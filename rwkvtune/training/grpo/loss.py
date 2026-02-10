"""
GRPO Loss Functions - supports multiple variants.

Implements various loss functions for GRPO training including
DAPO, Dr. GRPO, and BNPO normalization strategies.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional


class GRPOLossBase(ABC):
    """Base class for GRPO loss functions."""

    def __init__(
        self,
        epsilon: float = 0.2,
        epsilon_high: Optional[float] = None,
        delta: Optional[float] = None,
        importance_sampling_level: str = "token"
    ):
        """
        Args:
            epsilon: PPO clip lower bound
            epsilon_high: PPO clip upper bound (None = same as epsilon)
            delta: Additional upper bound clip (optional)
            importance_sampling_level: Importance sampling level
                - 'token': Token level (GRPO, default)
                - 'sequence': Sequence level (GSPO)
        """
        self.epsilon = epsilon
        self.epsilon_high = epsilon_high if epsilon_high is not None else epsilon
        self.delta = delta
        self.importance_sampling_level = importance_sampling_level

        assert importance_sampling_level in ['token', 'sequence'], \
            f"importance_sampling_level must be 'token' or 'sequence', got {importance_sampling_level}"

    @abstractmethod
    def compute_loss(self, per_token_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Aggregate per-token loss.

        Args:
            per_token_loss: [B*G, C] - Per-token loss values
            mask: [B*G, C] - Valid token mask (1 = valid, 0 = padding)

        Returns:
            loss: Scalar loss value
        """
        raise NotImplementedError

    def compute_per_token_loss(
        self,
        logps_policy: torch.Tensor,
        logps_ref: Optional[torch.Tensor],
        logps_sampling: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        beta: float = 0.0
    ) -> torch.Tensor:
        """
        Compute per-token GRPO/GSPO loss.

        Args:
            logps_policy: [B*G, C] - Policy model log probabilities
            logps_ref: [B*G, C] - Reference model log probabilities (optional)
            logps_sampling: [B*G, C] - Sampling log probabilities
            advantages: [B*G] - Advantage values
            mask: [B*G, C] - Valid token mask
            beta: KL penalty coefficient

        Returns:
            per_token_loss: [B*G, C] - Per-token loss values
        """
        # Token-level log ratio
        log_ratio = logps_policy - logps_sampling  # [B*G, C]

        # Compute importance weights based on sampling level
        if self.importance_sampling_level == "token":
            # GRPO: Token-level importance weights
            log_importance_weights = log_ratio  # [B*G, C]
        elif self.importance_sampling_level == "sequence":
            # GSPO: Sequence-level importance weights
            # s_i(theta) = (pi_theta(o_i|q) / pi_theta_old(o_i|q))^(1/|o_i|)
            # log(s_i(theta)) = mean(log_ratio) over tokens in sequence
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)  # [B*G]
            log_importance_weights = log_importance_weights.unsqueeze(-1)  # [B*G, 1]
        else:
            raise ValueError(f"Unknown importance_sampling_level: {self.importance_sampling_level}")

        # Importance sampling ratio
        ratio = torch.exp(log_importance_weights)  # [B*G, C] or [B*G, 1]

        # Expand advantages to token dimension
        advantages = advantages.unsqueeze(1)  # [B*G, 1]

        # PPO-Clip objective
        clip_low = 1.0 - self.epsilon
        clip_high = 1.0 + self.epsilon_high

        ratio_clipped = torch.clamp(ratio, clip_low, clip_high)

        # Policy gradient loss (negative because we minimize)
        # Standard PPO: max E[min(ratio * A, clip(ratio) * A)]
        # Convert to minimization: min -E[min(ratio * A, clip(ratio) * A)]
        policy_loss = -torch.min(ratio * advantages, ratio_clipped * advantages)

        # Add KL penalty if beta > 0
        if beta > 0.0 and logps_ref is not None:
            # KL(pi || pi_ref) approx exp(log_ref - log_policy) - (log_ref - log_policy) - 1
            # This is an f-divergence form, always >= 0
            log_ratio_kl = logps_ref - logps_policy  # [B*G, C]
            per_token_kl = torch.exp(log_ratio_kl) - log_ratio_kl - 1
            policy_loss = policy_loss + beta * per_token_kl

        # Apply mask
        policy_loss = policy_loss * mask

        return policy_loss


class DAPOLoss(GRPOLossBase):
    """
    DAPO loss (default, recommended).

    Normalization: Global valid token count
    Loss = sum(per_token_loss * mask) / sum(mask)
    """

    def compute_loss(self, per_token_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_loss = per_token_loss * mask
        total_loss = masked_loss.sum()
        num_tokens = mask.sum()
        num_tokens = torch.clamp(num_tokens, min=1.0)
        loss = total_loss / num_tokens
        return loss


class DrGRPOLoss(GRPOLossBase):
    """
    Dr. GRPO loss.

    Normalization: Constant (max_completion_length)
    Loss = sum(per_token_loss * mask) / (B * G * L)
    """

    def __init__(self, max_completion_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_completion_length = max_completion_length

    def compute_loss(self, per_token_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_loss = per_token_loss * mask
        total_loss = masked_loss.sum()
        B_times_G = per_token_loss.shape[0]
        normalizer = B_times_G * self.max_completion_length
        loss = total_loss / normalizer
        return loss


class BNPOLoss(GRPOLossBase):
    """
    BNPO loss.

    Normalization: Local batch valid token count
    Loss = sum(per_token_loss * mask) / local_num_tokens

    Note: Does not sync across processes in distributed training.
    """

    def compute_loss(self, per_token_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        local_num_tokens = mask.sum()
        local_num_tokens = torch.clamp(local_num_tokens, min=1.0)
        masked_loss = per_token_loss * mask
        total_loss = masked_loss.sum()
        loss = total_loss / local_num_tokens
        return loss


def get_loss_function(loss_type: str, config) -> GRPOLossBase:
    """
    Loss function factory.

    Args:
        loss_type: Loss function type ('dapo', 'dr_grpo', 'bnpo', 'grpo')
        config: GRPO config object

    Returns:
        Loss function instance
    """
    loss_registry = {
        'dapo': DAPOLoss,
        'dr_grpo': lambda *args, **kwargs: DrGRPOLoss(
            max_completion_length=config.max_completion_length,
            *args, **kwargs
        ),
        'bnpo': BNPOLoss,
        'grpo': DAPOLoss,  # 'grpo' as alias for 'dapo'
    }

    if loss_type not in loss_registry:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Available types: {list(loss_registry.keys())}"
        )

    loss_class = loss_registry[loss_type]
    return loss_class(
        epsilon=config.epsilon,
        epsilon_high=config.epsilon_high,
        delta=config.delta,
        importance_sampling_level=config.importance_sampling_level
    )
