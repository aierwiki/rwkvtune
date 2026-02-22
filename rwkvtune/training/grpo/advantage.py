"""
Advantage Function Calculator - GRPO core component.

Implements group-wise normalization for computing advantages in GRPO training.
The key insight is to compare multiple completions generated for the same prompt.
"""
import torch
from typing import Optional


class AdvantageCalculator:
    """
    Advantage function calculator for GRPO.

    Core concept: Group-wise normalization
    GRPO compares multiple completions generated for the same prompt,
    computing relative advantages within each group.
    """

    def __init__(self, scale_rewards: str = 'group', num_generations: int = 8,
                 advantage_clip: Optional[float] = None,
                 low_reward_threshold: Optional[float] = None,
                 low_reward_scale: float = 0.01):
        """
        Args:
            scale_rewards: Reward scaling method
                - 'group': Group-wise normalization (default, standard GRPO)
                - 'batch': Batch-level normalization
                - 'none': No normalization
            num_generations: Number of generations per prompt (G)
            advantage_clip: If set, clamp advantages to [-clip, +clip] after
                standardization.  Prevents extreme gradient magnitudes caused
                by near-zero group std.  None = disabled (default).
                Recommended value: 5.0~10.0.
            low_reward_threshold: If set, groups whose max reward is below this
                threshold will have their advantages scaled down by
                ``low_reward_scale``, effectively suppressing gradient updates
                from low-quality rollout groups.  None = disabled (default).
            low_reward_scale: Multiplier applied to advantages of groups that
                fall below ``low_reward_threshold``.  Default 0.01 makes the
                gradient contribution negligible without introducing NaN.
        """
        assert scale_rewards in ['group', 'batch', 'none'], \
            f"scale_rewards must be one of ['group', 'batch', 'none'], got {scale_rewards}"
        self.scale_rewards = scale_rewards
        self.G = num_generations
        self.advantage_clip = advantage_clip
        self.low_reward_threshold = low_reward_threshold
        self.low_reward_scale = low_reward_scale

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute advantage function.

        Args:
            rewards: [B*G] - Reward values, where B is prompt count, G is generations per prompt

        Returns:
            advantages: [B*G] - Advantage values

        Algorithm:
            1. Reshape rewards to [B, G]
            2. Compute group-wise mean: mean_r[i] = mean(rewards[i, :])
            3. Compute raw advantage: adv_raw = rewards - mean_r
            4. (optional) Standardize: adv = adv_raw / std
            5. Flatten to [B*G]
        """
        if self.scale_rewards == 'none':
            return rewards

        total_rewards = rewards.shape[0]
        if total_rewards % self.G != 0:
            raise ValueError(
                f"Rewards count ({total_rewards}) must be divisible by num_generations ({self.G})!"
            )

        # Reshape to [B, G]
        B = total_rewards // self.G
        rewards_matrix = rewards.view(B, self.G)

        if self.scale_rewards == 'group':
            advantages = self._compute_group_advantages(rewards_matrix)
        elif self.scale_rewards == 'batch':
            advantages = self._compute_batch_advantages(rewards_matrix)
        else:
            raise ValueError(f"Unknown scale_rewards: {self.scale_rewards}")

        return advantages.view(-1)

    def _compute_group_advantages(self, rewards_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute group-wise advantages [B, G] -> [B, G].

        Core idea: For multiple completions of the same prompt, compute advantages
        relative to the group mean. This avoids scale differences between prompts.

        When ``low_reward_threshold`` is set, groups where **all** completions
        scored poorly (max reward < threshold) get their advantages scaled down
        to near-zero so that they contribute almost no gradient, preventing the
        model from fitting to noise among uniformly bad samples.
        """
        # Group-wise mean [B, G] -> [B, 1]
        mean_r = rewards_matrix.mean(dim=1, keepdim=True)

        # Raw advantages
        advantages = rewards_matrix - mean_r

        # Group-wise standard deviation [B, G] -> [B, 1]
        std_r = rewards_matrix.std(dim=1, keepdim=True, unbiased=False)

        # Avoid division by zero
        std_r = torch.clamp(std_r, min=1e-8)

        # Standardize
        advantages = advantages / std_r

        # Clip extreme advantages (e.g. from near-zero std groups)
        if self.advantage_clip is not None:
            pre_clip_abs_max = advantages.abs().max().item()
            advantages = advantages.clamp(-self.advantage_clip, self.advantage_clip)
            if pre_clip_abs_max > self.advantage_clip:
                print(
                    f"[ADVANTAGE] Clipped advantages: "
                    f"abs_max {pre_clip_abs_max:.2f} -> {self.advantage_clip:.1f}"
                )

        # Suppress advantages for groups where all completions are poor
        if self.low_reward_threshold is not None:
            max_r = rewards_matrix.max(dim=1, keepdim=True).values  # [B, 1]
            low_quality_mask = (max_r < self.low_reward_threshold)  # [B, 1] bool
            if low_quality_mask.any():
                n_suppressed = low_quality_mask.sum().item()
                print(
                    f"[ADVANTAGE] Suppressing {n_suppressed}/{rewards_matrix.shape[0]} groups "
                    f"(max_reward < {self.low_reward_threshold}, scale={self.low_reward_scale})"
                )
            advantages = torch.where(
                low_quality_mask.expand_as(advantages),
                advantages * self.low_reward_scale,
                advantages,
            )

        return advantages

    def _compute_batch_advantages(self, rewards_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute batch-level advantages [B, G] -> [B, G].

        Normalizes across the entire batch.
        """
        mean_r = rewards_matrix.mean()
        std_r = rewards_matrix.std(unbiased=False)

        # Avoid division by zero
        std_r = torch.clamp(std_r, min=1e-8)

        advantages = (rewards_matrix - mean_r) / std_r

        if self.advantage_clip is not None:
            advantages = advantages.clamp(-self.advantage_clip, self.advantage_clip)

        return advantages

    def compute_group_statistics(self, rewards: torch.Tensor) -> dict:
        """
        Compute group statistics (for logging).

        Args:
            rewards: [B*G] - Reward values

        Returns:
            Statistics dict containing:
            - group_mean_rewards: [B] Mean reward for each group
            - group_std_rewards: [B] Standard deviation for each group
            - global_mean_reward: Global mean reward
            - global_std_reward: Global standard deviation
        """
        total_rewards = rewards.shape[0]
        if total_rewards % self.G != 0:
            raise ValueError(
                f"Rewards count ({total_rewards}) must be divisible by num_generations ({self.G})!"
            )

        B = total_rewards // self.G
        rewards_matrix = rewards.view(B, self.G)

        return {
            'group_mean_rewards': rewards_matrix.mean(dim=1),  # [B]
            'group_std_rewards': rewards_matrix.std(dim=1, unbiased=False),  # [B]
            'global_mean_reward': rewards.mean(),
            'global_std_reward': rewards.std(unbiased=False),
        }
