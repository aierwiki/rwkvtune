"""
RWKV Inference Core - Unified Sampling and State Management

This module provides core inference engine functionality shared by
InferencePipeline and BatchGenerator.

Components:
1. TokenSampler - Unified token sampler (supports top-k, top-p, temperature)
"""

import torch
import torch.nn.functional as F
from typing import Any, Callable, List, Tuple, Optional


class TokenSampler:
    """
    Unified Token Sampler
    
    Supports multiple sampling strategies:
    - greedy (argmax)
    - temperature sampling
    - top-k sampling
    - top-p (nucleus) sampling
    - combined strategies
    
    Features:
    - Numerically stable (uses log_softmax or max subtraction)
    - Supports CPU/GPU
    - Supports single and batch sampling
    - Handles NaN/Inf cases
    
    Config:
    - _max_valid_token_id: Maximum valid token ID (tokens beyond are banned)
    
    Note: token_id=0 is not banned as it may be a valid EOS token (e.g., <|im_end|>)
    """
    
    @staticmethod
    def sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        logit_bias: Optional[dict] = None,
        logits_processor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[int, float]:
        """
        Single sample (sample one token from logits)
        
        Args:
            logits: [vocab_size] - unnormalized logits (should be fp32)
            temperature: Temperature parameter (higher = more random)
            top_p: Nucleus sampling parameter (keep tokens with cumulative prob <= top_p)
            top_k: Top-k sampling parameter (only consider top k tokens, 0 = disabled)
            logit_bias: Token ID to bias value mapping (adjust specific token probabilities)
            logits_processor: Optional logits processor (for JSON Schema constraints etc.)
        
        Returns:
            token_id: Sampled token ID
            log_prob: Corresponding log probability
        """
        if logits.dim() != 1:
            raise ValueError(f"Single sample requires 1D logits, got shape {logits.shape}")
        
        # Ban tokens beyond vocabulary range
        max_valid_token_id = getattr(TokenSampler, '_max_valid_token_id', None)
        if max_valid_token_id is not None:
            if max_valid_token_id + 1 < len(logits):
                logits[max_valid_token_id + 1:] = float('-inf')
            elif max_valid_token_id >= len(logits):
                # This should not happen if tokenizer and model vocab_size match
                import warnings
                warnings.warn(
                    f"TokenSampler: max_valid_token_id ({max_valid_token_id}) >= logits size ({len(logits)}). "
                    f"This may indicate a mismatch between tokenizer vocab and model vocab_size."
                )
        
        # Greedy sampling: return argmax when temperature=0
        if temperature == 0 or temperature < 1e-6:
            token = logits.argmax().item()
            log_prob = F.log_softmax(logits, dim=-1)[token].item()
            return token, log_prob
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply logit_bias (OpenAI compatible)
        if logit_bias is not None and len(logit_bias) > 0:
            for token_id_str, bias in logit_bias.items():
                token_id = int(token_id_str)
                if 0 <= token_id < len(logits):
                    clamped_bias = min(100.0, max(-100.0, bias))
                    logits[token_id] += clamped_bias
        
        # Apply logits_processor (e.g., JSON Schema constraints)
        if logits_processor is not None:
            logits = logits_processor(logits)
        
        # Numerically stable softmax (subtract max)
        logits_max = torch.max(logits)
        exp_logits = torch.exp(logits - logits_max)
        probs = exp_logits / torch.sum(exp_logits)
        
        # Handle NaN or Inf with uniform distribution fallback
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.ones_like(logits) / len(logits)
        
        # Top-k sampling
        if top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, min(top_k, len(probs)))
            top_k_probs = top_k_probs / top_k_probs.sum()
            sampled_idx = torch.multinomial(top_k_probs, 1)
            token = top_k_indices[sampled_idx].item()
            logp = torch.log(top_k_probs[sampled_idx] + 1e-10).item()
        
        # Top-p (nucleus) sampling
        elif top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            cutoff = (cumsum <= top_p).sum().item() + 1
            cutoff = max(1, cutoff)
            
            nucleus_probs = sorted_probs[:cutoff]
            nucleus_indices = sorted_indices[:cutoff]
            nucleus_probs = nucleus_probs / nucleus_probs.sum()
            
            sampled_idx = torch.multinomial(nucleus_probs, 1)
            token = nucleus_indices[sampled_idx].item()
            logp = torch.log(nucleus_probs[sampled_idx] + 1e-10).item()
        
        # Standard sampling (no filtering)
        else:
            sampled_idx = torch.multinomial(probs, 1)
            token = sampled_idx.item()
            logp = torch.log(probs[token] + 1e-10).item()
        
        return token, logp
    
    @staticmethod
    def sample_batch(
        logits_batch: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        logit_bias: Optional[dict] = None,
        logits_processor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch sampling (GPU optimized)
        
        Args:
            logits_batch: [B, vocab_size] - batch logits
            temperature: Temperature parameter
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            logit_bias: Token ID to bias value mapping
            logits_processor: Optional logits processor
        
        Returns:
            tokens: [B] - sampled token IDs
            logps: [B] - corresponding log probabilities
        """
        if logits_batch.dim() != 2:
            raise ValueError(f"Batch sampling requires 2D logits, got shape {logits_batch.shape}")
        
        B, embedding_size = logits_batch.shape
        
        # Ban tokens beyond vocabulary range
        max_valid_token_id = getattr(TokenSampler, '_max_valid_token_id', None)
        if max_valid_token_id is not None:
            if max_valid_token_id + 1 < embedding_size:
                logits_batch[:, max_valid_token_id + 1:] = float('-inf')
            elif max_valid_token_id >= embedding_size:
                # This should not happen if tokenizer and model vocab_size match
                import warnings
                warnings.warn(
                    f"TokenSampler: max_valid_token_id ({max_valid_token_id}) >= logits size ({embedding_size}). "
                    f"This may indicate a mismatch between tokenizer vocab and model vocab_size."
                )
        
        # Greedy sampling: return argmax when temperature=0
        if temperature == 0 or temperature < 1e-6:
            tokens = logits_batch.argmax(dim=-1)
            log_probs = F.log_softmax(logits_batch, dim=-1)
            logps = log_probs.gather(1, tokens.unsqueeze(1)).squeeze(1)
            return tokens, logps
        
        # Apply temperature
        if temperature != 1.0:
            logits_batch = logits_batch / temperature
        
        # Apply logit_bias (OpenAI compatible)
        if logit_bias is not None and len(logit_bias) > 0:
            for token_id_str, bias in logit_bias.items():
                token_id = int(token_id_str)
                if 0 <= token_id < logits_batch.shape[1]:
                    clamped_bias = min(100.0, max(-100.0, bias))
                    logits_batch[:, token_id] += clamped_bias
        
        # Apply logits_processor
        if logits_processor is not None:
            logits_batch = logits_processor(logits_batch)
        
        # Top-k filtering
        if top_k > 0:
            top_k_values, top_k_indices = torch.topk(logits_batch, top_k, dim=-1)
            logits_filtered = torch.full_like(logits_batch, float('-inf'))
            logits_filtered.scatter_(-1, top_k_indices, top_k_values)
            logits_batch = logits_filtered
        
        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits_batch, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Find cutoff position
            cutoff_mask = cumsum_probs <= top_p
            cutoff_mask[:, 0] = True  # Keep at least one
            
            # Apply mask
            sorted_logits = sorted_logits.masked_fill(~cutoff_mask, float('-inf'))
            
            # Restore original order
            _, restore_indices = torch.sort(sorted_indices, dim=-1)
            logits_batch = torch.gather(sorted_logits, -1, restore_indices)
        
        # Compute probabilities and sample
        probs = F.softmax(logits_batch, dim=-1)
        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits_batch, dim=-1)
        logps = log_probs.gather(1, tokens.unsqueeze(1)).squeeze(1)
        
        return tokens, logps
