"""
GRPO Training Utilities

This module contains utility functions for GRPO training, optimized for memory efficiency.
Inspired by trl-main (https://github.com/huggingface/trl).
"""

import torch
import torch.nn.functional as F
from typing import Optional


def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    However, it avoids materializing the full log_softmax tensor, which can be very large
    when vocab_size is large (e.g., 65536 for RWKV7).

    Memory Savings:
    - For a batch of [B, T, V] logits, naive approach needs [B, T, V] space for log_softmax
    - This approach only needs [B, T] space for selected logps + temporary buffers
    - Example: [256, 3800, 65536] @ bf16 -> saves ~120GB of memory

    Args:
        logits (`torch.Tensor`): Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`): Index tensor of shape `(...)`, specifying the positions
            to gather from the log-softmax output.

    Returns:
        `torch.Tensor`: Gathered log probabilities with the same shape as `index`.

    References:
        - TRL implementation: https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L1459
        - Mathematical equivalence: log_softmax(x_i) = x_i - logsumexp(x)
    """
    if logits.dtype in [torch.float32, torch.float64]:
        # For fp32/fp64: use mathematically stable logsumexp approach
        # First gather the selected logits
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)

        # Compute logsumexp for each row (loop to reduce peak memory consumption)
        # This is more memory-efficient than torch.logsumexp(logits, dim=-1)
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])

        # log_softmax(x_i) = x_i - logsumexp(x)
        per_token_logps = selected_logits - logsumexp_values
    else:
        # For bf16/fp16: logsumexp approach can be numerically unstable
        # Fall back to direct computation but still avoid materializing full log_softmax
        # We only loop over the batch dimension (first dim), not all elements
        # Reshape to [B, -1, vocab_size] to handle any input shape
        original_shape = logits.shape

        if len(original_shape) > 2:
            # Flatten all dimensions except last (vocab) into batch dimension
            batch_dim = original_shape[0]
            seq_dim = original_shape[1] if len(original_shape) > 2 else 1
            vocab_dim = original_shape[-1]

            # Process in batches to avoid slow element-wise loop
            # For [32, 4096, 65536], we loop 32 times (not 32*4096 times)
            per_token_logps = []
            for i in range(batch_dim):
                batch_logits = logits[i]  # [seq_len, vocab_size]
                batch_labels = index[i]   # [seq_len]

                # Use memory-efficient logsumexp instead of full log_softmax
                # First gather selected logits
                batch_selected_logits = torch.gather(
                    batch_logits, dim=-1, index=batch_labels.unsqueeze(-1)
                ).squeeze(-1)

                # Compute logsumexp for numerical stability (convert to float32 if needed)
                if batch_logits.dtype == torch.bfloat16:
                    batch_logsumexp = torch.logsumexp(
                        batch_logits.float(), dim=-1
                    ).to(batch_logits.dtype)
                else:
                    batch_logsumexp = torch.logsumexp(batch_logits, dim=-1)

                # log_softmax(x_i) = x_i - logsumexp(x)
                batch_per_token_logps = batch_selected_logits - batch_logsumexp
                per_token_logps.append(batch_per_token_logps)

            per_token_logps = torch.stack(per_token_logps)  # [batch_dim, seq_len]
        else:
            # Handle 2D input using logsumexp method
            selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)

            if logits.dtype == torch.bfloat16:
                logsumexp_values = torch.logsumexp(logits.float(), dim=-1).to(logits.dtype)
            else:
                logsumexp_values = torch.logsumexp(logits, dim=-1)

            per_token_logps = selected_logits - logsumexp_values

    return per_token_logps


def entropy_from_logits(logits: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    """
    Compute the Shannon entropy (in nats) for each row of *logits* in a memory-efficient way.

    Instead of materializing the full softmax for all rows at once, the logits are flattened
    to shape (N, num_classes), where N is the product of all leading dimensions.
    Computation is then performed in chunks of size `chunk_size` along this flattened
    dimension, reducing peak memory usage. The result is reshaped back to match the
    input's leading dimensions.

    Args:
        logits (`torch.Tensor`): Logits tensor of shape `(..., num_classes)`.
            Entropy is taken along the last axis; all leading dimensions are
            preserved in the output.
        chunk_size (`int`, *optional*, defaults to `128`): Number of rows from the
            flattened logits to process per iteration. Smaller values reduce memory
            usage at the cost of additional overhead.

    Returns:
        `torch.Tensor`: Entropy tensor of shape `(...)` (same as input except for
            the last dimension removed).

    References:
        - TRL implementation: https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L1494
    """
    original_shape = logits.shape[:-1]
    num_classes = logits.shape[-1]

    # Flatten all leading dimensions
    logits_flat = logits.reshape(-1, num_classes)
    N = logits_flat.shape[0]

    entropies = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = logits_flat[start:end]

        # Compute entropy for this chunk: H(p) = -sum(p * log(p))
        probs = F.softmax(chunk, dim=-1)
        log_probs = F.log_softmax(chunk, dim=-1)
        chunk_entropy = -(probs * log_probs).sum(dim=-1)
        entropies.append(chunk_entropy)

    entropies = torch.cat(entropies, dim=0)
    entropies = entropies.reshape(original_shape)

    return entropies
