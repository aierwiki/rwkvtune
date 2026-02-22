"""
Generation Mixin - Transformers-style generation interface

Adds .generate() method to RWKV models.

Design:
1. Provides Transformers-style API interface
2. Internally uses inference engine from rwkvtune.inference (GenerationCore)
3. Handles parameter adaptation and conversion
"""

import torch
from typing import Optional, Union, List, Dict, Any


class GenerationMixin:
    """
    Provides Transformers-style generation interface for RWKV models.
    
    This Mixin is a lightweight adapter layer that reuses existing inference code.
    
    Usage:
        model = RWKV7Model(config)
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            temperature=1.0,
            top_p=0.85,
            do_sample=True,
        )
    """
    
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        pad_token_id: Optional[int] = None,
        token_ban: Optional[List[int]] = None,
        return_dict_in_generate: bool = False,
        output_scores: bool = False,
        initial_states: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Transformers-style generation interface (naive implementation).
        
        Args:
            input_ids: Input token IDs, shape [batch_size, seq_len] or [seq_len]
            inputs: Alias for input_ids (compatibility)
            max_new_tokens: Maximum number of new tokens to generate (recommended)
            max_length: Maximum total length (input + output)
            min_length: Minimum total length (not supported)
            do_sample: Whether to use sampling (True) or greedy search (False)
            temperature: Sampling temperature, higher = more random
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty coefficient (not supported)
            num_return_sequences: Number of sequences to generate per input
            eos_token_id: End of sequence token ID (supports single int or List[int])
            pad_token_id: Padding token ID (not supported)
            token_ban: List of token IDs to ban (set logits to -inf)
            return_dict_in_generate: Whether to return dict (with more info)
            output_scores: Whether to output logits scores (not supported)
            initial_states: Initial RNN states (for State Cache, skip prefill)
            **kwargs: Other arguments
        
        Returns:
            If return_dict_in_generate=False (default):
                torch.Tensor: Generated token IDs, shape [batch_size * num_return_sequences, total_length]
            
            If return_dict_in_generate=True:
                Dict: Contains:
                    - sequences: Generated token IDs
        """
        import torch.nn.functional as F
        
        # Parameter processing
        if input_ids is None and inputs is not None:
            input_ids = inputs
        
        if input_ids is None:
            raise ValueError("Must provide input_ids or inputs parameter")
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        batch_size, input_length = input_ids.shape
        device = input_ids.device
        
        # Handle max length parameters
        if max_new_tokens is not None:
            actual_max_new_tokens = max_new_tokens
        elif max_length is not None:
            actual_max_new_tokens = max_length - input_length
        else:
            actual_max_new_tokens = 512  # Default: generate 512 tokens
        
        # Handle EOS token
        if eos_token_id is None and hasattr(self, "config"):
            eos_token_id = getattr(self.config, "eos_token_id", None)
        
        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                eos_token_ids = [eos_token_id]
            else:
                eos_token_ids = eos_token_id
        else:
            eos_token_ids = []
        
        # Handle do_sample
        if not do_sample:
            temperature = 1.0
            top_p = 0.0
            top_k = 0
        
        # Expand batch for multiple sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
            batch_size = batch_size * num_return_sequences
        
        # Helper: apply repetition penalty (same semantics as GRPOBatchGenerator)
        def _apply_repetition_penalty(
            logits: torch.Tensor,
            token_ids_per_row,
            penalty: float,
        ):
            if penalty == 1.0 or penalty <= 0:
                return
            vocab_size = logits.shape[-1]
            for i in range(logits.shape[0]):
                for tid in token_ids_per_row[i]:
                    if 0 <= tid < vocab_size:
                        if logits[i, tid].item() > 0:
                            logits[i, tid] = logits[i, tid] / penalty
                        else:
                            logits[i, tid] = logits[i, tid] * penalty

        # Naive inference loop
        with torch.no_grad():
            logits, states = self.forward_with_state(input_ids, states=initial_states)
            next_token_logits = logits[:, -1, :].to(torch.float32)
        
        output_ids = input_ids.clone()
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # Autoregressive generation loop
        for step in range(actual_max_new_tokens):
            # Apply repetition penalty based on history (excluding current token to sample)
            if repetition_penalty is not None and repetition_penalty != 1.0 and repetition_penalty > 0:
                # token history per batch row: all tokens generated so far (input + previous outputs)
                token_ids_per_row = [
                    output_ids[i].tolist()
                    for i in range(batch_size)
                ]
                _apply_repetition_penalty(next_token_logits, token_ids_per_row, repetition_penalty)
            
            # Apply temperature
            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply token_ban
            if token_ban is not None and len(token_ban) > 0:
                next_token_logits[:, token_ban] = float('-inf')
            
            # Sampling
            if not do_sample or temperature == 0:
                # Greedy search
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            else:
                # Probability sampling
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Top-k
                if top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)))
                    probs = torch.zeros_like(probs).scatter_(1, top_k_indices, top_k_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Top-p (Nucleus)
                if top_p > 0 and top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    probs[indices_to_remove] = 0
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            output_ids = torch.cat([output_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Check for EOS tokens
            if eos_token_ids:
                is_eos = torch.zeros_like(unfinished_sequences)
                for eos_id in eos_token_ids:
                    is_eos = is_eos | (next_tokens == eos_id)
                unfinished_sequences = unfinished_sequences & (~is_eos)
            
            if not unfinished_sequences.any():
                break
            
            # Forward pass for next token logits
            with torch.no_grad():
                current_input = next_tokens.unsqueeze(-1)
                logits, states = self.forward_with_state(current_input, states)
                next_token_logits = logits[:, -1, :].to(torch.float32)
        
        if return_dict_in_generate:
            return {"sequences": output_ids}
        else:
            return output_ids
