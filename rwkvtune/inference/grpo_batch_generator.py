"""
GRPO Batch Generator - Simple and Efficient with Chunked Prefill

Design Principles:
1. Simple and efficient: direct batch forward, no complex scheduling
2. Supports Chunked Prefill: control memory peak
3. High throughput: maximize batching efficiency
4. Low overhead: avoid SchedulerCore state management overhead

Difference from BatchGenerator:
- BatchGenerator: designed for serving, supports dynamic sequence addition
- GRPOBatchGenerator: designed for training, all sequences start together with same params
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time

from rwkvtune.inference.core import TokenSampler


def _apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids_per_row: List[List[int]],
    penalty: float,
) -> None:
    """Apply repetition penalty in-place to suppress repeated tokens.
    logits: [B, V]. token_ids_per_row: for each row, list of token ids already generated (to penalize).
    penalty > 1: reduce probability of repeating these tokens.
    """
    if penalty == 1.0 or penalty <= 0:
        return
    for i in range(logits.shape[0]):
        for tid in token_ids_per_row[i]:
            if 0 <= tid < logits.shape[1]:
                if logits[i, tid].item() > 0:
                    logits[i, tid] = logits[i, tid] / penalty
                else:
                    logits[i, tid] = logits[i, tid] * penalty


@dataclass
class GRPOSequence:
    """Simplified sequence state (for GRPO only)"""
    seq_id: int
    prompt_tokens: List[int]
    generated_tokens: List[int] = None
    prefill_offset: int = 0
    batch_idx: int = -1
    is_finished: bool = False
    eos_token_id: Optional[int] = None
    num_generated: int = 0
    
    def __post_init__(self):
        if self.generated_tokens is None:
            self.generated_tokens = []
            self.num_generated = 0


class GRPOBatchGenerator:
    """
    GRPO Batch Generator
    
    Features:
    - Simple and efficient: direct batch forward, no complex scheduling
    - Supports Chunked Prefill: process in chunks, control memory peak
    - High throughput: group by length, maximize batching efficiency
    - Low overhead: pre-allocate state, avoid dynamic allocation
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        chunk_size: int = 2048,
        max_prefill_batch_size: int = -1,
        max_decode_batch_size: int = -1,
        eos_token_id: Optional[int] = None,
    ):
        """
        Args:
            model: RWKV model
            tokenizer: Tokenizer
            chunk_size: Prefill chunk size (tokens)
            max_prefill_batch_size: Max batch size for prefill (-1 = unlimited)
            max_decode_batch_size: Max batch size for decode (-1 = unlimited)
            eos_token_id: EOS token ID
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.chunk_size = chunk_size
        self.max_prefill_batch_size = max_prefill_batch_size
        self.max_decode_batch_size = max_decode_batch_size
        self.eos_token_id = eos_token_id
        
        self.sampler = TokenSampler
        
        # Set max valid token ID to ban out-of-vocab tokens
        if hasattr(tokenizer, 'idx2token') and tokenizer.idx2token:
            max_token_id = max(tokenizer.idx2token.keys())
            TokenSampler._max_valid_token_id = max_token_id
            model_vocab_size = getattr(model.config, 'vocab_size', None)
            if model_vocab_size and model_vocab_size > max_token_id + 1:
                print(f"[GRPOBatchGenerator] [INFO] Model vocab_size={model_vocab_size}, tokenizer max ID={max_token_id}")
                print(f"[GRPOBatchGenerator] [INFO] Will ban token IDs {max_token_id + 1} to {model_vocab_size - 1} during sampling")
        
        self.n_layer = model.config.n_layer
        self.n_embd = model.config.n_embd
        
    def generate(
        self,
        input_ids: List[List[int]],
        max_new_tokens: int,
        num_generations: int = 1,
        temperature: float = 0.9,
        top_p: float = 0.9,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        logit_bias: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        Batch generation with Chunked Prefill and state reuse optimization
        
        For same prompt with multiple generation paths, only prefill once then copy state.
        This saves computation especially when num_generations is large.
        
        Args:
            input_ids: Tokenized inputs [[tok1, tok2, ...], ...] (B prompts)
            max_new_tokens: Maximum tokens to generate
            num_generations: Number of completions per prompt (G)
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            repetition_penalty: Penalize already-generated tokens (>1 suppresses repetition)
            logit_bias: Token ID to bias mapping (ban certain tokens)
        
        Returns:
            Dict with:
                - completions: List[str] - generated texts (B*G)
                - completion_ids: torch.Tensor [B*G, max_len] - token IDs
                - masks: torch.Tensor [B*G, max_len] - valid token mask
        """
        B = len(input_ids)
        G = num_generations
        total = B * G
        
        # Identify unique prompts (deduplication)
        unique_prompts = {}
        prompt_to_unique_idx = []
        
        for tokens in input_ids:
            tokens_tuple = tuple(tokens)
            if tokens_tuple not in unique_prompts:
                unique_idx = len(unique_prompts)
                unique_prompts[tokens_tuple] = (unique_idx, tokens)
            prompt_to_unique_idx.append(unique_prompts[tokens_tuple][0])
        
        B_unique = len(unique_prompts)
        print(f"[GRPOBatchGenerator] State reuse optimization:")
        print(f"   - original_prompts: {B}")
        print(f"   - unique_prompts: {B_unique}")
        print(f"   - generations_per_prompt: {G}")
        print(f"   - total_sequences: {total}")
        print(f"   - prefill_reduction: {B}/{B_unique} = {B/B_unique:.2f}x")
        
        # Create all sequence objects (B*G)
        sequences = []
        for b_idx in range(B):
            prompt_tokens = input_ids[b_idx]
            for g_idx in range(G):
                seq_id = b_idx * G + g_idx
                sequences.append(
                    GRPOSequence(
                        seq_id=seq_id,
                        prompt_tokens=prompt_tokens,
                        eos_token_id=self.eos_token_id,
                    )
                )
        
        # Pre-allocate all states (B*G sequences)
        all_states = self.model.init_state(total)
        
        for i, seq in enumerate(sequences):
            seq.batch_idx = i
        
        # Create unique prompt sequences for prefill
        unique_sequences = []
        for unique_idx, (_, prompt_tokens) in enumerate(unique_prompts.values()):
            unique_sequences.append(
                GRPOSequence(
                    seq_id=unique_idx,
                    prompt_tokens=list(prompt_tokens),
                    eos_token_id=self.eos_token_id,
                )
            )
        
        # Allocate temporary states for unique prompts
        unique_states = self.model.init_state(B_unique)
        for i, seq in enumerate(unique_sequences):
            seq.batch_idx = i
        
        # Prefill unique prompts only
        print(f"[GRPOBatchGenerator] Starting Prefill ({B_unique} unique prompts)...")
        prefill_start = time.time()
        
        total_prompt_tokens = sum(len(seq.prompt_tokens) for seq in unique_sequences)
        print(f"[GRPOBatchGenerator]   - total_prompt_tokens: {total_prompt_tokens:,}")
        print(f"[GRPOBatchGenerator]   - avg_prompt_length: {total_prompt_tokens / B_unique:.1f} tokens")
        print(f"[GRPOBatchGenerator]   - chunk_size: {self.chunk_size}")
        print(f"[GRPOBatchGenerator]   - max_prefill_batch_size: {self.max_prefill_batch_size if self.max_prefill_batch_size > 0 else 'unlimited'}")
        
        self._chunked_prefill(unique_sequences, unique_states, temperature, top_p, top_k, repetition_penalty, logit_bias)
        
        prefill_time = time.time() - prefill_start
        prefill_throughput = total_prompt_tokens / prefill_time if prefill_time > 0 else 0
        print(f"[GRPOBatchGenerator] [OK] Prefill complete:")
        print(f"   - time: {prefill_time:.2f}s")
        print(f"   - tokens: {total_prompt_tokens:,}")
        print(f"   - throughput: {prefill_throughput:.1f} tokens/s")
        print(f"   - avg_per_sequence: {prefill_time / B_unique:.3f}s")
        
        # Copy states to all generation paths and sample first token
        print(f"[GRPOBatchGenerator] Copying states to {total} generation paths and sampling first token...")
        copy_start = time.time()
        
        n_layers = len(all_states)
        
        # Build copy mapping (batch operation)
        src_indices = []
        dst_indices = []
        unique_seq_map = {}
        
        for b_idx in range(B):
            unique_idx = prompt_to_unique_idx[b_idx]
            unique_seq = unique_sequences[unique_idx]
            
            for g_idx in range(G):
                seq_idx = b_idx * G + g_idx
                src_indices.append(unique_idx)
                dst_indices.append(seq_idx)
                unique_seq_map[seq_idx] = unique_seq
        
        # Batch copy all layer states
        src_indices_tensor = torch.tensor(src_indices, dtype=torch.long, device=self.device)
        dst_indices_tensor = torch.tensor(dst_indices, dtype=torch.long, device=self.device)
        
        for layer_idx in range(len(all_states)):
            src_states = unique_states[layer_idx]['att_x_prev'][src_indices_tensor].clone()
            all_states[layer_idx]['att_x_prev'][dst_indices_tensor] = src_states
            
            src_states = unique_states[layer_idx]['att_kv'][src_indices_tensor].clone()
            all_states[layer_idx]['att_kv'][dst_indices_tensor] = src_states
            
            src_states = unique_states[layer_idx]['ffn_x_prev'][src_indices_tensor].clone()
            all_states[layer_idx]['ffn_x_prev'][dst_indices_tensor] = src_states
        
        # Sample first token for each generation path (apply repetition penalty to prompt tokens)
        for seq_idx in range(total):
            seq = sequences[seq_idx]
            unique_seq = unique_seq_map[seq_idx]
            
            if hasattr(unique_seq, '_prefill_logits') and unique_seq._prefill_logits is not None:
                prefill_logits = unique_seq._prefill_logits.clone()
                _apply_repetition_penalty(
                    prefill_logits.unsqueeze(0),
                    [seq.prompt_tokens],
                    repetition_penalty,
                )
                prefill_logits = prefill_logits.squeeze(0)
                token_id, _ = self.sampler.sample(
                    prefill_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    logit_bias=logit_bias,
                )
                seq.generated_tokens.append(token_id)
                seq.num_generated = 1
                if token_id == seq.eos_token_id:
                    seq.is_finished = True
            else:
                print(f"[GRPOBatchGenerator] [WARN] Sequence {seq.seq_id} missing prefill logits, recomputing")
                prompt_tokens = seq.prompt_tokens
                last_token = torch.tensor([[prompt_tokens[-1]]], dtype=torch.long, device=self.device)
                batch_states = self._extract_batch_states(all_states, [seq_idx])
                with torch.no_grad():
                    logits, _ = self.model.forward_with_state(last_token, states=batch_states)
                next_token_logits = logits[0, -1, :].to(torch.float32).clone()
                _apply_repetition_penalty(
                    next_token_logits.unsqueeze(0),
                    [seq.prompt_tokens],
                    repetition_penalty,
                )
                next_token_logits = next_token_logits.squeeze(0)
                token_id, _ = self.sampler.sample(
                    next_token_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    logit_bias=logit_bias,
                )
                seq.generated_tokens.append(token_id)
                seq.num_generated = 1
                if token_id == seq.eos_token_id:
                    seq.is_finished = True
        
        copy_time = time.time() - copy_start
        print(f"[GRPOBatchGenerator] [OK] State copy and first sampling complete: {copy_time:.2f}s")
        
        # Release unique prompt states
        del unique_states
        del unique_sequences
        torch.cuda.empty_cache()
        
        # Batch decode
        print(f"[GRPOBatchGenerator] Starting Decode ({total} sequences, max_new_tokens={max_new_tokens})...")
        print(f"[GRPOBatchGenerator]   - max_decode_batch_size: {self.max_decode_batch_size if self.max_decode_batch_size > 0 else 'unlimited'}")
        decode_start = time.time()
        decode_stats = self._batch_decode(sequences, all_states, max_new_tokens, temperature, top_p, top_k, repetition_penalty, logit_bias)
        decode_time = time.time() - decode_start
        
        total_generated_tokens = sum(seq.num_generated for seq in sequences)
        decode_throughput = total_generated_tokens / decode_time if decode_time > 0 else 0
        avg_tokens_per_seq = total_generated_tokens / total if total > 0 else 0
        
        print(f"[GRPOBatchGenerator] [OK] Decode complete:")
        print(f"   - time: {decode_time:.2f}s")
        print(f"   - total_steps: {decode_stats.get('total_steps', 0)}")
        print(f"   - generated_tokens: {total_generated_tokens:,}")
        print(f"   - avg_per_sequence: {avg_tokens_per_seq:.1f} tokens")
        print(f"   - throughput: {decode_throughput:.1f} tokens/s")
        
        # Summary
        total_time = prefill_time + copy_time + decode_time
        total_tokens = total_prompt_tokens + total_generated_tokens
        overall_throughput = total_tokens / total_time if total_time > 0 else 0
        
        print(f"[GRPOBatchGenerator] Overall performance:")
        print(f"   - total_time: {total_time:.2f}s")
        print(f"   - prefill_ratio: {prefill_time / total_time * 100:.1f}%")
        print(f"   - state_copy_ratio: {copy_time / total_time * 100:.1f}%")
        print(f"   - decode_ratio: {decode_time / total_time * 100:.1f}%")
        print(f"   - total_tokens: {total_tokens:,} (prompt: {total_prompt_tokens:,} + generated: {total_generated_tokens:,})")
        print(f"   - overall_throughput: {overall_throughput:.1f} tokens/s")
        
        return self._collect_results(sequences)
    
    def _chunked_prefill(
        self,
        sequences: List[GRPOSequence],
        all_states: List[Dict[str, torch.Tensor]],
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float = 1.0,
        logit_bias: Optional[dict] = None,
    ):
        """
        Chunked Prefill: process prompts in chunks
        
        Strategy:
        1. Group by prompt length (same length processed together, no padding)
        2. Split each prompt into chunks of chunk_size
        3. Batch process each chunk, update batch states
        """
        # Group by prompt length
        sequences_by_len = defaultdict(list)
        for seq in sequences:
            prompt_len = len(seq.prompt_tokens)
            sequences_by_len[prompt_len].append(seq)
        
        print(f"[GRPOBatchGenerator]   - length_groups: {len(sequences_by_len)} different lengths")
        for prompt_len, seq_group in sequences_by_len.items():
            num_chunks = (prompt_len + self.chunk_size - 1) // self.chunk_size
            print(f"     * length {prompt_len}: {len(seq_group)} sequences, {num_chunks} chunks")
        
        # Process each length group in chunks
        for prompt_len, seq_group in sequences_by_len.items():
            num_chunks = (prompt_len + self.chunk_size - 1) // self.chunk_size
            
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * self.chunk_size
                chunk_end = min(chunk_start + self.chunk_size, prompt_len)
                
                chunk_inputs = []
                batch_indices = []
                
                for seq in seq_group:
                    if seq.prefill_offset < prompt_len:
                        chunk = seq.prompt_tokens[chunk_start:chunk_end]
                        chunk_inputs.append(chunk)
                        batch_indices.append(seq.batch_idx)
                
                if not chunk_inputs:
                    continue
                
                # Group by chunk length (avoid padding)
                chunks_by_len = defaultdict(list)
                for i, chunk in enumerate(chunk_inputs):
                    chunks_by_len[len(chunk)].append((batch_indices[i], chunk))
                
                # Process each length group
                for chunk_len, chunk_group in chunks_by_len.items():
                    if self.max_prefill_batch_size > 0 and len(chunk_group) > self.max_prefill_batch_size:
                        for batch_start in range(0, len(chunk_group), self.max_prefill_batch_size):
                            batch_end_idx = min(batch_start + self.max_prefill_batch_size, len(chunk_group))
                            sub_chunk_group = chunk_group[batch_start:batch_end_idx]
                            logits = self._process_prefill_batch(sub_chunk_group, all_states, sequences)
                            
                            for i, (batch_idx, _) in enumerate(sub_chunk_group):
                                seq = sequences[batch_idx]
                                seq.prefill_offset = chunk_end
                                
                            if chunk_end >= prompt_len and seq.prefill_offset >= len(seq.prompt_tokens):
                                last_logits = logits[:, -1, :].to(torch.float32)
                                seq._prefill_logits = last_logits[i].clone()
                    else:
                        logits = self._process_prefill_batch(chunk_group, all_states, sequences)
                        
                        for i, (batch_idx, _) in enumerate(chunk_group):
                            seq = sequences[batch_idx]
                            seq.prefill_offset = chunk_end
                            
                            if chunk_end >= prompt_len and seq.prefill_offset >= len(seq.prompt_tokens):
                                last_logits = logits[:, -1, :].to(torch.float32)
                                seq._prefill_logits = last_logits[i].clone()
    
    def _process_prefill_batch(
        self,
        chunk_group: List[Tuple[int, List[int]]],
        all_states: List[Dict[str, torch.Tensor]],
        sequences: List[GRPOSequence],
    ):
        """Process a prefill batch"""
        batch_input = torch.tensor(
            [chunk for _, chunk in chunk_group],
            dtype=torch.long,
            device=self.device
        )
        
        batch_states = self._extract_batch_states(all_states, [idx for idx, _ in chunk_group])
        
        with torch.no_grad():
            logits, new_states = self.model.forward_with_state(
                batch_input,
                states=batch_states
            )
        
        self._update_batch_states(all_states, new_states, [idx for idx, _ in chunk_group])
        
        return logits
    
    def _batch_decode(
        self,
        sequences: List[GRPOSequence],
        all_states: List[Dict[str, torch.Tensor]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float = 1.0,
        logit_bias: Optional[dict] = None,
    ):
        """Optimized batch decode with vectorized operations"""
        total_seqs = len(sequences)
        finished_mask = [False] * total_seqs
        
        step_times = []
        active_seq_counts = []
        
        for step in range(max_new_tokens):
            step_start_time = time.time()
            
            active_indices = []
            for i in range(total_seqs):
                if not finished_mask[i]:
                    active_indices.append(i)
            
            if not active_indices:
                break
            
            active_sequences = [sequences[i] for i in active_indices]
            active_seq_counts.append(len(active_sequences))
            
            if step % 50 == 0 and step > 0:
                completed = sum(finished_mask)
                progress = completed / total_seqs * 100
                avg_step_time = sum(step_times[-50:]) / min(50, len(step_times))
                print(f"[GRPOBatchGenerator]   - Decode Step {step}: {len(active_sequences)} active, "
                      f"{completed}/{total_seqs} complete ({progress:.1f}%), "
                      f"avg_step: {avg_step_time:.3f}s")
            
            if self.max_decode_batch_size > 0 and len(active_sequences) > self.max_decode_batch_size:
                for batch_start in range(0, len(active_sequences), self.max_decode_batch_size):
                    batch_end = min(batch_start + self.max_decode_batch_size, len(active_sequences))
                    batch_active_seqs = active_sequences[batch_start:batch_end]
                    batch_active_indices = active_indices[batch_start:batch_end]
                    self._process_decode_batch(
                        batch_active_seqs, all_states,
                        temperature, top_p, top_k, repetition_penalty,
                        max_new_tokens, finished_mask, batch_active_indices, logit_bias
                    )
            else:
                self._process_decode_batch(
                    active_sequences, all_states,
                    temperature, top_p, top_k, repetition_penalty,
                    max_new_tokens, finished_mask, active_indices, logit_bias
                )
            
            step_time = time.time() - step_start_time
            step_times.append(step_time)
        
        total_steps = len(step_times)
        avg_step_time = sum(step_times) / len(step_times) if step_times else 0
        min_active = min(active_seq_counts) if active_seq_counts else 0
        max_active = max(active_seq_counts) if active_seq_counts else 0
        avg_active = sum(active_seq_counts) / len(active_seq_counts) if active_seq_counts else 0
        
        print(f"[GRPOBatchGenerator]   - Decode stats:")
        print(f"     * total_steps: {total_steps}")
        print(f"     * avg_step_time: {avg_step_time:.3f}s")
        print(f"     * active_sequences: {min_active} ~ {max_active} (avg: {avg_active:.1f})")
        
        return {
            'total_steps': total_steps,
            'avg_step_time': avg_step_time,
            'min_active_seqs': min_active,
            'max_active_seqs': max_active,
            'avg_active_seqs': avg_active,
        }
    
    def _process_decode_batch(
        self,
        active_sequences: List[GRPOSequence],
        all_states: List[Dict[str, torch.Tensor]],
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        max_new_tokens: int,
        finished_mask: List[bool],
        active_indices: List[int],
        logit_bias: Optional[dict] = None,
    ):
        """Process a decode batch"""
        batch_input_list = []
        valid_sequences = []
        valid_active_indices = []
        
        for idx, seq in enumerate(active_sequences):
            if seq.num_generated > 0:
                batch_input_list.append([seq.generated_tokens[-1]])
                valid_sequences.append(seq)
                valid_active_indices.append(active_indices[idx])
            else:
                if len(seq.prompt_tokens) > 0:
                    batch_input_list.append([seq.prompt_tokens[-1]])
                    valid_sequences.append(seq)
                    valid_active_indices.append(active_indices[idx])
        
        if not batch_input_list:
            return
        
        batch_input = torch.tensor(
            batch_input_list,
            dtype=torch.long,
            device=self.device
        )
        
        batch_indices = [seq.batch_idx for seq in valid_sequences]
        batch_states = self._extract_batch_states(all_states, batch_indices)
        
        with torch.no_grad():
            logits, new_states = self.model.forward_with_state(
                batch_input,
                states=batch_states
            )
        
        self._update_batch_states(all_states, new_states, batch_indices)
        
        next_token_logits = logits[:, -1, :].to(torch.float32)
        # Apply repetition penalty: penalize tokens already in prompt + generated so far
        token_ids_per_row = [
            seq.prompt_tokens + seq.generated_tokens
            for seq in valid_sequences
        ]
        _apply_repetition_penalty(next_token_logits, token_ids_per_row, repetition_penalty)
        
        token_ids, _ = self.sampler.sample_batch(
            next_token_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logit_bias=logit_bias,
        )
        
        for i, seq in enumerate(valid_sequences):
            token_id = token_ids[i].item()
            
            seq.generated_tokens.append(token_id)
            seq.num_generated += 1
            
            if token_id == self.eos_token_id or seq.num_generated >= max_new_tokens:
                seq.is_finished = True
                finished_mask[valid_active_indices[i]] = True
    
    def _extract_batch_states(
        self,
        all_states: List[Dict[str, torch.Tensor]],
        batch_indices: List[int]
    ) -> List[Dict[str, torch.Tensor]]:
        """Extract batch states for specified sequence indices"""
        if len(batch_indices) == 0:
            return []
        
        # Check if indices are contiguous for optimization
        is_contiguous = all(batch_indices[i] == batch_indices[0] + i for i in range(len(batch_indices)))
        
        if is_contiguous and len(batch_indices) > 1:
            start_idx = batch_indices[0]
            end_idx = batch_indices[-1] + 1
            extracted = []
            for layer_state in all_states:
                extracted.append({
                    'att_x_prev': layer_state['att_x_prev'][start_idx:end_idx],
                    'att_kv': layer_state['att_kv'][start_idx:end_idx],
                    'ffn_x_prev': layer_state['ffn_x_prev'][start_idx:end_idx],
                })
            return extracted
        else:
            indices_tensor = torch.tensor(batch_indices, dtype=torch.long, device=self.device)
            extracted = []
            for layer_state in all_states:
                extracted.append({
                    'att_x_prev': layer_state['att_x_prev'][indices_tensor],
                    'att_kv': layer_state['att_kv'][indices_tensor],
                    'ffn_x_prev': layer_state['ffn_x_prev'][indices_tensor],
                })
            return extracted
    
    def _update_batch_states(
        self,
        all_states: List[Dict[str, torch.Tensor]],
        new_states: List[Dict[str, torch.Tensor]],
        batch_indices: List[int]
    ):
        """Update batch states back to all_states"""
        if len(batch_indices) == 0:
            return
        
        is_contiguous = all(batch_indices[i] == batch_indices[0] + i for i in range(len(batch_indices)))
        
        if is_contiguous and len(batch_indices) > 1:
            start_idx = batch_indices[0]
            end_idx = batch_indices[-1] + 1
            for layer_idx, layer_state in enumerate(all_states):
                new_att_x_prev = new_states[layer_idx]['att_x_prev'].to(layer_state['att_x_prev'].dtype)
                new_att_kv = new_states[layer_idx]['att_kv'].to(layer_state['att_kv'].dtype)
                new_ffn_x_prev = new_states[layer_idx]['ffn_x_prev'].to(layer_state['ffn_x_prev'].dtype)
                
                layer_state['att_x_prev'][start_idx:end_idx] = new_att_x_prev
                layer_state['att_kv'][start_idx:end_idx] = new_att_kv
                layer_state['ffn_x_prev'][start_idx:end_idx] = new_ffn_x_prev
        else:
            indices_tensor = torch.tensor(batch_indices, dtype=torch.long, device=self.device)
            for layer_idx, layer_state in enumerate(all_states):
                new_att_x_prev = new_states[layer_idx]['att_x_prev'].to(layer_state['att_x_prev'].dtype)
                new_att_kv = new_states[layer_idx]['att_kv'].to(layer_state['att_kv'].dtype)
                new_ffn_x_prev = new_states[layer_idx]['ffn_x_prev'].to(layer_state['ffn_x_prev'].dtype)
                
                layer_state['att_x_prev'][indices_tensor] = new_att_x_prev
                layer_state['att_kv'][indices_tensor] = new_att_kv
                layer_state['ffn_x_prev'][indices_tensor] = new_ffn_x_prev
    
    def _collect_results(self, sequences: List[GRPOSequence]) -> Dict[str, Any]:
        """Collect generation results"""
        completions = []
        completion_ids_list = []
        
        token_lengths = [len(seq.generated_tokens) for seq in sequences]
        max_len = max(token_lengths) if token_lengths else 0
        
        if max_len == 0:
            completions = [""] * len(sequences)
            completion_ids_list = [[0]] * len(sequences)
        else:
            for seq in sequences:
                tokens_to_decode = seq.generated_tokens
                
                if len(tokens_to_decode) == 0:
                    completion_text = ""
                else:
                    try:
                        completion_text = self.tokenizer.decode(tokens_to_decode)
                    except (KeyError, UnicodeDecodeError) as e:
                        # Tolerant fallback: avoid whole completion becoming single \ufffd
                        tok = getattr(self.tokenizer, "idx2token", None)
                        if tok is not None:
                            parts = [tok.get(tid, b"?") for tid in tokens_to_decode]
                            completion_text = b"".join(parts).decode("utf-8", errors="replace")
                            if not hasattr(GRPOBatchGenerator, "_decode_exc_warn_count"):
                                GRPOBatchGenerator._decode_exc_warn_count = 0
                            GRPOBatchGenerator._decode_exc_warn_count += 1
                            if GRPOBatchGenerator._decode_exc_warn_count <= 20:
                                bad_id = e.args[0] if isinstance(e, KeyError) else None
                                print(
                                    f"[GRPOBatchGenerator._collect_results] [WARN] decode raised {type(e).__name__}: {e}"
                                    + (f" (bad token_id={bad_id})" if bad_id is not None else "")
                                )
                        else:
                            completion_text = ""
                    # Check for decode errors (replacement char indicates invalid token sequence)
                    if completion_text and "\ufffd" in completion_text:
                        if not hasattr(GRPOBatchGenerator, '_decode_fail_warn_count'):
                            GRPOBatchGenerator._decode_fail_warn_count = 0
                        GRPOBatchGenerator._decode_fail_warn_count += 1
                        
                        if GRPOBatchGenerator._decode_fail_warn_count <= 10:
                            print(f"[GRPOBatchGenerator._collect_results] [WARN] Sequence {seq.seq_id} decode failed: "
                                  f"tokens={tokens_to_decode[:20]}{'...' if len(tokens_to_decode) > 20 else ''}, "
                                  f"decoded='{completion_text[:50]}...'")
                        # Keep the decoded text (even if it has errors) so we can see what happened
                
                completions.append(completion_text)
                
                tokens = seq.generated_tokens + [0] * (max_len - len(seq.generated_tokens))
                completion_ids_list.append(tokens)
        
        completion_ids = torch.tensor(
            completion_ids_list,
            dtype=torch.long,
            device=self.device
        )
        
        masks = torch.zeros_like(completion_ids, dtype=torch.bool)
        for i, seq in enumerate(sequences):
            masks[i, :len(seq.generated_tokens)] = True

        return {
            'completions': completions,
            'completion_ids': completion_ids,
            'masks': masks,
        }
