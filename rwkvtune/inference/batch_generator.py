"""
RWKV Batch Inference Engine - For GRPO Training and Batch Scenarios

This module is a thin wrapper around SchedulerCore, providing convenient
batch generation interfaces.

Features:
- Reuses SchedulerCore's Continuous Batching capability
- Provides simple synchronous interface for offline batch inference
- Supports num_generations > 1 (multiple completions per prompt)
"""

import torch
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from rwkvtune.inference.scheduler_core import (
    SequenceState, 
    SeqStatus,
    create_scheduler_core,
)


# Backward compatibility alias
ReqStatus = SeqStatus


@dataclass
class GenerationConfig:
    """Generation configuration"""
    max_length: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    num_generations: int = 1
    
    # Scheduling config (passed to SchedulerCore)
    max_batch_tokens: int = 8192
    chunk_size: int = 512


class BatchGenerator:
    """
    Batch Inference Engine - Synchronous wrapper for SchedulerCore
    
    Features:
    - Continuous Batching: parallel generation of multiple sequences
    - Chunked Prefill: control memory peak for long prompts
    - State Cache: accelerate repeated inference with same prefix (optional)
    
    Usage:
        generator = BatchGenerator(model, tokenizer, use_state_cache=True)
        results = generator.generate(prompts, generation_config)
    """
    
    def __init__(self, model, tokenizer, config=None, use_state_cache: bool = False, **kwargs):
        """
        Args:
            model: RWKV model
            tokenizer: Tokenizer
            config: Optional config (kept for compatibility)
            use_state_cache: Whether to enable State Cache
            **kwargs: Ignored parameters (kept for compatibility)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        self.eos_id = getattr(tokenizer, 'eos_token_id', None)
        
        # State Cache (optional)
        self.state_cache = None
        if use_state_cache:
            from rwkvtune.inference.state_cache import create_state_cache
            self.state_cache = create_state_cache(
                model=model,
                cache_level="prefix",
            )
            print(f"[OK] BatchGenerator State Cache enabled")
        
        # Pre-create SchedulerCore to avoid reallocating State Pool on each generate()
        self._scheduler = None
    
    def generate(
        self,
        prompts: Optional[List[str]] = None,
        generation_config: GenerationConfig = None,
        input_ids: Optional[List[List[int]]] = None,
    ) -> Dict[str, Any]:
        """
        Batch generation (Continuous Batching)
        
        Args:
            prompts: List of prompt texts [B]
            generation_config: Generation config
            input_ids: Pre-tokenized input [B] (takes priority)
            
        Returns:
            Dict with completions [B*G], completion_ids, masks
        """
        if input_ids is None:
            if prompts is None:
                raise ValueError("Must provide prompts or input_ids")
            input_ids = [self.tokenizer.encode(p) for p in prompts]
        
        B = len(input_ids)
        G = generation_config.num_generations
        total = B * G
        
        # Expand input_ids: copy each prompt G times
        expanded_ids = [list(ids) for ids in input_ids for _ in range(G)]
        
        # Reuse SchedulerCore to avoid reallocating State Pool
        max_batch_size = min(getattr(self.config, "generation_max_batch_size", 256), total)
        
        if self._scheduler is None or \
           self._scheduler.max_batch_size < total or \
           self._scheduler.max_batch_tokens != generation_config.max_batch_tokens or \
           self._scheduler.prefill_chunk_size != generation_config.chunk_size:
            self._scheduler = create_scheduler_core(
                model=self.model,
                tokenizer=self.tokenizer,
                max_batch_size=max_batch_size,
                max_batch_tokens=generation_config.max_batch_tokens,
                prefill_chunk_size=generation_config.chunk_size,
                state_cache=self.state_cache,
            )
        else:
            self._scheduler.reset()
        
        scheduler = self._scheduler
        
        # Log generation parameters
        if len(expanded_ids) > 0:
            print(f"[BatchGenerator] Creating sequences with params:")
            print(f"[BatchGenerator]   - max_new_tokens: {generation_config.max_length}")
            print(f"[BatchGenerator]   - eos_token_id: {self.eos_id}")
            print(f"[BatchGenerator]   - temperature: {generation_config.temperature}")
            print(f"[BatchGenerator]   - top_p: {generation_config.top_p}")
            print(f"[BatchGenerator]   - top_k: {generation_config.top_k}")
            print(f"[BatchGenerator]   - num_sequences: {len(expanded_ids)}")
            if len(expanded_ids) > 0:
                print(f"[BatchGenerator]   - first_prompt_length: {len(expanded_ids[0])} tokens")
        
        sequences = [
            SequenceState(
                seq_id=i,
                prompt_tokens=tokens,
                max_new_tokens=generation_config.max_length,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                eos_token_id=self.eos_id,
            )
            for i, tokens in enumerate(expanded_ids)
        ]
        scheduler.add_sequences(sequences)
        
        # Validate max_new_tokens
        if len(sequences) > 0:
            actual_max_new_tokens = sequences[0].max_new_tokens
            if actual_max_new_tokens != generation_config.max_length:
                print(f"[BatchGenerator] [WARN] Sequence max_new_tokens({actual_max_new_tokens}) != config max_length({generation_config.max_length})")
            else:
                print(f"[BatchGenerator] [OK] Sequence max_new_tokens verified: {actual_max_new_tokens}")
        
        import time
        
        print(f"[BatchGenerator] Starting generation of {total} sequences...")
        gen_start = time.time()
        
        all_tokens = [[] for _ in range(total)]
        update_count = 0
        
        for updates in scheduler.run_to_completion_sync():
            for seq_id, token_id in updates:
                all_tokens[seq_id].append(token_id)
                update_count += 1
            
            if update_count % 1000 == 0:
                elapsed = time.time() - gen_start
                completed_seq = sum(1 for t in all_tokens if t)
                print(f"[BatchGenerator] Progress: {completed_seq}/{total} sequences, "
                      f"{update_count} tokens, {elapsed:.1f}s elapsed")
        
        gen_time = time.time() - gen_start
        avg_len = sum(len(t) for t in all_tokens) / total if total > 0 else 0
        print(f"[BatchGenerator] [OK] Generation complete: {total} sequences, avg_length={avg_len:.1f}, time={gen_time:.2f}s, throughput={total*avg_len/gen_time:.1f} tokens/s")
        
        return self._postprocess(all_tokens, total, generation_config)
    
    def _postprocess(self, all_tokens: List[List[int]], total: int, generation_config: GenerationConfig) -> Dict[str, Any]:
        """Post-process: decode text, convert to tensor"""
        completions = []
        eos_stopped_count = 0
        max_length_stopped_count = 0
        for i, tokens in enumerate(all_tokens):
            original_len = len(tokens)
            if self.eos_id is not None and tokens and tokens[-1] == self.eos_id:
                tokens = tokens[:-1]
                all_tokens[i] = tokens
                eos_stopped_count += 1
            if original_len >= generation_config.max_length:
                max_length_stopped_count += 1
            completions.append(self.tokenizer.decode(tokens))
        
        # Print statistics
        if len(all_tokens) > 0:
            token_lengths = [len(t) for t in all_tokens]
            avg_length = sum(token_lengths) / len(token_lengths)
            print(f"[BatchGenerator] Generation statistics:")
            print(f"[BatchGenerator]   - total_sequences: {len(all_tokens)}")
            print(f"[BatchGenerator]   - avg_token_length: {avg_length:.1f}")
            print(f"[BatchGenerator]   - min_token_length: {min(token_lengths)}")
            print(f"[BatchGenerator]   - max_token_length: {max(token_lengths)}")
            print(f"[BatchGenerator]   - stopped_by_eos: {eos_stopped_count} ({eos_stopped_count/len(all_tokens)*100:.1f}%)")
            print(f"[BatchGenerator]   - stopped_by_max_length: {max_length_stopped_count} ({max_length_stopped_count/len(all_tokens)*100:.1f}%)")
            print(f"[BatchGenerator]   - config_max_length: {generation_config.max_length}")
            if max(token_lengths) < generation_config.max_length * 0.5:
                print(f"[BatchGenerator]   [WARN] max_length({max(token_lengths)}) much smaller than config({generation_config.max_length})")
        
        # Convert to padded tensor
        max_len = max((len(t) for t in all_tokens), default=0)
        
        if max_len > 0:
            padded_ids = []
            masks = []
            for tokens in all_tokens:
                pad_len = max_len - len(tokens)
                padded_ids.append(tokens + [0] * pad_len)
                masks.append([1] * len(tokens) + [0] * pad_len)
            
            completion_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
            masks_tensor = torch.tensor(masks, dtype=torch.float32, device=self.device)
        else:
            completion_ids = torch.zeros((total, 0), dtype=torch.long, device=self.device)
            masks_tensor = torch.zeros((total, 0), dtype=torch.float32, device=self.device)
        
        return {
            'completions': completions,
            'completion_ids': completion_ids,
            'completion_logps': torch.tensor([], device=self.device),
            'masks': masks_tensor,
        }
