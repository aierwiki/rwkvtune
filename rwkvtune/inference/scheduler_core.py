"""
Unified Scheduler Core - Shared by RWKVEngine and BatchGenerator

This module provides low-level Continuous Batching scheduling logic, including:
1. Slot management (allocation, release)
2. State Pool management (GPU state tensors)
3. Prefill/Decode scheduling
4. Chunked Prefill
5. State Cache integration (optional)

Design Principles:
- Provides both sync and async execution modes
- Upper-level interfaces (RWKVEngine, BatchGenerator) can choose as needed
"""

import torch
import collections
import os
import time
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto

# Debug switch
DEBUG_MEMORY = os.environ.get("RWKV_DEBUG_MEMORY", "0") == "1"

def log_memory(tag: str):
    """Print GPU memory usage"""
    if DEBUG_MEMORY and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_alloc = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[MEM] {tag}: alloc={allocated:.2f}GB, reserved={reserved:.2f}GB, max={max_alloc:.2f}GB")

if TYPE_CHECKING:
    from rwkvtune.inference.state_cache import StateCache

class SeqStatus(Enum):
    """Sequence status"""
    WAITING = auto()      # Waiting to be scheduled
    PREFILLING = auto()   # In prefill phase
    RUNNING = auto()      # In decode phase
    FINISHED = auto()     # Completed


@dataclass
class SequenceState:
    """
    Unified sequence state
    
    Supports both RWKVEngine (async serving) and BatchGenerator (offline batch)
    """
    seq_id: Union[int, str]           # Sequence ID (int for batch, str for serving)
    prompt_tokens: List[int]          # Prompt tokens to process
    status: SeqStatus = SeqStatus.WAITING
    
    # Execution state
    prefill_offset: int = 0           # Current prefill position
    generated_tokens: List[int] = field(default_factory=list)
    last_token_id: Optional[int] = None
    
    # Config
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    logit_bias: Optional[dict] = None
    stop: List[str] = field(default_factory=list)
    eos_token_id: Optional[int] = None
    
    # Stop sequence detection state
    generated_text: str = ""
    
    # Resources
    slot_id: int = -1                 # Physical slot index
    
    # State Cache related
    original_tokens: Optional[List[int]] = None
    cached_prefix_len: int = 0
    cached_state: Optional[Any] = None
    cached_output: Optional[torch.Tensor] = None
    
    # Async output (RWKVEngine only)
    output_queue: Optional[Any] = None
    
    # JSON Schema Enforcer (optional)
    enforcer: Optional[Any] = None
    enforcer_finished: bool = False
    
    @property
    def is_finished(self) -> bool:
        """Check if finished"""
        import os
        debug_finish = os.environ.get("RWKV_DEBUG_DECODE", "0") == "1"
        
        if self.enforcer is not None and self.enforcer_finished:
            if debug_finish:
                print(f"[DEBUG SEQ] seq_id={self.seq_id} finished: enforcer_finished")
            return True
        if self.eos_token_id is not None and self.last_token_id == self.eos_token_id:
            if debug_finish:
                print(f"[DEBUG SEQ] seq_id={self.seq_id} finished: EOS token {self.eos_token_id}")
            return True
        if len(self.generated_tokens) >= self.max_new_tokens:
            if debug_finish:
                print(f"[DEBUG SEQ] seq_id={self.seq_id} finished: reached max_new_tokens={self.max_new_tokens}")
            return True
        return False


class SchedulerCore:
    """
    Unified Scheduler Core
    
    Responsibilities:
    - Slot management (allocation, release)
    - State Pool management (GPU state tensors)
    - Prefill/Decode scheduling logic
    - Chunked Prefill
    - State Cache integration (optional)
    
    Usage:
    - BatchGenerator: use step_sync() for synchronous execution
    - RWKVEngine: use step_async() for asynchronous execution
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size: int = 64,
        max_batch_tokens: int = 8192,
        prefill_chunk_size: int = 512,
        state_cache: Optional["StateCache"] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        # Config
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.prefill_chunk_size = prefill_chunk_size
        
        # State Cache
        self.state_cache = state_cache
        if state_cache is not None:
            print(f"[OK] SchedulerCore State Cache integration enabled")
        
        # Get model config
        self.n_layer = model.config.n_layer
        self.n_embd = model.config.n_embd
        self.head_size = getattr(model.config, 'head_size_a', 64)
        dim_att = getattr(model.config, 'dim_att', self.n_embd)
        self.n_head = getattr(model.config, 'n_head', dim_att // self.head_size)
        
        # Initialize State Pool
        log_memory("Before State Pool init")
        self.state_pool = self._init_state_pool()
        log_memory("After State Pool init")
        
        # Slot management
        self.slots: List[Optional[SequenceState]] = [None] * max_batch_size
        self.free_slots = collections.deque(range(max_batch_size))
        
        # Sequence queues
        self.waiting_queue: collections.deque = collections.deque()
        self.prefilling_seqs: List[SequenceState] = []
        self.running_seqs: List[SequenceState] = []
        
        # Sampler
        from rwkvtune.inference.core import TokenSampler
        self.sampler = TokenSampler
        
        # Set max valid token ID to ban out-of-vocab tokens
        if hasattr(tokenizer, 'idx2token') and tokenizer.idx2token:
            TokenSampler._max_valid_token_id = max(tokenizer.idx2token.keys())
        
    def _init_state_pool(self) -> Dict[str, torch.Tensor]:
        """
        Initialize State Pool
        
        RWKV7 State structure:
        - att_x_prev: [n_layer, max_batch, n_embd]
        - att_kv: [n_layer, max_batch, n_head, head_size, head_size]
        - ffn_x_prev: [n_layer, max_batch, n_embd]
        """
        state_dtype = self.dtype
        kv_dtype = torch.float32  # KV state keeps FP32 for precision
        
        return {
            "att_x_prev": torch.zeros(
                (self.n_layer, self.max_batch_size, self.n_embd),
                dtype=state_dtype, device=self.device
            ),
            "att_kv": torch.zeros(
                (self.n_layer, self.max_batch_size, self.n_head, self.head_size, self.head_size),
                dtype=kv_dtype, device=self.device
            ),
            "ffn_x_prev": torch.zeros(
                (self.n_layer, self.max_batch_size, self.n_embd),
                dtype=state_dtype, device=self.device
            ),
        }
    
    # =========================================================================
    # Public Interface
    # =========================================================================
    
    def reset(self) -> None:
        """
        Reset scheduler state (reuse State Pool, only clear queues)
        
        Used to reuse resources when calling generate() multiple times on same LLM instance
        """
        self.waiting_queue.clear()
        self.prefilling_seqs.clear()
        self.running_seqs.clear()
        
        self.slots = [None] * self.max_batch_size
        self.free_slots = collections.deque(range(self.max_batch_size))
    
    def add_sequences(self, sequences: List[SequenceState]) -> None:
        """Add sequences to waiting queue in batch"""
        for seq in sequences:
            self.waiting_queue.append(seq)
    
    def add_sequence(self, seq: SequenceState) -> None:
        """Add single sequence to waiting queue"""
        self.waiting_queue.append(seq)
    
    def has_pending_work(self) -> bool:
        """Check if there is pending work"""
        return bool(self.waiting_queue or self.prefilling_seqs or self.running_seqs)
    
    def get_stats(self) -> Dict[str, int]:
        """Get current state statistics"""
        return {
            "waiting": len(self.waiting_queue),
            "prefilling": len(self.prefilling_seqs),
            "running": len(self.running_seqs),
            "free_slots": len(self.free_slots),
        }
    
    # =========================================================================
    # Core Scheduling Logic
    # =========================================================================
    
    def _schedule(self) -> Tuple[List[SequenceState], List[Tuple[SequenceState, List[int]]]]:
        """
        Schedule one step of work
        
        Returns:
            decode_batch: List of sequences to decode
            prefill_batch: List of (sequence, chunk) to prefill
        """
        current_budget = self.max_batch_tokens
        
        # 1. Allocate slots from waiting queue
        while self.waiting_queue and self.free_slots:
            seq = self.waiting_queue.popleft()
            slot_id = self.free_slots.popleft()
            seq.slot_id = slot_id
            seq.status = SeqStatus.PREFILLING
            self.slots[slot_id] = seq
            self.prefilling_seqs.append(seq)
            
            # Initialize slot state (may directly move to RUNNING on cache hit)
            self._init_slot_state(seq, slot_id)
            
            # If cache fully hit, move to running_seqs
            if seq.status == SeqStatus.RUNNING:
                self.prefilling_seqs.remove(seq)
                self.running_seqs.append(seq)
        
        # 2. Schedule Decode (higher priority)
        decode_batch: List[SequenceState] = []
        
        running_seqs_list = list(self.running_seqs)
        for seq in running_seqs_list:
            if len(decode_batch) < self.max_batch_size:
                decode_batch.append(seq)
            else:
                break
        
        # 3. Schedule Prefill (use remaining budget)
        prefill_batch: List[Tuple[SequenceState, List[int]]] = []
        for seq in list(self.prefilling_seqs):
            if current_budget <= 0:
                break
            remaining = len(seq.prompt_tokens) - seq.prefill_offset
            chunk_len = min(remaining, self.prefill_chunk_size, current_budget)
            if chunk_len > 0:
                chunk = seq.prompt_tokens[seq.prefill_offset : seq.prefill_offset + chunk_len]
                prefill_batch.append((seq, chunk))
                current_budget -= chunk_len
        
        return decode_batch, prefill_batch
    
    def _init_slot_state(self, seq: SequenceState, slot_id: int) -> None:
        """
        Initialize slot state
        
        - Query State Cache (if enabled)
        - If cache hit, load state to slot
        - Otherwise reset to 0
        """
        # Query State Cache
        if self.state_cache is not None and seq.prompt_tokens:
            prefix_len, state, output = self.state_cache.lookup(seq.prompt_tokens)
            if prefix_len > 0 and state is not None:
                seq.cached_state = state
                seq.cached_prefix_len = prefix_len
                seq.prefill_offset = prefix_len
        
        # Load state to slot
        if seq.cached_state is not None:
            self._load_state_to_slot(slot_id, seq.cached_state)
            seq.cached_state = None
            
            # If cache fully hit, directly move to Decode
            if seq.prefill_offset >= len(seq.prompt_tokens):
                seq.status = SeqStatus.RUNNING
                seq.last_token_id = seq.prompt_tokens[-1]
        else:
            self._reset_slot_state(slot_id)
    
    def _reset_slot_state(self, slot_id: int) -> None:
        """Reset slot state to 0"""
        idx = torch.tensor([slot_id], device=self.device, dtype=torch.long)
        self.state_pool["att_x_prev"][:, idx, :] = 0
        self.state_pool["att_kv"][:, idx, ...] = 0
        self.state_pool["ffn_x_prev"][:, idx, :] = 0
    
    def _load_state_to_slot(self, slot_id: int, state: List[Dict[str, torch.Tensor]]) -> None:
        """Load CPU state to GPU slot"""
        with torch.no_grad():
            for layer_idx, layer_state in enumerate(state):
                if 'att_x_prev' in layer_state:
                    self.state_pool["att_x_prev"][layer_idx, slot_id, :] = \
                        layer_state['att_x_prev'].to(self.device, dtype=self.dtype)
                if 'att_kv' in layer_state:
                    self.state_pool["att_kv"][layer_idx, slot_id, ...] = \
                        layer_state['att_kv'].to(self.device, dtype=torch.float32)
                if 'ffn_x_prev' in layer_state:
                    self.state_pool["ffn_x_prev"][layer_idx, slot_id, :] = \
                        layer_state['ffn_x_prev'].to(self.device, dtype=self.dtype)
    
    def _back_state_from_slot(self, slot_id: int) -> List[Dict[str, torch.Tensor]]:
        """Backup GPU slot state to CPU"""
        state = []
        with torch.no_grad():
            for layer_idx in range(self.n_layer):
                layer_state = {
                    'att_x_prev': self.state_pool["att_x_prev"][layer_idx, slot_id, :].cpu().clone(),
                    'att_kv': self.state_pool["att_kv"][layer_idx, slot_id, ...].cpu().clone(),
                    'ffn_x_prev': self.state_pool["ffn_x_prev"][layer_idx, slot_id, :].cpu().clone(),
                }
                state.append(layer_state)
        return state
    
    def _finish_sequence(self, seq: SequenceState) -> None:
        """Finish sequence, release resources"""
        seq.status = SeqStatus.FINISHED
        if seq in self.running_seqs:
            self.running_seqs.remove(seq)
        if seq in self.prefilling_seqs:
            self.prefilling_seqs.remove(seq)
        self.slots[seq.slot_id] = None
        self.free_slots.append(seq.slot_id)
    
    # =========================================================================
    # Execution Methods
    # =========================================================================
    
    def _gather_states(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather states for given slot indices"""
        idx_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
        return (
            self.state_pool["att_x_prev"][:, idx_tensor, :],
            self.state_pool["att_kv"][:, idx_tensor, ...],
            self.state_pool["ffn_x_prev"][:, idx_tensor, :],
        )
    
    def _scatter_states(
        self,
        indices: List[int],
        new_states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """Scatter states back to pool"""
        idx_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
        self.state_pool["att_x_prev"][:, idx_tensor, :] = new_states[0].to(dtype=self.dtype)
        self.state_pool["att_kv"][:, idx_tensor, ...] = new_states[1].to(dtype=torch.float32)
        self.state_pool["ffn_x_prev"][:, idx_tensor, :] = new_states[2].to(dtype=self.dtype)
    
    def _model_forward_with_state(
        self,
        input_ids: torch.Tensor,
        att_x_prevs: torch.Tensor,
        att_kvs: torch.Tensor,
        ffn_x_prevs: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Execute model forward with state
        
        Args:
            input_ids: [B, T]
            att_x_prevs: [L, B, C]
            att_kvs: [L, B, H, N, N]
            ffn_x_prevs: [L, B, C]
            
        Returns:
            final_x: [B, T, C] - last layer hidden states
            new_states: (att_x_prevs, att_kvs, ffn_x_prevs)
        """
        x = self.model.emb(input_ids)
        v_first = torch.empty_like(x)
        
        new_att_x_prevs = []
        new_att_kvs = []
        new_ffn_x_prevs = []
        
        for i, block in enumerate(self.model.blocks):
            layer_state = {
                'att_x_prev': att_x_prevs[i],
                'att_kv': att_kvs[i],
                'ffn_x_prev': ffn_x_prevs[i]
            }
            
            x, v_first, new_layer_state = block.forward_with_state(x, v_first, layer_state)
            
            new_att_x_prevs.append(new_layer_state['att_x_prev'])
            new_att_kvs.append(new_layer_state['att_kv'])
            new_ffn_x_prevs.append(new_layer_state['ffn_x_prev'])
        
        x = self.model.ln_out(x)
        
        return x, (
            torch.stack(new_att_x_prevs),
            torch.stack(new_att_kvs),
            torch.stack(new_ffn_x_prevs)
        )
    
    def _execute_decode(self, decode_batch: List[SequenceState]) -> List[Tuple[int, int]]:
        """
        Execute Decode
        
        Returns:
            new_tokens: [(seq_id, token_id), ...]
        """
        if not decode_batch:
            return []
        
        import time
        start_time = time.time()
        
        if DEBUG_MEMORY:
            log_memory(f"Before decode batch (batch={len(decode_batch)})")
        
        indices = [s.slot_id for s in decode_batch]
        att_x, att_kv, ffn_x = self._gather_states(indices)
        
        input_tensor = torch.tensor(
            [[s.last_token_id] for s in decode_batch],
            dtype=torch.long, device=self.device
        )
        
        final_x, new_states = self._model_forward_with_state(
            input_tensor, att_x, att_kv, ffn_x
        )
        self._scatter_states(indices, new_states)
        
        if DEBUG_MEMORY:
            log_memory(f"After decode forward (batch={len(decode_batch)})")
        
        import os
        debug_decode = os.environ.get("RWKV_DEBUG_DECODE", "0") == "1"
        
        logits = self.model.head(final_x[:, -1, :]).to(torch.float32)
        new_tokens = []
        
        for i, seq in enumerate(decode_batch):
            token_id, _ = self.sampler.sample(
                logits[i],
                temperature=seq.temperature,
                top_p=seq.top_p,
                top_k=seq.top_k,
                logit_bias=seq.logit_bias
            )
            
            if debug_decode:
                gen_len = len(seq.generated_tokens)
                if seq.seq_id == 0 and (gen_len < 50 or gen_len % 100 == 0):
                    print(f"[DEBUG SAMPLE] seq_id={seq.seq_id}, step={gen_len}, token_id={token_id}, eos={seq.eos_token_id}")
                if token_id >= 65529:
                    print(f"[DEBUG SAMPLE] [WARN] seq_id={seq.seq_id} sampled special token: {token_id}")
            
            seq.last_token_id = token_id
            seq.generated_tokens.append(token_id)
            new_tokens.append((seq.seq_id, token_id))
            
            if seq.is_finished:
                self._finish_sequence(seq)
        
        elapsed = time.time() - start_time
        if len(decode_batch) > 0:
            if len(new_tokens) != len(decode_batch):
                print(f"[Scheduler] [WARN] Decode batch={len(decode_batch)}, but only {len(new_tokens)} tokens returned, time={elapsed:.3f}s")
        
        return new_tokens
    
    def _execute_prefill(
        self,
        prefill_batch: List[Tuple[SequenceState, List[int]]]
    ) -> List[Tuple[int, int]]:
        """
        Execute Prefill
        
        Groups by length to avoid padding overhead
        
        Returns:
            new_tokens: [(seq_id, token_id), ...] - first generated token for sequences completing prefill
        """
        if not prefill_batch:
            return []
        
        import time
        start_time = time.time()
        total_tokens = sum(len(chunk) for _, chunk in prefill_batch)
        
        new_tokens = []
        
        # Group by chunk length
        jobs_by_len = collections.defaultdict(list)
        for seq, chunk in prefill_batch:
            jobs_by_len[len(chunk)].append((seq, chunk))
        
        for length, jobs in jobs_by_len.items():
            indices = [j[0].slot_id for j in jobs]
            chunks = [j[1] for j in jobs]
            
            if DEBUG_MEMORY:
                log_memory(f"Before prefill batch (len={length}, batch={len(jobs)})")
            
            att_x, att_kv, ffn_x = self._gather_states(indices)
            input_tensor = torch.tensor(chunks, dtype=torch.long, device=self.device)
            
            if DEBUG_MEMORY:
                print(f"[DEBUG] Prefill: input_tensor.shape={input_tensor.shape}, "
                      f"att_x.shape={att_x.shape}, att_kv.shape={att_kv.shape}")
            
            final_x, new_states = self._model_forward_with_state(
                input_tensor, att_x, att_kv, ffn_x
            )
            
            if DEBUG_MEMORY:
                log_memory(f"After prefill forward (len={length})")
            
            self._scatter_states(indices, new_states)
            
            # Update state and sample (if prefill complete)
            for i, (seq, chunk) in enumerate(jobs):
                seq.prefill_offset += len(chunk)
                
                if seq.prefill_offset >= len(seq.prompt_tokens):
                    seq.status = SeqStatus.RUNNING
                    self.prefilling_seqs.remove(seq)
                    self.running_seqs.append(seq)
                    
                    # Insert into State Cache (if enabled)
                    if self.state_cache is not None and seq.prompt_tokens:
                        state_backup = self._back_state_from_slot(seq.slot_id)
                        self.state_cache.create_checkpoints(
                            tokens=seq.prompt_tokens,
                            messages=[],
                            state=state_backup,
                            output=final_x[i, -1, :].cpu(),
                        )
                    
                    # Sample first token
                    logits = self.model.head(final_x[i, -1, :]).to(torch.float32)
                    token_id, _ = self.sampler.sample(
                        logits,
                        temperature=seq.temperature,
                        top_p=seq.top_p,
                        top_k=seq.top_k,
                        logit_bias=seq.logit_bias
                    )
                    
                    seq.last_token_id = token_id
                    seq.generated_tokens.append(token_id)
                    new_tokens.append((seq.seq_id, token_id))
                    
                    if seq.is_finished:
                        self._finish_sequence(seq)
            
            del final_x, new_states, att_x, att_kv, ffn_x, input_tensor
        
        return new_tokens
    
    # =========================================================================
    # Synchronous Execution Interface (for BatchGenerator)
    # =========================================================================
    
    def step_sync(self) -> List[Tuple[int, int]]:
        """
        Execute one step synchronously
        
        Returns:
            new_tokens: [(seq_id, token_id), ...]
        """
        max_attempts = 10
        
        for attempt in range(max_attempts):
            try:
                with torch.no_grad():
                    decode_batch, prefill_batch = self._schedule()
                    
                    if not decode_batch and not prefill_batch:
                        if not self.has_pending_work():
                            return []
                        if attempt < 5:
                            time.sleep(0.01)
                            continue
                        else:
                            return []
                    
                    new_tokens = []
                    
                    if decode_batch:
                        decode_result = self._execute_decode(decode_batch)
                        new_tokens.extend(decode_result)
                    
                    if prefill_batch:
                        new_tokens.extend(self._execute_prefill(prefill_batch))
                    
                    if hasattr(self, '_step_counter'):
                        self._step_counter += 1
                    else:
                        self._step_counter = 1
                    
                    if self._step_counter % 10 == 0:
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    return new_tokens
                    
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e):
                    print(f"[ERROR] step_sync attempt {attempt} failed with CUDA error: {e}")
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(0.5)
                else:
                    raise
        
        return []
    
    def run_to_completion_sync(self) -> Iterator[List[Tuple[int, int]]]:
        """
        Run synchronously until all sequences complete
        
        Yields:
            Each step's new tokens: [(seq_id, token_id), ...]
        """
        import time
        start_time = time.time()
        step_count = 0
        
        # Get all sequences
        all_sequences = []
        all_sequences.extend(self.waiting_queue)
        all_sequences.extend(self.prefilling_seqs)
        all_sequences.extend(self.running_seqs)
        all_sequences.extend([seq for seq in self.slots if seq is not None])
        unique_sequences = list({id(seq): seq for seq in all_sequences}.values())
        
        if len(unique_sequences) > 0:
            print(f"[Scheduler] Starting generation of {len(unique_sequences)} sequences...")
        
        last_progress_step = 0
        total_generated_tokens = 0
        
        while self.has_pending_work():
            step_start = time.time()
            stats_before = self.get_stats()
            
            new_tokens = self.step_sync()
            
            if new_tokens:
                last_progress_step = step_count
                total_generated_tokens += len(new_tokens)
            
            step_time = time.time() - step_start
            if step_time > 5.0:
                print(f"[WARNING] Step {step_count} took {step_time:.2f}s! "
                      f"waiting={stats_before['waiting']}, prefilling={stats_before['prefilling']}, running={stats_before['running']}")
            
            if new_tokens:
                yield new_tokens
            step_count += 1
            
            if step_count > 100000:
                print(f"[ERROR] Too many steps ({step_count}), breaking to prevent infinite loop")
                final_stats = self.get_stats()
                print(f"[ERROR] Final stats: waiting={final_stats['waiting']}, prefilling={final_stats['prefilling']}, running={final_stats['running']}")
                break
            
            if step_count % 100 == 0:
                stats_after = self.get_stats()
                elapsed = time.time() - start_time
                print(f"[Scheduler] Step {step_count}, elapsed={elapsed:.1f}s, waiting={stats_after['waiting']}, prefilling={stats_after['prefilling']}, running={stats_after['running']}, total_tokens={total_generated_tokens}")
                
                steps_since_progress = step_count - last_progress_step
                if steps_since_progress > 500 and elapsed > 60:
                    print(f"[ERROR] No progress for {steps_since_progress} steps (>60s), breaking to prevent hang")
                    break
        
        elapsed = time.time() - start_time
        final_stats = self.get_stats()
        print(f"[Scheduler] [OK] Generation complete, total_steps: {step_count}, total_time: {elapsed:.2f}s")
        print(f"[Scheduler]   Final state: waiting={final_stats['waiting']}, prefilling={final_stats['prefilling']}, running={final_stats['running']}")
        
        # Analyze stop reasons
        all_sequences = [seq for seq in self.slots if seq is not None]
        all_sequences.extend(self.waiting_queue)
        all_sequences.extend(self.prefilling_seqs)
        all_sequences.extend(self.running_seqs)
        unique_sequences = list({id(seq): seq for seq in all_sequences}.values())
        
        if len(unique_sequences) > 0:
            eos_stopped = 0
            max_length_stopped = 0
            enforcer_stopped = 0
            unknown_stopped = 0
            
            for seq in unique_sequences:
                if seq.status == SeqStatus.FINISHED:
                    if seq.enforcer is not None and seq.enforcer_finished:
                        enforcer_stopped += 1
                    elif seq.eos_token_id is not None and len(seq.generated_tokens) > 0 and seq.generated_tokens[-1] == seq.eos_token_id:
                        eos_stopped += 1
                    elif len(seq.generated_tokens) >= seq.max_new_tokens:
                        max_length_stopped += 1
                    else:
                        unknown_stopped += 1
            
            total_finished = eos_stopped + max_length_stopped + enforcer_stopped + unknown_stopped
            if total_finished > 0:
                print(f"[Scheduler] Stop reason statistics ({total_finished} sequences):")
                print(f"[Scheduler]   - stopped_by_eos: {eos_stopped} ({eos_stopped/total_finished*100:.1f}%)")
                print(f"[Scheduler]   - stopped_by_max_length: {max_length_stopped} ({max_length_stopped/total_finished*100:.1f}%)")
                print(f"[Scheduler]   - stopped_by_enforcer: {enforcer_stopped} ({enforcer_stopped/total_finished*100:.1f}%)")
                print(f"[Scheduler]   - unknown: {unknown_stopped} ({unknown_stopped/total_finished*100:.1f}%)")
            
            gen_lengths = [len(seq.generated_tokens) for seq in unique_sequences if seq.status == SeqStatus.FINISHED]
            if gen_lengths:
                print(f"[Scheduler] Generation length statistics:")
                print(f"[Scheduler]   - min_length: {min(gen_lengths)} tokens")
                print(f"[Scheduler]   - max_length: {max(gen_lengths)} tokens")
                print(f"[Scheduler]   - avg_length: {sum(gen_lengths)/len(gen_lengths):.1f} tokens")
        
        total_time = time.time() - start_time
        print(f"[Scheduler] [OK] Completed in {total_time:.2f}s, total_steps={step_count}")
    
    # =========================================================================
    # Asynchronous Execution Interface (for RWKVEngine)
    # =========================================================================
    
    async def step_async(self) -> List[Tuple[Union[int, str], int]]:
        """
        Execute one step asynchronously
        
        Returns:
            new_tokens: [(seq_id, token_id), ...]
        """
        return self.step_sync()


# =========================================================================
# Convenience Factory Function
# =========================================================================

def create_scheduler_core(
    model,
    tokenizer,
    max_batch_size: int = 64,
    max_batch_tokens: int = 8192,
    prefill_chunk_size: int = 512,
    state_cache: Optional["StateCache"] = None,
) -> SchedulerCore:
    """
    Create SchedulerCore instance
    
    Args:
        model: RWKV model
        tokenizer: Tokenizer
        max_batch_size: Maximum concurrent sequences
        max_batch_tokens: Maximum tokens per forward pass
        prefill_chunk_size: Chunked Prefill chunk size
        state_cache: State Cache instance (optional)
    """
    return SchedulerCore(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
        max_batch_tokens=max_batch_tokens,
        prefill_chunk_size=prefill_chunk_size,
        state_cache=state_cache,
    )
