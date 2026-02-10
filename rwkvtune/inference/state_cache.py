"""
State Cache System for RWKV Inference

Intelligent state caching system to accelerate RWKV model inference
by caching intermediate inference states.

Core Features:
- Trie-based indexing with prefix matching
- Three checkpoint types: system_prompt / interval / full_prompt
- Unified LRU eviction policy
- Auto-configuration (calculates optimal params based on memory limits)
- CPU memory storage, saves GPU memory

Reference: ai00_server project cache design
"""

import time
import threading
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from copy import deepcopy


class CacheLevel(Enum):
    """Cache level"""
    NONE = "none"       # Disable cache
    EXACT = "exact"     # Exact match
    PREFIX = "prefix"   # Prefix match (recommended)


class EvictionPolicy(Enum):
    """Eviction policy"""
    LRU = "lru"         # Least Recently Used


@dataclass
class CacheConfig:
    """State Cache configuration"""
    
    # Memory config
    max_cache_memory_gb: float = 4.0
    max_cache_entries: Optional[int] = None
    
    # Cache level
    cache_level: CacheLevel = CacheLevel.PREFIX
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    
    # Checkpoint config
    cache_first_message: bool = True
    
    def __post_init__(self):
        """Parameter validation"""
        if self.max_cache_memory_gb <= 0:
            raise ValueError("max_cache_memory_gb must be positive")


@dataclass(eq=False)
class CachedState:
    """
    Cached state entry
    
    Contains RWKV model state and output, plus metadata.
    """
    state: List[Dict[str, torch.Tensor]]    # RWKV state (CPU tensors)
    output: torch.Tensor                     # Last token logits (CPU tensor)
    token_count: int                         # Token count
    created_at: float                        # Creation time
    last_access: float                       # Last access time
    access_count: int                        # Access count
    memory_bytes: int                        # Memory usage (bytes)
    checkpoint_type: str                     # Checkpoint type: system_prompt / interval / full_prompt
    tokens: List[int] = field(default_factory=list)  # Token sequence (for debugging)
    
    def __eq__(self, other):
        """Use id for comparison to avoid tensor comparison issues"""
        return self is other
    
    def __hash__(self):
        """Use id as hash"""
        return id(self)
    
    def update_access(self):
        """Update access info"""
        self.last_access = time.time()
        self.access_count += 1


class TrieNode:
    """Trie tree node"""
    
    def __init__(self):
        self.children: Dict[int, "TrieNode"] = {}
        self.cached_state: Optional[CachedState] = None


class CacheIndex:
    """
    Trie-based cache index
    
    Supports:
    - Exact match and prefix match
    - Fast insert and lookup
    - Memory statistics
    """
    
    def __init__(self):
        self.root = TrieNode()
        self._entries: List[CachedState] = []
        self._total_memory_bytes: int = 0
        self._lock = threading.RLock()
    
    def insert(self, tokens: List[int], cached_state: CachedState) -> None:
        """Insert cache entry"""
        with self._lock:
            node = self.root
            for token in tokens:
                if token not in node.children:
                    node.children[token] = TrieNode()
                node = node.children[token]
            
            if node.cached_state is not None:
                old_state = node.cached_state
                self._total_memory_bytes -= old_state.memory_bytes
                self._entries.remove(old_state)
            
            node.cached_state = cached_state
            self._entries.append(cached_state)
            self._total_memory_bytes += cached_state.memory_bytes
    
    def find_exact(self, tokens: List[int]) -> Optional[CachedState]:
        """Exact match lookup"""
        with self._lock:
            node = self.root
            for token in tokens:
                if token not in node.children:
                    return None
                node = node.children[token]
            
            if node.cached_state is not None:
                node.cached_state.update_access()
                return node.cached_state
            return None
    
    def find_longest_cached_prefix(
        self, tokens: List[int]
    ) -> Tuple[int, Optional[CachedState]]:
        """
        Find longest cached prefix
        
        Args:
            tokens: Token sequence
        
        Returns:
            (matched token count, CachedState or None)
        """
        with self._lock:
            node = self.root
            last_cached_pos = 0
            last_cached_state = None
            
            for i, token in enumerate(tokens):
                if token not in node.children:
                    break
                node = node.children[token]
                
                if node.cached_state is not None:
                    last_cached_pos = i + 1
                    last_cached_state = node.cached_state
            
            if last_cached_state is not None:
                last_cached_state.update_access()
            
            return last_cached_pos, last_cached_state
    
    def remove(self, cached_state: CachedState) -> bool:
        """Remove cache entry"""
        with self._lock:
            if cached_state not in self._entries:
                return False
            
            tokens = cached_state.tokens
            node = self.root
            path = [node]
            
            for token in tokens:
                if token not in node.children:
                    return False
                node = node.children[token]
                path.append(node)
            
            if node.cached_state is cached_state:
                node.cached_state = None
                self._entries.remove(cached_state)
                self._total_memory_bytes -= cached_state.memory_bytes
                
                # Clean up empty nodes (leaf to root)
                for i in range(len(path) - 1, 0, -1):
                    parent = path[i - 1]
                    current = path[i]
                    token = tokens[i - 1]
                    
                    if not current.children and current.cached_state is None:
                        del parent.children[token]
                    else:
                        break
                
                return True
            
            return False
    
    @property
    def total_memory_bytes(self) -> int:
        """Total memory usage"""
        return self._total_memory_bytes
    
    def __len__(self) -> int:
        return len(self._entries)
    
    def get_all_entries(self) -> List[CachedState]:
        """Get all entries (for eviction)"""
        with self._lock:
            return list(self._entries)


class AutoConfig:
    """Auto-configuration calculator"""
    
    @staticmethod
    def calculate(
        max_memory_gb: float,
        model_state_size_bytes: int,
        safety_factor: float = 0.9,
    ) -> CacheConfig:
        """
        Calculate configuration based on memory limits
        
        Args:
            max_memory_gb: Maximum memory (GB)
            model_state_size_bytes: Memory size per state (bytes)
            safety_factor: Safety factor (default 0.9, leaves 10% margin)
        
        Returns:
            CacheConfig
        """
        max_memory_bytes = int(max_memory_gb * 1e9 * safety_factor)
        max_entries = max(1, max_memory_bytes // model_state_size_bytes)
        
        config = CacheConfig(
            max_cache_memory_gb=max_memory_gb,
            max_cache_entries=max_entries,
        )
        
        print(f"[AutoConfig] Memory limit: {max_memory_gb:.1f} GB")
        print(f"[AutoConfig] State size: {model_state_size_bytes / 1e6:.2f} MB")
        print(f"[AutoConfig] Max entries: {max_entries}")
        
        return config
    
    @staticmethod
    def estimate_state_size(model, device: str = "cpu") -> int:
        """
        Estimate memory size per state
        
        Runs a dummy forward pass to get accurate state size.
        
        Args:
            model: RWKV model
            device: Target device
        
        Returns:
            State memory size (bytes)
        """
        model_device = next(model.parameters()).device
        dummy_input = torch.tensor([[1]], dtype=torch.long, device=model_device)
        
        with torch.no_grad():
            _, state = model.forward_with_state(dummy_input, None)
        
        total_bytes = 0
        for layer_state in state:
            for key, tensor in layer_state.items():
                cpu_tensor = tensor.cpu() if tensor.device.type != 'cpu' else tensor
                total_bytes += cpu_tensor.numel() * cpu_tensor.element_size()
        
        vocab_size = getattr(model.config, 'vocab_size', 65536)
        output_bytes = vocab_size * 4  # float32
        total_bytes += output_bytes
        
        return total_bytes


@dataclass
class CacheStats:
    """Cache statistics"""
    total_hits: int = 0
    total_misses: int = 0
    prefix_hits: int = 0
    exact_hits: int = 0
    total_evictions: int = 0
    total_memory_bytes: int = 0
    entry_count: int = 0
    
    # Stats by checkpoint type
    system_prompt_entries: int = 0
    interval_entries: int = 0
    full_prompt_entries: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Hit rate"""
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "prefix_hits": self.prefix_hits,
            "exact_hits": self.exact_hits,
            "hit_rate": f"{self.hit_rate:.2%}",
            "total_evictions": self.total_evictions,
            "total_memory_mb": f"{self.total_memory_bytes / 1e6:.2f}",
            "entry_count": self.entry_count,
            "checkpoints": {
                "system_prompt": self.system_prompt_entries,
                "interval": self.interval_entries,
                "full_prompt": self.full_prompt_entries,
            }
        }


class StateCache:
    """
    RWKV State Cache System
    
    Main Features:
    - Cache intermediate states to reduce redundant computation
    - Supports Trie prefix matching
    - Smart checkpoints (system_prompt / interval / full_prompt)
    - Unified LRU eviction policy
    
    Usage:
    ```python
    # Create cache
    cache = StateCache(model, config)
    
    # Lookup cache
    cached_pos, state, output = cache.lookup(tokens)
    
    # Compute remaining part
    remaining_tokens = tokens[cached_pos:]
    new_state, new_output = model.forward_with_state(remaining_tokens, state)
    
    # Create checkpoints
    cache.create_checkpoints(tokens, messages, new_state, new_output)
    ```
    """
    
    def __init__(
        self,
        model,
        config: Optional[CacheConfig] = None,
        auto_config: bool = True,
        max_cache_memory_gb: float = 4.0,
    ):
        """
        Initialize State Cache
        
        Args:
            model: RWKV model
            config: Cache config (if provided, ignores auto_config)
            auto_config: Whether to auto-configure (default True)
            max_cache_memory_gb: Max cache memory (only used when auto_config=True)
        """
        self.model = model
        self.device = next(model.parameters()).device
        
        # Config
        if config is not None:
            self.config = config
        elif auto_config:
            state_size = AutoConfig.estimate_state_size(model)
            self.config = AutoConfig.calculate(max_cache_memory_gb, state_size)
        else:
            self.config = CacheConfig(max_cache_memory_gb=max_cache_memory_gb)
        
        # Cache index
        self.index = CacheIndex()
        
        # Statistics
        self.stats = CacheStats()
        
        # Thread lock
        self._lock = threading.RLock()
        
        # Print config
        print(f"\n{'='*60}")
        print(f"State Cache Initialized")
        print(f"{'='*60}")
        print(f"  Cache level: {self.config.cache_level.value}")
        print(f"  Memory limit: {self.config.max_cache_memory_gb:.1f} GB")
        print(f"  Max entries: {self.config.max_cache_entries}")
        print(f"  Cache first message: {self.config.cache_first_message}")
        print(f"  Eviction policy: {self.config.eviction_policy.value}")
        print(f"{'='*60}\n")
    
    def lookup(
        self, tokens: List[int]
    ) -> Tuple[int, Optional[List[Dict[str, torch.Tensor]]], Optional[torch.Tensor]]:
        """
        Lookup cache
        
        Args:
            tokens: Token sequence
        
        Returns:
            (cached_pos, state, output)
            - cached_pos: Token count covered by cache
            - state: RWKV state (if hit)
            - output: Last token logits (if hit)
        """
        if self.config.cache_level == CacheLevel.NONE:
            self.stats.total_misses += 1
            return 0, None, None
        
        with self._lock:
            if self.config.cache_level == CacheLevel.EXACT:
                cached = self.index.find_exact(tokens)
                if cached is not None:
                    self.stats.total_hits += 1
                    self.stats.exact_hits += 1
                    state = self._clone_state_to_device(cached.state)
                    output = cached.output.to(self.device)
                    return len(tokens), state, output
                else:
                    self.stats.total_misses += 1
                    return 0, None, None
            
            else:  # PREFIX
                cached_pos, cached = self.index.find_longest_cached_prefix(tokens)
                if cached is not None:
                    self.stats.total_hits += 1
                    if cached_pos == len(tokens):
                        self.stats.exact_hits += 1
                    else:
                        self.stats.prefix_hits += 1
                    
                    state = self._clone_state_to_device(cached.state)
                    output = cached.output.to(self.device)
                    return cached_pos, state, output
                else:
                    self.stats.total_misses += 1
                    return 0, None, None
    
    def create_checkpoints(
        self,
        tokens: List[int],
        messages: List[Dict[str, str]],
        state: List[Dict[str, torch.Tensor]],
        output: torch.Tensor,
    ) -> None:
        """
        Create checkpoints
        
        Creates the following checkpoints based on config:
        1. system_prompt: First message end position
        2. interval: Every N tokens
        3. full_prompt: Complete prompt end position
        
        Args:
            tokens: Complete token sequence
            messages: Message list (for identifying first message boundary)
            state: State after processing complete prompt
            output: Last token logits
        """
        if self.config.cache_level == CacheLevel.NONE:
            return
        
        with self._lock:
            checkpoints = []
            
            # 1. System Prompt checkpoint
            if self.config.cache_first_message and len(messages) > 0:
                first_msg_tokens = self._estimate_first_message_tokens(tokens, messages)
                
                if first_msg_tokens > 0 and first_msg_tokens < len(tokens):
                    checkpoints.append((first_msg_tokens, "system_prompt"))
            
            # 2. Full Prompt checkpoint
            if not any(cp[0] == len(tokens) for cp in checkpoints):
                checkpoints.append((len(tokens), "full_prompt"))
            
            # Create cache entry for each checkpoint
            for token_pos, checkpoint_type in checkpoints:
                self._create_checkpoint_at(
                    tokens[:token_pos],
                    state,
                    output,
                    checkpoint_type,
                    token_pos,
                    len(tokens),
                )
    
    def _create_checkpoint_at(
        self,
        tokens: List[int],
        full_state: List[Dict[str, torch.Tensor]],
        full_output: torch.Tensor,
        checkpoint_type: str,
        token_pos: int,
        total_tokens: int,
    ) -> None:
        """Create checkpoint at specified position"""
        if token_pos < total_tokens:
            existing = self.index.find_exact(tokens)
            if existing is not None:
                return
            
            if checkpoint_type != "system_prompt":
                return
            
            with torch.no_grad():
                prefix_tokens = torch.tensor([tokens], dtype=torch.long, device=self.device)
                logits, prefix_state = self.model.forward_with_state(prefix_tokens, None)
                prefix_output = logits[:, -1, :].to(torch.float32).squeeze(0)
            
            state_to_cache = prefix_state
            output_to_cache = prefix_output
        else:
            state_to_cache = full_state
            output_to_cache = full_output
        
        # Move to CPU
        cpu_state = self._clone_state_to_cpu(state_to_cache)
        cpu_output = output_to_cache.cpu().clone()
        
        # Calculate memory size
        memory_bytes = self._calculate_state_memory(cpu_state) + \
                       cpu_output.numel() * cpu_output.element_size()
        
        # Create cache entry
        cached = CachedState(
            state=cpu_state,
            output=cpu_output,
            token_count=len(tokens),
            created_at=time.time(),
            last_access=time.time(),
            access_count=0,
            memory_bytes=memory_bytes,
            checkpoint_type=checkpoint_type,
            tokens=list(tokens),
        )
        
        # Check if eviction needed
        self._maybe_evict()
        
        # Insert into index
        self.index.insert(tokens, cached)
        
        # Update statistics
        self._update_stats_after_insert(checkpoint_type)
        
        print(f"[StateCache] Created checkpoint: {checkpoint_type}, "
              f"tokens={len(tokens)}, memory={memory_bytes/1e6:.2f}MB")
    
    def _estimate_first_message_tokens(
        self, tokens: List[int], messages: List[Dict[str, str]]
    ) -> int:
        """
        Estimate first message token count
        
        Simplified strategy: assume first message token ratio equals character ratio
        """
        if len(messages) == 0:
            return 0
        
        first_msg_len = len(messages[0].get("content", ""))
        total_msg_len = sum(len(m.get("content", "")) for m in messages)
        
        if total_msg_len == 0:
            return 0
        
        ratio = first_msg_len / total_msg_len
        estimated = int(len(tokens) * ratio)
        
        return max(1, min(estimated, len(tokens) - 1))
    
    def _maybe_evict(self) -> None:
        """Check and perform eviction"""
        max_memory = int(self.config.max_cache_memory_gb * 1e9)
        max_entries = self.config.max_cache_entries or float('inf')
        
        while self.index.total_memory_bytes > max_memory:
            if not self._evict_one():
                break
        
        while len(self.index) > max_entries:
            if not self._evict_one():
                break
    
    def _evict_one(self) -> bool:
        """Evict one entry (unified LRU policy)"""
        entries = self.index.get_all_entries()
        if not entries:
            return False
        
        # LRU: evict least recently accessed
        victim = min(entries, key=lambda e: e.last_access)
        
        if self.index.remove(victim):
            self.stats.total_evictions += 1
            self._update_stats_after_remove(victim.checkpoint_type)
            print(f"[StateCache] Evicted entry: {victim.checkpoint_type}, "
                  f"tokens={victim.token_count}, "
                  f"last_access={time.time() - victim.last_access:.1f}s ago")
            return True
        
        return False
    
    def _clone_state_to_cpu(
        self, state: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Clone state to CPU"""
        cpu_state = []
        for layer_state in state:
            cpu_layer = {}
            for key, tensor in layer_state.items():
                cpu_layer[key] = tensor.cpu().clone()
            cpu_state.append(cpu_layer)
        return cpu_state
    
    def _clone_state_to_device(
        self, state: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Clone state to target device"""
        device_state = []
        for layer_state in state:
            device_layer = {}
            for key, tensor in layer_state.items():
                device_layer[key] = tensor.to(self.device).clone()
            device_state.append(device_layer)
        return device_state
    
    def _calculate_state_memory(
        self, state: List[Dict[str, torch.Tensor]]
    ) -> int:
        """Calculate state memory size"""
        total_bytes = 0
        for layer_state in state:
            for tensor in layer_state.values():
                total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes
    
    def _update_stats_after_insert(self, checkpoint_type: str) -> None:
        """Update stats after insert"""
        self.stats.entry_count = len(self.index)
        self.stats.total_memory_bytes = self.index.total_memory_bytes
        
        if checkpoint_type == "system_prompt":
            self.stats.system_prompt_entries += 1
        elif checkpoint_type == "interval":
            self.stats.interval_entries += 1
        else:
            self.stats.full_prompt_entries += 1
    
    def _update_stats_after_remove(self, checkpoint_type: str) -> None:
        """Update stats after remove"""
        self.stats.entry_count = len(self.index)
        self.stats.total_memory_bytes = self.index.total_memory_bytes
        
        if checkpoint_type == "system_prompt":
            self.stats.system_prompt_entries -= 1
        elif checkpoint_type == "interval":
            self.stats.interval_entries -= 1
        else:
            self.stats.full_prompt_entries -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.stats.to_dict()
    
    def clear(self) -> None:
        """Clear cache"""
        with self._lock:
            self.index = CacheIndex()
            self.stats = CacheStats()
            print("[StateCache] Cache cleared")
    
    @property
    def is_enabled(self) -> bool:
        """Check if cache is enabled"""
        return self.config.cache_level != CacheLevel.NONE


# === Convenience Functions ===

def create_state_cache(
    model,
    max_cache_memory_gb: float = 4.0,
    cache_level: str = "prefix",
    cache_first_message: bool = True,
) -> StateCache:
    """
    Create State Cache (convenience function)
    
    Args:
        model: RWKV model
        max_cache_memory_gb: Max cache memory (GB)
        cache_level: Cache level (none/exact/prefix)
        cache_first_message: Whether to cache first message
    
    Returns:
        StateCache instance
    
    Note:
        Interval checkpoints are automatically controlled by RWKVEngine's prefill_chunk_size
    """
    level_map = {
        "none": CacheLevel.NONE,
        "exact": CacheLevel.EXACT,
        "prefix": CacheLevel.PREFIX,
    }
    cache_level_enum = level_map.get(cache_level.lower(), CacheLevel.PREFIX)
    
    state_size = AutoConfig.estimate_state_size(model)
    config = AutoConfig.calculate(max_cache_memory_gb, state_size)
    
    config.cache_level = cache_level_enum
    config.cache_first_message = cache_first_message
    
    return StateCache(model, config=config, auto_config=False)
