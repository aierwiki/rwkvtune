"""
RWKV7 Infinite Context Training - State Management Module
"""
import torch
from typing import Optional


class TimeMixState:
    """TimeMix layer state."""
    
    def __init__(self, shift_state: torch.Tensor, wkv_state: torch.Tensor):
        """
        Args:
            shift_state: [B, C] - Time shift state
            wkv_state: [B, H, N, N] - WKV state matrix
        """
        self.shift_state = shift_state
        self.wkv_state = wkv_state


class ChannelMixState:
    """ChannelMix layer state."""
    
    def __init__(self, shift_state: torch.Tensor):
        """
        Args:
            shift_state: [B, C] - Time shift state
        """
        self.shift_state = shift_state


class BlockState:
    """Single block state."""
    
    def __init__(
        self, 
        time_mix_state: TimeMixState,
        channel_mix_state: ChannelMixState
    ):
        """
        Args:
            time_mix_state: TimeMix state
            channel_mix_state: ChannelMix state
        """
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state


class BlockStateList:
    """Manages states for all layers."""
    
    def __init__(self, shift_states: torch.Tensor, wkv_states: torch.Tensor):
        """
        Args:
            shift_states: [N_layer, 2, B, C] - All layers' shift states
                         [layer, 0]: TimeMix shift_state
                         [layer, 1]: ChannelMix shift_state
            wkv_states: [N_layer, B, H, N, N] - All layers' WKV states
        """
        self.shift_states = shift_states
        self.wkv_states = wkv_states
    
    @staticmethod
    def create(
        n_layer: int,
        batch_size: int,
        n_embd: int,
        n_head: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16
    ) -> 'BlockStateList':
        """
        Create zero-initialized state.
        
        Args:
            n_layer: Number of layers
            batch_size: Batch size
            n_embd: Model dimension
            n_head: Number of attention heads
            device: Device
            dtype: Data type
        
        Returns:
            Initialized BlockStateList
        """
        head_size = n_embd // n_head
        
        # WKV state must use float32 for high-precision accumulation in CUDA kernel
        wkv_states = torch.zeros(
            (n_layer, batch_size, n_head, head_size, head_size),
            device=device,
            dtype=torch.float32,
            requires_grad=True
        )
        
        shift_states = torch.zeros(
            (n_layer, 2, batch_size, n_embd),
            device=device,
            dtype=dtype,
            requires_grad=True
        )
        
        return BlockStateList(shift_states, wkv_states)
    
    @staticmethod
    def empty(
        n_layer: int,
        batch_size: int,
        n_embd: int,
        n_head: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16
    ) -> 'BlockStateList':
        """
        Create uninitialized empty state (for performance optimization).
        
        Args:
            n_layer: Number of layers
            batch_size: Batch size
            n_embd: Model dimension
            n_head: Number of attention heads
            device: Device
            dtype: Data type
        
        Returns:
            Uninitialized BlockStateList
        """
        head_size = n_embd // n_head
        
        wkv_states = torch.empty(
            (n_layer, batch_size, n_head, head_size, head_size),
            device=device,
            dtype=torch.float32
        )
        
        shift_states = torch.empty(
            (n_layer, 2, batch_size, n_embd),
            device=device,
            dtype=dtype
        )
        
        return BlockStateList(shift_states, wkv_states)
    
    def __getitem__(self, layer: int) -> BlockState:
        """
        Get state for specified layer.
        
        Args:
            layer: Layer index
        
        Returns:
            BlockState
        """
        return BlockState(
            TimeMixState(
                self.shift_states[layer, 0],
                self.wkv_states[layer]
            ),
            ChannelMixState(
                self.shift_states[layer, 1]
            )
        )
    
    def __setitem__(self, layer: int, state: BlockState):
        """
        Set state for specified layer.
        
        Args:
            layer: Layer index
            state: BlockState
        """
        self.shift_states[layer, 0] = state.time_mix_state.shift_state
        self.wkv_states[layer] = state.time_mix_state.wkv_state
        self.shift_states[layer, 1] = state.channel_mix_state.shift_state
    
    def detach(self) -> 'BlockStateList':
        """
        Detach all states (stop gradient propagation).
        
        Returns:
            New BlockStateList (detached)
        """
        return BlockStateList(
            self.shift_states.clone().detach(),
            self.wkv_states.clone().detach()
        )
    
    def to(self, device: torch.device) -> 'BlockStateList':
        """
        Move to specified device.
        
        Args:
            device: Target device
        
        Returns:
            New BlockStateList
        """
        return BlockStateList(
            self.shift_states.to(device),
            self.wkv_states.to(device)
        )
