"""
RWKV7 Model - Infinite Context Training Support
Extends model.py with infctx mode
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional

from .model import RWKV7TimeMix, RWKV7ChannelMix, RWKV7Block, RWKV7Model
from .infctx_module import TimeMixState, ChannelMixState, BlockState, BlockStateList
from .operators.wkv_infctx import rwkv7_wkv_infctx


class RWKV7TimeMixInfctx(RWKV7TimeMix):
    """RWKV7 TimeMix with infinite context training support."""
    
    def forward_infctx(
        self, 
        x: torch.Tensor,
        v_first: torch.Tensor,
        last_state: TimeMixState
    ) -> Tuple[torch.Tensor, torch.Tensor, TimeMixState]:
        """Forward pass for infinite context training."""
        B, T, C = x.shape
        H = self.n_head
        N = self.head_size
        
        shift_state = last_state.shift_state
        wkv_state = last_state.wkv_state.contiguous()
        
        shift_state = shift_state.to(x.dtype)
        xx = torch.cat([shift_state.unsqueeze(1), x[:, :-1]], dim=1) - x
        
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g
        
        r = self.receptance(xr)
        w0 = self.w0.to(xw.dtype)
        w1 = self.w1.to(xw.dtype)
        w2 = self.w2.to(xw.dtype)
        w = -torch.nn.functional.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        
        kk = k * self.k_k
        kk = torch.nn.functional.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)
        
        r = r.view(B, T, H, N)
        w = w.view(B, T, H, N)
        k = k.view(B, T, H, N)
        v = v.view(B, T, H, N)
        kk = kk.view(B, T, H, N)
        a = a.view(B, T, H, N)
        
        x_wkv, new_wkv_state = rwkv7_wkv_infctx(
            w=w,
            q=r,
            k=k,
            v=v,
            a=-kk,
            b=kk * a,
            state=wkv_state
        )
        
        x_wkv = x_wkv.view(B, T, C)
        
        original_dtype = x_wkv.dtype
        x_wkv = self.ln_x(x_wkv.view(B * T, C).to(self.ln_x.weight.dtype)).view(B, T, C).to(original_dtype)
        
        residual = ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * 
                    v.view(B, T, H, -1)).view(B, T, C)
        x_wkv = x_wkv + residual.to(x_wkv.dtype)
        
        x_out = self.output((x_wkv * g).to(self.output.weight.dtype))
        
        new_shift_state = x[:, -1, :]
        
        return x_out, v_first, TimeMixState(new_shift_state, new_wkv_state)


class RWKV7ChannelMixInfctx(RWKV7ChannelMix):
    """RWKV7 ChannelMix with infinite context training support."""
    
    def forward_infctx(
        self,
        x: torch.Tensor,
        last_state: ChannelMixState
    ) -> Tuple[torch.Tensor, ChannelMixState]:
        """
        Infinite context forward pass.
        
        Args:
            x: [B, T, C] - Input
            last_state: ChannelMixState - Previous chunk state
        
        Returns:
            x: [B, T, C] - Output
            new_state: ChannelMixState - New state
        """
        shift_state = last_state.shift_state
        
        shift_state = shift_state.to(x.dtype)
        xx = torch.cat([shift_state.unsqueeze(1), x[:, :-1]], dim=1) - x
        
        xk = x + xx * self.x_k
        k = self.key(xk)
        k = torch.relu(k) ** 2
        
        out = self.value(k)
        
        new_shift_state = x[:, -1, :]
        
        return out, ChannelMixState(new_shift_state)


class RWKV7BlockInfctx(nn.Module):
    """RWKV7 Block with infinite context training support."""
    
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)
            self.skip_ln0 = False
        
        self.att = RWKV7TimeMixInfctx(config, layer_id)
        self.ffn = RWKV7ChannelMixInfctx(config, layer_id)
    
    def forward_infctx(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor,
        last_state: BlockState
    ) -> Tuple[torch.Tensor, torch.Tensor, BlockState]:
        """
        Infinite context forward pass.
        
        Args:
            x: [B, T, C] - Input
            v_first: [B, T, C] - v_first state
            last_state: BlockState - Previous chunk state
        
        Returns:
            x: [B, T, C] - Output
            v_first: [B, T, C] - Updated v_first
            new_state: BlockState - New state
        """
        if self.layer_id == 0 and not getattr(self, 'skip_ln0', False):
            x = self.ln0(x)
        
        x_att, v_first, att_state = self.att.forward_infctx(
            self.ln1(x), v_first, last_state.time_mix_state
        )
        
        x = x + x_att
        
        ffn_out, ffn_state = self.ffn.forward_infctx(
            self.ln2(x), last_state.channel_mix_state
        )
        
        x = x + ffn_out
        
        return x, v_first, BlockState(att_state, ffn_state)


class RWKV7Infctx(RWKV7Model):
    """RWKV7 Model with infinite context training support."""
    
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([
            RWKV7BlockInfctx(config, i) for i in range(config.n_layer)
        ])
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self._emb_preprocessed = False
    
    def forward_infctx(
        self,
        idx: torch.Tensor,
        last_shift_states: torch.Tensor,
        last_wkv_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Infinite context forward pass.
        
        Args:
            idx: [B, T] - Input token IDs
            last_shift_states: [N_layer, 2, B, C] - Shift states
            last_wkv_states: [N_layer, B, H, N, N] - WKV states
        
        Returns:
            logits: [B, T, vocab_size] - Output logits
            new_shift_states: [N_layer, 2, B, C] - New shift states
            new_wkv_states: [N_layer, B, H, N, N] - New WKV states
        """
        B, T = idx.size()
        C = self.config.n_embd
        H = self.config.dim_att // self.config.head_size_a
        N = self.config.head_size_a
        
        x = self.emb(idx)
        
        new_states = BlockStateList.empty(
            self.config.n_layer, B, C, H, x.device, x.dtype
        )
        
        v_first = torch.empty_like(x)
        
        for i, (block, block_state) in enumerate(
            zip(self.blocks, BlockStateList(last_shift_states, last_wkv_states))
        ):
            x, v_first, new_block_state = block.forward_infctx(x, v_first, block_state)
            new_states[i] = new_block_state
        
        import os
        debug = os.environ.get('RWKV_DEBUG_GRADIENT', '0') == '1'
        
        if debug:
            print(f"\n[Model Output]")
            print(f"  Before ln_out: x.requires_grad={x.requires_grad}")
        
        x = self.ln_out(x)
        
        if debug:
            print(f"  After ln_out: x.requires_grad={x.requires_grad}")
        
        logits = self.head(x)
        
        if debug:
            print(f"  Final logits: requires_grad={logits.requires_grad}, grad_fn={logits.grad_fn}")
        
        return logits, new_states.shift_states, new_states.wkv_states
