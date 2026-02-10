"""
RWKV7 Infinite Context - Naive PyTorch Implementation

Based on FLA's correct gradient derivation, implemented in PyTorch.
For training (correctness first), CUDA optimized version can be added later.
"""
import torch
from typing import Optional, Tuple


def naive_recurrent_rwkv7_fwd(
    q: torch.Tensor,  # [B, T, H, C]
    k: torch.Tensor,  # [B, T, H, C]
    v: torch.Tensor,  # [B, T, H, C]
    w: torch.Tensor,  # [B, T, H, C]
    a: torch.Tensor,  # [B, T, H, C]
    b: torch.Tensor,  # [B, T, H, C]
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None  # [B, H, C, C]
):
    """
    Pure PyTorch RWKV7 WKV forward pass (recurrent form).
    
    Supports any dtype, computes internally in float32, outputs float32.
    """
    input_dtype = q.dtype
    
    if input_dtype != torch.float32:
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        w = w.to(torch.float32)
        a = a.to(torch.float32)
        b = b.to(torch.float32)
        if initial_state is not None and initial_state.dtype != torch.float32:
            initial_state = initial_state.to(torch.float32)
    
    B, T, H, C = q.shape
    
    if initial_state is None:
        state = torch.zeros(B, H, C, C, dtype=torch.float32, device=q.device)
    else:
        state = initial_state
    
    outputs = []
    
    for t in range(T):
        q_t = q[:, t, :, :] * scale
        k_t = k[:, t, :, :]
        v_t = v[:, t, :, :]
        w_t_raw = w[:, t, :, :]
        a_t = a[:, t, :, :]
        b_t = b[:, t, :, :]
        
        w_t = torch.exp(-torch.exp(w_t_raw))
        
        sa = (state * a_t[:, :, None, :]).sum(dim=-1)
        
        state = (
            w_t[:, :, None, :] * state +
            sa[:, :, :, None] * b_t[:, :, None, :] +
            k_t[:, :, None, :] * v_t[:, :, :, None]
        )
        
        output_t = (state * q_t[:, :, None, :]).sum(dim=-1)
        outputs.append(output_t)
    
    output = torch.stack(outputs, dim=1)
    
    return output, state


def naive_recurrent_rwkv7_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    doutput: torch.Tensor,
    dh_t: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None
) -> Tuple:
    """
    RWKV7 backward pass - Naive implementation.
    
    Based on FLA's correct gradient derivation.
    
    Args:
        q, k, v, w, a, b: [B, T, H, K] forward inputs
        doutput: [B, T, H, V] output gradient
        dh_t: [B, H, V, K] final state gradient (optional)
        scale: scaling factor
        initial_state: [B, H, V, K] initial state (optional)
    
    Returns:
        dq, dk, dv, dw, da, db, dh0: gradients
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    
    dtype = torch.float32
    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)
    w = w.to(dtype)
    a = a.to(dtype)
    b = b.to(dtype)
    doutput = doutput.to(dtype)
    
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dw = torch.zeros_like(w)
    da = torch.zeros_like(a)
    db = torch.zeros_like(b)
    
    dstate = torch.zeros(B, H, V, K, dtype=dtype, device=q.device)
    if dh_t is not None:
        dstate = dstate + dh_t.to(dtype)
    
    # Rebuild all states (forward pass)
    states = []
    if initial_state is None:
        state = torch.zeros(B, H, V, K, dtype=dtype, device=q.device)
    else:
        state = initial_state.to(dtype)
    states.append(state.clone())
    
    for t in range(T):
        q_t = q[:, t, :, :] * scale
        k_t = k[:, t, :, :]
        v_t = v[:, t, :, :]
        w_t_raw = w[:, t, :, :]
        a_t = a[:, t, :, :]
        b_t = b[:, t, :, :]
        
        w_t = torch.exp(-torch.exp(w_t_raw))
        sa = (state * a_t[:, :, None, :]).sum(dim=-1)
        
        state = (
            w_t[:, :, None, :] * state +
            sa[:, :, :, None] * b_t[:, :, None, :] +
            k_t[:, :, None, :] * v_t[:, :, :, None]
        )
        states.append(state.clone())
    
    # Backward pass (reverse time)
    for t in range(T-1, -1, -1):
        q_t = q[:, t, :, :] * scale
        k_t = k[:, t, :, :]
        v_t = v[:, t, :, :]
        w_scalar = w[:, t, :, :]
        a_t = a[:, t, :, :]
        b_t = b[:, t, :, :]
        
        w_exp = torch.exp(w_scalar)
        w_t = torch.exp(-w_exp)
        
        curr_state = states[t+1]
        prev_state = states[t]
        
        # dq: from output = state @ q
        dq[:, t, :, :] = (doutput[:, t, :, :, None] * curr_state).sum(dim=2) * scale
        
        # dstate from output
        dstate_from_out = q_t[:, :, None, :] * doutput[:, t, :, :, None]
        dstate_curr = dstate + dstate_from_out
        
        # Compute sa (needed for subsequent gradients)
        sa = (prev_state * a_t[:, :, None, :]).sum(dim=-1)
        
        # dw: from w_t * prev_state
        dw[:, t, :, :] = -(dstate_curr * prev_state).sum(dim=2) * w_t * w_exp
        
        # dk, dv: from k * v^T
        dk[:, t, :, :] = (dstate_curr * v_t[:, :, :, None]).sum(dim=2)
        dv[:, t, :, :] = (dstate_curr * k_t[:, :, None, :]).sum(dim=-1)
        
        # db: from sa * b_t
        db[:, t, :, :] = (dstate_curr * sa[:, :, :, None]).sum(dim=2)
        
        # dsa: from sa * b_t
        dsa = (dstate_curr * b_t[:, :, None, :]).sum(dim=-1)
        
        # da: from sa = sum(prev_state * a_t)
        da[:, t, :, :] = (prev_state * dsa[:, :, :, None]).sum(dim=2)
        
        # dstate propagate to previous timestep
        dstate_from_sa = a_t[:, :, None, :] * dsa[:, :, :, None]
        dstate_from_decay = dstate_curr * w_t[:, :, None, :]
        dstate = dstate_from_sa + dstate_from_decay
    
    dh0 = dstate if initial_state is not None else None
    
    return dq, dk, dv, dw, da, db, dh0


class RWKV7InfctxNaiveFunction(torch.autograd.Function):
    """RWKV7 Infctx - Naive PyTorch autograd Function."""
    
    @staticmethod
    def forward(ctx, q, k, v, w, a, b, scale, initial_state):
        output, final_state = naive_recurrent_rwkv7_fwd(
            q, k, v, w, a, b, scale, initial_state
        )
        
        ctx.save_for_backward(q, k, v, w, a, b, initial_state)
        ctx.scale = scale
        ctx.use_initial_state = (initial_state is not None)
        
        return output, final_state
    
    @staticmethod
    def backward(ctx, doutput, dh_t):
        q, k, v, w, a, b, initial_state = ctx.saved_tensors
        
        dq, dk, dv, dw, da, db, dh0 = naive_recurrent_rwkv7_bwd(
            q, k, v, w, a, b, doutput, dh_t, ctx.scale, initial_state
        )
        
        dh0 = dh0 if ctx.use_initial_state else None
        
        return dq, dk, dv, dw, da, db, None, dh0


def rwkv7_wkv_infctx_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    scale: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RWKV7 Infctx WKV - Naive PyTorch implementation.
    
    Args:
        q, k, v, w, a, b: [B, T, H, K] tensors
        state: [B, H, V, K] initial state (optional)
        scale: scaling factor
    
    Returns:
        output: [B, T, H, V]
        final_state: [B, H, V, K]
    """
    return RWKV7InfctxNaiveFunction.apply(q, k, v, w, a, b, scale, state)
