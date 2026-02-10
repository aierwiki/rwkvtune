"""
RWKV7 Infinite Context Training - WKV Operator Wrapper (High-Performance CUDA Version)

Features:
- Forward and backward both use CUDA kernel
- Uses Recompute mode to save memory
- Supports cross-chunk gradient propagation

Note: Does not use Fast mode (saving intermediate states) as it defeats the purpose
of memory-efficient infinite context training.
"""
import torch
from typing import Tuple, Optional

from .wkv_loader import load_cuda_operator_infctx


class WKV7InfctxFunction(torch.autograd.Function):
    """
    RWKV7 Infctx WKV operator autograd wrapper.
    
    Uses high-performance CUDA kernel for forward and backward passes.
    Uses Recompute mode:
    - Forward: Only saves initial state and input parameters
    - Backward: Recomputes intermediate state sequence
    
    This maximizes memory savings for true "infinite context" training.
    """
    
    @staticmethod
    def forward(ctx, w, q, k, v, a, b, state):
        """
        Forward pass - uses CUDA kernel.
        
        Args:
            w: [B, T, H, C] - time decay (log space)
            q: [B, T, H, C] - query
            k: [B, T, H, C] - key
            v: [B, T, H, C] - value
            a: [B, T, H, C] - alpha
            b: [B, T, H, C] - beta
            state: [B, H, C, C] - initial state
        
        Returns:
            y: [B, T, H, C] - output
            new_state: [B, H, C, C] - final state
        """
        B, T, H, C = q.shape
        
        if state is None:
            state = torch.zeros(B, H, C, C, dtype=torch.float32, device=q.device)
        
        if state.dtype != torch.float32:
            state = state.to(torch.float32)
        
        cuda_op = load_cuda_operator_infctx()
        
        if cuda_op is not None and q.is_cuda:
            y = torch.empty_like(q)
            
            s = state.clone()
            cuda_op.forward(w, q, k, v, a, b, s, y)
            new_state = s
            
            ctx.save_for_backward(w, q, k, v, a, b, state)
            ctx.use_naive = False
        else:
            from .wkv_infctx_naive import naive_recurrent_rwkv7_fwd
            y, new_state = naive_recurrent_rwkv7_fwd(
                q, k, v, w, a, b, scale=1.0, initial_state=state
            )
            ctx.save_for_backward(w, q, k, v, a, b, state)
            ctx.use_naive = True
        
        return y, new_state
    
    @staticmethod
    def backward(ctx, dy, dstate_out):
        """
        Backward pass - uses CUDA kernel (recompute mode).
        
        Args:
            dy: [B, T, H, C] - gradient of output y
            dstate_out: [B, H, C, C] - state gradient from subsequent chunk
        
        Returns:
            dw, dq, dk, dv, da, db, dstate
        """
        w, q, k, v, a, b, s0 = ctx.saved_tensors
        
        if ctx.use_naive:
            from .wkv_infctx_naive import naive_recurrent_rwkv7_fwd
            
            with torch.enable_grad():
                w_grad = w.detach().clone().requires_grad_(True)
                q_grad = q.detach().clone().requires_grad_(True)
                k_grad = k.detach().clone().requires_grad_(True)
                v_grad = v.detach().clone().requires_grad_(True)
                a_grad = a.detach().clone().requires_grad_(True)
                b_grad = b.detach().clone().requires_grad_(True)
                state_grad = s0.detach().clone().requires_grad_(True)
                
                y_recompute, new_state_recompute = naive_recurrent_rwkv7_fwd(
                    q_grad, k_grad, v_grad, w_grad, a_grad, b_grad,
                    scale=1.0,
                    initial_state=state_grad
                )
                
                if dstate_out is not None:
                    outputs = [y_recompute, new_state_recompute]
                    grad_outputs = [dy, dstate_out]
                else:
                    outputs = [y_recompute]
                    grad_outputs = [dy]
                
                grads = torch.autograd.grad(
                    outputs=outputs,
                    inputs=[w_grad, q_grad, k_grad, v_grad, a_grad, b_grad, state_grad],
                    grad_outputs=grad_outputs,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False
                )
                
                dw, dq, dk, dv, da, db, dstate = grads
            
            return dw, dq, dk, dv, da, db, dstate
        
        cuda_op = load_cuda_operator_infctx()
        
        B, T, H, C = q.shape
        
        dy = dy.contiguous()
        
        dw = torch.empty_like(w)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        da = torch.empty_like(a)
        db = torch.empty_like(b)
        ds_out = torch.empty_like(s0)
        
        if dstate_out is None:
            ds_in = torch.empty(0, device=q.device, dtype=torch.float32)
        else:
            ds_in = dstate_out.to(torch.float32).contiguous()
        
        cuda_op.backward(
            w, q, k, v, a, b, dy, s0, ds_in,
            dw, dq, dk, dv, da, db, ds_out
        )
        
        return dw, dq, dk, dv, da, db, ds_out


def rwkv7_wkv_infctx(
    w: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    scale: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RWKV7 Infctx WKV operator.
    
    Uses high-performance CUDA kernel for forward and backward passes.
    Uses Recompute mode to maximize memory savings.
    
    Args:
        w: [B, T, H, C] - gk (time decay, log space)
        q: [B, T, H, C] - Query/Receptance
        k: [B, T, H, C] - Key
        v: [B, T, H, C] - Value
        a: [B, T, H, C] - Alpha
        b: [B, T, H, C] - Beta
        state: [B, H, C, C] - Input state (optional)
        scale: float - Scaling factor (default: 1.0)
    
    Returns:
        output: [B, T, H, C] - Output
        final_state: [B, H, C, C] - Final state
    """
    if scale != 1.0:
        q = q * scale
    
    B, T, H, C = q.shape
    
    if state is None:
        state = torch.zeros(B, H, C, C, dtype=torch.float32, device=q.device)
    
    target_dtype = q.dtype
    w = w.to(target_dtype)
    k = k.to(target_dtype)
    v = v.to(target_dtype)
    a = a.to(target_dtype)
    b = b.to(target_dtype)
    
    w = w.contiguous()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    
    return WKV7InfctxFunction.apply(w, q, k, v, a, b, state)
