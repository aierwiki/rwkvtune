"""
RWKV7 CPU Operator - Official Implementation
Copied from RWKV-LM/RWKV-v7/rwkv_v7_demo.py
"""

import torch


def rwkv7_wkv_cpu(r, w, k, v, a, b, HEAD_SIZE=64, state=None):
    """
    RWKV7 CPU implementation (from RWKV official repo, rwkv_v7_demo.py).
    
    Args:
        r: receptance [B, T, C] or [T, C]
        w: decay weights [B, T, C] or [T, C]
        k: key [B, T, C] or [T, C]
        v: value [B, T, C] or [T, C]
        a: [B, T, C] or [T, C]
        b: [B, T, C] or [T, C]
        HEAD_SIZE: head size (default 64)
        state: initial state [B, H, N, N] or [H, N, N] (optional)
    
    Returns:
        y: output [B, T, C] or [T, C] (matches input dimensions)
        state: final state [B, H, N, N] or [H, N, N] (if state is not None)
    """
    # Handle input dimensions (support [T, C] and [B, T, C])
    if r.ndim == 2:
        r = r.unsqueeze(0)
        w = w.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
        had_batch_dim = False
    else:
        had_batch_dim = True
    
    B, T, C = r.size()
    H = C // HEAD_SIZE
    N = HEAD_SIZE
    
    # Official implementation: compute in float
    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
    
    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
    
    # Initialize or use provided state
    if state is not None:
        if state.ndim == 3:
            state = state.unsqueeze(0)
        state = state.to(torch.float32)
    else:
        state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)
    
    for t in range(T):
        kk = k[:, t, :].view(B, H, 1, N)
        rr = r[:, t, :].view(B, H, N, 1)
        vv = v[:, t, :].view(B, H, N, 1)
        aa = a[:, t, :].view(B, H, N, 1)
        bb = b[:, t, :].view(B, H, 1, N)
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t, :] = (state @ rr).view(B, H, N)
    
    # Restore output dimensions
    out = out.view(B, T, C)
    if not had_batch_dim:
        out = out.squeeze(0)
        state = state.squeeze(0)
    
    return out, state


# Export
RUN_CPU_RWKV7 = rwkv7_wkv_cpu
