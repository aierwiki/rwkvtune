"""
Backward Compatibility Layer - DEPRECATED

This file has been refactored, please use wkv_loader.py instead.

New API:
- load_wkv_operator(device='cuda'|'cpu', for_inference=False, head_size=64)
- load_cpu_operator(head_size=64)
- load_cuda_operator_train(head_size=64)
- load_cuda_operator_inference(head_size=64)
- rwkv7_wkv(q, w, k, v, a, b, ...)

Migration Guide:
    Old code:
        from rwkvtune.models.rwkv7.operators.wkv_cuda import load_wkv7_operator
        op = load_wkv7_operator(head_size=64, device='cuda')
    
    New code:
        from rwkvtune.models.rwkv7.operators import load_wkv_operator
        op = load_wkv_operator(device='cuda', head_size=64)
"""

import warnings
from typing import Callable

from .wkv_loader import (
    load_wkv_operator as _load_wkv_operator,
    rwkv7_wkv as _rwkv7_wkv,
)


def load_wkv7_operator(head_size: int = 64, device: str = 'cuda', with_state: bool = False) -> Callable:
    """
    Backward compatible loading function (DEPRECATED).
    
    This function has been replaced by load_wkv_operator.
    
    Args:
        head_size: Attention head size
        device: 'cuda' or 'cpu'
        with_state: Whether for inference (supports state)
    
    Returns:
        WKV operator function
    """
    warnings.warn(
        "load_wkv7_operator() is deprecated, use load_wkv_operator(device, for_inference, head_size)",
        DeprecationWarning,
        stacklevel=2
    )
    return _load_wkv_operator(device=device, for_inference=with_state, head_size=head_size)


def load_wkv7_operator_cuda(head_size: int = 64) -> Callable:
    """
    Backward compatible CUDA loading function (DEPRECATED).
    
    This function has been replaced by load_cuda_operator_train.
    """
    warnings.warn(
        "load_wkv7_operator_cuda() is deprecated, use load_cuda_operator_train(head_size)",
        DeprecationWarning,
        stacklevel=2
    )
    from .wkv_loader import load_cuda_operator_train
    return load_cuda_operator_train(head_size)


def rwkv7_wkv(*args, **kwargs):
    """
    Backward compatible convenience interface.
    
    This function is still available, directly proxies to new implementation.
    """
    return _rwkv7_wkv(*args, **kwargs)


# Warning: This module is deprecated
warnings.warn(
    "wkv_cuda.py module has been refactored to wkv_loader.py\n"
    "Please update import statements:\n"
    "  from rwkvtune.models.rwkv7.operators import load_wkv_operator, rwkv7_wkv",
    DeprecationWarning,
    stacklevel=2
)
