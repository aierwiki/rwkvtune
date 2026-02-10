"""
RWKV7 CUDA Operators - Backward Compatibility Layer

[WARN] This module has been refactored. Operator loading logic moved to wkv_loader.py

New Interface:
    from rwkvtune.models.rwkv7.operators import load_cuda_operator_infctx
    op = load_cuda_operator_infctx()

Migration Guide:
    Old code:
        from rwkvtune.models.rwkv7.operators.cuda import load_wkv7_infctx_operator
        op = load_wkv7_infctx_operator()
    
    New code:
        from rwkvtune.models.rwkv7.operators import load_cuda_operator_infctx
        op = load_cuda_operator_infctx()
"""

import warnings

# Import new implementation
from ..wkv_loader import load_cuda_operator_infctx as _load_cuda_operator_infctx


def load_wkv7_infctx_operator():
    """
    Load infinite context CUDA operator (backward compatible, deprecated)
    
    [WARN] This function has been replaced by load_cuda_operator_infctx
    
    Returns:
        CUDA infctx operator module
    """
    warnings.warn(
        "load_wkv7_infctx_operator() is deprecated. Please use:\n"
        "  from rwkvtune.models.rwkv7.operators import load_cuda_operator_infctx",
        DeprecationWarning,
        stacklevel=2
    )
    return _load_cuda_operator_infctx()


# Backward compatibility: keep old global variable name
wkv7_infctx_op = None


def __getattr__(name):
    """Lazy load global variable (backward compatible)"""
    if name == 'wkv7_infctx_op':
        global wkv7_infctx_op
        if wkv7_infctx_op is None:
            warnings.warn(
                "Directly accessing wkv7_infctx_op is deprecated. "
                "Please use load_cuda_operator_infctx()",
                DeprecationWarning,
                stacklevel=2
            )
            wkv7_infctx_op = _load_cuda_operator_infctx()
        return wkv7_infctx_op
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ['wkv7_infctx_op', 'load_wkv7_infctx_operator']
