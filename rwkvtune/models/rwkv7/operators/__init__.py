"""
RWKV7 WKV Operators Module

Provides CUDA and CPU WKV operator implementations:
- load_wkv_operator: Load standard operator (recommended)
- load_cuda_operator_infctx: Load infinite context operator
- rwkv7_wkv: Convenient interface for WKV operations
- load_wkv7_operator: Backward compatible alias
"""

from rwkvtune.models.rwkv7.operators.wkv_loader import (
    load_wkv_operator,
    rwkv7_wkv,
    load_cpu_operator,
    load_cuda_operator_train,
    load_cuda_operator_inference,
    load_cuda_operator_infctx,
)

# Backward compatibility
load_wkv7_operator = load_wkv_operator

__all__ = [
    "load_wkv_operator",
    "rwkv7_wkv",
    "load_wkv7_operator",
    "load_cpu_operator",
    "load_cuda_operator_train", 
    "load_cuda_operator_inference",
    "load_cuda_operator_infctx",
]
