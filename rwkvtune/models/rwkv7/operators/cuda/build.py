"""
RWKV7 Infinite Context CUDA Operator Build Script

Usage:
    python build.py

Or from project root:
    python rwkvtune/models/rwkv7/operators/cuda/build.py
"""

import os
import sys
from pathlib import Path
from torch.utils.cpp_extension import load

# Current directory
current_dir = Path(__file__).parent

# CUDA source files
cuda_sources = [
    str(current_dir / "wkv7_infctx_op.cpp"),
    str(current_dir / "wkv7_infctx_cuda.cu"),
]

# Compile options
extra_compile_args = {
    "cxx": ["-O3", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0"],
    "nvcc": [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        # Support multiple GPU architectures
        "-gencode=arch=compute_70,code=sm_70",  # V100
        "-gencode=arch=compute_75,code=sm_75",  # Turing
        "-gencode=arch=compute_80,code=sm_80",  # A100
        "-gencode=arch=compute_86,code=sm_86",  # RTX 30xx
        "-gencode=arch=compute_89,code=sm_89",  # RTX 40xx
        "-gencode=arch=compute_90,code=sm_90",  # H100
    ],
}


def build():
    """Build CUDA operator"""
    print("=" * 70)
    print("RWKV7 Infinite Context CUDA Operator Build")
    print("=" * 70)
    print()
    
    print("Source files:")
    for src in cuda_sources:
        print(f"  - {Path(src).name}")
    print()
    
    print("Starting compilation...")
    try:
        wkv7_infctx_op = load(
            name="wkv7_infctx_op",
            sources=cuda_sources,
            extra_cflags=extra_compile_args["cxx"],
            extra_cuda_cflags=extra_compile_args["nvcc"],
            verbose=True,
        )
        
        print()
        print("=" * 70)
        print("[OK] Build successful!")
        print("=" * 70)
        print()
        print("Verifying...")
        
        # Verify functions exist
        assert hasattr(wkv7_infctx_op, "forward"), "forward function not found"
        assert hasattr(wkv7_infctx_op, "backward"), "backward function not found"
        
        print("[OK] Interface verification passed")
        print()
        print("You can now import and use:")
        print("  from rwkvtune.models.rwkv7.operators.cuda import wkv7_infctx_op")
        print()
        
        return wkv7_infctx_op
        
    except Exception as e:
        print()
        print("=" * 70)
        print("[ERROR] Build failed")
        print("=" * 70)
        print()
        print(f"Error: {e}")
        print()
        print("Possible causes:")
        print("  1. CUDA not properly installed")
        print("  2. PyTorch version incompatible")
        print("  3. GPU does not support required CUDA architecture")
        print()
        print("Solutions:")
        print("  1. Check CUDA installation: nvcc --version")
        print("  2. Check PyTorch CUDA support: python -c 'import torch; print(torch.cuda.is_available())'")
        print("  3. See detailed error message above")
        print()
        sys.exit(1)


if __name__ == "__main__":
    build()
