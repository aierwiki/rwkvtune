#include <torch/extension.h>
#include "ATen/ATen.h"

// ============================================================================
// Forward CUDA Kernel Declarations
// ============================================================================

// fp16
void cuda_forward_infctx_fp16(
    int B, int T, int H, int C,
    at::Half* w, at::Half* q, at::Half* k, at::Half* v, at::Half* a, at::Half* b,
    float* s,
    at::Half* y
);

// bf16
void cuda_forward_infctx_bf16(
    int B, int T, int H, int C,
    at::BFloat16* w, at::BFloat16* q, at::BFloat16* k, at::BFloat16* v, at::BFloat16* a, at::BFloat16* b,
    float* s,
    at::BFloat16* y
);

// fp32
void cuda_forward_infctx_fp32(
    int B, int T, int H, int C,
    float* w, float* q, float* k, float* v, float* a, float* b,
    float* s,
    float* y
);

// ============================================================================
// Backward CUDA Kernel Declarations (Recompute Mode)
// ============================================================================

// fp16
void cuda_backward_infctx_fp16(
    int B, int T, int H, int C,
    at::Half* w, at::Half* q, at::Half* k, at::Half* v, at::Half* a, at::Half* b,
    at::Half* dy,
    float* s,
    float* ds_in,
    at::Half* dw, at::Half* dq, at::Half* dk, at::Half* dv, at::Half* da, at::Half* db,
    float* ds
);

// bf16
void cuda_backward_infctx_bf16(
    int B, int T, int H, int C,
    at::BFloat16* w, at::BFloat16* q, at::BFloat16* k, at::BFloat16* v, at::BFloat16* a, at::BFloat16* b,
    at::BFloat16* dy,
    float* s,
    float* ds_in,
    at::BFloat16* dw, at::BFloat16* dq, at::BFloat16* dk, at::BFloat16* dv, at::BFloat16* da, at::BFloat16* db,
    float* ds
);

// fp32
void cuda_backward_infctx_fp32(
    int B, int T, int H, int C,
    float* w, float* q, float* k, float* v, float* a, float* b,
    float* dy,
    float* s,
    float* ds_in,
    float* dw, float* dq, float* dk, float* dv, float* da, float* db,
    float* ds
);

// ============================================================================
// PyTorch Interface Functions
// ============================================================================

// Forward (in-place state update)
void forward_infctx(
    torch::Tensor &w,
    torch::Tensor &q,
    torch::Tensor &k,
    torch::Tensor &v,
    torch::Tensor &a,
    torch::Tensor &b,
    torch::Tensor &s,
    torch::Tensor &y
) {
    int B = w.size(0);
    int T = w.size(1);
    int H = w.size(2);
    int C = w.size(3);
    
    auto dtype = w.dtype();
    TORCH_CHECK(q.dtype() == dtype, "All input tensors must have the same dtype");
    TORCH_CHECK(k.dtype() == dtype, "All input tensors must have the same dtype");
    TORCH_CHECK(v.dtype() == dtype, "All input tensors must have the same dtype");
    TORCH_CHECK(a.dtype() == dtype, "All input tensors must have the same dtype");
    TORCH_CHECK(b.dtype() == dtype, "All input tensors must have the same dtype");
    TORCH_CHECK(y.dtype() == dtype, "All input tensors must have the same dtype");
    TORCH_CHECK(s.dtype() == torch::kFloat32, "s must be float32");
    
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(s.is_contiguous(), "s must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    
    if (dtype == torch::kFloat16) {
        cuda_forward_infctx_fp16(
            B, T, H, C,
            w.data_ptr<at::Half>(),
            q.data_ptr<at::Half>(),
            k.data_ptr<at::Half>(),
            v.data_ptr<at::Half>(),
            a.data_ptr<at::Half>(),
            b.data_ptr<at::Half>(),
            s.data_ptr<float>(),
            y.data_ptr<at::Half>()
        );
    } else if (dtype == torch::kBFloat16) {
        cuda_forward_infctx_bf16(
            B, T, H, C,
            w.data_ptr<at::BFloat16>(),
            q.data_ptr<at::BFloat16>(),
            k.data_ptr<at::BFloat16>(),
            v.data_ptr<at::BFloat16>(),
            a.data_ptr<at::BFloat16>(),
            b.data_ptr<at::BFloat16>(),
            s.data_ptr<float>(),
            y.data_ptr<at::BFloat16>()
        );
    } else if (dtype == torch::kFloat32) {
        cuda_forward_infctx_fp32(
            B, T, H, C,
            w.data_ptr<float>(),
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            s.data_ptr<float>(),
            y.data_ptr<float>()
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Supported: fp16, bf16, fp32");
    }
}

// Backward (recompute mode)
void backward_infctx(
    torch::Tensor &w,
    torch::Tensor &q,
    torch::Tensor &k,
    torch::Tensor &v,
    torch::Tensor &a,
    torch::Tensor &b,
    torch::Tensor &dy,
    torch::Tensor &s,
    torch::Tensor &ds_in,
    torch::Tensor &dw,
    torch::Tensor &dq,
    torch::Tensor &dk,
    torch::Tensor &dv,
    torch::Tensor &da,
    torch::Tensor &db,
    torch::Tensor &ds
) {
    int B = w.size(0);
    int T = w.size(1);
    int H = w.size(2);
    int C = w.size(3);
    
    auto dtype = w.dtype();
    
    // ds_in is gradient from subsequent chunk, pass nullptr if empty
    float* ds_in_ptr = ds_in.numel() > 0 ? ds_in.data_ptr<float>() : nullptr;
    
    if (dtype == torch::kFloat16) {
        cuda_backward_infctx_fp16(
            B, T, H, C,
            w.data_ptr<at::Half>(),
            q.data_ptr<at::Half>(),
            k.data_ptr<at::Half>(),
            v.data_ptr<at::Half>(),
            a.data_ptr<at::Half>(),
            b.data_ptr<at::Half>(),
            dy.data_ptr<at::Half>(),
            s.data_ptr<float>(),
            ds_in_ptr,
            dw.data_ptr<at::Half>(),
            dq.data_ptr<at::Half>(),
            dk.data_ptr<at::Half>(),
            dv.data_ptr<at::Half>(),
            da.data_ptr<at::Half>(),
            db.data_ptr<at::Half>(),
            ds.data_ptr<float>()
        );
    } else if (dtype == torch::kBFloat16) {
        cuda_backward_infctx_bf16(
            B, T, H, C,
            w.data_ptr<at::BFloat16>(),
            q.data_ptr<at::BFloat16>(),
            k.data_ptr<at::BFloat16>(),
            v.data_ptr<at::BFloat16>(),
            a.data_ptr<at::BFloat16>(),
            b.data_ptr<at::BFloat16>(),
            dy.data_ptr<at::BFloat16>(),
            s.data_ptr<float>(),
            ds_in_ptr,
            dw.data_ptr<at::BFloat16>(),
            dq.data_ptr<at::BFloat16>(),
            dk.data_ptr<at::BFloat16>(),
            dv.data_ptr<at::BFloat16>(),
            da.data_ptr<at::BFloat16>(),
            db.data_ptr<at::BFloat16>(),
            ds.data_ptr<float>()
        );
    } else if (dtype == torch::kFloat32) {
        cuda_backward_infctx_fp32(
            B, T, H, C,
            w.data_ptr<float>(),
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            dy.data_ptr<float>(),
            s.data_ptr<float>(),
            ds_in_ptr,
            dw.data_ptr<float>(),
            dq.data_ptr<float>(),
            dk.data_ptr<float>(),
            dv.data_ptr<float>(),
            da.data_ptr<float>(),
            db.data_ptr<float>(),
            ds.data_ptr<float>()
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Supported: fp16, bf16, fp32");
    }
}

// Register module with PYBIND11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_infctx, "RWKV7 Infctx forward (CUDA)");
    m.def("backward", &backward_infctx, "RWKV7 Infctx backward (CUDA, recompute mode)");
}
