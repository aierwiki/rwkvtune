#include <torch/extension.h>
#include "ATen/ATen.h"

void cuda_forward_fp16(int B, int T, int C, int H, float *state, at::Half *r, at::Half *w, at::Half *k, at::Half *v, at::Half *a, at::Half *b, at::Half *y);
void cuda_forward_bf16(int B, int T, int C, int H, float *state, at::BFloat16 *r, at::BFloat16 *w, at::BFloat16 *k, at::BFloat16 *v, at::BFloat16 *a, at::BFloat16 *b, at::BFloat16 *y);
void cuda_forward_fp32(int B, int T, int C, int H, float *state, float *r, float *w, float *k, float *v, float *a, float *b, float *y);

void forward_fp16(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &y) {
    cuda_forward_fp16(B, T, C, H, state.data_ptr<float>(), r.data_ptr<at::Half>(), w.data_ptr<at::Half>(), k.data_ptr<at::Half>(), v.data_ptr<at::Half>(), a.data_ptr<at::Half>(), b.data_ptr<at::Half>(), y.data_ptr<at::Half>());
}

void forward_bf16(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &y) {
    cuda_forward_bf16(B, T, C, H, state.data_ptr<float>(), r.data_ptr<at::BFloat16>(), w.data_ptr<at::BFloat16>(), k.data_ptr<at::BFloat16>(), v.data_ptr<at::BFloat16>(), a.data_ptr<at::BFloat16>(), b.data_ptr<at::BFloat16>(), y.data_ptr<at::BFloat16>());
}

void forward_fp32(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &y) {
    cuda_forward_fp32(B, T, C, H, state.data_ptr<float>(), r.data_ptr<float>(), w.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_fp16", &forward_fp16, "wkv7s_custom forward fp16");
    m.def("forward_bf16", &forward_bf16, "wkv7s_custom forward bf16");
    m.def("forward_fp32", &forward_fp32, "wkv7s_custom forward fp32");
}

TORCH_LIBRARY(wkv7s_custom, m) {
    m.def("forward_fp16", forward_fp16);
    m.def("forward_bf16", forward_bf16);
    m.def("forward_fp32", forward_fp32);
}
