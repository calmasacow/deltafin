#pragma once

#include <ATen/Context.h>

#include <stdexcept>

namespace deltafin::provider_internal {

// K3's authoritative target keeps fp32 activations and fp32 accumulation.
// PyTorch initializes this policy from process-wide state, including
// TORCH_ALLOW_TF32_CUBLAS_OVERRIDE, so a dedicated CUDA provider session must
// not merely trust the inherited default. Use the non-deprecated, scoped
// backend/operator API and verify the resolved policy after setting it.
inline void enforce_authoritative_cuda_fp32_precision() {
  auto& context = at::globalContext();
  context.setFloat32Precision(at::Float32Backend::CUDA,
                              at::Float32Op::MATMUL,
                              at::Float32Precision::IEEE);
  if (context.float32Precision(at::Float32Backend::CUDA,
                               at::Float32Op::MATMUL) !=
      at::Float32Precision::IEEE) {
    throw std::runtime_error(
        "CUDA target provider could not enforce IEEE fp32 matmul precision");
  }
}

}  // namespace deltafin::provider_internal
