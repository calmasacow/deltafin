#include "provider_precision.h"

#include <ATen/Context.h>

#include <iostream>
#include <stdexcept>

int main() {
  try {
    auto& context = at::globalContext();
    // Simulate inherited process state without requiring CUDA hardware.
    context.setFloat32Precision(at::Float32Backend::CUDA,
                                at::Float32Op::MATMUL,
                                at::Float32Precision::TF32);
    if (context.float32Precision(at::Float32Backend::CUDA,
                                 at::Float32Op::MATMUL) !=
        at::Float32Precision::TF32) {
      throw std::runtime_error("test could not install simulated TF32 policy");
    }
    deltafin::provider_internal::enforce_authoritative_cuda_fp32_precision();
    if (context.float32Precision(at::Float32Backend::CUDA,
                                 at::Float32Op::MATMUL) !=
        at::Float32Precision::IEEE) {
      throw std::runtime_error("authoritative policy did not restore IEEE");
    }
    std::cout << "provider_precision.cuda_ieee=PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "provider_precision.error=" << error.what() << '\n';
    return 1;
  }
}
