#ifndef DELTAFIN_PROVIDER_DEVICE_H
#define DELTAFIN_PROVIDER_DEVICE_H

#include "provider_abi.h"

#include <ATen/Context.h>
#include <c10/core/Device.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace deltafin::provider_internal {

inline bool mps_available() {
#if defined(__APPLE__)
  return at::hasMPS();
#else
  return false;
#endif
}

inline std::uint32_t cuda_device_count() {
#if defined(__linux__) || defined(_WIN32)
  if (!at::hasCUDA()) {
    return 0;
  }
  return static_cast<std::uint32_t>(std::min<std::size_t>(
      at::getNumGPUs(), std::numeric_limits<std::uint32_t>::max()));
#else
  return 0;
#endif
}

struct SelectedDevice {
  at::Device device;
  std::uint32_t kind;
  std::uint32_t index;
};

inline SelectedDevice select_device(const std::uint32_t requested,
                                    const std::uint32_t requested_index) {
  std::uint32_t kind = requested;
  if (kind == DELTAFIN_PROVIDER_DEVICE_AUTO_V1) {
    if (mps_available()) {
      kind = DELTAFIN_PROVIDER_DEVICE_MPS_V1;
    } else if (cuda_device_count() != 0) {
      kind = DELTAFIN_PROVIDER_DEVICE_CUDA_V1;
    } else {
      kind = DELTAFIN_PROVIDER_DEVICE_CPU_V1;
    }
  }
  switch (kind) {
    case DELTAFIN_PROVIDER_DEVICE_CPU_V1:
      if (requested_index != 0) {
        throw std::invalid_argument("CPU device index must be zero");
      }
      return {at::Device(at::kCPU), kind, 0};
    case DELTAFIN_PROVIDER_DEVICE_MPS_V1:
      if (!mps_available()) {
        throw std::runtime_error(
            "MPS was requested but this LibTorch has no usable MPS device");
      }
      if (requested_index != 0) {
        throw std::invalid_argument("MPS device index must be zero");
      }
      // LibTorch canonicalizes tensors allocated through `.to(mps)` to
      // `mps:0`.  A bare `at::Device(at::kMPS)` retains index -1 and compares
      // unequal to those tensors even though both name the sole Apple GPU.
      // Store the canonical indexed device so ownership checks cannot reject
      // a tensor the same session just allocated.
      return {at::Device(at::kMPS, 0), kind, 0};
    case DELTAFIN_PROVIDER_DEVICE_CUDA_V1: {
      const std::uint32_t count = cuda_device_count();
      if (requested_index >= count) {
        throw std::runtime_error(
            "CUDA device index is outside the LibTorch provider inventory");
      }
      return {at::Device(at::kCUDA,
                         static_cast<c10::DeviceIndex>(requested_index)),
              kind, requested_index};
    }
    default:
      throw std::invalid_argument("unknown provider device kind");
  }
}

// The shared native test matrix names a CUDA case on every host, so the decode
// path is gated wherever that hardware exists rather than only where the build
// happens to run.  A host with no visible CUDA device skips the case and still
// reports its pass marker, the same way provider_cuda_moe_test skips its
// physical section.
//
// MPS is deliberately not covered here.  Its case is already platform-gated to
// macOS, where an unusable MPS device is a genuine failure and must not be
// downgraded to a skip.
inline bool cuda_case_should_skip(const std::uint32_t requested) {
  return requested == DELTAFIN_PROVIDER_DEVICE_CUDA_V1 &&
         cuda_device_count() == 0;
}

}  // namespace deltafin::provider_internal

#endif
