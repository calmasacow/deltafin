// Runtime shim shared by Deltafin's device kernel translation units.
//
// A ROCm build of PyTorch keeps the torch_cuda/c10_cuda pair and the at::kCUDA
// device type, so the provider layer above these kernels is vendor-neutral
// already. The kernels themselves are not: NVCC compiles against the CUDA
// runtime and HIPCC against HIP, and the two spell the same operations with
// different identifiers.
//
// This header does not redefine any `cuda*` name. Hijacking vendor identifiers
// with macros makes header ordering load-bearing and hides which runtime a call
// actually reached. Instead every kernel source spells the neutral `k3_gpu`
// names below, and exactly one of the two branches here is compiled.
//
// Device-side code needs no shim for arithmetic: fmaf, ldexpf, tanhf, expf,
// __syncthreads, the vector types and the bit-cast intrinsics carry the same
// names and the same IEEE semantics on both toolchains. Wavefront width is the
// one real divergence, and it is handled where it is used, against `warpSize`,
// not here.

#ifndef K3_GPU_RUNTIME_COMPAT_H
#define K3_GPU_RUNTIME_COMPAT_H

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__) || \
    defined(__HIP_PLATFORM_HCC__)
#define K3_GPU_HIP 1
#else
#define K3_GPU_HIP 0
#endif

#if K3_GPU_HIP
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

namespace k3_gpu {

#if K3_GPU_HIP

using Error = hipError_t;
using Stream = hipStream_t;

inline constexpr Error kSuccess = hipSuccess;
inline constexpr int kComputeModeProhibited = hipComputeModeProhibited;
inline constexpr const char* kRuntimeName = "HIP";

inline const char* error_string(const Error status) {
  return hipGetErrorString(status);
}
inline Error get_device_count(int* count) { return hipGetDeviceCount(count); }
inline Error last_error() { return hipGetLastError(); }
inline Error peek_error() { return hipPeekAtLastError(); }

inline Error compute_mode(int* value, const int device) {
  return hipDeviceGetAttribute(value, hipDeviceAttributeComputeMode, device);
}
inline Error architecture_major(int* value, const int device) {
  return hipDeviceGetAttribute(
      value, hipDeviceAttributeComputeCapabilityMajor, device);
}
inline Error architecture_minor(int* value, const int device) {
  return hipDeviceGetAttribute(
      value, hipDeviceAttributeComputeCapabilityMinor, device);
}
inline Error wave_width(int* value, const int device) {
  return hipDeviceGetAttribute(value, hipDeviceAttributeWarpSize, device);
}

#else

using Error = cudaError_t;
using Stream = cudaStream_t;

inline constexpr Error kSuccess = cudaSuccess;
inline constexpr int kComputeModeProhibited = cudaComputeModeProhibited;
inline constexpr const char* kRuntimeName = "CUDA";

inline const char* error_string(const Error status) {
  return cudaGetErrorString(status);
}
inline Error get_device_count(int* count) { return cudaGetDeviceCount(count); }
inline Error last_error() { return cudaGetLastError(); }
inline Error peek_error() { return cudaPeekAtLastError(); }

inline Error compute_mode(int* value, const int device) {
  return cudaDeviceGetAttribute(value, cudaDevAttrComputeMode, device);
}
inline Error architecture_major(int* value, const int device) {
  return cudaDeviceGetAttribute(
      value, cudaDevAttrComputeCapabilityMajor, device);
}
inline Error architecture_minor(int* value, const int device) {
  return cudaDeviceGetAttribute(
      value, cudaDevAttrComputeCapabilityMinor, device);
}
inline Error wave_width(int* value, const int device) {
  return cudaDeviceGetAttribute(value, cudaDevAttrWarpSize, device);
}

#endif

// A CDNA wavefront is 64 lanes, an RDNA or CUDA warp is 32. Both are refused
// unless the kernel that queries this can reduce the width it actually gets.
inline constexpr int kMinimumWave = 32;
inline constexpr int kMaximumWave = 64;

}  // namespace k3_gpu

#endif  // K3_GPU_RUNTIME_COMPAT_H
