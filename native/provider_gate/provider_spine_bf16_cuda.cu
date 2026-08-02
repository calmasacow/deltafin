// Exact RAW_BF16 x FP32 CUDA projection kernels for Deltafin.
//
// The source matrix remains uint16 BF16 bits. Each lane expands those bits by
// the exact operation uint32(bits) << 16 and performs FP32 fmaf against an
// FP32 activation. No half/bfloat activation, Tensor Core mode, TF32, cuBLAS,
// allocation, stream creation, retained pointer, or synchronization is hidden
// in this translation unit.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

constexpr std::uint32_t kAbiVersion = 1;
constexpr int kThreads = 256;

thread_local char g_last_error[512] = "no error";

enum Status : int {
  kOk = 0,
  kInvalidArgument = -1,
  kCudaError = -2,
  kUnsupported = -3,
};

void set_error(const char* operation, const char* detail) {
  std::snprintf(g_last_error, sizeof(g_last_error), "%s: %s",
                operation == nullptr ? "CUDA RAW_BF16" : operation,
                detail == nullptr ? "unknown error" : detail);
}

int reject(const char* operation, const char* detail) {
  set_error(operation, detail);
  return kInvalidArgument;
}

int cuda_status(const char* operation, const cudaError_t status) {
  if (status == cudaSuccess) {
    return kOk;
  }
  set_error(operation, cudaGetErrorString(status));
  return kCudaError;
}

__device__ __forceinline__ float exact_bf16(const std::uint16_t bits) {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

// One block owns one output row. Aligned real-model shapes use four BF16/FP32
// lanes per iteration; odd/tail shapes remain exact through the scalar path.
// Warp shuffles avoid a block-wide 256-float shared reduction while preserving
// one fixed FP32 operation order across repeated launches.
__global__ void raw_bf16_fp32_gemv_kernel(
    const std::uint16_t* __restrict__ weights,
    const float* __restrict__ activation,
    float* __restrict__ output,
    const int rows,
    const int columns,
    const int positions) {
  const int row = static_cast<int>(blockIdx.x);
  const int position = static_cast<int>(blockIdx.y);
  if (row >= rows || position >= positions) {
    return;
  }
  const std::uint16_t* row_weights =
      weights + static_cast<std::int64_t>(row) * columns;
  activation += static_cast<std::int64_t>(position) * columns;
  output += static_cast<std::int64_t>(position) * rows;
  float local = 0.0F;
  const bool vector_aligned =
      (reinterpret_cast<std::uintptr_t>(row_weights) & 7U) == 0 &&
      (reinterpret_cast<std::uintptr_t>(activation) & 15U) == 0;
  const int vectors = columns / 4;
  if (vector_aligned) {
    const auto* packed_weights = reinterpret_cast<const uint2*>(row_weights);
    const auto* packed_activation = reinterpret_cast<const float4*>(activation);
    for (int vector = static_cast<int>(threadIdx.x); vector < vectors;
         vector += blockDim.x) {
      const uint2 bits = packed_weights[vector];
      const float4 values = packed_activation[vector];
      local = fmaf(exact_bf16(static_cast<std::uint16_t>(bits.x)), values.x,
                   local);
      local = fmaf(exact_bf16(static_cast<std::uint16_t>(bits.x >> 16)),
                   values.y, local);
      local = fmaf(exact_bf16(static_cast<std::uint16_t>(bits.y)), values.z,
                   local);
      local = fmaf(exact_bf16(static_cast<std::uint16_t>(bits.y >> 16)),
                   values.w, local);
    }
    for (int column = vectors * 4 + static_cast<int>(threadIdx.x);
         column < columns; column += blockDim.x) {
      local = fmaf(exact_bf16(row_weights[column]), activation[column], local);
    }
  } else {
    for (int column = static_cast<int>(threadIdx.x); column < columns;
         column += blockDim.x) {
      local = fmaf(exact_bf16(row_weights[column]), activation[column], local);
    }
  }

  constexpr unsigned kFullWarp = 0xffffffffU;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local += __shfl_down_sync(kFullWarp, local, offset);
  }
  __shared__ float warp_partial[kThreads / 32];
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  if (lane == 0) {
    warp_partial[warp] = local;
  }
  __syncthreads();

  if (warp == 0) {
    float total = lane < (kThreads / 32) ? warp_partial[lane] : 0.0F;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      total += __shfl_down_sync(kFullWarp, total, offset);
    }
    if (lane == 0) {
      output[row] = total;
    }
  }
}

__global__ void decode_bits_kernel(const std::uint16_t* input,
                                   std::uint32_t* output,
                                   const int count) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = __float_as_uint(exact_bf16(input[index]));
  }
}

}  // namespace

extern "C" std::uint32_t k3_cuda_spine_bf16_abi_version(void) {
  return kAbiVersion;
}

extern "C" const char* k3_cuda_spine_bf16_last_error(void) {
  return g_last_error;
}

extern "C" int k3_cuda_spine_bf16_available(const int device,
                                             int* compute_major,
                                             int* compute_minor) {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    return cuda_status("cudaGetDeviceCount", status);
  }
  if (device < 0 || device >= count) {
    set_error("CUDA RAW_BF16 availability",
              "device index is outside the visible range");
    return kUnsupported;
  }
  int compute_mode = -1;
  status = cudaDeviceGetAttribute(&compute_mode, cudaDevAttrComputeMode, device);
  if (status != cudaSuccess) {
    return cuda_status("cudaDeviceGetAttribute(compute mode)", status);
  }
  if (compute_mode == cudaComputeModeProhibited) {
    set_error("CUDA RAW_BF16 availability",
              "device compute mode is prohibited");
    return kUnsupported;
  }
  int major = 0;
  int minor = 0;
  status = cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, device);
  if (status != cudaSuccess) {
    return cuda_status("cudaDeviceGetAttribute(compute major)", status);
  }
  status = cudaDeviceGetAttribute(
      &minor, cudaDevAttrComputeCapabilityMinor, device);
  if (status != cudaSuccess) {
    return cuda_status("cudaDeviceGetAttribute(compute minor)", status);
  }
  if (compute_major != nullptr) {
    *compute_major = major;
  }
  if (compute_minor != nullptr) {
    *compute_minor = minor;
  }
  std::snprintf(g_last_error, sizeof(g_last_error),
                "available: device %d compute capability %d.%d", device,
                major, minor);
  return 1;
}

extern "C" int k3_cuda_spine_bf16_launch_v1(
    const std::uint16_t* weights,
    const float* activation,
    float* output,
    const int rows,
    const int columns,
    const int positions,
    void* stream_pointer) {
  if (weights == nullptr || activation == nullptr || output == nullptr) {
    return reject("CUDA RAW_BF16 GEMV", "null input or output pointer");
  }
  if (rows <= 0 || columns <= 0 || positions <= 0 || positions > 64) {
    return reject("CUDA RAW_BF16 GEMV",
                  "rows/columns must be positive and positions must be 1..64");
  }
  // A non-success preflight is returned before any new work is enqueued.
  // This makes the host-side lifetime contract unambiguous: a successful
  // return means the launch owns its pointers until the caller's event, while
  // a failed return means there is no new launch to drain.
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return cuda_status("CUDA RAW_BF16 GEMV launch preflight", status);
  }
  raw_bf16_fp32_gemv_kernel<<<dim3(rows, positions, 1), kThreads, 0,
                              reinterpret_cast<cudaStream_t>(
                                  stream_pointer)>>>(
      weights, activation, output, rows, columns, positions);
  return cuda_status("CUDA RAW_BF16 GEMV launch", cudaGetLastError());
}

// Test-only in practice, but kept as a checked ABI canary so a physical CUDA
// gate can exhaust all 65,536 BF16 encodings without duplicating device code.
extern "C" int k3_cuda_spine_bf16_decode_bits(
    const std::uint16_t* input,
    std::uint32_t* output,
    const int count,
    void* stream_pointer) {
  if (input == nullptr || output == nullptr || count <= 0) {
    return reject("CUDA RAW_BF16 decode", "invalid input, output, or count");
  }
  const std::int64_t blocks =
      (static_cast<std::int64_t>(count) + kThreads - 1) / kThreads;
  if (blocks > std::numeric_limits<int>::max()) {
    return reject("CUDA RAW_BF16 decode", "launch grid is too large");
  }
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return cuda_status("CUDA RAW_BF16 decode launch preflight", status);
  }
  decode_bits_kernel<<<static_cast<int>(blocks), kThreads, 0,
                       reinterpret_cast<cudaStream_t>(stream_pointer)>>>(
      input, output, count);
  return cuda_status("CUDA RAW_BF16 decode launch", cudaGetLastError());
}
