// Exact RAW_BF16 x FP32 device projection kernels for Deltafin, on CUDA and
// ROCm/HIP.
//
// The source matrix remains uint16 BF16 bits. Each lane expands those bits by
// the exact operation uint32(bits) << 16 and performs FP32 fmaf against an
// FP32 activation. No half/bfloat activation, Tensor Core or MFMA mode, TF32,
// cuBLAS/hipBLAS, allocation, stream creation, retained pointer, or
// synchronization is hidden in this translation unit.
//
// The reduction follows the code object's own wave width rather than a
// hardcoded 32, so the addition tree is fixed per build but differs between a
// 32-lane warp and a 64-lane CDNA wavefront. Every 32-lane device keeps the
// exact bits it produced before that change.

#include "gpu_runtime_compat.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

constexpr std::uint32_t kAbiVersion = 1;
constexpr int kThreads = 256;
// The shared reduction slab cannot be sized from `warpSize`, which is a
// runtime value on both toolchains, so it is sized for the narrowest wave the
// shim admits.
constexpr int kMaximumWaves = kThreads / k3_gpu::kMinimumWave;
static_assert(kMaximumWaves == 8, "shared reduction capacity changed");

thread_local char g_last_error[512] = "no error";

enum Status : int {
  kOk = 0,
  kInvalidArgument = -1,
  kCudaError = -2,
  kUnsupported = -3,
};

void set_error(const char* operation, const char* detail) {
  std::snprintf(g_last_error, sizeof(g_last_error), "%s: %s",
                operation == nullptr ? "RAW_BF16" : operation,
                detail == nullptr ? "unknown error" : detail);
}

int reject(const char* operation, const char* detail) {
  set_error(operation, detail);
  return kInvalidArgument;
}

int device_status(const char* operation, const k3_gpu::Error status) {
  if (status == k3_gpu::kSuccess) {
    return kOk;
  }
  set_error(operation, k3_gpu::error_string(status));
  return kCudaError;
}

__device__ __forceinline__ float exact_bf16(const std::uint16_t bits) {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

// `warpSize` is the lane count of the selected code object: 32 on every CUDA
// device and on RDNA, 64 on CDNA. Reducing against it instead of a literal 32
// keeps one fixed addition tree per code object without transplanting the CUDA
// warp width onto a 64-lane wavefront. HIP's `__shfl_down_sync` mask is 64 bits
// wide, so the CUDA-side 0xffffffff would describe only half a CDNA wavefront;
// the HIP branch uses the maskless primitive rather than relying on that mask
// being ignored.
__device__ __forceinline__ float wave_shfl_down(const float value,
                                                const int delta) {
#if K3_GPU_HIP
  return __shfl_down(value, delta, warpSize);
#else
  return __shfl_down_sync(0xffffffffU, value, delta, warpSize);
#endif
}

// Not unrolled: `warpSize` has no compile-time value on either toolchain.
__device__ __forceinline__ float wave_reduce(float value) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += wave_shfl_down(value, offset);
  }
  return value;
}

// One block owns one output row. Aligned real-model shapes use four BF16/FP32
// lanes per iteration; odd/tail shapes remain exact through the scalar path.
// Wave shuffles avoid a block-wide 256-float shared reduction while preserving
// one fixed FP32 operation order across repeated launches on a given code
// object. The tree shape follows the wave width, so a 64-lane build sums in a
// different order than a 32-lane one; both stay within the FP32 parity bound
// the gate enforces, and every 32-lane device keeps its existing bits.
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

  __shared__ float wave_partial[kMaximumWaves];
  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  const int wave = static_cast<int>(threadIdx.x) / warpSize;
  const int waves =
      (static_cast<int>(blockDim.x) + warpSize - 1) / warpSize;

  local = wave_reduce(local);
  if (lane == 0) {
    wave_partial[wave] = local;
  }
  __syncthreads();

  // `wave == 0` selects one whole hardware wavefront at either width, so the
  // second reduction never shuffles against an inactive lane. The host rejects
  // any device whose wave width would overflow `wave_partial`.
  if (wave == 0) {
    float total = lane < waves ? wave_partial[lane] : 0.0F;
    total = wave_reduce(total);
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
  k3_gpu::Error status = k3_gpu::get_device_count(&count);
  if (status != k3_gpu::kSuccess) {
    return device_status("device count query", status);
  }
  if (device < 0 || device >= count) {
    set_error("RAW_BF16 availability",
              "device index is outside the visible range");
    return kUnsupported;
  }
  int compute_mode = -1;
  status = k3_gpu::compute_mode(&compute_mode, device);
  if (status != k3_gpu::kSuccess) {
    return device_status("device attribute (compute mode)", status);
  }
  if (compute_mode == k3_gpu::kComputeModeProhibited) {
    set_error("RAW_BF16 availability",
              "device compute mode is prohibited");
    return kUnsupported;
  }
  // The reduction slab holds one float per wave of a `kThreads` block, sized
  // for the narrowest wave this build accepts. A device outside 32/64 lanes
  // would overflow it or leave a wave unaccumulated, both silently, so it is
  // refused here once per device rather than misreducing on every launch.
  int wave_width = 0;
  status = k3_gpu::wave_width(&wave_width, device);
  if (status != k3_gpu::kSuccess) {
    return device_status("device attribute (wave width)", status);
  }
  if ((wave_width != k3_gpu::kMinimumWave &&
       wave_width != k3_gpu::kMaximumWave) ||
      kThreads % wave_width != 0 ||
      kThreads / wave_width > kMaximumWaves) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "RAW_BF16 availability: %s device %d reports a %d-lane "
                  "wavefront; this build reduces only 32- or 64-lane waves",
                  k3_gpu::kRuntimeName, device, wave_width);
    return kUnsupported;
  }
  int major = 0;
  int minor = 0;
  status = k3_gpu::architecture_major(&major, device);
  if (status != k3_gpu::kSuccess) {
    return device_status("device attribute (architecture major)", status);
  }
  status = k3_gpu::architecture_minor(&minor, device);
  if (status != k3_gpu::kSuccess) {
    return device_status("device attribute (architecture minor)", status);
  }
  if (compute_major != nullptr) {
    *compute_major = major;
  }
  if (compute_minor != nullptr) {
    *compute_minor = minor;
  }
  std::snprintf(g_last_error, sizeof(g_last_error),
                "available: %s device %d architecture %d.%d",
                k3_gpu::kRuntimeName, device, major, minor);
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
    return reject("RAW_BF16 GEMV", "null input or output pointer");
  }
  if (rows <= 0 || columns <= 0 || positions <= 0 || positions > 64) {
    return reject("RAW_BF16 GEMV",
                  "rows/columns must be positive and positions must be 1..64");
  }
  // A non-success preflight is returned before any new work is enqueued.
  // This makes the host-side lifetime contract unambiguous: a successful
  // return means the launch owns its pointers until the caller's event, while
  // a failed return means there is no new launch to drain.
  k3_gpu::Error status = k3_gpu::last_error();
  if (status != k3_gpu::kSuccess) {
    return device_status("RAW_BF16 GEMV launch preflight", status);
  }
  raw_bf16_fp32_gemv_kernel<<<dim3(rows, positions, 1), kThreads, 0,
                              reinterpret_cast<k3_gpu::Stream>(
                                  stream_pointer)>>>(
      weights, activation, output, rows, columns, positions);
  return device_status("RAW_BF16 GEMV launch", k3_gpu::last_error());
}

// Test-only in practice, but kept as a checked ABI canary so a physical CUDA
// gate can exhaust all 65,536 BF16 encodings without duplicating device code.
extern "C" int k3_cuda_spine_bf16_decode_bits(
    const std::uint16_t* input,
    std::uint32_t* output,
    const int count,
    void* stream_pointer) {
  if (input == nullptr || output == nullptr || count <= 0) {
    return reject("RAW_BF16 decode", "invalid input, output, or count");
  }
  const std::int64_t blocks =
      (static_cast<std::int64_t>(count) + kThreads - 1) / kThreads;
  if (blocks > std::numeric_limits<int>::max()) {
    return reject("RAW_BF16 decode", "launch grid is too large");
  }
  k3_gpu::Error status = k3_gpu::last_error();
  if (status != k3_gpu::kSuccess) {
    return device_status("RAW_BF16 decode launch preflight", status);
  }
  decode_bits_kernel<<<static_cast<int>(blocks), kThreads, 0,
                       reinterpret_cast<k3_gpu::Stream>(stream_pointer)>>>(
      input, output, count);
  return device_status("RAW_BF16 decode launch", k3_gpu::last_error());
}
