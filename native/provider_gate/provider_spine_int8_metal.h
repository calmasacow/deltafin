#ifndef DELTAFIN_PROVIDER_SPINE_INT8_METAL_H
#define DELTAFIN_PROVIDER_SPINE_INT8_METAL_H

#include <ATen/core/Tensor.h>

#include <cstdint>

namespace deltafin::provider_internal {

constexpr std::uint32_t kSpineInt8MetalAbiV1 = 1;
constexpr std::uint32_t kSpineInt8MetalEmbeddedLibraryV1 = 1U << 0;
constexpr std::uint32_t kSpineInt8MetalFp32DestinationV1 = 1U << 1;
constexpr std::uint32_t kSpineInt8MetalFp32ScaleV1 = 1U << 2;
constexpr std::uint32_t kSpineInt8MetalCurrentStreamV1 = 1U << 3;
constexpr std::uint32_t kSpineInt8MetalAsyncDispatchV1 = 1U << 4;
constexpr std::uint32_t kSpineInt8MetalStorageOffsetV1 = 1U << 5;
constexpr std::uint32_t kSpineInt8MetalRequiredCapabilitiesV1 =
    kSpineInt8MetalEmbeddedLibraryV1 |
    kSpineInt8MetalFp32DestinationV1 | kSpineInt8MetalFp32ScaleV1 |
    kSpineInt8MetalCurrentStreamV1 | kSpineInt8MetalAsyncDispatchV1 |
    kSpineInt8MetalStorageOffsetV1;

struct SpineInt8MetalCapabilities {
  std::uint32_t abi_version = 0;
  std::uint32_t flags = 0;
  std::uint32_t threads_per_threadgroup = 0;
  std::uint32_t reserved = 0;
};

struct SpineInt8MetalCanaryReport {
  std::uint32_t rows = 0;
  std::uint32_t columns = 0;
  std::uint32_t compared_elements = 0;
  std::uint32_t equal_bits = 0;
  std::uint32_t nonfinite = 0;
  std::uint32_t source_offset_elements = 0;
  std::uint32_t scale_offset_elements = 0;
  std::uint32_t destination_offset_elements = 0;
};

#if defined(__APPLE__)

/* Load and validate only the embedded int8 spine-dequant pipeline. */
[[nodiscard]] SpineInt8MetalCapabilities
spine_int8_metal_capabilities_v1();

/*
 * Encode one bit-exact row-scaled expansion on PyTorch's current MPS stream:
 *
 *   destination[row, column] =
 *       float(quantized[row, column]) * row_scales[row]
 *
 * All tensors are contiguous MPS views. Non-zero storage offsets are part of
 * the production contract because both packed spine uploads and the shared
 * FP32 execution arena expose subviews. The dispatch is asynchronous and
 * performs no command-buffer commit or host wait.
 */
void spine_int8_metal_dequant_f32(const at::Tensor& destination,
                                 const at::Tensor& quantized,
                                 const at::Tensor& row_scales);

/* Small deterministic qualification, including non-zero offsets. */
[[nodiscard]] SpineInt8MetalCanaryReport
spine_int8_metal_canary_v1();

#endif  // defined(__APPLE__)

}  // namespace deltafin::provider_internal

#endif
