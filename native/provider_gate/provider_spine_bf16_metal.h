#ifndef DELTAFIN_PROVIDER_SPINE_BF16_METAL_H
#define DELTAFIN_PROVIDER_SPINE_BF16_METAL_H

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace deltafin::provider_internal {

constexpr std::uint32_t kSpineBf16MetalAbiV1 = 1;
constexpr std::uint32_t kSpineBf16MetalEmbeddedLibraryV1 = 1U << 0;
constexpr std::uint32_t kSpineBf16MetalT1V1 = 1U << 1;
constexpr std::uint32_t kSpineBf16MetalExactBitDecodeV1 = 1U << 2;
constexpr std::uint32_t kSpineBf16MetalFp32InputV1 = 1U << 3;
constexpr std::uint32_t kSpineBf16MetalFp32AccumulateV1 = 1U << 4;
constexpr std::uint32_t kSpineBf16MetalFp32OutputV1 = 1U << 5;
constexpr std::uint32_t kSpineBf16MetalCurrentStreamV1 = 1U << 6;
constexpr std::uint32_t kSpineBf16MetalAsyncDispatchV1 = 1U << 7;
constexpr std::uint32_t kSpineBf16MetalT1To64V1 = 1U << 8;
constexpr std::uint32_t kSpineBf16MetalOwnedBf16V1 = 1U << 9;
constexpr std::uint32_t kSpineBf16MetalRequiredCapabilitiesV1 =
    kSpineBf16MetalEmbeddedLibraryV1 | kSpineBf16MetalT1V1 |
    kSpineBf16MetalExactBitDecodeV1 | kSpineBf16MetalFp32InputV1 |
    kSpineBf16MetalFp32AccumulateV1 | kSpineBf16MetalFp32OutputV1 |
    kSpineBf16MetalCurrentStreamV1 | kSpineBf16MetalAsyncDispatchV1 |
    kSpineBf16MetalT1To64V1 | kSpineBf16MetalOwnedBf16V1;

enum class SpineBf16MetalStorageKind : std::uint32_t {
  BorrowedNoCopy = 0,
  OwnedSharedCopy = 1,
  RetainedMpsBf16 = 2,
};

struct SpineBf16MetalCapabilities {
  std::uint32_t abi_version = 0;
  std::uint32_t flags = 0;
  std::uint32_t positions = 0;
  std::uint32_t rows_per_simdgroup = 0;
  std::uint32_t threads_per_threadgroup = 0;
  std::uint32_t column_alignment = 0;
};

struct SpineBf16MetalCanaryReport {
  std::uint32_t decoded_elements = 0;
  std::uint32_t decoded_equal_bits = 0;
  std::uint32_t rows = 0;
  std::uint32_t one_hot_equal_bits = 0;
  std::uint32_t dense_equal_bits = 0;
  std::uint32_t nonfinite = 0;
  float dense_maximum_absolute = 0.0F;
  std::int64_t dense_reference_argmax = -1;
  std::int64_t dense_candidate_argmax = -1;
};

#if defined(__APPLE__)

/*
 * Non-owning Metal wrapper for one complete page-aligned Rust reader slab.
 * The caller remains the sole owner of `host_pointer` and must keep both the
 * wrapper and its host allocation alive until a later source-use fence proves
 * that every encoded projection has completed. This module deliberately does
 * not guess that lifetime and performs no commit or wait itself.
 */
class SpineBf16MetalBuffer final {
 public:
  ~SpineBf16MetalBuffer();
  SpineBf16MetalBuffer(SpineBf16MetalBuffer&&) noexcept;
  SpineBf16MetalBuffer& operator=(SpineBf16MetalBuffer&&) noexcept;
  SpineBf16MetalBuffer(const SpineBf16MetalBuffer&) = delete;
  SpineBf16MetalBuffer& operator=(const SpineBf16MetalBuffer&) = delete;

  [[nodiscard]] std::size_t logical_bytes() const noexcept;
  [[nodiscard]] std::size_t allocation_bytes() const noexcept;
  [[nodiscard]] SpineBf16MetalStorageKind storage_kind() const noexcept;
  [[nodiscard]] std::size_t bytes_per_element() const noexcept;

 private:
  struct Impl;
  explicit SpineBf16MetalBuffer(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;

  friend SpineBf16MetalBuffer wrap_spine_bf16_metal_buffer(
      const void*, std::size_t, std::size_t);
  friend SpineBf16MetalBuffer copy_spine_bf16_metal_buffer(
      const void*, std::size_t);
  friend SpineBf16MetalBuffer retain_spine_bf16_metal_tensor(
      const at::Tensor&);
  friend at::Tensor spine_bf16_metal_gemv(
      const SpineBf16MetalBuffer&, std::size_t, std::uint32_t,
      std::uint32_t, const at::Tensor&);
  friend at::Tensor spine_bf16_metal_gemv_t1(
      const SpineBf16MetalBuffer&, std::size_t, std::uint32_t,
      std::uint32_t, const at::Tensor&);
  friend SpineBf16MetalCanaryReport spine_bf16_metal_canary_v1();
};

/* Load and qualify only the embedded metallib and its rows4 T=1 pipeline. */
[[nodiscard]] SpineBf16MetalCapabilities
spine_bf16_metal_capabilities_v1();

/*
 * Wrap `allocation_bytes` without copying. `logical_bytes` is the checked
 * descriptor-visible prefix; the larger allocation length is required by
 * Metal's page-granular newBufferWithBytesNoCopy contract.
 */
[[nodiscard]] SpineBf16MetalBuffer wrap_spine_bf16_metal_buffer(
    const void* host_pointer, std::size_t logical_bytes,
    std::size_t allocation_bytes);

/*
 * Copy one raw BF16 slab into a provider-owned shared Metal allocation once.
 * The caller may mutate or free `source` immediately after this returns. The
 * provider stores exactly two logical bytes per weight and never creates an
 * FP32 weight allocation.
 */
[[nodiscard]] SpineBf16MetalBuffer copy_spine_bf16_metal_buffer(
    const void* source, std::size_t logical_bytes);

/*
 * Retain one contiguous MPS BFloat16, UInt16, or Short tensor (for a retained
 * layer or global vocabulary head) without copying it. Integer tensors carry
 * opaque BF16 bits; all accepted storage remains exactly two bytes/weight.
 */
[[nodiscard]] SpineBf16MetalBuffer retain_spine_bf16_metal_tensor(
    const at::Tensor& tensor);

/*
 * Encode [T,columns] fp32 by [rows,columns] original BF16 for 1 <= T <= 64.
 * Each position is an async rows4 T=1 dispatch on the current MPS encoder;
 * this function performs no command-buffer commit or wait. Returns
 * contiguous [T,rows] fp32 on MPS.
 */
[[nodiscard]] at::Tensor spine_bf16_metal_gemv(
    const SpineBf16MetalBuffer& weight, std::size_t weight_byte_offset,
    std::uint32_t rows, std::uint32_t columns, const at::Tensor& input);

/*
 * Encode one [1,columns] fp32 by [rows,columns] BF16-bit GEMV onto PyTorch's
 * current MPS stream and return a [1,rows] fp32 MPS tensor. This function does
 * not end kernel coalescing, commit, synchronize, or cross a host boundary.
 */
[[nodiscard]] at::Tensor spine_bf16_metal_gemv_t1(
    const SpineBf16MetalBuffer& weight, std::size_t weight_byte_offset,
    std::uint32_t rows, std::uint32_t columns, const at::Tensor& input);

/*
 * Small self-contained qualification. It batches exact decode, one-hot GEMV,
 * dense GEMV, and its MPS reference before one final canary-only wait. The
 * decode arm exhaustively sweeps all 65,536 BF16 bit patterns, including NaNs.
 */
[[nodiscard]] SpineBf16MetalCanaryReport
spine_bf16_metal_canary_v1();

#endif  // defined(__APPLE__)

}  // namespace deltafin::provider_internal

#endif
