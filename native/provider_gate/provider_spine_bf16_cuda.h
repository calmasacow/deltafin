#ifndef DELTAFIN_PROVIDER_SPINE_BF16_CUDA_H
#define DELTAFIN_PROVIDER_SPINE_BF16_CUDA_H

#include <ATen/ATen.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace deltafin::provider_internal {

/*
 * Production CUDA carrier for original RAW_BF16 spine data. A whole upload
 * run is prepared once at bind; every matrix projection is an offset view of
 * that preparation. No matrix is materialized as FP32.
 */
enum class CudaSpineBf16SourceKind : std::uint32_t {
  DetachedStaged = 0,
  BorrowedDirectHostAts = 1,
  DetachedDeviceOwned = 2,
};

enum class CudaSpineBf16HostPolicy : std::uint32_t {
  /* Safe default remains staged until a real-shape ATS crossover is gated. */
  Auto = 0,
  StageOnly = 1,
};

struct CudaSpineBf16Capability {
  bool compiled = false;
  bool available = false;
  /* Runtime attributes plus an ordinary-pageable-host physical canary. */
  bool direct_host_ats = false;
  /* Deliberately false until device+shape real-weight benchmarks pass. */
  bool direct_host_runtime_activation_qualified = false;
  std::int32_t device_index = -1;
  std::int32_t compute_major = 0;
  std::int32_t compute_minor = 0;
  std::uint32_t maximum_positions = 0;
  std::string detail;
};

struct CudaSpineBf16HostSlab {
  const std::byte* allocation_base = nullptr;
  /* Valid model bytes within the rounded reader allocation. */
  std::size_t logical_slab_bytes = 0;
  std::size_t allocation_bytes = 0;
};

struct CudaSpineBf16DeviceSlab {
  /*
   * Contiguous CUDA Byte, BFloat16, UInt16, or Short storage, owned outside
   * this wrapper. The integer forms are opaque BF16 bits, never numerically
   * converted integers.
   */
  at::Tensor storage;
  std::size_t logical_slab_bytes = 0;
};

struct CudaSpineBf16MatrixView {
  std::size_t matrix_byte_offset = 0;
  std::size_t logical_bytes = 0;
  std::int64_t rows = 0;
  std::int64_t columns = 0;
};

/*
 * Device-neutral consume-once protocol used by both production state and the
 * portable no-CUDA gate. One prepared layer may submit many projections, then
 * publishes exactly one seal/reclaim lifecycle to the source-use ABI.
 */
class CudaSpineBf16LifetimeState final {
 public:
  void note_submission();
  void seal();
  void require_reclaim_query() const;
  void complete_reclaim();
  void complete_abort();
  void poison() noexcept;

  [[nodiscard]] std::size_t submissions() const noexcept;
  [[nodiscard]] bool open() const noexcept;
  [[nodiscard]] bool sealed() const noexcept;
  [[nodiscard]] bool reclaimed() const noexcept;

 private:
  enum class Phase : std::uint8_t { Open, Sealed, Reclaimed, Poisoned };
  Phase phase_ = Phase::Open;
  std::size_t submissions_ = 0;
};

/* Portable descriptor gate shared by CUDA and no-CUDA builds. */
void validate_cuda_spine_bf16_host_view(
    const CudaSpineBf16HostSlab& slab,
    const CudaSpineBf16MatrixView& matrix);

/* Portable dtype half of the device-slab gate, shared with integration. */
[[nodiscard]] bool cuda_spine_bf16_device_storage_dtype_supported(
    at::ScalarType scalar_type) noexcept;

/*
 * One bind-time preparation, any number of T=1..64 projections, one composite
 * completion event. For staged storage, the Rust reader slab is Detached as
 * soon as its exact bytes have been copied into the owned pinned slab. The
 * prepared object must nevertheless remain live until its event is reclaimed.
 * For ATS storage, the entire caller allocation is Borrowed until reclaim.
 */
class CudaSpineBf16PreparedLayer final {
 public:
  ~CudaSpineBf16PreparedLayer();

  CudaSpineBf16PreparedLayer(const CudaSpineBf16PreparedLayer&) = delete;
  CudaSpineBf16PreparedLayer& operator=(const CudaSpineBf16PreparedLayer&) =
      delete;
  CudaSpineBf16PreparedLayer(CudaSpineBf16PreparedLayer&&) noexcept;
  CudaSpineBf16PreparedLayer& operator=(
      CudaSpineBf16PreparedLayer&&) noexcept;

  [[nodiscard]] CudaSpineBf16SourceKind source_kind() const noexcept;
  [[nodiscard]] std::size_t submissions() const noexcept;
  [[nodiscard]] bool sealed() const noexcept;
  [[nodiscard]] bool reclaimed() const noexcept;

  /*
   * `activation` must be contiguous CUDA float32 [T,columns], 1 <= T <= 64,
   * on the prepared device. Weight decode is exact uint16<<16; activation,
   * FMA accumulation, reduction, and output remain FP32. One kernel grid
   * shares the prepared weights across all positions without staging again.
   * Detached device-owned storage follows the caller's current stream and
   * keeps a bounded active-stream event set; staged/ATS storage remains bound
   * to the stream captured during preparation.
   */
  [[nodiscard]] at::Tensor submit(
      const CudaSpineBf16MatrixView& matrix,
      const at::Tensor& activation);

  [[nodiscard]] at::Tensor submit_t1(
      const CudaSpineBf16MatrixView& matrix,
      const at::Tensor& activation);

  /* Public seal is state-only; each submit already advanced one private event. */
  void seal();
  /* Event query only. Never waits and never synchronizes a device or stream. */
  [[nodiscard]] bool try_reclaim();
  /* Exceptional path: waits only this prepared layer's latest event. */
  void abort_and_reclaim();

 private:
  struct Impl;
  explicit CudaSpineBf16PreparedLayer(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
  friend class CudaSpineBf16Projector;
};

class CudaSpineBf16Projector final {
 public:
  explicit CudaSpineBf16Projector(const at::Device& device);
  ~CudaSpineBf16Projector();

  CudaSpineBf16Projector(const CudaSpineBf16Projector&) = delete;
  CudaSpineBf16Projector& operator=(const CudaSpineBf16Projector&) = delete;
  CudaSpineBf16Projector(CudaSpineBf16Projector&&) = delete;
  CudaSpineBf16Projector& operator=(CudaSpineBf16Projector&&) = delete;

  [[nodiscard]] const CudaSpineBf16Capability& capability();

  /*
   * Copies/pins/uploads the complete raw BF16 slab once, never per matrix.
   * Auto intentionally chooses this safe path until an ATS real-shape gate
   * exists; StageOnly is the explicit equivalent.
   */
  [[nodiscard]] std::unique_ptr<CudaSpineBf16PreparedLayer>
  prepare_host_layer(const CudaSpineBf16HostSlab& slab,
                     CudaSpineBf16HostPolicy policy =
                         CudaSpineBf16HostPolicy::Auto);

  /*
   * Qualification/benchmark-only ATS entry point. Runtime code must not call
   * this until direct_host_ats is true AND its exact device/shape crossover
   * beats staging. It wraps the whole slab so 256-byte matrix offsets remain
   * valid under one borrowed source lease.
   */
  [[nodiscard]] std::unique_ptr<CudaSpineBf16PreparedLayer>
  prepare_direct_host_layer_for_benchmark(
      const CudaSpineBf16HostSlab& slab);

  /*
   * Persistent/global path (for example the vocabulary head): retain one
   * already device-owned BF16/opaque-16-bit allocation and project views
   * without a second BF16 allocation or any FP32 weight residency.
   */
  [[nodiscard]] std::unique_ptr<CudaSpineBf16PreparedLayer>
  prepare_device_layer(const CudaSpineBf16DeviceSlab& slab);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] bool cuda_spine_bf16_compiled() noexcept;
[[nodiscard]] float cuda_spine_bf16_reference_decode(
    std::uint16_t bits) noexcept;

}  // namespace deltafin::provider_internal

#endif
