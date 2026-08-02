#ifndef DELTAFIN_PROVIDER_BF16_CPU_H
#define DELTAFIN_PROVIDER_BF16_CPU_H

#include <ATen/ATen.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace deltafin::provider_internal {

/*
 * Original-BF16 T=1 projection kernel.
 *
 * The weights remain in their original 16-bit storage. Every element is
 * expanded by placing its bits in the high half of one IEEE-754 binary32
 * value, and multiplication and accumulation are both fp32. There is no
 * quantization, activation conversion, or approximate arithmetic mode.
 */
enum class Bf16CpuT1Dispatch : std::uint32_t {
  Auto = 0,
  Scalar = 1,
  Neon = 2,
  Avx2Fma = 3,
};

inline constexpr std::size_t kBf16CpuT1MaximumWorkers = 256;
inline constexpr std::size_t kBf16CpuMaximumPositions = 64;

[[nodiscard]] float decode_bf16_exact(std::uint16_t bits) noexcept;
[[nodiscard]] const char *
bf16_cpu_t1_dispatch_name(Bf16CpuT1Dispatch dispatch) noexcept;
[[nodiscard]] bool
bf16_cpu_t1_dispatch_available(Bf16CpuT1Dispatch dispatch) noexcept;
[[nodiscard]] bool bf16_cpu_fp_environment_qualified() noexcept;

/*
 * One session should retain one instance. Its bounded worker pool is created
 * only in the constructor and reused for every synchronous apply. Calls on a
 * single instance serialize; separate instances are independent.
 */
class Bf16CpuT1Kernel final {
public:
  explicit Bf16CpuT1Kernel(
      std::size_t worker_count,
      Bf16CpuT1Dispatch dispatch = Bf16CpuT1Dispatch::Auto);
  ~Bf16CpuT1Kernel();

  Bf16CpuT1Kernel(const Bf16CpuT1Kernel &) = delete;
  Bf16CpuT1Kernel &operator=(const Bf16CpuT1Kernel &) = delete;
  Bf16CpuT1Kernel(Bf16CpuT1Kernel &&) = delete;
  Bf16CpuT1Kernel &operator=(Bf16CpuT1Kernel &&) = delete;

  [[nodiscard]] std::size_t worker_count() const noexcept;
  [[nodiscard]] Bf16CpuT1Dispatch selected_dispatch() const noexcept;

  /*
   * Compute output[row] = sum(weight[row, column] * input[column]).
   * Passing active_workers == 0 selects the complete retained pool. Pointer
   * ownership remains with the caller and is borrowed only until return.
   */
  void apply(std::span<const std::uint16_t> weights,
             std::span<const float> input, std::span<float> output,
             std::size_t rows, std::size_t columns,
             std::size_t active_workers = 0);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/*
 * Representation-aware original-BF16 matrix carrier.
 *
 * Provider-owned matrices retain contiguous opaque 16-bit checkpoint bits. A
 * transient streamed layer may instead retain a borrowed CPU pointer protected
 * by the V2 source-use lease. Exactly one storage arm is active. The carrier is
 * deliberately backend-neutral at its ownership boundary: a later CUDA or
 * Metal original-BF16 kernel can consume the shared storage without another
 * model representation rewrite.
 */
class ExactBf16ProjectionBackend {
public:
  virtual ~ExactBf16ProjectionBackend() = default;

  [[nodiscard]] virtual bool
  matches_device(const at::Device &device) const noexcept = 0;
  [[nodiscard]] virtual at::Tensor
  linear(const at::Tensor &input, std::size_t element_offset,
         std::size_t rows, std::size_t columns) = 0;
};

struct ExactBf16Storage {
  at::Tensor tensor;
  std::shared_ptr<ExactBf16ProjectionBackend> projection_backend;
};

struct OriginalBf16Matrix {
  std::shared_ptr<ExactBf16Storage> owned_storage;
  std::size_t owned_element_offset = 0;
  const std::uint16_t *borrowed_cpu_data = nullptr;
  std::size_t rows = 0;
  std::size_t columns = 0;
  Bf16CpuT1Kernel *cpu_kernel = nullptr;

  [[nodiscard]] bool defined() const noexcept {
    return owned_storage != nullptr || borrowed_cpu_data != nullptr ||
        owned_element_offset != 0 || rows != 0 || columns != 0 ||
        cpu_kernel != nullptr;
  }
  [[nodiscard]] bool is_owned() const noexcept {
    return owned_storage != nullptr;
  }
  [[nodiscard]] bool is_borrowed_cpu() const noexcept {
    return borrowed_cpu_data != nullptr;
  }
};

[[nodiscard]] std::shared_ptr<ExactBf16Storage>
make_exact_bf16_storage(at::Tensor storage);

[[nodiscard]] OriginalBf16Matrix
make_owned_original_bf16(std::shared_ptr<ExactBf16Storage> storage,
                         std::size_t element_offset, std::size_t rows,
                         std::size_t columns,
                         Bf16CpuT1Kernel *cpu_kernel = nullptr);

[[nodiscard]] OriginalBf16Matrix
make_owned_original_bf16_cpu(std::shared_ptr<ExactBf16Storage> storage,
                             std::size_t element_offset, std::size_t rows,
                             std::size_t columns,
                             Bf16CpuT1Kernel *kernel);

[[nodiscard]] OriginalBf16Matrix make_borrowed_original_bf16_cpu(
    const std::uint16_t *data, std::size_t rows, std::size_t columns,
    Bf16CpuT1Kernel *kernel);

[[nodiscard]] bool original_bf16_cpu_matrix_matches(
    const OriginalBf16Matrix &matrix, const at::Device &device,
    std::size_t rows, std::size_t columns) noexcept;

[[nodiscard]] bool original_bf16_matrix_matches(
    const OriginalBf16Matrix &matrix, const at::Device &device,
    std::size_t rows, std::size_t columns) noexcept;

[[nodiscard]] at::Device
original_bf16_device(const OriginalBf16Matrix &matrix);

[[nodiscard]] std::optional<OriginalBf16Matrix>
adjacent_original_bf16_cpu_matrices(
    std::span<const OriginalBf16Matrix *const> matrices);

[[nodiscard]] std::optional<OriginalBf16Matrix>
adjacent_original_bf16_matrices(
    std::span<const OriginalBf16Matrix *const> matrices);

[[nodiscard]] at::Tensor
materialize_original_bf16_cpu_f32(const OriginalBf16Matrix &matrix);

/* Explicit diagnostic/bind-time expansion on the carrier's selected device. */
[[nodiscard]] at::Tensor
materialize_original_bf16_f32(const OriginalBf16Matrix &matrix);

/*
 * Apply an original-BF16 matrix to 1..64 contiguous fp32 rows. Input may be
 * [T,C] or [1,T,C]; output preserves the leading dimensions and is fp32 CPU
 * storage. Multiplication and accumulation remain fp32 FMA. The call is
 * synchronous: no borrowed source pointer survives its return.
 */
[[nodiscard]] at::Tensor
bf16_cpu_linear(const at::Tensor &input, const OriginalBf16Matrix &matrix);

/* Backend-neutral exact projection; never falls back to a dense FP32 weight. */
[[nodiscard]] at::Tensor
original_bf16_linear(const at::Tensor &input,
                     const OriginalBf16Matrix &matrix);

} // namespace deltafin::provider_internal

#endif
