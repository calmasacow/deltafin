#include "provider_bf16_device.h"

#include "provider_precision.h"
#include "provider_spine_bf16_cuda.h"
#if defined(__APPLE__)
#include "provider_spine_bf16_metal.h"
#endif

#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace deltafin::provider_internal {
namespace {

class CudaExactBf16Backend final : public ExactBf16ProjectionBackend {
public:
  CudaExactBf16Backend(
      const at::Device &device,
      std::unique_ptr<CudaSpineBf16PreparedLayer> prepared)
      : device_(device), prepared_(std::move(prepared)) {
    if (!device_.is_cuda() || prepared_ == nullptr) {
      throw std::invalid_argument(
          "CUDA exact-BF16 backend received invalid prepared storage");
    }
  }

  [[nodiscard]] bool
  matches_device(const at::Device &device) const noexcept override {
    return device == device_;
  }

  [[nodiscard]] at::Tensor
  linear(const at::Tensor &input, const std::size_t element_offset,
         const std::size_t rows, const std::size_t columns) override {
    if (element_offset > std::numeric_limits<std::size_t>::max() / 2 ||
        rows > static_cast<std::size_t>(
                   std::numeric_limits<std::int64_t>::max()) ||
        columns > static_cast<std::size_t>(
                      std::numeric_limits<std::int64_t>::max()) ||
        rows > std::numeric_limits<std::size_t>::max() / columns ||
        rows * columns > std::numeric_limits<std::size_t>::max() / 2) {
      throw std::overflow_error(
          "CUDA exact-BF16 matrix view byte span overflows");
    }
    enforce_authoritative_cuda_fp32_precision();
    return prepared_->submit(
        CudaSpineBf16MatrixView{
            .matrix_byte_offset = element_offset * 2,
            .logical_bytes = rows * columns * 2,
            .rows = static_cast<std::int64_t>(rows),
            .columns = static_cast<std::int64_t>(columns),
        },
        input);
  }

private:
  at::Device device_;
  std::unique_ptr<CudaSpineBf16PreparedLayer> prepared_;
};

#if defined(__APPLE__)
class MetalExactBf16Backend final : public ExactBf16ProjectionBackend {
public:
  MetalExactBf16Backend(const at::Device &device,
                        SpineBf16MetalBuffer buffer)
      : device_(device), buffer_(std::move(buffer)) {
    if (!device_.is_mps()) {
      throw std::invalid_argument(
          "Metal exact-BF16 backend received a non-MPS device");
    }
  }

  [[nodiscard]] bool
  matches_device(const at::Device &device) const noexcept override {
    return device == device_;
  }

  [[nodiscard]] at::Tensor
  linear(const at::Tensor &input, const std::size_t element_offset,
         const std::size_t rows, const std::size_t columns) override {
    if (element_offset > std::numeric_limits<std::size_t>::max() / 2 ||
        rows > std::numeric_limits<std::uint32_t>::max() ||
        columns > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error(
          "Metal exact-BF16 matrix view exceeds its checked ABI");
    }
    return spine_bf16_metal_gemv(
        buffer_, element_offset * 2, static_cast<std::uint32_t>(rows),
        static_cast<std::uint32_t>(columns), input);
  }

private:
  at::Device device_;
  SpineBf16MetalBuffer buffer_;
};
#endif

} // namespace

struct ExactBf16DeviceProjector::Impl {
  explicit Impl(const at::Device &selected) : device(selected) {
    if (!device.is_mps() && !device.is_cuda()) {
      throw std::invalid_argument(
          "exact-BF16 device projector requires MPS or CUDA");
    }
    if (device.is_cuda()) {
      enforce_authoritative_cuda_fp32_precision();
      cuda = std::make_unique<CudaSpineBf16Projector>(device);
    }
  }

#if defined(__APPLE__)
  void require_metal_qualified() {
    std::call_once(metal_once, [this] {
      try {
        const SpineBf16MetalCapabilities capabilities =
            spine_bf16_metal_capabilities_v1();
        if (capabilities.abi_version != kSpineBf16MetalAbiV1 ||
            (capabilities.flags & kSpineBf16MetalRequiredCapabilitiesV1) !=
                kSpineBf16MetalRequiredCapabilitiesV1 ||
            capabilities.positions < 64) {
          throw std::runtime_error(
              "Metal exact-BF16 capability contract is incomplete");
        }
        const SpineBf16MetalCanaryReport report =
            spine_bf16_metal_canary_v1();
        if (report.decoded_elements != 65'536 ||
            report.decoded_equal_bits != report.decoded_elements ||
            report.rows == 0 || report.one_hot_equal_bits != report.rows ||
            report.dense_equal_bits != report.rows ||
            report.nonfinite != 0 || report.dense_maximum_absolute != 0.0F ||
            report.dense_reference_argmax !=
                report.dense_candidate_argmax) {
          throw std::runtime_error(
              "Metal exact-BF16 numeric canary missed parity");
        }
        metal_qualified = true;
      } catch (const std::exception &error) {
        metal_failure = error.what();
      } catch (...) {
        metal_failure = "non-standard Metal exact-BF16 qualification failure";
      }
    });
    if (!metal_qualified) {
      throw std::runtime_error("Metal exact-BF16 unavailable: " +
                               metal_failure);
    }
  }
#endif

  at::Device device;
  std::unique_ptr<CudaSpineBf16Projector> cuda;
#if defined(__APPLE__)
  std::once_flag metal_once;
  bool metal_qualified = false;
  std::string metal_failure = "numeric canary did not run";
#endif
};

ExactBf16DeviceProjector::ExactBf16DeviceProjector(
    const at::Device &device)
    : impl_(std::make_unique<Impl>(device)) {}

ExactBf16DeviceProjector::~ExactBf16DeviceProjector() = default;

at::ScalarType ExactBf16DeviceProjector::storage_scalar_type() const noexcept {
  // Signed 16-bit storage is universally supported by CUDA tensor copies and
  // preserves every raw bit when both source and target use the same dtype.
  // MPS UInt16 support is physically covered by the retained-carrier test.
  return impl_->device.is_cuda() ? at::kShort : at::kUInt16;
}

std::shared_ptr<ExactBf16Storage>
ExactBf16DeviceProjector::prepare(at::Tensor storage) {
  if (!storage.defined() || storage.device() != impl_->device ||
      storage.scalar_type() != storage_scalar_type() ||
      !storage.is_contiguous() || storage.dim() != 1 ||
      storage.numel() <= 0) {
    throw std::invalid_argument(
        "exact-BF16 device upload run has invalid storage");
  }
  auto exact = make_exact_bf16_storage(std::move(storage));
  if (impl_->device.is_cuda()) {
    enforce_authoritative_cuda_fp32_precision();
    const CudaSpineBf16Capability &capability = impl_->cuda->capability();
    if (!capability.available || capability.maximum_positions < 64) {
      throw std::runtime_error("CUDA exact-BF16 unavailable: " +
                               capability.detail);
    }
    const std::size_t logical_bytes =
        static_cast<std::size_t>(exact->tensor.numel()) * 2;
    exact->projection_backend = std::make_shared<CudaExactBf16Backend>(
        impl_->device,
        impl_->cuda->prepare_device_layer(CudaSpineBf16DeviceSlab{
            .storage = exact->tensor,
            .logical_slab_bytes = logical_bytes,
        }));
    return exact;
  }
#if defined(__APPLE__)
  impl_->require_metal_qualified();
  exact->projection_backend = std::make_shared<MetalExactBf16Backend>(
      impl_->device, retain_spine_bf16_metal_tensor(exact->tensor));
  return exact;
#else
  throw std::runtime_error(
      "MPS exact-BF16 is unavailable outside an Apple build");
#endif
}

} // namespace deltafin::provider_internal
