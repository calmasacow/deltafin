#ifndef DELTAFIN_PROVIDER_BF16_DEVICE_H
#define DELTAFIN_PROVIDER_BF16_DEVICE_H

#include "provider_bf16_cpu.h"

#include <ATen/ATen.h>

#include <memory>

namespace deltafin::provider_internal {

/*
 * Session-owned qualifier/factory for exact accelerator BF16 storage.
 * Qualification is performed once per selected-device instance. Every
 * prepared upload run retains one backend wrapper shared by all of its matrix
 * offset views; no matrix creates an FP32 weight tensor.
 */
class ExactBf16DeviceProjector final {
public:
  explicit ExactBf16DeviceProjector(const at::Device &device);
  ~ExactBf16DeviceProjector();

  ExactBf16DeviceProjector(const ExactBf16DeviceProjector &) = delete;
  ExactBf16DeviceProjector &
  operator=(const ExactBf16DeviceProjector &) = delete;
  ExactBf16DeviceProjector(ExactBf16DeviceProjector &&) = delete;
  ExactBf16DeviceProjector &operator=(ExactBf16DeviceProjector &&) = delete;

  [[nodiscard]] at::ScalarType storage_scalar_type() const noexcept;
  [[nodiscard]] std::shared_ptr<ExactBf16Storage>
  prepare(at::Tensor storage);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace deltafin::provider_internal

#endif
