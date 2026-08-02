#include "provider_spine_int8_metal.h"

#if !defined(__APPLE__)
#error "provider_spine_int8_metal_test.mm is Apple-only"
#endif
#if !defined(DELTAFIN_HAVE_SPINE_INT8_METAL_V1)
#error "int8 spine Metal test requires the production capability guard"
#endif
#if !defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
#error "int8 spine Metal test requires embedded metallibs"
#endif

#include <ATen/ATen.h>
#include <ATen/Context.h>

#import <Foundation/Foundation.h>

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(const bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void expect_failure(Function&& function, const char* name) {
  try {
    function();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(std::string(name) + " unexpectedly succeeded");
}

void capability_and_parity() {
  const auto capabilities =
      deltafin::provider_internal::spine_int8_metal_capabilities_v1();
  require(capabilities.abi_version ==
              deltafin::provider_internal::kSpineInt8MetalAbiV1,
          "int8 spine Metal ABI changed");
  require(capabilities.flags ==
              deltafin::provider_internal::
                  kSpineInt8MetalRequiredCapabilitiesV1,
          "int8 spine Metal capability flags changed");
  require(capabilities.threads_per_threadgroup == 256 &&
              capabilities.reserved == 0,
          "int8 spine Metal schedule changed");

  const auto report =
      deltafin::provider_internal::spine_int8_metal_canary_v1();
  require(report.rows == 257 && report.columns == 263 &&
              report.compared_elements == report.rows * report.columns &&
              report.equal_bits == report.compared_elements &&
              report.nonfinite == 0,
          "int8 spine Metal dequant was not bit exact");
  require(report.source_offset_elements != 0 &&
              report.scale_offset_elements != 0 &&
              report.destination_offset_elements != 0,
          "int8 spine Metal canary lost its offset coverage");
  std::cout << "provider_spine_int8_metal.bits=" << report.equal_bits << '/'
            << report.compared_elements << '\n'
            << "provider_spine_int8_metal.capability=PASS\n";
}

void contract_rejections() {
  const at::Tensor q_mps = at::zeros(
      {3, 5}, at::TensorOptions().dtype(at::kChar).device(at::kMPS));
  const at::Tensor scales_mps = at::ones(
      {3}, at::TensorOptions().dtype(at::kFloat).device(at::kMPS));
  const at::Tensor destination_mps = at::zeros(
      {3, 5}, at::TensorOptions().dtype(at::kFloat).device(at::kMPS));
  expect_failure(
      [&] {
        deltafin::provider_internal::spine_int8_metal_dequant_f32(
            destination_mps, q_mps,
            at::ones({2}, at::TensorOptions()
                              .dtype(at::kFloat)
                              .device(at::kMPS)));
      },
      "wrong scale rows");
  expect_failure(
      [&] {
        deltafin::provider_internal::spine_int8_metal_dequant_f32(
            at::zeros({3, 5}, at::TensorOptions()
                                  .dtype(at::kFloat)
                                  .device(at::kCPU)),
            q_mps, scales_mps);
      },
      "CPU destination");
  expect_failure(
      [&] {
        deltafin::provider_internal::spine_int8_metal_dequant_f32(
            destination_mps,
            at::zeros({3, 5}, at::TensorOptions()
                                  .dtype(at::kFloat)
                                  .device(at::kMPS)),
            scales_mps);
      },
      "wrong quantized dtype");
  std::cout << "provider_spine_int8_metal.rejections=PASS\n";
}

}  // namespace

int main() {
  @autoreleasepool {
    try {
      require(at::hasMPS(), "MPS is unavailable");
      capability_and_parity();
      contract_rejections();
      std::cout << "provider_spine_int8_metal.mps=PASS\n";
      return 0;
    } catch (const std::exception& error) {
      std::cerr << "provider_spine_int8_metal=FAIL: " << error.what()
                << '\n';
      return 1;
    }
  }
}
