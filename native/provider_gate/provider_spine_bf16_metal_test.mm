#include "provider_spine_bf16_metal.h"

#if !defined(__APPLE__)
#error "provider_spine_bf16_metal_test.mm is Apple-only"
#endif
#if !defined(DELTAFIN_HAVE_SPINE_BF16_METAL_V1)
#error "BF16 spine Metal test requires the production capability guard"
#endif
#if !defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
#error "BF16 spine Metal test requires embedded metallibs"
#endif

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/mps/MPSStream.h>

#import <Foundation/Foundation.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using deltafin::provider_internal::SpineBf16MetalBuffer;

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

struct FreeAligned {
  void operator()(std::uint8_t* pointer) const noexcept {
    std::free(pointer);
  }
};

void contract_rejections() {
  const long raw_page_size = sysconf(_SC_PAGESIZE);
  require(raw_page_size > 0, "test could not query host page size");
  const std::size_t page_size = static_cast<std::size_t>(raw_page_size);
  void* raw = nullptr;
  require(posix_memalign(&raw, page_size, page_size) == 0 && raw != nullptr,
          "test page allocation failed");
  std::unique_ptr<std::uint8_t, FreeAligned> bytes(
      static_cast<std::uint8_t*>(raw));
  SpineBf16MetalBuffer wrapped =
      deltafin::provider_internal::wrap_spine_bf16_metal_buffer(
          bytes.get(), 8, page_size);
  const at::Tensor cpu = at::zeros(
      {1, 4}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  expect_failure(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::spine_bf16_metal_gemv_t1(
                wrapped, 0, 1, 4, cpu));
      },
      "CPU-input qualification");
  const at::Tensor t65 = at::zeros(
      {65, 4}, at::TensorOptions().dtype(at::kFloat).device(at::kMPS));
  expect_failure(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::spine_bf16_metal_gemv(
                wrapped, 0, 1, 4, t65));
      },
      "T=65 qualification");
  const at::Tensor t0 = at::zeros(
      {0, 4}, at::TensorOptions().dtype(at::kFloat).device(at::kMPS));
  expect_failure(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::spine_bf16_metal_gemv(
                wrapped, 0, 1, 4, t0));
      },
      "T=0 qualification");
  expect_failure(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::spine_bf16_metal_gemv(
                wrapped, 256, 1, 4,
                at::zeros({1, 4}, at::TensorOptions()
                                       .dtype(at::kFloat)
                                       .device(at::kMPS))));
      },
      "matrix range qualification");
  expect_failure(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::copy_spine_bf16_metal_buffer(
                bytes.get(), 7));
      },
      "odd owned-copy length");
  expect_failure(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::retain_spine_bf16_metal_tensor(
                at::zeros({4}, at::TensorOptions()
                                   .dtype(at::kBFloat16)
                                   .device(at::kCPU))));
      },
      "CPU retained tensor");
  expect_failure(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::retain_spine_bf16_metal_tensor(
                at::zeros({4}, at::TensorOptions()
                                   .dtype(at::kFloat)
                                   .device(at::kMPS))));
      },
      "FP32 retained tensor");
  expect_failure(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::wrap_spine_bf16_metal_buffer(
                bytes.get() + 2, 8, page_size));
      },
      "unaligned no-copy wrapper");
  expect_failure(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::wrap_spine_bf16_metal_buffer(
                bytes.get(), 8, page_size - 2));
      },
      "non-page-sized no-copy wrapper");
  std::cout << "provider_spine_bf16_metal.rejections=PASS\n";
}

void capability_and_parity() {
  const auto capabilities =
      deltafin::provider_internal::spine_bf16_metal_capabilities_v1();
  require(capabilities.abi_version ==
              deltafin::provider_internal::kSpineBf16MetalAbiV1,
          "BF16 spine Metal ABI changed");
  require(capabilities.flags ==
              deltafin::provider_internal::
                  kSpineBf16MetalRequiredCapabilitiesV1,
          "BF16 spine Metal capability flags changed");
  require(capabilities.positions == 64 &&
              capabilities.rows_per_simdgroup == 4 &&
              capabilities.threads_per_threadgroup == 128 &&
              capabilities.column_alignment == 4,
          "BF16 spine Metal schedule changed");

  const auto report =
      deltafin::provider_internal::spine_bf16_metal_canary_v1();
  require(report.decoded_elements != 0 &&
              report.decoded_equal_bits == report.decoded_elements,
          "BF16 spine Metal decode was not bit-exact");
  require(report.rows != 0 &&
              report.one_hot_equal_bits == report.rows,
          "BF16 spine Metal one-hot GEMV was not bit-exact");
  require(report.nonfinite == 0 &&
              report.dense_maximum_absolute <= 2.0e-4F &&
              report.dense_reference_argmax ==
                  report.dense_candidate_argmax,
          "BF16 spine Metal dense GEMV missed its parity gate");
  std::cout << "provider_spine_bf16_metal.decode_bits="
            << report.decoded_equal_bits << '/' << report.decoded_elements
            << '\n'
            << "provider_spine_bf16_metal.one_hot_bits="
            << report.one_hot_equal_bits << '/' << report.rows << '\n'
            << "provider_spine_bf16_metal.dense_bits="
            << report.dense_equal_bits << '/' << report.rows << '\n'
            << "provider_spine_bf16_metal.dense_max_abs="
            << report.dense_maximum_absolute << '\n'
            << "provider_spine_bf16_metal.capability=PASS\n";
}

std::uint16_t finite_weight_bits(const std::size_t index) {
  const std::int32_t centered =
      static_cast<std::int32_t>((index * 37U + 13U) % 113U) - 56;
  const float value = static_cast<float>(centered) / 32.0F;
  return static_cast<std::uint16_t>(
      std::bit_cast<std::uint32_t>(value) >> 16);
}

struct PendingPositions {
  std::uint32_t positions = 0;
  at::Tensor input;
  at::Tensor output;
};

PendingPositions encode_one_hot_positions(
    const SpineBf16MetalBuffer& weight, const std::uint32_t positions,
    const std::uint32_t rows, const std::uint32_t columns) {
  std::vector<float> values(
      static_cast<std::size_t>(positions) * columns, 0.0F);
  for (std::uint32_t position = 0; position < positions; ++position) {
    values[static_cast<std::size_t>(position) * columns +
           (position % columns)] = 1.0F;
  }
  const at::Tensor cpu = at::from_blob(
      values.data(), {positions, columns},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  const at::Tensor input = cpu.to(at::kMPS).contiguous();
  return PendingPositions{
      .positions = positions,
      .input = input,
      .output = deltafin::provider_internal::spine_bf16_metal_gemv(
          weight, 0, rows, columns, input)};
}

void require_exact_one_hot(
    const PendingPositions& pending,
    const std::vector<std::uint16_t>& weight_words,
    const std::uint32_t rows, const std::uint32_t columns) {
  require(pending.output.defined() && pending.output.device().is_mps() &&
              pending.output.scalar_type() == at::kFloat &&
              pending.output.is_contiguous() && pending.output.dim() == 2 &&
              pending.output.size(0) == pending.positions &&
              pending.output.size(1) == rows,
          "multi-position BF16 Metal output contract changed");
  const at::Tensor cpu = pending.output.to(at::kCPU).contiguous();
  const float* found = cpu.const_data_ptr<float>();
  for (std::uint32_t position = 0; position < pending.positions; ++position) {
    const std::uint32_t column = position % columns;
    for (std::uint32_t row = 0; row < rows; ++row) {
      const std::uint32_t expected =
          static_cast<std::uint32_t>(
              weight_words[static_cast<std::size_t>(row) * columns + column])
          << 16;
      const std::uint32_t actual = std::bit_cast<std::uint32_t>(
          found[static_cast<std::size_t>(position) * rows + row]);
      require(actual == expected,
              "multi-position BF16 Metal one-hot result was not bit exact");
    }
  }
}

void owned_and_retained_multi_position() {
  constexpr std::uint32_t kRows = 37;
  constexpr std::uint32_t kColumns = 256;
  constexpr std::size_t kElements =
      static_cast<std::size_t>(kRows) * kColumns;
  constexpr std::size_t kLogicalBytes =
      kElements * sizeof(std::uint16_t);
  std::vector<std::uint16_t> expected_words(kElements);
  for (std::size_t index = 0; index < expected_words.size(); ++index) {
    expected_words[index] = finite_weight_bits(index);
  }

  auto source = std::make_unique<std::uint16_t[]>(kElements);
  std::memcpy(source.get(), expected_words.data(), kLogicalBytes);
  SpineBf16MetalBuffer owned =
      deltafin::provider_internal::copy_spine_bf16_metal_buffer(
          source.get(), kLogicalBytes);
  require(owned.storage_kind() ==
              deltafin::provider_internal::
                  SpineBf16MetalStorageKind::OwnedSharedCopy &&
              owned.logical_bytes() == kLogicalBytes &&
              owned.bytes_per_element() == sizeof(std::uint16_t) &&
              owned.logical_bytes() / kElements == 2,
          "owned BF16 Metal storage is not exactly two logical bytes/weight");
  std::memset(source.get(), 0, kLogicalBytes);
  source.reset();

  std::vector<PendingPositions> owned_pending;
  for (const std::uint32_t positions : {1U, 2U, 9U, 64U}) {
    owned_pending.push_back(
        encode_one_hot_positions(owned, positions, kRows, kColumns));
  }

  at::Tensor cpu_bf16 = at::empty(
      {kRows, kColumns},
      at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU));
  static_assert(sizeof(c10::BFloat16) == sizeof(std::uint16_t));
  std::memcpy(cpu_bf16.data_ptr<c10::BFloat16>(), expected_words.data(),
              kLogicalBytes);
  at::Tensor mps_bf16 = cpu_bf16.to(at::kMPS).contiguous();
  SpineBf16MetalBuffer retained =
      deltafin::provider_internal::retain_spine_bf16_metal_tensor(mps_bf16);
  require(retained.storage_kind() ==
              deltafin::provider_internal::
                  SpineBf16MetalStorageKind::RetainedMpsBf16 &&
              retained.logical_bytes() == kLogicalBytes &&
              retained.bytes_per_element() == 2,
          "retained MPS BF16 storage is not exactly two logical bytes/weight");
  mps_bf16 = at::Tensor();
  cpu_bf16 = at::Tensor();
  PendingPositions retained_pending =
      encode_one_hot_positions(retained, 9, kRows, kColumns);

  at::Tensor cpu_uint16 = at::empty(
      {kRows, kColumns},
      at::TensorOptions().dtype(at::kUInt16).device(at::kCPU));
  std::memcpy(cpu_uint16.data_ptr<std::uint16_t>(), expected_words.data(),
              kLogicalBytes);
  at::Tensor mps_uint16 = cpu_uint16.to(at::kMPS).contiguous();
  SpineBf16MetalBuffer retained_uint16 =
      deltafin::provider_internal::retain_spine_bf16_metal_tensor(
          mps_uint16);
  mps_uint16 = at::Tensor();
  cpu_uint16 = at::Tensor();
  PendingPositions retained_uint16_pending =
      encode_one_hot_positions(retained_uint16, 9, kRows, kColumns);

  at::Tensor cpu_short = at::empty(
      {kRows, kColumns},
      at::TensorOptions().dtype(at::kShort).device(at::kCPU));
  static_assert(sizeof(std::int16_t) == sizeof(std::uint16_t));
  std::memcpy(cpu_short.data_ptr<std::int16_t>(), expected_words.data(),
              kLogicalBytes);
  at::Tensor mps_short = cpu_short.to(at::kMPS).contiguous();
  SpineBf16MetalBuffer retained_short =
      deltafin::provider_internal::retain_spine_bf16_metal_tensor(mps_short);
  mps_short = at::Tensor();
  cpu_short = at::Tensor();
  PendingPositions retained_short_pending =
      encode_one_hot_positions(retained_short, 9, kRows, kColumns);

  // Every T and both ownership modes were encoded before this single explicit
  // test-only boundary. Production entry points contain no commit or wait.
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr,
          "current MPS stream disappeared during ownership test");
  stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
  for (const PendingPositions& pending : owned_pending) {
    require_exact_one_hot(pending, expected_words, kRows, kColumns);
  }
  require_exact_one_hot(retained_pending, expected_words, kRows, kColumns);
  require_exact_one_hot(retained_uint16_pending, expected_words, kRows,
                        kColumns);
  require_exact_one_hot(retained_short_pending, expected_words, kRows,
                        kColumns);
  std::cout << "provider_spine_bf16_metal.positions=1,2,9,64\n"
            << "provider_spine_bf16_metal.owned_bf16=PASS\n"
            << "provider_spine_bf16_metal.retained_bf16=PASS\n"
            << "provider_spine_bf16_metal.retained_uint16=PASS\n"
            << "provider_spine_bf16_metal.retained_short=PASS\n";
}

}  // namespace

int main() {
  @autoreleasepool {
    try {
      if (!at::hasMPS()) {
        std::cout << "provider_spine_bf16_metal.mps=SKIP\n";
        return 0;
      }
      capability_and_parity();
      contract_rejections();
      owned_and_retained_multi_position();
      std::cout << "provider_spine_bf16_metal.mps=PASS\n"
                << "provider_spine_bf16_metal.runtime_msl=ABSENT\n"
                << "provider_spine_bf16_metal.per_projection_wait=ABSENT\n";
      return 0;
    } catch (const std::exception& error) {
      std::cerr << "provider_spine_bf16_metal=FAIL: " << error.what()
                << '\n';
      return 1;
    }
  }
}
