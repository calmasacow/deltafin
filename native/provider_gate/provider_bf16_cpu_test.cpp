#include "provider_bf16_cpu.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/ops/matmul.h>
#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <bit>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

#if defined(__APPLE__) && defined(__aarch64__)
#include <sys/sysctl.h>
#endif

namespace {

using deltafin::provider_internal::bf16_cpu_t1_dispatch_available;
using deltafin::provider_internal::bf16_cpu_t1_dispatch_name;
using deltafin::provider_internal::bf16_cpu_fp_environment_qualified;
using deltafin::provider_internal::bf16_cpu_linear;
using deltafin::provider_internal::Bf16CpuT1Dispatch;
using deltafin::provider_internal::Bf16CpuT1Kernel;
using deltafin::provider_internal::decode_bf16_exact;
using deltafin::provider_internal::kBf16CpuT1MaximumWorkers;
using deltafin::provider_internal::make_borrowed_original_bf16_cpu;
using deltafin::provider_internal::make_exact_bf16_storage;
using deltafin::provider_internal::make_owned_original_bf16_cpu;
using Clock = std::chrono::steady_clock;

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

void require(const bool condition, const std::string &message) {
  if (!condition) {
    fail(message);
  }
}

template <typename Function>
void require_throws(Function &&function, const std::string &message) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  fail(message);
}

[[nodiscard]] std::uint16_t float_to_bf16(const float value) noexcept {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t retained_low_bit = (bits >> 16U) & 1U;
  bits += 0x7fffU + retained_low_bit;
  return static_cast<std::uint16_t>(bits >> 16U);
}

void fill_finite(std::vector<std::uint16_t> &weights, std::vector<float> &input,
                 const std::size_t rows, const std::size_t columns) {
  require(weights.size() == rows * columns, "test weight shape mismatch");
  require(input.size() == columns, "test input shape mismatch");
  for (std::size_t column = 0; column < columns; ++column) {
    const std::int32_t centered =
        static_cast<std::int32_t>((column * 29U + 7U) % 251U) - 125;
    input[column] =
        static_cast<float>(centered) * (1.0F / 127.0F) +
        std::sin(static_cast<float>(column + 1U) * 0.03125F) * 0.0625F;
  }
  std::uint32_t state = 0x1234567U;
  for (std::size_t index = 0; index < weights.size(); ++index) {
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    const std::int32_t centered =
        static_cast<std::int32_t>((state >> 8U) & 0xffffU) - 32768;
    weights[index] =
        float_to_bf16(static_cast<float>(centered) * (1.0F / 262144.0F));
  }
}

[[nodiscard]] std::vector<double>
double_reference(const std::vector<std::uint16_t> &weights,
                 const std::vector<float> &input, const std::size_t rows,
                 const std::size_t columns) {
  std::vector<double> result(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    double accumulator = 0.0;
    for (std::size_t column = 0; column < columns; ++column) {
      accumulator += static_cast<double>(
                         decode_bf16_exact(weights[row * columns + column])) *
                     static_cast<double>(input[column]);
    }
    result[row] = accumulator;
  }
  return result;
}

void check_numerical(const std::vector<std::uint16_t> &weights,
                     const std::vector<float> &input,
                     const std::vector<float> &output,
                     const std::vector<double> &reference,
                     const std::size_t rows, const std::size_t columns,
                     const std::string &label) {
  for (std::size_t row = 0; row < rows; ++row) {
    double absolute_product_sum = 0.0;
    for (std::size_t column = 0; column < columns; ++column) {
      absolute_product_sum += std::abs(static_cast<double>(decode_bf16_exact(
                                           weights[row * columns + column])) *
                                       static_cast<double>(input[column]));
    }
    const double observed = static_cast<double>(output[row]);
    const double error = std::abs(observed - reference[row]);
    const double bound = 1.0e-5 + absolute_product_sum * 2.0e-5;
    require(std::isfinite(observed) && error <= bound,
            label + " row " + std::to_string(row) +
                " exceeds fp32 accumulation error bound");
  }
}

void test_all_decode_codes() {
  for (std::uint32_t code = 0; code <= 0xffffU; ++code) {
    const std::uint32_t observed = std::bit_cast<std::uint32_t>(
        decode_bf16_exact(static_cast<std::uint16_t>(code)));
    require(observed == code << 16U,
            "BF16 decode changed the source payload for code " +
                std::to_string(code));
  }
  std::cout << "provider_bf16_cpu.decode_65536=PASS\n";
}

void test_fp_environment_and_edge_values() {
  require(bf16_cpu_fp_environment_qualified(),
          "test process began with a hostile FP environment");
  require(std::bit_cast<std::uint32_t>(decode_bf16_exact(0x8000U)) ==
              0x80000000U,
          "BF16 decode lost negative zero");
  require(std::bit_cast<std::uint32_t>(decode_bf16_exact(0x7fc1U)) ==
              0x7fc10000U,
          "BF16 decode changed a NaN payload");

  Bf16CpuT1Kernel kernel(2, Bf16CpuT1Dispatch::Scalar);
  const std::vector<std::uint16_t> subnormal_weight{0x0001U};
  const std::vector<float> one{1.0F};
  std::vector<float> subnormal_output(1);
  kernel.apply(subnormal_weight, one, subnormal_output, 1, 1);
  require(std::bit_cast<std::uint32_t>(subnormal_output[0]) == 0x00010000U,
          "BF16 CPU projection flushed a finite subnormal");

  const int original_rounding = std::fegetround();
  require(original_rounding == FE_TONEAREST,
          "BF16 CPU test requires nearest-even ambient rounding");
  require(std::fesetround(FE_DOWNWARD) == 0,
          "test could not install hostile downward rounding");
  bool rejected_rounding = false;
  try {
    kernel.apply(subnormal_weight, one, subnormal_output, 1, 1);
  } catch (const std::exception &) {
    rejected_rounding = true;
  }
  const bool hostile_visible = !bf16_cpu_fp_environment_qualified();
  require(std::fesetround(original_rounding) == 0,
          "test could not restore nearest-even rounding");
  require(hostile_visible && rejected_rounding &&
              bf16_cpu_fp_environment_qualified(),
          "BF16 CPU projection did not fail closed under hostile rounding");

#if defined(__x86_64__) || defined(_M_X64)
  const unsigned int original_control = _mm_getcsr();
  _mm_setcsr(original_control | (1U << 6U) | (1U << 15U));
  const bool x86_hostile_visible = !bf16_cpu_fp_environment_qualified();
  bool rejected_flush = false;
  try {
    kernel.apply(subnormal_weight, one, subnormal_output, 1, 1);
  } catch (const std::exception &) {
    rejected_flush = true;
  }
  _mm_setcsr(original_control);
  require(x86_hostile_visible && rejected_flush &&
              bf16_cpu_fp_environment_qualified(),
          "BF16 CPU projection accepted x86 DAZ/FTZ");
#elif defined(__aarch64__)
  std::uint64_t original_control = 0;
  __asm__ volatile("mrs %0, fpcr" : "=r"(original_control));
  const std::uint64_t hostile_control = original_control | (1ULL << 24U);
  __asm__ volatile("msr fpcr, %0\n\tisb" : : "r"(hostile_control) : "memory");
  const bool arm_hostile_visible = !bf16_cpu_fp_environment_qualified();
  bool rejected_flush = false;
  try {
    kernel.apply(subnormal_weight, one, subnormal_output, 1, 1);
  } catch (const std::exception &) {
    rejected_flush = true;
  }
  __asm__ volatile("msr fpcr, %0\n\tisb" : : "r"(original_control) : "memory");
  require(arm_hostile_visible && rejected_flush &&
              bf16_cpu_fp_environment_qualified(),
          "BF16 CPU projection accepted AArch64 flush-to-zero");
#endif
  std::cout << "provider_bf16_cpu.fp_environment=PASS\n";
}

void test_owned_multirow_projection_and_token_parity() {
  constexpr std::size_t rows = 17;
  constexpr std::size_t columns = 19;
  std::vector<std::uint16_t> weights(rows * columns);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      const float value =
          static_cast<float>((row + 1) * (column + 3)) * (1.0F / 512.0F);
      weights[row * columns + column] = float_to_bf16(value);
    }
  }
  at::Tensor storage = at::empty(
      {static_cast<std::int64_t>(weights.size())},
      at::TensorOptions().dtype(at::kUInt16).device(at::kCPU));
  std::memcpy(storage.mutable_data_ptr<std::uint16_t>(), weights.data(),
              weights.size() * sizeof(std::uint16_t));
  require(static_cast<std::size_t>(storage.nbytes()) ==
              weights.size() * sizeof(std::uint16_t),
          "owned original-BF16 storage is not two bytes per element");
  Bf16CpuT1Kernel kernel(7, Bf16CpuT1Dispatch::Auto);
  const auto shared = make_exact_bf16_storage(std::move(storage));
  const auto owned = make_owned_original_bf16_cpu(
      shared, 0, rows, columns, &kernel);
  const auto borrowed = make_borrowed_original_bf16_cpu(
      weights.data(), rows, columns, &kernel);

  for (const std::int64_t positions : {1, 2, 7, 64}) {
    at::Tensor input = at::empty(
        {positions, static_cast<std::int64_t>(columns)},
        at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    float *values = input.mutable_data_ptr<float>();
    for (std::int64_t position = 0; position < positions; ++position) {
      for (std::size_t column = 0; column < columns; ++column) {
        values[position * columns + column] =
            static_cast<float>((position + 2) *
                               static_cast<std::int64_t>(column + 1)) *
            (1.0F / 257.0F);
      }
    }
    const at::Tensor owned_output = bf16_cpu_linear(input, owned);
    const at::Tensor borrowed_output = bf16_cpu_linear(input, borrowed);
    require(at::equal(owned_output, borrowed_output),
            "owned and borrowed original-BF16 projections disagree");
    require(owned_output.sizes() ==
                at::IntArrayRef({positions,
                                 static_cast<std::int64_t>(rows)}),
            "multirow original-BF16 output shape changed");
    const at::Tensor decisions = at::argmax(owned_output, 1);
    require(at::all(decisions == static_cast<std::int64_t>(rows - 1))
                .item<bool>(),
            "head-style original-BF16 logits changed greedy token decisions");
    const at::Tensor rank_three = input.reshape(
        {1, positions, static_cast<std::int64_t>(columns)});
    require(at::equal(bf16_cpu_linear(rank_three, owned).reshape(
                          {positions, static_cast<std::int64_t>(rows)}),
                      owned_output),
            "rank-three original-BF16 multirow projection disagrees");
  }

  at::Tensor too_wide = at::zeros(
      {static_cast<std::int64_t>(
           deltafin::provider_internal::kBf16CpuMaximumPositions + 1),
       static_cast<std::int64_t>(columns)},
      at::TensorOptions().dtype(at::kFloat));
  require_throws([&] { static_cast<void>(bf16_cpu_linear(too_wide, owned)); },
                 "65-position original-BF16 input was accepted");
  std::cout << "provider_bf16_cpu.owned_2byte=PASS\n"
            << "provider_bf16_cpu.multirow_1_64=PASS\n"
            << "provider_bf16_cpu.token_parity=PASS\n";
}

[[nodiscard]] std::vector<Bf16CpuT1Dispatch> available_dispatches() {
  std::vector<Bf16CpuT1Dispatch> result{Bf16CpuT1Dispatch::Scalar};
  for (const Bf16CpuT1Dispatch dispatch :
       {Bf16CpuT1Dispatch::Neon, Bf16CpuT1Dispatch::Avx2Fma}) {
    if (bf16_cpu_t1_dispatch_available(dispatch)) {
      result.push_back(dispatch);
    }
  }
  return result;
}

void test_odd_tails_numerical_and_determinism() {
  constexpr std::size_t pool_workers = 7;
  for (const Bf16CpuT1Dispatch dispatch : available_dispatches()) {
    Bf16CpuT1Kernel kernel(pool_workers, dispatch);
    for (const std::size_t rows : {1U, 3U, 11U}) {
      for (const std::size_t columns :
           {1U, 3U, 7U, 15U, 16U, 17U, 31U, 32U, 33U, 63U, 65U, 127U}) {
        std::vector<std::uint16_t> weights(rows * columns);
        std::vector<float> input(columns);
        std::vector<float> baseline(rows);
        std::vector<float> observed(rows);
        fill_finite(weights, input, rows, columns);
        const std::vector<double> reference =
            double_reference(weights, input, rows, columns);

        kernel.apply(weights, input, baseline, rows, columns, 1);
        check_numerical(weights, input, baseline, reference, rows, columns,
                        bf16_cpu_t1_dispatch_name(dispatch));
        for (std::size_t threads = 1; threads <= pool_workers; ++threads) {
          std::fill(observed.begin(), observed.end(),
                    std::numeric_limits<float>::quiet_NaN());
          kernel.apply(weights, input, observed, rows, columns, threads);
          require(std::memcmp(observed.data(), baseline.data(),
                              rows * sizeof(float)) == 0,
                  std::string(bf16_cpu_t1_dispatch_name(dispatch)) +
                      " changes output with worker partitioning");
          std::vector<float> repeated(rows);
          kernel.apply(weights, input, repeated, rows, columns, threads);
          require(std::memcmp(repeated.data(), observed.data(),
                              rows * sizeof(float)) == 0,
                  std::string(bf16_cpu_t1_dispatch_name(dispatch)) +
                      " is not bit-deterministic");
        }
      }
    }
  }
  std::cout << "provider_bf16_cpu.odd_tail=PASS\n"
            << "provider_bf16_cpu.thread_determinism=PASS\n";
}

void test_runtime_dispatch() {
  Bf16CpuT1Kernel automatic(2, Bf16CpuT1Dispatch::Auto);
  const Bf16CpuT1Dispatch expected =
      bf16_cpu_t1_dispatch_available(Bf16CpuT1Dispatch::Avx2Fma)
          ? Bf16CpuT1Dispatch::Avx2Fma
          : (bf16_cpu_t1_dispatch_available(Bf16CpuT1Dispatch::Neon)
                 ? Bf16CpuT1Dispatch::Neon
                 : Bf16CpuT1Dispatch::Scalar);
  require(automatic.selected_dispatch() == expected,
          "automatic BF16 CPU runtime dispatch chose the wrong kernel");
  require(automatic.worker_count() == 2,
          "BF16 CPU kernel changed its retained worker count");
  require(bf16_cpu_t1_dispatch_available(Bf16CpuT1Dispatch::Auto) &&
              bf16_cpu_t1_dispatch_available(Bf16CpuT1Dispatch::Scalar),
          "automatic/scalar BF16 dispatch must always be available");
  for (const Bf16CpuT1Dispatch dispatch :
       {Bf16CpuT1Dispatch::Neon, Bf16CpuT1Dispatch::Avx2Fma}) {
    if (!bf16_cpu_t1_dispatch_available(dispatch)) {
      require_throws([dispatch] { Bf16CpuT1Kernel unavailable(1, dispatch); },
                     "unavailable BF16 CPU ISA did not fail closed");
    }
  }
  require_throws(
      [] {
        Bf16CpuT1Kernel invalid(1, static_cast<Bf16CpuT1Dispatch>(0xffffffffU));
      },
      "unknown BF16 CPU dispatch selector was accepted");
  std::cout << "provider_bf16_cpu.dispatch="
            << bf16_cpu_t1_dispatch_name(automatic.selected_dispatch()) << '\n'
            << "provider_bf16_cpu.runtime_dispatch=PASS\n";
}

void test_validation() {
  require_throws([] { Bf16CpuT1Kernel invalid(0); },
                 "zero BF16 CPU workers were accepted");
  require_throws([] { Bf16CpuT1Kernel invalid(kBf16CpuT1MaximumWorkers + 1); },
                 "unbounded BF16 CPU worker pool was accepted");

  Bf16CpuT1Kernel kernel(2, Bf16CpuT1Dispatch::Scalar);
  std::vector<std::uint16_t> weights(12, float_to_bf16(0.25F));
  std::vector<float> input(4, 0.5F);
  std::vector<float> output(3);
  kernel.apply(weights, input, output, 3, 4);
  require_throws([&] { kernel.apply(weights, input, output, 3, 4, 3); },
                 "active workers beyond retained pool were accepted");
  require_throws([&] { kernel.apply(weights, input, output, 0, 4); },
                 "zero rows were accepted");
  require_throws([&] { kernel.apply(weights, input, output, 3, 0); },
                 "zero columns were accepted");
  require_throws(
      [&] {
        kernel.apply(weights, input, output,
                     std::numeric_limits<std::size_t>::max(), 2);
      },
      "overflowing BF16 projection shape was accepted");

  std::span<const std::uint16_t> short_weights(weights.data(),
                                               weights.size() - 1);
  require_throws([&] { kernel.apply(short_weights, input, output, 3, 4); },
                 "short BF16 weight span was accepted");

  std::vector<float> overlap(4, 0.5F);
  require_throws(
      [&] {
        kernel.apply(weights, std::span<const float>(overlap.data(), 4),
                     std::span<float>(overlap.data(), 3), 3, 4);
      },
      "overlapping FP32 input/output spans were accepted");

  std::vector<std::uint8_t> unaligned_storage(weights.size() * 2 + 1);
  std::span<const std::uint16_t> unaligned_weights(
      reinterpret_cast<const std::uint16_t *>(unaligned_storage.data() + 1),
      weights.size());
  require_throws([&] { kernel.apply(unaligned_weights, input, output, 3, 4); },
                 "unaligned BF16 storage was accepted");
  std::cout << "provider_bf16_cpu.validation=PASS\n";
}

#if defined(__APPLE__) && defined(__aarch64__)
[[nodiscard]] double milliseconds(const Clock::time_point begin,
                                  const Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

[[nodiscard]] double median(std::vector<double> values) {
  require(!values.empty(), "cannot calculate an empty timing median");
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

[[nodiscard]] bool is_m1_family() noexcept {
  constexpr std::uint32_t m1_family = 0x1b588bb3U;
  std::uint32_t observed = 0;
  std::size_t bytes = sizeof(observed);
  return sysctlbyname("hw.cpufamily", &observed, &bytes, nullptr, 0) == 0 &&
         bytes == sizeof(observed) && observed == m1_family;
}

void run_m1_performance_canary() {
  if (!is_m1_family()) {
    std::cout << "provider_bf16_cpu.m1_performance=SKIP(non-M1)\n";
    return;
  }
  constexpr std::size_t rows = 12288;
  constexpr std::size_t columns = 7168;
  constexpr std::size_t direct_threads = 8;
  constexpr std::size_t torch_threads = 4;
  constexpr std::size_t rounds = 5;
  std::vector<std::uint16_t> weights(rows * columns);
  std::vector<float> input(columns);
  std::vector<float> direct_output(rows);
  fill_finite(weights, input, rows, columns);

  Bf16CpuT1Kernel direct(direct_threads, Bf16CpuT1Dispatch::Neon);
  direct.apply(weights, input, direct_output, rows, columns);
  at::set_num_threads(static_cast<int>(torch_threads));
  at::Tensor source = at::from_blob(
      weights.data(),
      {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(columns)},
      at::TensorOptions().dtype(at::kBFloat16));
  at::Tensor activation =
      at::from_blob(input.data(), {1, static_cast<std::int64_t>(columns)},
                    at::TensorOptions().dtype(at::kFloat));

  std::vector<double> direct_times;
  std::vector<double> transient_times;
  direct_times.reserve(rounds);
  transient_times.reserve(rounds);
  at::Tensor torch_output;
  for (std::size_t round = 0; round < rounds; ++round) {
    const Clock::time_point direct_begin = Clock::now();
    direct.apply(weights, input, direct_output, rows, columns);
    const Clock::time_point direct_end = Clock::now();
    direct_times.push_back(milliseconds(direct_begin, direct_end));

    const Clock::time_point transient_begin = Clock::now();
    at::Tensor expanded = at::empty(
        {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(columns)},
        at::TensorOptions().dtype(at::kFloat));
    expanded.copy_(source, false);
    torch_output = at::matmul(activation, expanded.transpose(0, 1));
    const Clock::time_point transient_end = Clock::now();
    transient_times.push_back(milliseconds(transient_begin, transient_end));
  }
  torch_output = torch_output.contiguous();
  const float *torch_data = torch_output.const_data_ptr<float>();
  for (std::size_t row = 0; row < rows; ++row) {
    const double expected = static_cast<double>(torch_data[row]);
    const double observed = static_cast<double>(direct_output[row]);
    const double bound = 2.0e-3 * std::max(1.0, std::abs(expected));
    require(std::isfinite(observed) && std::abs(observed - expected) <= bound,
            "M1 performance canary failed projection parity");
  }
  const double direct_median = median(direct_times);
  const double transient_median = median(transient_times);
  std::cout << "provider_bf16_cpu.m1_direct_median_ms=" << direct_median << '\n'
            << "provider_bf16_cpu.m1_transient_median_ms=" << transient_median
            << '\n';
  require(direct_median < transient_median * 0.8,
          "direct M1 BF16 kernel did not beat transient expansion by its "
          "conservative canary margin");
  std::cout << "provider_bf16_cpu.m1_performance=PASS\n";
}
#else
void run_m1_performance_canary() {
  std::cout << "provider_bf16_cpu.m1_performance=SKIP(non-Apple-aarch64)\n";
}
#endif

} // namespace

int main(int argc, char **argv) {
  try {
    bool benchmark = false;
    if (argc == 2 && std::string(argv[1]) == "--benchmark") {
      benchmark = true;
    } else if (argc != 1) {
      fail("usage: provider_bf16_cpu_test [--benchmark]");
    }
    c10::InferenceMode inference_mode;
    test_all_decode_codes();
    test_fp_environment_and_edge_values();
    test_owned_multirow_projection_and_token_parity();
    test_odd_tails_numerical_and_determinism();
    test_runtime_dispatch();
    test_validation();
    if (benchmark) {
      run_m1_performance_canary();
    }
    std::cout << "provider_bf16_cpu=PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "provider_bf16_cpu.error=" << error.what() << '\n';
    return 1;
  }
}
