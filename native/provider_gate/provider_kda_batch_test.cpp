#include "provider_device.h"
#include "provider_bf16_device.h"
#include "provider_kda_batch.h"

#include <ATen/ATen.h>
#include <ATen/ops/_weight_int8pack_mm.h>
#include <c10/core/InferenceMode.h>

#if defined(__APPLE__)
#include <torch/mps.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using deltafin::provider_internal::KdaBatchInputProjections;
using deltafin::provider_internal::KdaBatchDependentProjections;
using deltafin::provider_internal::KdaBatchProjectionPath;
using deltafin::provider_internal::KdaProjection;
using deltafin::provider_internal::KdaWeights;

constexpr std::int64_t kCanaryHidden = 32;
constexpr std::int64_t kCanaryProjection = 32 * 32;
constexpr std::int64_t kCanaryFeatureA = 32;
constexpr std::int64_t kK3Hidden = 7168;
constexpr std::int64_t kK3Projection = 96 * 128;
constexpr std::int64_t kK3FeatureA = 128;

struct Options {
  std::uint32_t device = DELTAFIN_PROVIDER_DEVICE_CPU_V1;
  bool benchmark_real = false;
  std::int64_t positions = 8;
};

struct ProjectionArena {
  at::Tensor weight;
  at::Tensor scale;
};

struct OriginalProjectionArena {
  std::unique_ptr<deltafin::provider_internal::Bf16CpuT1Kernel> cpu_kernel;
  std::unique_ptr<deltafin::provider_internal::ExactBf16DeviceProjector>
      device_projector;
  std::shared_ptr<deltafin::provider_internal::ExactBf16Storage> storage;
};

void synchronize_device(const at::Device& device) {
  if (device.type() == at::kMPS) {
#if defined(__APPLE__)
    torch::mps::synchronize();
#else
    throw std::runtime_error("MPS synchronization is unavailable");
#endif
  }
}

ProjectionArena make_arena(const std::int64_t hidden,
                           const std::int64_t projection,
                           const std::int64_t feature_a,
                           const at::Device& device) {
  const std::int64_t rows = projection * 4 + feature_a;
  at::Tensor weight = at::zeros(
      {rows, hidden},
      at::TensorOptions().dtype(at::kChar).device(at::kCPU));
  at::Tensor scale = at::empty(
      {rows}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  auto weights = weight.accessor<std::int8_t, 2>();
  auto scales = scale.accessor<float, 1>();
  constexpr std::array<float, 4> kScales{
      0.015625F, 0.03125F, 0.0625F, 0.125F};
  for (std::int64_t row = 0; row < rows; ++row) {
    const std::int64_t first = (row * 17 + 3) % hidden;
    const std::int64_t second = (row * 29 + 11) % hidden;
    weights[row][first] = static_cast<std::int8_t>((row % 7) + 1);
    if (second != first) {
      weights[row][second] =
          static_cast<std::int8_t>(-((row % 5) + 1));
    }
    scales[row] = kScales[static_cast<std::size_t>(row % 4)];
  }
  return ProjectionArena{weight.to(device), scale.to(device)};
}

KdaProjection make_separate_projection(const std::int64_t rows,
                                       const std::int64_t columns,
                                       const at::Device& device,
                                       const std::int64_t seed) {
  at::Tensor weight = at::zeros(
      {rows, columns},
      at::TensorOptions().dtype(at::kChar).device(at::kCPU));
  at::Tensor scale = at::empty(
      {rows}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  auto weights = weight.accessor<std::int8_t, 2>();
  auto scales = scale.accessor<float, 1>();
  for (std::int64_t row = 0; row < rows; ++row) {
    weights[row][(row * 17 + seed) % columns] =
        static_cast<std::int8_t>((row + seed) % 7 + 1);
    scales[row] = static_cast<float>((row + seed) % 4 + 1) / 128.0F;
  }
  return KdaProjection{weight.to(device), scale.to(device)};
}

KdaProjection make_dense_projection(const std::int64_t rows,
                                    const std::int64_t columns,
                                    const at::Device& device,
                                    const std::int64_t seed) {
  at::Tensor weight = at::zeros(
      {rows, columns},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  auto values = weight.accessor<float, 2>();
  for (std::int64_t row = 0; row < rows; ++row) {
    values[row][(row * 13 + seed) % columns] =
        static_cast<float>((row + seed) % 5 + 1) / 64.0F;
  }
  return KdaProjection{weight.to(device), at::Tensor()};
}

KdaProjection view(const ProjectionArena& arena, const std::int64_t first,
                   const std::int64_t rows) {
  return KdaProjection{arena.weight.narrow(0, first, rows),
                       arena.scale.narrow(0, first, rows)};
}

KdaProjection separate(const KdaProjection& projection) {
  return KdaProjection{projection.weight.clone(), projection.scale.clone()};
}

KdaWeights make_weights(const std::int64_t hidden,
                        const std::int64_t projection,
                        const std::int64_t feature_a,
                        const at::Device& device,
                        ProjectionArena& arena) {
  arena = make_arena(hidden, projection, feature_a, device);
  std::int64_t row = 0;
  const auto take = [&](const std::int64_t count) {
    KdaProjection result = view(arena, row, count);
    row += count;
    return result;
  };
  KdaWeights weights;
  weights.query_projection = take(projection);
  weights.key_projection = take(projection);
  weights.value_projection = take(projection);
  weights.recurrent_gate_projection = take(projection);
  weights.feature_a_projection = take(feature_a);
  const std::int64_t heads = hidden == kCanaryHidden ? 32 : 96;
  weights.feature_b_projection =
      make_separate_projection(projection, feature_a, device, 71);
  weights.beta_projection =
      make_separate_projection(heads, hidden, device, 73);
  weights.output_projection =
      make_separate_projection(hidden, projection, device, 79);
  weights.output_norm = at::ones(
      {projection / heads},
      at::TensorOptions().dtype(at::kFloat).device(device));
  return weights;
}

KdaWeights make_original_weights(const std::int64_t hidden,
                                 const std::int64_t projection,
                                 const std::int64_t feature_a,
                                 const at::Device& device,
                                 OriginalProjectionArena& arena) {
  const std::int64_t rows = projection * 4 + feature_a;
  at::Tensor cpu_bits = at::empty(
      {rows * hidden},
      at::TensorOptions().dtype(at::kUInt16).device(at::kCPU));
  auto* bits = cpu_bits.mutable_data_ptr<std::uint16_t>();
  std::fill(bits, bits + cpu_bits.numel(), UINT16_C(0));
  for (std::int64_t row = 0; row < rows; ++row) {
    const std::int64_t first = (row * 17 + 3) % hidden;
    const std::int64_t second = (row * 29 + 11) % hidden;
    bits[row * hidden + first] = UINT16_C(0x3f80);
    if (second != first) {
      bits[row * hidden + second] = UINT16_C(0xbf00);
    }
  }
  if (device.is_cpu()) {
    arena.cpu_kernel = std::make_unique<
        deltafin::provider_internal::Bf16CpuT1Kernel>(2);
    arena.storage =
        deltafin::provider_internal::make_exact_bf16_storage(cpu_bits);
  } else {
    arena.device_projector = std::make_unique<
        deltafin::provider_internal::ExactBf16DeviceProjector>(device);
    const at::ScalarType carrier_type =
        arena.device_projector->storage_scalar_type();
    at::Tensor opaque_cpu = at::empty(
        {rows * hidden},
        at::TensorOptions().dtype(carrier_type).device(at::kCPU));
    std::memcpy(opaque_cpu.mutable_data_ptr(), cpu_bits.const_data_ptr(),
                static_cast<std::size_t>(cpu_bits.numel()) *
                    sizeof(std::uint16_t));
    arena.storage = arena.device_projector->prepare(
        opaque_cpu.to(device).contiguous());
  }
  std::size_t element_offset = 0;
  const auto take = [&](const std::int64_t count) {
    const auto original =
        deltafin::provider_internal::make_owned_original_bf16(
            arena.storage, element_offset, static_cast<std::size_t>(count),
            static_cast<std::size_t>(hidden), arena.cpu_kernel.get());
    element_offset += static_cast<std::size_t>(count * hidden);
    return KdaProjection{at::Tensor(), at::Tensor(), original};
  };
  KdaWeights weights;
  weights.query_projection = take(projection);
  weights.key_projection = take(projection);
  weights.value_projection = take(projection);
  weights.recurrent_gate_projection = take(projection);
  weights.feature_a_projection = take(feature_a);
  const std::int64_t heads = hidden == kCanaryHidden ? 32 : 96;
  weights.feature_b_projection =
      make_dense_projection(projection, feature_a, device, 83);
  weights.beta_projection =
      make_dense_projection(heads, hidden, device, 89);
  weights.output_projection =
      make_dense_projection(hidden, projection, device, 97);
  weights.output_norm = at::ones(
      {projection / heads},
      at::TensorOptions().dtype(at::kFloat).device(device));
  return weights;
}

KdaProjection separate_original(const KdaProjection& projection,
                                OriginalProjectionArena& arena) {
  const auto& source = projection.original_bf16;
  const std::size_t elements = source.rows * source.columns;
  const at::Device device =
      deltafin::provider_internal::original_bf16_device(source);
  at::Tensor copied = source.owned_storage->tensor
      .narrow(0, static_cast<std::int64_t>(source.owned_element_offset),
              static_cast<std::int64_t>(elements))
      .clone();
  if (device.is_cpu()) {
    arena.cpu_kernel = std::make_unique<
        deltafin::provider_internal::Bf16CpuT1Kernel>(2);
    arena.storage =
        deltafin::provider_internal::make_exact_bf16_storage(std::move(copied));
  } else {
    arena.device_projector = std::make_unique<
        deltafin::provider_internal::ExactBf16DeviceProjector>(device);
    arena.storage = arena.device_projector->prepare(std::move(copied));
  }
  return KdaProjection{
      at::Tensor(), at::Tensor(),
      deltafin::provider_internal::make_owned_original_bf16(
          arena.storage, 0, source.rows, source.columns,
          arena.cpu_kernel.get())};
}

at::Tensor make_hidden(const std::int64_t positions,
                       const std::int64_t hidden,
                       const at::Device& device) {
  at::Tensor cpu = at::empty(
      {positions, hidden},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  auto values = cpu.accessor<float, 2>();
  for (std::int64_t row = 0; row < positions; ++row) {
    for (std::int64_t column = 0; column < hidden; ++column) {
      const std::int64_t numerator =
          ((row + 3) * (column + 5) * 13) % 257 - 128;
      values[row][column] = static_cast<float>(numerator) / 1024.0F;
    }
  }
  return cpu.to(device);
}

at::Tensor rowwise_projection(const at::Tensor& hidden,
                              const KdaProjection& projection) {
  std::vector<at::Tensor> rows;
  rows.reserve(static_cast<std::size_t>(hidden.size(0)));
  for (std::int64_t row = 0; row < hidden.size(0); ++row) {
    if (projection.original_bf16.defined()) {
      rows.push_back(deltafin::provider_internal::original_bf16_linear(
          hidden.narrow(0, row, 1), projection.original_bf16));
    } else if (projection.scale.defined()) {
      rows.push_back(at::_weight_int8pack_mm(
          hidden.narrow(0, row, 1), projection.weight, projection.scale));
    } else {
      rows.push_back(at::matmul(hidden.narrow(0, row, 1),
                                projection.weight.transpose(0, 1)));
    }
  }
  return at::cat(rows, 0);
}

std::array<at::Tensor, 6> rowwise_reference(
    const at::Tensor& hidden, const KdaWeights& weights) {
  const at::Tensor feature_a =
      rowwise_projection(hidden, weights.feature_a_projection);
  return {
      rowwise_projection(hidden, weights.query_projection),
      rowwise_projection(hidden, weights.key_projection),
      rowwise_projection(hidden, weights.value_projection),
      feature_a,
      rowwise_projection(feature_a, weights.feature_b_projection),
      rowwise_projection(hidden, weights.beta_projection),
  };
}

void require_equal(const at::Tensor& actual, const at::Tensor& expected,
                   const char* name) {
  const at::Tensor actual_cpu = actual.to(at::kCPU);
  const at::Tensor expected_cpu = expected.to(at::kCPU);
  if (!at::equal(actual_cpu, expected_cpu)) {
    const double maximum =
        at::max(at::abs(actual_cpu - expected_cpu)).item<double>();
    throw std::runtime_error(std::string(name) +
                             " differs from rowwise projection, max_abs=" +
                             std::to_string(maximum));
  }
}

void require_result_equal(const KdaBatchInputProjections& actual,
                          const std::array<at::Tensor, 6>& expected) {
  require_equal(actual.query, expected[0], "query");
  require_equal(actual.key, expected[1], "key");
  require_equal(actual.value, expected[2], "value");
}

void require_dependent_equal(const KdaBatchDependentProjections& actual,
                             const std::array<at::Tensor, 6>& expected) {
  require_equal(actual.feature_a, expected[3], "feature A");
  require_equal(actual.feature_b, expected[4], "feature B");
  require_equal(actual.beta, expected[5], "beta");
}

void run_canary(const at::Device& device) {
  ProjectionArena arena;
  KdaWeights grouped = make_weights(
      kCanaryHidden, kCanaryProjection, kCanaryFeatureA, device, arena);
  const at::Tensor hidden = make_hidden(7, kCanaryHidden, device);
  const auto expected = rowwise_reference(hidden, grouped);
  KdaBatchInputProjections adjacent =
      deltafin::provider_internal::kda_project_inputs_batch(
          hidden, grouped, false);
  KdaBatchDependentProjections adjacent_dependent =
      deltafin::provider_internal::kda_project_dependent_batch(
          hidden, grouped, false);
  synchronize_device(device);
  require_result_equal(adjacent, expected);
  require_dependent_equal(adjacent_dependent, expected);
  if (adjacent.path != KdaBatchProjectionPath::Separate ||
      adjacent.provider_dispatches != 3 ||
      adjacent.equivalent_rowwise_dispatches != 21 ||
      adjacent.established_separate_rowwise_dispatches != 21 ||
      adjacent_dependent.dependent_provider_dispatches != 3 ||
      adjacent_dependent.dependent_equivalent_rowwise_dispatches != 21 ||
      adjacent.query.is_alias_of(adjacent.key) ||
      adjacent.query.is_alias_of(adjacent.value)) {
    throw std::runtime_error(
        "adjacent weights did not preserve live separate Q/K/V dispatches");
  }

  KdaWeights split_feature = grouped;
  split_feature.feature_a_projection =
      separate(grouped.feature_a_projection);
  KdaBatchInputProjections split_feature_result =
      deltafin::provider_internal::kda_project_inputs_batch(
          hidden, split_feature, false);
  KdaBatchDependentProjections split_feature_dependent =
      deltafin::provider_internal::kda_project_dependent_batch(
          hidden, split_feature, false);
  synchronize_device(device);
  require_result_equal(split_feature_result, expected);
  require_dependent_equal(split_feature_dependent, expected);
  if (split_feature_result.path != KdaBatchProjectionPath::Separate ||
      split_feature_result.provider_dispatches != 3 ||
      split_feature_result.equivalent_rowwise_dispatches != 21 ||
      split_feature_dependent.dependent_provider_dispatches != 3) {
    throw std::runtime_error(
        "split feature-A changed the live projection schedule");
  }

  KdaWeights separate_weights = grouped;
  separate_weights.query_projection = separate(grouped.query_projection);
  separate_weights.key_projection = separate(grouped.key_projection);
  separate_weights.value_projection = separate(grouped.value_projection);
  separate_weights.recurrent_gate_projection =
      separate(grouped.recurrent_gate_projection);
  separate_weights.feature_a_projection =
      separate(grouped.feature_a_projection);
  KdaBatchInputProjections separate_result =
      deltafin::provider_internal::kda_project_inputs_batch(
          hidden, separate_weights, false);
  KdaBatchDependentProjections separate_dependent =
      deltafin::provider_internal::kda_project_dependent_batch(
          hidden, separate_weights, false);
  synchronize_device(device);
  require_result_equal(separate_result, expected);
  require_dependent_equal(separate_dependent, expected);
  if (separate_result.path != KdaBatchProjectionPath::Separate ||
      separate_result.provider_dispatches != 3 ||
      separate_result.equivalent_rowwise_dispatches != 21 ||
      separate_dependent.dependent_provider_dispatches != 3 ||
      separate_dependent.dependent_equivalent_rowwise_dispatches != 21) {
    throw std::runtime_error("separate batch fallback contract changed");
  }

  bool rejected = false;
  try {
    (void)deltafin::provider_internal::kda_project_inputs_batch(
        hidden.slice(1, 0, kCanaryHidden - 1), grouped, false);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) {
    throw std::runtime_error("KDA batch accepted a malformed hidden width");
  }
}

void run_original_bf16_canary(const at::Device& device) {
  OriginalProjectionArena arena;
  KdaWeights grouped = make_original_weights(
      kCanaryHidden, kCanaryProjection, kCanaryFeatureA, device, arena);
  constexpr std::array<std::int64_t, 10> position_roster{
      1, 2, 3, 4, 5, 6, 7, 8, 9, 64};
  for (const std::int64_t positions : position_roster) {
    const at::Tensor hidden = make_hidden(positions, kCanaryHidden, device);
    const auto expected = rowwise_reference(hidden, grouped);
    const KdaBatchInputProjections result =
        deltafin::provider_internal::kda_project_inputs_batch(
            hidden, grouped, false);
    const KdaBatchDependentProjections dependent =
        deltafin::provider_internal::kda_project_dependent_batch(
            hidden, grouped, false);
    synchronize_device(device);
    require_result_equal(result, expected);
    require_dependent_equal(dependent, expected);
    if (result.path != KdaBatchProjectionPath::Separate ||
        result.provider_dispatches != 3 ||
        result.equivalent_rowwise_dispatches != positions * 3 ||
        result.established_separate_rowwise_dispatches != positions * 3 ||
        dependent.dependent_provider_dispatches != 3 ||
        dependent.dependent_equivalent_rowwise_dispatches != positions * 3 ||
        result.query.is_alias_of(result.key) ||
        result.query.is_alias_of(result.value)) {
      throw std::runtime_error(
          "original-BF16 KDA batch lost its live separate-Q/K/V contract");
    }
  }

  OriginalProjectionArena feature_arena;
  KdaWeights four = grouped;
  four.feature_a_projection =
      separate_original(grouped.feature_a_projection, feature_arena);
  const at::Tensor hidden = make_hidden(9, kCanaryHidden, device);
  const auto expected = rowwise_reference(hidden, four);
  const KdaBatchInputProjections result =
      deltafin::provider_internal::kda_project_inputs_batch(
          hidden, four, false);
  const KdaBatchDependentProjections dependent =
      deltafin::provider_internal::kda_project_dependent_batch(
          hidden, four, false);
  synchronize_device(device);
  require_result_equal(result, expected);
  require_dependent_equal(dependent, expected);
  if (result.path != KdaBatchProjectionPath::Separate ||
      result.provider_dispatches != 3 ||
      result.equivalent_rowwise_dispatches != 27 ||
      dependent.dependent_provider_dispatches != 3) {
    throw std::runtime_error(
        "original-BF16 feature split changed the live dispatch schedule");
  }
}

template <typename Function>
double median_milliseconds(Function&& function, const at::Device& device) {
  constexpr std::int64_t kWarmups = 1;
  constexpr std::int64_t kSamples = 5;
  for (std::int64_t index = 0; index < kWarmups; ++index) {
    function();
  }
  synchronize_device(device);
  std::vector<double> values;
  values.reserve(kSamples);
  for (std::int64_t sample = 0; sample < kSamples; ++sample) {
    const auto started = std::chrono::steady_clock::now();
    function();
    synchronize_device(device);
    values.push_back(std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - started)
                         .count());
  }
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void run_real_benchmark(const at::Device& device,
                        const std::int64_t positions) {
  ProjectionArena arena;
  KdaWeights weights = make_weights(
      kK3Hidden, kK3Projection, kK3FeatureA, device, arena);
  const at::Tensor hidden = make_hidden(positions, kK3Hidden, device);
  KdaBatchInputProjections batch;
  KdaBatchDependentProjections dependent;
  std::array<at::Tensor, 6> rowwise;
  const auto run_batch = [&] {
    batch = deltafin::provider_internal::kda_project_inputs_batch(
        hidden, weights, true);
    dependent = deltafin::provider_internal::kda_project_dependent_batch(
        hidden, weights, true);
  };
  const auto run_rowwise = [&] { rowwise = rowwise_reference(hidden, weights); };
  run_batch();
  run_rowwise();
  synchronize_device(device);
  require_result_equal(batch, rowwise);
  require_dependent_equal(dependent, rowwise);

  // Alternate measurement order to reduce thermal/order bias.
  const double batch_first = median_milliseconds(run_batch, device);
  const double rowwise_second = median_milliseconds(run_rowwise, device);
  const double rowwise_first = median_milliseconds(run_rowwise, device);
  const double batch_second = median_milliseconds(run_batch, device);
  const double batch_ms = (batch_first + batch_second) * 0.5;
  const double rowwise_ms = (rowwise_first + rowwise_second) * 0.5;
  std::cout << "benchmark.kda_batch_positions=" << positions << '\n'
            << "benchmark.kda_batch_projection_ms=" << batch_ms << '\n'
            << "benchmark.kda_rowwise_projection_ms=" << rowwise_ms << '\n'
            << "benchmark.kda_batch_projection_speedup="
            << rowwise_ms / batch_ms << "x\n"
            << "benchmark.kda_batch_provider_dispatches=6\n"
            << "benchmark.kda_grouped_rowwise_provider_dispatches="
            << positions * 6 << '\n'
            << "benchmark.kda_separate_rowwise_provider_dispatches="
            << positions * 6 << '\n';
}

Options parse_options(const int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--benchmark-real") {
      options.benchmark_real = true;
      continue;
    }
    if (argument == "--device" && ++index < argc) {
      const std::string_view value(argv[index]);
      if (value == "cpu") {
        options.device = DELTAFIN_PROVIDER_DEVICE_CPU_V1;
      } else if (value == "mps") {
        options.device = DELTAFIN_PROVIDER_DEVICE_MPS_V1;
      } else if (value == "cuda") {
        options.device = DELTAFIN_PROVIDER_DEVICE_CUDA_V1;
      } else {
        throw std::invalid_argument(
            "KDA batch device must be cpu, mps, or cuda");
      }
      continue;
    }
    if (argument == "--positions" && ++index < argc) {
      options.positions = std::stoll(argv[index]);
      if (options.positions < 1 ||
          options.positions > static_cast<std::int64_t>(
                                  deltafin::provider_internal::
                                      kKdaBatchMaximumPositions)) {
        throw std::invalid_argument("positions must be in 1..64");
      }
      continue;
    }
    throw std::invalid_argument(
        "usage: deltafin-provider-kda-batch-test [--device cpu|mps|cuda] "
        "[--benchmark-real] [--positions 1..64]");
  }
  return options;
}

}  // namespace

int main(const int argc, char** argv) {
  try {
    c10::InferenceMode inference_guard;
    const Options options = parse_options(argc, argv);
    if (deltafin::provider_internal::cuda_case_should_skip(options.device)) {
      std::cout << "check.kda_batch_projection=PASS\n"
                << "check.kda_batch_projection.cuda="
                   "skipped(no visible CUDA device)\n";
      return 0;
    }
    const at::Device device =
        deltafin::provider_internal::select_device(options.device, 0).device;
    run_canary(device);
    run_original_bf16_canary(device);
    if (options.benchmark_real) {
      run_real_benchmark(device, options.positions);
    }
    std::cout << "check.kda_batch_projection=PASS (bit-exact)\n"
              << "check.kda_batch_original_bf16=PASS (T=1..9,64)\n"
              << "check.kda_batch_fallbacks=PASS\n"
              << "check.kda_batch_contract=PASS\n"
              << "device=" << device.str() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "result=FAIL\nerror=\"" << error.what() << "\"\n";
    return 1;
  }
}
