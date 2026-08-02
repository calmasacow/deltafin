#include "provider_route_mailbox.h"

#include <ATen/ATen.h>
#include <ATen/Context.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using deltafin::provider_internal::RouteMailboxT1;
using deltafin::provider_internal::kRouteMailboxTopK;

RouteMailboxT1 ordinary_materialize(const at::Tensor& expert_ids,
                                    const at::Tensor& weights) {
  const at::Tensor ids_cpu = expert_ids.to(at::kCPU).contiguous();
  const at::Tensor weights_cpu = weights.to(at::kCPU).contiguous();
  RouteMailboxT1 result;
  std::memcpy(result.expert_ids.data(),
              ids_cpu.const_data_ptr<std::int64_t>(),
              result.expert_ids.size() * sizeof(result.expert_ids[0]));
  const float* values = weights_cpu.const_data_ptr<float>();
  for (std::size_t edge = 0; edge < result.weight_bits.size(); ++edge) {
    std::memcpy(&result.weight_bits[edge], &values[edge], sizeof(float));
  }
  return result;
}

void require_equal(const RouteMailboxT1& actual,
                   const RouteMailboxT1& expected, const char* name) {
  if (actual.expert_ids != expected.expert_ids ||
      actual.weight_bits != expected.weight_bits) {
    throw std::runtime_error(std::string(name) + " was not bit-exact");
  }
}

std::pair<at::Tensor, at::Tensor> offset_inputs(const at::Device& device) {
  constexpr std::int64_t kIdOffset = 3;
  constexpr std::int64_t kWeightOffset = 5;
  at::Tensor ids_cpu = at::full(
      {kIdOffset + static_cast<std::int64_t>(kRouteMailboxTopK) + 4},
      -777, at::TensorOptions().dtype(at::kLong));
  at::Tensor weights_cpu = at::zeros(
      {kWeightOffset + static_cast<std::int64_t>(kRouteMailboxTopK) + 4},
      at::TensorOptions().dtype(at::kFloat));
  auto* ids = ids_cpu.data_ptr<std::int64_t>();
  auto* weights = weights_cpu.data_ptr<float>();
  constexpr std::array<std::uint32_t, kRouteMailboxTopK> kUnusualBits = {
      0x00000000U, 0x80000000U, 0x00000001U, 0x007fffffU,
      0x00800000U, 0x00800001U, 0x3f000001U, 0x3f7fffffU,
      0x3f800000U, 0x3f800001U, 0x40000001U, 0x4b000001U,
      0x7f7ffffeU, 0x7f7fffffU, 0x80800001U, 0xbf000001U};
  for (std::size_t edge = 0; edge < kRouteMailboxTopK; ++edge) {
    ids[kIdOffset + static_cast<std::int64_t>(edge)] =
        static_cast<std::int64_t>(811 - 37 * edge);
    std::memcpy(&weights[kWeightOffset + static_cast<std::int64_t>(edge)],
                &kUnusualBits[edge], sizeof(float));
  }
  const at::Tensor ids_storage = ids_cpu.to(device);
  const at::Tensor weight_storage = weights_cpu.to(device);
  at::Tensor ids_view = ids_storage
                            .slice(0, kIdOffset,
                                   kIdOffset + kRouteMailboxTopK)
                            .reshape({1, static_cast<std::int64_t>(
                                             kRouteMailboxTopK)});
  at::Tensor weights_view = weight_storage
                                .slice(0, kWeightOffset,
                                       kWeightOffset + kRouteMailboxTopK)
                                .reshape({1, static_cast<std::int64_t>(
                                                 kRouteMailboxTopK)});
  if (ids_view.storage_offset() != kIdOffset ||
      weights_view.storage_offset() != kWeightOffset ||
      !ids_view.is_contiguous() || !weights_view.is_contiguous()) {
    throw std::runtime_error(
        "test failed to construct contiguous nonzero-offset MPS views");
  }
  return {std::move(ids_view), std::move(weights_view)};
}

std::pair<at::Tensor, at::Tensor> real_route(const at::Device& device) {
  at::Tensor logits = at::linspace(
      -4.0, 5.0, 896,
      at::TensorOptions().dtype(at::kFloat).device(device));
  logits = at::add(logits.reshape({1, 896}),
                   at::sin(at::linspace(
                       -2.0, 7.0, 896,
                       at::TensorOptions().dtype(at::kFloat).device(device)))
                       .reshape({1, 896}));
  const at::Tensor scores = at::sigmoid(logits);
  const at::Tensor bias =
      at::cos(at::linspace(
          -1.0, 11.0, 896,
          at::TensorOptions().dtype(at::kFloat).device(device))) /
      64.0;
  const at::Tensor choice = at::add(scores, bias.reshape({1, 896}));
  const auto [ignored, expert_ids] =
      at::topk(choice, static_cast<std::int64_t>(kRouteMailboxTopK), -1,
               true, false);
  static_cast<void>(ignored);
  const at::Tensor unnormalized = at::gather(scores, 1, expert_ids);
  const at::Tensor weights =
      unnormalized /
      at::add(at::sum(unnormalized, std::vector<std::int64_t>{-1}, true),
              1.0e-20);
  return {expert_ids, weights};
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

double trimmed_mean(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t trim = values.size() / 10;
  double total = 0.0;
  for (std::size_t index = trim; index < values.size() - trim; ++index) {
    total += values[index];
  }
  return total / static_cast<double>(values.size() - 2 * trim);
}

template <typename Function>
double elapsed_milliseconds(Function&& function) {
  const auto started = std::chrono::steady_clock::now();
  function();
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - started)
      .count();
}

void parity_test(const at::Device& device) {
  auto [expert_ids, weights] = offset_inputs(device);
  const RouteMailboxT1 expected = ordinary_materialize(expert_ids, weights);
  RouteMailboxT1 actual;
  if (!deltafin::provider_internal::try_materialize_mps_route_t1(
          expert_ids, weights, actual)) {
    throw std::runtime_error(
        "route mailbox rejected qualified nonzero-offset MPS inputs");
  }
  require_equal(actual, expected, "nonzero-offset route mailbox");

  // A nonqualifying call must fail harmlessly and leave the output untouched.
  RouteMailboxT1 sentinel = actual;
  const at::Tensor cpu_ids = expert_ids.to(at::kCPU);
  if (deltafin::provider_internal::try_materialize_mps_route_t1(
          cpu_ids, weights, sentinel) ||
      sentinel.expert_ids != actual.expert_ids ||
      sentinel.weight_bits != actual.weight_bits) {
    throw std::runtime_error(
        "route mailbox did not preserve output on qualification failure");
  }
  std::cout << "provider_route_mailbox.offset_bits=PASS\n";
}

void timing_test(const at::Device& device) {
  auto [expert_ids, weights] = real_route(device);
  const RouteMailboxT1 expected = ordinary_materialize(expert_ids, weights);
  constexpr std::size_t kWarmups = 12;
  constexpr std::size_t kSamples = 101;
  for (std::size_t index = 0; index < kWarmups; ++index) {
    RouteMailboxT1 mailbox;
    if (!deltafin::provider_internal::try_materialize_mps_route_t1(
            expert_ids.clone(), weights.clone(), mailbox)) {
      throw std::runtime_error("route mailbox failed during timing warmup");
    }
    require_equal(mailbox, expected, "warm route mailbox");
    require_equal(ordinary_materialize(expert_ids.clone(), weights.clone()),
                  expected, "warm ordinary materialization");
  }

  std::vector<double> ordinary_samples;
  std::vector<double> mailbox_samples;
  ordinary_samples.reserve(kSamples);
  mailbox_samples.reserve(kSamples);
  std::size_t mailbox_wins = 0;
  for (std::size_t sample = 0; sample < kSamples; ++sample) {
    const auto measure_ordinary = [&] {
      at::Tensor fresh_ids = expert_ids.clone();
      at::Tensor fresh_weights = weights.clone();
      RouteMailboxT1 result;
      const double elapsed = elapsed_milliseconds(
          [&] { result = ordinary_materialize(fresh_ids, fresh_weights); });
      require_equal(result, expected, "timed ordinary materialization");
      return elapsed;
    };
    const auto measure_mailbox = [&] {
      at::Tensor fresh_ids = expert_ids.clone();
      at::Tensor fresh_weights = weights.clone();
      RouteMailboxT1 result;
      const double elapsed = elapsed_milliseconds([&] {
        if (!deltafin::provider_internal::try_materialize_mps_route_t1(
                fresh_ids, fresh_weights, result)) {
          throw std::runtime_error("route mailbox failed during timing");
        }
      });
      require_equal(result, expected, "timed route mailbox");
      return elapsed;
    };

    double ordinary = 0.0;
    double mailbox = 0.0;
    if ((sample & 1U) == 0) {
      ordinary = measure_ordinary();
      mailbox = measure_mailbox();
    } else {
      mailbox = measure_mailbox();
      ordinary = measure_ordinary();
    }
    ordinary_samples.push_back(ordinary);
    mailbox_samples.push_back(mailbox);
    mailbox_wins += mailbox < ordinary ? 1 : 0;
  }

  const double ordinary_median = median(ordinary_samples);
  const double mailbox_median = median(mailbox_samples);
  const double ordinary_trimmed = trimmed_mean(ordinary_samples);
  const double mailbox_trimmed = trimmed_mean(mailbox_samples);
  const double speedup = ordinary_median / mailbox_median;
  std::cout << "provider_route_mailbox.samples=" << kSamples << '\n'
            << "provider_route_mailbox.ordinary_median_ms="
            << ordinary_median << '\n'
            << "provider_route_mailbox.mailbox_median_ms=" << mailbox_median
            << '\n'
            << "provider_route_mailbox.speedup=" << speedup << '\n'
            << "provider_route_mailbox.paired_wins=" << mailbox_wins << '/'
            << kSamples << '\n';
  if (!(speedup >= 1.05 && mailbox_trimmed < ordinary_trimmed &&
        mailbox_wins >= 67)) {
    throw std::runtime_error(
        "route mailbox did not clear its conservative physical-MPS speed gate");
  }
}

}  // namespace

int main() {
  @autoreleasepool {
    try {
      if (!at::hasMPS()) {
        std::cout << "provider_route_mailbox.mps=SKIP\n";
        return 0;
      }
      const at::Device device(at::kMPS);
      parity_test(device);
      timing_test(device);
      std::cout << "provider_route_mailbox.mps=PASS\n"
                << "provider_route_mailbox.python_runtime=ABSENT\n";
      return 0;
    } catch (const std::exception& error) {
      std::cerr << "provider_route_mailbox=FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
