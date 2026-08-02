#include "provider_pilot.h"

#include <ATen/ATen.h>
#include <ATen/ops/_weight_int8pack_mm.h>
#include <ATen/ops/add.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/topk.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using deltafin::provider_internal::CanonicalPilotPrefetchT1;
using deltafin::provider_internal::MoeRowInt8Matrix;
using deltafin::provider_internal::MoeGeometry;
using deltafin::provider_internal::MoeSpineT1;
using deltafin::provider_internal::PilotPredictionT1;
using deltafin::provider_internal::PilotPredictionRows;
using deltafin::provider_internal::PilotRouterT1;
using deltafin::provider_internal::kPilotTopK;

constexpr std::int64_t kHidden = 32;
constexpr std::int64_t kExperts = 24;
constexpr float kEpsilon = 1.0e-5F;

at::Tensor source() {
  return at::linspace(-0.75, 0.875, kHidden,
                      at::TensorOptions().dtype(at::kFloat))
      .reshape({1, kHidden})
      .contiguous();
}

at::Tensor norm() {
  return at::linspace(0.625, 1.375, kHidden,
                      at::TensorOptions().dtype(at::kFloat));
}

at::Tensor dense_router() {
  return at::linspace(-0.0625, 0.078125, kExperts * kHidden,
                      at::TensorOptions().dtype(at::kFloat))
      .reshape({kExperts, kHidden})
      .contiguous();
}

at::Tensor bias() {
  return at::arange(kExperts, at::TensorOptions().dtype(at::kFloat))
      .mul_(1.0F / 4096.0F)
      .contiguous();
}

PilotRouterT1 dense_weights() {
  PilotRouterT1 weights;
  weights.layer_index = 12;
  weights.generation = 41;
  weights.hidden_size = kHidden;
  weights.expert_count = kExperts;
  weights.post_attention_norm = norm();
  weights.router = MoeRowInt8Matrix{at::Tensor(), at::Tensor(),
                                    dense_router(), {}};
  weights.correction_bias = bias();
  return weights;
}

std::pair<at::Tensor, at::Tensor> reference(
    const at::Tensor& input, const PilotRouterT1& weights,
    const at::Tensor& dense) {
  const at::Tensor variance = at::mean(
      at::pow(input, 2), std::vector<std::int64_t>{-1}, true);
  const at::Tensor normalized = at::mul(
      weights.post_attention_norm,
      at::mul(input, at::rsqrt(at::add(variance, kEpsilon))));
  const at::Tensor logits = at::matmul(normalized, dense.transpose(0, 1));
  const at::Tensor choice =
      at::add(at::sigmoid(logits), weights.correction_bias);
  auto [values, ids] = at::topk(
      choice, static_cast<std::int64_t>(kPilotTopK), -1, true, false);
  return {std::move(ids), std::move(values)};
}

void require_equal(const at::Tensor& actual, const at::Tensor& expected,
                   const char* name) {
  if (!at::equal(actual, expected)) {
    throw std::runtime_error(std::string(name) + " differed");
  }
}

void require_throws(const std::function<void()>& operation, const char* name) {
  try {
    operation();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(std::string(name) + " did not fail closed");
}

void dense_cpu_parity() {
  const at::Tensor input = source();
  const PilotRouterT1 weights = dense_weights();
  const PilotPredictionT1 actual =
      deltafin::provider_internal::predict_pilot_router_t1(input, weights,
                                                           false);
  const auto expected = reference(input, weights, weights.router.dense_f32);
  require_equal(actual.expert_ids, expected.first, "dense pilot IDs");
  require_equal(actual.choice_scores, expected.second,
                "dense pilot choice scores");
  if (actual.layer_index != weights.layer_index ||
      actual.generation != weights.generation ||
      actual.expert_count != weights.expert_count ||
      actual.expert_ids.device().type() != at::kCPU ||
      actual.choice_scores.device().type() != at::kCPU ||
      actual.expert_ids.scalar_type() != at::kLong ||
      actual.choice_scores.scalar_type() != at::kFloat ||
      actual.expert_ids.sizes() != at::IntArrayRef({1, kPilotTopK}) ||
      actual.choice_scores.sizes() != at::IntArrayRef({1, kPilotTopK})) {
    throw std::runtime_error("dense pilot metadata contract changed");
  }
  const float sum = actual.choice_scores.sum().item<float>();
  if (std::abs(sum - 1.0F) < 1.0e-4F) {
    throw std::runtime_error(
        "pilot returned normalized route weights instead of choice scores");
  }
  std::cout << "provider_pilot.dense_cpu=PASS\n";
}

void live_width_router_and_union_parity() {
  const PilotRouterT1 weights = dense_weights();
  for (std::int64_t positions = 1; positions <= 9; ++positions) {
    std::vector<at::Tensor> inputs;
    inputs.reserve(static_cast<std::size_t>(positions));
    for (std::int64_t row = 0; row < positions; ++row) {
      inputs.push_back((source() + static_cast<float>(row) / 37.0F)
                           .contiguous());
    }
    const at::Tensor input = at::cat(inputs, 0).contiguous();
    const PilotPredictionRows actual =
        deltafin::provider_internal::predict_pilot_router_rows(
            input, weights, false);
    const auto expected = reference(input, weights,
                                    weights.router.dense_f32);
    require_equal(actual.expert_ids, expected.first,
                  "multirow dense pilot IDs");
    require_equal(actual.choice_scores, expected.second,
                  "multirow dense pilot scores");
    if (actual.position_count != positions ||
        actual.expert_ids.sizes() !=
            at::IntArrayRef({positions, kPilotTopK}) ||
        actual.choice_scores.sizes() !=
            at::IntArrayRef({positions, kPilotTopK})) {
      throw std::runtime_error("multirow pilot metadata changed its width");
    }
    if (positions == 1) {
      const PilotPredictionT1 scalar =
          deltafin::provider_internal::predict_pilot_router_t1(
              input, weights, false);
      require_equal(actual.expert_ids, scalar.expert_ids,
                    "T=1 rows/scalar pilot IDs");
      require_equal(actual.choice_scores, scalar.choice_scores,
                    "T=1 rows/scalar pilot scores");
    }
  }

  constexpr std::size_t positions = 3;
  constexpr std::uint32_t experts = 64;
  std::array<std::int64_t, positions * kPilotTopK> ids{};
  std::array<float, positions * kPilotTopK> scores{};
  std::array<float, experts> best{};
  best.fill(-std::numeric_limits<float>::infinity());
  for (std::size_t row = 0; row < positions; ++row) {
    for (std::size_t slot = 0; slot < kPilotTopK; ++slot) {
      const std::size_t index = row * kPilotTopK + slot;
      const std::size_t expert = row * kPilotTopK + slot;
      ids[index] = static_cast<std::int64_t>(expert);
      scores[index] = static_cast<float>((index * 29) % 47) / 47.0F;
      best[expert] = scores[index];
    }
  }
  std::vector<std::uint16_t> ranked(experts);
  for (std::size_t expert = 0; expert < experts; ++expert) {
    ranked[expert] = static_cast<std::uint16_t>(expert);
  }
  ranked.erase(
      std::remove_if(ranked.begin(), ranked.end(), [&](const auto expert) {
        return !std::isfinite(best[expert]);
      }),
      ranked.end());
  std::sort(ranked.begin(), ranked.end(), [&](const auto left,
                                               const auto right) {
    if (best[left] != best[right]) {
      return best[left] > best[right];
    }
    return left < right;
  });
  ranked.resize(deltafin::provider_internal::kPilotMaxPrefetch);
  std::sort(ranked.begin(), ranked.end());
  const auto unioned =
      deltafin::provider_internal::canonicalize_pilot_prefetch_rows(
          ids, scores, positions, experts,
          deltafin::provider_internal::kPilotMaxPrefetch);
  if (unioned.candidate_count != 48 || unioned.count != ranked.size() ||
      !std::equal(ranked.begin(), ranked.end(),
                  unioned.expert_ids.begin())) {
    throw std::runtime_error(
        "multirow PILOT union/cap diverged from independent oracle");
  }
  std::cout << "provider_pilot.live_widths=PASS (T=1..9, cap=32)\n";
}

PilotRouterT1 router_on(const PilotRouterT1& source_router,
                        const at::Device& device) {
  PilotRouterT1 result = source_router;
  result.post_attention_norm =
      source_router.post_attention_norm.to(device).contiguous();
  result.correction_bias =
      source_router.correction_bias.to(device).contiguous();
  if (source_router.packed_int8_qualified) {
    result.router.quantized =
        source_router.router.quantized.to(device).contiguous();
    result.router.row_scales =
        source_router.router.row_scales.to(device).contiguous();
    result.router.dense_f32 = at::Tensor();
  } else {
    result.router.dense_f32 =
        source_router.router.dense_f32.to(device).contiguous();
  }
  return result;
}

void live_width_mps_parity() {
  const at::Device device(at::kMPS);
  PilotRouterT1 dense = router_on(dense_weights(), device);
  for (std::int64_t positions = 1; positions <= 9; ++positions) {
    const at::Tensor input =
        source().repeat({positions, 1}).to(device).contiguous();
    const PilotPredictionRows actual =
        deltafin::provider_internal::predict_pilot_router_rows(
            input, dense, false);
    const auto expected = reference(input, dense, dense.router.dense_f32);
    require_equal(actual.expert_ids, expected.first,
                  "MPS multirow dense pilot IDs");
    require_equal(actual.choice_scores, expected.second,
                  "MPS multirow dense pilot scores");
  }

  constexpr std::int64_t kMpsPackedExperts = 32;
  const at::Tensor quantized =
      at::arange(kMpsPackedExperts * kHidden,
                 at::TensorOptions().dtype(at::kLong))
          .remainder(15)
          .sub_(7)
          .to(at::kChar)
          .reshape({kMpsPackedExperts, kHidden})
          .contiguous();
  const at::Tensor scales =
      at::linspace(1.0F / 256.0F, 1.0F / 64.0F, kMpsPackedExperts,
                   at::TensorOptions().dtype(at::kFloat))
          .contiguous();
  PilotRouterT1 packed = dense_weights();
  packed.expert_count = kMpsPackedExperts;
  packed.packed_int8_qualified = true;
  packed.router = MoeRowInt8Matrix{quantized, scales, at::Tensor(), {}};
  packed.correction_bias =
      at::arange(kMpsPackedExperts,
                 at::TensorOptions().dtype(at::kFloat))
          .mul_(1.0F / 4096.0F)
          .contiguous();
  packed = router_on(packed, device);
  for (std::int64_t positions = 1; positions <= 9; ++positions) {
    const at::Tensor input =
        source().repeat({positions, 1}).to(device).contiguous();
    const PilotPredictionRows actual =
        deltafin::provider_internal::predict_pilot_router_rows(
            input, packed, false);
    const at::Tensor variance = at::mean(
        at::pow(input, 2), std::vector<std::int64_t>{-1}, true);
    const at::Tensor normalized = at::mul(
        packed.post_attention_norm,
        at::mul(input, at::rsqrt(at::add(variance, kEpsilon))));
    const at::Tensor logits = at::_weight_int8pack_mm(
        normalized, packed.router.quantized, packed.router.row_scales);
    const at::Tensor choice =
        at::add(at::sigmoid(logits), packed.correction_bias);
    auto [expected_scores, expected_ids] = at::topk(
        choice, static_cast<std::int64_t>(kPilotTopK), -1, true, false);
    require_equal(actual.expert_ids, expected_ids,
                  "MPS multirow packed pilot IDs");
    require_equal(actual.choice_scores, expected_scores,
                  "MPS multirow packed pilot scores");
  }
  std::cout << "provider_pilot.mps_live_widths=PASS (T=1..9)\n";
}

void row_int8_cpu_parity() {
  const at::Tensor quantized =
      at::arange(kExperts * kHidden,
                 at::TensorOptions().dtype(at::kLong))
          .remainder(15)
          .sub_(7)
          .to(at::kChar)
          .reshape({kExperts, kHidden})
          .contiguous();
  const at::Tensor scales =
      at::linspace(1.0F / 256.0F, 1.0F / 64.0F, kExperts,
                   at::TensorOptions().dtype(at::kFloat))
          .contiguous();
  PilotRouterT1 weights = dense_weights();
  weights.packed_int8_qualified = true;
  weights.router =
      MoeRowInt8Matrix{quantized, scales, at::Tensor(), {}};

  const at::Tensor input = source();
  const PilotPredictionT1 actual =
      deltafin::provider_internal::predict_pilot_router_t1(input, weights,
                                                           false);
  const at::Tensor variance = at::mean(
      at::pow(input, 2), std::vector<std::int64_t>{-1}, true);
  const at::Tensor normalized = at::mul(
      weights.post_attention_norm,
      at::mul(input, at::rsqrt(at::add(variance, kEpsilon))));
  const at::Tensor logits =
      at::_weight_int8pack_mm(normalized, quantized, scales);
  const at::Tensor choice =
      at::add(at::sigmoid(logits), weights.correction_bias);
  auto [expected_values, expected_ids] = at::topk(
      choice, static_cast<std::int64_t>(kPilotTopK), -1, true, false);
  require_equal(actual.expert_ids, expected_ids, "row-int8 pilot IDs");
  require_equal(actual.choice_scores, expected_values,
                "row-int8 pilot choice scores");
  std::cout << "provider_pilot.row_int8_cpu=PASS\n";
}

void compact_clone_lifetime_and_quantization() {
  MoeSpineT1 dense_source;
  dense_source.layer_index = 17;
  dense_source.generation = 9001;
  dense_source.geometry = MoeGeometry{
      static_cast<std::uint32_t>(kHidden),
      static_cast<std::uint32_t>(kHidden),
      static_cast<std::uint32_t>(kHidden),
      static_cast<std::uint32_t>(kExperts), 64};
  dense_source.router.dense_f32 = dense_router();
  // Exercise the all-zero-row scale rule as well as ordinary rows.
  dense_source.router.dense_f32.select(0, 0).zero_();
  dense_source.router_correction_bias = bias();
  at::Tensor source_norm = norm();
  const PilotRouterT1 compact =
      deltafin::provider_internal::clone_compact_pilot_router_t1(
          dense_source, source_norm, false);
  const at::Tensor peaks =
      dense_source.router.dense_f32.abs().amax(1, false);
  const at::Tensor expected_scales = at::where(
      peaks > 0.0F, peaks / 127.0F, at::ones_like(peaks));
  const at::Tensor expected_q =
      at::round(dense_source.router.dense_f32 /
                expected_scales.unsqueeze(1))
          .clamp(-127.0F, 127.0F)
          .to(at::kChar)
          .contiguous();
  require_equal(compact.router.quantized, expected_q,
                "dense compact pilot qweight");
  require_equal(compact.router.row_scales, expected_scales,
                "dense compact pilot scales");
  if (compact.router.dense_f32.defined() ||
      compact.router.quantized.is_alias_of(
          dense_source.router.dense_f32) ||
      compact.post_attention_norm.is_alias_of(source_norm) ||
      compact.correction_bias.is_alias_of(
          dense_source.router_correction_bias)) {
    throw std::runtime_error(
        "compact pilot retained an authoritative source allocation");
  }
  const PilotPredictionT1 before =
      deltafin::provider_internal::predict_pilot_router_t1(
          source(), compact, false);
  dense_source.router.dense_f32.fill_(99.0F);
  dense_source.router_correction_bias.fill_(-99.0F);
  source_norm.fill_(77.0F);
  const PilotPredictionT1 after =
      deltafin::provider_internal::predict_pilot_router_t1(
          source(), compact, false);
  require_equal(after.expert_ids, before.expert_ids,
                "detached dense pilot IDs");
  require_equal(after.choice_scores, before.choice_scores,
                "detached dense pilot scores");

  MoeSpineT1 packed_source = dense_source;
  packed_source.layer_index = 18;
  packed_source.generation = 9002;
  packed_source.packed_int8_qualified = true;
  packed_source.router = MoeRowInt8Matrix{
      expected_q.clone(), expected_scales.clone(), at::Tensor(), {}};
  packed_source.router_correction_bias = bias();
  at::Tensor packed_norm = norm();
  const PilotRouterT1 packed =
      deltafin::provider_internal::clone_compact_pilot_router_t1(
          packed_source, packed_norm, false);
  if (packed.router.quantized.is_alias_of(
          packed_source.router.quantized) ||
      packed.router.row_scales.is_alias_of(
          packed_source.router.row_scales)) {
    throw std::runtime_error(
        "compact row-int8 pilot retained its source slab");
  }
  require_equal(packed.router.quantized, packed_source.router.quantized,
                "packed compact pilot qweight");
  require_equal(packed.router.row_scales,
                packed_source.router.row_scales,
                "packed compact pilot scales");
  std::cout << "provider_pilot.compact_lifetime=PASS\n";
}

void tie_and_canonical_cap() {
  std::array<std::int64_t, kPilotTopK> ids{
      9, 3, 7, 2, 18, 1, 14, 22, 5, 16, 10, 8, 21, 4, 12, 6};
  std::array<float, kPilotTopK> scores{
      0.8F, 0.8F, 0.8F, 0.7F, 0.6F, 0.5F, 0.4F, 0.3F,
      0.2F, 0.1F, 0.0F, -0.1F, -0.2F, -0.3F, -0.4F, -0.5F};
  const CanonicalPilotPrefetchT1 capped =
      deltafin::provider_internal::canonicalize_pilot_prefetch_t1(
          ids, scores, kExperts, 2);
  if (capped.count != 2 || capped.expert_ids[0] != 3 ||
      capped.expert_ids[1] != 7) {
    throw std::runtime_error(
        "pilot cap did not apply score/ID tie ordering then canonical order");
  }

  const CanonicalPilotPrefetchT1 all =
      deltafin::provider_internal::canonicalize_pilot_prefetch_t1(
          ids, scores, kExperts, 1000);
  if (all.count != kPilotTopK ||
      !std::is_sorted(all.expert_ids.begin(),
                      all.expert_ids.begin() +
                          static_cast<std::ptrdiff_t>(all.count))) {
    throw std::runtime_error("pilot canonical cap did not clamp/sort");
  }
  const CanonicalPilotPrefetchT1 disabled =
      deltafin::provider_internal::canonicalize_pilot_prefetch_t1(
          ids, scores, kExperts, 0);
  if (disabled.count != 0) {
    throw std::runtime_error("zero pilot cap did not disable speculation");
  }

  PilotRouterT1 tied = dense_weights();
  tied.router.dense_f32 = at::zeros({kExperts, kHidden},
                                    at::TensorOptions().dtype(at::kFloat));
  tied.correction_bias =
      at::zeros({kExperts}, at::TensorOptions().dtype(at::kFloat));
  const PilotPredictionT1 tied_actual =
      deltafin::provider_internal::predict_pilot_router_t1(source(), tied,
                                                           false);
  const auto tied_expected =
      reference(source(), tied, tied.router.dense_f32);
  require_equal(tied_actual.expert_ids, tied_expected.first,
                "ATen tie-preserving pilot IDs");
  const at::Tensor tied_ids_cpu = tied_actual.expert_ids.contiguous();
  const auto* tied_ids = tied_ids_cpu.const_data_ptr<std::int64_t>();
  std::array<bool, kExperts> seen{};
  for (std::size_t index = 0; index < kPilotTopK; ++index) {
    if (tied_ids[index] < 0 || tied_ids[index] >= kExperts ||
        seen[static_cast<std::size_t>(tied_ids[index])]) {
      throw std::runtime_error("tied pilot topk repeated an expert");
    }
    seen[static_cast<std::size_t>(tied_ids[index])] = true;
  }
  std::cout << "provider_pilot.tie_and_cap=PASS\n";
}

void validation_and_fail_soft() {
  const PilotRouterT1 weights = dense_weights();
  require_throws(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::predict_pilot_router_t1(
                source().to(at::kBFloat16), weights, false));
      },
      "bf16 source");
  require_throws(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::predict_pilot_router_t1(
                source(), weights, true));
      },
      "non-K3 production geometry");
  PilotRouterT1 missing = weights;
  missing.generation = 0;
  require_throws(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::predict_pilot_router_t1(
                source(), missing, false));
      },
      "missing generation");
  if (deltafin::provider_internal::try_predict_pilot_router_t1(
          source(), missing, false)
          .has_value()) {
    throw std::runtime_error("fail-soft wrapper published an invalid result");
  }

  std::array<std::int64_t, kPilotTopK> ids{};
  std::array<float, kPilotTopK> scores{};
  for (std::size_t index = 0; index < kPilotTopK; ++index) {
    ids[index] = static_cast<std::int64_t>(index);
    scores[index] = static_cast<float>(index);
  }
  ids[4] = ids[3];
  require_throws(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::canonicalize_pilot_prefetch_t1(
                ids, scores, kExperts, kPilotTopK));
      },
      "duplicate prefetch expert");
  ids[4] = 4;
  scores[6] = std::numeric_limits<float>::quiet_NaN();
  require_throws(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::canonicalize_pilot_prefetch_t1(
                ids, scores, kExperts, kPilotTopK));
      },
      "non-finite prefetch score");
  std::cout << "provider_pilot.validation=PASS\n";
}

}  // namespace

int main(const int argc, const char* const* argv) {
  try {
    bool run_mps = false;
    for (int index = 1; index < argc; ++index) {
      const std::string argument(argv[index]);
      if (argument == "--device" && index + 1 < argc) {
        run_mps = std::string(argv[++index]) == "mps";
      } else {
        throw std::invalid_argument("unknown provider PILOT test argument");
      }
    }
    dense_cpu_parity();
    live_width_router_and_union_parity();
    row_int8_cpu_parity();
    compact_clone_lifetime_and_quantization();
    tie_and_canonical_cap();
    validation_and_fail_soft();
    if (run_mps) {
      live_width_mps_parity();
    }
    std::cout << "provider_pilot.result=PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "provider_pilot.result=FAIL: " << error.what() << '\n';
    return 1;
  }
}
