#include "provider_device.h"
#include "provider_mla.h"

#include <ATen/ATen.h>
#include <ATen/ops/_weight_int8pack_mm.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/softmax.h>
#include <c10/core/InferenceMode.h>

#if defined(__APPLE__)
#include <torch/mps.h>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using deltafin::provider_internal::MlaCache;
using deltafin::provider_internal::MlaCacheRepresentation;
using deltafin::provider_internal::MlaCacheTransaction;
using deltafin::provider_internal::MlaAbsorbedKeyValue;
using deltafin::provider_internal::MlaExecutionStage;
using deltafin::provider_internal::MlaExecutionTrace;
using deltafin::provider_internal::MlaInputBundle;
using deltafin::provider_internal::MlaLinearEncoding;
using deltafin::provider_internal::MlaLinearWeight;
using deltafin::provider_internal::MlaPreparedDecode;
using deltafin::provider_internal::MlaShape;
using deltafin::provider_internal::MlaWeights;

struct ReferenceCache {
  at::Tensor keys;
  at::Tensor values;
};

struct Options {
  std::uint32_t device = DELTAFIN_PROVIDER_DEVICE_CPU_V1;
  bool benchmark_input_bundle = false;
};

at::Tensor deterministic_tensor(const at::IntArrayRef shape,
                                const at::Device& device,
                                const std::int64_t salt,
                                const float scale = 0.03125F) {
  std::int64_t elements = 1;
  for (const std::int64_t dimension : shape) {
    elements *= dimension;
  }
  at::Tensor cpu = at::empty(shape, at::TensorOptions().dtype(at::kFloat));
  float* values = cpu.data_ptr<float>();
  for (std::int64_t index = 0; index < elements; ++index) {
    const auto centered = static_cast<std::int64_t>(
        ((index + 1) * (salt * 13 + 17) + salt * 7) % 61) - 30;
    values[index] = static_cast<float>(centered) * scale;
  }
  return cpu.to(device);
}

MlaLinearWeight dense_weight(const std::int64_t rows,
                             const std::int64_t columns,
                             const at::Device& device,
                             const std::int64_t salt) {
  return MlaLinearWeight{
      .encoding = MlaLinearEncoding::DenseF32,
      .data = deterministic_tensor({rows, columns}, device, salt, 0.0078125F),
      .row_scale = {},
      // Explicit: GCC errors on omitted members under -Werror; Clang does not.
      .original_bf16 = {},
  };
}

MlaLinearWeight packed_weight(const std::int64_t rows,
                              const std::int64_t columns,
                              const at::Device& device,
                              const std::int64_t salt) {
  at::Tensor data =
      at::empty({rows, columns}, at::TensorOptions().dtype(at::kChar));
  at::Tensor scales =
      at::empty({rows}, at::TensorOptions().dtype(at::kFloat));
  auto values = data.accessor<std::int8_t, 2>();
  auto row_scales = scales.accessor<float, 1>();
  constexpr float scale_cycle[] = {
      0.015625F, 0.03125F, 0.0625F, 0.125F};
  for (std::int64_t row = 0; row < rows; ++row) {
    row_scales[row] = scale_cycle[(row + salt) % 4];
    for (std::int64_t column = 0; column < columns; ++column) {
      std::int64_t value =
          ((row + 3) * (column + salt + 5) + salt * 11) % 15 - 7;
      if (value == 0) {
        value = 3;
      }
      values[row][column] = static_cast<std::int8_t>(value);
    }
  }
  return MlaLinearWeight{
      .encoding = MlaLinearEncoding::RowI8F32Scale,
      .data = data.to(device),
      .row_scale = scales.to(device),
      // Explicit: GCC errors on omitted members under -Werror; Clang does not.
      .original_bf16 = {},
  };
}

MlaWeights make_weights(const MlaShape& shape, const at::Device& device) {
  const std::int64_t query_width =
      shape.num_heads * shape.query_head_dim();
  const std::int64_t value_width =
      shape.num_heads * shape.value_head_dim;
  return MlaWeights{
      .query_a =
          dense_weight(shape.q_lora_rank, shape.hidden_size, device, 1),
      .query_a_norm =
          deterministic_tensor({shape.q_lora_rank}, device, 2, 0.015625F) +
          1.0,
      .query_b = dense_weight(query_width, shape.q_lora_rank, device, 3),
      .key_value_a = dense_weight(
          shape.kv_lora_rank + shape.qk_rope_head_dim,
          shape.hidden_size, device, 4),
      .key_value_a_norm =
          deterministic_tensor({shape.kv_lora_rank}, device, 5, 0.015625F) +
          1.0,
      .key_value_b = dense_weight(
          shape.num_heads *
              (shape.qk_nope_head_dim + shape.value_head_dim),
          shape.kv_lora_rank, device, 6),
      .output_gate =
          dense_weight(value_width, shape.hidden_size, device, 7),
      .output = dense_weight(shape.hidden_size, value_width, device, 8),
  };
}

MlaWeights make_packed_weights(const MlaShape& shape,
                               const at::Device& device) {
  const std::int64_t query_width =
      shape.num_heads * shape.query_head_dim();
  const std::int64_t value_width =
      shape.num_heads * shape.value_head_dim;
  return MlaWeights{
      .query_a =
          packed_weight(shape.q_lora_rank, shape.hidden_size, device, 21),
      .query_a_norm =
          deterministic_tensor({shape.q_lora_rank}, device, 22, 0.015625F) +
          1.0,
      .query_b =
          packed_weight(query_width, shape.q_lora_rank, device, 23),
      .key_value_a = packed_weight(
          shape.kv_lora_rank + shape.qk_rope_head_dim,
          shape.hidden_size, device, 24),
      .key_value_a_norm =
          deterministic_tensor({shape.kv_lora_rank}, device, 25, 0.015625F) +
          1.0,
      .key_value_b = packed_weight(
          shape.num_heads *
              (shape.qk_nope_head_dim + shape.value_head_dim),
          shape.kv_lora_rank, device, 26),
      .output_gate =
          packed_weight(value_width, shape.hidden_size, device, 27),
      .output =
          packed_weight(shape.hidden_size, value_width, device, 28),
  };
}

at::Tensor reference_linear(const at::Tensor& input,
                            const MlaLinearWeight& weight) {
  const std::int64_t columns = input.size(-1);
  const at::Tensor flat = input.reshape({-1, columns});
  at::Tensor output;
  switch (weight.encoding) {
    case MlaLinearEncoding::DenseF32:
      output = at::matmul(flat, weight.data.transpose(0, 1));
      break;
    case MlaLinearEncoding::RowI8F32Scale:
      output = at::matmul(
          flat,
          (weight.data.to(at::kFloat) * weight.row_scale.unsqueeze(1))
              .transpose(0, 1));
      break;
    default:
      throw std::logic_error("test reference saw an unknown weight encoding");
  }
  auto shape = input.sizes().vec();
  shape.back() = weight.data.size(0);
  return output.view(shape);
}

at::Tensor reference_norm(const at::Tensor& input, const at::Tensor& weight,
                          const double epsilon) {
  const at::Tensor variance = at::mean(at::pow(input, 2), {-1}, true);
  return weight * (input * at::rsqrt(variance + epsilon));
}

at::Tensor reference_decode(const at::Tensor& hidden,
                            const MlaWeights& weights,
                            const MlaShape& shape, ReferenceCache& cache) {
  const std::int64_t query_head_dim = shape.query_head_dim();
  const std::int64_t value_width =
      shape.num_heads * shape.value_head_dim;
  const at::Tensor raw_query = reference_linear(
      reference_norm(reference_linear(hidden, weights.query_a),
                     weights.query_a_norm, shape.rms_epsilon),
      weights.query_b);
  const at::Tensor old_query =
      raw_query.view({1, 1, shape.num_heads, query_head_dim}).transpose(1, 2);
  const at::Tensor query_nope =
      old_query.narrow(-1, 0, shape.qk_nope_head_dim);
  const at::Tensor query_rope = old_query.narrow(
      -1, shape.qk_nope_head_dim, shape.qk_rope_head_dim);
  const at::Tensor query = at::cat({query_nope, query_rope}, -1);

  const at::Tensor compressed = reference_linear(hidden, weights.key_value_a);
  const at::Tensor latent =
      compressed.narrow(-1, 0, shape.kv_lora_rank);
  const at::Tensor rope = compressed.narrow(
      -1, shape.kv_lora_rank, shape.qk_rope_head_dim);
  const at::Tensor expanded = reference_linear(
      reference_norm(latent, weights.key_value_a_norm, shape.rms_epsilon),
      weights.key_value_b);
  const at::Tensor heads =
      expanded
          .view({1, 1, shape.num_heads,
                 shape.qk_nope_head_dim + shape.value_head_dim})
          .transpose(1, 2);
  const at::Tensor key_nope =
      heads.narrow(-1, 0, shape.qk_nope_head_dim);
  const at::Tensor new_value = heads.narrow(
      -1, shape.qk_nope_head_dim, shape.value_head_dim);
  const at::Tensor expanded_rope =
      rope.view({1, 1, 1, shape.qk_rope_head_dim})
          .expand({1, shape.num_heads, 1, shape.qk_rope_head_dim});
  const at::Tensor new_key = at::cat({key_nope, expanded_rope}, -1);
  cache.keys = cache.keys.defined() ? at::cat({cache.keys, new_key}, 2)
                                    : new_key;
  cache.values = cache.values.defined()
                     ? at::cat({cache.values, new_value}, 2)
                     : new_value;

  const double scaling =
      std::pow(static_cast<double>(query_head_dim), -0.5);
  const at::Tensor scores =
      at::matmul(query, cache.keys.transpose(-1, -2)) * scaling;
  const at::Tensor probabilities =
      at::softmax(scores, -1, at::kFloat).to(query.scalar_type());
  at::Tensor attention =
      at::matmul(probabilities, cache.values).transpose(1, 2).contiguous();
  attention = attention.reshape({1, 1, value_width}).contiguous();
  attention = attention *
              at::sigmoid(reference_linear(hidden, weights.output_gate));
  return reference_linear(attention, weights.output);
}

at::Tensor reference_positions_empty_cache(const at::Tensor& hidden,
                                           const MlaWeights& weights,
                                           const MlaShape& shape) {
  const std::int64_t positions = hidden.size(1);
  const std::int64_t query_head_dim = shape.query_head_dim();
  const std::int64_t value_width =
      shape.num_heads * shape.value_head_dim;
  const at::Tensor raw_query = reference_linear(
      reference_norm(reference_linear(hidden, weights.query_a),
                     weights.query_a_norm, shape.rms_epsilon),
      weights.query_b);
  const at::Tensor original =
      raw_query
          .view({1, positions, shape.num_heads, query_head_dim})
          .transpose(1, 2);
  const at::Tensor query = at::cat(
      {original.narrow(-1, 0, shape.qk_nope_head_dim),
       original.narrow(-1, shape.qk_nope_head_dim,
                       shape.qk_rope_head_dim)},
      -1);

  const at::Tensor compressed = reference_linear(hidden, weights.key_value_a);
  const at::Tensor latent =
      compressed.narrow(-1, 0, shape.kv_lora_rank);
  const at::Tensor rope = compressed.narrow(
      -1, shape.kv_lora_rank, shape.qk_rope_head_dim);
  const at::Tensor expanded = reference_linear(
      reference_norm(latent, weights.key_value_a_norm, shape.rms_epsilon),
      weights.key_value_b);
  const at::Tensor heads =
      expanded
          .view({1, positions, shape.num_heads,
                 shape.qk_nope_head_dim + shape.value_head_dim})
          .transpose(1, 2);
  const at::Tensor key_nope =
      heads.narrow(-1, 0, shape.qk_nope_head_dim);
  const at::Tensor values = heads.narrow(
      -1, shape.qk_nope_head_dim, shape.value_head_dim);
  const at::Tensor expanded_rope =
      rope.view({1, 1, positions, shape.qk_rope_head_dim})
          .expand({1, shape.num_heads, positions,
                   shape.qk_rope_head_dim});
  const at::Tensor keys = at::cat({key_nope, expanded_rope}, -1);

  at::Tensor scores =
      at::einsum("bhqd,bhkd->bhqk", {query, keys}) *
      std::pow(static_cast<double>(query_head_dim), -0.5);
  if (positions > 1) {
    const at::Tensor mask = at::triu(
        at::full({positions, positions},
                 -std::numeric_limits<float>::infinity(), scores.options()),
        1);
    scores = scores + mask.view({1, 1, positions, positions});
  }
  const at::Tensor probabilities =
      at::softmax(scores, -1, at::kFloat).to(query.scalar_type());
  at::Tensor attention =
      at::einsum("bhqk,bhkd->bhqd", {probabilities, values})
          .transpose(1, 2)
          .contiguous()
          .reshape({1, positions, value_width})
          .contiguous();
  attention = attention *
      at::sigmoid(reference_linear(hidden, weights.output_gate));
  return reference_linear(attention, weights.output);
}

void require_trace(const MlaExecutionTrace& actual,
                   const std::span<const MlaExecutionStage> expected,
                   const char* name) {
  if (actual.count != expected.size()) {
    throw std::runtime_error(std::string(name) +
                             " had the wrong stage count");
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (actual.stages[index] != expected[index]) {
      throw std::runtime_error(std::string(name) +
                               " changed provider submission order");
    }
  }
}

void require_close(const at::Tensor& actual, const at::Tensor& expected,
                   const char* name, double rtol = 2.0e-5,
                   double atol = 2.0e-6) {
  const at::Tensor actual_cpu = actual.to(at::kCPU);
  const at::Tensor expected_cpu = expected.to(at::kCPU);
  const double maximum =
      at::max(at::abs(actual_cpu - expected_cpu)).item<double>();
  if (!at::allclose(actual_cpu, expected_cpu, rtol, atol, true)) {
    throw std::runtime_error(std::string(name) +
                             " parity failed; max_abs=" +
                             std::to_string(maximum));
  }
}

void require_equal(const at::Tensor& actual, const at::Tensor& expected,
                   const char* name) {
  if (!at::equal(actual.to(at::kCPU), expected.to(at::kCPU))) {
    throw std::runtime_error(std::string(name) + " was not bit-exact");
  }
}

struct CompactReassociationFalsifier {
  std::uint64_t differing_elements = 0;
  double maximum_absolute_difference = 0.0;
};

void observe_fp32_bits(const at::Tensor& candidate,
                       const at::Tensor& expanded,
                       CompactReassociationFalsifier& observation) {
  const at::Tensor candidate_cpu = candidate.to(at::kCPU).contiguous();
  const at::Tensor expanded_cpu = expanded.to(at::kCPU).contiguous();
  if (candidate_cpu.scalar_type() != at::kFloat ||
      expanded_cpu.scalar_type() != at::kFloat ||
      candidate_cpu.sizes() != expanded_cpu.sizes()) {
    throw std::runtime_error(
        "compact MLA falsifier received incompatible tensors");
  }
  const float* candidate_values = candidate_cpu.const_data_ptr<float>();
  const float* expanded_values = expanded_cpu.const_data_ptr<float>();
  for (std::int64_t index = 0; index < candidate_cpu.numel(); ++index) {
    if (std::bit_cast<std::uint32_t>(candidate_values[index]) !=
        std::bit_cast<std::uint32_t>(expanded_values[index])) {
      ++observation.differing_elements;
    }
    observation.maximum_absolute_difference = std::max(
        observation.maximum_absolute_difference,
        std::abs(static_cast<double>(candidate_values[index]) -
                 static_cast<double>(expanded_values[index])));
  }
}

template <typename Function>
void require_throws(Function&& function, const char* name) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception&) {
    rejected = true;
  }
  if (!rejected) {
    throw std::runtime_error(std::string(name) + " was not rejected");
  }
}

MlaShape test_shape() {
  return MlaShape{
      .hidden_size = 12,
      .num_heads = 4,
      .q_lora_rank = 5,
      .kv_lora_rank = 3,
      .qk_nope_head_dim = 4,
      .qk_rope_head_dim = 2,
      .value_head_dim = 4,
      .max_context = 64,
      .rms_epsilon = 1.0e-5,
  };
}

MlaShape packed_test_shape() {
  return MlaShape{
      .hidden_size = 32,
      .num_heads = 2,
      .q_lora_rank = 32,
      .kv_lora_rank = 32,
      .qk_nope_head_dim = 16,
      .qk_rope_head_dim = 32,
      .value_head_dim = 16,
      .max_context = 32,
      .rms_epsilon = 1.0e-5,
  };
}

void run_packed_parity(const at::Device& device) {
  const MlaShape shape = packed_test_shape();
  const MlaWeights reference_weights = make_packed_weights(shape, device);
  MlaWeights bundled_weights = reference_weights;
  const MlaInputBundle bundle =
      deltafin::provider_internal::bundle_mla_input_weights(
          shape, bundled_weights);
  const std::int64_t component_elements =
      reference_weights.query_a.data.numel() +
      reference_weights.key_value_a.data.numel() +
      reference_weights.output_gate.data.numel();
  const std::int64_t scale_elements =
      reference_weights.query_a.row_scale.numel() +
      reference_weights.key_value_a.row_scale.numel() +
      reference_weights.output_gate.row_scale.numel();
  if (bundle.projection.data.numel() != component_elements ||
      bundle.projection.row_scale.numel() != scale_elements) {
    throw std::runtime_error(
        "MLA same-input bundle changed packed payload residency");
  }
  ReferenceCache reference_cache;
  MlaCache fallback_cache(shape);
  MlaCache bundled_cache(shape);
  MlaCache compact_cache(shape, MlaCacheRepresentation::CompactLatentF32);
  const MlaAbsorbedKeyValue absorbed =
      deltafin::provider_internal::absorb_mla_key_value(
          shape, bundled_weights.key_value_b);
  for (std::int64_t token = 0; token < 5; ++token) {
    const at::Tensor hidden = deterministic_tensor(
        {1, 1, shape.hidden_size}, device, 1200 + token, 0.03125F);
    const at::Tensor expected =
        reference_decode(hidden, reference_weights, shape, reference_cache);
    MlaPreparedDecode fallback =
        deltafin::provider_internal::prepare_mla_decode(
            hidden, bundled_weights, fallback_cache, true);
    MlaPreparedDecode candidate =
        deltafin::provider_internal::prepare_mla_decode(
            hidden, bundled_weights, bundled_cache, true, &bundle);
    MlaPreparedDecode compact =
        deltafin::provider_internal::prepare_mla_positions(
            hidden, bundled_weights, compact_cache, &absorbed, true, &bundle);
    require_close(candidate.output, expected, "row-int8 MLA decode");
    require_equal(candidate.output, fallback.output,
                  "same-input bundled MLA projection");
    // packed_test_shape's kv_lora_rank/qk_rope_head_dim (32) are ~10x
    // test_shape's, so the documented reassociation drift from moving kv_b
    // across the score/value contractions (see prepare_mla_positions) grows
    // past the default tolerance on some platforms' BLAS reduction order.
    require_close(compact.output, expected,
                  "row-int8 absorbed compact MLA decode", 5.0e-5, 1.0e-5);
    deltafin::provider_internal::commit_mla_decode(fallback_cache, fallback);
    deltafin::provider_internal::commit_mla_decode(bundled_cache, candidate);
    deltafin::provider_internal::commit_mla_decode(compact_cache, compact);
  }
}

void run_live_shape_schedule(const at::Device& device) {
  const MlaShape shape = test_shape();
  const MlaWeights weights = make_weights(shape, device);
  constexpr std::array<MlaExecutionStage, 7> kLiveOrder{
      MlaExecutionStage::QueryA,
      MlaExecutionStage::QueryB,
      MlaExecutionStage::KeyValueA,
      MlaExecutionStage::KeyValueB,
      MlaExecutionStage::Attention,
      MlaExecutionStage::OutputGate,
      MlaExecutionStage::Output,
  };
  constexpr std::array<MlaExecutionStage, 7> kDecodeOrder{
      MlaExecutionStage::QueryA,
      MlaExecutionStage::KeyValueA,
      MlaExecutionStage::OutputGate,
      MlaExecutionStage::QueryB,
      MlaExecutionStage::KeyValueB,
      MlaExecutionStage::Attention,
      MlaExecutionStage::Output,
  };
  constexpr std::array<std::int64_t, 9> kPositions{1, 2, 3, 4, 5,
                                                   6, 7, 8, 9};
  for (const std::int64_t positions : kPositions) {
    const at::Tensor hidden = deterministic_tensor(
        {1, positions, shape.hidden_size}, device, 2600 + positions,
        0.03125F);
    const at::Tensor expected =
        reference_positions_empty_cache(hidden, weights, shape);
    MlaCache cache(shape);
    MlaExecutionTrace trace;
    MlaPreparedDecode prepared =
        deltafin::provider_internal::prepare_mla_positions(
            hidden, weights, cache, nullptr, true, nullptr, &trace);
    require_equal(prepared.output, expected,
                  "independent live-shape MLA oracle");
    if (positions == 1) {
      require_trace(trace, kDecodeOrder,
                    "one-position MLA submission schedule");
    } else {
      require_trace(trace, kLiveOrder,
                    "T-wide MLA submission schedule");
    }
    deltafin::provider_internal::cancel_mla_decode(cache, prepared);

    if (positions > 1) {
      // Production may retain a zero-copy same-input storage descriptor for
      // T=1. A poisoned descriptor proves the T-wide path does not execute its
      // schedule-changing super-projection.
      MlaInputBundle poisoned{
          .projection = MlaLinearWeight{
              .encoding = MlaLinearEncoding::RowI8F32Scale,
              .data = at::zeros(
                  {shape.q_lora_rank + shape.kv_lora_rank +
                       shape.qk_rope_head_dim +
                       shape.num_heads * shape.value_head_dim,
                   shape.hidden_size},
                  hidden.options().dtype(at::kChar)),
              .row_scale = at::ones(
                  {shape.q_lora_rank + shape.kv_lora_rank +
                   shape.qk_rope_head_dim +
                   shape.num_heads * shape.value_head_dim},
                  hidden.options().dtype(at::kFloat)),
          // Explicit: GCC errors on omitted members under -Werror; Clang does not.
          .original_bf16 = {},
          },
          .query_a_rows = shape.q_lora_rank,
          .key_value_a_rows =
              shape.kv_lora_rank + shape.qk_rope_head_dim,
          .output_gate_rows = shape.num_heads * shape.value_head_dim,
      };
      MlaCache bundled_cache(shape);
      MlaExecutionTrace bundled_trace;
      MlaPreparedDecode bundled =
          deltafin::provider_internal::prepare_mla_positions(
              hidden, weights, bundled_cache, nullptr, true, &poisoned,
              &bundled_trace);
      require_equal(bundled.output, expected,
                    "T-wide MLA ignored same-input bundle");
      require_trace(bundled_trace, kLiveOrder,
                    "T-wide MLA bundled storage fallback schedule");
      deltafin::provider_internal::cancel_mla_decode(bundled_cache, bundled);
    }
  }
}

void run_parity(const at::Device& device) {
  const c10::InferenceMode inference_guard;
  const MlaShape shape = test_shape();
  shape.validate();
  if (!MlaShape::k3().is_exact_k3()) {
    throw std::runtime_error("exact K3 MLA shape contract changed");
  }
  const MlaWeights weights = make_weights(shape, device);
  ReferenceCache reference_cache;
  MlaCache candidate_cache(shape);
  MlaCache no_alias_cache(shape);

  at::Tensor retained_snapshot;
  at::Tensor retained_growth_view;
  at::Tensor retained_growth_bits;
  const void* retained_growth_pointer = nullptr;
  for (std::int64_t token = 0; token < 33; ++token) {
    const at::Tensor hidden =
        deterministic_tensor({1, 1, shape.hidden_size}, device, 100 + token,
                             0.03125F);
    const at::Tensor expected =
        reference_decode(hidden, weights, shape, reference_cache);
    MlaPreparedDecode candidate = deltafin::provider_internal::prepare_mla_decode(
        hidden, weights, candidate_cache, true);
    MlaPreparedDecode no_alias = deltafin::provider_internal::prepare_mla_decode(
        hidden, weights, no_alias_cache, false);
    require_close(candidate.output, expected, "expanded MLA decode");
    require_equal(candidate.output, no_alias.output,
                  "qualified T=1 query alias");
    deltafin::provider_internal::commit_mla_decode(candidate_cache, candidate);
    deltafin::provider_internal::commit_mla_decode(no_alias_cache, no_alias);
    require_close(candidate_cache.committed_keys(), reference_cache.keys,
                  "geometric key cache");
    require_close(candidate_cache.committed_values(), reference_cache.values,
                  "geometric value cache");
    if (token == 4) {
      retained_snapshot = candidate_cache.committed_keys().clone();
    }
    if (token > 4) {
      require_equal(candidate_cache.committed_keys().narrow(2, 0, 5),
                    retained_snapshot, "committed cache prefix");
    }
    if (token == 14) {
      retained_growth_view = candidate_cache.committed_keys();
      retained_growth_bits = retained_growth_view.clone();
      retained_growth_pointer = retained_growth_view.const_data_ptr();
    }
    if (token == 16) {
      require_equal(retained_growth_view, retained_growth_bits,
                    "pre-growth cache snapshot");
      if (candidate_cache.committed_keys().const_data_ptr() ==
          retained_growth_pointer) {
        throw std::runtime_error(
            "geometric MLA growth reused a retained cache allocation");
      }
    }
  }
  if (candidate_cache.length() != 33 || candidate_cache.version() != 33 ||
      candidate_cache.capacity() < 33 ||
      candidate_cache.capacity() >= shape.max_context) {
    throw std::runtime_error("geometric MLA cache accounting is invalid");
  }

  {
    const at::Tensor parent_keys = candidate_cache.committed_keys().clone();
    const at::Tensor parent_values = candidate_cache.committed_values().clone();
    auto child = candidate_cache.fork_committed();
    const at::Tensor branch_hidden = deterministic_tensor(
        {1, 1, shape.hidden_size}, device, 777, 0.03125F);
    MlaPreparedDecode branch =
        deltafin::provider_internal::prepare_mla_decode(
            branch_hidden, weights, *child, true);
    deltafin::provider_internal::commit_mla_decode(*child, branch);
    if (child->length() != 34 || child->version() != 34 ||
        candidate_cache.length() != 33 || candidate_cache.version() != 33) {
      throw std::runtime_error(
          "copy-on-write MLA branch changed its parent metadata");
    }
    require_equal(candidate_cache.committed_keys(), parent_keys,
                  "copy-on-write MLA parent keys");
    require_equal(candidate_cache.committed_values(), parent_values,
                  "copy-on-write MLA parent values");
    require_equal(child->committed_keys().narrow(2, 0, 33), parent_keys,
                  "copy-on-write MLA child key prefix");
    require_equal(child->committed_values().narrow(2, 0, 33), parent_values,
                  "copy-on-write MLA child value prefix");
  }

  const std::int64_t length_before_cancel = candidate_cache.length();
  const std::uint64_t version_before_cancel = candidate_cache.version();
  const at::Tensor key_before_cancel =
      candidate_cache.committed_keys().clone();
  const at::Tensor cancel_hidden = deterministic_tensor(
      {1, 1, shape.hidden_size}, device, 999, 0.03125F);
  MlaPreparedDecode cancelled = deltafin::provider_internal::prepare_mla_decode(
      cancel_hidden, weights, candidate_cache, true);
  if (!candidate_cache.has_pending_prepare()) {
    throw std::runtime_error("MLA prepare did not reserve its cache");
  }
  deltafin::provider_internal::cancel_mla_decode(candidate_cache, cancelled);
  if (candidate_cache.has_pending_prepare() ||
      candidate_cache.length() != length_before_cancel ||
      candidate_cache.version() != version_before_cancel) {
    throw std::runtime_error("MLA cancellation published staged cache state");
  }
  require_equal(candidate_cache.committed_keys(), key_before_cancel,
                "cancelled MLA cache");

  MlaWeights invalid = weights;
  invalid.output.data = at::empty({1, 1}, cancel_hidden.options());
  bool rejected = false;
  try {
    static_cast<void>(deltafin::provider_internal::prepare_mla_decode(
        cancel_hidden, invalid, candidate_cache, true));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected || candidate_cache.has_pending_prepare() ||
      candidate_cache.length() != length_before_cancel ||
      candidate_cache.version() != version_before_cancel) {
    throw std::runtime_error(
        "invalid MLA tape was not rejected without cache mutation");
  }

  rejected = false;
  try {
    static_cast<void>(deltafin::provider_internal::prepare_k3_mla_decode(
        cancel_hidden, weights, candidate_cache));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected || candidate_cache.has_pending_prepare()) {
    throw std::runtime_error(
        "production MLA entry accepted a non-K3 test contract");
  }
}

void run_compact_sequence_and_transactions(const at::Device& device) {
  const c10::InferenceMode inference_guard;
  const MlaShape shape = test_shape();
  const MlaWeights weights = make_weights(shape, device);
  const MlaAbsorbedKeyValue absorbed =
      deltafin::provider_internal::absorb_mla_key_value(
          shape, weights.key_value_b);
  const at::Tensor hidden = deterministic_tensor(
      {1, 9, shape.hidden_size}, device, 1600, 0.03125F);

  MlaCache missing_descriptor(
      shape, MlaCacheRepresentation::CompactLatentF32);
  require_throws(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::prepare_mla_positions(
                hidden.narrow(1, 0, 1), weights, missing_descriptor,
                nullptr, true));
      },
      "compact MLA without absorbed descriptor");
  if (missing_descriptor.has_pending_prepare() ||
      missing_descriptor.length() != 0) {
    throw std::runtime_error(
        "failed compact MLA admission mutated its cache");
  }
  MlaCache expanded_with_descriptor(shape);
  require_throws(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::prepare_mla_positions(
                hidden.narrow(1, 0, 1), weights,
                expanded_with_descriptor, &absorbed, true));
      },
      "absorbed descriptor on expanded MLA cache");
  require_throws(
      [&] {
        static_cast<void>(deltafin::provider_internal::prepare_mla_decode(
            hidden.narrow(1, 0, 1), weights, missing_descriptor, true));
      },
      "legacy expanded-only decode on compact MLA cache");

  MlaCache sequential(shape);
  std::vector<at::Tensor> expected_rows;
  expected_rows.reserve(9);
  for (std::int64_t row = 0; row < hidden.size(1); ++row) {
    MlaPreparedDecode prepared =
        deltafin::provider_internal::prepare_mla_decode(
            hidden.narrow(1, row, 1), weights, sequential, true);
    expected_rows.push_back(prepared.output);
    deltafin::provider_internal::commit_mla_decode(sequential, prepared);
  }
  const at::Tensor expected = at::cat(expected_rows, 1);

  MlaCache expanded_bulk(shape);
  MlaPreparedDecode expanded =
      deltafin::provider_internal::prepare_mla_positions(
          hidden, weights, expanded_bulk, nullptr, true);
  require_close(expanded.output, expected,
                "multi-position expanded causal MLA");
  deltafin::provider_internal::commit_mla_decode(expanded_bulk, expanded);

  MlaCache compact(shape, MlaCacheRepresentation::CompactLatentF32);
  MlaPreparedDecode compact_prepared =
      deltafin::provider_internal::prepare_mla_positions(
          hidden, weights, compact, &absorbed, true);
  require_close(compact_prepared.output, expected,
                "multi-position compact causal MLA");
  deltafin::provider_internal::commit_mla_decode(compact, compact_prepared);

  const at::Tensor compressed =
      reference_linear(hidden, weights.key_value_a);
  const at::Tensor raw_latent =
      compressed.narrow(-1, 0, shape.kv_lora_rank);
  const at::Tensor expected_latent =
      reference_norm(raw_latent, weights.key_value_a_norm,
                     shape.rms_epsilon)
          .unsqueeze(1);
  const at::Tensor expected_position =
      compressed
          .narrow(-1, shape.kv_lora_rank, shape.qk_rope_head_dim)
          .unsqueeze(1);
  require_equal(compact.committed_keys(), expected_latent,
                "compact normalized latent storage");
  require_equal(compact.committed_values(), expected_position,
                "compact positional storage");
  if (compact.representation() !=
          MlaCacheRepresentation::CompactLatentF32 ||
      compact.length() != 9 || compact.version() != 9 ||
      compact.bytes_per_position() * 8 !=
          expanded_bulk.bytes_per_position()) {
    throw std::runtime_error(
        "compact MLA representation/accounting changed");
  }

  // A changed future suffix must not influence earlier rows in a single
  // multi-position provider call. This catches an accidentally unmasked
  // verification batch even when expanded/compact algebra still agrees.
  at::Tensor changed_future = hidden.clone();
  changed_future.narrow(1, 4, 5).add_(17.0);
  MlaCache causal_left(shape, MlaCacheRepresentation::CompactLatentF32);
  MlaCache causal_right(shape, MlaCacheRepresentation::CompactLatentF32);
  MlaPreparedDecode left =
      deltafin::provider_internal::prepare_mla_positions(
          hidden, weights, causal_left, &absorbed, true);
  MlaPreparedDecode right =
      deltafin::provider_internal::prepare_mla_positions(
          changed_future, weights, causal_right, &absorbed, true);
  require_close(left.output.narrow(1, 0, 4),
                right.output.narrow(1, 0, 4),
                "compact MLA future-token isolation");
  deltafin::provider_internal::cancel_mla_decode(causal_left, left);
  deltafin::provider_internal::cancel_mla_decode(causal_right, right);

  // Work on an unpublished branch, publish only three of five completed
  // positions, and prove the parent prefix/metadata never changed early.
  const at::Tensor parent_keys = compact.committed_keys().clone();
  const at::Tensor parent_values = compact.committed_values().clone();
  const std::int64_t parent_length = compact.length();
  const std::uint64_t parent_version = compact.version();
  MlaCacheTransaction transaction(compact, 5);
  const at::Tensor branch_hidden = deterministic_tensor(
      {1, 5, shape.hidden_size}, device, 1700, 0.03125F);
  MlaPreparedDecode branch =
      deltafin::provider_internal::prepare_mla_positions(
          branch_hidden, weights, transaction.working_cache(), &absorbed,
          true);
  deltafin::provider_internal::commit_mla_decode(
      transaction.working_cache(), branch);
  if (!transaction.active() || transaction.completed_positions() != 5 ||
      compact.length() != parent_length ||
      compact.version() != parent_version) {
    throw std::runtime_error(
        "compact MLA transaction published before prefix commit");
  }
  require_equal(compact.committed_keys(), parent_keys,
                "unpublished compact parent latent");
  require_equal(compact.committed_values(), parent_values,
                "unpublished compact parent positional");
  const at::Tensor publish_keys =
      transaction.working_cache().committed_keys().narrow(
          2, 0, parent_length + 3).clone();
  const at::Tensor publish_values =
      transaction.working_cache().committed_values().narrow(
          2, 0, parent_length + 3).clone();
  transaction.preflight_publish_prefix(3);
  transaction.publish_prefix_noexcept(3);
  if (transaction.active() || compact.length() != parent_length + 3 ||
      compact.version() != parent_version + 3) {
    throw std::runtime_error(
        "compact MLA partial-prefix publication metadata is invalid");
  }
  require_equal(compact.committed_keys(), publish_keys,
                "published compact latent prefix");
  require_equal(compact.committed_values(), publish_values,
                "published compact positional prefix");

  // Cancellation keeps the representation, prefix bits, and version intact.
  const at::Tensor cancel_keys = compact.committed_keys().clone();
  const at::Tensor cancel_values = compact.committed_values().clone();
  const std::int64_t cancel_length = compact.length();
  const std::uint64_t cancel_version = compact.version();
  MlaPreparedDecode cancelled =
      deltafin::provider_internal::prepare_mla_positions(
          branch_hidden.narrow(1, 0, 2), weights, compact, &absorbed, true);
  deltafin::provider_internal::cancel_mla_decode(compact, cancelled);
  if (compact.length() != cancel_length ||
      compact.version() != cancel_version ||
      compact.representation() !=
          MlaCacheRepresentation::CompactLatentF32) {
    throw std::runtime_error(
        "compact MLA cancel changed committed metadata/representation");
  }
  require_equal(compact.committed_keys(), cancel_keys,
                "cancelled compact latent prefix");
  require_equal(compact.committed_values(), cancel_values,
                "cancelled compact positional prefix");

  // A branch whose base advances cannot publish, and an explicit transaction
  // cancel permanently drops its authority to publish.
  MlaCacheTransaction stale(compact, 2);
  MlaPreparedDecode stale_branch =
      deltafin::provider_internal::prepare_mla_positions(
          branch_hidden.narrow(1, 0, 2), weights, stale.working_cache(),
          &absorbed, true);
  deltafin::provider_internal::commit_mla_decode(stale.working_cache(),
                                                 stale_branch);
  MlaPreparedDecode base_advance =
      deltafin::provider_internal::prepare_mla_positions(
          branch_hidden.narrow(1, 4, 1), weights, compact, &absorbed, true);
  deltafin::provider_internal::commit_mla_decode(compact, base_advance);
  require_throws([&] { stale.preflight_publish_prefix(1); },
                 "stale compact MLA transaction");
  stale.cancel_noexcept();
  if (stale.active()) {
    throw std::runtime_error("cancelled compact transaction stayed active");
  }
  require_throws([&] { stale.preflight_publish_prefix(0); },
                 "cancelled compact MLA transaction publication");

  auto fork = compact.fork_committed();
  if (fork->representation() != compact.representation() ||
      fork->bytes_per_position() != compact.bytes_per_position()) {
    throw std::runtime_error(
        "compact MLA fork lost its immutable representation tag");
  }
}

CompactReassociationFalsifier run_compact_reassociation_falsifier(
    const at::Device& device) {
  const c10::InferenceMode inference_guard;
  const MlaShape shape = test_shape();
  const MlaWeights weights = make_weights(shape, device);
  const MlaAbsorbedKeyValue absorbed =
      deltafin::provider_internal::absorb_mla_key_value(
          shape, weights.key_value_b);
  MlaCache expanded(shape);
  MlaCache compact(shape, MlaCacheRepresentation::CompactLatentF32);
  CompactReassociationFalsifier observation;
  for (std::int64_t position = 0; position < 33; ++position) {
    const at::Tensor hidden = deterministic_tensor(
        {1, 1, shape.hidden_size}, device, 8100 + position, 0.03125F);
    MlaPreparedDecode expected =
        deltafin::provider_internal::prepare_mla_decode(
            hidden, weights, expanded, true);
    MlaPreparedDecode candidate =
        deltafin::provider_internal::prepare_mla_positions(
            hidden, weights, compact, &absorbed, true);
    require_close(candidate.output, expected.output,
                  "compact MLA reassociation tolerance");
    observe_fp32_bits(candidate.output, expected.output, observation);
    deltafin::provider_internal::commit_mla_decode(expanded, expected);
    deltafin::provider_internal::commit_mla_decode(compact, candidate);
  }
  if (observation.differing_elements == 0 ||
      !(observation.maximum_absolute_difference > 0.0) ||
      !std::isfinite(observation.maximum_absolute_difference)) {
    throw std::runtime_error(
        "compact MLA reassociation falsifier did not observe the expected "
        "non-bit-exact fp32 result");
  }
  return observation;
}

void run_k3_cache_geometry() {
  const MlaShape shape = MlaShape::k3();
  MlaCache expanded(shape);
  MlaCache compact(shape, MlaCacheRepresentation::CompactLatentF32);
  if (expanded.representation() != MlaCacheRepresentation::ExpandedExact ||
      expanded.bytes_per_position() != 122880 ||
      expanded.admitted_max_context() != 4369 ||
      expanded.full_context_storage_bytes() != 128849018880ULL ||
      compact.representation() !=
          MlaCacheRepresentation::CompactLatentF32 ||
      compact.bytes_per_position() != 2304 ||
      compact.full_context_storage_bytes() != 2415919104ULL ||
      compact.storage_budget_bytes() != 2415919104ULL ||
      compact.admitted_max_context() != 1048576 ||
      !compact.can_append(1048576) || compact.can_append(1048577)) {
    throw std::runtime_error(
        "exact K3 compact/expanded 1M-context accounting changed");
  }
  if (compact.full_context_storage_bytes() * 24ULL != 57982058496ULL) {
    throw std::runtime_error(
        "24-layer K3 compact MLA full-context accounting changed");
  }

  MlaWeights sentinel;
  for (MlaLinearWeight* weight :
       {&sentinel.query_a, &sentinel.query_b, &sentinel.key_value_a,
        &sentinel.key_value_b, &sentinel.output_gate, &sentinel.output}) {
    weight->encoding = MlaLinearEncoding::DenseF32;
  }
  require_throws(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::prepare_k3_mla_positions(
                at::Tensor(), sentinel, compact, nullptr));
      },
      "production compact MLA qualification gate");
  require_throws(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::prepare_mla_positions(
                at::Tensor(), sentinel, compact, nullptr));
      },
      "low-level exact-K3 compact MLA qualification gate");
}

void synchronize_device(const at::Device& device) {
  if (device.type() == at::kMPS) {
#if defined(__APPLE__)
    torch::mps::synchronize();
#else
    throw std::runtime_error("MPS synchronization is unavailable");
#endif
  }
}

template <typename Function>
double median_milliseconds(Function&& function, const at::Device& device) {
  constexpr std::int64_t warmups = 5;
  constexpr std::int64_t iterations = 20;
  constexpr std::int64_t samples = 7;
  for (std::int64_t index = 0; index < warmups; ++index) {
    function();
  }
  synchronize_device(device);
  std::vector<double> values;
  values.reserve(samples);
  for (std::int64_t sample = 0; sample < samples; ++sample) {
    const auto started = std::chrono::steady_clock::now();
    for (std::int64_t index = 0; index < iterations; ++index) {
      function();
    }
    synchronize_device(device);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started);
    values.push_back(elapsed.count() / static_cast<double>(iterations));
  }
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void run_input_bundle_benchmark(const at::Device& device) {
  constexpr std::int64_t hidden_columns = 7168;
  constexpr std::int64_t query_rows = 1536;
  constexpr std::int64_t key_value_rows = 576;
  constexpr std::int64_t gate_rows = 12288;
  const MlaShape shape = MlaShape::k3();
  MlaWeights weights;
  weights.query_a = packed_weight(query_rows, hidden_columns, device, 41);
  weights.key_value_a =
      packed_weight(key_value_rows, hidden_columns, device, 42);
  weights.output_gate =
      packed_weight(gate_rows, hidden_columns, device, 43);
  const std::uint64_t component_payload_bytes =
      static_cast<std::uint64_t>(
          weights.query_a.data.numel() + weights.key_value_a.data.numel() +
          weights.output_gate.data.numel()) +
      static_cast<std::uint64_t>(
          weights.query_a.row_scale.numel() +
          weights.key_value_a.row_scale.numel() +
          weights.output_gate.row_scale.numel()) *
          sizeof(float);
  const MlaInputBundle bundle =
      deltafin::provider_internal::bundle_mla_input_weights(shape, weights);
  const std::uint64_t bundle_payload_bytes =
      static_cast<std::uint64_t>(bundle.projection.data.numel()) +
      static_cast<std::uint64_t>(bundle.projection.row_scale.numel()) *
          sizeof(float);
  if (bundle_payload_bytes != component_payload_bytes) {
    throw std::runtime_error(
        "real-shape MLA bundle changed steady payload residency");
  }
  const at::Tensor hidden = deterministic_tensor(
      {1, hidden_columns}, device, 1300, 0.03125F);
  std::array<at::Tensor, 3> separate;
  at::Tensor combined;
  const auto run_separate = [&] {
    separate[0] = at::_weight_int8pack_mm(
        hidden, weights.query_a.data, weights.query_a.row_scale);
    separate[1] = at::_weight_int8pack_mm(
        hidden, weights.key_value_a.data,
        weights.key_value_a.row_scale);
    separate[2] = at::_weight_int8pack_mm(
        hidden, weights.output_gate.data,
        weights.output_gate.row_scale);
  };
  const auto run_bundle = [&] {
    combined = at::_weight_int8pack_mm(
        hidden, bundle.projection.data, bundle.projection.row_scale);
  };
  run_separate();
  run_bundle();
  synchronize_device(device);
  require_equal(combined, at::cat({separate[0], separate[1], separate[2]}, 1),
                "real-shape same-input bundle split");

  // Alternate the measurement order to make thermal drift affect both arms.
  const double separate_first = median_milliseconds(run_separate, device);
  const double bundle_second = median_milliseconds(run_bundle, device);
  const double bundle_first = median_milliseconds(run_bundle, device);
  const double separate_second = median_milliseconds(run_separate, device);
  const double separate_ms = (separate_first + separate_second) * 0.5;
  const double bundle_ms = (bundle_first + bundle_second) * 0.5;
  std::cout << "benchmark.mla_input_separate_ms=" << separate_ms << '\n'
            << "benchmark.mla_input_bundle_ms=" << bundle_ms << '\n'
            << "benchmark.mla_input_speedup=" << separate_ms / bundle_ms
            << "x\n"
            << "benchmark.mla_input_component_bytes="
            << component_payload_bytes << '\n'
            << "benchmark.mla_input_bundle_bytes=" << bundle_payload_bytes
            << '\n'
            << "benchmark.mla_input_resident_delta_bytes="
            << static_cast<std::int64_t>(bundle_payload_bytes) -
                   static_cast<std::int64_t>(component_payload_bytes)
            << '\n';
}

Options parse_options(const int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--benchmark-input-bundle") {
      options.benchmark_input_bundle = true;
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
            "MLA test device must be cpu, mps, or cuda");
      }
      continue;
    }
    throw std::invalid_argument(
        "usage: deltafin-provider-mla-test [--device cpu|mps|cuda] "
        "[--benchmark-input-bundle]");
  }
  return options;
}

}  // namespace

int main(const int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (deltafin::provider_internal::cuda_case_should_skip(options.device)) {
      std::cout << "check.mla_decode=PASS\n"
                << "check.mla_decode.cuda=skipped(no visible CUDA device)\n";
      return 0;
    }
    const auto selected = deltafin::provider_internal::select_device(
        options.device, 0);
    run_k3_cache_geometry();
    run_parity(selected.device);
    run_compact_sequence_and_transactions(selected.device);
    const CompactReassociationFalsifier falsifier =
        run_compact_reassociation_falsifier(selected.device);
    run_packed_parity(selected.device);
    run_live_shape_schedule(selected.device);
    if (options.benchmark_input_bundle) {
      run_input_bundle_benchmark(selected.device);
    }
    std::cout << "check.mla_decode=PASS\n"
              << "check.mla_row_int8=PASS\n"
              << "check.mla_query_alias=PASS (bit-exact)\n"
              << "check.mla_geometric_cache=PASS\n"
              << "check.mla_transaction=PASS\n"
              << "check.mla_compact_fp32_research=PASS (tolerance-only)\n"
              << "qualification.mla_compact_production=REFUSED "
                 "(fp32 reassociation is not bit-exact)\n"
              << "falsifier.mla_compact_differing_elements="
              << falsifier.differing_elements << '\n'
              << "falsifier.mla_compact_max_abs="
              << falsifier.maximum_absolute_difference << '\n'
              << "check.mla_multi_position_causal=PASS\n"
              << "check.mla_live_shape_schedule=PASS (T=1..9)\n"
              << "check.mla_k3_1m_accounting=PASS\n"
              << "device=" << selected.device.str() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "result=FAIL\nerror=\"" << error.what() << "\"\n";
    return 1;
  }
}
