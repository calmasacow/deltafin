#include "provider_target.h"

#include "provider_device.h"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/ops/_weight_int8pack_mm.h>
#include <ATen/ops/add.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/rms_norm.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/tanh.h>

#if defined(__APPLE__)
#include <torch/mps.h>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using deltafin::provider_internal::MoeRowInt8Matrix;
using deltafin::provider_internal::Bf16CpuT1Kernel;
using deltafin::provider_internal::TargetBlockResidual;
using deltafin::provider_internal::TargetDenseWeights;
using deltafin::provider_internal::TargetResidualWeights;
using deltafin::provider_internal::TargetTailWeights;

constexpr float kEpsilon = 1.0e-5F;

at::Tensor vector(const std::vector<float> &values) {
  return at::from_blob(const_cast<float *>(values.data()),
                       {static_cast<std::int64_t>(values.size())},
                       at::TensorOptions().dtype(at::kFloat))
      .clone();
}

at::Tensor matrix(const std::vector<float> &values, const std::int64_t rows,
                  const std::int64_t columns) {
  return vector(values).reshape({rows, columns}).contiguous();
}

std::uint16_t float_to_bf16(const float value) {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

MoeRowInt8Matrix original_bf16_weight(const at::Tensor &dense,
                                      Bf16CpuT1Kernel &kernel) {
  if (!dense.defined() || dense.device() != at::Device(at::kCPU) ||
      dense.scalar_type() != at::kFloat || !dense.is_contiguous() ||
      dense.dim() != 2) {
    throw std::invalid_argument(
        "test original-BF16 source must be contiguous CPU fp32");
  }
  std::vector<std::uint16_t> bits(
      static_cast<std::size_t>(dense.numel()));
  const float *source = dense.const_data_ptr<float>();
  for (std::size_t index = 0; index < bits.size(); ++index) {
    bits[index] = float_to_bf16(source[index]);
  }
  at::Tensor storage = at::empty(
      {dense.numel()},
      at::TensorOptions().dtype(at::kUInt16).device(at::kCPU));
  std::memcpy(storage.mutable_data_ptr<std::uint16_t>(), bits.data(),
              bits.size() * sizeof(std::uint16_t));
  auto shared =
      deltafin::provider_internal::make_exact_bf16_storage(std::move(storage));
  return MoeRowInt8Matrix{
      at::Tensor(), at::Tensor(), at::Tensor(),
      deltafin::provider_internal::make_owned_original_bf16_cpu(
          std::move(shared), 0, static_cast<std::size_t>(dense.size(0)),
          static_cast<std::size_t>(dense.size(1)), &kernel)};
}

MoeRowInt8Matrix linear_weight(const at::Tensor &dense) {
  const std::int64_t rows = dense.size(0);
  const std::int64_t columns = dense.size(1);
  // The fallback is authoritative in this small CPU test; q/sc merely satisfy
  // the same structural contract production binding supplies.
  return MoeRowInt8Matrix{
      at::zeros({rows, columns}, dense.options().dtype(at::kChar)),
      at::ones({rows}, dense.options().dtype(at::kFloat)), dense.contiguous(),
      {}};
}

MoeRowInt8Matrix packed_weight(const at::Tensor &quantized,
                               const at::Tensor &scales,
                               const at::Device &device) {
  return MoeRowInt8Matrix{quantized.to(device).contiguous(),
                          scales.to(device).contiguous(), at::Tensor(), {}};
}

void synchronize_device(const at::Device &device) {
  if (device.type() == at::kMPS) {
#if defined(__APPLE__)
    torch::mps::synchronize();
#else
    throw std::runtime_error("MPS synchronization is unavailable");
#endif
  }
}

struct PairedTiming {
  double control_ms = 0.0;
  double candidate_ms = 0.0;
  double median_speedup = 0.0;
  std::int64_t candidate_wins = 0;
  std::int64_t samples = 0;
};

template <typename Control, typename Candidate>
PairedTiming paired_milliseconds(Control &&control, Candidate &&candidate,
                                 const at::Device &device,
                                 const std::int64_t iterations) {
  constexpr std::int64_t warmups = 8;
  constexpr std::int64_t samples = 21;
  for (std::int64_t index = 0; index < warmups; ++index) {
    control();
    candidate();
  }
  synchronize_device(device);
  std::vector<double> control_values;
  std::vector<double> candidate_values;
  std::vector<double> ratios;
  control_values.reserve(samples);
  candidate_values.reserve(samples);
  ratios.reserve(samples);
  const auto measure = [&](auto &&operation) {
    const auto started = std::chrono::steady_clock::now();
    for (std::int64_t index = 0; index < iterations; ++index) {
      operation();
    }
    synchronize_device(device);
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - started)
               .count() /
           static_cast<double>(iterations);
  };
  std::int64_t wins = 0;
  for (std::int64_t sample = 0; sample < samples; ++sample) {
    double control_ms = 0.0;
    double candidate_ms = 0.0;
    if (sample % 2 == 0) {
      control_ms = measure(control);
      candidate_ms = measure(candidate);
    } else {
      candidate_ms = measure(candidate);
      control_ms = measure(control);
    }
    control_values.push_back(control_ms);
    candidate_values.push_back(candidate_ms);
    ratios.push_back(control_ms / candidate_ms);
    wins += candidate_ms < control_ms ? 1 : 0;
  }
  std::sort(control_values.begin(), control_values.end());
  std::sort(candidate_values.begin(), candidate_values.end());
  std::sort(ratios.begin(), ratios.end());
  return PairedTiming{control_values[control_values.size() / 2],
                      candidate_values[candidate_values.size() / 2],
                      ratios[ratios.size() / 2], wins, samples};
}

at::Tensor rms_reference(const at::Tensor &input, const at::Tensor &weight) {
  const at::Tensor variance =
      at::mean(at::pow(input, 2), std::vector<std::int64_t>{-1}, true);
  return at::mul(weight,
                 at::mul(input, at::rsqrt(at::add(variance, kEpsilon))));
}

at::Tensor residual_reference(const at::Tensor &prefix,
                              const at::Tensor &anchors,
                              const at::Tensor &projection,
                              const at::Tensor &norm) {
  const at::Tensor values = at::cat({anchors, prefix.unsqueeze(1)}, 1);
  const at::Tensor variance =
      at::mean(at::pow(values, 2), std::vector<std::int64_t>{-1}, true);
  const at::Tensor keys =
      at::mul(values, at::rsqrt(at::add(variance, kEpsilon)));
  const at::Tensor scores =
      at::sum(at::mul(keys, at::mul(norm, projection.squeeze(0))),
              std::vector<std::int64_t>{-1}, false);
  const at::Tensor probabilities =
      at::softmax(scores, -1, at::kFloat).unsqueeze(1);
  return at::matmul(probabilities, values).squeeze(1);
}

at::Tensor situ_reference(const at::Tensor &gate, const at::Tensor &up) {
  // Match KimiMLP/SituAndMul's public formulation: concatenate both linear
  // results, split the final dimension, then evaluate in fp32.
  const at::Tensor gate_up = at::cat({gate, up}, -1);
  const std::int64_t half = gate_up.size(-1) / 2;
  const at::Tensor gate_half = gate_up.slice(-1, 0, half).to(at::kFloat);
  const at::Tensor up_half = gate_up.slice(-1, half).to(at::kFloat);
  return at::mul(at::mul(at::mul(at::tanh(gate_half / 4.0F), 4.0F),
                         at::sigmoid(gate_half)),
                 at::mul(at::tanh(up_half / 25.0F), 25.0F));
}

void require_equal(const at::Tensor &actual, const at::Tensor &expected,
                   const char *name) {
  if (!at::equal(actual, expected)) {
    const double error = at::max(at::abs(actual - expected)).item<double>();
    throw std::runtime_error(std::string(name) +
                             " max_abs=" + std::to_string(error));
  }
}

void require_throws(const std::function<void()> &operation, const char *name) {
  try {
    operation();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error(std::string(name) + " did not fail closed");
}

at::Tensor bf16_expanded_linspace(const std::vector<std::int64_t> &shape,
                                  const float begin, const float end,
                                  const at::Device &device) {
  std::int64_t count = 1;
  for (const std::int64_t extent : shape) {
    count *= extent;
  }
  return at::linspace(begin, end, count, at::TensorOptions().dtype(at::kFloat))
      .to(at::kBFloat16)
      .to(at::kFloat)
      .reshape(shape)
      .to(device)
      .contiguous();
}

double max_abs_error(const at::Tensor &left, const at::Tensor &right) {
  return at::max(at::abs(left - right)).item<double>();
}

TargetResidualWeights residual_weights() {
  return TargetResidualWeights{vector({0.75F, 1.0F, 1.25F, 1.5F}),
                               vector({1.1F, 0.9F, 1.2F, 0.8F}),
                               matrix({0.3F, -0.2F, 0.1F, 0.4F}, 1, 4),
                               vector({1.0F, 1.1F, 0.9F, 1.2F}),
                               vector({0.8F, 1.2F, 1.1F, 0.7F}),
                               matrix({-0.1F, 0.2F, 0.35F, -0.25F}, 1, 4)};
}

void residual_tape_test() {
  const at::Tensor hidden = matrix({0.2F, -0.4F, 0.8F, 0.5F}, 1, 4);
  const at::Tensor anchor =
      matrix({-0.3F, 0.7F, 0.1F, -0.2F}, 1, 4).unsqueeze(1).contiguous();
  const TargetBlockResidual residual{anchor};
  const TargetResidualWeights weights = residual_weights();
  const TargetResidualWeights prepared_weights =
      deltafin::provider_internal::precompute_target_residual_score_weights(
          weights);
  require_equal(prepared_weights.self_attention_score_weight,
                weights.self_attention_res_norm *
                    weights.self_attention_res_projection.squeeze(0),
                "prepared attention score weight");
  require_equal(prepared_weights.mlp_score_weight,
                weights.mlp_res_norm * weights.mlp_res_projection.squeeze(0),
                "prepared MLP score weight");

  const auto prepared = deltafin::provider_internal::prepare_target_attention(
      hidden, residual, weights, 5, false);
  const auto bind_prepared =
      deltafin::provider_internal::prepare_target_attention(
          hidden, residual, prepared_weights, 5, false);
  require_equal(bind_prepared.normalized, prepared.normalized,
                "bind-time attention score product");
  const at::Tensor expected_mixed =
      residual_reference(hidden, anchor, weights.self_attention_res_projection,
                         weights.self_attention_res_norm);
  require_equal(prepared.normalized,
                rms_reference(expected_mixed, weights.input_norm),
                "attention preparation");
  require_equal(prepared.next_anchors, anchor, "non-boundary anchors");

  const at::Tensor attention = matrix({0.05F, 0.1F, -0.2F, 0.15F}, 1, 4);
  const auto mlp = deltafin::provider_internal::prepare_target_mlp(
      prepared, attention, weights, false);
  const auto bind_prepared_mlp =
      deltafin::provider_internal::prepare_target_mlp(bind_prepared, attention,
                                                      prepared_weights, false);
  require_equal(bind_prepared_mlp.normalized, mlp.normalized,
                "bind-time MLP score product");
  const at::Tensor prefix = hidden + attention;
  const at::Tensor expected_post = residual_reference(
      prefix, anchor, weights.mlp_res_projection, weights.mlp_res_norm);
  require_equal(mlp.lookahead_source, expected_post,
                "exact MLP lookahead source");
  require_equal(bind_prepared_mlp.lookahead_source, mlp.lookahead_source,
                "bind-time lookahead source");
  require_equal(mlp.normalized,
                rms_reference(expected_post, weights.post_attention_norm),
                "MLP preparation");
  const at::Tensor mlp_output = matrix({-0.02F, 0.03F, 0.04F, -0.01F}, 1, 4);
  require_equal(deltafin::provider_internal::complete_target_layer(
                    mlp, mlp_output, false),
                prefix + mlp_output, "layer completion");

  const auto boundary = deltafin::provider_internal::prepare_target_attention(
      hidden, residual, weights, 12, false);
  if (boundary.prefix_sum.defined() || boundary.next_anchors.size(1) != 2) {
    throw std::runtime_error("residual block boundary was not transactional");
  }
  // The caller's committed state remains unchanged until the whole layer
  // succeeds and publishes boundary.next_anchors.
  if (residual.anchors.size(1) != 1) {
    throw std::runtime_error("residual preparation mutated committed anchors");
  }
}

void residual_rows_live_shape_test() {
  constexpr std::array<std::int64_t, 8> kPositions{2, 3, 4, 5, 6, 7, 8, 9};
  constexpr std::int64_t kWidth = 4;
  const TargetResidualWeights weights = residual_weights();
  for (const std::int64_t positions : kPositions) {
    const at::Tensor hidden = at::linspace(
        -0.625F, 0.875F, positions * kWidth,
        at::TensorOptions().dtype(at::kFloat))
                                  .reshape({positions, kWidth})
                                  .contiguous();
    const at::Tensor anchors = at::linspace(
        -0.375F, 0.5F, positions * 2 * kWidth,
        at::TensorOptions().dtype(at::kFloat))
                                   .reshape({positions, 2, kWidth})
                                   .contiguous();
    const at::Tensor attention_output = at::linspace(
        0.1875F, -0.3125F, positions * kWidth,
        at::TensorOptions().dtype(at::kFloat))
                                             .reshape({positions, kWidth})
                                             .contiguous();
    const at::Tensor mlp_output = at::linspace(
        -0.09375F, 0.15625F, positions * kWidth,
        at::TensorOptions().dtype(at::kFloat))
                                      .reshape({positions, kWidth})
                                      .contiguous();

    // A non-boundary layer must evaluate the residual softmax and RMS once
    // over the authoritative [T,A,H] / [T,H] shapes.
    const auto attention =
        deltafin::provider_internal::prepare_target_attention_rows(
            hidden, anchors, weights, 5, false);
    const at::Tensor mixed_attention = residual_reference(
        hidden, anchors, weights.self_attention_res_projection,
        weights.self_attention_res_norm);
    require_equal(attention.normalized,
                  rms_reference(mixed_attention, weights.input_norm),
                  "T-wide attention preparation");
    require_equal(attention.prefix_sum, hidden,
                  "T-wide non-boundary prefix");
    require_equal(attention.next_anchors, anchors,
                  "T-wide non-boundary anchors");

    const auto mlp = deltafin::provider_internal::prepare_target_mlp_rows(
        attention, attention_output, weights, false);
    const at::Tensor prefix = hidden + attention_output;
    const at::Tensor mixed_mlp = residual_reference(
        prefix, anchors, weights.mlp_res_projection, weights.mlp_res_norm);
    require_equal(mlp.lookahead_source, mixed_mlp,
                  "T-wide MLP lookahead source");
    require_equal(mlp.normalized,
                  rms_reference(mixed_mlp, weights.post_attention_norm),
                  "T-wide MLP preparation");
    require_equal(
        deltafin::provider_internal::complete_target_layer_rows(
            mlp, mlp_output, false),
        prefix + mlp_output, "T-wide layer completion");

    // A block boundary appends every row's authoritative hidden state before
    // the attention projection and starts the new prefix at attention output.
    const auto boundary =
        deltafin::provider_internal::prepare_target_attention_rows(
            hidden, anchors, weights, 12, false);
    const at::Tensor boundary_anchors =
        at::cat({anchors, hidden.unsqueeze(1)}, 1).contiguous();
    if (boundary.prefix_sum.defined()) {
      throw std::runtime_error(
          "T-wide residual boundary retained an obsolete prefix");
    }
    require_equal(boundary.next_anchors, boundary_anchors,
                  "T-wide boundary anchors");
    const auto boundary_mlp =
        deltafin::provider_internal::prepare_target_mlp_rows(
            boundary, attention_output, weights, false);
    const at::Tensor boundary_mixed = residual_reference(
        attention_output, boundary_anchors, weights.mlp_res_projection,
        weights.mlp_res_norm);
    require_equal(boundary_mlp.lookahead_source, boundary_mixed,
                  "T-wide boundary lookahead source");
    require_equal(boundary_mlp.normalized,
                  rms_reference(boundary_mixed,
                                weights.post_attention_norm),
                  "T-wide boundary MLP preparation");
    require_equal(
        deltafin::provider_internal::complete_target_layer_rows(
            boundary_mlp, mlp_output, false),
        attention_output + mlp_output,
        "T-wide boundary layer completion");
  }
}

void dense_and_tail_test() {
  const at::Tensor hidden = matrix({0.15F, -0.25F, 0.35F, 0.45F}, 1, 4);
  const at::Tensor gate =
      matrix({0.1F,  0.2F,  -0.1F, 0.3F,  -0.2F, 0.4F,  0.25F,  -0.15F,
              0.3F,  -0.1F, 0.2F,  0.05F, 0.4F,  0.15F, -0.25F, 0.1F,
              -0.3F, 0.2F,  0.1F,  0.35F, 0.2F,  -0.4F, 0.3F,   0.1F},
             6, 4);
  const at::Tensor up = gate.flip({0}).contiguous();
  const at::Tensor down =
      matrix({0.2F,   -0.1F, 0.3F, 0.1F,   -0.2F, 0.4F, -0.3F, 0.25F,
              0.15F,  -0.1F, 0.2F, 0.05F,  0.1F,  0.3F, -0.2F, 0.35F,
              -0.15F, 0.2F,  0.4F, -0.25F, 0.05F, 0.2F, 0.1F,  -0.3F},
             4, 6);
  const TargetDenseWeights dense{linear_weight(gate), linear_weight(up),
                                 linear_weight(down), false};
  const at::Tensor expected_dense =
      at::matmul(situ_reference(at::matmul(hidden, gate.transpose(0, 1)),
                                at::matmul(hidden, up.transpose(0, 1))),
                 down.transpose(0, 1));
  require_equal(
      deltafin::provider_internal::run_target_dense(hidden, dense, false),
      expected_dense, "dense SiTU tape");
  const at::Tensor hidden_rows =
      at::cat({hidden, hidden * 0.5F, hidden * -1.25F}, 0).contiguous();
  const at::Tensor expected_dense_rows = at::matmul(
      situ_reference(at::matmul(hidden_rows, gate.transpose(0, 1)),
                     at::matmul(hidden_rows, up.transpose(0, 1))),
      down.transpose(0, 1));
  require_equal(deltafin::provider_internal::run_target_dense_rows(
                    hidden_rows, dense, false),
                expected_dense_rows, "T-wide dense SiTU tape");
  require_equal(deltafin::provider_internal::run_target_dense_rows(
                    hidden, dense, false),
                deltafin::provider_internal::run_target_dense(
                    hidden, dense, false),
                "one-row dense wrapper");

  const at::Tensor anchors =
      at::cat({matrix({0.2F, 0.1F, -0.3F, 0.5F}, 1, 4).unsqueeze(1),
               matrix({-0.1F, 0.4F, 0.25F, -0.2F}, 1, 4).unsqueeze(1)},
              1);
  const at::Tensor output_norm = vector({1.0F, 0.9F, 1.1F, 1.2F});
  const at::Tensor output_proj = matrix({0.2F, -0.3F, 0.1F, 0.4F}, 1, 4);
  const at::Tensor final_norm = vector({0.8F, 1.0F, 1.2F, 0.7F});
  const at::Tensor head = matrix(
      {0.1F, 0.2F,  0.3F, 0.4F,  -0.2F,  0.1F,  0.25F, 0.35F, 0.4F, -0.1F,
       0.2F, -0.3F, 0.3F, 0.15F, -0.25F, 0.05F, 0.2F,  -0.4F, 0.1F, 0.25F},
      5, 4);
  const TargetTailWeights tail{output_norm, output_proj, final_norm,
                               linear_weight(head), false};
  const TargetTailWeights prepared_tail =
      deltafin::provider_internal::precompute_target_tail_score_weight(tail);
  const at::Tensor mixed =
      residual_reference(hidden, anchors, output_proj, output_norm);
  const at::Tensor expected_logits =
      at::matmul(rms_reference(mixed, final_norm), head.transpose(0, 1));
  require_equal(deltafin::provider_internal::finish_target_tail(
                    hidden, TargetBlockResidual{anchors}, tail, false),
                expected_logits, "target tail");
  require_equal(deltafin::provider_internal::finish_target_tail(
                    hidden, TargetBlockResidual{anchors}, prepared_tail, false),
                expected_logits, "bind-time target-tail score product");
}

void original_bf16_head_parity_test() {
  constexpr std::int64_t kRows = 7;
  constexpr std::int64_t kHidden = 4;
  constexpr std::int64_t kVocabulary = 11;
  Bf16CpuT1Kernel kernel(4);
  const at::Tensor source_head = at::linspace(
      -0.375F, 0.625F, kVocabulary * kHidden,
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU))
                                     .reshape({kVocabulary, kHidden})
                                     .contiguous();
  const MoeRowInt8Matrix head = original_bf16_weight(source_head, kernel);
  if (!head.original_bf16.is_owned() || head.dense_f32.defined() ||
      head.original_bf16.owned_storage == nullptr ||
      head.original_bf16.owned_storage->tensor.nbytes() !=
          kVocabulary * kHidden * 2) {
    throw std::runtime_error(
        "original-BF16 language-model head is not two bytes per element");
  }
  const at::Tensor hidden = at::linspace(
      -0.5F, 0.75F, kRows * kHidden,
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU))
                                .reshape({kRows, kHidden})
                                .contiguous();
  const at::Tensor logits =
      deltafin::provider_internal::target_language_model_head_rows(
          hidden, head, false, false);
  const at::Tensor expanded =
      deltafin::provider_internal::materialize_original_bf16_cpu_f32(
          head.original_bf16);
  const at::Tensor reference =
      at::matmul(hidden, expanded.transpose(0, 1)).contiguous();
  const double error = max_abs_error(logits, reference);
  if (!std::isfinite(error) || error > 1.0e-5) {
    throw std::runtime_error(
        "original-BF16 head logits exceeded fp32 accumulation tolerance");
  }
  if (!at::equal(at::argmax(logits, 1), at::argmax(reference, 1))) {
    throw std::runtime_error(
        "original-BF16 head changed greedy token decisions");
  }
  std::cout << "provider_target.original_bf16_head_2byte=PASS\n"
            << "provider_target.original_bf16_head_logits=PASS\n"
            << "provider_target.original_bf16_head_tokens=PASS\n";
}

void packed_dense_bundle_test(const at::Device &device) {
  constexpr std::int64_t kWidth = 32;
  constexpr std::int64_t kIntermediate = 32;
  at::manual_seed(6031);
  const at::Tensor hidden =
      at::linspace(-0.45F, 0.55F, kWidth, at::TensorOptions().dtype(at::kFloat))
          .reshape({1, kWidth})
          .to(device);
  const at::Tensor gate_q =
      at::empty({kIntermediate, kWidth}, at::TensorOptions().dtype(at::kChar))
          .random_(-16, 17);
  const at::Tensor up_q = gate_q.flip({0}).contiguous();
  const at::Tensor down_q =
      at::empty({kWidth, kIntermediate}, at::TensorOptions().dtype(at::kChar))
          .random_(-16, 17);
  const at::Tensor gate_scale = at::full({kIntermediate}, 0.03125F,
                                         at::TensorOptions().dtype(at::kFloat));
  const at::Tensor down_scale =
      at::full({kWidth}, 0.03125F, at::TensorOptions().dtype(at::kFloat));
  TargetDenseWeights baseline{packed_weight(gate_q, gate_scale, device),
                              packed_weight(up_q, gate_scale, device),
                              packed_weight(down_q, down_scale, device), true};
  const at::Tensor separate =
      deltafin::provider_internal::run_target_dense(hidden, baseline, false);
  TargetDenseWeights storage_only =
      deltafin::provider_internal::bundle_target_dense_gate_up(
          baseline, true);
  if (storage_only.bundled_gate_up_enabled) {
    throw std::runtime_error(
        "dense gate/up adjacency enabled experimental execution by default");
  }
  require_equal(
      deltafin::provider_internal::run_target_dense(
          hidden, storage_only, false),
      separate, "default separate dense gate/up execution");
  TargetDenseWeights bundled =
      deltafin::provider_internal::bundle_target_dense_gate_up(
          std::move(baseline), true, true);
  if (!bundled.gate.quantized.is_alias_of(bundled.gate_up.quantized) ||
      !bundled.up.quantized.is_alias_of(bundled.gate_up.quantized) ||
      !bundled.gate.row_scales.is_alias_of(bundled.gate_up.row_scales) ||
      !bundled.up.row_scales.is_alias_of(bundled.gate_up.row_scales)) {
    throw std::runtime_error(
        "dense gate/up bundle retained duplicate resident storage");
  }
  require_equal(
      deltafin::provider_internal::run_target_dense(hidden, bundled, false),
      separate, "packed dense gate/up bundle");
}

void embedding_test() {
  std::vector<std::int8_t> quantized_values{1,  -2, 3,  -4, 5,  6,
                                            -7, 8,  -9, 10, 11, -12};
  const at::Tensor quantized =
      at::from_blob(quantized_values.data(), {3, 4},
                    at::TensorOptions().dtype(at::kChar))
          .clone();
  const at::Tensor scales = vector({0.5F, 0.25F, 0.125F});
  const MoeRowInt8Matrix embedding{quantized, scales, at::Tensor(), {}};
  const at::Tensor expected =
      quantized[1].to(at::kFloat).mul(scales[1]).reshape({1, 4});
  require_equal(
      deltafin::provider_internal::target_embedding_row(1, embedding, false),
      expected, "embedding row");
  require_throws(
      [&] {
        static_cast<void>(deltafin::provider_internal::target_embedding_row(
            3, embedding, false));
      },
      "embedding range contract");
}

void exact_residual_schedule_test() {
  constexpr std::int64_t kHidden = 7168;
  const at::Tensor one =
      at::ones({kHidden}, at::TensorOptions().dtype(at::kFloat));
  const at::Tensor projection =
      at::zeros({1, kHidden}, at::TensorOptions().dtype(at::kFloat));
  const TargetResidualWeights weights{one, one, projection,
                                      one, one, projection};
  at::Tensor hidden =
      at::linspace(-0.5F, 0.5F, kHidden, at::TensorOptions().dtype(at::kFloat))
          .reshape({1, kHidden})
          .contiguous();
  TargetBlockResidual residual =
      deltafin::provider_internal::empty_target_block_residual(
          at::Device(at::kCPU), kHidden);

  for (std::uint32_t layer = 0; layer < 93; ++layer) {
    const auto attention =
        deltafin::provider_internal::prepare_target_attention(
            hidden, residual, weights, layer, true);
    const auto mlp = deltafin::provider_internal::prepare_target_mlp(
        attention, at::zeros_like(hidden), weights, true);
    hidden = deltafin::provider_internal::complete_target_layer(
        mlp, at::zeros_like(hidden), true);
    residual.anchors = mlp.next_anchors;
  }
  if (residual.anchors.size(1) != 8) {
    throw std::runtime_error(
        "93-layer residual schedule did not retain 8 anchors");
  }

  TargetBlockResidual wrong =
      deltafin::provider_internal::empty_target_block_residual(
          at::Device(at::kCPU), kHidden);
  require_throws(
      [&] {
        static_cast<void>(deltafin::provider_internal::prepare_target_attention(
            hidden, wrong, weights, 1, true));
      },
      "exact anchor count");

  auto forged = deltafin::provider_internal::prepare_target_attention(
      hidden, wrong, weights, 0, true);
  forged.prefix_sum = hidden;
  require_throws(
      [&] {
        static_cast<void>(deltafin::provider_internal::prepare_target_mlp(
            forged, at::zeros_like(hidden), weights, true));
      },
      "exact boundary state");
}

void run_residual_score_benchmark(const at::Device &device) {
  constexpr std::int64_t kHidden = 7168;
  const at::Tensor hidden =
      bf16_expanded_linspace({1, kHidden}, -0.75F, 0.625F, device);
  const at::Tensor anchors =
      bf16_expanded_linspace({1, 8, kHidden}, -0.5F, 0.75F, device);
  const TargetBlockResidual residual{anchors};
  const TargetResidualWeights baseline{
      bf16_expanded_linspace({kHidden}, 0.75F, 1.25F, device),
      bf16_expanded_linspace({kHidden}, 0.625F, 1.375F, device),
      bf16_expanded_linspace({1, kHidden}, -0.125F, 0.125F, device),
      bf16_expanded_linspace({kHidden}, 0.875F, 1.125F, device),
      bf16_expanded_linspace({kHidden}, 0.5F, 1.5F, device),
      bf16_expanded_linspace({1, kHidden}, 0.1875F, -0.1875F, device)};
  const TargetResidualWeights prepared =
      deltafin::provider_internal::precompute_target_residual_score_weights(
          baseline);
  const at::Tensor control =
      deltafin::provider_internal::prepare_target_attention(hidden, residual,
                                                            baseline, 92, true)
          .normalized;
  const at::Tensor candidate =
      deltafin::provider_internal::prepare_target_attention(hidden, residual,
                                                            prepared, 92, true)
          .normalized;
  synchronize_device(device);
  require_equal(candidate, control,
                "real-shape prepared residual score product");

  at::Tensor sink;
  const auto run_control = [&] {
    sink = deltafin::provider_internal::prepare_target_attention(
               hidden, residual, baseline, 92, true)
               .normalized;
  };
  const auto run_prepared = [&] {
    sink = deltafin::provider_internal::prepare_target_attention(
               hidden, residual, prepared, 92, true)
               .normalized;
  };
  const PairedTiming timing =
      paired_milliseconds(run_control, run_prepared, device, 500);
  if (!sink.defined()) {
    throw std::runtime_error("residual benchmark did not publish output");
  }
  std::cout << "benchmark.target_residual_score_control_ms="
            << timing.control_ms << '\n'
            << "benchmark.target_residual_score_prepared_ms="
            << timing.candidate_ms << '\n'
            << "benchmark.target_residual_score_speedup="
            << timing.median_speedup << "x\n"
            << "benchmark.target_residual_score_wins=" << timing.candidate_wins
            << '/' << timing.samples << '\n'
            << "check.target_residual_score=PASS (bit-exact)\n";
}

void run_rms_norm_experiment(const at::Device &device) {
  constexpr std::int64_t kHidden = 7168;
  const std::array<std::int64_t, 1> normalized_shape{kHidden};
  const at::Tensor weight =
      bf16_expanded_linspace({kHidden}, 0.5F, 1.5F, device);
  std::vector<float> adversarial_values(kHidden);
  constexpr std::array<float, 16> pattern{
      0.0F, -0.0F, 1.0e-12F, -1.0e-12F, 1.0e-4F, -1.0e-4F, 0.125F,  -0.125F,
      1.0F, -1.0F, 31.75F,   -31.75F,   1024.0F, -1024.0F, 8192.0F, -8192.0F};
  for (std::int64_t index = 0; index < kHidden; ++index) {
    adversarial_values[static_cast<std::size_t>(index)] =
        pattern[static_cast<std::size_t>(index) % pattern.size()];
  }
  const std::vector<at::Tensor> cases{
      bf16_expanded_linspace({1, kHidden}, -0.75F, 0.625F, device),
      vector(adversarial_values).reshape({1, kHidden}).to(device),
      at::full({1, kHidden}, 0.03125F,
               at::TensorOptions().dtype(at::kFloat).device(device)),
      at::zeros({1, kHidden},
                at::TensorOptions().dtype(at::kFloat).device(device))};

  bool bit_exact = true;
  double worst_error = 0.0;
  for (const at::Tensor &input : cases) {
    const at::Tensor control = rms_reference(input, weight);
    const at::Tensor exported = at::rms_norm(input, normalized_shape, weight,
                                             static_cast<double>(kEpsilon));
    synchronize_device(device);
    if (!at::equal(exported, control)) {
      bit_exact = false;
      worst_error = std::max(worst_error, max_abs_error(exported, control));
    }
  }
  if (!bit_exact) {
    std::cout << "check.exported_rms_norm=REJECT (not bit-exact)\n"
              << "check.exported_rms_norm_max_abs=" << worst_error << '\n';
    return;
  }

  const at::Tensor input = cases.front();
  at::Tensor sink;
  const auto run_control = [&] { sink = rms_reference(input, weight); };
  const auto run_exported = [&] {
    sink = at::rms_norm(input, normalized_shape, weight,
                        static_cast<double>(kEpsilon));
  };
  const PairedTiming timing =
      paired_milliseconds(run_control, run_exported, device, 200);
  if (!sink.defined()) {
    throw std::runtime_error("RMS benchmark did not publish output");
  }
  std::cout << "benchmark.rms_expression_ms=" << timing.control_ms << '\n'
            << "benchmark.rms_exported_ms=" << timing.candidate_ms << '\n'
            << "benchmark.rms_exported_speedup=" << timing.median_speedup
            << "x\n"
            << "benchmark.rms_exported_wins=" << timing.candidate_wins << '/'
            << timing.samples << '\n'
            << "check.exported_rms_norm=PASS (bit-exact)\n";
}

MoeRowInt8Matrix synthetic_packed_weight(const std::int64_t rows,
                                         const std::int64_t columns,
                                         const at::Device &device,
                                         const std::int64_t seed) {
  at::manual_seed(seed);
  const at::Tensor quantized =
      at::empty({rows, columns}, at::TensorOptions().dtype(at::kChar))
          .random_(-31, 32);
  const at::Tensor scales = bf16_expanded_linspace(
      {rows}, 0.0009765625F, 0.00390625F, at::Device(at::kCPU));
  return packed_weight(quantized, scales, device);
}

void run_dense_bundle_benchmark(const at::Device &device) {
  constexpr std::int64_t kHidden = 7168;
  constexpr std::int64_t kIntermediate = 33792;
  TargetDenseWeights weights{
      synthetic_packed_weight(kIntermediate, kHidden, device, 7401),
      synthetic_packed_weight(kIntermediate, kHidden, device, 7402),
      MoeRowInt8Matrix{}, true};
  const std::uint64_t component_bytes =
      static_cast<std::uint64_t>(weights.gate.quantized.numel() +
                                 weights.up.quantized.numel()) +
      static_cast<std::uint64_t>(weights.gate.row_scales.numel() +
                                 weights.up.row_scales.numel()) *
          sizeof(float);
  weights = deltafin::provider_internal::bundle_target_dense_gate_up(
      std::move(weights), true, true);
  if (!weights.gate.quantized.is_alias_of(weights.gate_up.quantized) ||
      !weights.up.quantized.is_alias_of(weights.gate_up.quantized) ||
      !weights.gate.row_scales.is_alias_of(weights.gate_up.row_scales) ||
      !weights.up.row_scales.is_alias_of(weights.gate_up.row_scales)) {
    throw std::runtime_error(
        "real-shape dense gate/up bundle duplicated resident storage");
  }
  const std::uint64_t bundle_bytes =
      static_cast<std::uint64_t>(weights.gate_up.quantized.numel()) +
      static_cast<std::uint64_t>(weights.gate_up.row_scales.numel()) *
          sizeof(float);
  if (component_bytes != bundle_bytes) {
    throw std::runtime_error(
        "real-shape dense gate/up bundle changed payload residency");
  }
  const at::Tensor hidden =
      bf16_expanded_linspace({1, kHidden}, -0.125F, 0.15625F, device);
  std::array<at::Tensor, 2> separate;
  at::Tensor combined;
  const auto run_separate = [&] {
    separate[0] = at::_weight_int8pack_mm(hidden, weights.gate.quantized,
                                          weights.gate.row_scales);
    separate[1] = at::_weight_int8pack_mm(hidden, weights.up.quantized,
                                          weights.up.row_scales);
  };
  const auto run_bundle = [&] {
    combined = at::_weight_int8pack_mm(hidden, weights.gate_up.quantized,
                                       weights.gate_up.row_scales);
  };
  run_separate();
  run_bundle();
  synchronize_device(device);
  require_equal(combined, at::cat({separate[0], separate[1]}, 1),
                "real-shape dense gate/up one-call projection");

  const PairedTiming timing =
      paired_milliseconds(run_separate, run_bundle, device, 50);
  std::cout << "benchmark.dense_gate_up_separate_ms=" << timing.control_ms
            << '\n'
            << "benchmark.dense_gate_up_bundle_ms=" << timing.candidate_ms
            << '\n'
            << "benchmark.dense_gate_up_speedup=" << timing.median_speedup
            << "x\n"
            << "benchmark.dense_gate_up_wins=" << timing.candidate_wins << '/'
            << timing.samples << '\n'
            << "benchmark.dense_gate_up_component_bytes=" << component_bytes
            << '\n'
            << "benchmark.dense_gate_up_bundle_bytes=" << bundle_bytes << '\n'
            << "benchmark.dense_gate_up_resident_delta_bytes=0\n"
            << "check.dense_gate_up_real_shape=PASS (bit-exact)\n";
}

void run_residual_workspace_experiment(const at::Device &device) {
  constexpr std::int64_t kHidden = 7168;
  const at::Tensor prefix =
      bf16_expanded_linspace({1, kHidden}, -0.75F, 0.625F, device);
  const at::Tensor anchors =
      bf16_expanded_linspace({1, 8, kHidden}, -0.5F, 0.75F, device);
  const at::Tensor norm =
      bf16_expanded_linspace({kHidden}, 0.625F, 1.375F, device);
  const at::Tensor projection =
      bf16_expanded_linspace({1, kHidden}, -0.125F, 0.125F, device);
  const at::Tensor score_weight = norm * projection.squeeze(0);
  at::Tensor workspace = at::empty(
      {1, 9, kHidden}, at::TensorOptions().dtype(at::kFloat).device(device));
  at::Tensor sink;
  const auto evaluate = [&](const at::Tensor &values) {
    const at::Tensor variance =
        at::mean(at::pow(values, 2), std::vector<std::int64_t>{-1}, true);
    const at::Tensor keys =
        values * at::rsqrt(variance + static_cast<double>(kEpsilon));
    const at::Tensor scores =
        at::sum(keys * score_weight, std::vector<std::int64_t>{-1}, false);
    const at::Tensor probabilities =
        at::softmax(scores, -1, at::kFloat).unsqueeze(1);
    sink = at::matmul(probabilities, values).squeeze(1);
  };
  const auto run_cat = [&] {
    evaluate(at::cat({anchors, prefix.unsqueeze(1)}, 1));
  };
  const auto run_workspace = [&] {
    workspace.slice(1, 0, 8).copy_(anchors);
    workspace.slice(1, 8, 9).copy_(prefix.unsqueeze(1));
    evaluate(workspace);
  };
  run_cat();
  synchronize_device(device);
  const at::Tensor control = sink.clone();
  run_workspace();
  synchronize_device(device);
  require_equal(sink, control, "fixed target residual workspace");
  const PairedTiming timing =
      paired_milliseconds(run_cat, run_workspace, device, 200);
  std::cout << "benchmark.target_residual_cat_ms=" << timing.control_ms << '\n'
            << "benchmark.target_residual_workspace_ms=" << timing.candidate_ms
            << '\n'
            << "benchmark.target_residual_workspace_speedup="
            << timing.median_speedup << "x\n"
            << "benchmark.target_residual_workspace_wins="
            << timing.candidate_wins << '/' << timing.samples << '\n'
            << "check.target_residual_workspace=NOT_INTEGRATED"
               " (shared async storage needs session/stream ownership)\n";
}

struct Options {
  at::Device device = at::Device(at::kCPU);
  bool benchmark = false;
};

Options parse_options(const int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--benchmark") {
      options.benchmark = true;
      continue;
    }
    if (argument == "--device" && ++index < argc) {
      const std::string_view value(argv[index]);
      if (value == "cpu") {
        options.device = at::Device(at::kCPU);
      } else if (value == "mps") {
#if defined(__APPLE__)
        if (!at::hasMPS()) {
          throw std::runtime_error("MPS is unavailable to provider_target");
        }
        options.device = at::Device(at::kMPS);
#else
        throw std::invalid_argument("MPS requires macOS");
#endif
      } else if (value == "cuda") {
        // Only the --benchmark path reads this device; the correctness body
        // stays on CPU with a separate MPS canary. An explicit request for
        // absent hardware is a failure here rather than a skip.
        if (deltafin::provider_internal::cuda_device_count() == 0) {
          throw std::runtime_error("CUDA is unavailable to provider_target");
        }
        options.device = at::Device(at::kCUDA, 0);
      } else {
        throw std::invalid_argument(
            "target test device must be cpu, mps, or cuda");
      }
      continue;
    }
    throw std::invalid_argument(
        "usage: deltafin-provider-target-test [--benchmark] "
        "[--device cpu|mps|cuda]");
  }
  return options;
}

void mps_canary_test() {
#if defined(__APPLE__)
  if (!at::hasMPS()) {
    std::cout << "provider_target.mps=UNAVAILABLE\n";
    return;
  }
  const at::Device device(at::kMPS);
  const at::Tensor hidden = matrix({0.2F, -0.4F, 0.8F, 0.5F}, 1, 4).to(device);
  const at::Tensor anchors = matrix({-0.3F, 0.7F, 0.1F, -0.2F}, 1, 4)
                                 .unsqueeze(1)
                                 .to(device)
                                 .contiguous();
  const TargetResidualWeights cpu_weights = residual_weights();
  const TargetResidualWeights weights{
      cpu_weights.input_norm.to(device),
      cpu_weights.self_attention_res_norm.to(device),
      cpu_weights.self_attention_res_projection.to(device),
      cpu_weights.post_attention_norm.to(device),
      cpu_weights.mlp_res_norm.to(device),
      cpu_weights.mlp_res_projection.to(device)};
  const auto attention = deltafin::provider_internal::prepare_target_attention(
      hidden, TargetBlockResidual{anchors}, weights, 5, false);
  const at::Tensor expected_attention = rms_reference(
      residual_reference(hidden, anchors, weights.self_attention_res_projection,
                         weights.self_attention_res_norm),
      weights.input_norm);
  require_equal(attention.normalized, expected_attention,
                "MPS attention preparation");

  const at::Tensor attention_output =
      matrix({0.05F, 0.1F, -0.2F, 0.15F}, 1, 4).to(device);
  const auto mlp = deltafin::provider_internal::prepare_target_mlp(
      attention, attention_output, weights, false);
  const at::Tensor prefix = hidden + attention_output;
  const at::Tensor expected_lookahead =
      residual_reference(prefix, anchors, weights.mlp_res_projection,
                         weights.mlp_res_norm);
  require_equal(mlp.lookahead_source, expected_lookahead,
                "MPS exact MLP lookahead source");
  require_equal(mlp.normalized,
                rms_reference(expected_lookahead, weights.post_attention_norm),
                "MPS MLP preparation");
  const at::Tensor mlp_output =
      matrix({-0.02F, 0.03F, 0.04F, -0.01F}, 1, 4).to(device);
  require_equal(deltafin::provider_internal::complete_target_layer(
                    mlp, mlp_output, false),
                prefix + mlp_output, "MPS layer completion");

  const at::Tensor gate =
      matrix({0.1F, 0.2F, -0.1F, 0.3F, -0.2F, 0.4F, 0.25F, -0.15F}, 2, 4)
          .to(device);
  const at::Tensor up = gate.flip({0}).contiguous();
  const at::Tensor down =
      matrix({0.2F, -0.1F, -0.3F, 0.25F, 0.1F, 0.3F, 0.4F, -0.25F}, 4, 2)
          .to(device);
  const TargetDenseWeights dense{linear_weight(gate), linear_weight(up),
                                 linear_weight(down), false};
  require_equal(
      deltafin::provider_internal::run_target_dense(hidden, dense, false),
      at::matmul(situ_reference(at::matmul(hidden, gate.transpose(0, 1)),
                                at::matmul(hidden, up.transpose(0, 1))),
                 down.transpose(0, 1)),
      "MPS dense SiTU tape");

  const at::Tensor output_norm = vector({1.0F, 0.9F, 1.1F, 1.2F}).to(device);
  const at::Tensor output_projection =
      matrix({0.2F, -0.3F, 0.1F, 0.4F}, 1, 4).to(device);
  const at::Tensor final_norm = vector({0.8F, 1.0F, 1.2F, 0.7F}).to(device);
  const at::Tensor head = matrix({0.1F, 0.2F, 0.3F, 0.4F, -0.2F, 0.1F, 0.25F,
                                  0.35F, 0.4F, -0.1F, 0.2F, -0.3F},
                                 3, 4)
                              .to(device);
  const TargetTailWeights tail{output_norm, output_projection, final_norm,
                               linear_weight(head), false};
  const at::Tensor mixed =
      residual_reference(hidden, anchors, output_projection, output_norm);
  require_equal(
      deltafin::provider_internal::finish_target_tail(
          hidden, TargetBlockResidual{anchors}, tail, false),
      at::matmul(rms_reference(mixed, final_norm), head.transpose(0, 1)),
      "MPS target tail");

  const at::Tensor quantized =
      at::arange(-8, 4, at::TensorOptions().dtype(at::kChar).device(device))
          .reshape({3, 4});
  const at::Tensor scales = vector({0.5F, 0.25F, 0.125F}).to(device);
  const MoeRowInt8Matrix embedding{quantized, scales, at::Tensor(), {}};
  require_equal(
      deltafin::provider_internal::target_embedding_row(1, embedding, false),
      quantized[1].to(at::kFloat).mul(scales[1]).reshape({1, 4}),
      "MPS embedding row");
  packed_dense_bundle_test(device);
  std::cout << "provider_target.mps=PASS\n";
#else
  std::cout << "provider_target.mps=NOT_APPLICABLE\n";
#endif
}

} // namespace

int main(const int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    residual_tape_test();
    residual_rows_live_shape_test();
    dense_and_tail_test();
    original_bf16_head_parity_test();
    packed_dense_bundle_test(at::Device(at::kCPU));
    embedding_test();
    exact_residual_schedule_test();
    mps_canary_test();
    if (options.benchmark) {
      run_residual_score_benchmark(options.device);
      run_rms_norm_experiment(options.device);
      run_dense_bundle_benchmark(options.device);
      run_residual_workspace_experiment(options.device);
      std::cout << "benchmark.device=" << options.device.str() << '\n';
    }
    std::cout << "provider_target.residual=PASS\n"
              << "provider_target.residual_rows_live_shape=PASS\n"
              << "provider_target.dense_tail=PASS\n"
              << "provider_target.dense_gate_up_bundle=PASS (bit-exact)\n"
              << "provider_target.embedding=PASS\n"
              << "provider_target.exact_schedule=PASS\n"
              << "provider_target.python_runtime=ABSENT\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "provider_target=FAIL: " << error.what() << '\n';
    return 1;
  }
}
