#include "provider_qwen.h"

#include <ATen/ops/arange.h>
#include <ATen/ops/argmax.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cos.h>
#include <ATen/ops/embedding.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/logsumexp.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/reciprocal.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/stack.h>
#include <c10/core/InferenceMode.h>

#if defined(__APPLE__)
#include <torch/mps.h>
#endif

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace deltafin::provider_internal {
namespace {

constexpr std::uint32_t kBf16 = DELTAFIN_PROVIDER_DSPARK_BF16_V1;
constexpr std::size_t kTensorCount = DELTAFIN_PROVIDER_QWEN_TENSOR_COUNT_V1;

at::Tensor dense_linear(const at::Tensor& hidden, const at::Tensor& weight) {
  return at::linear(hidden, weight, std::nullopt);
}

at::Tensor rms_norm(const at::Tensor& value, const at::Tensor& weight) {
  const at::ScalarType input_type = value.scalar_type();
  const at::Tensor promoted = value.to(at::kFloat);
  const at::Tensor variance = at::mean(at::pow(promoted, 2), {-1}, true);
  const at::Tensor normalized =
      promoted * at::rsqrt(variance + QwenShape::rms_epsilon);
  return weight * normalized.to(input_type);
}

at::Tensor qwen_inverse_frequency(const at::Device& device) {
  // Mirrors Transformers 4.56.2 `_compute_default_rope_parameters`: create
  // the integer roster on CPU, convert it to provider FP32, then use the
  // scalar-base pow/reciprocal operators in the same order.
  const at::Tensor pairs =
      at::arange(0, QwenShape::head_dim, 2,
                 at::TensorOptions().dtype(at::kLong))
          .to(at::TensorOptions().dtype(at::kFloat).device(device));
  return at::reciprocal(at::pow(
      at::Scalar(QwenShape::rope_theta),
      pairs / static_cast<double>(QwenShape::head_dim)));
}

std::pair<at::Tensor, at::Tensor> rotary_embeddings(
    const at::Tensor& hidden, const at::Tensor& position_ids,
    const at::Tensor& inverse_frequency) {
  const std::int64_t batch = position_ids.size(0);
  const at::Tensor expanded = inverse_frequency
                                  .unsqueeze(0)
                                  .unsqueeze(-1)
                                  .to(at::kFloat)
                                  .expand({batch, -1, 1});
  const at::Tensor positions = position_ids.unsqueeze(1).to(at::kFloat);
  const at::Tensor frequencies =
      at::matmul(expanded.to(at::kFloat), positions.to(at::kFloat))
          .transpose(1, 2);
  const at::Tensor embedding = at::cat({frequencies, frequencies}, -1);
  const at::Tensor cosine = (at::cos(embedding) * 1.0).to(hidden.scalar_type());
  const at::Tensor sine = (at::sin(embedding) * 1.0).to(hidden.scalar_type());
  return {cosine, sine};
}

at::Tensor rotate_half(const at::Tensor& value) {
  const std::int64_t half = QwenShape::head_dim / 2;
  return at::cat(
      {-value.narrow(-1, half, half), value.narrow(-1, 0, half)}, -1);
}

at::Tensor apply_rope(const at::Tensor& value, const at::Tensor& cosine,
                      const at::Tensor& sine) {
  const at::Tensor broadcast_cosine = cosine.unsqueeze(1);
  const at::Tensor broadcast_sine = sine.unsqueeze(1);
  return value * broadcast_cosine + rotate_half(value) * broadcast_sine;
}

at::Tensor update_dynamic_cache(at::Tensor& cache,
                                const at::Tensor& current) {
  if (!cache.defined()) {
    const at::Tensor empty = at::empty({0}, current.options());
    cache = at::cat({empty, current}, -2);
  } else {
    cache = at::cat({cache, current}, -2);
  }
  return cache;
}

at::Tensor qwen_forward(
    const at::Tensor& input_ids, const at::Tensor& position_ids,
    const QwenWeights& weights, const at::Tensor& inverse_frequency,
    std::array<at::Tensor, QwenShape::layers>& keys,
    std::array<at::Tensor, QwenShape::layers>& values) {
  at::Tensor hidden = at::embedding(weights.embedding, input_ids);
  const std::int64_t batch = hidden.size(0);
  const std::int64_t rows = hidden.size(1);
  const auto [cosine, sine] =
      rotary_embeddings(hidden, position_ids, inverse_frequency);

  for (std::size_t layer = 0; layer < QwenShape::layers; ++layer) {
    const auto& weight = weights.layers[layer];
    const at::Tensor residual = hidden;
    const at::Tensor normalized = rms_norm(hidden, weight.input_norm);
    at::Tensor query = rms_norm(
        dense_linear(normalized, weight.query)
            .view({batch, rows, QwenShape::heads, QwenShape::head_dim}),
        weight.query_norm);
    at::Tensor key = rms_norm(
        dense_linear(normalized, weight.key)
            .view({batch, rows, QwenShape::kv_heads, QwenShape::head_dim}),
        weight.key_norm);
    query = apply_rope(query.transpose(1, 2), cosine, sine);
    key = apply_rope(key.transpose(1, 2), cosine, sine);
    const at::Tensor value =
        dense_linear(normalized, weight.value)
            .view({batch, rows, QwenShape::kv_heads, QwenShape::head_dim})
            .transpose(1, 2);

    const at::Tensor history_keys = update_dynamic_cache(keys[layer], key);
    const at::Tensor history_values =
        update_dynamic_cache(values[layer], value);
    const bool is_causal = rows > 1;
    at::Tensor attended = at::scaled_dot_product_attention(
        query, history_keys, history_values, std::nullopt, 0.0, is_causal,
        std::pow(static_cast<double>(QwenShape::head_dim), -0.5), true);
    attended = attended.transpose(1, 2).contiguous();
    attended = attended
                   .reshape({batch, rows,
                             QwenShape::heads * QwenShape::head_dim})
                   .contiguous();
    hidden = residual + dense_linear(attended, weight.output);

    const at::Tensor mlp_residual = hidden;
    const at::Tensor mlp_input = rms_norm(hidden, weight.post_attention_norm);
    hidden = mlp_residual +
             dense_linear(at::silu(dense_linear(mlp_input, weight.gate)) *
                              dense_linear(mlp_input, weight.up),
                          weight.down);
  }
  return rms_norm(hidden, weights.final_norm);
}

std::vector<std::int64_t> expected_shape(const QwenShape& shape,
                                         const std::uint32_t slot) {
  if (slot == 0) return {QwenShape::vocabulary, shape.hidden};
  if (slot == 1) return {shape.hidden};
  const std::uint32_t component = (slot - 2) % 11;
  switch (component) {
    case 0: case 1: return {shape.hidden};
    case 2: case 3: return {QwenShape::head_dim};
    case 4: return {QwenShape::heads * QwenShape::head_dim, shape.hidden};
    case 5: case 6:
      return {QwenShape::kv_heads * QwenShape::head_dim, shape.hidden};
    case 7: return {shape.hidden, QwenShape::heads * QwenShape::head_dim};
    case 8: case 9: return {shape.intermediate, shape.hidden};
    case 10: return {shape.hidden, shape.intermediate};
    default: throw std::logic_error("Qwen component index escaped roster");
  }
}

at::Tensor copy_tensor(const DeltafinProviderQwenTensorV1& descriptor,
                       const QwenShape& shape, const at::Device& device) {
  if (descriptor.slot >= kTensorCount || descriptor.scalar_type != kBf16 ||
      descriptor.flags != 0 || descriptor.data == nullptr ||
      descriptor.rank < 1 || descriptor.rank > 2 ||
      descriptor.reserved[0] != 0 || descriptor.reserved[1] != 0) {
    throw std::invalid_argument("Qwen tensor descriptor is invalid");
  }
  const auto expected = expected_shape(shape, descriptor.slot);
  if (descriptor.rank != expected.size()) {
    throw std::invalid_argument("Qwen tensor rank differs from fixed roster");
  }
  std::uint64_t elements = 1;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (descriptor.shape[index] != static_cast<std::uint64_t>(expected[index]) ||
        elements > std::numeric_limits<std::uint64_t>::max() /
                       descriptor.shape[index]) {
      throw std::invalid_argument("Qwen tensor shape differs from fixed roster");
    }
    elements *= descriptor.shape[index];
  }
  if (elements > std::numeric_limits<std::uint64_t>::max() / 2 ||
      descriptor.data_length != elements * 2) {
    throw std::invalid_argument("Qwen tensor byte extent differs from shape");
  }
  at::Tensor borrowed = at::from_blob(
      const_cast<std::uint8_t*>(descriptor.data), expected,
      at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU));
  const at::ScalarType resident_type = device.is_cpu() ? at::kFloat : at::kHalf;
  return borrowed.to(at::TensorOptions().device(device).dtype(resident_type),
                     false, true).contiguous();
}

}  // namespace

QwenShape QwenShape::pinned(const std::uint32_t variant) {
  if (variant == DELTAFIN_PROVIDER_QWEN_06B_V1) return {1024, 3072};
  if (variant == DELTAFIN_PROVIDER_QWEN_17B_V1) return {2048, 6144};
  throw std::invalid_argument("Qwen variant is not pinned");
}

void QwenShape::validate() const {
  if (!((hidden == 1024 && intermediate == 3072) ||
        (hidden == 2048 && intermediate == 6144))) {
    throw std::invalid_argument("Qwen shape is not a pinned architecture");
  }
}

QwenWeights bind_qwen_roster(const DeltafinProviderQwenCreateV1& request,
                             const QwenShape& shape,
                             const at::Device& device) {
  shape.validate();
  if (request.tensor_count != kTensorCount || request.tensors == nullptr) {
    throw std::invalid_argument("Qwen create requires exactly 310 tensors");
  }
  std::array<at::Tensor, kTensorCount> slots;
  for (std::size_t index = 0; index < kTensorCount; ++index) {
    const auto& descriptor = request.tensors[index];
    if (descriptor.slot >= kTensorCount || slots[descriptor.slot].defined()) {
      throw std::invalid_argument("Qwen tensor slot is duplicate or invalid");
    }
    slots[descriptor.slot] = copy_tensor(descriptor, shape, device);
  }
  for (const auto& tensor : slots) {
    if (!tensor.defined()) throw std::invalid_argument("Qwen roster is incomplete");
  }
#if defined(__APPLE__)
  if (device.is_mps()) {
    // Every descriptor points into a Rust-owned read-only checkpoint map that
    // is released immediately after this FFI call.  MPS may queue a no-copy
    // upload even when Tensor::to was requested with non_blocking=false.  Keep
    // the mapping authoritative until the complete 310-tensor upload batch is
    // consumed; one synchronization here avoids both a per-tensor barrier and
    // a multi-gigabyte host clone of the proposal model.
    torch::mps::synchronize();
  }
#endif
  // `layers` is populated by the loop below. State the empty initializer
  // rather than leaving it implicit: GCC rejects the omission under
  // -Werror=missing-field-initializers while Clang accepts it, so an implicit
  // member builds on macOS and fails the Linux provider build.
  QwenWeights weights{.embedding = std::move(slots[0]),
                      .final_norm = std::move(slots[1]),
                      .layers = {}};
  for (std::size_t layer = 0; layer < QwenShape::layers; ++layer) {
    const std::size_t base = 2 + layer * 11;
    weights.layers[layer] = {
        .input_norm = std::move(slots[base]),
        .post_attention_norm = std::move(slots[base + 1]),
        .query_norm = std::move(slots[base + 2]),
        .key_norm = std::move(slots[base + 3]),
        .query = std::move(slots[base + 4]),
        .key = std::move(slots[base + 5]),
        .value = std::move(slots[base + 6]),
        .output = std::move(slots[base + 7]),
        .gate = std::move(slots[base + 8]),
        .up = std::move(slots[base + 9]),
        .down = std::move(slots[base + 10]),
    };
  }
  return weights;
}

QwenModel::QwenModel(QwenShape shape, QwenWeights weights)
    : shape_(shape),
      weights_(std::move(weights)),
      inverse_frequency_(qwen_inverse_frequency(weights_.embedding.device())) {
  shape_.validate();
}

QwenGeneration QwenModel::generate(const std::uint32_t* input_ids,
                                    const std::size_t input_count,
                                    const std::size_t maximum_new_tokens) const {
  const c10::InferenceMode inference_guard;
  if (input_ids == nullptr || input_count == 0 || maximum_new_tokens == 0 ||
      maximum_new_tokens > DELTAFIN_PROVIDER_QWEN_MAX_PROPOSAL_TOKENS_V1 ||
      input_count + maximum_new_tokens >
          static_cast<std::size_t>(QwenShape::maximum_position)) {
    throw std::invalid_argument("Qwen generation bounds are invalid");
  }
  std::vector<std::int64_t> host_input(input_count);
  for (std::size_t index = 0; index < input_count; ++index) {
    if (input_ids[index] >= static_cast<std::uint32_t>(QwenShape::vocabulary)) {
      throw std::invalid_argument("Qwen input token lies outside vocabulary");
    }
    host_input[index] = static_cast<std::int64_t>(input_ids[index]);
  }
  const at::Device device = weights_.embedding.device();
  const at::Tensor provider_input =
      at::from_blob(host_input.data(),
                    {1, static_cast<std::int64_t>(input_count)},
                    at::TensorOptions().dtype(at::kLong).device(at::kCPU))
          .to(at::TensorOptions().dtype(at::kLong).device(device), false, true);
  const at::Tensor provider_positions =
      at::arange(static_cast<std::int64_t>(input_count),
                 at::TensorOptions().dtype(at::kLong).device(device))
          .unsqueeze(0);

  std::array<at::Tensor, QwenShape::layers> keys;
  std::array<at::Tensor, QwenShape::layers> values;
  at::Tensor hidden = qwen_forward(provider_input, provider_positions,
                                   weights_, inverse_frequency_, keys, values);
  std::vector<at::Tensor> selected_tokens;
  std::vector<at::Tensor> scores;
  selected_tokens.reserve(maximum_new_tokens);
  scores.reserve(maximum_new_tokens);

  for (std::size_t step = 0; step < maximum_new_tokens; ++step) {
    const at::Tensor logits =
        dense_linear(hidden.narrow(1, hidden.size(1) - 1, 1),
                     weights_.embedding);
    const at::Tensor next_scores =
        logits.select(1, logits.size(1) - 1)
            .to(at::TensorOptions().dtype(at::kFloat).device(device), false,
                true);
    const at::Tensor selected = at::argmax(next_scores, -1, false);
    scores.push_back(next_scores);
    selected_tokens.push_back(selected);
    if (step + 1 == maximum_new_tokens) break;

    const at::Tensor next_input = selected.view({1, 1});
    const at::Tensor next_position =
        at::arange(static_cast<std::int64_t>(input_count + step),
                   static_cast<std::int64_t>(input_count + step + 1),
                   at::TensorOptions().dtype(at::kLong).device(device))
            .unsqueeze(0);
    hidden = qwen_forward(next_input, next_position, weights_,
                          inverse_frequency_, keys, values);
  }

  const at::Tensor token_tensor = at::cat(selected_tokens, 0);
  std::vector<at::Tensor> selected_logits;
  std::vector<at::Tensor> normalizers;
  selected_logits.reserve(selected_tokens.size());
  normalizers.reserve(selected_tokens.size());
  for (std::size_t index = 0; index < selected_tokens.size(); ++index) {
    const at::Tensor row = scores[index].select(0, 0);
    selected_logits.push_back(at::gather(row, 0, selected_tokens[index]));
    normalizers.push_back(at::logsumexp(row, {0}));
  }
  const at::Tensor probability_tensor = at::exp(
      at::cat(selected_logits, 0) - at::stack(normalizers, 0));
  const at::Tensor host_tokens =
      token_tensor
          .to(at::TensorOptions().dtype(at::kLong).device(at::kCPU), false,
              true)
          .contiguous();
  const at::Tensor host_probabilities =
      probability_tensor
          .to(at::TensorOptions().dtype(at::kFloat).device(at::kCPU), false,
              true)
          .contiguous();

  QwenGeneration result;
  result.token_ids.reserve(maximum_new_tokens);
  result.probabilities.reserve(maximum_new_tokens);
  const auto* token_values = host_tokens.const_data_ptr<std::int64_t>();
  const auto* probability_values =
      host_probabilities.const_data_ptr<float>();
  for (std::size_t index = 0; index < selected_tokens.size(); ++index) {
    const auto token = static_cast<std::uint32_t>(token_values[index]);
    result.token_ids.push_back(token);
    result.probabilities.push_back(probability_values[index]);
    if (token == static_cast<std::uint32_t>(QwenShape::eos_token)) break;
  }
  return result;
}

}  // namespace deltafin::provider_internal
