#include "provider_dspark_model.h"

#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using deltafin::provider_internal::DSparkDecoderWeights;
using deltafin::provider_internal::DSparkModel;
using deltafin::provider_internal::DSparkModelWeights;
using deltafin::provider_internal::DSparkMlaWeights;
using deltafin::provider_internal::DSparkMlpWeights;
using deltafin::provider_internal::DSparkShape;
using deltafin::provider_internal::DSparkTargetIo;
using deltafin::provider_internal::MoeRowInt8Matrix;
using deltafin::provider_internal::kDSparkLayers;
using deltafin::provider_internal::prepare_dspark_model_weights;
using deltafin::provider_internal::target_embedding_rows;
using deltafin::provider_internal::target_language_model_head_rows;

at::Tensor deterministic(const at::IntArrayRef shape, const std::int64_t salt,
                         const float scale = 0.03125F) {
  std::int64_t elements = 1;
  for (const std::int64_t dimension : shape) {
    elements *= dimension;
  }
  at::Tensor tensor =
      at::empty(shape, at::TensorOptions().dtype(at::kFloat));
  float* values = tensor.data_ptr<float>();
  for (std::int64_t index = 0; index < elements; ++index) {
    const std::int64_t centered =
        ((index + 1) * (salt * 11 + 13) + salt * 3) % 31 - 15;
    values[index] = static_cast<float>(centered) * scale;
  }
  return tensor;
}

at::Tensor bf16(const at::IntArrayRef shape, const std::int64_t salt,
                const float scale = 0.03125F) {
  return deterministic(shape, salt, scale).to(at::kBFloat16).contiguous();
}

at::Tensor norm(const std::int64_t width, const std::int64_t salt) {
  return (bf16({width}, salt, 0.015625F) +
          at::ones({width}, at::TensorOptions().dtype(at::kBFloat16)))
      .contiguous();
}

DSparkMlaWeights make_attention(const DSparkShape& shape,
                                const std::int64_t salt) {
  return DSparkMlaWeights{
      .query_a = bf16({shape.q_lora_rank, shape.hidden_size}, salt),
      .query_a_norm = norm(shape.q_lora_rank, salt + 1),
      .query_b = bf16({shape.num_heads * shape.query_head_dim(),
                       shape.q_lora_rank},
                      salt + 2),
      .key_value_a =
          bf16({shape.kv_lora_rank + shape.qk_rope_head_dim,
                shape.hidden_size},
               salt + 3),
      .key_value_a_norm = norm(shape.kv_lora_rank, salt + 4),
      .key_value_b =
          bf16({shape.num_heads *
                    (shape.qk_nope_head_dim + shape.value_head_dim),
                shape.kv_lora_rank},
               salt + 5),
      .output = bf16({shape.hidden_size,
                      shape.num_heads * shape.value_head_dim},
                     salt + 6),
  };
}

DSparkMlpWeights make_mlp(const DSparkShape& shape,
                          const std::int64_t salt) {
  return DSparkMlpWeights{
      .gate = bf16({shape.intermediate_size, shape.hidden_size}, salt),
      .up = bf16({shape.intermediate_size, shape.hidden_size}, salt + 1),
      .down = bf16({shape.hidden_size, shape.intermediate_size}, salt + 2),
  };
}

DSparkModelWeights make_weights(const DSparkShape& shape) {
  DSparkModelWeights weights{
      .context_projection =
          bf16({shape.hidden_size, shape.target_context_width()}, 1),
      .context_norm = norm(shape.hidden_size, 2),
      .layers = {},
      .fused_context_projection = {},
      .final_norm = norm(shape.hidden_size, 80),
      .markov_embedding =
          bf16({shape.vocabulary_size, shape.markov_rank}, 81),
      .markov_output =
          bf16({shape.vocabulary_size, shape.markov_rank}, 82),
      .confidence_weight =
          bf16({1, shape.hidden_size + shape.markov_rank}, 83),
      .confidence_bias = bf16({1}, 84),
  };
  for (std::size_t layer = 0; layer < kDSparkLayers; ++layer) {
    const auto salt = static_cast<std::int64_t>(10 + layer * 12);
    weights.layers[layer] = DSparkDecoderWeights{
        .input_norm = norm(shape.hidden_size, salt),
        .attention = make_attention(shape, salt + 1),
        .post_attention_norm = norm(shape.hidden_size, salt + 8),
        .mlp = make_mlp(shape, salt + 9),
    };
  }
  return weights;
}

DSparkTargetIo make_target_io(const DSparkShape& shape) {
  at::Tensor quantized = at::empty(
      {shape.vocabulary_size, shape.hidden_size},
      at::TensorOptions().dtype(at::kChar));
  auto values = quantized.accessor<std::int8_t, 2>();
  for (std::int64_t row = 0; row < shape.vocabulary_size; ++row) {
    for (std::int64_t column = 0; column < shape.hidden_size; ++column) {
      values[row][column] = static_cast<std::int8_t>(
          ((row + 2) * (column + 3)) % 11 - 5);
    }
  }
  const at::Tensor scales =
      deterministic({shape.vocabulary_size}, 91, 0.00390625F).abs() +
      0.0078125F;
  return DSparkTargetIo{
      .embedding = MoeRowInt8Matrix{
          .quantized = quantized,
          .row_scales = scales.contiguous(),
          .dense_f32 = {},
          .original_bf16 = {},
      },
      .language_model_head = MoeRowInt8Matrix{
          .quantized = {},
          .row_scales = {},
          .dense_f32 = deterministic(
              {shape.vocabulary_size, shape.hidden_size}, 92, 0.015625F),
          .original_bf16 = {},
      },
      .head_packed_int8_qualified = false,
      .exact_k3 = false,
  };
}

void require_close(const at::Tensor& actual, const at::Tensor& expected,
                   const char* name, const double tolerance = 0.004) {
  if (actual.sizes() != expected.sizes() ||
      actual.scalar_type() != expected.scalar_type()) {
    throw std::runtime_error(std::string(name) +
                             " has the wrong shape or dtype");
  }
  const double difference =
      (actual.to(at::kFloat) - expected.to(at::kFloat))
          .abs()
          .max()
          .item<double>();
  if (!std::isfinite(difference) || difference > tolerance) {
    throw std::runtime_error(std::string(name) + " differs by " +
                             std::to_string(difference));
  }
}

template <typename Operation>
void require_invalid(Operation&& operation, const char* name) {
  try {
    operation();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(std::string(name) + " did not fail closed");
}

void test_weight_roster_and_target_helpers() {
  const DSparkShape shape = DSparkShape::small_canary();
  DSparkModelWeights weights =
      prepare_dspark_model_weights(shape, make_weights(shape));
  const std::int64_t compressed =
      shape.kv_lora_rank + shape.qk_rope_head_dim;
  if (weights.fused_context_projection.sizes() !=
      at::IntArrayRef({static_cast<std::int64_t>(kDSparkLayers) * compressed,
                       shape.hidden_size}) ||
      weights.fused_context_projection.scalar_type() != at::kBFloat16) {
    throw std::runtime_error("fused five-layer context roster is invalid");
  }
  DSparkTargetIo io = make_target_io(shape);
  const at::Tensor ids =
      at::tensor({2, 7}, at::TensorOptions().dtype(at::kLong));
  const at::Tensor embedded =
      target_embedding_rows(ids, io.embedding, false);
  const at::Tensor expected_embedding =
      io.embedding.quantized.index_select(0, ids).to(at::kFloat) *
      io.embedding.row_scales.index_select(0, ids).unsqueeze(1);
  require_close(embedded, expected_embedding.contiguous(),
                "shared target embedding", 0.0);
  const at::Tensor hidden = bf16({2, shape.hidden_size}, 95);
  const at::Tensor logits = target_language_model_head_rows(
      hidden, io.language_model_head, false, false);
  const at::Tensor expected_logits = at::matmul(
      hidden.to(at::kFloat), io.language_model_head.dense_f32.transpose(0, 1));
  require_close(logits, expected_logits.contiguous(), "shared target head",
                0.0);

  DSparkModelWeights bad = make_weights(shape);
  bad.confidence_bias = at::Tensor();
  require_invalid(
      [&] { (void)prepare_dspark_model_weights(shape, std::move(bad)); },
      "incomplete DSpark weight roster");
}

void test_cache_backbone_and_scored_prefix() {
  const DSparkShape shape = DSparkShape::small_canary();
  DSparkModel model(shape, make_weights(shape), make_target_io(shape));
  const at::Tensor first_context =
      bf16({2, shape.target_context_width()}, 101);
  model.append_target_context(
      first_context,
      at::tensor({0, 1}, at::TensorOptions().dtype(at::kLong)));
  if (model.cache().length() != 2 || model.cache().capacity() != 8 ||
      model.cache().generation() != 1) {
    throw std::runtime_error("initial compact-cache append is wrong");
  }
  const auto short_snapshot = model.cache().snapshot();
  const at::Tensor original_third =
      bf16({1, shape.target_context_width()}, 102);
  model.append_target_context(
      original_third,
      at::tensor({2}, at::TensorOptions().dtype(at::kLong)));
  const auto long_snapshot = model.cache().snapshot();
  const at::Tensor abandoned_row =
      long_snapshot.context(0).latent.narrow(0, 2, 1).clone();
  model.cache().restore(short_snapshot);
  if (!model.cache().requires_fork() || model.cache().length() != 2 ||
      model.cache().generation() <= long_snapshot.generation) {
    throw std::runtime_error("cache restore did not establish COW state");
  }
  const at::Tensor alternative_third =
      bf16({1, shape.target_context_width()}, 109);
  model.append_target_context(
      alternative_third,
      at::tensor({2}, at::TensorOptions().dtype(at::kLong)));
  require_close(long_snapshot.context(0).latent.narrow(0, 2, 1), abandoned_row,
                "snapshot COW preservation", 0.0);
  if (model.cache().requires_fork() || model.cache().length() != 3) {
    throw std::runtime_error("cache COW append did not publish correctly");
  }

  const std::uint64_t generation = model.cache().generation();
  const auto narrow = model.propose(4, 2);
  const auto wide = model.propose(4, 7);
  if (narrow.token_ids.sizes() != at::IntArrayRef({2}) ||
      narrow.token_ids.scalar_type() != at::kLong ||
      narrow.confidence_logits.sizes() != at::IntArrayRef({2}) ||
      narrow.confidence_logits.scalar_type() != at::kFloat ||
      narrow.anchor_position != 3 ||
      narrow.cache_generation != generation || model.cache().length() != 3 ||
      model.cache().generation() != generation) {
    throw std::runtime_error("proposal result/state contract is wrong");
  }
  require_close(narrow.token_ids, wide.token_ids.narrow(0, 0, 2),
                "seven-row Markov scored prefix", 0.0);
  require_close(narrow.confidence_logits,
                wide.confidence_logits.narrow(0, 0, 2),
                "seven-row confidence scored prefix", 0.0);

  require_invalid([&] { (void)model.propose(4, 0); },
                  "empty proposal score width");
  require_invalid(
      [&] {
        model.append_target_context(
            bf16({1, shape.target_context_width()}, 110),
            at::tensor({4}, at::TensorOptions().dtype(at::kLong)));
      },
      "noncontiguous target context");
}

}  // namespace

int main() {
  try {
    const c10::InferenceMode inference_guard;
    test_weight_roster_and_target_helpers();
    test_cache_backbone_and_scored_prefix();
    std::cout << "provider_dspark_model.synthetic=PASS\n";
    std::cout << "provider_dspark_model.query_rows=7\n";
    std::cout << "provider_dspark_model.authority=PROPOSAL_ONLY\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "provider_dspark_model.synthetic=FAIL: " << error.what()
              << '\n';
    return 1;
  }
}
