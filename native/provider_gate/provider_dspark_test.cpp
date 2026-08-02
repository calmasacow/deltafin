#include "provider_dspark.h"

#include <ATen/ATen.h>
#include <ATen/ops/einsum.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/softmax.h>
#include <c10/core/InferenceMode.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using deltafin::provider_internal::DSparkDecoderWeights;
using deltafin::provider_internal::DSparkLatentContext;
using deltafin::provider_internal::DSparkMlaOutput;
using deltafin::provider_internal::DSparkMlaWeights;
using deltafin::provider_internal::DSparkMlpWeights;
using deltafin::provider_internal::DSparkShape;
using deltafin::provider_internal::dspark_apply_yarn_rotary_bf16;
using deltafin::provider_internal::dspark_rms_norm_bf16;
using deltafin::provider_internal::dspark_yarn_inverse_frequencies;
using deltafin::provider_internal::run_dspark_decoder_layer;
using deltafin::provider_internal::run_dspark_mla;
using deltafin::provider_internal::run_dspark_mlp;

at::Tensor deterministic_bf16(const at::IntArrayRef shape,
                              const std::int64_t salt,
                              const float scale = 0.03125F) {
  std::int64_t elements = 1;
  for (const std::int64_t dimension : shape) {
    elements *= dimension;
  }
  at::Tensor result = at::empty(shape, at::TensorOptions().dtype(at::kFloat));
  float* values = result.data_ptr<float>();
  for (std::int64_t index = 0; index < elements; ++index) {
    const std::int64_t centered =
        ((index + 3) * (salt * 7 + 11) + salt * 5) % 29 - 14;
    values[index] = static_cast<float>(centered) * scale;
  }
  return result.to(at::kBFloat16).contiguous();
}

at::Tensor weight(const std::int64_t rows, const std::int64_t columns,
                  const std::int64_t salt) {
  return deterministic_bf16({rows, columns}, salt, 0.015625F);
}

DSparkMlaWeights mla_weights(const DSparkShape& shape) {
  return DSparkMlaWeights{
      .query_a = weight(shape.q_lora_rank, shape.hidden_size, 1),
      .query_a_norm = deterministic_bf16({shape.q_lora_rank}, 2) +
                      at::ones({shape.q_lora_rank},
                               at::TensorOptions().dtype(at::kBFloat16)),
      .query_b =
          weight(shape.num_heads * shape.query_head_dim(), shape.q_lora_rank,
                 3),
      .key_value_a =
          weight(shape.kv_lora_rank + shape.qk_rope_head_dim,
                 shape.hidden_size, 4),
      .key_value_a_norm =
          deterministic_bf16({shape.kv_lora_rank}, 5) +
          at::ones({shape.kv_lora_rank},
                   at::TensorOptions().dtype(at::kBFloat16)),
      .key_value_b =
          weight(shape.num_heads *
                     (shape.qk_nope_head_dim + shape.value_head_dim),
                 shape.kv_lora_rank, 6),
      .output = weight(shape.hidden_size,
                       shape.num_heads * shape.value_head_dim, 7),
  };
}

DSparkMlpWeights mlp_weights(const DSparkShape& shape) {
  return DSparkMlpWeights{
      .gate = weight(shape.intermediate_size, shape.hidden_size, 8),
      .up = weight(shape.intermediate_size, shape.hidden_size, 9),
      .down = weight(shape.hidden_size, shape.intermediate_size, 10),
  };
}

at::Tensor reference_linear(const at::Tensor& input,
                            const at::Tensor& matrix) {
  return at::matmul(input, matrix.transpose(0, 1));
}

void require_close(const at::Tensor& actual, const at::Tensor& expected,
                   const char* name, const double tolerance = 0.004) {
  if (actual.sizes() != expected.sizes() ||
      actual.scalar_type() != expected.scalar_type()) {
    throw std::runtime_error(std::string(name) +
                             " returned the wrong shape or dtype");
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

DSparkMlaOutput reference_mla(const at::Tensor& hidden,
                             const at::Tensor& positions,
                             const DSparkLatentContext& context,
                             const DSparkMlaWeights& weights,
                             const DSparkShape& shape) {
  const std::int64_t rows = hidden.size(0);
  const std::int64_t head_dim = shape.query_head_dim();
  const at::Tensor q_low = dspark_rms_norm_bf16(
      reference_linear(hidden, weights.query_a), weights.query_a_norm,
      shape.rms_epsilon);
  const at::Tensor query =
      reference_linear(q_low, weights.query_b)
          .view({rows, shape.num_heads, head_dim});
  const at::Tensor q_nope = query.narrow(-1, 0, shape.qk_nope_head_dim);
  const at::Tensor q_rope = dspark_apply_yarn_rotary_bf16(
      query.narrow(-1, shape.qk_nope_head_dim, shape.qk_rope_head_dim)
          .contiguous(),
      positions, shape);
  const at::Tensor projected =
      reference_linear(hidden, weights.key_value_a);
  const at::Tensor latent = dspark_rms_norm_bf16(
      projected.narrow(-1, 0, shape.kv_lora_rank).contiguous(),
      weights.key_value_a_norm, shape.rms_epsilon);
  const at::Tensor positional = dspark_apply_yarn_rotary_bf16(
      projected
          .narrow(-1, shape.kv_lora_rank, shape.qk_rope_head_dim)
          .contiguous(),
      positions, shape);
  const at::Tensor all_latent = at::cat({context.latent, latent}, 0);
  const at::Tensor all_positional =
      at::cat({context.positional, positional}, 0);
  const at::Tensor expanded = weights.key_value_b.view(
      {shape.num_heads, shape.qk_nope_head_dim + shape.value_head_dim,
       shape.kv_lora_rank});
  const at::Tensor key_weight =
      expanded.narrow(1, 0, shape.qk_nope_head_dim);
  const at::Tensor value_weight =
      expanded.narrow(1, shape.qk_nope_head_dim, shape.value_head_dim);
  const at::Tensor latent_query =
      at::einsum("qhd,hdl->qhl", {q_nope, key_weight});
  at::Tensor scores =
      at::einsum("qhl,kl->hqk", {latent_query, all_latent});
  scores += at::einsum("qhr,kr->hqk", {q_rope, all_positional});
  const double mscale =
      0.1 * shape.rope_mscale_all_dim * std::log(shape.rope_factor) + 1.0;
  const double scale = std::pow(static_cast<double>(head_dim), -0.5) *
                       mscale * mscale;
  const at::Tensor probabilities =
      at::softmax(scores.to(at::kFloat) * scale, -1, at::kFloat)
          .to(at::kBFloat16);
  const at::Tensor latent_output =
      at::einsum("hqk,kl->qhl", {probabilities, all_latent});
  const at::Tensor value_output =
      at::einsum("qhl,hvl->qhv", {latent_output, value_weight});
  return DSparkMlaOutput{
      .hidden = reference_linear(
          value_output.reshape({rows, shape.num_heads * shape.value_head_dim}),
          weights.output),
      .query_context = DSparkLatentContext{
          .latent = latent,
          .positional = positional,
      },
  };
}

void test_shape_and_rms() {
  const DSparkShape k3 = DSparkShape::k3();
  k3.validate();
  if (!k3.is_exact_k3() || k3.query_head_dim() != 192) {
    throw std::runtime_error("exact K3 DSpark geometry is wrong");
  }
  const at::Tensor input = deterministic_bf16({2, 8}, 11);
  const at::Tensor norm = deterministic_bf16({8}, 12) +
                          at::ones({8}, at::TensorOptions().dtype(at::kBFloat16));
  const at::Tensor promoted = input.to(at::kFloat);
  const at::Tensor expected =
      (promoted * at::rsqrt(at::mean(promoted.pow(2), {-1}, true) + 1.0e-5))
          .to(at::kBFloat16) *
      norm;
  require_close(dspark_rms_norm_bf16(input, norm, 1.0e-5), expected,
                "BF16 RMSNorm", 0.0);
  require_invalid(
      [&] { (void)dspark_rms_norm_bf16(input.to(at::kFloat), norm, 1.0e-5); },
      "fp32 RMSNorm input");
}

void test_adjacent_pair_yarn() {
  const DSparkShape shape = DSparkShape::small_canary();
  const at::Tensor inverse =
      dspark_yarn_inverse_frequencies(shape, at::Device(at::kCPU));
  if (inverse.scalar_type() != at::kFloat || inverse.sizes() != at::IntArrayRef({2}) ||
      !inverse.isfinite().all().item<bool>()) {
    throw std::runtime_error("YaRN inverse frequencies violate their contract");
  }
  const at::Tensor value =
      at::tensor({1.0F, 0.0F, 0.0F, 1.0F},
                 at::TensorOptions().dtype(at::kFloat))
          .view({1, 1, 4})
          .to(at::kBFloat16);
  const at::Tensor positions =
      at::tensor({1}, at::TensorOptions().dtype(at::kLong));
  const at::Tensor actual =
      dspark_apply_yarn_rotary_bf16(value, positions, shape).to(at::kFloat);
  const float first_frequency = inverse[0].item<float>();
  const float second_frequency = inverse[1].item<float>();
  const at::Tensor expected =
      at::tensor({std::cos(first_frequency), std::sin(first_frequency),
                  -std::sin(second_frequency), std::cos(second_frequency)},
                 at::TensorOptions().dtype(at::kFloat))
          .view({1, 1, 4})
          .to(at::kBFloat16)
          .to(at::kFloat);
  require_close(actual, expected, "adjacent-pair YaRN", 0.0);
}

void test_mlp_mla_and_decoder() {
  const DSparkShape shape = DSparkShape::small_canary();
  const DSparkMlaWeights attention = mla_weights(shape);
  const DSparkMlpWeights mlp = mlp_weights(shape);
  const at::Tensor hidden = deterministic_bf16({3, shape.hidden_size}, 13);
  const at::Tensor positions =
      at::tensor({2, 3, 4}, at::TensorOptions().dtype(at::kLong));
  const DSparkLatentContext context{
      .latent = deterministic_bf16({2, shape.kv_lora_rank}, 14),
      .positional = deterministic_bf16({2, shape.qk_rope_head_dim}, 15),
  };

  const at::Tensor expected_mlp =
      reference_linear(
          at::silu(reference_linear(hidden, mlp.gate)) *
              reference_linear(hidden, mlp.up),
          mlp.down);
  require_close(run_dspark_mlp(hidden, mlp, shape), expected_mlp,
                "DSpark SiLU MLP", 0.0);

  const DSparkMlaOutput expected_attention =
      reference_mla(hidden, positions, context, attention, shape);
  const DSparkMlaOutput actual_attention =
      run_dspark_mla(hidden, positions, context, attention, shape);
  require_close(actual_attention.hidden, expected_attention.hidden,
                "latent MLA output", 0.0);
  require_close(actual_attention.query_context.latent,
                expected_attention.query_context.latent,
                "latent MLA staged latent", 0.0);
  require_close(actual_attention.query_context.positional,
                expected_attention.query_context.positional,
                "latent MLA staged positional", 0.0);
  if (context.latent.size(0) != 2 || context.positional.size(0) != 2) {
    throw std::runtime_error("MLA arithmetic mutated committed context");
  }
  const DSparkLatentContext empty_context{
      .latent = at::empty({0, shape.kv_lora_rank},
                          at::TensorOptions().dtype(at::kBFloat16)),
      .positional = at::empty({0, shape.qk_rope_head_dim},
                              at::TensorOptions().dtype(at::kBFloat16)),
  };
  const at::Tensor initial_positions =
      at::tensor({0, 1, 2}, at::TensorOptions().dtype(at::kLong));
  const DSparkMlaOutput initial_expected =
      reference_mla(hidden, initial_positions, empty_context, attention, shape);
  const DSparkMlaOutput initial_actual =
      run_dspark_mla(hidden, initial_positions, empty_context, attention, shape);
  require_close(initial_actual.hidden, initial_expected.hidden,
                "empty-prefix latent MLA", 0.0);

  const DSparkDecoderWeights decoder{
      .input_norm = deterministic_bf16({shape.hidden_size}, 16) +
                    at::ones({shape.hidden_size},
                             at::TensorOptions().dtype(at::kBFloat16)),
      .attention = attention,
      .post_attention_norm =
          deterministic_bf16({shape.hidden_size}, 17) +
          at::ones({shape.hidden_size},
                   at::TensorOptions().dtype(at::kBFloat16)),
      .mlp = mlp,
  };
  const at::Tensor prior_residual =
      deterministic_bf16({3, shape.hidden_size}, 18);
  const at::Tensor first_residual = prior_residual + hidden;
  const at::Tensor normalized = dspark_rms_norm_bf16(
      first_residual, decoder.input_norm, shape.rms_epsilon);
  const DSparkMlaOutput reference_attention =
      reference_mla(normalized, positions, context, attention, shape);
  const at::Tensor final_residual = first_residual + reference_attention.hidden;
  const at::Tensor expected_hidden = run_dspark_mlp(
      dspark_rms_norm_bf16(final_residual, decoder.post_attention_norm,
                           shape.rms_epsilon),
      mlp, shape);
  const auto actual = run_dspark_decoder_layer(
      hidden, prior_residual, positions, context, decoder, shape);
  require_close(actual.residual, final_residual, "decoder residual", 0.0);
  require_close(actual.hidden, expected_hidden, "decoder hidden", 0.0);

  require_invalid(
      [&] {
        (void)run_dspark_mla(hidden.to(at::kFloat), positions, context,
                             attention, shape);
      },
      "fp32 MLA input");
  require_invalid(
      [&] {
        const at::Tensor bad_positions =
            at::tensor({2, 3}, at::TensorOptions().dtype(at::kLong));
        (void)run_dspark_mla(hidden, bad_positions, context, attention, shape);
      },
      "mismatched MLA positions");
  require_invalid(
      [&] {
        const at::Tensor wrong_boundary =
            at::tensor({3, 4, 5}, at::TensorOptions().dtype(at::kLong));
        (void)run_dspark_mla(hidden, wrong_boundary, context, attention, shape);
      },
      "non-boundary MLA positions");
}

}  // namespace

int main() {
  try {
    const c10::InferenceMode inference_guard;
    test_shape_and_rms();
    test_adjacent_pair_yarn();
    test_mlp_mla_and_decoder();
    std::cout << "provider_dspark.synthetic=PASS\n";
    std::cout << "provider_dspark.authority=PROPOSAL_ONLY\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "provider_dspark.synthetic=FAIL: " << error.what() << '\n';
    return 1;
  }
}
