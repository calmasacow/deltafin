#include "provider_kda_batch.h"

#include <ATen/ops/_weight_int8pack_mm.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/sigmoid.h>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace deltafin::provider_internal {
namespace {

constexpr std::int64_t kK3Hidden = 7168;
constexpr std::int64_t kK3Heads = 96;
constexpr std::int64_t kK3Projection = 96 * 128;
constexpr std::int64_t kK3FeatureA = 128;
constexpr std::int64_t kCanaryHidden = 32;
constexpr std::int64_t kCanaryHeads = 32;
constexpr std::int64_t kCanaryProjection = 32 * 32;
constexpr std::int64_t kCanaryFeatureA = 32;
constexpr double kRmsEpsilon = 1.0e-5;

void require_projection(const KdaProjection& projection,
                        const at::Device& device,
                        const std::int64_t rows,
                        const std::int64_t columns,
                        const char* name) {
  if (projection.original_bf16.defined()) {
    if (projection.weight.defined() || projection.scale.defined() ||
        !original_bf16_matrix_matches(
            projection.original_bf16, device,
            static_cast<std::size_t>(rows),
            static_cast<std::size_t>(columns))) {
      throw std::invalid_argument(std::string("KDA batch ") + name +
                                  " has an invalid original-BF16 carrier");
    }
    return;
  }
  if (!projection.scale.defined()) {
    if (!projection.weight.defined() ||
        projection.weight.device() != device ||
        projection.weight.scalar_type() != at::kFloat ||
        projection.weight.sizes() != at::IntArrayRef({rows, columns}) ||
        !projection.weight.is_contiguous()) {
      throw std::invalid_argument(std::string("KDA batch ") + name +
                                  " violates its dense projection contract");
    }
    return;
  }
  if (!projection.weight.defined() || !projection.scale.defined() ||
      projection.weight.device() != device ||
      projection.scale.device() != device ||
      projection.weight.scalar_type() != at::kChar ||
      projection.scale.scalar_type() != at::kFloat ||
      projection.weight.sizes() != at::IntArrayRef({rows, columns}) ||
      projection.scale.sizes() != at::IntArrayRef({rows}) ||
      !projection.weight.is_contiguous() ||
      !projection.scale.is_contiguous()) {
    throw std::invalid_argument(std::string("KDA batch ") + name +
                                " violates its packed projection contract");
  }
}

template <std::size_t Size>
std::optional<KdaProjection> adjacent_projection_bundle(
    const std::array<const KdaProjection*, Size>& parts,
    const std::int64_t columns) {
  static_assert(Size != 0);
  if (columns <= 0 || parts.front() == nullptr) {
    return std::nullopt;
  }
  if (parts.front()->original_bf16.defined()) {
    std::array<const OriginalBf16Matrix*, Size> matrices = {};
    for (std::size_t index = 0; index < Size; ++index) {
      if (parts[index] == nullptr || parts[index]->weight.defined() ||
          parts[index]->scale.defined() ||
          !parts[index]->original_bf16.defined()) {
        return std::nullopt;
      }
      matrices[index] = &parts[index]->original_bf16;
    }
    auto combined = adjacent_original_bf16_matrices(matrices);
    if (!combined.has_value() ||
        combined->columns != static_cast<std::size_t>(columns)) {
      return std::nullopt;
    }
    return KdaProjection{at::Tensor(), at::Tensor(), std::move(*combined)};
  }
  if (!parts.front()->scale.defined()) {
    // The established dense prompt path uses separate T-wide linears. Merely
    // observing adjacent template-arena views is not a numeric/performance
    // admission for a wider-N fused GEMM, whose backend schedule may differ.
    return std::nullopt;
  }
  std::int64_t rows = 0;
  std::int64_t next_weight_offset = 0;
  std::int64_t next_scale_offset = 0;
  for (std::size_t index = 0; index < Size; ++index) {
    const KdaProjection& part = *parts[index];
    if (index == 0) {
      next_weight_offset = part.weight.storage_offset();
      next_scale_offset = part.scale.storage_offset();
    } else if (!parts.front()->weight.is_alias_of(part.weight) ||
               !parts.front()->scale.is_alias_of(part.scale)) {
      return std::nullopt;
    }
    if (part.weight.storage_offset() != next_weight_offset ||
        part.scale.storage_offset() != next_scale_offset ||
        part.weight.numel() >
            std::numeric_limits<std::int64_t>::max() - next_weight_offset ||
        part.scale.numel() >
            std::numeric_limits<std::int64_t>::max() - next_scale_offset ||
        part.weight.size(0) >
            std::numeric_limits<std::int64_t>::max() - rows) {
      return std::nullopt;
    }
    next_weight_offset += part.weight.numel();
    next_scale_offset += part.scale.numel();
    rows += part.weight.size(0);
  }
  return KdaProjection{
      parts.front()->weight.as_strided(
          {rows, columns}, {columns, 1},
          parts.front()->weight.storage_offset()),
      parts.front()->scale.as_strided(
          {rows}, {1}, parts.front()->scale.storage_offset()),
  };
}

at::Tensor batch_linear(const at::Tensor& input,
                        const KdaProjection& projection) {
  if (projection.original_bf16.defined()) {
    return original_bf16_linear(input, projection.original_bf16);
  }
  if (!projection.scale.defined()) {
    return at::linear(input, projection.weight, std::nullopt);
  }
  return at::_weight_int8pack_mm(input, projection.weight,
                                 projection.scale);
}

}  // namespace

KdaBatchInputProjections kda_project_inputs_batch(
    const at::Tensor& hidden_rows, const KdaWeights& weights,
    const bool exact_k3) {
  const std::int64_t hidden = exact_k3 ? kK3Hidden : kCanaryHidden;
  const std::int64_t projection =
      exact_k3 ? kK3Projection : kCanaryProjection;
  if (!hidden_rows.defined() || hidden_rows.dim() != 2 ||
      hidden_rows.size(0) < 1 ||
      hidden_rows.size(0) >
          static_cast<std::int64_t>(kKdaBatchMaximumPositions) ||
      hidden_rows.size(1) != hidden ||
      hidden_rows.scalar_type() != at::kFloat ||
      !hidden_rows.is_contiguous() || hidden_rows.device().is_meta()) {
    throw std::invalid_argument(
        "KDA batch input must be contiguous fp32 [1..64,hidden]");
  }
  const at::Device device = hidden_rows.device();
  require_projection(weights.query_projection, device, projection, hidden,
                     "query projection");
  require_projection(weights.key_projection, device, projection, hidden,
                     "key projection");
  require_projection(weights.value_projection, device, projection, hidden,
                     "value projection");

  KdaBatchInputProjections result;
  result.positions = static_cast<std::uint32_t>(hidden_rows.size(0));
  result.established_separate_rowwise_dispatches = result.positions * 3;
  // Match the established public prompt path exactly: three independent
  // T-wide Q/K/V linears. Physical adjacency alone is not a qualification for
  // a wider-N fused GEMM because provider scheduling/rounding may change.
  result.query = batch_linear(hidden_rows, weights.query_projection);
  result.key = batch_linear(hidden_rows, weights.key_projection);
  result.value = batch_linear(hidden_rows, weights.value_projection);
  result.path = KdaBatchProjectionPath::Separate;
  result.provider_dispatches = 3;
  result.equivalent_rowwise_dispatches =
      result.positions * result.provider_dispatches;
  return result;
}

KdaBatchDependentProjections kda_project_dependent_batch(
    const at::Tensor& hidden_rows, const KdaWeights& weights,
    const bool exact_k3) {
  const std::int64_t hidden = exact_k3 ? kK3Hidden : kCanaryHidden;
  const std::int64_t heads = exact_k3 ? kK3Heads : kCanaryHeads;
  const std::int64_t projection =
      exact_k3 ? kK3Projection : kCanaryProjection;
  const std::int64_t feature_a =
      exact_k3 ? kK3FeatureA : kCanaryFeatureA;
  if (!hidden_rows.defined() || hidden_rows.dim() != 2 ||
      hidden_rows.size(0) < 1 ||
      hidden_rows.size(0) >
          static_cast<std::int64_t>(kKdaBatchMaximumPositions) ||
      hidden_rows.size(1) != hidden ||
      hidden_rows.scalar_type() != at::kFloat ||
      !hidden_rows.is_contiguous() || hidden_rows.device().is_meta()) {
    throw std::invalid_argument(
        "KDA dependent input must be contiguous fp32 [1..64,hidden]");
  }
  const at::Device device = hidden_rows.device();
  require_projection(weights.feature_a_projection, device, feature_a, hidden,
                     "feature-A projection");
  require_projection(weights.feature_b_projection, device, projection,
                     feature_a, "feature-B projection");
  require_projection(weights.beta_projection, device, heads, hidden,
                     "beta projection");

  KdaBatchDependentProjections result;
  result.positions = static_cast<std::uint32_t>(hidden_rows.size(0));
  result.feature_a =
      batch_linear(hidden_rows, weights.feature_a_projection);
  result.feature_b =
      batch_linear(result.feature_a, weights.feature_b_projection);
  result.beta = batch_linear(hidden_rows, weights.beta_projection);
  result.dependent_provider_dispatches = 3;
  result.dependent_equivalent_rowwise_dispatches = result.positions * 3;
  return result;
}

KdaBatchOutputProjection kda_finish_output_batch(
    const at::Tensor& hidden_rows,
    const at::Tensor& recurrent_output_rows, const KdaWeights& weights,
    const bool exact_k3) {
  const std::int64_t hidden = exact_k3 ? kK3Hidden : kCanaryHidden;
  const std::int64_t heads = exact_k3 ? kK3Heads : kCanaryHeads;
  const std::int64_t projection =
      exact_k3 ? kK3Projection : kCanaryProjection;
  const std::int64_t head_width = projection / heads;
  if (!hidden_rows.defined() || hidden_rows.dim() != 2 ||
      hidden_rows.size(0) < 1 ||
      hidden_rows.size(0) >
          static_cast<std::int64_t>(kKdaBatchMaximumPositions) ||
      hidden_rows.size(1) != hidden ||
      hidden_rows.scalar_type() != at::kFloat ||
      !hidden_rows.is_contiguous() || hidden_rows.device().is_meta() ||
      !recurrent_output_rows.defined() ||
      recurrent_output_rows.dim() != 2 ||
      recurrent_output_rows.size(0) != hidden_rows.size(0) ||
      recurrent_output_rows.size(1) != projection ||
      recurrent_output_rows.scalar_type() != at::kFloat ||
      !recurrent_output_rows.is_contiguous() ||
      recurrent_output_rows.device() != hidden_rows.device()) {
    throw std::invalid_argument(
        "KDA batch finish requires matching contiguous fp32 hidden/recurrent rows");
  }
  require_projection(weights.recurrent_gate_projection,
                     hidden_rows.device(), projection, hidden,
                     "output-gate projection");
  require_projection(weights.output_projection,
                     hidden_rows.device(), hidden, projection,
                     "output projection");
  if (!weights.output_norm.defined() ||
      weights.output_norm.device() != hidden_rows.device() ||
      weights.output_norm.scalar_type() != at::kFloat ||
      weights.output_norm.sizes() != at::IntArrayRef({head_width})) {
    throw std::invalid_argument(
        "KDA batch output norm violates its fp32 head-width contract");
  }
  const at::Tensor output_gate = batch_linear(
      hidden_rows, weights.recurrent_gate_projection);
  at::Tensor output = recurrent_output_rows.view(
      {hidden_rows.size(0), heads, head_width});
  const at::Tensor variance = output.pow(2).mean(-1, true);
  output = output * at::rsqrt(variance + kRmsEpsilon);
  output = output * weights.output_norm;
  output = output * at::sigmoid(
      output_gate.view({hidden_rows.size(0), heads, head_width}));
  KdaBatchOutputProjection result;
  result.output = batch_linear(
      output.reshape({hidden_rows.size(0), projection}),
      weights.output_projection).contiguous();
  result.positions =
      static_cast<std::uint32_t>(recurrent_output_rows.size(0));
  result.provider_dispatches = 2;
  result.equivalent_rowwise_dispatches = result.positions * 2;
  return result;
}

}  // namespace deltafin::provider_internal
