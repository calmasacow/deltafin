#include "provider_target_tape.h"

#include <ATen/ATen.h>
#include <ATen/ops/argmax.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using deltafin::provider_internal::CanonicalExpertBatchT1;
using deltafin::provider_internal::KdaProjection;
using deltafin::provider_internal::KdaWeights;
using deltafin::provider_internal::kMoeRouteTopK;
using deltafin::provider_internal::kTargetKdaLayerCount;
using deltafin::provider_internal::kTargetLayerCount;
using deltafin::provider_internal::kTargetMlaLayerCount;
using deltafin::provider_internal::MlaCache;
using deltafin::provider_internal::MlaLinearEncoding;
using deltafin::provider_internal::MlaLinearWeight;
using deltafin::provider_internal::MlaShape;
using deltafin::provider_internal::MlaWeights;
using deltafin::provider_internal::MoeGeometry;
using deltafin::provider_internal::MoeExpertLayout;
using deltafin::provider_internal::MoeRowInt8Matrix;
using deltafin::provider_internal::MoeRunOptions;
using deltafin::provider_internal::MoeSpineT1;
using deltafin::provider_internal::TargetAttentionKind;
using deltafin::provider_internal::TargetBlockResidual;
using deltafin::provider_internal::TargetDenseWeights;
using deltafin::provider_internal::TargetKdaCache;
using deltafin::provider_internal::TargetLayerBinding;
using deltafin::provider_internal::TargetLayerCacheBinding;
using deltafin::provider_internal::TargetLayerPrepareKind;
using deltafin::provider_internal::TargetPositionBindings;
using deltafin::provider_internal::TargetPositionState;
using deltafin::provider_internal::TargetPositionTape;
using deltafin::provider_internal::TargetResidualWeights;
using deltafin::provider_internal::TargetTailWeights;
using deltafin::provider_internal::TargetTapeContract;

constexpr std::int64_t kHidden = 32;
constexpr std::int64_t kKdaHeads = 32;
constexpr std::int64_t kKdaHeadWidth = 32;
constexpr std::int64_t kKdaProjection = kKdaHeads * kKdaHeadWidth;
constexpr std::int64_t kKdaConvolution = 4;
constexpr std::int64_t kVocabulary = 64;
constexpr MoeGeometry kMoeGeometry{32, 32, 32, 16, 64};

at::Tensor deterministic_tensor(const at::IntArrayRef shape,
                                const std::int64_t seed,
                                const float scale = 0.015625F) {
  std::int64_t count = 1;
  for (const std::int64_t extent : shape) {
    count *= extent;
  }
  at::Tensor tensor = at::empty(shape, at::TensorOptions().dtype(at::kFloat));
  auto *values = tensor.data_ptr<float>();
  for (std::int64_t index = 0; index < count; ++index) {
    const std::int64_t numerator =
        ((index * 17 + seed * 29 + (index / 7) * 5) % 61) - 30;
    values[index] = static_cast<float>(numerator) * scale;
  }
  return tensor;
}

MoeRowInt8Matrix row_int8(const std::int64_t rows, const std::int64_t columns,
                          const std::int64_t seed,
                          const bool dense_fallback = true) {
  at::Tensor quantized =
      at::empty({rows, columns}, at::TensorOptions().dtype(at::kChar));
  at::Tensor scales = at::empty({rows}, at::TensorOptions().dtype(at::kFloat));
  auto q = quantized.accessor<std::int8_t, 2>();
  auto s = scales.accessor<float, 1>();
  for (std::int64_t row = 0; row < rows; ++row) {
    s[row] = static_cast<float>(1 + ((row + seed) % 3)) / 256.0F;
    for (std::int64_t column = 0; column < columns; ++column) {
      q[row][column] = static_cast<std::int8_t>(
          ((row * 3 + column * 5 + seed * 7) % 15) - 7);
    }
  }
  at::Tensor dense;
  if (dense_fallback) {
    dense = (quantized.to(at::kFloat) * scales.unsqueeze(1)).contiguous();
  }
  return {std::move(quantized), std::move(scales), std::move(dense), {}};
}

KdaProjection kda_projection(const std::int64_t rows,
                             const std::int64_t columns,
                             const std::int64_t seed) {
  MoeRowInt8Matrix matrix = row_int8(rows, columns, seed, false);
  return {std::move(matrix.quantized), std::move(matrix.row_scales)};
}

KdaWeights make_kda_weights() {
  const at::Tensor convolution = deterministic_tensor(
      {kKdaProjection * 3, 1, kKdaConvolution}, 101, 0.00390625F);
  const at::Tensor a_log =
      at::full({kKdaHeadWidth}, -2.0F, at::TensorOptions().dtype(at::kFloat));
  return KdaWeights{
      .a_log = a_log,
      .dt_bias = deterministic_tensor({kKdaProjection}, 102, 0.001953125F),
      .query_convolution = convolution.narrow(0, 0, kKdaProjection),
      .key_convolution = convolution.narrow(0, kKdaProjection, kKdaProjection),
      .value_convolution =
          convolution.narrow(0, kKdaProjection * 2, kKdaProjection),
      .output_norm = deterministic_tensor({kKdaHeadWidth}, 103) + 1.0F,
      .query_projection = kda_projection(kKdaProjection, kHidden, 104),
      .key_projection = kda_projection(kKdaProjection, kHidden, 105),
      .value_projection = kda_projection(kKdaProjection, kHidden, 106),
      .recurrent_gate_projection = kda_projection(kKdaProjection, kHidden, 107),
      .feature_a_projection = kda_projection(kKdaHeadWidth, kHidden, 108),
      .feature_b_projection =
          kda_projection(kKdaProjection, kKdaHeadWidth, 109),
      .beta_projection = kda_projection(kKdaHeads, kHidden, 110),
      .output_projection = kda_projection(kHidden, kKdaProjection, 111),
  };
}

MlaLinearWeight mla_weight(const std::int64_t rows, const std::int64_t columns,
                           const std::int64_t seed) {
  MoeRowInt8Matrix matrix = row_int8(rows, columns, seed, false);
  return MlaLinearWeight{
      .encoding = MlaLinearEncoding::RowI8F32Scale,
      .data = std::move(matrix.quantized),
      .row_scale = std::move(matrix.row_scales),
      // Explicit: GCC errors on omitted members under -Werror; Clang does not.
      .original_bf16 = {},
  };
}

MlaWeights make_mla_weights(const MlaShape &shape) {
  const std::int64_t query_width = shape.num_heads * shape.query_head_dim();
  const std::int64_t value_width = shape.num_heads * shape.value_head_dim;
  return MlaWeights{
      .query_a = mla_weight(shape.q_lora_rank, shape.hidden_size, 201),
      .query_a_norm = deterministic_tensor({shape.q_lora_rank}, 202) + 1.0F,
      .query_b = mla_weight(query_width, shape.q_lora_rank, 203),
      .key_value_a = mla_weight(shape.kv_lora_rank + shape.qk_rope_head_dim,
                                shape.hidden_size, 204),
      .key_value_a_norm =
          deterministic_tensor({shape.kv_lora_rank}, 205) + 1.0F,
      .key_value_b = mla_weight(
          shape.num_heads * (shape.qk_nope_head_dim + shape.value_head_dim),
          shape.kv_lora_rank, 206),
      .output_gate = mla_weight(value_width, shape.hidden_size, 207),
      .output = mla_weight(shape.hidden_size, value_width, 208),
  };
}

TargetResidualWeights make_residual_weights() {
  TargetResidualWeights weights{
      deterministic_tensor({kHidden}, 301) + 1.0F,
      deterministic_tensor({kHidden}, 302) + 1.0F,
      deterministic_tensor({1, kHidden}, 303),
      deterministic_tensor({kHidden}, 304) + 1.0F,
      deterministic_tensor({kHidden}, 305) + 1.0F,
      deterministic_tensor({1, kHidden}, 306),
  };
  return deltafin::provider_internal::precompute_target_residual_score_weights(
      std::move(weights));
}

TargetDenseWeights make_dense_weights() {
  return TargetDenseWeights{
      row_int8(kHidden, kHidden, 401),
      row_int8(kHidden, kHidden, 402),
      row_int8(kHidden, kHidden, 403),
      false,
  };
}

MoeSpineT1 make_moe_spine(const std::uint32_t layer_index) {
  return MoeSpineT1{
      .layer_index = layer_index,
      .generation = 1000 + layer_index,
      .geometry = kMoeGeometry,
      .packed_int8_qualified = false,
      .router = row_int8(kMoeGeometry.experts, kHidden, 501),
      .router_correction_bias =
          at::arange(kMoeGeometry.experts,
                     at::TensorOptions().dtype(at::kFloat)) *
          (1.0F / 4096.0F),
      .routed_down = row_int8(kMoeGeometry.routed_hidden, kHidden, 502),
      .routed_norm =
          deterministic_tensor({kMoeGeometry.routed_hidden}, 503) + 1.0F,
      .routed_up = row_int8(kHidden, kMoeGeometry.routed_hidden, 504),
      .shared_gate = row_int8(kMoeGeometry.shared_intermediate, kHidden, 505),
      .shared_up = row_int8(kMoeGeometry.shared_intermediate, kHidden, 506),
      .shared_down = row_int8(kHidden, kMoeGeometry.shared_intermediate, 507),
      // Explicit: GCC errors on omitted members under -Werror; Clang does not.
      .shared_gate_up = {},
  };
}

struct MatrixLayout {
  std::size_t packed_offset = 0;
  std::size_t scale_offset = 0;
  std::size_t rows = 0;
  std::size_t columns = 0;
};

std::array<MatrixLayout, 3> expert_layouts() {
  const auto packed = [](const std::size_t rows, const std::size_t columns) {
    return rows * columns / 2;
  };
  const auto scales = [](const std::size_t rows, const std::size_t columns) {
    return rows * columns / 32;
  };
  const std::size_t p13 =
      packed(kMoeGeometry.intermediate, kMoeGeometry.routed_hidden);
  const std::size_t s13 =
      scales(kMoeGeometry.intermediate, kMoeGeometry.routed_hidden);
  const std::size_t p2 =
      packed(kMoeGeometry.routed_hidden, kMoeGeometry.intermediate);
  const std::size_t s2 =
      scales(kMoeGeometry.routed_hidden, kMoeGeometry.intermediate);
  return {{{0, p13, kMoeGeometry.intermediate, kMoeGeometry.routed_hidden},
           {p13 + s13, p13 + s13 + p2, kMoeGeometry.routed_hidden,
            kMoeGeometry.intermediate},
           {p13 + s13 + p2 + s2, p13 + s13 + p2 + s2 + p13,
            kMoeGeometry.intermediate, kMoeGeometry.routed_hidden}}};
}

void encode_expert(std::span<std::uint8_t> destination,
                   const std::size_t expert) {
  const auto layouts = expert_layouts();
  for (std::size_t matrix = 0; matrix < layouts.size(); ++matrix) {
    const MatrixLayout &layout = layouts[matrix];
    for (std::size_t row = 0; row < layout.rows; ++row) {
      for (std::size_t column = 0; column < layout.columns; column += 2) {
        const auto code = [&](const std::size_t offset) {
          std::uint8_t value = static_cast<std::uint8_t>(
              1 +
              ((expert * 3 + matrix * 5 + row * 7 + (column + offset) * 11) %
               4));
          if (((expert + matrix + row + column + offset) & 1U) != 0) {
            value = static_cast<std::uint8_t>(value | 8U);
          }
          return value;
        };
        destination[layout.packed_offset + row * (layout.columns / 2) +
                    column / 2] =
            static_cast<std::uint8_t>(code(0) | (code(1) << 4));
      }
      for (std::size_t group = 0; group < layout.columns / 32; ++group) {
        destination[layout.scale_offset + row * (layout.columns / 32) + group] =
            static_cast<std::uint8_t>(123 +
                                      ((expert + matrix + row + group) % 4));
      }
    }
  }
}

struct Experts {
  std::array<std::uint16_t, kMoeRouteTopK> ids{};
  std::vector<std::uint8_t> bytes;

  Experts() {
    const std::size_t span =
        static_cast<std::size_t>(kMoeGeometry.expert_span_bytes());
    bytes.resize(kMoeRouteTopK * span);
    for (std::size_t expert = 0; expert < kMoeRouteTopK; ++expert) {
      ids[expert] = static_cast<std::uint16_t>(expert);
      encode_expert(std::span<std::uint8_t>(bytes).subspan(expert * span, span),
                    expert);
    }
  }

  CanonicalExpertBatchT1 view() const {
    return CanonicalExpertBatchT1{
        std::span<const std::uint16_t>(ids),
        std::span<const std::uint8_t>(bytes),
        MoeExpertLayout::RawV1,
        kMoeGeometry.expert_span_bytes(),
    };
  }
};

struct Fixture {
  MlaShape mla_shape = MlaShape::small_canary();
  KdaWeights kda_weights = make_kda_weights();
  MlaWeights mla_weights = make_mla_weights(mla_shape);
  TargetResidualWeights residual_weights = make_residual_weights();
  TargetDenseWeights dense_weights = make_dense_weights();
  MoeRowInt8Matrix embedding = row_int8(kVocabulary, kHidden, 601, false);
  TargetTailWeights tail{
      deterministic_tensor({kHidden}, 602) + 1.0F,
      deterministic_tensor({1, kHidden}, 603),
      deterministic_tensor({kHidden}, 604) + 1.0F,
      row_int8(kVocabulary, kHidden, 605),
      false,
  };
  std::array<MoeSpineT1, kTargetLayerCount> moe_spines{};
  std::vector<std::unique_ptr<TargetKdaCache>> kda_caches;
  std::vector<std::unique_ptr<MlaCache>> mla_caches;
  std::array<TargetLayerBinding, kTargetLayerCount> layers{};
  std::array<TargetLayerCacheBinding, kTargetLayerCount> cache_bindings{};

  Fixture() {
    tail = deltafin::provider_internal::precompute_target_tail_score_weight(
        std::move(tail));
    kda_caches.reserve(kTargetKdaLayerCount);
    mla_caches.reserve(kTargetMlaLayerCount);
    for (std::uint32_t layer = 0; layer < kTargetLayerCount; ++layer) {
      TargetLayerBinding binding;
      TargetLayerCacheBinding cache_binding;
      binding.layer_index = layer;
      cache_binding.layer_index = layer;
      binding.residual = &residual_weights;
      if (deltafin::provider_internal::target_layer_uses_mla(layer)) {
        binding.attention_kind = TargetAttentionKind::Mla;
        cache_binding.attention_kind = TargetAttentionKind::Mla;
        mla_caches.push_back(std::make_unique<MlaCache>(mla_shape));
        binding.mla_weights = &mla_weights;
        cache_binding.mla_cache = mla_caches.back().get();
      } else {
        binding.attention_kind = TargetAttentionKind::Kda;
        cache_binding.attention_kind = TargetAttentionKind::Kda;
        auto cache = std::make_unique<TargetKdaCache>();
        cache->layer_index = layer;
        cache->state = deltafin::provider_internal::zero_small_kda_canary_state(
            at::Device(at::kCPU));
        kda_caches.push_back(std::move(cache));
        binding.kda_weights = &kda_weights;
        cache_binding.kda_cache = kda_caches.back().get();
      }
      if (layer == 0) {
        binding.dense = &dense_weights;
      } else {
        moe_spines[layer] = make_moe_spine(layer);
        binding.moe = &moe_spines[layer];
      }
      layers[layer] = binding;
      cache_bindings[layer] = cache_binding;
    }
  }

  TargetPositionBindings bindings() const {
    return TargetPositionBindings{
        .contract = TargetTapeContract::SyntheticK3Schedule,
        .caches = std::span<const TargetLayerCacheBinding>(cache_bindings),
        .tail = &tail,
    };
  }

  at::Tensor input_hidden(const std::uint32_t token_id) const {
    return deltafin::provider_internal::target_embedding_row(token_id,
                                                             embedding, false);
  }
};

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
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error(std::string(name) + " did not fail closed");
}

void require_unpublished(const Fixture &fixture) {
  for (const auto &cache : fixture.kda_caches) {
    if (cache->version != 0) {
      throw std::runtime_error("cancelled target position published KDA state");
    }
  }
  for (const auto &cache : fixture.mla_caches) {
    if (cache->version() != 0 || cache->length() != 0 ||
        cache->has_pending_prepare()) {
      throw std::runtime_error("cancelled target position published MLA state");
    }
  }
}

void run_all_routed_layers(TargetPositionTape &tape, Fixture &fixture,
                           const Experts &experts) {
  const MoeRunOptions options{
      .expert_backend = deltafin::provider_internal::MoeExpertBackend::CpuMxfp4,
      .cpu_threads = 1,
      .metal_shader_path = {},
      .cuda_cache = nullptr,
  };
  const auto dense = tape.prepare_layer(fixture.layers[0]);
  if (dense.kind != TargetLayerPrepareKind::DenseCompleted ||
      tape.next_layer_index() != 1) {
    throw std::runtime_error("target tape did not complete dense layer zero");
  }
  for (std::uint32_t layer = 1; layer < kTargetLayerCount; ++layer) {
    TargetLayerBinding streamed = fixture.layers[layer];
    MoeSpineT1 transient_spine = *streamed.moe;
    streamed.moe = &transient_spine;
    const auto prepared = tape.prepare_layer(streamed);
    // The tape must retain only this current spine's tensor handles across the
    // expert read, never a caller-owned pointer or all 93 layer weights.
    transient_spine = MoeSpineT1{};
    if (prepared.kind != TargetLayerPrepareKind::ExpertsRequired) {
      throw std::runtime_error("routed target layer did not request experts");
    }
    const auto &route = prepared.route;
    if (route.layer_index != layer || route.spine_generation != 1000 + layer) {
      throw std::runtime_error("target tape returned an out-of-order route");
    }
    tape.finish_moe_layer(route.layer_index, route.spine_generation,
                          experts.view(), options);
  }
}

std::uint32_t run_reference(Fixture &fixture, const Experts &experts,
                            const std::uint32_t token_id) {
  at::Tensor hidden = deltafin::provider_internal::target_embedding_row(
      token_id, fixture.embedding, false);
  TargetBlockResidual residual =
      deltafin::provider_internal::empty_target_block_residual(
          at::Device(at::kCPU), kHidden);
  std::size_t kda = 0;
  std::size_t mla = 0;
  const MoeRunOptions options{
      .expert_backend = deltafin::provider_internal::MoeExpertBackend::CpuMxfp4,
      .cpu_threads = 1,
      .metal_shader_path = {},
      .cuda_cache = nullptr,
  };
  for (std::uint32_t layer = 0; layer < kTargetLayerCount; ++layer) {
    const TargetLayerBinding &binding = fixture.layers[layer];
    const TargetLayerCacheBinding &cache = fixture.cache_bindings[layer];
    auto attention = deltafin::provider_internal::prepare_target_attention(
        hidden, residual, fixture.residual_weights, layer, false);
    at::Tensor attention_output;
    if (binding.attention_kind == TargetAttentionKind::Kda) {
      auto result = deltafin::provider_internal::kda_decode_one(
          attention.normalized, fixture.kda_weights, cache.kda_cache->state,
          false);
      attention_output = result.output;
      cache.kda_cache->state = std::move(result.next_state);
      ++cache.kda_cache->version;
      ++kda;
    } else {
      auto result = deltafin::provider_internal::prepare_mla_decode(
          attention.normalized.view({1, 1, kHidden}), fixture.mla_weights,
          *cache.mla_cache, true);
      attention_output = result.output.view({1, kHidden}).contiguous();
      deltafin::provider_internal::commit_mla_decode(*cache.mla_cache, result);
      ++mla;
    }
    auto mlp = deltafin::provider_internal::prepare_target_mlp(
        attention, attention_output, fixture.residual_weights, false);
    at::Tensor mlp_output =
        layer == 0
            ? deltafin::provider_internal::run_target_dense(
                  mlp.normalized, fixture.dense_weights, false)
            : deltafin::provider_internal::run_moe_t1(mlp.normalized,
                                                      fixture.moe_spines[layer],
                                                      experts.view(), options);
    hidden = deltafin::provider_internal::complete_target_layer(mlp, mlp_output,
                                                                false);
    residual.anchors = std::move(mlp.next_anchors);
  }
  if (kda != kTargetKdaLayerCount || mla != kTargetMlaLayerCount ||
      residual.anchors.size(1) != 8) {
    throw std::runtime_error("reference target schedule counts changed");
  }
  const at::Tensor logits = deltafin::provider_internal::finish_target_tail(
      hidden, residual, fixture.tail, false);
  return static_cast<std::uint32_t>(
      at::argmax(logits, -1, false).item<std::int64_t>());
}

void require_cache_equal(const Fixture &actual, const Fixture &expected) {
  if (actual.kda_caches.size() != expected.kda_caches.size() ||
      actual.mla_caches.size() != expected.mla_caches.size()) {
    throw std::runtime_error("target fixture cache counts disagree");
  }
  for (std::size_t index = 0; index < actual.kda_caches.size(); ++index) {
    const TargetKdaCache &left = *actual.kda_caches[index];
    const TargetKdaCache &right = *expected.kda_caches[index];
    if (left.version != right.version || left.version != 1) {
      throw std::runtime_error("committed KDA versions disagree");
    }
    require_equal(left.state.query_convolution, right.state.query_convolution,
                  "KDA query cache");
    require_equal(left.state.key_convolution, right.state.key_convolution,
                  "KDA key cache");
    require_equal(left.state.value_convolution, right.state.value_convolution,
                  "KDA value cache");
    require_equal(left.state.recurrent, right.state.recurrent,
                  "KDA recurrent cache");
  }
  for (std::size_t index = 0; index < actual.mla_caches.size(); ++index) {
    const MlaCache &left = *actual.mla_caches[index];
    const MlaCache &right = *expected.mla_caches[index];
    if (left.version() != right.version() || left.version() != 1 ||
        left.length() != right.length() || left.length() != 1 ||
        left.has_pending_prepare()) {
      throw std::runtime_error("committed MLA versions disagree");
    }
    require_equal(left.committed_keys(), right.committed_keys(),
                  "MLA key cache");
    require_equal(left.committed_values(), right.committed_values(),
                  "MLA value cache");
  }
}

void full_commit_test(const Experts &experts) {
  constexpr std::uint32_t token_id = 7;
  Fixture expected;
  const std::uint32_t expected_token =
      run_reference(expected, experts, token_id);
  Fixture actual;
  TargetPositionTape tape(actual.bindings(), actual.input_hidden(token_id));
  run_all_routed_layers(tape, actual, experts);
  if (tape.state() != TargetPositionState::ReadyForTail ||
      tape.next_layer_index() != kTargetLayerCount ||
      tape.staged_kda_count() != kTargetKdaLayerCount ||
      tape.staged_mla_count() != kTargetMlaLayerCount) {
    throw std::runtime_error("target tape did not stage the full K3 schedule");
  }
  for (const auto &cache : actual.kda_caches) {
    if (cache->version != 0) {
      throw std::runtime_error("KDA cache published before the tail");
    }
  }
  for (const auto &cache : actual.mla_caches) {
    if (cache->version() != 0 || cache->length() != 0 ||
        !cache->has_pending_prepare()) {
      throw std::runtime_error("MLA cache published before the tail");
    }
  }
  const std::uint32_t actual_token = tape.finish_greedy();
  if (actual_token != expected_token ||
      tape.state() != TargetPositionState::Committed) {
    throw std::runtime_error("transactional target greedy token disagrees");
  }
  require_cache_equal(actual, expected);
  require_throws([&] { static_cast<void>(tape.finish_greedy()); },
                 "double target commit");
}

void rollback_test(const Experts &experts) {
  Fixture fixture;
  TargetPositionTape tape(fixture.bindings(), fixture.input_hidden(9));
  run_all_routed_layers(tape, fixture, experts);
  tape.cancel();
  if (tape.state() != TargetPositionState::Cancelled) {
    throw std::runtime_error("explicit target cancellation did not finalize");
  }
  require_unpublished(fixture);
}

void order_failure_test(const Experts &experts) {
  Fixture fixture;
  TargetPositionTape tape(fixture.bindings(), fixture.input_hidden(11));
  const MoeRunOptions options{
      .expert_backend = deltafin::provider_internal::MoeExpertBackend::CpuMxfp4,
      .cpu_threads = 1,
      .metal_shader_path = {},
      .cuda_cache = nullptr,
  };
  require_throws(
      [&] { tape.finish_moe_layer(1, 1001, experts.view(), options); },
      "expert finish before route");
  const auto dense = tape.prepare_layer(fixture.layers[0]);
  if (dense.kind != TargetLayerPrepareKind::DenseCompleted) {
    throw std::runtime_error("target order test did not complete layer zero");
  }
  const auto first = tape.prepare_layer(fixture.layers[1]);
  require_throws(
      [&] { static_cast<void>(tape.prepare_layer(fixture.layers[2])); },
      "second route while experts are pending");
  tape.finish_moe_layer(first.route.layer_index, first.route.spine_generation,
                        experts.view(), options);
  const auto second = tape.prepare_layer(fixture.layers[2]);
  require_throws(
      [&] {
        tape.finish_moe_layer(second.route.layer_index,
                              second.route.spine_generation + 1, experts.view(),
                              options);
      },
      "stale target spine generation");
  if (tape.state() != TargetPositionState::Cancelled) {
    throw std::runtime_error("target ordering failure did not cancel");
  }
  require_unpublished(fixture);
}

void plan_validation_test() {
  Fixture fixture;
  auto invalid_caches = fixture.cache_bindings;
  invalid_caches[3].attention_kind = TargetAttentionKind::Kda;
  TargetPositionBindings invalid = fixture.bindings();
  invalid.caches = std::span<const TargetLayerCacheBinding>(invalid_caches);
  require_throws(
      [&] { TargetPositionTape tape(invalid, fixture.input_hidden(1)); },
      "changed K3 attention schedule");

  invalid = fixture.bindings();
  invalid.contract = TargetTapeContract::ExactK3;
  require_throws(
      [&] { TargetPositionTape tape(invalid, fixture.input_hidden(1)); },
      "synthetic geometry under exact K3 contract");

  std::vector<std::unique_ptr<MlaCache>> compact_caches;
  compact_caches.reserve(kTargetMlaLayerCount);
  auto compact_bindings = fixture.cache_bindings;
  for (std::uint32_t layer = 0; layer < kTargetLayerCount; ++layer) {
    if (deltafin::provider_internal::target_layer_uses_mla(layer)) {
      compact_caches.push_back(std::make_unique<MlaCache>(
          MlaShape::k3(),
          deltafin::provider_internal::MlaCacheRepresentation::
              CompactLatentF32));
      compact_bindings[layer].mla_cache = compact_caches.back().get();
    }
  }
  invalid = fixture.bindings();
  invalid.contract = TargetTapeContract::ExactK3;
  invalid.caches =
      std::span<const TargetLayerCacheBinding>(compact_bindings);
  require_throws(
      [&] { TargetPositionTape tape(invalid, at::Tensor()); },
      "exact target position compact MLA cache state");

  invalid_caches = fixture.cache_bindings;
  invalid_caches[1].kda_cache = invalid_caches[0].kda_cache;
  invalid = fixture.bindings();
  invalid.caches = std::span<const TargetLayerCacheBinding>(invalid_caches);
  require_throws(
      [&] { TargetPositionTape tape(invalid, fixture.input_hidden(1)); },
      "duplicate target cache ownership");

  TargetPositionTape streamed(fixture.bindings(), fixture.input_hidden(2));
  auto wrong_layer = fixture.layers[1];
  require_throws(
      [&] { static_cast<void>(streamed.prepare_layer(wrong_layer)); },
      "out-of-order streamed weight binding");
}

} // namespace

int main() {
  try {
    const Experts experts;
    full_commit_test(experts);
    rollback_test(experts);
    order_failure_test(experts);
    plan_validation_test();
    std::cout << "provider_target_tape.schedule=PASS (69 KDA, 24 MLA)\n"
              << "provider_target_tape.greedy=PASS (token only)\n"
              << "provider_target_tape.transaction=PASS\n"
              << "provider_target_tape.rollback=PASS\n"
              << "provider_target_tape.order=PASS\n"
              << "provider_target_tape.streamed_weights=ONE_LAYER\n"
              << "provider_target_tape.embedding=EXACT_ROW_ONLY\n"
              << "provider_target_tape.borrowed_experts=SYNC_ONLY\n"
              << "provider_target_tape.python_runtime=ABSENT\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "provider_target_tape=FAIL: " << error.what() << '\n';
    return 1;
  }
}
