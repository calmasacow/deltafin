#ifndef DELTAFIN_PROVIDER_TARGET_H
#define DELTAFIN_PROVIDER_TARGET_H

#include "provider_moe.h"

#include <ATen/ATen.h>

#include <cstdint>

namespace deltafin::provider_internal {

/*
 * The target-layer residual tape is shared by KDA and MLA layers.  Keeping it
 * in the provider means Rust crosses the ABI only at the two unavoidable MoE
 * boundaries: once to obtain the route, and once after the requested expert
 * bytes have arrived.  Individual RMS, softmax, residual, and dense operators
 * never become host-language calls.
 */
struct TargetResidualWeights {
  at::Tensor input_norm;
  at::Tensor self_attention_res_norm;
  at::Tensor self_attention_res_projection;
  at::Tensor post_attention_norm;
  at::Tensor mlp_res_norm;
  at::Tensor mlp_res_projection;

  /*
   * Optional bind-time products.  The authoritative checkpoint tensors stay
   * above; these cache the exact `norm * projection.squeeze(0)` expression
   * that would otherwise launch once per token.  They are deliberately
   * provider-owned tensors rather than host-side approximations.
   */
  at::Tensor self_attention_score_weight = {};
  at::Tensor mlp_score_weight = {};
};

struct TargetDenseWeights {
  MoeRowInt8Matrix gate;
  MoeRowInt8Matrix up;
  MoeRowInt8Matrix down;
  bool packed_int8_qualified = false;

  /*
   * A row-concatenated gate/up matrix prepared while the layer is bound.
   * gate and up become zero-copy row views into this same allocation, so the
   * qualified one-call token path does not increase steady-state residency.
   */
  MoeRowInt8Matrix gate_up = {};
  bool bundled_gate_up_qualified = false;
  bool bundled_gate_up_enabled = false;
};

/* One forward position owns this state only while it crosses all 93 layers. */
struct TargetBlockResidual {
  at::Tensor anchors;
};

struct TargetAttentionInput {
  at::Tensor normalized;
  at::Tensor prefix_sum;
  at::Tensor next_anchors;
  std::uint32_t layer_index = 0;
};

struct TargetMlpInput {
  at::Tensor normalized;
  /*
   * Exact pre-normalization residual used by the next layer's scheduling-only
   * router predictor.  It is provider-owned and never substitutes for
   * `normalized`, the authoritative route, or any route weight.  Retaining it
   * here avoids recomputing the residual expression and, critically, avoids
   * approximating the Python pilot's input from a later tensor.
   */
  at::Tensor lookahead_source;
  at::Tensor prefix_sum;
  at::Tensor next_anchors;
  std::uint32_t layer_index = 0;
};

/* Live [T,H] residual carriers used only when T>1. Anchor storage is one
 * contiguous [T,A,H] tensor, matching KimiDecoderLayer's public prompt shape
 * and avoiding T independent softmax/RMS submissions. */
struct TargetAttentionRowsInput {
  at::Tensor normalized;
  at::Tensor prefix_sum;
  at::Tensor next_anchors;
  std::uint32_t layer_index = 0;
};

struct TargetMlpRowsInput {
  at::Tensor normalized;
  at::Tensor lookahead_source;
  at::Tensor prefix_sum;
  at::Tensor next_anchors;
  std::uint32_t layer_index = 0;
};

struct TargetTailWeights {
  at::Tensor output_res_norm;
  at::Tensor output_res_projection;
  at::Tensor final_norm;
  MoeRowInt8Matrix language_model_head;
  bool packed_int8_qualified = false;
  at::Tensor output_score_weight = {};
};

/* Bind-time helpers.  They consume their argument so obsolete allocations can
 * be released before generation begins. */
[[nodiscard]] TargetResidualWeights
precompute_target_residual_score_weights(TargetResidualWeights weights);

[[nodiscard]] TargetDenseWeights
bundle_target_dense_gate_up(TargetDenseWeights weights, bool qualified,
                            bool enable_experimental = false);

[[nodiscard]] TargetTailWeights
precompute_target_tail_score_weight(TargetTailWeights weights);

[[nodiscard]] TargetBlockResidual
empty_target_block_residual(const at::Device &device,
                            std::int64_t hidden_size = 7168);

[[nodiscard]] TargetAttentionInput
prepare_target_attention(const at::Tensor &hidden,
                         const TargetBlockResidual &residual,
                         const TargetResidualWeights &weights,
                         std::uint32_t layer_index, bool exact_k3 = true);

[[nodiscard]] TargetMlpInput
prepare_target_mlp(const TargetAttentionInput &prepared,
                   const at::Tensor &attention_output,
                   const TargetResidualWeights &weights, bool exact_k3 = true);

[[nodiscard]] TargetAttentionRowsInput prepare_target_attention_rows(
    const at::Tensor &hidden_rows, const at::Tensor &residual_anchor_rows,
    const TargetResidualWeights &weights, std::uint32_t layer_index,
    bool exact_k3 = true);

[[nodiscard]] TargetMlpRowsInput prepare_target_mlp_rows(
    const TargetAttentionRowsInput &prepared,
    const at::Tensor &attention_output_rows,
    const TargetResidualWeights &weights, bool exact_k3 = true);

[[nodiscard]] at::Tensor complete_target_layer(const TargetMlpInput &prepared,
                                               const at::Tensor &mlp_output,
                                               bool exact_k3 = true);

[[nodiscard]] at::Tensor complete_target_layer_rows(
    const TargetMlpRowsInput &prepared, const at::Tensor &mlp_output_rows,
    bool exact_k3 = true);

[[nodiscard]] at::Tensor run_target_dense(const at::Tensor &hidden,
                                          const TargetDenseWeights &weights,
                                          bool exact_k3 = true);

/*
 * Multi-position layer-zero FFN.  It preserves the same gate/up, SiTU and
 * down equations while reading each resident projection once for [T,H].
 * Decode continues to use run_target_dense's exact one-row path.
 */
[[nodiscard]] at::Tensor run_target_dense_rows(
    const at::Tensor &hidden_rows, const TargetDenseWeights &weights,
    bool exact_k3 = true);

[[nodiscard]] at::Tensor finish_target_tail(const at::Tensor &hidden,
                                            const TargetBlockResidual &residual,
                                            const TargetTailWeights &weights,
                                            bool exact_k3 = true);

/*
 * Wide verification tail.  All rows have already run through canonical T=1
 * attention/cache transitions, so the independent final residual/norm/head
 * work may use the checkpoint's ordinary [T,H] linear contract.  This avoids
 * T full-vocabulary GEMVs and T device synchronizations without changing any
 * recurrent or routed-expert order.
 */
[[nodiscard]] at::Tensor finish_target_tail_rows(
    const at::Tensor &hidden_rows, const at::Tensor &residual_anchor_rows,
    const TargetTailWeights &weights, bool exact_k3 = true);

[[nodiscard]] at::Tensor target_embedding_row(std::uint32_t token_id,
                                              const MoeRowInt8Matrix &embedding,
                                              bool exact_k3 = true);

/* Proposal-only consumers reuse the resident K3 I/O matrices through these
 * tensor helpers. They do not verify, accept, or emit the resulting IDs. */
[[nodiscard]] at::Tensor target_embedding_rows(
    const at::Tensor &token_ids, const MoeRowInt8Matrix &embedding,
    bool exact_k3 = true);

[[nodiscard]] at::Tensor target_language_model_head_rows(
    const at::Tensor &hidden_rows, const MoeRowInt8Matrix &language_model_head,
    bool packed_int8_qualified, bool exact_k3 = true);

} // namespace deltafin::provider_internal

#endif
