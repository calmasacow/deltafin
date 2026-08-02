#ifndef DELTAFIN_PROVIDER_PILOT_H
#define DELTAFIN_PROVIDER_PILOT_H

#include "provider_moe.h"

#include <ATen/ATen.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace deltafin::provider_internal {

constexpr std::size_t kPilotTopK = kMoeRouteTopK;
constexpr std::size_t kPilotMaxPrefetch = 2 * kPilotTopK;

/*
 * Immutable, session-owned inputs for one next-layer scheduling prediction.
 * These tensors are intentionally separate from MoeSpineT1: retaining a
 * PilotRouterT1 must never retain the rest of a streamed layer allocation.
 */
struct PilotRouterT1 {
  std::uint32_t layer_index = 0;
  std::uint64_t generation = 0;
  std::uint32_t hidden_size = 0;
  std::uint32_t expert_count = 0;
  bool packed_int8_qualified = false;
  at::Tensor post_attention_norm;
  MoeRowInt8Matrix router;
  at::Tensor correction_bias;
};

/*
 * Scheduling data only. `expert_ids` is int64 [1,16] and `choice_scores` is
 * fp32 [1,16], both on the source device and in ATen topk(sorted=false)
 * order.  This type deliberately has no route weights, uncorrected sigmoid
 * scores, hidden state, or model-output surface.  The authoritative K3 router
 * still runs independently and exclusively decides which experts execute.
 */
struct PilotPredictionT1 {
  std::uint32_t layer_index = 0;
  std::uint64_t generation = 0;
  std::uint32_t expert_count = 0;
  at::Tensor expert_ids;
  at::Tensor choice_scores;
};

/*
 * The same scheduling-only prediction for a live prompt/verification width.
 * Both tensors are [position_count,16].  A single full-width router call is
 * important here: the established Python PILOT path predicts every row, then
 * unions those candidates before applying its bounded read budget.  It never
 * substitutes this prediction for an authoritative route.
 */
struct PilotPredictionRows {
  std::uint32_t layer_index = 0;
  std::uint64_t generation = 0;
  std::uint32_t expert_count = 0;
  std::uint16_t position_count = 0;
  at::Tensor expert_ids;
  at::Tensor choice_scores;
};

/* A canonical ascending disk-read set, never an execution route. */
struct CanonicalPilotPrefetchT1 {
  std::array<std::uint16_t, kPilotMaxPrefetch> expert_ids = {};
  std::size_t count = 0;
  /* Unique candidates before the bounded read budget was applied. */
  std::size_t candidate_count = 0;
};

/*
 * Detach the four scheduling-only tensors needed by PILOT from one complete
 * authoritative MoE binding. Existing qualified row-int8 router storage is
 * cloned byte-for-byte. An original-BF16/fp32 router is converted to a private
 * symmetric row-int8 prediction copy (absmax/127, ties-to-even, [-127,127]).
 * No routed/shared projection or source storage view can escape this call.
 * Approximation is confined to the optional prediction; K3's independent
 * router still supplies every executed expert and route weight.
 */
[[nodiscard]] PilotRouterT1 clone_compact_pilot_router_t1(
    const MoeSpineT1& authoritative,
    const at::Tensor& post_attention_norm, bool exact_k3 = true);

/*
 * Mirror the established K3 PILOT arithmetic for one decode row:
 *
 *   RMS(lookahead_source, next-layer post-attention norm, eps=1e-5)
 *   -> dense or qualified row-int8 router
 *   -> sigmoid -> add correction bias -> topk(16, sorted=false)
 *
 * `exact_k3` is true by default and admits only K3's production geometry.
 * Standalone canaries pass false to exercise the same arithmetic on tiny
 * reviewed tensors.  The operation owns no mutable state and publishes its
 * result only after every validation and ATen operation succeeds.
 */
[[nodiscard]] PilotPredictionT1 predict_pilot_router_t1(
    const at::Tensor& lookahead_source, const PilotRouterT1& router,
    bool exact_k3 = true);

/*
 * Fail-soft decode wrapper. Optional scheduling work may call this surface and
 * simply fall back to an authoritative demand read on nullopt.  It catches all
 * synchronous failures, owns no global disable flag, and cannot partially
 * publish a prediction.
 */
[[nodiscard]] std::optional<PilotPredictionT1> try_predict_pilot_router_t1(
    const at::Tensor& lookahead_source, const PilotRouterT1& router,
    bool exact_k3 = true) noexcept;

/*
 * Full-width counterpart used by prompt ingestion and speculative verify.
 * Accepted source shape is contiguous fp32 [1..64,H].  T=1 deliberately uses
 * the identical ATen operation sequence as predict_pilot_router_t1.
 */
[[nodiscard]] PilotPredictionRows predict_pilot_router_rows(
    const at::Tensor& lookahead_source, const PilotRouterT1& router,
    bool exact_k3 = true);

[[nodiscard]] std::optional<PilotPredictionRows>
try_predict_pilot_router_rows(const at::Tensor& lookahead_source,
                              const PilotRouterT1& router,
                              bool exact_k3 = true) noexcept;

/*
 * Bound already-materialized top-k candidates to at most `cap` speculative
 * reads. Candidates are ranked by choice score descending with expert ID as a
 * deterministic tie-breaker, then returned in canonical ascending ID order.
 * A cap of zero disables the read; this decode-only wrapper remains clamped
 * to its one row of 16 candidates.
 */
[[nodiscard]] CanonicalPilotPrefetchT1 canonicalize_pilot_prefetch_t1(
    std::span<const std::int64_t> expert_ids,
    std::span<const float> choice_scores, std::uint32_t expert_count,
    std::size_t cap);

/*
 * Union per-position top-k candidates, retaining each expert's best score,
 * cap by score descending/expert ID ascending, then return canonical ascending
 * IDs.  `choice_scores` may be empty only when the unique union already fits
 * under `cap`; this mirrors Python PILOT's score-transfer elision.
 */
[[nodiscard]] CanonicalPilotPrefetchT1 canonicalize_pilot_prefetch_rows(
    std::span<const std::int64_t> expert_ids,
    std::span<const float> choice_scores, std::size_t position_count,
    std::uint32_t expert_count, std::size_t cap);

}  // namespace deltafin::provider_internal

#endif
