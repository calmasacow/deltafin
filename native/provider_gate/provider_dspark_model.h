#ifndef DELTAFIN_PROVIDER_DSPARK_MODEL_H
#define DELTAFIN_PROVIDER_DSPARK_MODEL_H

#include "provider_dspark.h"
#include "provider_target.h"

#include <ATen/ATen.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace deltafin::provider_internal {

constexpr std::size_t kDSparkLayers = 5;
constexpr std::int64_t kDSparkTrainedQueryRows = 7;

struct DSparkModelWeights {
  at::Tensor context_projection;
  at::Tensor context_norm;
  std::array<DSparkDecoderWeights, kDSparkLayers> layers;
  at::Tensor fused_context_projection;
  at::Tensor final_norm;
  at::Tensor markov_embedding;
  at::Tensor markov_output;
  at::Tensor confidence_weight;
  at::Tensor confidence_bias;
};

// Bind-time staging validates the fixed 68-tensor-derived arithmetic roster
// represented above and builds the non-persistent five-way context projection.
[[nodiscard]] DSparkModelWeights prepare_dspark_model_weights(
    const DSparkShape& shape, DSparkModelWeights weights);

struct DSparkTargetIo {
  MoeRowInt8Matrix embedding;
  MoeRowInt8Matrix language_model_head;
  bool head_packed_int8_qualified = false;
  bool exact_k3 = true;
};

struct DSparkCacheSnapshot {
  const void* owner = nullptr;
  std::uint64_t generation = 0;
  std::int64_t length = 0;
  std::int64_t capacity = 0;
  std::array<at::Tensor, kDSparkLayers> latent_storage;
  std::array<at::Tensor, kDSparkLayers> positional_storage;

  [[nodiscard]] DSparkLatentContext context(std::size_t layer) const;
};

class DSparkCache {
 public:
  DSparkCache(DSparkShape shape, at::Device device);

  DSparkCache(const DSparkCache&) = delete;
  DSparkCache& operator=(const DSparkCache&) = delete;
  DSparkCache(DSparkCache&&) = delete;
  DSparkCache& operator=(DSparkCache&&) = delete;

  [[nodiscard]] std::int64_t length() const noexcept;
  [[nodiscard]] std::int64_t capacity() const noexcept;
  [[nodiscard]] std::uint64_t generation() const noexcept;
  [[nodiscard]] bool requires_fork() const noexcept;
  [[nodiscard]] DSparkLatentContext context(std::size_t layer) const;
  [[nodiscard]] DSparkCacheSnapshot snapshot() const;
  void restore(const DSparkCacheSnapshot& snapshot);
  void clear();

 private:
  friend class DSparkModel;
  void append(const std::array<DSparkLatentContext, kDSparkLayers>& rows);
  void ensure_capacity(std::int64_t required);

  DSparkShape shape_;
  at::Device device_;
  std::int64_t length_ = 0;
  std::int64_t capacity_ = 0;
  std::uint64_t generation_ = 0;
  bool requires_fork_ = false;
  std::array<at::Tensor, kDSparkLayers> latent_storage_;
  std::array<at::Tensor, kDSparkLayers> positional_storage_;
};

struct DSparkProposalScores {
  // Unverified candidate IDs. Full K3 must verify every row before emission.
  at::Tensor token_ids;
  at::Tensor confidence_logits;
  std::int64_t anchor_token_id = 0;
  std::int64_t anchor_position = 0;
  std::uint64_t cache_generation = 0;
};

class DSparkModel {
 public:
  DSparkModel(DSparkShape shape, DSparkModelWeights weights,
              DSparkTargetIo target_io);

  DSparkModel(const DSparkModel&) = delete;
  DSparkModel& operator=(const DSparkModel&) = delete;
  DSparkModel(DSparkModel&&) = delete;
  DSparkModel& operator=(DSparkModel&&) = delete;

  [[nodiscard]] const DSparkShape& shape() const noexcept;
  [[nodiscard]] DSparkCache& cache() noexcept;
  [[nodiscard]] const DSparkCache& cache() const noexcept;

  void append_target_context(const at::Tensor& combined_target_hidden,
                             const at::Tensor& positions);

  [[nodiscard]] at::Tensor forward_backbone(const at::Tensor& query_token_ids,
                                            const at::Tensor& positions) const;
  [[nodiscard]] at::Tensor forward_backbone_embeddings(
      const at::Tensor& query_embeddings, const at::Tensor& positions) const;

  // Always executes the trained seven-row query. score_rows selects only the
  // leading vocabulary/Markov/confidence rows (1..7).
  [[nodiscard]] DSparkProposalScores propose(std::int64_t anchor_token_id,
                                             std::int64_t score_rows) const;
  [[nodiscard]] DSparkProposalScores propose_from_embeddings(
      std::int64_t anchor_token_id, std::int64_t score_rows,
      const at::Tensor& trained_query_embeddings) const;

 private:
  DSparkShape shape_;
  DSparkModelWeights weights_;
  DSparkTargetIo target_io_;
  DSparkCache cache_;
};

}  // namespace deltafin::provider_internal

#endif
