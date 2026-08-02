#include "provider_target_tape.h"

#include "provider_cuda_moe.h"

#include <ATen/ops/argmax.h>
#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace deltafin::provider_internal {
namespace {

constexpr std::int64_t kExactHidden = 7168;
constexpr std::int64_t kSyntheticHidden = 32;

static_assert(std::is_nothrow_move_constructible_v<MlaPreparedDecode>);
static_assert(std::is_nothrow_move_assignable_v<KdaState>);
static_assert(std::is_nothrow_move_assignable_v<at::Tensor>);

bool same_geometry(const MoeGeometry &left, const MoeGeometry &right) {
  return left.hidden == right.hidden &&
         left.routed_hidden == right.routed_hidden &&
         left.intermediate == right.intermediate &&
         left.experts == right.experts &&
         left.shared_intermediate == right.shared_intermediate;
}

struct CacheStage {
  enum class Kind { Kda, Mla };

  Kind kind = Kind::Kda;
  std::uint32_t layer_index = 0;
  TargetKdaCache *kda_cache = nullptr;
  std::uint64_t expected_kda_version = 0;
  std::optional<KdaState> next_kda_state;
  MlaCache *mla_cache = nullptr;
  std::optional<MlaPreparedDecode> prepared_mla;

  CacheStage() = default;
  CacheStage(const CacheStage &) = delete;
  CacheStage &operator=(const CacheStage &) = delete;

  ~CacheStage() { static_cast<void>(cancel_noexcept()); }

  bool cancel_noexcept() noexcept {
    if (kind != Kind::Mla || !prepared_mla.has_value() ||
        prepared_mla->finalized) {
      return true;
    }
    try {
      cancel_mla_decode(*mla_cache, *prepared_mla);
      return true;
    } catch (...) {
      return false;
    }
  }

  void preflight() const {
    if (kind == Kind::Kda) {
      if (kda_cache == nullptr || !next_kda_state.has_value() ||
          kda_cache->layer_index != layer_index ||
          kda_cache->version != expected_kda_version ||
          kda_cache->version == std::numeric_limits<std::uint64_t>::max()) {
        throw std::runtime_error(
            "target position has a stale or invalid staged KDA cache");
      }
      return;
    }

    if (mla_cache == nullptr || !prepared_mla.has_value()) {
      throw std::runtime_error(
          "target position has an incomplete staged MLA cache");
    }
    const MlaPreparedDecode &prepared = *prepared_mla;
    const MlaShape &shape = mla_cache->shape();
    if (prepared.owner != mla_cache || prepared.finalized ||
        prepared.nonce == 0 || !prepared.output.defined() ||
        !mla_cache->has_pending_prepare() ||
        mla_cache->version() != prepared.expected_version ||
        mla_cache->length() != prepared.expected_length ||
        mla_cache->version() == std::numeric_limits<std::uint64_t>::max() ||
        prepared.next_length != prepared.expected_length + 1 ||
        prepared.next_capacity < prepared.next_length ||
        prepared.next_capacity > shape.max_context) {
      throw std::runtime_error(
          "target position has a stale or invalid staged MLA cache");
    }
    if (prepared.uses_grown_storage) {
      if (!prepared.grown_key_storage.defined() ||
          !prepared.grown_value_storage.defined() ||
          prepared.grown_key_storage.sizes() !=
              at::IntArrayRef({1, shape.num_heads, prepared.next_capacity,
                               shape.query_head_dim()}) ||
          prepared.grown_value_storage.sizes() !=
              at::IntArrayRef({1, shape.num_heads, prepared.next_capacity,
                               shape.value_head_dim})) {
        throw std::runtime_error(
            "target position has invalid staged MLA growth storage");
      }
    } else if (prepared.next_capacity != mla_cache->capacity()) {
      throw std::runtime_error(
          "target position staged MLA capacity changed without growth");
    }
  }
};

struct PendingMoeLayer {
  std::uint32_t layer_index = 0;
  TargetMlpInput mlp_input;
  PreparedMoeT1 moe;
  MoeSpineT1 spine;
  std::unique_ptr<CacheStage> cache;
};

} // namespace

class TargetPositionTape::Impl {
public:
  Impl(const TargetPositionBindings &bindings, at::Tensor input_hidden)
      : exact_k3_(bindings.contract == TargetTapeContract::ExactK3),
        tail_(bindings.tail),
        pilot_routers_(bindings.pilot_routers) {
    validate_and_copy_bindings(bindings);
    const c10::InferenceMode inference_guard;
    const std::int64_t hidden = exact_k3_ ? kExactHidden : kSyntheticHidden;
    if (!input_hidden.defined() || input_hidden.scalar_type() != at::kFloat ||
        !input_hidden.is_contiguous() ||
        input_hidden.sizes() != at::IntArrayRef({1, hidden})) {
      throw std::invalid_argument(
          "target position input must be an exact contiguous fp32 hidden row");
    }
    hidden_ = std::move(input_hidden);
    residual_ = empty_target_block_residual(hidden_.device(), hidden_.size(1));
    staged_.reserve(kTargetLayerCount);
  }

  ~Impl() { abort_unlocked(); }

  TargetLayerPrepareResult prepare_layer(const TargetLayerBinding &binding) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_state(TargetPositionState::Active, "prepare the next target layer");
    try {
      validate_current_layer(binding);
      const TargetLayerCacheBinding &cache = caches_[next_layer_];
      TargetAttentionInput attention = prepare_target_attention(
          hidden_, residual_, *binding.residual, next_layer_, exact_k3_);
      auto cache_stage = std::make_unique<CacheStage>();
      cache_stage->layer_index = next_layer_;
      at::Tensor attention_output;
      if (binding.attention_kind == TargetAttentionKind::Kda) {
        KdaDecodeResult decoded =
            kda_decode_one(attention.normalized, *binding.kda_weights,
                           cache.kda_cache->state, exact_k3_);
        attention_output = decoded.output;
        cache_stage->kind = CacheStage::Kind::Kda;
        cache_stage->kda_cache = cache.kda_cache;
        cache_stage->expected_kda_version = cache.kda_cache->version;
        cache_stage->next_kda_state.emplace(std::move(decoded.next_state));
      } else {
        MlaPreparedDecode decoded =
            exact_k3_ ? prepare_k3_mla_decode(
                            attention.normalized.view({1, 1, hidden_.size(1)}),
                            *binding.mla_weights, *cache.mla_cache,
                            binding.mla_input_bundle)
                      : prepare_mla_decode(
                            attention.normalized.view({1, 1, hidden_.size(1)}),
                            *binding.mla_weights, *cache.mla_cache, true,
                            binding.mla_input_bundle);
        attention_output =
            decoded.output.view({1, hidden_.size(1)}).contiguous();
        cache_stage->kind = CacheStage::Kind::Mla;
        cache_stage->mla_cache = cache.mla_cache;
        cache_stage->prepared_mla.emplace(std::move(decoded));
      }

      TargetMlpInput mlp_input = prepare_target_mlp(
          attention, attention_output, *binding.residual, exact_k3_);
      if (next_layer_ == 0) {
        const at::Tensor mlp_output =
            run_target_dense(mlp_input.normalized, *binding.dense, exact_k3_);
        at::Tensor next_hidden =
            complete_target_layer(mlp_input, mlp_output, exact_k3_);
        publish_completed_layer(std::move(next_hidden),
                                std::move(mlp_input.next_anchors),
                                std::move(cache_stage));
        return TargetLayerPrepareResult{
            .kind = TargetLayerPrepareKind::DenseCompleted,
            .route = {},
        };
      }

      // Copying this descriptor only retains tensor handles for the one
      // transient layer currently split across expert I/O. It never pins the
      // other 92 layer spines.
      MoeSpineT1 pending_spine = *binding.moe;
      PreparedMoeT1 moe = prepare_moe_t1(mlp_input.normalized, pending_spine);
      TargetRouteRequest request{
          .layer_index = next_layer_,
          .spine_generation = pending_spine.generation,
          .route = moe.route,
      };
      pending_ = std::make_unique<PendingMoeLayer>(
          PendingMoeLayer{next_layer_, std::move(mlp_input), std::move(moe),
                          std::move(pending_spine), std::move(cache_stage)});
      state_ = TargetPositionState::WaitingForExperts;
      return TargetLayerPrepareResult{
          .kind = TargetLayerPrepareKind::ExpertsRequired,
          .route = request,
      };
    } catch (...) {
      abort_unlocked();
      throw;
    }
  }

  void finish_moe_layer(const std::uint32_t layer_index,
                        const std::uint64_t spine_generation,
                        const CanonicalExpertBatchT1 &experts,
                        const MoeRunOptions &options) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_state(TargetPositionState::WaitingForExperts,
                  "finish a target MoE layer");
    try {
      if (pending_ == nullptr || pending_->layer_index != layer_index ||
          layer_index != next_layer_) {
        throw std::invalid_argument(
            "target expert finish is out of layer order");
      }
      if (pending_->spine.generation != spine_generation ||
          pending_->moe.spine_generation != spine_generation) {
        throw std::invalid_argument(
            "target expert finish has a stale spine generation");
      }
      const at::Tensor routed =
          execute_routed_moe_t1(pending_->moe, experts, options);
      const at::Tensor mlp_output =
          complete_moe_t1(pending_->moe, routed, pending_->spine);
      at::Tensor next_hidden =
          complete_target_layer(pending_->mlp_input, mlp_output, exact_k3_);
      std::unique_ptr<PendingMoeLayer> completed = std::move(pending_);
      publish_completed_layer(std::move(next_hidden),
                              std::move(completed->mlp_input.next_anchors),
                              std::move(completed->cache));
      state_ = next_layer_ == kTargetLayerCount
                   ? TargetPositionState::ReadyForTail
                   : TargetPositionState::Active;
    } catch (...) {
      if (options.cuda_cache != nullptr &&
          (options.expert_backend == MoeExpertBackend::CudaMxfp4 ||
           options.cuda_plan != 0)) {
        options.cuda_cache->poison_external(
            "target MoE transaction failed after CUDA selection");
      }
      abort_unlocked();
      throw;
    }
  }

  std::uint32_t finish_greedy() {
    std::lock_guard<std::mutex> lock(mutex_);
    require_state(TargetPositionState::ReadyForTail,
                  "finish the target position");
    try {
      if (next_layer_ != kTargetLayerCount || pending_ != nullptr ||
          staged_.size() != kTargetLayerCount ||
          residual_.anchors.size(1) != 8) {
        throw std::runtime_error(
            "target position is incomplete at its greedy tail");
      }
      const at::Tensor logits =
          finish_target_tail(hidden_, residual_, *tail_, exact_k3_);
      if (!logits.defined() || logits.dim() != 2 || logits.size(0) != 1 ||
          logits.size(1) <= 0) {
        throw std::runtime_error(
            "target tail did not produce one vocabulary row");
      }
      const std::int64_t token =
          at::argmax(logits, -1, false).item<std::int64_t>();
      if (token < 0 || token > static_cast<std::int64_t>(
                                   std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error(
            "target greedy token is outside the public token range");
      }

      preflight_commit();
      // After the complete preflight, each MLA commit consists solely of
      // checked scalar publication and noexcept tensor-handle moves.  The
      // provider session must retain exclusive ownership for this short loop.
      for (const auto &stage : staged_) {
        if (stage->kind == CacheStage::Kind::Mla) {
          commit_mla_decode(*stage->mla_cache, *stage->prepared_mla);
        }
      }
      for (auto &stage : staged_) {
        if (stage->kind == CacheStage::Kind::Kda) {
          stage->kda_cache->state = std::move(*stage->next_kda_state);
          ++stage->kda_cache->version;
        }
      }
      state_ = TargetPositionState::Committed;
      staged_.clear();
      hidden_ = at::Tensor();
      residual_.anchors = at::Tensor();
      return static_cast<std::uint32_t>(token);
    } catch (...) {
      abort_unlocked();
      throw;
    }
  }

  void cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == TargetPositionState::Committed) {
      throw std::logic_error("a committed target position cannot be cancelled");
    }
    if (state_ == TargetPositionState::Cancelled ||
        state_ == TargetPositionState::Poisoned) {
      return;
    }
    abort_unlocked();
    if (state_ == TargetPositionState::Poisoned) {
      throw std::runtime_error(
          "target position could not release every staged MLA cache");
    }
  }

  TargetPositionState state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
  }

  std::uint32_t next_layer_index() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return next_layer_;
  }

  std::size_t staged_count(const CacheStage::Kind kind) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<std::size_t>(std::count_if(
        staged_.begin(), staged_.end(),
        [kind](const auto &stage) { return stage->kind == kind; }));
  }

private:
  void validate_and_copy_bindings(const TargetPositionBindings &bindings) {
    if (bindings.contract != TargetTapeContract::ExactK3 &&
        bindings.contract != TargetTapeContract::SyntheticK3Schedule) {
      throw std::invalid_argument("target position contract is unknown");
    }
    if (bindings.caches.size() != kTargetLayerCount ||
        bindings.tail == nullptr) {
      throw std::invalid_argument(
          "target position requires 93 caches and tail weights");
    }
    if (pilot_routers_ != nullptr) {
      for (std::size_t layer = 0; layer < pilot_routers_->size(); ++layer) {
        const auto &router = (*pilot_routers_)[layer];
        if (router.has_value() &&
            (layer == 0 || router->layer_index != layer)) {
          throw std::invalid_argument(
              "target position pilot roster has an invalid immutable slot");
        }
      }
    }
    std::copy(bindings.caches.begin(), bindings.caches.end(), caches_.begin());
    std::unordered_set<const void *> cache_addresses;
    std::size_t kda_count = 0;
    std::size_t mla_count = 0;
    for (std::uint32_t index = 0; index < kTargetLayerCount; ++index) {
      const TargetLayerCacheBinding &layer = caches_[index];
      const TargetAttentionKind expected_kind = target_layer_uses_mla(index)
                                                    ? TargetAttentionKind::Mla
                                                    : TargetAttentionKind::Kda;
      if (layer.layer_index != index || layer.attention_kind != expected_kind) {
        throw std::invalid_argument(
            "target layer order or KDA/MLA schedule is invalid");
      }
      if (expected_kind == TargetAttentionKind::Kda) {
        ++kda_count;
        if (layer.kda_cache == nullptr || layer.mla_cache != nullptr ||
            layer.kda_cache->layer_index != index ||
            layer.kda_cache->version ==
                std::numeric_limits<std::uint64_t>::max() ||
            !cache_addresses.insert(layer.kda_cache).second) {
          throw std::invalid_argument(
              "target KDA binding/cache ownership is invalid");
        }
      } else {
        ++mla_count;
        if (layer.mla_cache == nullptr || layer.kda_cache != nullptr ||
            layer.mla_cache->has_pending_prepare() ||
            layer.mla_cache->version() ==
                std::numeric_limits<std::uint64_t>::max() ||
            !cache_addresses.insert(layer.mla_cache).second) {
          throw std::invalid_argument(
              "target MLA binding/cache ownership is invalid");
        }
        const MlaShape &shape = layer.mla_cache->shape();
        if ((exact_k3_ &&
             (!shape.is_exact_k3() ||
              layer.mla_cache->representation() !=
                  MlaCacheRepresentation::ExpandedExact)) ||
            (!exact_k3_ &&
             (shape.hidden_size != kSyntheticHidden || shape.is_exact_k3()))) {
          throw std::invalid_argument(
              "target MLA cache does not match the tape contract");
        }
      }
    }
    if (kda_count != kTargetKdaLayerCount ||
        mla_count != kTargetMlaLayerCount) {
      throw std::logic_error("target KDA/MLA schedule counts changed");
    }
  }

  void validate_current_layer(const TargetLayerBinding &layer) const {
    if (next_layer_ >= kTargetLayerCount || layer.layer_index != next_layer_) {
      throw std::invalid_argument(
          "target streamed layer is out of schedule order");
    }
    const TargetAttentionKind expected_kind = target_layer_uses_mla(next_layer_)
                                                  ? TargetAttentionKind::Mla
                                                  : TargetAttentionKind::Kda;
    if (layer.attention_kind != expected_kind || layer.residual == nullptr) {
      throw std::invalid_argument(
          "target streamed layer has an invalid attention binding");
    }
    if (expected_kind == TargetAttentionKind::Kda) {
      if (layer.kda_weights == nullptr || layer.mla_weights != nullptr ||
          layer.mla_input_bundle != nullptr) {
        throw std::invalid_argument("target streamed KDA weights are invalid");
      }
    } else if (layer.mla_weights == nullptr || layer.kda_weights != nullptr) {
      throw std::invalid_argument("target streamed MLA weights are invalid");
    }

    if (next_layer_ == 0) {
      if (layer.dense == nullptr || layer.moe != nullptr) {
        throw std::invalid_argument(
            "target layer zero must be dense and cannot be routed");
      }
      return;
    }
    const std::int64_t hidden = exact_k3_ ? kExactHidden : kSyntheticHidden;
    const MoeGeometry expected_moe = k3_moe_geometry();
    if (layer.dense != nullptr || layer.moe == nullptr ||
        layer.moe->layer_index != next_layer_ || layer.moe->generation == 0 ||
        layer.moe->geometry.hidden != static_cast<std::uint32_t>(hidden) ||
        (exact_k3_ && !same_geometry(layer.moe->geometry, expected_moe))) {
      throw std::invalid_argument(
          "target routed-MoE binding is invalid for its streamed layer");
    }
  }

  void require_state(const TargetPositionState expected,
                     const char *operation) const {
    if (state_ != expected) {
      throw std::logic_error(std::string("cannot ") + operation +
                             " in the current target-position state");
    }
  }

  void publish_completed_layer(at::Tensor next_hidden, at::Tensor next_anchors,
                               std::unique_ptr<CacheStage> cache_stage) {
    if (cache_stage == nullptr || staged_.size() >= staged_.capacity()) {
      throw std::logic_error(
          "target cache staging capacity invariant was violated");
    }
    staged_.push_back(std::move(cache_stage));
    hidden_ = std::move(next_hidden);
    residual_.anchors = std::move(next_anchors);
    ++next_layer_;
  }

  void preflight_commit() const {
    std::size_t kda_count = 0;
    std::size_t mla_count = 0;
    for (const auto &stage : staged_) {
      stage->preflight();
      if (stage->kind == CacheStage::Kind::Kda) {
        ++kda_count;
      } else {
        ++mla_count;
      }
    }
    if (kda_count != kTargetKdaLayerCount ||
        mla_count != kTargetMlaLayerCount) {
      throw std::runtime_error(
          "target position did not stage every attention cache exactly once");
    }
  }

  void abort_unlocked() noexcept {
    if (state_ == TargetPositionState::Committed ||
        state_ == TargetPositionState::Cancelled) {
      return;
    }
    bool released = true;
    if (pending_ != nullptr && pending_->cache != nullptr) {
      released = pending_->cache->cancel_noexcept() && released;
    }
    for (const auto &stage : staged_) {
      released = stage->cancel_noexcept() && released;
    }
    pending_.reset();
    staged_.clear();
    hidden_ = at::Tensor();
    residual_.anchors = at::Tensor();
    state_ = released ? TargetPositionState::Cancelled
                      : TargetPositionState::Poisoned;
  }

  bool exact_k3_ = true;
  const TargetTailWeights *tail_ = nullptr;
  // Session-owned and address-stable for the tape lifetime. The position path
  // binds the same immutable roster as the sequence path even though only the
  // latter currently exposes a public asynchronous read-hint boundary.
  const TargetPilotRoster *pilot_routers_ = nullptr;
  std::array<TargetLayerCacheBinding, kTargetLayerCount> caches_{};
  at::Tensor hidden_;
  TargetBlockResidual residual_;
  std::uint32_t next_layer_ = 0;
  std::vector<std::unique_ptr<CacheStage>> staged_;
  std::unique_ptr<PendingMoeLayer> pending_;
  TargetPositionState state_ = TargetPositionState::Active;
  mutable std::mutex mutex_;
};

TargetPositionTape::TargetPositionTape(const TargetPositionBindings &bindings,
                                       at::Tensor input_hidden)
    : impl_(std::make_unique<Impl>(bindings, std::move(input_hidden))) {}

TargetPositionTape::~TargetPositionTape() = default;

TargetLayerPrepareResult
TargetPositionTape::prepare_layer(const TargetLayerBinding &layer) {
  return impl_->prepare_layer(layer);
}

void TargetPositionTape::finish_moe_layer(const std::uint32_t layer_index,
                                          const std::uint64_t spine_generation,
                                          const CanonicalExpertBatchT1 &experts,
                                          const MoeRunOptions &options) {
  impl_->finish_moe_layer(layer_index, spine_generation, experts, options);
}

std::uint32_t TargetPositionTape::finish_greedy() {
  return impl_->finish_greedy();
}

void TargetPositionTape::cancel() { impl_->cancel(); }

TargetPositionState TargetPositionTape::state() const { return impl_->state(); }

std::uint32_t TargetPositionTape::next_layer_index() const {
  return impl_->next_layer_index();
}

std::size_t TargetPositionTape::staged_kda_count() const {
  return impl_->staged_count(CacheStage::Kind::Kda);
}

std::size_t TargetPositionTape::staged_mla_count() const {
  return impl_->staged_count(CacheStage::Kind::Mla);
}

} // namespace deltafin::provider_internal
