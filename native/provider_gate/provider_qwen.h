#ifndef DELTAFIN_PROVIDER_QWEN_H
#define DELTAFIN_PROVIDER_QWEN_H

#include "provider_abi.h"

#include <ATen/ATen.h>

#include <array>
#include <cstdint>
#include <vector>

namespace deltafin::provider_internal {

struct QwenShape {
  std::int64_t hidden = 0;
  std::int64_t intermediate = 0;
  static constexpr std::int64_t layers = 28;
  static constexpr std::int64_t heads = 16;
  static constexpr std::int64_t kv_heads = 8;
  static constexpr std::int64_t head_dim = 128;
  static constexpr std::int64_t vocabulary = 151936;
  static constexpr std::int64_t maximum_position = 32768;
  static constexpr std::int64_t eos_token = 151643;
  static constexpr double rope_theta = 1000000.0;
  static constexpr double rms_epsilon = 1.0e-6;

  [[nodiscard]] static QwenShape pinned(std::uint32_t variant);
  void validate() const;
};

struct QwenLayerWeights {
  at::Tensor input_norm;
  at::Tensor post_attention_norm;
  at::Tensor query_norm;
  at::Tensor key_norm;
  at::Tensor query;
  at::Tensor key;
  at::Tensor value;
  at::Tensor output;
  at::Tensor gate;
  at::Tensor up;
  at::Tensor down;
};

struct QwenWeights {
  at::Tensor embedding;
  at::Tensor final_norm;
  std::array<QwenLayerWeights, QwenShape::layers> layers;
};

struct QwenGeneration {
  std::vector<std::uint32_t> token_ids;
  std::vector<float> probabilities;
};

/* Owns immutable copied weights. Every generate call creates private KV state. */
class QwenModel final {
 public:
  QwenModel(QwenShape shape, QwenWeights weights);
  [[nodiscard]] const QwenShape& shape() const noexcept { return shape_; }
  [[nodiscard]] QwenGeneration generate(
      const std::uint32_t* input_ids, std::size_t input_count,
      std::size_t maximum_new_tokens) const;

 private:
  QwenShape shape_;
  QwenWeights weights_;
  at::Tensor inverse_frequency_;
};

[[nodiscard]] QwenWeights bind_qwen_roster(
    const DeltafinProviderQwenCreateV1& request, const QwenShape& shape,
    const at::Device& device);

}  // namespace deltafin::provider_internal

#endif
