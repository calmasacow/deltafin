#include "provider_cuda_moe.h"

#include <ATen/ATen.h>

#include <array>
#include <bit>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(DELTAFIN_HAVE_CUDA_MOE_V1)
#include <cuda_runtime_api.h>
#endif

namespace {

using deltafin::provider_internal::CudaMoeCachePolicy;
using deltafin::provider_internal::CudaMoeCacheReserveKind;
using deltafin::provider_internal::CudaMoeExpertCache;

template <typename Function>
void require_failure(Function&& function, const char* expected) {
  try {
    function();
  } catch (const std::exception& error) {
    if (std::string(error.what()).find(expected) == std::string::npos) {
      throw std::runtime_error(
          std::string("unexpected CUDA provider failure: ") + error.what());
    }
    return;
  }
  throw std::runtime_error(
      std::string("expected CUDA provider failure containing: ") + expected);
}

void portable_policy_test() {
  const at::Device cpu(at::kCPU);

  CudaMoeExpertCache invalid_auto(cpu);
  require_failure(
      [&] {
        invalid_auto.configure(CudaMoeCachePolicy{
            .automatic_capacity = true,
            .capacity_experts = 1,
            .reserve_kind = CudaMoeCacheReserveKind::Auto,
            .reserve_value = 0});
      },
      "outside its bounds");

  CudaMoeExpertCache invalid_reserve(cpu);
  require_failure(
      [&] {
        invalid_reserve.configure(CudaMoeCachePolicy{
            .automatic_capacity = false,
            .capacity_experts = 0,
            .reserve_kind = CudaMoeCacheReserveKind::Auto,
            .reserve_value = 1});
      },
      "outside its bounds");

  CudaMoeExpertCache disabled(cpu);
  disabled.configure(CudaMoeCachePolicy{
      .automatic_capacity = false,
      .capacity_experts = 0,
      .reserve_kind = CudaMoeCacheReserveKind::Bytes,
      .reserve_value = UINT64_MAX});
  require_failure(
      [&] {
        disabled.configure(CudaMoeCachePolicy{
            .automatic_capacity = false,
            .capacity_experts = 1,
            .reserve_kind = CudaMoeCacheReserveKind::Bytes,
            .reserve_value = 0});
      },
      "already frozen");
}

#if !defined(DELTAFIN_HAVE_CUDA_MOE_V1)

void portable_stub_test() {
  using deltafin::provider_internal::cuda_moe_compiled;
  if (cuda_moe_compiled()) {
    throw std::runtime_error(
        "portable CUDA provider stub advertised a linked CUDA kernel");
  }

  CudaMoeExpertCache cache{at::Device(at::kCPU)};
  if (cache.available() ||
      cache.detail().find("not compiled") == std::string::npos) {
    throw std::runtime_error(
        "portable CUDA provider stub did not fail closed");
  }
  require_failure(
      [&] {
        cache.configure(CudaMoeCachePolicy{
            .automatic_capacity = false,
            .capacity_experts = 0,
            .reserve_kind = CudaMoeCacheReserveKind::Bytes,
            .reserve_value = 0});
      },
      "already frozen");

  const std::array<std::uint16_t, 1> experts{0};
  require_failure(
      [&] { static_cast<void>(cache.plan(1, 1, experts)); }, "not compiled");
  cache.cancel_plan(1);
  cache.poison_external("portable poison canary");
  if (cache.detail().find("portable poison canary") == std::string::npos) {
    throw std::runtime_error(
        "portable CUDA provider stub lost its poison diagnostic");
  }
}

#else

void physical_cuda_test() {
  using deltafin::provider_internal::cuda_moe_compiled;
  if (!cuda_moe_compiled()) {
    throw std::runtime_error(
        "CUDA-gated provider test did not advertise its linked kernel");
  }

  int devices = 0;
  const cudaError_t probe = cudaGetDeviceCount(&devices);
  if (probe != cudaSuccess || devices == 0) {
    static_cast<void>(cudaGetLastError());
    std::cout << "provider_cuda_moe.physical=skipped(no visible CUDA device)\n";
    return;
  }

  CudaMoeExpertCache cache{at::Device(at::kCUDA, 0)};
  cache.configure(CudaMoeCachePolicy{
      .automatic_capacity = false,
      .capacity_experts = 0,
      .reserve_kind = CudaMoeCacheReserveKind::Bytes,
      .reserve_value = 0});
  if (!cache.available()) {
    throw std::runtime_error(
        std::string("physical CUDA qualification failed: ") + cache.detail());
  }

  const std::array<std::uint16_t, 2> experts{0, 7};
  const auto report = cache.plan(1, 1, experts);
  if (report.residency_enabled || report.capacity_experts != 0 ||
      report.missing_experts.size() != experts.size() ||
      report.missing_experts[0] != experts[0] ||
      report.missing_experts[1] != experts[1]) {
    throw std::runtime_error(
        "exact CUDA cache capacity zero did not disable residency");
  }
  cache.cancel_plan(1);
  require_failure(
      [&] { static_cast<void>(cache.materialize_plan_for_cpu(1)); },
      "stale or unknown");

#if defined(DELTAFIN_PROVIDER_CUDA_MOE_TESTING)
  // Capacity one is deliberately assigned to layer 84: the production
  // stratum permutation maps its zero-based slot 83 to rank zero, so exactly
  // this layer receives the sole cache entry. This exercises the real pinned
  // staging, asynchronous upload, admission, hit snapshot and D2H fallback
  // path without allocating a second expert.
  constexpr std::uint32_t layer = 84;
  constexpr std::int64_t hidden = 3584;
  constexpr std::size_t expert_span = 17'547'264;
  CudaMoeExpertCache resident_cache{at::Device(at::kCUDA, 0)};
  resident_cache.configure(CudaMoeCachePolicy{
      .automatic_capacity = false,
      .capacity_experts = 1,
      .reserve_kind = CudaMoeCacheReserveKind::Bytes,
      .reserve_value = 0});
  if (!resident_cache.available()) {
    throw std::runtime_error(
        std::string("physical CUDA residency qualification failed: ") +
        resident_cache.detail());
  }

  const std::array<std::uint16_t, 1> one_expert{0};
  const auto miss = resident_cache.plan(101, layer, one_expert);
  if (!miss.residency_enabled || miss.capacity_experts != 1 ||
      miss.missing_experts !=
          std::vector<std::uint16_t>(one_expert.begin(), one_expert.end())) {
    throw std::runtime_error(
        "one-expert CUDA residency canary did not begin with one miss");
  }

  std::vector<std::uint8_t> exact_bytes(expert_span);
  for (std::size_t index = 0; index < exact_bytes.size(); ++index) {
    exact_bytes[index] = static_cast<std::uint8_t>(
        (index * std::size_t{131} + std::size_t{17}) & 0xffU);
  }
  deltafin::provider_internal::PreparedMoeT1 prepared;
  prepared.layer_index = layer;
  prepared.geometry = deltafin::provider_internal::k3_moe_geometry();
  prepared.routed_input = at::zeros(
      {1, hidden},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  prepared.route.expert_ids.fill(0);
  prepared.route.weight_bits.fill(std::bit_cast<std::uint32_t>(1.0F / 16.0F));
  const std::array<const deltafin::provider_internal::PreparedMoeT1*, 1>
      prepared_rows{&prepared};
  const deltafin::provider_internal::CanonicalExpertPositionTileT1 tile{
      .expert_ids = std::span<const std::uint16_t>(one_expert),
      .expert_major_bytes = std::span<const std::uint8_t>(exact_bytes),
      .layout = deltafin::provider_internal::MoeExpertLayout::RawV1,
      .expert_span_bytes = expert_span};
  const at::Tensor output = resident_cache.execute_positions_plan_t1(
      101,
      std::span<const deltafin::provider_internal::PreparedMoeT1* const>(
          prepared_rows),
      tile);
  if (!output.defined() || !output.device().is_cuda() ||
      output.sizes() != at::IntArrayRef({1, hidden})) {
    throw std::runtime_error(
        "one-expert CUDA residency canary returned an invalid output");
  }

  const auto hit = resident_cache.plan(102, layer, one_expert);
  if (!hit.residency_enabled || hit.capacity_experts != 1 ||
      !hit.missing_experts.empty()) {
    throw std::runtime_error(
        "one-expert CUDA residency canary did not produce a zero-miss hit");
  }
  const auto fallback = resident_cache.materialize_plan_for_cpu(102);
  if (fallback.canonical_experts !=
          std::vector<std::uint16_t>(one_expert.begin(), one_expert.end()) ||
      fallback.resident_experts.size() != 1 ||
      fallback.resident_experts.front().expert != 0) {
    throw std::runtime_error(
        "one-expert CUDA residency canary returned an invalid hit snapshot");
  }
  const at::Tensor copied = fallback.resident_experts.front().bytes;
  if (!copied.device().is_cpu() || copied.scalar_type() != at::kByte ||
      !copied.is_contiguous() ||
      copied.numel() != static_cast<std::int64_t>(exact_bytes.size()) ||
      std::memcmp(copied.const_data_ptr<std::uint8_t>(), exact_bytes.data(),
                  exact_bytes.size()) != 0) {
    throw std::runtime_error(
        "one-expert CUDA residency canary failed its exact D2H comparison");
  }
  resident_cache.cancel_plan(102);
  std::cout << "provider_cuda_moe.residency=ok\n";
#endif
}

#endif

}  // namespace

int main() {
  try {
    portable_policy_test();
#if !defined(DELTAFIN_HAVE_CUDA_MOE_V1)
    portable_stub_test();
    std::cout << "provider_cuda_moe.stub=ok\n";
#else
    physical_cuda_test();
    std::cout << "provider_cuda_moe.cuda_gate=ok\n";
#endif
    std::cout << "provider_cuda_moe.result=PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "provider_cuda_moe_test failed: " << error.what() << '\n';
    return 1;
  }
}
