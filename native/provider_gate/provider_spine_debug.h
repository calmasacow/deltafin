#ifndef DELTAFIN_PROVIDER_SPINE_DEBUG_H
#define DELTAFIN_PROVIDER_SPINE_DEBUG_H

#include "provider_abi.h"

#include <cstdint>
#include <span>
#include <vector>

namespace deltafin::provider_internal {

// Development-only observability for the provider gate.  This deliberately
// stays outside the stable C ABI: normal Rust callers neither pay for nor
// depend on a diagnostic request.  The counts describe the currently bound
// generation after its transactional commit.
struct SpineBindingDebugStats {
  std::uint64_t source_component_count = 0;
  std::uint64_t upload_run_count = 0;
  std::uint64_t direct_upload_run_count = 0;
  std::uint64_t gathered_upload_run_count = 0;
  std::uint64_t source_component_bytes = 0;
  std::uint64_t logical_target_bytes = 0;
  std::uint64_t resident_storage_bytes = 0;
  std::uint64_t mla_input_bundle_count = 0;
};

struct SpineStoreDebugStats {
  std::uint32_t resident_prefix_layers = 0;
  std::uint64_t resident_storage_bytes = 0;
  bool transient_bound = false;
  std::uint32_t transient_layer = 0;
  std::uint64_t transient_generation = 0;
  std::uint64_t transient_storage_bytes = 0;
  std::uint64_t last_generation = 0;
};

struct SpineFp32ExecutionDebugReport {
  std::vector<float> values;
  std::uint64_t owner = 0;
  std::uint64_t spine_generation = 0;
  std::uint64_t required_elements = 0;
  std::uint64_t capacity_elements = 0;
  std::uintptr_t storage_identity = 0;
  std::uint32_t layer_index = 0;
};

[[nodiscard]] SpineBindingDebugStats spine_binding_debug_stats(
    DeltafinProviderSessionHandleV1 session, std::uint32_t layer_index,
    std::uint64_t generation);

[[nodiscard]] SpineStoreDebugStats spine_store_debug_stats(
    DeltafinProviderSessionHandleV1 session);

/*
 * Development-only production-carrier projection oracle.  The public ABI
 * performs the bind; this helper then exercises the exact backend retained by
 * that published slot, so ownership tests can destroy the ABI source before
 * any projection is encoded.
 */
[[nodiscard]] std::vector<float> spine_original_bf16_debug_project(
    DeltafinProviderSessionHandleV1 session, std::uint32_t layer_index,
    std::uint64_t generation, std::uint32_t slot,
    std::span<const float> input);

/*
 * Development-only direct exercise of the shared FP32 execution arena. It
 * uses an already bound compact layer, invokes the same production
 * materializer, synchronizes only for diagnostic readback, and exposes enough
 * identity metadata to prove bounded reuse and sequencing.
 */
[[nodiscard]] SpineFp32ExecutionDebugReport
spine_fp32_execution_debug(
    DeltafinProviderSessionHandleV1 session, std::uint32_t layer_index,
    std::uint64_t generation, std::uint64_t owner, std::uint32_t slot);

}  // namespace deltafin::provider_internal

#endif
