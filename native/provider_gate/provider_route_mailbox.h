#ifndef DELTAFIN_PROVIDER_ROUTE_MAILBOX_H
#define DELTAFIN_PROVIDER_ROUTE_MAILBOX_H

#include <ATen/core/Tensor.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace deltafin::provider_internal {

constexpr std::size_t kRouteMailboxTopK = 16;

/*
 * Host-visible, byte-exact result of one current-stream MPS boundary.  Keep
 * weights as bits: routing is an exact target decision, so crossing the host
 * boundary must not perform a floating-point conversion or Python boxing.
 */
struct RouteMailboxT1 {
  std::array<std::int64_t, kRouteMailboxTopK> expert_ids = {};
  std::array<std::uint32_t, kRouteMailboxTopK> weight_bits = {};
};

#if defined(__APPLE__)
/*
 * Returns false without modifying `output` when the tensors do not qualify or
 * when Metal setup/execution fails.  Callers must retain their established
 * tensor-to-CPU path as the authoritative fallback.
 */
[[nodiscard]] bool try_materialize_mps_route_t1(
    const at::Tensor& expert_ids, const at::Tensor& weights,
    RouteMailboxT1& output) noexcept;
#endif

}  // namespace deltafin::provider_internal

#endif
