#include <metal_stdlib>
using namespace metal;

struct DeltafinRouteMailboxT1 {
  long expert_ids[16];
  uint weight_bits[16];
};

kernel void deltafin_route_mailbox_t1(
    device const long* expert_ids [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device DeltafinRouteMailboxT1* output [[buffer(2)]],
    uint edge [[thread_position_in_grid]]) {
  if (edge < 16) {
    output->expert_ids[edge] = expert_ids[edge];
    output->weight_bits[edge] = as_type<uint>(weights[edge]);
  }
}
