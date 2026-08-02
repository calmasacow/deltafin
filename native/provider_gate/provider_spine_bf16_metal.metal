#include <metal_stdlib>
using namespace metal;

struct DeltafinSpineBf16GemvDimsV1 {
  uint rows;
  uint columns;
  uint reserved0;
  uint reserved1;
};

inline float4 deltafin_spine_decode_bf16x4_v1(const ushort4 bits) {
  return as_type<float4>(uint4(bits) << 16);
}

kernel void deltafin_spine_decode_bf16_bits_v1(
    device const ushort* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& elements [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < elements) {
    destination[index] = as_type<float>(uint(source[index]) << 16);
  }
}

/*
 * T=1 production candidate. One SIMD group owns four adjacent rows. Every
 * checkpoint ushort is expanded by placing its bits in the high half of an
 * IEEE fp32 word; inputs, accumulators, explicit FMAs, SIMD reductions, and
 * outputs all remain fp32. Four SIMD groups share each 128-thread group.
 */
kernel void deltafin_spine_bf16_gemv_rows4_t1_v1(
    device const ushort* weight [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant DeltafinSpineBf16GemvDimsV1& dims [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  const uint row0 = (tg * 4 + simdgroup) * 4;
  float a0 = 0.0f;
  float a1 = 0.0f;
  float a2 = 0.0f;
  float a3 = 0.0f;
  for (uint column = lane * 4; column < dims.columns; column += 128) {
    const float4 x = *((device const float4*)(input + column));
    if (row0 + 0 < dims.rows) {
      const float4 w = deltafin_spine_decode_bf16x4_v1(
          *((device const ushort4*)(weight +
              (row0 + 0) * dims.columns + column)));
      a0 = fma(w.x, x.x, a0);
      a0 = fma(w.y, x.y, a0);
      a0 = fma(w.z, x.z, a0);
      a0 = fma(w.w, x.w, a0);
    }
    if (row0 + 1 < dims.rows) {
      const float4 w = deltafin_spine_decode_bf16x4_v1(
          *((device const ushort4*)(weight +
              (row0 + 1) * dims.columns + column)));
      a1 = fma(w.x, x.x, a1);
      a1 = fma(w.y, x.y, a1);
      a1 = fma(w.z, x.z, a1);
      a1 = fma(w.w, x.w, a1);
    }
    if (row0 + 2 < dims.rows) {
      const float4 w = deltafin_spine_decode_bf16x4_v1(
          *((device const ushort4*)(weight +
              (row0 + 2) * dims.columns + column)));
      a2 = fma(w.x, x.x, a2);
      a2 = fma(w.y, x.y, a2);
      a2 = fma(w.z, x.z, a2);
      a2 = fma(w.w, x.w, a2);
    }
    if (row0 + 3 < dims.rows) {
      const float4 w = deltafin_spine_decode_bf16x4_v1(
          *((device const ushort4*)(weight +
              (row0 + 3) * dims.columns + column)));
      a3 = fma(w.x, x.x, a3);
      a3 = fma(w.y, x.y, a3);
      a3 = fma(w.z, x.z, a3);
      a3 = fma(w.w, x.w, a3);
    }
  }
  a0 = simd_sum(a0);
  a1 = simd_sum(a1);
  a2 = simd_sum(a2);
  a3 = simd_sum(a3);
  if (lane == 0) {
    if (row0 + 0 < dims.rows) output[row0 + 0] = a0;
    if (row0 + 1 < dims.rows) output[row0 + 1] = a1;
    if (row0 + 2 < dims.rows) output[row0 + 2] = a2;
    if (row0 + 3 < dims.rows) output[row0 + 3] = a3;
  }
}
