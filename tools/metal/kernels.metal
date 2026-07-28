// Kimi-K3 fused MXFP4 dequant+GEMV kernels (M1 Max prototype).
// Standalone copy of the MSL embedded in bench.mm (runtime-compiled there
// via newLibraryWithSource; BLUT is host-generated in bench.mm, materialized here).

#include <metal_stdlib>
using namespace metal;

constant float LUT[16] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f,
                          -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};

struct GemvDesc { uint woff; uint soff; uint xoff; uint yoff; };

// One simdgroup computes one output row. Per 32-elem group: 16 packed bytes as
// 4x uchar4, nibble->float via constant table, fma against float4 x, then one
// e8m0 scale via exponent-bit trick 2^(b-127) == as_type<float>(b<<23).
inline float mxdot_row(device const uchar *wrow, device const uchar *srow,
                       device const float4 *x4, uint lane, int NG) {
  float acc = 0.0f;
  device const uchar4 *w4 = (device const uchar4 *)wrow;
  for (int g = (int)lane; g < NG; g += 32) {
    int b4 = g * 4;                 // uchar4 index of this group's 16 bytes
    int xb = g * 8;                 // float4 index of this group's 32 floats
    float gacc = 0.0f;
    for (int j = 0; j < 4; ++j) {
      uchar4 b = w4[b4 + j];
      float4 wa = float4(LUT[b.x & 15], LUT[b.x >> 4], LUT[b.y & 15], LUT[b.y >> 4]);
      float4 wb = float4(LUT[b.z & 15], LUT[b.z >> 4], LUT[b.w & 15], LUT[b.w >> 4]);
      gacc += dot(wa, x4[xb + 2 * j]) + dot(wb, x4[xb + 2 * j + 1]);
    }
    acc += gacc * as_type<float>((uint)srow[g] << 23);
  }
  return acc;
}

kernel void gemv_mx(device const uchar *wp [[buffer(0)]],
                    device const uchar *sc [[buffer(1)]],
                    device const float *x  [[buffer(2)]],
                    device float *y        [[buffer(3)]],
                    constant int &K        [[buffer(4)]],
                    constant int &NR       [[buffer(5)]],
                    uint tg   [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]]) {
  int row = (int)tg * 4 + (int)sg;
  if (row >= NR) return;
  int NG = K / 32;
  float acc = mxdot_row(wp + (long)row * (K / 2), sc + (long)row * NG,
                        (device const float4 *)x, lane, NG);
  acc = simd_sum(acc);
  if (lane == 0) y[row] = acc;
}

// Batched: NT rows spread over NT/RPG same-shape GEMVs, descriptor per GEMV.
kernel void gemv_mx_batch(device const uchar *slab   [[buffer(0)]],
                          device const GemvDesc *d   [[buffer(1)]],
                          device float *pool         [[buffer(2)]],
                          constant int &K            [[buffer(3)]],
                          constant int &RPG          [[buffer(4)]],
                          constant int &NT           [[buffer(5)]],
                          uint tg   [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg   [[simdgroup_index_in_threadgroup]]) {
  int row = (int)tg * 4 + (int)sg;
  if (row >= NT) return;
  int gi = row / RPG, o = row % RPG;
  GemvDesc dd = d[gi];
  int NG = K / 32;
  float acc = mxdot_row(slab + dd.woff + (long)o * (K / 2),
                        slab + dd.soff + (long)o * NG,
                        (device const float4 *)(pool + dd.xoff), lane, NG);
  acc = simd_sum(acc);
  if (lane == 0) pool[dd.yoff + o] = acc;
}

// SiTU (tools/fast_moe.py): a = 4*tanh(g/4)/(1+exp(-g)); h = a * 25*tanh(u/25)
kernel void situ(device const float *g [[buffer(0)]],
                 device const float *u [[buffer(1)]],
                 device float *h       [[buffer(2)]],
                 uint i [[thread_position_in_grid]]) {
  float gv = g[i], uv = u[i];
  float a = 4.0f * tanh(gv * 0.25f) / (1.0f + exp(-gv));
  h[i] = a * (25.0f * tanh(uv * 0.04f));
}

// v2: 4 consecutive rows per simdgroup (x loads amortized 4x), 16-byte uint4
// weight loads, byte -> float2 via 256-entry constant table (BLUT, host-generated).
constant float2 BLUT[256] = {
float2(0.0f,0.0f), float2(0.5f,0.0f), float2(1.0f,0.0f), float2(1.5f,0.0f), float2(2.0f,0.0f), float2(3.0f,0.0f), float2(4.0f,0.0f), float2(6.0f,0.0f),
float2(-0.0f,0.0f), float2(-0.5f,0.0f), float2(-1.0f,0.0f), float2(-1.5f,0.0f), float2(-2.0f,0.0f), float2(-3.0f,0.0f), float2(-4.0f,0.0f), float2(-6.0f,0.0f),
float2(0.0f,0.5f), float2(0.5f,0.5f), float2(1.0f,0.5f), float2(1.5f,0.5f), float2(2.0f,0.5f), float2(3.0f,0.5f), float2(4.0f,0.5f), float2(6.0f,0.5f),
float2(-0.0f,0.5f), float2(-0.5f,0.5f), float2(-1.0f,0.5f), float2(-1.5f,0.5f), float2(-2.0f,0.5f), float2(-3.0f,0.5f), float2(-4.0f,0.5f), float2(-6.0f,0.5f),
float2(0.0f,1.0f), float2(0.5f,1.0f), float2(1.0f,1.0f), float2(1.5f,1.0f), float2(2.0f,1.0f), float2(3.0f,1.0f), float2(4.0f,1.0f), float2(6.0f,1.0f),
float2(-0.0f,1.0f), float2(-0.5f,1.0f), float2(-1.0f,1.0f), float2(-1.5f,1.0f), float2(-2.0f,1.0f), float2(-3.0f,1.0f), float2(-4.0f,1.0f), float2(-6.0f,1.0f),
float2(0.0f,1.5f), float2(0.5f,1.5f), float2(1.0f,1.5f), float2(1.5f,1.5f), float2(2.0f,1.5f), float2(3.0f,1.5f), float2(4.0f,1.5f), float2(6.0f,1.5f),
float2(-0.0f,1.5f), float2(-0.5f,1.5f), float2(-1.0f,1.5f), float2(-1.5f,1.5f), float2(-2.0f,1.5f), float2(-3.0f,1.5f), float2(-4.0f,1.5f), float2(-6.0f,1.5f),
float2(0.0f,2.0f), float2(0.5f,2.0f), float2(1.0f,2.0f), float2(1.5f,2.0f), float2(2.0f,2.0f), float2(3.0f,2.0f), float2(4.0f,2.0f), float2(6.0f,2.0f),
float2(-0.0f,2.0f), float2(-0.5f,2.0f), float2(-1.0f,2.0f), float2(-1.5f,2.0f), float2(-2.0f,2.0f), float2(-3.0f,2.0f), float2(-4.0f,2.0f), float2(-6.0f,2.0f),
float2(0.0f,3.0f), float2(0.5f,3.0f), float2(1.0f,3.0f), float2(1.5f,3.0f), float2(2.0f,3.0f), float2(3.0f,3.0f), float2(4.0f,3.0f), float2(6.0f,3.0f),
float2(-0.0f,3.0f), float2(-0.5f,3.0f), float2(-1.0f,3.0f), float2(-1.5f,3.0f), float2(-2.0f,3.0f), float2(-3.0f,3.0f), float2(-4.0f,3.0f), float2(-6.0f,3.0f),
float2(0.0f,4.0f), float2(0.5f,4.0f), float2(1.0f,4.0f), float2(1.5f,4.0f), float2(2.0f,4.0f), float2(3.0f,4.0f), float2(4.0f,4.0f), float2(6.0f,4.0f),
float2(-0.0f,4.0f), float2(-0.5f,4.0f), float2(-1.0f,4.0f), float2(-1.5f,4.0f), float2(-2.0f,4.0f), float2(-3.0f,4.0f), float2(-4.0f,4.0f), float2(-6.0f,4.0f),
float2(0.0f,6.0f), float2(0.5f,6.0f), float2(1.0f,6.0f), float2(1.5f,6.0f), float2(2.0f,6.0f), float2(3.0f,6.0f), float2(4.0f,6.0f), float2(6.0f,6.0f),
float2(-0.0f,6.0f), float2(-0.5f,6.0f), float2(-1.0f,6.0f), float2(-1.5f,6.0f), float2(-2.0f,6.0f), float2(-3.0f,6.0f), float2(-4.0f,6.0f), float2(-6.0f,6.0f),
float2(0.0f,-0.0f), float2(0.5f,-0.0f), float2(1.0f,-0.0f), float2(1.5f,-0.0f), float2(2.0f,-0.0f), float2(3.0f,-0.0f), float2(4.0f,-0.0f), float2(6.0f,-0.0f),
float2(-0.0f,-0.0f), float2(-0.5f,-0.0f), float2(-1.0f,-0.0f), float2(-1.5f,-0.0f), float2(-2.0f,-0.0f), float2(-3.0f,-0.0f), float2(-4.0f,-0.0f), float2(-6.0f,-0.0f),
float2(0.0f,-0.5f), float2(0.5f,-0.5f), float2(1.0f,-0.5f), float2(1.5f,-0.5f), float2(2.0f,-0.5f), float2(3.0f,-0.5f), float2(4.0f,-0.5f), float2(6.0f,-0.5f),
float2(-0.0f,-0.5f), float2(-0.5f,-0.5f), float2(-1.0f,-0.5f), float2(-1.5f,-0.5f), float2(-2.0f,-0.5f), float2(-3.0f,-0.5f), float2(-4.0f,-0.5f), float2(-6.0f,-0.5f),
float2(0.0f,-1.0f), float2(0.5f,-1.0f), float2(1.0f,-1.0f), float2(1.5f,-1.0f), float2(2.0f,-1.0f), float2(3.0f,-1.0f), float2(4.0f,-1.0f), float2(6.0f,-1.0f),
float2(-0.0f,-1.0f), float2(-0.5f,-1.0f), float2(-1.0f,-1.0f), float2(-1.5f,-1.0f), float2(-2.0f,-1.0f), float2(-3.0f,-1.0f), float2(-4.0f,-1.0f), float2(-6.0f,-1.0f),
float2(0.0f,-1.5f), float2(0.5f,-1.5f), float2(1.0f,-1.5f), float2(1.5f,-1.5f), float2(2.0f,-1.5f), float2(3.0f,-1.5f), float2(4.0f,-1.5f), float2(6.0f,-1.5f),
float2(-0.0f,-1.5f), float2(-0.5f,-1.5f), float2(-1.0f,-1.5f), float2(-1.5f,-1.5f), float2(-2.0f,-1.5f), float2(-3.0f,-1.5f), float2(-4.0f,-1.5f), float2(-6.0f,-1.5f),
float2(0.0f,-2.0f), float2(0.5f,-2.0f), float2(1.0f,-2.0f), float2(1.5f,-2.0f), float2(2.0f,-2.0f), float2(3.0f,-2.0f), float2(4.0f,-2.0f), float2(6.0f,-2.0f),
float2(-0.0f,-2.0f), float2(-0.5f,-2.0f), float2(-1.0f,-2.0f), float2(-1.5f,-2.0f), float2(-2.0f,-2.0f), float2(-3.0f,-2.0f), float2(-4.0f,-2.0f), float2(-6.0f,-2.0f),
float2(0.0f,-3.0f), float2(0.5f,-3.0f), float2(1.0f,-3.0f), float2(1.5f,-3.0f), float2(2.0f,-3.0f), float2(3.0f,-3.0f), float2(4.0f,-3.0f), float2(6.0f,-3.0f),
float2(-0.0f,-3.0f), float2(-0.5f,-3.0f), float2(-1.0f,-3.0f), float2(-1.5f,-3.0f), float2(-2.0f,-3.0f), float2(-3.0f,-3.0f), float2(-4.0f,-3.0f), float2(-6.0f,-3.0f),
float2(0.0f,-4.0f), float2(0.5f,-4.0f), float2(1.0f,-4.0f), float2(1.5f,-4.0f), float2(2.0f,-4.0f), float2(3.0f,-4.0f), float2(4.0f,-4.0f), float2(6.0f,-4.0f),
float2(-0.0f,-4.0f), float2(-0.5f,-4.0f), float2(-1.0f,-4.0f), float2(-1.5f,-4.0f), float2(-2.0f,-4.0f), float2(-3.0f,-4.0f), float2(-4.0f,-4.0f), float2(-6.0f,-4.0f),
float2(0.0f,-6.0f), float2(0.5f,-6.0f), float2(1.0f,-6.0f), float2(1.5f,-6.0f), float2(2.0f,-6.0f), float2(3.0f,-6.0f), float2(4.0f,-6.0f), float2(6.0f,-6.0f),
float2(-0.0f,-6.0f), float2(-0.5f,-6.0f), float2(-1.0f,-6.0f), float2(-1.5f,-6.0f), float2(-2.0f,-6.0f), float2(-3.0f,-6.0f), float2(-4.0f,-6.0f), float2(-6.0f,-6.0f),
};
inline float dot_group16(uint4 w, thread const float2 *xv) {
  float g = 0.0f; uint b;
  b = w.x; g += dot(BLUT[b & 255], xv[0]);  g += dot(BLUT[(b >> 8) & 255], xv[1]);
           g += dot(BLUT[(b >> 16) & 255], xv[2]); g += dot(BLUT[b >> 24], xv[3]);
  b = w.y; g += dot(BLUT[b & 255], xv[4]);  g += dot(BLUT[(b >> 8) & 255], xv[5]);
           g += dot(BLUT[(b >> 16) & 255], xv[6]); g += dot(BLUT[b >> 24], xv[7]);
  b = w.z; g += dot(BLUT[b & 255], xv[8]);  g += dot(BLUT[(b >> 8) & 255], xv[9]);
           g += dot(BLUT[(b >> 16) & 255], xv[10]); g += dot(BLUT[b >> 24], xv[11]);
  b = w.w; g += dot(BLUT[b & 255], xv[12]); g += dot(BLUT[(b >> 8) & 255], xv[13]);
           g += dot(BLUT[(b >> 16) & 255], xv[14]); g += dot(BLUT[b >> 24], xv[15]);
  return g;
}
kernel void gemv_mx4(device const uchar *slab   [[buffer(0)]],
                     device const GemvDesc *d   [[buffer(1)]],
                     device float *pool         [[buffer(2)]],
                     constant int &K            [[buffer(3)]],
                     constant int &RPG          [[buffer(4)]],
                     constant int &NT           [[buffer(5)]],
                     uint tg   [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg   [[simdgroup_index_in_threadgroup]]) {
  int r0 = ((int)tg * 4 + (int)sg) * 4;       // 4 rows per simdgroup
  if (r0 >= NT) return;
  int gi = r0 / RPG, o0 = r0 % RPG;           // RPG % 4 == 0 -> no straddle
  GemvDesc dd = d[gi];
  int NG = K / 32, rb = K / 2, rstep16 = rb / 16;
  device const uint4 *w16 = (device const uint4 *)(slab + dd.woff + (long)o0 * rb);
  device const uchar *s0 = slab + dd.soff + (long)o0 * NG;
  device const float2 *x2 = (device const float2 *)(pool + dd.xoff);
  float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  for (int g = (int)lane; g < NG; g += 32) {
    float2 xv[16];
    device const float2 *xg = x2 + g * 16;
    for (int j = 0; j < 16; ++j) xv[j] = xg[j];
    a0 += dot_group16(w16[0 * rstep16 + g], xv) * as_type<float>((uint)s0[0 * NG + g] << 23);
    a1 += dot_group16(w16[1 * rstep16 + g], xv) * as_type<float>((uint)s0[1 * NG + g] << 23);
    a2 += dot_group16(w16[2 * rstep16 + g], xv) * as_type<float>((uint)s0[2 * NG + g] << 23);
    a3 += dot_group16(w16[3 * rstep16 + g], xv) * as_type<float>((uint)s0[3 * NG + g] << 23);
  }
  a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
  if (lane == 0) {
    device float *y = pool + dd.yoff + o0;
    y[0] = a0; y[1] = a1; y[2] = a2; y[3] = a3;
  }
}

// v3: fused w1+w3+SiTU stage. Each simdgroup computes 4 gate rows AND the 4
// matching up rows (8 row-dots sharing one set of x loads), then lane 0 applies
// SiTU and writes h directly. Triple becomes 2 dispatches: this then w2 (v2).
struct GluDesc { uint w1off; uint s1off; uint w3off; uint s3off; uint xoff; uint hoff; };
kernel void gemv_mx4_glu(device const uchar *slab [[buffer(0)]],
                         device const GluDesc *d  [[buffer(1)]],
                         device float *pool       [[buffer(2)]],
                         constant int &K          [[buffer(3)]],
                         constant int &RPG        [[buffer(4)]],
                         constant int &NT         [[buffer(5)]],
                         uint tg   [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg   [[simdgroup_index_in_threadgroup]]) {
  int r0 = ((int)tg * 4 + (int)sg) * 4;
  if (r0 >= NT) return;
  int gi = r0 / RPG, o0 = r0 % RPG;
  GluDesc dd = d[gi];
  int NG = K / 32, rb = K / 2, rstep16 = rb / 16;
  device const uint4 *g16 = (device const uint4 *)(slab + dd.w1off + (long)o0 * rb);
  device const uint4 *u16 = (device const uint4 *)(slab + dd.w3off + (long)o0 * rb);
  device const uchar *gs = slab + dd.s1off + (long)o0 * NG;
  device const uchar *us = slab + dd.s3off + (long)o0 * NG;
  device const float2 *x2 = (device const float2 *)(pool + dd.xoff);
  float ag0 = 0, ag1 = 0, ag2 = 0, ag3 = 0, au0 = 0, au1 = 0, au2 = 0, au3 = 0;
  for (int g = (int)lane; g < NG; g += 32) {
    float2 xv[16];
    device const float2 *xg = x2 + g * 16;
    for (int j = 0; j < 16; ++j) xv[j] = xg[j];
    ag0 += dot_group16(g16[0 * rstep16 + g], xv) * as_type<float>((uint)gs[0 * NG + g] << 23);
    ag1 += dot_group16(g16[1 * rstep16 + g], xv) * as_type<float>((uint)gs[1 * NG + g] << 23);
    ag2 += dot_group16(g16[2 * rstep16 + g], xv) * as_type<float>((uint)gs[2 * NG + g] << 23);
    ag3 += dot_group16(g16[3 * rstep16 + g], xv) * as_type<float>((uint)gs[3 * NG + g] << 23);
    au0 += dot_group16(u16[0 * rstep16 + g], xv) * as_type<float>((uint)us[0 * NG + g] << 23);
    au1 += dot_group16(u16[1 * rstep16 + g], xv) * as_type<float>((uint)us[1 * NG + g] << 23);
    au2 += dot_group16(u16[2 * rstep16 + g], xv) * as_type<float>((uint)us[2 * NG + g] << 23);
    au3 += dot_group16(u16[3 * rstep16 + g], xv) * as_type<float>((uint)us[3 * NG + g] << 23);
  }
  ag0 = simd_sum(ag0); ag1 = simd_sum(ag1); ag2 = simd_sum(ag2); ag3 = simd_sum(ag3);
  au0 = simd_sum(au0); au1 = simd_sum(au1); au2 = simd_sum(au2); au3 = simd_sum(au3);
  if (lane == 0) {
    device float *h = pool + dd.hoff + o0;
    float gv[4] = {ag0, ag1, ag2, ag3}, uv[4] = {au0, au1, au2, au3};
    for (int r = 0; r < 4; ++r) {
      float a = 4.0f * tanh(gv[r] * 0.25f) / (1.0f + exp(-gv[r]));
      h[r] = a * (25.0f * tanh(uv[r] * 0.04f));
    }
  }
}

// raw read-bandwidth ceiling: stream the whole slab as uint4, defeat DCE cheaply.
kernel void bw_read(device const uint4 *w [[buffer(0)]],
                    device float *out     [[buffer(1)]],
                    constant uint &n16    [[buffer(2)]],
                    constant uint &total  [[buffer(3)]],
                    uint tid  [[thread_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]]) {
  uint acc = 0;
  for (uint j = tid; j < n16; j += total) { uint4 v = w[j]; acc += v.x + v.y + v.z + v.w; }
  float f = simd_sum((float)(acc & 1023));
  if (lane == 0) out[tid / 32 & 4095] = f;
}
