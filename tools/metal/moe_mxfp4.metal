// Kimi-K3 MoE expert stack on the GPU: fused MXFP4 dequant + GEMV, batched so one
// dispatch covers every selected expert of a layer.  Runtime-compiled by
// tools/metal_moe.mm via newLibraryWithSource (no Xcode toolchain needed).
//
// Decode shape per expert (hidden H=3584, intermediate I=3072):
//     y1 = w1 @ x     w1 [I,H]   x [H]
//     y3 = w3 @ x     w3 [I,H]
//     h  = SiTU(y1, y3)                       [I]
//     ye = w2 @ h     w2 [H,I]                [H]
//     out = sum_e weight_e * ye
//
// Weights are the verbatim on-disk MXFP4 shard span (17,547,264 B per expert):
//     w1_p 5505024 | w1_s 344064 | w2_p 5505024 | w2_s 344064 | w3_p 5505024 | w3_s 344064
// packed = two e2m1 nibbles per byte (low nibble first), scale = one e8m0 byte per
// 32 elements.  Dequant happens in-register; fp32 weights are never materialized.
//
// Kernel structure (inherited from the validated tools/metal/kernels.metal prototype,
// relL2 9e-7 vs the CPU reference, 150 GB/s):
//   * one simdgroup owns 4 consecutive output rows -> x loads amortized 4x
//   * 16-byte uint4 weight loads; byte -> float2 via a 256-entry constant table
//   * e8m0 applied as 2^(b-127) == as_type<float>(b << 23)
//   * fp32 accumulation throughout, simd_sum reduction, lane 0 stores
//
// Two batching flavours, both live in one command buffer:
//   *_batch : bindless.  ExpertRef[] is an argument buffer of raw GPU addresses
//             (Tier-2 argument buffers); ONE dispatch spans all experts.
//   *_one   : the expert blob is bound directly; the host issues one dispatch per
//             expert into a single concurrent encoder.  Fallback for devices
//             without Tier-2 argument buffers, and an A/B knob for benchmarking.

#include <metal_stdlib>
using namespace metal;

// byte -> (E2M1[b & 15], E2M1[b >> 4]); low nibble first, matching tools/mxfp4.py
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

// Layout of one expert blob, and the problem dims.  Filled by the host; kept in a
// struct so a future shape change is a host-side edit only.
struct MoeDims {
  uint n_experts;   // experts batched in this dispatch
  uint H;           // hidden / x length            (3584)
  uint I;           // intermediate / h length      (3072)
  uint w1p, w1s;    // byte offsets inside an expert blob
  uint w2p, w2s;
  uint w3p, w3s;
  uint expert;      // *_one variants: which expert slot this dispatch writes
  uint nt;          // total output rows covered by this dispatch
};

struct ExpertRef { device const uchar *blob; };   // argument buffer entry (8 B, Tier 2)

// SiTU, Moonshot modeling code, beta=4, linear_beta=25 — evaluated in fp32 with the
// precise transcendentals so it tracks the numpy reference in tools/fast_moe.py:
//     a = beta*tanh(gate/beta)*sigmoid(gate);  up' = linear_beta*tanh(up/linear_beta)
inline float situ(float gate, float up) {
  float a = 4.0f * precise::tanh(gate * 0.25f) / (1.0f + precise::exp(-gate));
  return a * (25.0f * precise::tanh(up * 0.04f));
}

// 32 MXFP4 elements (one scale group) x 32 activations, dequantized in-register.
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

// ---------------------------------------------------------------------------
// stage 1 body: 4 gate rows + the 4 matching up rows on one set of x loads,
// SiTU applied inline, h written straight out.  o0 = first row, must be % 4 == 0.
// ---------------------------------------------------------------------------
inline void glu_rows(device const uchar *blob, constant MoeDims &D,
                     device const float *x, device float *h_out,
                     uint o0, uint lane) {
  const uint K = D.H, NG = K / 32, rb = K / 2, rstep16 = rb / 16;
  device const uint4 *g16 = (device const uint4 *)(blob + D.w1p + (ulong)o0 * rb);
  device const uint4 *u16 = (device const uint4 *)(blob + D.w3p + (ulong)o0 * rb);
  device const uchar *gs = blob + D.w1s + (ulong)o0 * NG;
  device const uchar *us = blob + D.w3s + (ulong)o0 * NG;
  device const float2 *x2 = (device const float2 *)x;
  float ag0 = 0, ag1 = 0, ag2 = 0, ag3 = 0, au0 = 0, au1 = 0, au2 = 0, au3 = 0;
  for (uint g = lane; g < NG; g += 32) {
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
    h_out[0] = situ(ag0, au0);
    h_out[1] = situ(ag1, au1);
    h_out[2] = situ(ag2, au2);
    h_out[3] = situ(ag3, au3);
  }
}

// stage 2 body: 4 rows of w2 against h.
inline void w2_rows(device const uchar *blob, constant MoeDims &D,
                    device const float *h, device float *y_out,
                    uint o0, uint lane) {
  const uint K = D.I, NG = K / 32, rb = K / 2, rstep16 = rb / 16;
  device const uint4 *w16 = (device const uint4 *)(blob + D.w2p + (ulong)o0 * rb);
  device const uchar *sc = blob + D.w2s + (ulong)o0 * NG;
  device const float2 *h2 = (device const float2 *)h;
  float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  for (uint g = lane; g < NG; g += 32) {
    float2 xv[16];
    device const float2 *hg = h2 + g * 16;
    for (int j = 0; j < 16; ++j) xv[j] = hg[j];
    a0 += dot_group16(w16[0 * rstep16 + g], xv) * as_type<float>((uint)sc[0 * NG + g] << 23);
    a1 += dot_group16(w16[1 * rstep16 + g], xv) * as_type<float>((uint)sc[1 * NG + g] << 23);
    a2 += dot_group16(w16[2 * rstep16 + g], xv) * as_type<float>((uint)sc[2 * NG + g] << 23);
    a3 += dot_group16(w16[3 * rstep16 + g], xv) * as_type<float>((uint)sc[3 * NG + g] << 23);
  }
  a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
  if (lane == 0) { y_out[0] = a0; y_out[1] = a1; y_out[2] = a2; y_out[3] = a3; }
}

// Row assignment shared by every kernel: threadgroup = 128 threads = 4 simdgroups,
// each simdgroup owns 4 consecutive rows -> 16 rows per threadgroup.
#define ROW0(tg, sg) ((((uint)(tg)) * 4u + (uint)(sg)) * 4u)

// ---------------------------------------------------------------------------
// bindless: ONE dispatch spans every selected expert of the layer.
// grid rows = n_experts * I (stage 1) or n_experts * H (stage 2); I and H are
// multiples of 4, so a simdgroup's 4 rows never straddle an expert boundary.
// ---------------------------------------------------------------------------
kernel void moe_glu_batch(device const ExpertRef *experts [[buffer(0)]],
                          device const float *x           [[buffer(1)]],
                          device float *h                 [[buffer(2)]],
                          constant MoeDims &D             [[buffer(3)]],
                          uint tg   [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg   [[simdgroup_index_in_threadgroup]]) {
  uint r0 = ROW0(tg, sg);
  if (r0 >= D.nt) return;
  uint e = r0 / D.I, o0 = r0 % D.I;
  glu_rows(experts[e].blob, D, x, h + (ulong)e * D.I + o0, o0, lane);
}

kernel void moe_w2_batch(device const ExpertRef *experts [[buffer(0)]],
                         device const float *h           [[buffer(1)]],
                         device float *y                 [[buffer(2)]],
                         constant MoeDims &D             [[buffer(3)]],
                         uint tg   [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg   [[simdgroup_index_in_threadgroup]]) {
  uint r0 = ROW0(tg, sg);
  if (r0 >= D.nt) return;
  uint e = r0 / D.H, o0 = r0 % D.H;
  w2_rows(experts[e].blob, D, h + (ulong)e * D.I, y + (ulong)e * D.H + o0, o0, lane);
}

// ---------------------------------------------------------------------------
// direct-bind: one dispatch per expert, all encoded into a single concurrent
// encoder in the same command buffer.  D.expert selects the h/y slice.
// ---------------------------------------------------------------------------
kernel void moe_glu_one(device const uchar *blob [[buffer(0)]],
                        device const float *x    [[buffer(1)]],
                        device float *h          [[buffer(2)]],
                        constant MoeDims &D      [[buffer(3)]],
                        uint tg   [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg   [[simdgroup_index_in_threadgroup]]) {
  uint o0 = ROW0(tg, sg);
  if (o0 >= D.I) return;
  glu_rows(blob, D, x, h + (ulong)D.expert * D.I + o0, o0, lane);
}

kernel void moe_w2_one(device const uchar *blob [[buffer(0)]],
                       device const float *h    [[buffer(1)]],
                       device float *y          [[buffer(2)]],
                       constant MoeDims &D      [[buffer(3)]],
                       uint tg   [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg   [[simdgroup_index_in_threadgroup]]) {
  uint o0 = ROW0(tg, sg);
  if (o0 >= D.H) return;
  w2_rows(blob, D, h + (ulong)D.expert * D.I, y + (ulong)D.expert * D.H + o0, o0, lane);
}

// ---------------------------------------------------------------------------
// combine: out[i] = sum_e weight[e] * y[e*H + i].  Summed in expert order so the
// fp32 rounding sequence matches the CPU path's `out += w * expert(x)` loop.
// ---------------------------------------------------------------------------
kernel void moe_reduce(device const float *y   [[buffer(0)]],
                       device const float *wts [[buffer(1)]],
                       device float *out       [[buffer(2)]],
                       constant MoeDims &D     [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
  if (i >= D.H) return;
  float acc = 0.0f;
  for (uint e = 0; e < D.n_experts; ++e) acc += wts[e] * y[(ulong)e * D.H + i];
  out[i] = acc;
}
