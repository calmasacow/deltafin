// Fused MXFP4 dequant+GEMV Metal benchmark for Kimi-K3 experts (M1 Max).
// Runtime-compiled MSL (no Xcode toolchain), zero-copy mmap slab, StorageModeShared.
// Build: clang++ -O2 -std=c++17 -fobjc-arc bench.mm -framework Metal -framework Foundation -o bench
// Pattern follows research/colibri/c/backend_metal.mm (one simdgroup per output row,
// 4 simdgroups per threadgroup, group-of-32 MXFP4 unpack via constant table).
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// ---- expert geometry (Kimi-K3): hidden 3584, moe-intermediate 3072 ----
static const int RIN  = 3584;                 // hidden dim (w1/w3 cols, w2 rows)
static const int RMID = 3072;                 // intermediate (w1/w3 rows, w2 cols)
static const int NE   = 16;                   // experts per layer per token (decode)
static const size_t PB = 5505024;             // packed bytes per matrix (all three)
static const size_t SB = 344064;              // scale bytes per matrix
static const size_t OFF_W1P = 0, OFF_W1S = PB;
static const size_t OFF_W3P = PB + SB, OFF_W3S = 2 * PB + SB;
static const size_t OFF_W2P = 2 * (PB + SB), OFF_W2S = 3 * PB + 2 * SB;
static const size_t EXP_STRIDE = 3 * (PB + SB);          // 17547264
static const size_t EXPERT_BYTES = EXP_STRIDE;           // packed+scale consumed per expert

static const char *SHADER = R"MSL(
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
//BLUT_HERE
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
)MSL";

struct GemvDesc { uint32_t woff, soff, xoff, yoff; };

static double now_s() {
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Stats { double med, mn, gpu_med; };
static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void *map_file(const char *path, size_t *len) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) { fprintf(stderr, "open %s failed\n", path); exit(1); }
  struct stat st; fstat(fd, &st);
  *len = (size_t)st.st_size;
  void *p = mmap(NULL, *len, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  close(fd);
  if (p == MAP_FAILED) { fprintf(stderr, "mmap %s failed\n", path); exit(1); }
  return p;
}
static std::vector<float> read_floats(const char *path, size_t n) {
  std::vector<float> v(n);
  FILE *f = fopen(path, "rb");
  if (!f || fread(v.data(), 4, n, f) != n) { fprintf(stderr, "read %s failed\n", path); exit(1); }
  fclose(f);
  return v;
}
static void check(const float *got, const float *ref, size_t n, const char *tag) {
  double maxrel = 0, num = 0, den = 0; size_t argmax = 0;
  double refmax = 0;
  for (size_t i = 0; i < n; i++) refmax = std::max(refmax, (double)fabsf(ref[i]));
  for (size_t i = 0; i < n; i++) {
    double e = (double)got[i] - (double)ref[i];
    num += e * e; den += (double)ref[i] * (double)ref[i];
    double rel = fabs(e) / std::max((double)fabsf(ref[i]), 1e-3 * refmax);
    if (rel > maxrel) { maxrel = rel; argmax = i; }
  }
  printf("  [%s] relL2=%.3e  max-elem-rel=%.3e (i=%zu got=%.6f ref=%.6f)  %s\n",
         tag, sqrt(num / den), maxrel, argmax, got[argmax], ref[argmax],
         (sqrt(num / den) < 1e-4 && maxrel < 1e-3) ? "PASS" : "FAIL");
}

int main() {
  @autoreleasepool {
    // ---- data ----
    size_t slab_len = 0;
    void *slab_host = map_file("slab.bin", &slab_len);
    if (slab_len != NE * EXP_STRIDE) { fprintf(stderr, "slab size mismatch\n"); return 1; }
    auto xh = read_floats("x.bin", RIN);
    auto ref_g = read_floats("ref_g.bin", (size_t)NE * RMID);
    auto ref_y = read_floats("ref_y.bin", (size_t)NE * RIN);

    // ---- device / pipelines ----
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }
    printf("device: %s  unified=%d  maxTG=%lu\n", [[dev name] UTF8String],
           (int)[dev hasUnifiedMemory], (unsigned long)[dev maxThreadsPerThreadgroup].width);
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError *err = nil;
    // generate byte -> float2 dequant table text (low nibble first)
    std::string blut = "constant float2 BLUT[256] = {\n";
    const float L16[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                           -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    for (int b = 0; b < 256; b++) {
      char t[64];
      snprintf(t, sizeof t, "float2(%.1ff,%.1ff),%s", L16[b & 15], L16[b >> 4],
               (b % 8 == 7) ? "\n" : " ");
      blut += t;
    }
    blut += "};\n";
    std::string src(SHADER);
    src.replace(src.find("//BLUT_HERE"), strlen("//BLUT_HERE"), blut);
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                           options:nil error:&err];
    if (!lib) { fprintf(stderr, "compile: %s\n", [[err localizedDescription] UTF8String]); return 1; }
    id<MTLComputePipelineState> pGemv = [dev newComputePipelineStateWithFunction:
                                          [lib newFunctionWithName:@"gemv_mx"] error:&err];
    id<MTLComputePipelineState> pBatch = [dev newComputePipelineStateWithFunction:
                                          [lib newFunctionWithName:@"gemv_mx_batch"] error:&err];
    id<MTLComputePipelineState> pSitu = [dev newComputePipelineStateWithFunction:
                                          [lib newFunctionWithName:@"situ"] error:&err];
    id<MTLComputePipelineState> pBatch4 = [dev newComputePipelineStateWithFunction:
                                          [lib newFunctionWithName:@"gemv_mx4"] error:&err];
    id<MTLComputePipelineState> pBw = [dev newComputePipelineStateWithFunction:
                                          [lib newFunctionWithName:@"bw_read"] error:&err];
    id<MTLComputePipelineState> pGlu = [dev newComputePipelineStateWithFunction:
                                          [lib newFunctionWithName:@"gemv_mx4_glu"] error:&err];
    if (!pGemv || !pBatch || !pSitu || !pBatch4 || !pBw || !pGlu) { fprintf(stderr, "pipeline fail\n"); return 1; }
    printf("simd width=%lu\n", (unsigned long)[pGemv threadExecutionWidth]);

    // ---- buffers ----
    // slab: zero-copy mmap wrap (page-aligned, len multiple of 16K by construction)
    id<MTLBuffer> bSlab = [dev newBufferWithBytesNoCopy:slab_host length:slab_len
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
    if (!bSlab) {  // fallback: copy
      bSlab = [dev newBufferWithBytes:slab_host length:slab_len
                              options:MTLResourceStorageModeShared];
      printf("slab: copied (no-copy wrap failed)\n");
    }
    // float pool: x | G[NE*RMID] | U | H | Y[NE*RIN]
    const uint32_t OX = 0, OG = RIN, OU = OG + NE * RMID, OH = OU + NE * RMID,
                   OY = OH + NE * RMID, POOL_N = OY + NE * RIN;
    id<MTLBuffer> bPool = [dev newBufferWithLength:POOL_N * 4
                                           options:MTLResourceStorageModeShared];
    memcpy(bPool.contents, xh.data(), RIN * 4);
    // descriptor tables
    std::vector<GemvDesc> d13(2 * NE), d2(NE);
    for (int e = 0; e < NE; e++) {
      size_t eb = (size_t)e * EXP_STRIDE;
      d13[2 * e + 0] = {(uint32_t)(eb + OFF_W1P), (uint32_t)(eb + OFF_W1S), OX,
                        OG + (uint32_t)e * RMID};
      d13[2 * e + 1] = {(uint32_t)(eb + OFF_W3P), (uint32_t)(eb + OFF_W3S), OX,
                        OU + (uint32_t)e * RMID};
      d2[e] = {(uint32_t)(eb + OFF_W2P), (uint32_t)(eb + OFF_W2S),
               OH + (uint32_t)e * RMID, OY + (uint32_t)e * RIN};
    }
    id<MTLBuffer> bD13 = [dev newBufferWithBytes:d13.data() length:d13.size() * sizeof(GemvDesc)
                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> bD2 = [dev newBufferWithBytes:d2.data() length:d2.size() * sizeof(GemvDesc)
                                        options:MTLResourceStorageModeShared];
    struct GluDesc { uint32_t w1, s1, w3, s3, xoff, hoff; };
    std::vector<GluDesc> dglu(NE);
    for (int e = 0; e < NE; e++) {
      size_t eb = (size_t)e * EXP_STRIDE;
      dglu[e] = {(uint32_t)(eb + OFF_W1P), (uint32_t)(eb + OFF_W1S),
                 (uint32_t)(eb + OFF_W3P), (uint32_t)(eb + OFF_W3S), OX,
                 OH + (uint32_t)e * RMID};
    }
    id<MTLBuffer> bDGlu = [dev newBufferWithBytes:dglu.data()
                                           length:dglu.size() * sizeof(GluDesc)
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> bYs = [dev newBufferWithLength:RMID * 4
                                         options:MTLResourceStorageModeShared]; // single-gemv out

    MTLSize tgsz = MTLSizeMake(128, 1, 1);   // 4 simdgroups
    auto rows_tg = [](int rows) { return MTLSizeMake((rows + 3) / 4, 1, 1); };
    int K_IN = RIN, K_MID = RMID, NR_MID = RMID;
    int NT13 = 2 * NE * RMID, RPG13 = RMID, NT2 = NE * RIN, RPG2 = RIN;

    auto encSingle = [&](id<MTLComputeCommandEncoder> enc) {
      [enc setComputePipelineState:pGemv];
      [enc setBuffer:bSlab offset:OFF_W1P atIndex:0];   // expert 0 w1
      [enc setBuffer:bSlab offset:OFF_W1S atIndex:1];
      [enc setBuffer:bPool offset:0 atIndex:2];
      [enc setBuffer:bYs offset:0 atIndex:3];
      [enc setBytes:&K_IN length:4 atIndex:4];
      [enc setBytes:&NR_MID length:4 atIndex:5];
      [enc dispatchThreadgroups:rows_tg(RMID) threadsPerThreadgroup:tgsz];
    };
    auto encBatch13 = [&](id<MTLComputeCommandEncoder> enc) {
      [enc setComputePipelineState:pBatch];
      [enc setBuffer:bSlab offset:0 atIndex:0];
      [enc setBuffer:bD13 offset:0 atIndex:1];
      [enc setBuffer:bPool offset:0 atIndex:2];
      [enc setBytes:&K_IN length:4 atIndex:3];
      [enc setBytes:&RPG13 length:4 atIndex:4];
      [enc setBytes:&NT13 length:4 atIndex:5];
      [enc dispatchThreadgroups:rows_tg(NT13) threadsPerThreadgroup:tgsz];
    };
    auto encSitu = [&](id<MTLComputeCommandEncoder> enc) {
      [enc setComputePipelineState:pSitu];
      [enc setBuffer:bPool offset:OG * 4 atIndex:0];
      [enc setBuffer:bPool offset:OU * 4 atIndex:1];
      [enc setBuffer:bPool offset:OH * 4 atIndex:2];
      [enc dispatchThreads:MTLSizeMake(NE * RMID, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    auto encBatch2 = [&](id<MTLComputeCommandEncoder> enc) {
      [enc setComputePipelineState:pBatch];
      [enc setBuffer:bSlab offset:0 atIndex:0];
      [enc setBuffer:bD2 offset:0 atIndex:1];
      [enc setBuffer:bPool offset:0 atIndex:2];
      [enc setBytes:&K_MID length:4 atIndex:3];
      [enc setBytes:&RPG2 length:4 atIndex:4];
      [enc setBytes:&NT2 length:4 atIndex:5];
      [enc dispatchThreadgroups:rows_tg(NT2) threadsPerThreadgroup:tgsz];
    };
    // 48 individual gemv_mx dispatches (per-dispatch overhead probe)
    auto enc48 = [&](id<MTLComputeCommandEncoder> enc) {
      [enc setComputePipelineState:pGemv];
      [enc setBuffer:bPool offset:0 atIndex:2];
      for (int e = 0; e < NE; e++) {
        size_t eb = (size_t)e * EXP_STRIDE;
        struct { size_t wp, ws; uint32_t yoff; int K, NR; } m[3] = {
            {eb + OFF_W1P, eb + OFF_W1S, OG + (uint32_t)e * RMID, RIN, RMID},
            {eb + OFF_W3P, eb + OFF_W3S, OU + (uint32_t)e * RMID, RIN, RMID},
            {eb + OFF_W2P, eb + OFF_W2S, OY + (uint32_t)e * RIN, RMID, RIN}};
        for (auto &mm : m) {
          [enc setBuffer:bSlab offset:mm.wp atIndex:0];
          [enc setBuffer:bSlab offset:mm.ws atIndex:1];
          [enc setBuffer:bPool offset:mm.yoff * 4 atIndex:3];
          [enc setBytes:&mm.K length:4 atIndex:4];
          [enc setBytes:&mm.NR length:4 atIndex:5];
          [enc dispatchThreadgroups:rows_tg(mm.NR) threadsPerThreadgroup:tgsz];
        }
      }
    };

    // v2 kernel encoders: 16 rows per TG (4 simdgroups x 4 rows)
    auto rows_tg4 = [](int rows) { return MTLSizeMake((rows + 15) / 16, 1, 1); };
    auto encBatch13v2 = [&](id<MTLComputeCommandEncoder> enc) {
      [enc setComputePipelineState:pBatch4];
      [enc setBuffer:bSlab offset:0 atIndex:0];
      [enc setBuffer:bD13 offset:0 atIndex:1];
      [enc setBuffer:bPool offset:0 atIndex:2];
      [enc setBytes:&K_IN length:4 atIndex:3];
      [enc setBytes:&RPG13 length:4 atIndex:4];
      [enc setBytes:&NT13 length:4 atIndex:5];
      [enc dispatchThreadgroups:rows_tg4(NT13) threadsPerThreadgroup:tgsz];
    };
    auto encBatch2v2 = [&](id<MTLComputeCommandEncoder> enc) {
      [enc setComputePipelineState:pBatch4];
      [enc setBuffer:bSlab offset:0 atIndex:0];
      [enc setBuffer:bD2 offset:0 atIndex:1];
      [enc setBuffer:bPool offset:0 atIndex:2];
      [enc setBytes:&K_MID length:4 atIndex:3];
      [enc setBytes:&RPG2 length:4 atIndex:4];
      [enc setBytes:&NT2 length:4 atIndex:5];
      [enc dispatchThreadgroups:rows_tg4(NT2) threadsPerThreadgroup:tgsz];
    };
    // v3: fused w1w3+situ over NE experts (NT = NE*RMID gate rows)
    int NTG = NE * RMID;
    auto encGlu = [&](id<MTLComputeCommandEncoder> enc) {
      [enc setComputePipelineState:pGlu];
      [enc setBuffer:bSlab offset:0 atIndex:0];
      [enc setBuffer:bDGlu offset:0 atIndex:1];
      [enc setBuffer:bPool offset:0 atIndex:2];
      [enc setBytes:&K_IN length:4 atIndex:3];
      [enc setBytes:&RPG13 length:4 atIndex:4];
      [enc setBytes:&NTG length:4 atIndex:5];
      [enc dispatchThreadgroups:rows_tg4(NTG) threadsPerThreadgroup:tgsz];
    };
    uint32_t n16 = (uint32_t)(slab_len / 16), bw_threads = 65536;
    auto encBw = [&](id<MTLComputeCommandEncoder> enc) {
      [enc setComputePipelineState:pBw];
      [enc setBuffer:bSlab offset:0 atIndex:0];
      [enc setBuffer:bPool offset:(OG + 100000) * 4 atIndex:1];  // scratch in pool tail
      [enc setBytes:&n16 length:4 atIndex:2];
      [enc setBytes:&bw_threads length:4 atIndex:3];
      [enc dispatchThreads:MTLSizeMake(bw_threads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    auto runCB = [&](void (^body)(id<MTLComputeCommandEncoder>), double *gpu_s) {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      if (body) {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        body(enc);
        [enc endEncoding];
      }
      double t0 = now_s();
      [cb commit];
      [cb waitUntilCompleted];
      double t = now_s() - t0;
      if (gpu_s) *gpu_s = [cb GPUEndTime] - [cb GPUStartTime];
      return t;
    };
    auto bench = [&](const char *tag, int warm, int iters, size_t bytes,
                     void (^body)(id<MTLComputeCommandEncoder>)) {
      for (int i = 0; i < warm; i++) runCB(body, NULL);
      std::vector<double> wall(iters), gpu(iters);
      for (int i = 0; i < iters; i++) wall[i] = runCB(body, &gpu[i]);
      Stats s{median(wall), *std::min_element(wall.begin(), wall.end()), median(gpu)};
      printf("%-34s wall med %8.1f us (min %8.1f)  gpu med %8.1f us", tag,
             s.med * 1e6, s.mn * 1e6, s.gpu_med * 1e6);
      if (bytes)
        printf("  |  %7.1f GB/s wall, %7.1f GB/s gpu", bytes / s.med / 1e9,
               bytes / s.gpu_med / 1e9);
      printf("\n");
      return s;
    };

    // pipelined: keep P command buffers in flight, wait only on the last.
    auto benchPipe = [&](const char *tag, int P, int reps, size_t bytes_per_cb,
                         void (^body)(id<MTLComputeCommandEncoder>)) {
      for (int i = 0; i < 3; i++) runCB(body, NULL);
      std::vector<double> per(reps);
      for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        id<MTLCommandBuffer> last = nil;
        for (int i = 0; i < P; i++) {
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          body(enc);
          [enc endEncoding];
          [cb commit];
          last = cb;
        }
        [last waitUntilCompleted];
        per[r] = (now_s() - t0) / P;
      }
      double med = median(per);
      printf("%-34s per-CB med %8.1f us  |  %7.1f GB/s sustained\n", tag, med * 1e6,
             bytes_per_cb / med / 1e9);
      return med;
    };

    // ---- validation ----
    printf("\n== validation ==\n");
    runCB(encSingle, NULL);
    check((float *)bYs.contents, ref_g.data(), RMID, "single gemv w1@x, expert0");
    runCB(^(id<MTLComputeCommandEncoder> e) { encBatch13(e); encSitu(e); encBatch2(e); }, NULL);
    check((float *)bPool.contents + OY, ref_y.data(), (size_t)NE * RIN, "full triple x16 experts");
    // also verify the 48-dispatch path writes identical G
    runCB(enc48, NULL);
    check((float *)bPool.contents + OG, ref_g.data(), (size_t)NE * RMID, "48-dispatch G block");
    // v2 kernel: full triple
    memset((char *)bPool.contents + OG * 4, 0, (POOL_N - OG) * 4);
    runCB(^(id<MTLComputeCommandEncoder> e) { encBatch13v2(e); encSitu(e); encBatch2v2(e); }, NULL);
    check((float *)bPool.contents + OY, ref_y.data(), (size_t)NE * RIN, "v2 full triple x16");
    // v3 fused glu: full triple in 2 dispatches
    memset((char *)bPool.contents + OG * 4, 0, (POOL_N - OG) * 4);
    runCB(^(id<MTLComputeCommandEncoder> e) { encGlu(e); encBatch2v2(e); }, NULL);
    check((float *)bPool.contents + OY, ref_y.data(), (size_t)NE * RIN, "v3 fused triple x16");

    // ---- benchmarks ----
    printf("\n== benchmarks (median over iters; bytes = packed+scale consumed) ==\n");
    bench("empty command buffer", 20, 200, 0, nil);
    bench("situ only (49k elems)", 10, 100, 0, ^(id<MTLComputeCommandEncoder> e) { encSitu(e); });
    size_t oneG = PB + SB;
    bench("single GEMV w1 [3072x3584]", 20, 200, oneG,
          ^(id<MTLComputeCommandEncoder> e) { encSingle(e); });
    size_t all16 = (size_t)NE * EXPERT_BYTES;
    bench("batch48 2 dispatches (no situ)", 10, 100, all16,
          ^(id<MTLComputeCommandEncoder> e) { encBatch13(e); encBatch2(e); });
    bench("full triple w1w3->situ->w2 x16", 10, 100, all16,
          ^(id<MTLComputeCommandEncoder> e) { encBatch13(e); encSitu(e); encBatch2(e); });
    bench("48 individual dispatches", 10, 100, all16,
          ^(id<MTLComputeCommandEncoder> e) { enc48(e); });
    bench("v2 batch48 2 dispatches", 10, 100, all16,
          ^(id<MTLComputeCommandEncoder> e) { encBatch13v2(e); encBatch2v2(e); });
    bench("v2 full triple x16", 10, 100, all16,
          ^(id<MTLComputeCommandEncoder> e) { encBatch13v2(e); encSitu(e); encBatch2v2(e); });
    bench("v3 fused triple x16 (2 disp)", 10, 100, all16,
          ^(id<MTLComputeCommandEncoder> e) { encGlu(e); encBatch2v2(e); });
    bench("raw uint4 read of slab", 5, 50, slab_len,
          ^(id<MTLComputeCommandEncoder> e) { encBw(e); });
    printf("\n== pipelined (8 CBs in flight; sustained, clock-ramp-free) ==\n");
    benchPipe("pipe single GEMV w1", 8, 25, PB + SB,
              ^(id<MTLComputeCommandEncoder> e) { encSingle(e); });
    benchPipe("pipe v1 full triple x16", 8, 12, all16,
              ^(id<MTLComputeCommandEncoder> e) { encBatch13(e); encSitu(e); encBatch2(e); });
    benchPipe("pipe v2 full triple x16", 8, 12, all16,
              ^(id<MTLComputeCommandEncoder> e) { encBatch13v2(e); encSitu(e); encBatch2v2(e); });
    benchPipe("pipe v3 fused triple x16", 8, 12, all16,
              ^(id<MTLComputeCommandEncoder> e) { encGlu(e); encBatch2v2(e); });
    benchPipe("pipe raw uint4 read", 8, 12, slab_len,
              ^(id<MTLComputeCommandEncoder> e) { encBw(e); });
    size_t one_expert = EXPERT_BYTES;
    bench("one expert triple (3 disp+situ)", 10, 200, one_expert,
          ^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pBatch];
            [e setBuffer:bSlab offset:0 atIndex:0];
            [e setBuffer:bD13 offset:0 atIndex:1];
            [e setBuffer:bPool offset:0 atIndex:2];
            int nt = 2 * RMID;
            [e setBytes:&K_IN length:4 atIndex:3];
            [e setBytes:&RPG13 length:4 atIndex:4];
            [e setBytes:&nt length:4 atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake((nt + 3) / 4, 1, 1) threadsPerThreadgroup:tgsz];
            [e setComputePipelineState:pSitu];
            [e setBuffer:bPool offset:OG * 4 atIndex:0];
            [e setBuffer:bPool offset:OU * 4 atIndex:1];
            [e setBuffer:bPool offset:OH * 4 atIndex:2];
            [e dispatchThreads:MTLSizeMake(RMID, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e setComputePipelineState:pBatch];
            [e setBuffer:bSlab offset:0 atIndex:0];
            [e setBuffer:bD2 offset:0 atIndex:1];
            [e setBuffer:bPool offset:0 atIndex:2];
            int nt2 = RIN;
            [e setBytes:&K_MID length:4 atIndex:3];
            [e setBytes:&RPG2 length:4 atIndex:4];
            [e setBytes:&nt2 length:4 atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake((nt2 + 3) / 4, 1, 1) threadsPerThreadgroup:tgsz];
          });
    printf("\nexpert bytes: %zu/expert, %zu/16 experts\n", EXPERT_BYTES, all16);
  }
  return 0;
}
