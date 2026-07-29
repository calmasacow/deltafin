// Kimi-K3 Metal MoE bridge — host side for tools/metal/moe_mxfp4.metal.
//
// Runs a whole layer's selected experts (MXFP4 dequant + w1/w3 GEMV + SiTU + w2
// GEMV + weighted sum) on the GPU from one command buffer.  Persistent device,
// queue, library and pipelines; expert bytes are wrapped zero-copy where the
// pointer allows it, so a cached expert costs no host-side traffic at all.
//
// BUILD (do this by hand; no Makefile is touched):
//   clang++ -O2 -std=c++17 -fobjc-arc -dynamiclib \
//       tools/metal_moe.mm -framework Metal -framework Foundation \
//       -o tools/libk3metalmoe.dylib
//
// The MSL is compiled at run time via newLibraryWithSource, so no Xcode toolchain
// and no .metallib are needed.  Source resolution order:
//   1. the path passed to k3_metal_init()
//   2. $K3_METAL_SRC
//   3. <dylib dir>/metal/moe_mxfp4.metal
//
// ---------------------------------------------------------------------------
// C ABI (all functions are thread-safe; a global mutex serialises GPU work)
//
//   int  k3_metal_available(void)
//        1 if a Metal device + library + pipelines came up, else 0.  Cheap after
//        the first call.  Never throws; inspect k3_metal_last_error() on 0.
//
//   int  k3_metal_init(const char *msl_path)
//        Explicit init with an MSL path (NULL = default resolution).  Returns 0
//        on success, negative on failure.  Idempotent.
//
//   int  k3_metal_moe_layer(const uint8_t *const *expert_blobs, int n_experts,
//                           const float *weights, const float *x, float *out)
//        expert_blobs[e] -> the expert's verbatim 17,547,264-byte shard span
//            (w1_p 5505024 | w1_s 344064 | w2_p 5505024 | w2_s 344064 |
//             w3_p 5505024 | w3_s 344064).  Must stay mapped and unmodified for
//            the duration of the call; if it is page-aligned it is ALSO retained
//            in the zero-copy buffer cache afterwards (see k3_metal_drop).
//        weights[n_experts]  fp32 routing weights, applied in the given order.
//        x[3584]             fp32 activations for one token.
//        out[3584]           fp32 result, overwritten.
//        Returns 0 on success, negative on error.  Synchronous.
//
//   int  k3_metal_moe_positions(
//            const uint8_t *const *expert_blobs, int n_edges,
//            const int *position_offsets, int n_positions,
//            const float *weights, const float *x, float *out)
//        Opt-in N5 path. `position_offsets` has n_positions+1 entries, starts at
//        zero, ends at n_edges, and partitions row-major expert_blobs/weights.
//        x/out are [n_positions, 3584]. The ordinary per-position kernels are
//        encoded in the same order into ONE command buffer and waited once.
//        Arithmetic and reduction order within every position are unchanged.
//
//   void k3_metal_drop(const uint8_t *blob)
//        Release the cached MTLBuffer for one blob pointer.  MUST be called
//        before the caller unmaps/frees that memory, otherwise a later call that
//        re-uses the pointer would read a dangling wrap.
//
//   void k3_metal_flush(void)   drop the whole buffer cache.
//
//   int  k3_metal_set_mode(int bindless)   1 = one dispatch for all experts via
//        a Tier-2 argument buffer (default when supported); 0 = one dispatch per
//        expert inside a single concurrent encoder.  Returns the mode in effect.
//   int  k3_metal_mode(void)
//
//   void k3_metal_shapes(int *hidden, int *inter, long long *span)
//   void k3_metal_stats(long long *five)
//        [calls, zero_copy_wraps, copies, cache_entries, bindless]
//   const char *k3_metal_last_error(void)
// ---------------------------------------------------------------------------

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

// ---- fixed model geometry (mirrors tools/fetch_v2.py LAYOUT) ---------------
static const uint32_t K3_H = 3584;                 // hidden / x length
static const uint32_t K3_I = 3072;                 // intermediate / h length
static const uint32_t P_BYTES = 5505024, S_BYTES = 344064;
static const uint32_t OFF_W1P = 0;
static const uint32_t OFF_W1S = P_BYTES;
static const uint32_t OFF_W2P = P_BYTES + S_BYTES;
static const uint32_t OFF_W2S = 2 * P_BYTES + S_BYTES;
static const uint32_t OFF_W3P = 2 * (P_BYTES + S_BYTES);
static const uint32_t OFF_W3S = 3 * P_BYTES + 2 * S_BYTES;
static const size_t   EXPERT_SPAN = 3 * (size_t)(P_BYTES + S_BYTES);   // 17,547,264

static const uint32_t ROWS_PER_TG = 16;            // 4 simdgroups x 4 rows
static const uint32_t TG_THREADS  = 128;

struct MoeDims {                                   // must match moe_mxfp4.metal
  uint32_t n_experts, H, I;
  uint32_t w1p, w1s, w2p, w2s, w3p, w3s;
  uint32_t expert, nt;
};

// ---------------------------------------------------------------------------
namespace {

std::mutex g_mu;
std::string g_err;
bool g_tried = false, g_ok = false;

id<MTLDevice> g_dev = nil;
id<MTLCommandQueue> g_q = nil;
id<MTLComputePipelineState> g_pGluB = nil, g_pW2B = nil;
id<MTLComputePipelineState> g_pGlu1 = nil, g_pW21 = nil;
id<MTLComputePipelineState> g_pRed = nil;

id<MTLBuffer> g_bX = nil, g_bH = nil, g_bY = nil, g_bOut = nil, g_bW = nil, g_bArgs = nil;
// N5 uses separate pools so the default one-position allocation and execution
// path above remain byte-for-byte untouched until explicitly requested.
id<MTLBuffer> g_bBatchX = nil, g_bBatchH = nil, g_bBatchY = nil;
id<MTLBuffer> g_bBatchOut = nil, g_bBatchW = nil, g_bBatchArgs = nil;
NSMutableDictionary<NSNumber *, id<MTLBuffer>> *g_cache = nil;   // blob ptr -> wrap
NSMutableArray<id<MTLBuffer>> *g_scratch = nil;                  // copy-path staging
int g_cap = 0;                                                   // pool capacity in experts
int g_batch_edge_cap = 0, g_batch_position_cap = 0;
int g_bindless = -1;                                             // -1 = auto
long long g_calls = 0, g_wraps = 0, g_copies = 0;

void seterr(const char *what, NSError *e) {
  g_err = what;
  if (e) { g_err += ": "; g_err += [[e localizedDescription] UTF8String]; }
}

std::string default_msl_path() {
  if (const char *env = getenv("K3_METAL_SRC")) return env;
  Dl_info info;
  if (dladdr((const void *)&default_msl_path, &info) && info.dli_fname) {
    std::string p(info.dli_fname);
    size_t slash = p.find_last_of('/');
    if (slash != std::string::npos)
      return p.substr(0, slash) + "/metal/moe_mxfp4.metal";
  }
  return "tools/metal/moe_mxfp4.metal";
}

id<MTLComputePipelineState> mkpipe(id<MTLLibrary> lib, NSString *name) {
  id<MTLFunction> fn = [lib newFunctionWithName:name];
  if (!fn) { seterr([[@"missing kernel " stringByAppendingString:name] UTF8String], nil); return nil; }
  NSError *err = nil;
  id<MTLComputePipelineState> p = [g_dev newComputePipelineStateWithFunction:fn error:&err];
  if (!p) seterr("pipeline", err);
  return p;
}

// Grow the float pools so they hold `n` experts' worth of h and y.
bool ensure_capacity(int n) {
  if (n <= g_cap) return true;
  int cap = n < 16 ? 16 : n;
  id<MTLBuffer> bH = [g_dev newBufferWithLength:(size_t)cap * K3_I * 4
                                        options:MTLResourceStorageModeShared];
  id<MTLBuffer> bY = [g_dev newBufferWithLength:(size_t)cap * K3_H * 4
                                        options:MTLResourceStorageModeShared];
  id<MTLBuffer> bW = [g_dev newBufferWithLength:(size_t)cap * 4
                                        options:MTLResourceStorageModeShared];
  id<MTLBuffer> bA = [g_dev newBufferWithLength:(size_t)cap * 8
                                        options:MTLResourceStorageModeShared];
  if (!bH || !bY || !bW || !bA) { seterr("pool alloc", nil); return false; }
  g_bH = bH; g_bY = bY; g_bW = bW; g_bArgs = bA; g_cap = cap;
  return true;
}

// Separate capacity for the opt-in multi-position ABI. Keeping it out of
// ensure_capacity() prevents a speculative T=8 call from permanently inflating
// the established T=1 scratch pools.
bool ensure_batch_capacity(int edges, int positions) {
  if (edges <= g_batch_edge_cap && positions <= g_batch_position_cap) return true;
  const int edge_cap = std::max(std::max(edges, g_batch_edge_cap), 1);
  const int position_cap =
      std::max(std::max(positions, g_batch_position_cap), 1);
  id<MTLBuffer> bX =
      [g_dev newBufferWithLength:(size_t)position_cap * K3_H * 4
                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> bH =
      [g_dev newBufferWithLength:(size_t)edge_cap * K3_I * 4
                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> bY =
      [g_dev newBufferWithLength:(size_t)edge_cap * K3_H * 4
                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> bOut =
      [g_dev newBufferWithLength:(size_t)position_cap * K3_H * 4
                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> bW =
      [g_dev newBufferWithLength:(size_t)edge_cap * 4
                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> bArgs =
      [g_dev newBufferWithLength:(size_t)edge_cap * 8
                         options:MTLResourceStorageModeShared];
  if (!bX || !bH || !bY || !bOut || !bW || !bArgs) {
    seterr("position-batch pool alloc", nil);
    return false;
  }
  g_bBatchX = bX;
  g_bBatchH = bH;
  g_bBatchY = bY;
  g_bBatchOut = bOut;
  g_bBatchW = bW;
  g_bBatchArgs = bArgs;
  g_batch_edge_cap = edge_cap;
  g_batch_position_cap = position_cap;
  return true;
}

bool init_locked(const char *msl_path) {
  if (g_tried) return g_ok;
  g_tried = true;
  @autoreleasepool {
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) { seterr("no Metal device", nil); return false; }
    g_q = [g_dev newCommandQueue];
    if (!g_q) { seterr("no command queue", nil); return false; }

    std::string path = (msl_path && *msl_path) ? msl_path : default_msl_path();
    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:@(path.c_str())
                                              encoding:NSUTF8StringEncoding error:&err];
    if (!src) { seterr(("read MSL " + path).c_str(), err); return false; }
    id<MTLLibrary> lib = [g_dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { seterr("compile MSL", err); return false; }

    g_pGluB = mkpipe(lib, @"moe_glu_batch");
    g_pW2B  = mkpipe(lib, @"moe_w2_batch");
    g_pGlu1 = mkpipe(lib, @"moe_glu_one");
    g_pW21  = mkpipe(lib, @"moe_w2_one");
    g_pRed  = mkpipe(lib, @"moe_reduce");
    if (!g_pGluB || !g_pW2B || !g_pGlu1 || !g_pW21 || !g_pRed) return false;
    if ([g_pGluB threadExecutionWidth] != 32) { seterr("simd width != 32", nil); return false; }

    g_bX   = [g_dev newBufferWithLength:K3_H * 4 options:MTLResourceStorageModeShared];
    g_bOut = [g_dev newBufferWithLength:K3_H * 4 options:MTLResourceStorageModeShared];
    if (!g_bX || !g_bOut) { seterr("pool alloc", nil); return false; }
    if (!ensure_capacity(16)) return false;

    g_cache = [NSMutableDictionary dictionary];
    g_scratch = [NSMutableArray array];

    if (g_bindless < 0) {
      bool tier2 = ([g_dev argumentBuffersSupport] == MTLArgumentBuffersTier2) &&
                   [g_bX respondsToSelector:@selector(gpuAddress)];
      const char *env = getenv("K3_METAL_BINDLESS");
      g_bindless = env ? (atoi(env) != 0 && tier2) : (tier2 ? 1 : 0);
    }
    g_ok = true;
  }
  return g_ok;
}

// Wrap one expert blob.  Page-aligned pointers (every mmap'd L*-E*.bin cache file
// qualifies: EXPERT_SPAN is exactly 1071 x 16 KB) get a zero-copy wrap that is
// cached for reuse; anything else is staged into a per-slot scratch buffer.
id<MTLBuffer> wrap_blob(const uint8_t *p, int slot) {
  NSNumber *key = @((unsigned long long)(uintptr_t)p);
  id<MTLBuffer> b = g_cache[key];
  if (b) return b;
  size_t pg = (size_t)getpagesize();
  if (((uintptr_t)p % pg) == 0 && (EXPERT_SPAN % pg) == 0) {
    b = [g_dev newBufferWithBytesNoCopy:(void *)p length:EXPERT_SPAN
                                options:MTLResourceStorageModeShared deallocator:nil];
    if (b) { g_cache[key] = b; g_wraps++; return b; }
  }
  while ((int)[g_scratch count] <= slot) {
    id<MTLBuffer> s = [g_dev newBufferWithLength:EXPERT_SPAN
                                         options:MTLResourceStorageModeShared];
    if (!s) { seterr("scratch alloc", nil); return nil; }
    [g_scratch addObject:s];
  }
  b = g_scratch[slot];              // never cached by pointer: contents are stale-able
  memcpy([b contents], p, EXPERT_SPAN);
  g_copies++;
  return b;
}

}  // namespace

// ---------------------------------------------------------------------------
extern "C" {

int k3_metal_init(const char *msl_path) {
  std::lock_guard<std::mutex> lk(g_mu);
  return init_locked(msl_path) ? 0 : -1;
}

int k3_metal_available(void) {
  std::lock_guard<std::mutex> lk(g_mu);
  return init_locked(nullptr) ? 1 : 0;
}

const char *k3_metal_last_error(void) { return g_err.c_str(); }

void k3_metal_shapes(int *hidden, int *inter, long long *span) {
  if (hidden) *hidden = (int)K3_H;
  if (inter) *inter = (int)K3_I;
  if (span) *span = (long long)EXPERT_SPAN;
}

void k3_metal_stats(long long *five) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!five) return;
  five[0] = g_calls; five[1] = g_wraps; five[2] = g_copies;
  five[3] = (long long)[g_cache count]; five[4] = g_bindless;
}

int k3_metal_mode(void) { std::lock_guard<std::mutex> lk(g_mu); return g_bindless; }

int k3_metal_set_mode(int bindless) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!init_locked(nullptr)) return -1;
  bool tier2 = ([g_dev argumentBuffersSupport] == MTLArgumentBuffersTier2) &&
               [g_bX respondsToSelector:@selector(gpuAddress)];
  g_bindless = (bindless && tier2) ? 1 : 0;
  return g_bindless;
}

void k3_metal_drop(const uint8_t *blob) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (g_cache) [g_cache removeObjectForKey:@((unsigned long long)(uintptr_t)blob)];
}

void k3_metal_flush(void) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (g_cache) [g_cache removeAllObjects];
  if (g_scratch) [g_scratch removeAllObjects];
}

int k3_metal_moe_layer(const uint8_t *const *expert_blobs, int n_experts,
                       const float *weights, const float *x, float *out) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!init_locked(nullptr)) return -1;
  if (!out || !x) return -2;
  if (n_experts < 0) return -2;
  if (n_experts == 0) { memset(out, 0, K3_H * 4); return 0; }
  if (!expert_blobs || !weights) return -2;
  if (!ensure_capacity(n_experts)) return -3;

  @autoreleasepool {
    std::vector<id<MTLBuffer>> bufs((size_t)n_experts);
    for (int e = 0; e < n_experts; ++e) {
      if (!expert_blobs[e]) return -2;
      bufs[e] = wrap_blob(expert_blobs[e], e);
      if (!bufs[e]) return -4;
    }
    memcpy([g_bX contents], x, K3_H * 4);
    memcpy([g_bW contents], weights, (size_t)n_experts * 4);

    MoeDims D = {(uint32_t)n_experts, K3_H, K3_I,
                 OFF_W1P, OFF_W1S, OFF_W2P, OFF_W2S, OFF_W3P, OFF_W3S, 0, 0};

    id<MTLCommandBuffer> cb = [g_q commandBuffer];

    if (g_bindless == 1) {
      uint64_t *addr = (uint64_t *)[g_bArgs contents];
      for (int e = 0; e < n_experts; ++e) addr[e] = [bufs[e] gpuAddress];
      // serial encoder: glu -> w2 -> reduce are dependent, the implicit barriers
      // between dispatches in a serial encoder are exactly the ordering we need.
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      for (int e = 0; e < n_experts; ++e)
        [enc useResource:bufs[e] usage:MTLResourceUsageRead];

      D.nt = (uint32_t)n_experts * K3_I;
      [enc setComputePipelineState:g_pGluB];
      [enc setBuffer:g_bArgs offset:0 atIndex:0];
      [enc setBuffer:g_bX offset:0 atIndex:1];
      [enc setBuffer:g_bH offset:0 atIndex:2];
      [enc setBytes:&D length:sizeof D atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake((D.nt + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];

      D.nt = (uint32_t)n_experts * K3_H;
      [enc setComputePipelineState:g_pW2B];
      [enc setBuffer:g_bArgs offset:0 atIndex:0];
      [enc setBuffer:g_bH offset:0 atIndex:1];
      [enc setBuffer:g_bY offset:0 atIndex:2];
      [enc setBytes:&D length:sizeof D atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake((D.nt + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];

      [enc setComputePipelineState:g_pRed];
      [enc setBuffer:g_bY offset:0 atIndex:0];
      [enc setBuffer:g_bW offset:0 atIndex:1];
      [enc setBuffer:g_bOut offset:0 atIndex:2];
      [enc setBytes:&D length:sizeof D atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake((K3_H + 255) / 256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [enc endEncoding];
    } else {
      // stage 1: every expert independent -> concurrent dispatches, no barriers.
      id<MTLComputeCommandEncoder> e1 =
          [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
      [e1 setComputePipelineState:g_pGlu1];
      [e1 setBuffer:g_bX offset:0 atIndex:1];
      [e1 setBuffer:g_bH offset:0 atIndex:2];
      for (int e = 0; e < n_experts; ++e) {
        D.expert = (uint32_t)e; D.nt = K3_I;
        [e1 setBuffer:bufs[e] offset:0 atIndex:0];
        [e1 setBytes:&D length:sizeof D atIndex:3];
        [e1 dispatchThreadgroups:MTLSizeMake((K3_I + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];
      }
      [e1 endEncoding];                       // encoder boundary == full barrier

      id<MTLComputeCommandEncoder> e2 =
          [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
      [e2 setComputePipelineState:g_pW21];
      [e2 setBuffer:g_bH offset:0 atIndex:1];
      [e2 setBuffer:g_bY offset:0 atIndex:2];
      for (int e = 0; e < n_experts; ++e) {
        D.expert = (uint32_t)e; D.nt = K3_H;
        [e2 setBuffer:bufs[e] offset:0 atIndex:0];
        [e2 setBytes:&D length:sizeof D atIndex:3];
        [e2 dispatchThreadgroups:MTLSizeMake((K3_H + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];
      }
      [e2 endEncoding];

      id<MTLComputeCommandEncoder> e3 = [cb computeCommandEncoder];
      [e3 setComputePipelineState:g_pRed];
      [e3 setBuffer:g_bY offset:0 atIndex:0];
      [e3 setBuffer:g_bW offset:0 atIndex:1];
      [e3 setBuffer:g_bOut offset:0 atIndex:2];
      [e3 setBytes:&D length:sizeof D atIndex:3];
      [e3 dispatchThreadgroups:MTLSizeMake((K3_H + 255) / 256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e3 endEncoding];
    }

    [cb commit];
    [cb waitUntilCompleted];
    if ([cb status] == MTLCommandBufferStatusError) {
      seterr("command buffer", [cb error]);
      return -5;
    }
    memcpy(out, [g_bOut contents], K3_H * 4);
    g_calls++;
  }
  return 0;
}

int k3_metal_moe_positions(const uint8_t *const *expert_blobs, int n_edges,
                           const int *position_offsets, int n_positions,
                           const float *weights, const float *x, float *out) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!init_locked(nullptr)) return -1;
  if (n_edges < 0 || n_positions < 0) return -2;
  if (n_positions == 0) return n_edges == 0 ? 0 : -2;
  if (!position_offsets || !x || !out) return -2;
  if (position_offsets[0] != 0) return -2;
  for (int p = 0; p < n_positions; ++p) {
    if (position_offsets[p] < 0 ||
        position_offsets[p] > position_offsets[p + 1] ||
        position_offsets[p + 1] > n_edges) {
      return -2;
    }
  }
  if (position_offsets[n_positions] != n_edges) return -2;
  if (n_edges > 0 && (!expert_blobs || !weights)) return -2;
  if (!ensure_batch_capacity(n_edges, n_positions)) return -3;

  @autoreleasepool {
    std::vector<id<MTLBuffer>> bufs((size_t)n_edges);
    for (int edge = 0; edge < n_edges; ++edge) {
      if (!expert_blobs[edge]) return -2;
      bufs[edge] = wrap_blob(expert_blobs[edge], edge);
      if (!bufs[edge]) return -4;
    }
    memcpy([g_bBatchX contents], x, (size_t)n_positions * K3_H * 4);
    if (n_edges > 0)
      memcpy([g_bBatchW contents], weights, (size_t)n_edges * 4);
    // Empty route rows are legal at the ABI boundary. Their reduce dispatch is
    // skipped, so pre-zero every output row before encoding the nonempty ones.
    memset([g_bBatchOut contents], 0, (size_t)n_positions * K3_H * 4);

    id<MTLCommandBuffer> cb = [g_q commandBuffer];
    if (!cb) {
      seterr("position-batch command buffer", nil);
      return -5;
    }

    if (g_bindless == 1) {
      uint64_t *addr = (uint64_t *)[g_bBatchArgs contents];
      for (int edge = 0; edge < n_edges; ++edge)
        addr[edge] = [bufs[edge] gpuAddress];

      // One serial encoder preserves every dependency in the proven N5 probe:
      // position 0 GLU -> W2 -> reduce, then position 1, and so on.
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      for (int edge = 0; edge < n_edges; ++edge)
        [enc useResource:bufs[edge] usage:MTLResourceUsageRead];

      for (int p = 0; p < n_positions; ++p) {
        const int edge0 = position_offsets[p];
        const int count = position_offsets[p + 1] - edge0;
        if (count == 0) continue;
        const size_t xoff = (size_t)p * K3_H * 4;
        const size_t hoff = (size_t)edge0 * K3_I * 4;
        const size_t yoff = (size_t)edge0 * K3_H * 4;
        const size_t woff = (size_t)edge0 * 4;
        const size_t aoff = (size_t)edge0 * 8;
        const size_t ooff = (size_t)p * K3_H * 4;
        MoeDims D = {(uint32_t)count, K3_H, K3_I,
                     OFF_W1P, OFF_W1S, OFF_W2P, OFF_W2S,
                     OFF_W3P, OFF_W3S, 0, (uint32_t)count * K3_I};

        [enc setComputePipelineState:g_pGluB];
        [enc setBuffer:g_bBatchArgs offset:aoff atIndex:0];
        [enc setBuffer:g_bBatchX offset:xoff atIndex:1];
        [enc setBuffer:g_bBatchH offset:hoff atIndex:2];
        [enc setBytes:&D length:sizeof D atIndex:3];
        [enc dispatchThreadgroups:
                 MTLSizeMake((D.nt + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];

        D.nt = (uint32_t)count * K3_H;
        [enc setComputePipelineState:g_pW2B];
        [enc setBuffer:g_bBatchArgs offset:aoff atIndex:0];
        [enc setBuffer:g_bBatchH offset:hoff atIndex:1];
        [enc setBuffer:g_bBatchY offset:yoff atIndex:2];
        [enc setBytes:&D length:sizeof D atIndex:3];
        [enc dispatchThreadgroups:
                 MTLSizeMake((D.nt + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];

        [enc setComputePipelineState:g_pRed];
        [enc setBuffer:g_bBatchY offset:yoff atIndex:0];
        [enc setBuffer:g_bBatchW offset:woff atIndex:1];
        [enc setBuffer:g_bBatchOut offset:ooff atIndex:2];
        [enc setBytes:&D length:sizeof D atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((K3_H + 255) / 256, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      [enc endEncoding];
    } else {
      // The direct-bind fallback keeps the established concurrent-per-stage
      // schedule. Encoder boundaries preserve GLU/W2/reduce dependencies while
      // all positions still share one command buffer and one completion wait.
      for (int p = 0; p < n_positions; ++p) {
        const int edge0 = position_offsets[p];
        const int count = position_offsets[p + 1] - edge0;
        if (count == 0) continue;
        const size_t xoff = (size_t)p * K3_H * 4;
        const size_t hoff = (size_t)edge0 * K3_I * 4;
        const size_t yoff = (size_t)edge0 * K3_H * 4;
        const size_t woff = (size_t)edge0 * 4;
        const size_t ooff = (size_t)p * K3_H * 4;
        MoeDims D = {(uint32_t)count, K3_H, K3_I,
                     OFF_W1P, OFF_W1S, OFF_W2P, OFF_W2S,
                     OFF_W3P, OFF_W3S, 0, 0};

        id<MTLComputeCommandEncoder> e1 =
            [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        [e1 setComputePipelineState:g_pGlu1];
        [e1 setBuffer:g_bBatchX offset:xoff atIndex:1];
        [e1 setBuffer:g_bBatchH offset:hoff atIndex:2];
        for (int edge = 0; edge < count; ++edge) {
          D.expert = (uint32_t)edge;
          D.nt = K3_I;
          [e1 setBuffer:bufs[(size_t)edge0 + edge] offset:0 atIndex:0];
          [e1 setBytes:&D length:sizeof D atIndex:3];
          [e1 dispatchThreadgroups:
                  MTLSizeMake((K3_I + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];
        }
        [e1 endEncoding];

        id<MTLComputeCommandEncoder> e2 =
            [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        [e2 setComputePipelineState:g_pW21];
        [e2 setBuffer:g_bBatchH offset:hoff atIndex:1];
        [e2 setBuffer:g_bBatchY offset:yoff atIndex:2];
        for (int edge = 0; edge < count; ++edge) {
          D.expert = (uint32_t)edge;
          D.nt = K3_H;
          [e2 setBuffer:bufs[(size_t)edge0 + edge] offset:0 atIndex:0];
          [e2 setBytes:&D length:sizeof D atIndex:3];
          [e2 dispatchThreadgroups:
                  MTLSizeMake((K3_H + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];
        }
        [e2 endEncoding];

        id<MTLComputeCommandEncoder> e3 = [cb computeCommandEncoder];
        [e3 setComputePipelineState:g_pRed];
        [e3 setBuffer:g_bBatchY offset:yoff atIndex:0];
        [e3 setBuffer:g_bBatchW offset:woff atIndex:1];
        [e3 setBuffer:g_bBatchOut offset:ooff atIndex:2];
        [e3 setBytes:&D length:sizeof D atIndex:3];
        [e3 dispatchThreadgroups:MTLSizeMake((K3_H + 255) / 256, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e3 endEncoding];
      }
    }

    [cb commit];
    [cb waitUntilCompleted];
    if ([cb status] == MTLCommandBufferStatusError) {
      seterr("position-batch command buffer", [cb error]);
      return -5;
    }
    memcpy(out, [g_bBatchOut contents], (size_t)n_positions * K3_H * 4);
    g_calls += n_positions;
  }
  return 0;
}

}  // extern "C"
