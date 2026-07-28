# M1 Max (64GB) capability report for Kimi K3 huge-MoE inference

All numbers below were measured on this machine today (macOS 26.5.2, clang 21.0.0, Metal 4 reported) unless marked [knowledge]. Benchmark sources live in `/private/tmp/claude-501/-Users-chris-Claude-kimi-k3/59064e0b-8864-414a-bf6f-055f098c486b/scratchpad/bench/` (ssdbench.c, membench.c, mmapbench.c, metaltest.m, gpuflops.m, amxbench.c). The 12GB SSD test file was deleted after the runs.

## 1. Hardware inventory

- MacBookPro18,2, Apple M1 Max, **32-core GPU** (system_profiler: "Total Number of Cores: 32", Metal Support: Metal 4).
- CPU: 10 cores = **8 P (Firestorm) + 2 E (Icestorm)** (`hw.perflevel0.physicalcpu=8`, `hw.perflevel1.physicalcpu=2`). L1d 64KB/128B lines, P-cluster L2 12MB (`hw.perflevel0.l2cachesize=12582912`), E-cluster L2 4MB. Page size **16KB** (`hw.pagesize=16384`).
- RAM 64GiB (`hw.memsize=68719476736`), usable 66.7GB (`hw.memsize_usable=66746712064`). [knowledge] LPDDR5-6400 on a 512-bit bus = **400GB/s theoretical**; GPU ~1296MHz, theoretical 10.4 TFLOPS fp32 (fp16 same rate on M1 — no double-rate fp16, no matrix units).
- ISA flags that matter for quant kernels: `FEAT_DotProd=1` (sdot/udot), `FEAT_FHM=1` (FMLAL fp16→fp32), `FEAT_FP16=1`, but **`FEAT_BF16=0` and `FEAT_I8MM=0`** — no CPU bf16, no i8mm 8×8 int8 matmul instruction. Int8 paths must use sdot; bf16 weights must be converted (bf16→fp32 is a shift, cheap).
- AMX: [knowledge] M1 Max has one AMX unit per cluster (2 P-clusters + 1 E). Reached via Accelerate only (see §7): measured 1.62 TFLOPS fp32 GEMM.
- SSD: **APPLE SSD AP8192R (8TB)**, APFS, 1.2TiB free on /System/Volumes/Data. The 8TB config is the widest-striped (fastest) Apple BGA SSD.
- Wiring/limits: `iogpu.wired_limit_mb=0` (default in effect), Metal `recommendedMaxWorkingSetSize=51.8GiB`, `maxBufferLength=38.9GiB` (single-buffer cap — a bigger model mapping must be split across MTLBuffers). `vm.user_wire_limit=56,349,970,923` (52.5GiB mlock ceiling). POSIX aio is crippled: `kern.aiomax=90, kern.aioprocmax=16, kern.aiothreads=4`.

## 2. SSD benchmark (F_NOCACHE = uncached, 12GB pseudorandom file)

Write: **1.91 GB/s** (12GB, F_NOCACHE, 16MB blocks, fsync included).

Sequential uncached read (single thread):
| block | GB/s |
|---|---|
| 1MB | 2.93 |
| 4MB | 4.93 |
| 16MB | 6.14 |

Random uncached reads (expert-sized chunks, N threads = effective queue depth):
| chunk | QD1 | QD2 | QD4 | QD8 | QD16 |
|---|---|---|---|---|---|
| 1MB | 2.59 | – | 5.36 | 6.60 | 6.65 |
| 2MB | 3.41 | 5.59 | 6.26 | 5.41 | 5.72 |
| 4MB | 5.00 | 5.50 | 5.25 | 6.56 | 6.64 |
| 8MB | 4.95 | 5.42 | 6.55 | 6.66 | 6.34 |

QD1 latency: **0.38ms per 1MB, 0.57ms per 2MB, 0.78ms per 4MB, 1.58ms per 8MB**. Device saturates at **~6.6 GB/s** with 4–8 concurrent threads at 1–8MB granularity; random ≈ sequential once QD≥4 (chunked random expert fetch costs nothing vs streaming — layout freedom). Headline: **the expert-streaming budget on this box is 6.6 GB/s, reachable only with a multi-threaded F_NOCACHE pread pool.**

## 3. CPU memory bandwidth (clang -O3, NEON, QoS user-interactive)

| threads | READ (NEON sum) | COPY (memcpy, R+W counted) | TRIAD |
|---|---|---|---|
| 1 | 39.2 | 66.0 | 67.2 |
| 2 | 70.4 | – | – |
| 4 | 77.8–81.1 | 136.3 | 103.9 |
| 8 | 92.3–105.3 | 152.0 | 131.3 |
| 10 | 99.7–107.2 | – | 131.1 |

Practical CPU-side ceiling ≈ **105–110 GB/s pure read, ~150 GB/s mixed traffic** — the CPU cluster cannot get near the 400GB/s fabric. (Apple's own memcpy into an L1-resident buffer was slower: 18–50 GB/s.) GPU is the bandwidth engine: measured **283–290 GB/s** GPU read on a 2GB shared MTLBuffer (simple strided uint4 sum kernel; better kernels reach ~330 [knowledge]). Consequence: any GEMV over resident weights should run on GPU (≈2.7× CPU bandwidth); CPU NEON is fully adequate for the SSD-streamed portion (6.6 GB/s ≪ 105 GB/s).

## 4. Metal on M1 Max (all probed programmatically)

- Families: **Apple7 = YES; Apple8 (M2) / Apple9 (M3) = NO; Metal3 = YES**; OS reports Metal 4 API. `supportsFunctionPointers=1`, `supportsRaytracing=1` (software), argumentBuffers tier 1, `maxThreadgroupMemoryLength=32KB`.
- **bfloat: MSL 3.1 `bfloat` kernels compile AND create pipelines on M1** (macOS 26). No native bf16 ALU (Apple7), so it's compiler-emulated via conversions — fine for storage-type bf16, don't expect rate gains.
- **simdgroup_half8x8 / simdgroup_float8x8 work** (8×8 is the only size). Measured GPU FLOPS (with `#pragma clang loop unroll_count(16)` — without it, dispatch loop overhead cost 55%!): **fp32 FMA 9.07 TFLOPS, fp16 FMA 8.13 TFLOPS, simdgroup_half8x8 MMA 7.42 TFLOPS**. So fp16 is NOT faster than fp32 on M1; simdgroup_matrix wins on locality, not rate.
- **Zero-copy: `newBufferWithBytesNoCopy` works on file-backed mmap** — both `PROT_READ, MAP_SHARED` and `PROT_READ, MAP_PRIVATE` mappings produced working MTLBuffers; GPU checksum matched CPU checksum. Requires page-aligned (16KB) address + length.
- **Residency is the trap**: cold GPU pass over a 4GB file-backed no-copy buffer ran at **2.0 GB/s** (driver faults/wires pages during use), warm passes **242–275 GB/s**. GPU-side demand paging from SSD is 3.3× slower than an explicit pread pool (6.6 GB/s). `setPurgeableState` works on such buffers (returned nonVolatile). Anonymous-mmap no-copy buffers: full speed (235 GB/s warm).
- Default GPU working set 51.8GiB; `iogpu.wired_limit_mb` (currently 0 = default) can be raised with sudo — ds4's README already documents this (ds4/README.md:610-612).

## 5. macOS I/O specifics (measured where possible)

- **No io_uring.** kqueue's EVFILT_READ is useless for regular files (always ready, read still blocks); POSIX aio capped at 90 in-flight ops / 4 kernel threads system-wide. **The fastest async path is a plain pthread pool doing F_NOCACHE preads — measured 6.6 GB/s at 4–8 threads**, which saturates the device; dispatch_io adds nothing over that but loses control.
- **mmap page-fault streaming is catastrophic cold**: plain touch = **0.66 GB/s**; `madvise(MADV_SEQUENTIAL)` = **0.69 GB/s** (no help); 8 threads faulting in parallel = **2.02 GB/s**. **`posix_madvise(POSIX_MADV_WILLNEED)`/`MADV_WILLNEED` is effectively a no-op on XNU for file pages.**
- **F_RDADVISE is the real macOS readahead**: issuing `fcntl(F_RDADVISE)` in 32MB chunks ahead of touching lifted mmap fault streaming to **4.96 GB/s** (7.5×). Still below the 6.6 GB/s pread pool, and it double-buffers through the unified buffer cache — which matters when ~50GB is wired and page cache has no room.
- F_NOCACHE semantics: bypasses UBC for 4KB-aligned transfers; unaligned head/tail silently go through cache. No O_DIRECT alignment constraints on the buffer.
- Memory pressure: this machine had 28.5GB swap in use at session start — macOS swaps aggressively at ~50GB working sets. Quantized weights are entropy-packed and **incompressible, so the VM compressor wastes CPU and then swaps to the same SSD you're streaming from**, stealing read bandwidth. Strategy: keep the resident expert pool wired (GPU no-copy buffers get wired on use; CPU-side cache can mlock up to the 52.5GiB `vm.user_wire_limit`), use F_NOCACHE so streaming never competes with the page cache, and size caches from `kern.memorystatus_level` / DISPATCH_SOURCE_TYPE_MEMORYPRESSURE callbacks. macOS jetsam rarely kills foreground processes [knowledge], but dirtying + compressor churn will halve effective SSD throughput.
- Budget guidance: ~51.8GiB GPU-wired max (default), leave ≥8GB for OS; practical resident model+KV budget ≈ **45–50GB**.

## 6. What MLX / llama.cpp do that a from-scratch engine should copy [knowledge, verified against local probes]

- llama.cpp: splits the model mapping into multiple no-copy MTLBuffers below `maxBufferLength` (38.9GiB here — mandatory for >39GB residents); **in-kernel dequant** — quantized blocks are read directly by Metal matmul kernels (simdgroup 8×8 tiles), never materializing fp16 weights; `MTLDispatchTypeConcurrent` encoders + untracked hazards with manual barriers; `--mlock`; respects `recommendedMaxWorkingSetSize`; documents raising `iogpu.wired_limit_mb`. CPU side uses sdot-based Q4/Q8 dot products (right call here given I8MM=0).
- MLX: lazy graph evaluation with kernel fusion of elementwise chains; steel GEMM kernels; **MTLResidencySet** (macOS 15+) to make the whole weight set resident once instead of per-command-buffer `useResource` churn — ds4 already adopted this (ds4_metal.m:652-690); wired-limit management API; unified-memory arrays shared CPU/GPU with no copies; async streams overlapping CPU and GPU work.
- Both: per-token encoder re-encoding is kept trivially cheap; there is no CUDA-graph equivalent needed — one command buffer, many encoders.

## 7. Accelerate/AMX vs hand NEON

- **GEMV (decode): Accelerate loses.** `cblas_sgemv` 8192×8192 fp32 = 8.36ms = **29.9 GB/s** vs hand NEON 45.4 GB/s (1 thread), 103.1 (4T), **105.7 GB/s (8T)**. GEMV is bandwidth-bound; 8 NEON threads hit the CPU fabric ceiling and beat AMX 3.5×. And Accelerate can't consume 2–6-bit quantized data — dequantizing first doubles traffic. **Verdict: for quantized dequant+GEMV, hand NEON (sdot + FHM fmlal) always wins on this chip.**
- **GEMM (prefill): AMX wins on CPU** — `cblas_sgemm` 2048³ fp32 = **1616 GFLOPS**, ~2× what 8 NEON cores can do in fp32 (~830 GFLOPS theoretical). Crossover: once the activation batch M ≳ 32–64, dequantize weight tiles once into an fp16/fp32 panel and feed AMX (or better, the GPU: 7.4–9 TFLOPS measured). BNNS exposes fp16 AMX GEMM (~2× fp32) [knowledge]. Below that M, stay in NEON.

## 8. Audit of the two engines (Apple-Silicon tricks present/missing)

**colibri** (`research/colibri/c/`):
- Has: F_NOCACHE twin fd (`compat.h:44-48`, used at `st.h:95`); `posix_fadvise(WILLNEED)`→`F_RDADVISE` shim (`compat.h:15,30-33`) — correct macOS readahead; layer-expert prefetch then demand pread (`st.h:382-397`); io_uring on Linux only (`uring.h`); Metal `newBufferWithBytesNoCopy` (`backend_metal.mm:411,481`) and opt-in `MTLResourceHazardTrackingModeUntracked` (`backend_metal.mm:418`).
- Missing: demand reads are **single-threaded sequential** (`st_pread_full`, `st.h:191`) — its F_RDADVISE+cached-pread pattern tops out ≈5 GB/s and routes bytes through the UBC (double copy + cache pressure at 50GB wired). A 4–8-thread F_NOCACHE pread pool on the existing `dfds` at 1–8MB granularity reaches 6.6 GB/s and skips the page cache. Also no MTLResidencySet, and many small per-call `newBufferWithBytes` uploads (`backend_metal.mm:575,869-873`) instead of persistent ring buffers.

**ds4** (`research/ds4/`):
- Has: mmap loader (MAP_SHARED for Metal / MAP_PRIVATE for CPU, `ds4.c:2447-2454`); `newBufferWithBytesNoCopy` over the model mapping (`ds4_metal.m:1791`); **MTLResidencySet** management (`ds4_metal.m:652-690`); reads `iogpu.wired_limit_mb` as an explicit GPU budget (`ds4.c:37385-37518`); Metal-4 MPP `matmul2d` prefill path with legacy fallback (`ds4_metal.m:2266-2303`); MoE mul_mm_id kernels; expert-cache streaming with hotlists (`ds4_streaming_hotlist*.inc`).
- Missing (the big one): its only readahead is `posix_madvise(POSIX_MADV_WILLNEED)` (`ds4.c:2327-2333, 3066`) — **a no-op on macOS**. Cold expert misses and cold starts run at mmap-fault speed: **0.66–2.0 GB/s measured, vs 6.6 GB/s available**. It needs an F_RDADVISE call (1-line macOS shim, colibri compat.h:30-33 shows how) and/or a threaded F_NOCACHE pread path for expert-cache fills. Also: GPU-faulting cold pages through no-copy buffers costs 2.0 GB/s — pages must be CPU-prefaulted/advised before dispatch. Its MPP matmul2d path likely falls back on M1 (Apple7) — the fallback exists (`ds4_metal.m:2277`), verify which branch actually runs.

## 9. Kimi K3 sizing math for this machine (estimates flagged as such)

- **Disk is the first wall**: 2.8T params at 4.25bpw ≈ 1.49TB > 1.2TiB free → **does not fit**. At ~3.0bpw ≈ 1.05TB it fits with ~15% headroom. Target ≤3.2bpw average (mix: 4-bit hot/shared + 2–3-bit cold experts), or prune provably-cold experts at conversion time.
- **Streaming budget**: 6.6 GB/s. If K3 scales K2's shape (8 routed experts/layer, ~60 layers, ~16MB/expert at 3bpw — estimate), zero-cache decode streams ~8GB/token → 1.2s/token floor. Every percentage of expert-cache hit rate in the ~45GB resident pool converts directly: 85% hit ≈ 1.2GB/token ≈ 5 tok/s I/O-bound ceiling. The QD1 latencies (0.57ms per 2MB) make just-in-time per-layer fetch viable inside layer compute time if fetches are issued at routing time with 4–8 parallel threads.
- Compute is never the decode bottleneck: resident-weight GEMV on GPU at 290 GB/s, streamed experts on CPU NEON at 105 GB/s (≫ 6.6 GB/s arrival rate). Prefill belongs on GPU (7.4–9 TFLOPS measured).

## KEY FACTS
- M1 Max here is the full chip: 32-core GPU, 8P+2E CPU, 64GiB RAM, 8TB APPLE SSD AP8192R with 1.2TiB free, macOS 26.5.2, 16KB pages
- SSD uncached (F_NOCACHE) read saturates at 6.6 GB/s with 4-8 pread threads at 1-8MB chunks; single-thread sequential is 2.93/4.93/6.14 GB/s at 1/4/16MB blocks; write 1.91 GB/s
- Random-read QD1 latency: 0.38ms/1MB, 0.57ms/2MB, 0.78ms/4MB, 1.58ms/8MB - random reads at QD>=4 equal sequential throughput, so expert layout on disk is unconstrained
- Cold mmap page-fault streaming: 0.66 GB/s plain, 0.69 with MADV_SEQUENTIAL, 2.02 with 8 fault threads; posix_madvise(WILLNEED) is a no-op on macOS; F_RDADVISE readahead lifts it to 4.96 GB/s; explicit F_NOCACHE pread pool (6.6 GB/s) beats everything
- CPU memory bandwidth measured: read 39 GB/s 1T -> 107 GB/s 10T; memcpy 66 -> 152 GB/s (R+W); triad 67 -> 131 GB/s; CPU cannot exceed ~110 GB/s pure read of the 400 GB/s fabric
- GPU read bandwidth measured 283-290 GB/s on shared MTLBuffer; GPU FLOPS measured: 9.07 TFLOPS fp32, 8.13 fp16, 7.42 simdgroup_half8x8 (fp16 is NOT double-rate on M1)
- newBufferWithBytesNoCopy on file-backed mmap (MAP_SHARED and MAP_PRIVATE, PROT_READ) works and validates; warm GPU reads 242-275 GB/s but cold GPU page-faulting runs at 2.0 GB/s - pages must be prefaulted CPU-side before dispatch
- Metal limits on this box: maxBufferLength 38.9GiB (must split larger residents across buffers), recommendedMaxWorkingSetSize 51.8GiB, iogpu.wired_limit_mb=0 (default, raiseable via sudo sysctl), vm.user_wire_limit 52.5GiB for mlock
- M1 CPU lacks FEAT_BF16 and FEAT_I8MM (has DotProd/sdot, FHM, FP16); GPU is Apple7 family only (not Apple8/9) but MSL 3.1 bfloat kernels compile and run (emulated); simdgroup_matrix is 8x8 only; function pointers supported
- AMX via Accelerate: cblas_sgemm 2048^3 = 1616 GFLOPS fp32 (wins CPU prefill), but cblas_sgemv = 29.9 GB/s loses 3.5x to 8-thread hand NEON GEMV at 105.7 GB/s - quantized decode GEMV must be hand NEON
- POSIX aio is unusable (kern.aiomax=90, 4 kernel threads); kqueue does not do async file reads; the fastest macOS async I/O is simply a pthread pool of F_NOCACHE preads
- ds4's only readahead is posix_madvise(POSIX_MADV_WILLNEED) (ds4.c:2327,3066) which does nothing on macOS -> its cold expert streaming runs at 0.66-2 GB/s vs 6.6 available; colibri already has the correct F_RDADVISE shim (compat.h:30-33) and F_NOCACHE twin fds (st.h:95) but demand-reads on a single thread (st_pread_full, st.h:191) capping it near 5 GB/s
- ds4 already uses MTLResidencySet (ds4_metal.m:652-690), no-copy model buffers (ds4_metal.m:1791), reads iogpu.wired_limit_mb (ds4.c:37385), and has a Metal-4 MPP matmul2d prefill path with fallback (ds4_metal.m:2266-2303)
- Kimi K3 2.8T at 4.25bpw = ~1.49TB exceeds the 1.2TiB free disk; needs <=3.2bpw average quant or expert pruning to fit; at 6.6 GB/s streaming, decode speed is entirely governed by resident expert-cache hit rate within a ~45-50GB wired budget
- VM compressor + swap actively fight streaming: the machine showed 28.5GB swap in use; quantized weights are incompressible so compression wastes CPU then swaps to the same SSD being streamed from - wire/mlock the resident set and use F_NOCACHE to keep the page cache out

## IMPROVEMENT IDEAS
- Replace ds4's posix_madvise(WILLNEED) with a macOS F_RDADVISE shim (copy colibri compat.h:30-33) and, better, route expert-cache fills through a 4-8 thread F_NOCACHE pread pool at 1-8MB granularity: measured 6.6 GB/s vs the 0.66-2.0 GB/s it gets today - a 3-10x cold-streaming win, the single biggest speedup available
- Upgrade colibri's demand path from single-threaded st_pread_full to a small parallel pread worker pool on its existing F_NOCACHE twin fds: 5.0 -> 6.6 GB/s and stops double-buffering through the unified buffer cache, which matters when ~50GB is wired
- Never let the GPU fault cold file-backed no-copy buffers (2.0 GB/s); prefault via CPU touch or F_RDADVISE immediately after router selection, then dispatch - warm no-copy buffers deliver 275 GB/s
- Split work by bandwidth domain: GPU (290 GB/s) runs attention, dense layers, shared+cached hot experts; CPU NEON (105 GB/s, sdot+fmlal since I8MM/BF16 are absent) runs the SSD-streamed cold experts whose arrival rate is only 6.6 GB/s - this hides nearly all streaming behind resident compute
- Issue expert fetches at routing time with QD4-8: per-expert 2MB reads cost 0.57ms at QD1, so all 8 routed experts of a layer can be in flight and complete within the previous layer's compute window; consider router-lookahead (compute layer L+1 gating from layer L activations) to double the overlap window
- Budget memory explicitly: raise iogpu.wired_limit_mb (sudo sysctl) toward ~52-56GB if GPU-resident, keep total resident+KV near 45-50GB, mlock CPU-side expert cache (limit 52.5GiB), use F_NOCACHE everywhere so the page cache and VM compressor never compete, and shrink the cache on DISPATCH_SOURCE_TYPE_MEMORYPRESSURE events
- Quantize for the disk wall, not just quality: 2.8T params must land at <=3.2bpw average to fit 1.2TiB - use 4-5bpw for shared/dense/hot experts and 2-3bpw for cold experts, or prune experts with near-zero routing mass; also consider storing a small fp16 'hot shard' resident and streaming only cold experts
- Split any resident weight pool into MTLBuffers under the 38.9GiB maxBufferLength, register them once in a MTLResidencySet (ds4 pattern, ds4_metal.m:652), and use untracked hazards + concurrent dispatch (colibri COLI_METAL_UNTRACKED pattern) to kill per-token encoder overhead
- Do prefill on GPU with simdgroup_half8x8 in-kernel dequant kernels (7.4-9 TFLOPS measured; llama.cpp-style block-quant matmul) - never dequantize to a buffer first; on CPU, only use Accelerate/AMX (1.6 TFLOPS sgemm) for batch>=32-64 tiles, never for GEMV
- Pin I/O and dequant-staging threads to the 2 E-cores (QoS background/utility) and keep all 8 P-cores on NEON compute; the SSD needs only 4-8 lightweight threads to saturate
- Unroll GPU inner loops explicitly (#pragma clang loop unroll_count(16) took the same kernel from 4.0 to 9.1 TFLOPS in measurement) - Metal's compiler under-unrolls dependency-chained FMA loops
- Exploit the random=sequential property of this SSD (at QD>=4): store experts in per-expert contiguous chunks in routing-frequency order with zero layout constraints, and fetch exactly the selected experts - no need for large sequential super-blocks