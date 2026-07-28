# Deltafin — Speed Optimization Brainstorm Log

Every idea raised for squeezing milliseconds out of Kimi K3 inference on this machine,
with status and measured results where tested. All measurements on real K3 weights,
this M1 Max (64 GB, 8 TB NVMe), 2026-07-27. Baseline governing equation:
`decode time ≈ missed expert bytes ÷ effective fetch bandwidth + resident I/O + compute`.

Timeline of the day: first run ~20 min/token → run 3 (mps+int8+v2) ~127 s → run 4
(quartet) **~76 s warm**. Prefill 2,429 s → 154 s (15.8×).

## ✅ Shipped & validated (in the driver today)

| # | Idea | Measured result |
|---|------|-----------------|
| S1 | Fused MXFP4 dequant+GEMV NEON kernel (`vqtbl1q` nibble LUT + e8m0 scale as integer exponent-add; never materialize fp32 weights) | bit-exact; 5.2 GB/s/core, 15.8 GB/s@4T packed; 41–123× the numpy+torch path; full expert triple 1.04 ms@4T |
| S2 | HTTP fetch v2: ONE range request per expert (6 tensors verified contiguous, fixed 17,547,264 B span) over 4 keep-alive CDN connections, hourly signed-URL cache | 6.4× wall (9.1→57.7 MB/s under live contention); per-expert latency 6.5–8.8 s → 1.1 s |
| S3 | torch-MPS resident compute (fp32) | logits identical to 4 decimals; resident compute 903→117 s (7.8×) |
| S4 | int8 resident spine (per-row symmetric, fp16 scales) | 114.4→60 GB/pass; argmax + top-5 order preserved, top-logit shift 0.07% |
| S5 | Raw-span expert cache (.bin = verbatim shard bytes, mmap on hit; replaced npz zip+CRC parsing ~1,472 loads/token) | in quartet; gate-clean |
| S6 | Double-buffered layer loader (worker thread reads layer N+1 blobs during layer N compute) | in quartet; gate-clean |
| S7 | Resident lm_head on GPU (was 2.35 GB re-read+dequant per token) | in quartet; gate-clean |
| S8 | Previous-token whole-depth expert prefetch (fires the prior token's per-layer sets at step start) | in quartet; honest recall **30.8%** on a deduplicated holdout (the 39.1–39.7% we first quoted was inflated by repeated prompts in the trace) |
| — | Quartet net (S5–S8) | decode 127→76 s warm; prefill 219→154 s |

## ✅ Validated prototype, integration pending

| # | Idea | Measured result |
|---|------|-----------------|
| P1 | Metal fused MXFP4 GEMV (in-shader dequant, simdgroup reduce, 16-expert batches per command buffer) | 150 GB/s sustained = 9.5× CPU kernel; per-expert dispatch unusable (29 GB/s) — must batch; zero-copy OK (expert stride is 16 KB-aligned) |
| P2 | Buffered reads + F_RDADVISE next-file readahead (vs F_NOCACHE pool) | +18% (7.0 GB/s ≈ SSD ceiling); F_SPECULATIVE_READ = statistical tie; re-test under wired-memory pressure before adopting in the C engine |

## 🔄 In test now

| # | Idea | Rationale |
|---|------|-----------|
| T2 | n-gram speculative 2-token batching | **CERTIFIED & DEFAULT-ON**: accept path sequence-exact (16/16 reference match; spec+2 = 43 s/token effective vs 60 s floor); rollback after a wrong draft restores state and logits **bit-exactly** (maxdiff 0.00e+00); causal isolation exact. Lossless speedup for all modes |
| T3 | int8→GPU transfer-order fix (ship int8 bytes, dequant on device — 4× less transfer on the ~40 s/token apply slice) | numerically identical; rides run-6 validation |
| T4 | Per-phase profiler (`K3_PROFILE=1`): KDA vs MLA vs moe-kernel vs fetch vs apply with MPS sync boundaries | directs the next compute fix (torch.compile vs batched dylib vs sync elimination) |
| T5 | RAM-adaptive layer pinning (auto budget = total − max(10 GB, 18%); ~40% of spend pins layers on GPU permanently; scales 64→128 GB Macs automatically) | **gate PASSED bit-exact**; speed effect measures next quiet run |

## 🧭 Queued (ranked by expected value)

| # | Idea | Expected win | Notes |
|---|------|--------------|-------|
| **Q0** | **Template-layer buffer reuse**: two persistent materialized layers (KDA-shape + MLA-shape — all 69/24 are identical), per-layer `copy_()` into the same MPS buffers, mutate `layer_idx` attrs | **measured 1,317→288 ms/layer (4.6×) in isolation; ~1 s/layer × 92 = the single largest remaining cost** | discovered via op-microbench: full KDA attention is only 11.2 ms — the "compute" was MPS alloc churn from materialize→free. Profiler caveat learned: the KDA/MLA buckets time whole decoder layers incl. MoE |


| # | Idea | Expected win | Notes |
|---|------|--------------|-------|
| Q1 | torch.compile the layer forward on MPS | cut per-op dispatch on the many small KDA-shim ops | one shared shape across 69 KDA layers = one compile; MPS backend maturity is the risk |
| Q2 | Metal expert kernel into the driver (P1) | expert compute 15.8→150 GB/s for cached experts | pairs with a GPU-resident hot-expert pool (dequantized fp16 MTLBuffers, ~10–20 GB) |
| Q3 | Batch API + persistent P-core-pinned threads in the GEMV dylib (one call per layer, not 48) | thread-spawn overhead × 92 layers × every token | small, easy |
| Q4 | Cache-aware expert substitution (CACHE_ROUTE-lite: only for misses in the bottom-k routing weights) | colibri measured +39% for the aggressive version | quality-gated, graduated knob |
| Q5 | Top-k truncation (16→12: ×1.33 on fetch+expert compute; 16→8: ×2) | biggest honest quality-for-speed dial | gate harness ready |
| Q6 | Router surrogate: predict ALL layers' experts at token start (train tiny per-layer heads on accumulating trace) | turns depth-serial fetch into one QD-saturating burst | needs more trace; token-id→experts prior table is the cheap v0 |
| Q7 | Pin hottest ~20 layers' int8 weights in RAM (~13 GB) + all layers' small tensors | kills a slice of per-layer apply/transfer churn | interacts with memory budget |
| Q8 | Idle-time cache warmer (convert old npz→bin, prefetch top-frequency uncached experts between runs) | first-token latency on fresh prompts | trivial daemon |
| Q9 | ANE toe-dip: lm_head as CoreML fp16 on the Neural Engine | frees MPS + power headroom; tests the three-engine split | narrow scope first |
| Q10 | ANE full resident spine (CoreML stateful model, all KDA/MLA/shared/latent projections) | the big power-envelope play — third compute engine | prototype after Q9 |
| Q11 | Thunderbolt cold-tier: full model on the LAN Mac (2 TB free), hot experts local — no requant needed | kills the disk wall AND the sub-4-bit quality question | needs a TB cable |
| Q12 | setiopolicy_np(IOPOL_IMPORTANT) on I/O threads; QoS userInteractive for the driver | protects 6.6 GB/s from background I/O; contention measurably hurt today | 15 lines |
| Q13 | Async cache writes (fetch path currently blocks on the cache store) | small | fold into Q3-era cleanup |
| Q14 | Deterministic-replay memoization: (prefix → next token) text-side cache | instant replay of repeated prompts/demos | trivial, free |
| Q15 | Compress e8m0 scale tensors only (83% ratio, ~6% of expert bytes) | ~5% fetch bytes | probably not worth the format change |
| Q16 | ds4-fork C engine (PLAN.md §5) — everything above feeds it | the next order of magnitude | the real destination |

## ❌ Tested and killed (negative results are results)

| # | Idea | Cause of death |
|---|------|----------------|
| K1 | Latent-expert SVD sketches: approximate-on-miss / progressive sketch+residual / shared per-layer basis | K3 experts are trained dense: flat spectra (rank-128 = 13–15% energy, 93% output error; stable rank ~700/3072); cross-expert basis projects at exactly the random floor; rank for <30% error would be 2.3× LARGER than the full expert |
| K2 | APFS transparent compression on expert files | MXFP4 payload entropy 7.51 bits/byte (max ~6%); decmpfs refuses to store it; kernel decompress 1.7 GB/s < SSD 6.6 GB/s — dies twice |
| K3 | lz4/zstd on-the-fly expert compression | lz4: 0.00%; zstd: 5.7% (at the entropy wall) |
| K4 | >20 GB/s/core CPU dequant | architecturally impossible: fp32 output is 8× packed input; Firestorm store bandwidth caps at ~13 GB/s/core — led directly to the fused-GEMV fix (S1) |
| K5 | HTTP/2 multiplexing for expert fetch | measured 1.7× SLOWER than HTTP/1.1 keep-alive (shared congestion window + client overhead) |
| K6 | MTP self-speculation | K3 ships `num_nextn_predict_layers: 0` — no MTP head exists |
| K7 | Draft-model speculation for pure expert-streaming engines | streamed bytes scale with accepted tokens (near-disjoint expert sets) — note: revived as T2 for THIS driver because its cost structure is resident-dominated |
| K8 | Two-Mac tensor parallel (ds4-style) | both expert halves must be resident: 2.8T at any bpw ≫ 2×64 GB |
| K9 | Cross-expert range coalescing beyond adjacency | non-adjacent gaps are always ≥17.55 MB (one full expert) — merging always downloads ≥1 expert of waste; adjacency-only version kept (free, ~1.7% of requests) |
| K10 | fp16 weights on MPS as a default | ~0.1 logit noise swapped near-tie candidates (ranks 3–4 at token 0) and **diverged the greedy sequence at token 3** (" The Eiffel" → " The population"); coherent but not logit-faithful, and divergence also leaves the warm expert cache. Survives as **approx mode** (`K3_APPROX=1`: fp16 + speculation) — deliberately NOT named "fast mode" until a quiet-machine A/B proves it faster; contended gate numbers showed no win |

## Measured constants (for future arithmetic)

SSD 6.6 GB/s (4–8-thread F_NOCACHE) / 7.0 GB/s (buffered+RDADVISE); GPU 283–290 GB/s read,
9.07 TFLOPS fp32; CPU ~105 GB/s read, no i8mm/bf16; HF single-stream 24 MB/s, 8-stream 56 MB/s,
v2 4-conn 57.7 MB/s; expert = 17,547,264 B (33.03 M params, MXFP4); per-token expert touch
= 16×92 = 1,472 selections ≈ 25.8 GB uncached; resident spine 113.5 GB bf16 / 60 GB int8;
consecutive-token expert overlap 39.1–39.7%; top-16/896 in-sample routing mass 40.8%.

---

## Post-download campaign (queued 2026-07-28, starts when the expert pool is 100% local)

Once all 82,432 experts are on local disk, expert *fetching* stops being the
bottleneck (25.8 GB/token from NVMe at 6.6 GB/s ≈ 4 s, versus minutes over the
network). The problem reorders: **expert compute and resident I/O become the
entire cost.** Measured shares from the last profiled token, and what each phase
of the campaign targets:

| phase | measured now | target | approach |
|---|---|---|---|
| expert fetch | 79 s | ~4 s | (free — the download itself) |
| MoE kernel | 43 s | ~5 s | Metal expert kernel, validated at 150 GB/s = 9.5× the CPU path |
| resident I/O | 49 s | ~25 s | int4 spine (opt-in, quality-gated), better overlap |
| other compute | ~7 s | ~3 s | torch.compile on the two layer templates |

**Order of work** (each step re-profiled before the next, since the bottleneck moves):

1. **Re-baseline.** Fresh `K3_PROFILE=1` run on a quiet, fully-local machine. Every
   number above was taken under contention and must be re-measured before decisions.
2. **Metal expert kernel** (prototype validated, bridge written in advance). Biggest
   single lossless win available. Must batch a whole layer per command buffer —
   per-expert dispatch measured 29 GB/s versus 150 GB/s batched.
3. **Batched CPU GEMV** — one dylib call per layer instead of 48, persistent
   P-core-pinned pool. Useful on its own and as the fallback path.
4. **torch.compile** on the KDA/MLA templates. Template-layer reuse made this viable
   by giving the graph stable shapes *and* stable parameter identities.
5. **RAM-pinned hot-expert tier** — with 1.45 TB on disk and ~40 GB of RAM spare, pin
   the highest-frequency experts. Sizing decided by the holdout locality study.
6. **int4 spine** — opt-in only, and only if the quality gate passes (int8 shifted the
   top logit 0.07% and preserved top-5 order; int4-g64 is expected to be ~3× worse).
7. **Quality harness** (avg NLL vs the official API) — the instrument that lets the
   lossy dials (top-k truncation ×1.33–2.0, cache-aware routing +39% on colibri's
   measurements) be *measured* rather than argued about.
8. **ANE** — lm_head as a CoreML fp16 model first, then possibly the whole resident
   spine. The third compute engine on the die is still completely idle.

The destination remains the ds4-style C engine (PLAN.md §5), where the Python
overhead disappears entirely. Everything above feeds it.

### Prep round results (2026-07-28, written while the expert download ran)

| item | status | evidence |
|---|---|---|
| **Metal MoE bridge** (`tools/metal_moe.{mm,py}`, `tools/metal/moe_mxfp4.metal`) | ready to benchmark | 2.5e-7 relative error vs the CPU path on a real 16-expert decode layer (target was 1e-4); bindless and per-expert dispatch modes bit-identical to each other; zero-copy confirmed (0 host copies on the mmap path); w1/w3 fused into one pass over x, so an expert costs 2 GEMV passes not 3 |
| **Batched CPU GEMV** (`tools/fused_gemv_batch.c`, `tools/fast_moe_batch.py`) | ready to benchmark | **max relative error exactly 0.000e+00** vs the shipped path — it `#include`s the shipped kernel rather than reimplementing it; 48 ctypes calls per layer become 2, with zero thread creations after pool init (was 192 create/join per layer) |
| **int4 spine** (`tools/convert_spine_int4.py`, `tools/int4_loader.py`) | **DROP for uniform int4** | built and bit-exact (25/25 checks, MPS == CPU), but the quality gate fails: int4-g64 weight error is **10.1× int8** (11.1% vs 1.1% relative Frobenius) — about 3× worse than predicted, and in the wrong direction. g=32 is 9.0×, g=128 is 11.1×. Not worth an end-to-end benchmark; kept in-tree for anyone who wants to try mixed precision |
| **Expert locality study** (`tools/expert_locality.py`) | done, corrective | see below |

Two latent bugs were found and fixed during bring-up, both of which would have been
painful later: the Metal wrap cache could alias a *stale* expert after an LRU eviction
unmapped its backing pages, and the batched pool could deadlock when a worker latched
its generation counter before the dispatcher bumped it.

**The locality study corrected our own published numbers.** The router trace is an
append log across many re-runs of a handful of prompts — only 55% of token-passes have
a distinct routing signature. Re-measured on unique contexts with a 60/40 chronological
holdout: previous-token recall is **30.8%** (not 39.7%), and a 40 GB frequency-pinned
RAM tier gets **41.1%** (not 51.9%). The gap between honest and in-sample numbers
*widens* with cache size, because the tail of the pinned set is exactly the part that
does not generalize. Also learned: global LRU is useless at any size below one token's
1,472-expert working set, and cross-layer co-activation lift is only ~1.20× median —
too weak to justify building a router surrogate on.


### Fully-local baseline (2026-07-28) — the campaign's assumptions were wrong

First profile with all 82,432 experts on local disk. **Decode token: 30 s** (was 60–76 s
while streaming). Per-token split, measured not estimated:

| phase | per token | share | what I predicted |
|---|---|---|---|
| expert reads | **12.9 s** | 43% | ~4 s |
| resident spine apply | **13.6 s** | 45% | ~25 s |
| MoE kernel | **1.9 s** | 6% | ~43 s |
| other compute (attention, norms) | 2.0 s | 7% | ~7 s |

Two corrections worth recording:

1. **The MoE kernel was never the problem.** The 43 s figure came from a run whose
   "compute" bucket also contained network waits and prefill. At 1.9 s it is 6% of a
   token, so the Metal expert kernel — validated at 9.5× the CPU path — is worth about
   1.7 s here, not tens of seconds. It stays on the list; it is no longer the headline.
2. **We repeated ds4's mistake in our own code.** Expert reads run at 25.8 GB / 12.9 s =
   **2.0 GB/s**, which is precisely the cold-mmap-demand-fault rate measured on day one
   (0.66 GB/s single-threaded, 2.0 GB/s with parallel faulting) — because `fetch_v2`
   returns `np.memmap` views and lets the GEMV kernel fault pages in as it computes. The
   threaded `F_NOCACHE` pread pool we benchmarked at 6.6 GB/s was never wired into the
   cache-hit path. We criticised ds4 for exactly this class of bug (`posix_madvise` being
   a no-op on macOS) and then shipped our own version of it.

Revised priority order: expert read path, then resident apply, then Metal MoE, then
torch.compile. The lesson that keeps repeating: profile on the real configuration before
choosing what to optimise.


### I/O campaign results (2026-07-28) — decode 32 s -> 16 s, both now default-on

Sequential A/B on a quiet machine, fully local, 2 decode tokens each. Every config
produced the reference completion " Paris.".

| config | decode token | wall (prefill + 2 tokens) | resident apply | expert fetch |
|---|---|---|---|---|
| baseline (mmap + torch dequant) | 32 s | 107 s | 28.2 s | 46.8 s |
| pread expert reads only | 24 s | 71 s | 27.9 s | 26.4 s |
| fast spine only | 23 s | 75 s | **3.6 s** | 50.7 s |
| **both (now the default)** | **16 s** | **57 s** | 4.1 s | 32.4 s |
| both + speculative L+1 prefetch | 16 s | 57 s | 4.1 s | 32.4 s |

**Exactly 2× on decode, and the two fixes compose** (each ~25% alone, 50% together).

**S9 — expert reads via threaded F_NOCACHE pread** (`K3_EXPERT_READ`, default `pread`).
The old path returned `np.memmap` views and let the GEMV kernel fault pages in while it
computed. Cold micro-benchmark: memmap+touch **0.87 GB/s**, threaded pread + `F_NOCACHE`
**6.85 GB/s**. Isolated read path 40.2 s -> 4.3 s per token (9.3×). A whole layer's 16
experts are now read in parallel from the router's ids instead of faulting in serially.
Note the old accounting understated this: under mmap, `fetch_experts` did almost no I/O,
so the cost was split between the `expert_fetch` and `moe_kernel` buckets.

**S10 — fast resident spine** (`K3_FAST_SPINE`, default on). Measurement, not guesswork,
found the cost was *not* file reads (already hidden in the preloader — proven with a
`preload_wait` counter reading 0.4 s) but torch's row-broadcast multiply on MPS: **43.7
GB/s versus 334 GB/s for a plain copy of the same traffic.** Nine PyTorch formulations
topped out at 60 GB/s; a custom Metal dequant kernel via `torch.mps.compile_shader` hits
**297 GB/s**. Two further finds: a 634 MB MPS alloc/free per layer cost more than the
dequant (fixed with persistent staging buffers), and blocking H2D transfers wedged
between kernel dispatches stalled the main thread on the GPU queue (fixed by hoisting
them). Per-layer apply 117.8 ms -> 21.2 ms; bit-exact, `max|diff| = 0.000e+00` across
nine real layers.

**Speculative L+1 expert prefetch: no measurable gain** (`K3_EXPERT_PREFETCH`, stays
off). Once reads run at disk speed there is nothing left to hide, and at ~31% recall it
reads ~1.6× the bytes for it. Kept as a flag, documented as a dead end.

Where a 16 s token now goes (approximate, from the D column): expert reads ~6 s,
attention/norm compute ~5 s, MoE kernel ~2 s, resident apply ~2 s, other ~1 s. The
remaining read cost is partly the 12,676 legacy `.npz` cache entries (~15% of selections)
that still go through `np.load` at ~2.9 GB/s — converting them to raw `.bin` is the next
cheap read-path win. After that the ranking is: Metal MoE kernel (~2 s), then
`torch.compile` on the layer templates (~5 s of attention compute).
