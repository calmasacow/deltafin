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
| S8 | Previous-token whole-depth expert prefetch (fires the prior token's per-layer sets at step start) | in quartet; recall basis measured 39.1–39.7% |
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
