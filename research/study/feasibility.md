# Kimi K3 on M1 Max 64GB — hard feasibility math

Assumed K3 shape (from kimiArch facts, to be re-derived when config.json drops): 2.8T total; ~92 MoE layers of 896 experts, 16 active + shared; ~34M params/routed expert (2.8T/(896×92)); routed pool ≈ 2.75T params; non-routed (attention ~100M/layer×93 ≈ 9.3B, shared experts 34M×92 ≈ 3.1B, lm_head 163840×7168 ≈ 1.2B, embeddings) ≈ 15B stored, ~13.5B touched per token. Cross-check: 16×92×34M = 50.0B routed active + 13.5B non-routed = 63.5B ≈ the official "~50-60B active" estimate. Hardware constants (measured, appleHw): SSD 6.6 GB/s (F_NOCACHE pread pool, 4-8 threads), GPU RAM read 283-290 GB/s, CPU read ~105-107 GB/s, GPU 7.4-9.1 TFLOPS, free disk ~1.2 TB.

## Task 1 — decode working set (bytes/token)

Routed active/token = 50.0B params (16 experts × 92 layers). Non-routed touched/token ≈ 13.5B params.

| avg bpw (incl scales) | (a) resident dense/attn/shared, stored / touched per token | (b) routed experts touched/token | routed pool on disk |
|---|---|---|---|
| 2.0625 (IQ2_XXS-class) | (dense should NOT be 2-bit) | **12.9 GB** | 709 GB |
| 2.5 (IQ2_XS/Q2_K mix) | — | **15.6 GB** | 859 GB |
| 3.0625 (E8/IQ3, colibri fmt=6) | — | **19.2 GB** | 1053 GB |
| 4.25 (MXFP4 native / int4-g32) | 8.0 GB stored / 7.2 GB touched | **26.6 GB** | 1461 GB |
| dense at int8 (ds4 recipe) | 15 GB stored / 13.5 GB touched | — | — |

Practical resident set: dense at mixed int4/int8 ≈ **8-15 GB RAM** (touched ~7-13.5 GB/token from RAM — 25-47 ms at 290 GB/s, never the bottleneck). KV: KDA hybrid means ~23 MLA layers × 576 dims ≈ 26 KB/token → ~3.3 GB at 128K ctx, plus ~150 MB constant KDA state. Long context is nearly free; nearly all remaining RAM goes to the expert cache.

**RAM budget:** 64 GB − dense 10 − KV 1.5-3 − OS/apps ~6 − page-cache reserve 2.5 (colibri measured 800→180 MB/s when starved) − slack 2 ≈ **~40-42 GB expert cache**. At 2.06 bpw that is ~4,900 experts = **5.9% of the routed pool**; at 3.06 bpw ~3,300 experts = 4.0%; at 4.25 bpw ~2,400 = 2.9%. This fraction is the single number that governs everything below. Note: the machine currently shows 28.5 GB swap in use — that must be cleared and the cache mlock'd/wired or the compressor eats the budget.

## Task 2 — tok/s ceilings (t = max(missed_bytes/6.6 GB/s, (dense+resident_expert bytes)/290 GB/s + ~30 ms dispatch slop))

| config | 2.06 bpw | 2.5 bpw | 3.06 bpw | 4.25 bpw |
|---|---|---|---|---|
| all-RAM hypothetical (GPU, 290 GB/s) | 17.7 | 14.9 | 11.9 | 8.6 |
| all-RAM hypothetical (CPU, 105 GB/s) | 6.4 | 5.4 | 4.3 | 3.1 |
| 90% hit | 5.1 | 4.2 | 3.5 | 2.5 |
| 70% hit | 1.7 | 1.4 | 1.15 | 0.83 |
| 50% hit | 1.0 | 0.84 | 0.69 | 0.50 |
| cold / 0% hit (6.6 GB/s) | 0.51 | 0.42 | 0.34 | 0.25 |
| cold via naive mmap faulting (0.66-2 GB/s, ds4 as-is on macOS) | 0.05-0.16 | 0.04-0.13 | 0.03-0.10 | 0.02-0.08 |

**The brutal part: what hit rate is actually reachable?** Under uniform routing, hit ≈ cache byte-fraction = ~6% → 0.54 tok/s, barely above cold. Every number above 1 tok/s is a bet on router skew. The only calibration point we have: GLM-5.2 with cache = 12.7% of routed bytes achieved 72.5-74.5% hit (colibri M5 report) and ds4's 59 GB cache on 188 GiB (31%) implies ~72% hit. K3 gives us **half GLM's cache fraction (5.9% vs 12.7%) on a router with 896 experts and modern load-balancing losses** (which are explicitly trained to flatten usage). A realistic band is **h ≈ 35-60%** with learned AUTOPIN + PILOT lookahead; 70% is optimistic; **90% is unreachable honestly** — it would need the cache to hold ~25-40% of a ≥709 GB pool, i.e. a 256 GB machine. 90%+ *effective* hit is reachable only by cache-aware routing (CACHE_ROUTE), which substitutes expert IDs and costs quality (measured +39% speed on GLM at 121 GB; the quality cost at K3's needed substitution rate is unmeasured and could be severe).

**Prefill:** 4096-token chunks activate essentially all 896 experts/layer (1−(1−16/896)^4096 ≈ 1), so a chunk streams ~the whole non-cached routed pool: at 2.06 bpw, 0.94×709 GB/6.6 GB/s = 101 s/chunk → **~40 tok/s prefill ceiling** (27 tok/s at 3.06 bpw); GPU compute floor for the chunk (~2×50B FLOP/token at 7.4 TFLOPS) is ~55 s, so prefill is jointly disk/compute-bound around 30-40 tok/s. Usable, and the KVC/session-persistence tricks make it pay once.

**MTP/speculation:** All K2-family configs shipped num_nextn_predict_layers=0 — assume no MTP head until config.json proves otherwise. Even if present: when disk-bound, verifying k tokens routes each token independently, touching up to k×16 near-disjoint experts/layer (at 896 experts, overlap between 3 tokens' top-16 is small), so streamed bytes grow ≈ linearly with accepted tokens — **speculation is roughly a wash in the disk-bound regime** (colibri's own best M5 recipe runs MTP=0). It becomes a ×1.5-2.5 lever only in a high-hit/resident regime this machine cannot reach. Same logic kills Kimi-Linear-48B draft speculation as a throughput play.

**A cut that DOES multiply throughput:** runtime top-k reduction. Dropping to the top-12-of-16 experts by score mass cuts streamed bytes ×0.75 (→ +33% tok/s); top-8-of-16 halves it (→ ×2). At h=50%, 2.06 bpw: 1.0 → 2.0 tok/s. Lossy, but the loss is graduated and measurable, unlike hoping for hit-rate miracles.

## Task 3 — disk budget (~1.2 TB free)

- MXFP4 native (4.25 bpw): **1.46-1.49 TB — does not fit. Full stop.** Requantization or pruning is mandatory, despite the QAT fidelity argument.
- 3.0625 bpw experts + int4/8 dense: 1053 + ~10 GB = **~1.06 TB — fits with only ~140 GB slack**. No room for a partial mirror, KV session files are fine, but this is the ceiling quant.
- 2.5 bpw experts: ~870 GB — fits with ~330 GB room.
- 2.0625 bpw experts: ~720 GB — fits with ~480 GB room, enough for a hot-expert dual-precision shard (e.g. top ~10-15k experts also stored at 4.25 bpw) or a partial external-SSD mirror.

**Answer: 3.06 bpw is the maximum that fits; 2-2.5 bpw is the maximum that fits AND leaves working room.** The cfse rANS layer (1.37× measured on int4) cannot rescue MXFP4: 2-bit-class weights are near-white (expect ≤1.1×), and 1.49/1.37 = 1.09 TB would fit only with zero margin and unbenchmarked decode throughput.

## Task 4 — do the engines' claims corroborate the model? Yes, tightly.

- **colibri "~11 GB/token cold" on GLM-5.2:** 8 experts × 75 layers × 19 MB = 11.4 GB. Exact match.
- **colibri 2.24 tok/s on M5 Max at ~74% hit:** missed bytes = 0.26×11.4 = 2.96 GB; t = 0.446 s → implied effective SSD bandwidth **6.5 GB/s** — i.e. the tuned recipe (DIRECT=1, PIPE, Metal overlap) achieves essentially 100% of raw SSD bandwidth. The ceiling model is not just an upper bound; it is *attained* by a well-engineered stack. This is the strongest corroboration.
- **colibri 0.05-0.1 tok/s on a 25 GB box:** implies effective 0.6-1.1 GB/s on 11.4 GB/token — consistent with single-threaded demand reads + starved page cache. Confirms naive streaming loses 5-10× vs raw SSD; the 0.05 regime is real and is exactly where an unported/untuned engine lands.
- **colibri 1.8 tok/s warm CPU-only 128 GB:** ~74% hit → 3 GB missed at ~5 GB/s single-thread ≈ 0.6 s + 0.2 s CPU compute ≈ 1.25-1.8 tok/s. Consistent.
- **colibri 5.8-6.8 tok/s on 6× RTX 5090 full residency:** ~21 GB touched/token over multi-TB/s aggregate → latency/kernel-bound, plausible; irrelevant to this machine except as proof that even full residency of a 744B MoE lands single-digit tok/s.
- **ds4 4.8 tok/s GLM SSD-streaming on M5 Max:** cold ≈ 5.2 GB/token at IQ2; t = 0.208 s at ~7 GB/s implies ~72% hit with a 59 GB cache on 188 GiB (31% cached). Consistent, and confirms the hit-vs-cache-fraction curve I used.
- **ds4 M1 Max extrapolation 11-13 tok/s for resident 13B-active q2:** M3 Max measures 26.7 tok/s at ~30% of theoretical bandwidth; scaling to 290 GB/s gives ~19, so 11-13 is conservative-plausible. Irrelevant to K3 (nothing resident-sized about it).

Every claim fits one model: **decode time ≈ missed expert bytes ÷ effective SSD bandwidth, with compute hidden underneath.** No engine number contradicts the physics; none of them promises what K3-on-64GB cannot deliver.

## Task 5 — verdict: three realistic configs, ranked

**Config 1 — "speed build": experts ~2.06-2.2 bpw imatrix (IQ2_XXS gate/up + Q2_K down, ds4 recipe), dense/attn/shared int4-int8 (~10 GB), ~40 GB mlock'd learned pin + LRU, F_NOCACHE pread pool (4-6 threads), Metal MoE path with resident-first submit, PILOT lookahead, moderate CACHE_ROUTE + top-12-of-16 score-mass cut, optional external TB4 SSD partial mirror (+3 GB/s on hot shards).**
Math: 12.9 GB/token cold; h ≈ 45-60% honest → 0.9-1.3 tok/s; top-12 cut ×1.33 → 1.2-1.7; CACHE_ROUTE pushing effective h to 75-85% → 2.2-3.5; external mirror ×~1.3-1.45 on the still-missed bytes → **expected 1.5-3 tok/s, central ~2**. Biggest risk: **compounded quality collapse** — 2-bit requant of a model QAT'd at MXFP4, plus substituted routing, plus dropped experts, on a 16-expert router; nobody has measured this stack, and ds4's avg_nll-vs-API gate must be run before believing any token.

**Config 2 — "honest build": same engineering, experts 2.06-2.5 bpw, true routing (no CACHE_ROUTE, no expert drop), learned AUTOPIN.**
Math: h ≈ 35-60% at 5-6% cache fraction → **0.7-1.3 tok/s, central ~1.0**. Biggest risk: **K3 router flatness** — if aux-loss balancing makes hit ≈ cache fraction, this degrades to 0.55 tok/s, indistinguishable from cold; the router-skew histogram from the first calibration run is the go/no-go measurement for the whole project.

**Config 3 — "max quality that fits": experts at 3.0625 bpw (colibri fmt=6 E8/IQ3 or IQ3_XXS), everything else as Config 2.**
Math: 19.2 GB/token cold, cache fraction 4.0% → h ≈ 30-50% → **0.45-0.7 tok/s**. Biggest risk: it fills the disk to ~1.06/1.2 TB (no mirror, no dual-precision shard, no slack) and lands at a speed that is not interactively usable — you pay maximum engineering for a model you will not want to wait for.

**Fantasy numbers to strike from any plan:** ≥5 tok/s decode (needs 90% honest hit = a 256 GB-class cache); MXFP4-native local storage (doesn't fit); MTP/draft speculation as a disk-bound multiplier (bytes scale with accepted tokens); ds4 unmodified on macOS (posix_madvise WILLNEED is a no-op → 0.66-2 GB/s → **0.03-0.15 tok/s, genuine 0.05 territory**); two-Mac TP (both expert halves must be resident — impossible at 2.8T); and any number quoted without stating the assumed hit rate. Even the physics ceiling of this machine — model magically all-resident at 2 bpw, GPU-fed at 290 GB/s — is **~18 tok/s**; everything real is a fraction of that governed by (1−h)×12.9 GB ÷ 6.6 GB/s.

## KEY FACTS
- K3 decode working set: 16 experts x 92 MoE layers x ~34M params = 50.0B routed params touched/token; plus ~13.5B non-routed (attention ~9.3B, shared experts ~3.1B, lm_head ~1.2B) touched from RAM
- Streamed expert bytes/token: 12.9 GB at 2.06 bpw, 15.6 GB at 2.5 bpw, 19.2 GB at 3.06 bpw, 26.6 GB at 4.25 bpw (MXFP4)
- Cold-streaming decode ceiling at measured 6.6 GB/s SSD: 0.51 / 0.42 / 0.34 / 0.25 tok/s at 2.06 / 2.5 / 3.06 / 4.25 bpw; via naive mmap faulting (0.66-2 GB/s, ds4 unmodified on macOS) it is 0.03-0.16 tok/s
- Hit-rate ceilings at 2.06 bpw: 50% hit -> 1.0 tok/s, 70% -> 1.7, 90% -> 5.1; all-RAM hypothetical GPU ceiling 17.7 tok/s (2 bpw) to 8.6 (4.25 bpw) at 290 GB/s measured GPU read
- Expert cache budget on 64 GB: ~40-42 GB after dense (~10 GB), KV (~1.5-3 GB, KDA hybrid), OS, page-cache reserve = only 5.9% of the 709 GB routed pool at 2.06 bpw (~4,900 of 82,432 experts); GLM achieved 72-74% hit with 12.7-31% cached, so K3 honest hit is realistically 35-60%, and 90% honest hit is unreachable on 64 GB
- Disk: MXFP4 native (1.46-1.49 TB) does NOT fit 1.2 TB free; 3.0625 bpw experts (~1.06 TB) is the maximum that fits (140 GB slack); 2.0625 bpw (~720 GB) fits with room for a hot-expert dual-precision shard or partial external mirror
- Engine claims corroborate the model exactly: colibri's 2.24 tok/s on M5 Max at 74% hit implies 6.5 GB/s effective SSD (full raw bandwidth attained); its ~11 GB/token GLM cold cost matches 8x75x19MB; ds4's 4.8 tok/s implies ~72% hit at 31% cache fraction; colibri's 0.05-0.1 tok/s floor implies 0.6-1.1 GB/s effective, i.e. untuned streaming loses 5-10x vs raw SSD
- MTP/speculative decoding is ~a wash when disk-bound: k verified tokens route near-disjoint expert sets (k x 16 of 896 per layer), so streamed bytes scale linearly with accepted tokens; all K2-family models shipped num_nextn_predict_layers=0 so K3 likely has no MTP head anyway; colibri's own best Apple recipe runs MTP=0
- Runtime top-k reduction is the strongest honest lever: dropping to top-12-of-16 experts by score mass gives x1.33 tok/s, top-8-of-16 gives x2.0, with graduated measurable quality cost
- Prefill ceiling ~30-40 tok/s: a 4096-token chunk activates essentially all 896 experts/layer, streaming ~the whole non-cached routed pool (~101 s/chunk at 2 bpw) with GPU compute floor ~55 s/chunk
- Verdict ranking: (1) 2.06-2.2 bpw + CACHE_ROUTE + top-12 cut + optional TB4 mirror = 1.5-3 tok/s central ~2, risk = compounded quality collapse; (2) same at true routing = 0.7-1.3 tok/s central ~1.0, risk = router flatness degrading to 0.55; (3) 3.06 bpw max-quality = 0.45-0.7 tok/s with zero disk slack
- Machine currently has 28.5 GB swap in use; the resident set must be mlock'd/wired and F_NOCACHE used everywhere or the macOS compressor+swap fight the streaming SSD directly

## IMPROVEMENT IDEAS
- Measure K3's router-skew histogram on a calibration set the day weights land: hit rate at 6% cache fraction is THE go/no-go number; if cumulative routing mass of the top ~5,000 experts is under ~40%, only CACHE_ROUTE-style substitution or expert pruning can save decode speed
- Implement runtime top-k reduction (drop lowest-score-mass experts of the 16, renormalize weights): x1.33 at top-12, x2.0 at top-8 on disk-bound decode - a bigger, more controllable lever than any cache trick, and composable with all of them
- Dual-precision hot shard: with 2.06 bpw cold experts (720 GB) there is ~400 GB disk slack - store the top ~10-15k hottest experts additionally at MXFP4/int4 and serve cache hits from the high-precision copy, so quality loss concentrates only on rarely-used streamed experts
- Add an external TB4/USB4 SSD (~3 GB/s) as a deterministic-hash partial mirror of the hottest shards (colibri COLI_MODEL_MIRROR accepts partial mirrors): aggregate ~9.6 GB/s on missed bytes = x1.3-1.45 on every streaming-bound config
- Port the exact proven Apple recipe as defaults: F_NOCACHE pread pool of 4-6 threads at 1-8 MB (6.6 GB/s measured vs 0.66-2 naive), F_RDADVISE prefetch from a PILOT-style lookahead thread, passive OMP wait (active spin cost -39% on M5 Max via shared power budget), one-command-buffer batching with MTLResidencySet, resident-experts-submitted-before-miss-preads overlap, and a DVFS keep-alive kernel during miss stalls
- Exploit KDA: ~70 of 93 layers need only a ~150 MB constant recurrent state, so KV at 128K ctx is ~3.3 GB not ~30 - explicitly reassign that saved RAM to the expert cache in the budget formula, and persist the tiny state for instant session resume
- Split compute by bandwidth domain: GPU (290 GB/s) runs attention, dense, shared and cached-hot experts; CPU NEON sdot (105 GB/s, no i8mm on M1) runs the cold streamed experts whose arrival rate is only 6.6 GB/s - hides essentially all streaming compute
- Skip MTP/draft speculation and DeepSeek-style compressed-attention porting entirely for week one; budget that engineering into streaming overlap quality and the quant pipeline, which the ceiling math shows are the only terms that matter
- Validate every quality-costing lever (2-bit requant, CACHE_ROUTE substitution rate, top-k cut, expert pruning) with ds4's avg_nll-vs-official-API-continuations gate before stacking the next one - the three levers compound and no one has measured the stack
- Benchmark cfse/rANS or zstd-light decode throughput per E-core before betting on on-disk compression: 2-bit-class weights are near-white so expect <=1.1x, useful only if decompress exceeds ~1 GB/s/core and runs in the pread pool threads