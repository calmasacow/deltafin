# How the native runtime works

One Rust executable owns long-lived state and calls a small, versioned C ABI into provider code linked into the same process:

```text
deltafin
  Rust: CLI, HTTP, tokenizer, chat template, scheduler, storage, cache, output
    -> versioned in-process ABI
       C++/LibTorch: K3 tensor program and provider-owned KDA/MLA state
         -> MPS + Metal on Apple Silicon
         -> CUDA + native CUDA MXFP4 on NVIDIA Linux
         -> CPU + NEON/AVX SIMD MXFP4 fallback
```

There are no internal HTTP calls between inference stages and no helper processes per layer or token. Weight downloads use authenticated HTTPS because they cross the network; local model execution uses files, memory and direct native calls.

The most unusual retained optimizations include exact next-layer expert prefetch, CUDA cache-aware pre-I/O planning, provider-owned speculative cache transactions, persistent chat state, lossless scale-sidecar decoding, GigaToken-inspired parallel tokenizer batching, DSpark target-row capture and wide target verification. The detailed design, qualification rules and measurements are in [How Deltafin's optimizations work](OPTIMIZATIONS.md). The compiled ownership boundary and bugs found during the rewrite are in [Deltafin's custom compiled runtime](COMPILED-RUNTIME.md).

## Router tracing and offline expert warming

Tracing is off by default. A buffered native trace records the authoritative 16 routed IDs and fp32 weight bits after routing; it does not change the route:

```bash
./target/release/deltafin run --chat \
  --prompt "Explain how tides work." \
  --router-trace k3-meta/router-trace.jsonl \
  --router-trace-mode buffered
```

`sync` flushes each completed pass for crash-focused diagnosis and is slower. The trace writer rejects symlinks, caps files at 8 GiB and reserves bounded memory. Use recorded routes to plan or explicitly fetch likely missing experts:

```bash
./target/release/deltafin warm-expert-cache \
  --trace k3-meta/router-trace.jsonl --show 50

./target/release/deltafin warm-expert-cache \
  --trace k3-meta/router-trace.jsonl --fetch 128 --workers 8
```

Planning is read-only unless `--fetch N` is present. Fetches reuse the same pinned inventory, HTTPS authentication and atomic publication rules as setup. Installations created by an early Deltafin release can migrate six-member expert NPZ files without Python:

```bash
./target/release/deltafin warm-expert-cache --convert-npz
```

The native converter validates ZIP/NPY structure, dtype, shapes and CRCs, reconstructs the canonical raw span, verifies its SHA-256 after a durable atomic publication, and only then removes the legacy NPZ. Add `--keep-npz` to retain the authenticated source during a cautious first pass.

## Automatic PILOT prefetch and CUDA expert residency

On a complete local expert corpus with CPU or Metal execution, the native provider keeps an immutable, scheduling-only roster of all 92 router/norm boundaries. PILOT derives one canonical 16-ID hint for the next layer from the current exact activation. Rust starts those one-expert reads in a bounded 17-slot arena while the current layer executes. After authoritative routing, predicted losers are cancelled before draining; hits are reused without a copy, misses use the ordinary authenticated reader, and experts are assembled in exact authoritative order. Prediction never supplies weights or replaces the real router.

CUDA uses a different exact boundary. Before opening expert files, the provider freezes a per-session residency snapshot for the authoritative route union and returns the complete ordered list of cache misses. Rust reads only those misses. Hits remain device-owned and ready events order their use on the active stream. Automatic capacity is derived from live free VRAM only after model allocations, retaining at least 2 GiB or 20% as headroom. A failed qualification falls back to the native CPU expert path without weakening resident CUDA work.

## GigaToken-inspired native tokenization

Deltafin credits [GigaToken](https://github.com/marcelroed/gigatoken) and its author [Marcel Rød](https://github.com/marcelroed) for the stable-order native batching design that motivated the server tokenizer experiment. The production runtime now implements K3's exact rank-file tokenizer directly in Rust; it does not install or load the GigaToken wheel. Small prompts stay sequential. For a large, already-classified chat history, independent XTML segments fan out automatically only after the measured crossover (at least eight segments and 128 KiB), then rejoin in stable order. The same native decoder remains the sole source of token IDs, and any worker failure rejects the whole encode.

The earlier isolated M1 Max server fixture measured segment encoding at 0.438 ms versus 0.155 ms for 453 rendered characters, 67.225 versus 9.735 ms at 100 synthetic turns, and 664.987 versus 95.986 ms at 1,000 turns. These are once-per-request preprocessing measurements, not token-generation claims.
