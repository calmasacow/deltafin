# Performance reference

For reproducible performance evidence, use the native benchmark harness rather than timing terminal output:

```bash
./target/release/deltafin benchmark \
  --prompt "The capital of France is" \
  --max-new 17 --reps 3 \
  --expect-text " Paris. The Eiffel Tower is located in Paris. The Louvre Museum is also"
```

The harness launches the current native executable, records structured events, supports interleaved named arms and rejects an arm whose exact token/text oracle does not match. See `deltafin benchmark --help` for `--arm`, `--configs`, `--warmup-steps` and output-directory controls.

The established public reference remains the interleaved 20-run M1 Max campaign; it is retained here rather than being replaced by a slower or uncontrolled rerun. The host was a **64 GB M1 Max MacBook Pro** with local experts, Metal MoE, tracing off, the optional Qwen pair, an explicit int8 resident spine and lossless scale4 expert sidecars. Because int8 changes the resident weights, this is a quantified performance arm—not a benchmark of the default original-BF16 path. Every arm kept the complete 93-layer K3 target, all top-16 routed experts and the established 17-token output oracle.

| Metric | First working version | Raw experts, 10 runs | Lossless scale4, 10 runs |
|---|---:|---:|---:|
| Optional-Qwen exact-oracle steady decode | ~20 min/token | **0.2669 token/s (3.75 s/token median)** | **0.2734 token/s (3.66 s/token median)**; 0.2630–0.2762 token/s range (3.62–3.80 s/token) |

Scale4 improved the median by **2.4%**, and the project moved more than 300× from its first version. Steady decode excludes the first output token, includes draft time and covers the following 16 tokens. This favorable short completion is not representative of long-context chat: draft acceptance changes with text and prefill grows with context.

K3's checkpoint advertises an architectural context capacity of one million tokens, but the current native provider's exact expanded fp32 MLA cache admits a smaller physical bound and prints it at startup. Deltafin does not claim the architectural maximum as usable memory today.

A contrasting 17-token planet completion measured **0.2136 token/s (4.682 s/token)** with the optional Qwen confidence policy versus **12.530 s/token** target-only, with identical output IDs. A creative-chat physical-M1 check measured target-only at **0.081220 token/s** and DSpark at **0.095027 token/s**; both produced the same 25 IDs. These are separate fixtures, not additions to the 20-run reference.

An earlier exact prefix-reuse fixture reused 117 of 174 tokens and reduced time to first token from **245.3 to 92.7 seconds** and four-token wall time from **281.9 to 128.8 seconds**, with the same four raw IDs. Its retained compact state occupied 0.451 GiB. This demonstrates work avoided across a growing conversation; it is not a steady decode-rate claim.

Newer hardware is not ordered simply by generation. Memory bandwidth, RAM, accelerator throughput and NVMe performance all matter; an older Max chip can outperform a newer base chip in this workload. Higher-RAM, higher-bandwidth Max/Ultra Macs and capable NVIDIA workstations should have more room to retain spine data and experts, but only measured results should be quoted as results.
