# How Deltafin's speed paths work

Deltafin is mostly an exercise in moving fewer bytes, moving the necessary
bytes at the right time, and declining an optimization when the running
machine has not proved that it can execute it safely. Kimi K3 is a 2.8-trillion
parameter Mixture-of-Experts model, but one token selects only 16 of 896 routed
experts at each MoE layer. That makes the model runnable on one workstation;
it does not make the remaining I/O small. A local decode still touches roughly
25.8 GB of routed-expert data and, with the int8 resident spine, about 53 GB of
non-expert weights per pass.

This document describes the paths retained in the runtime. It is deliberately
not a catalogue of every experiment we tried. We do not mean to suggest that
the underlying ideas are all new; the interesting part here is often the
particular ownership, fallback, or data-layout contract needed to make an idea
work with K3.

The status labels used below are important:

| Label | Meaning |
|---|---|
| **Default** | Used automatically in the ordinary configuration. |
| **Capability-gated** | Used only after the live runtime or native library proves the required operator, shape, and device behavior. |
| **Opt-in** | Implemented and correctness-gated, but off by default while its complete-token speed or wider hardware coverage is still being established. |
| **Structural evidence** | Exact byte, allocation, call, or synchronization work was removed, but we do not yet publish a tokens-per-second gain for that change alone. |

## Reference-only speculative snapshots

**Status: Default, exact.**

The unusual part of speculative decoding in Deltafin is not the n-gram draft;
it is making a failed draft cheap enough to abandon.

The original safe implementation cloned every KDA recurrent tensor, all three
short-convolution histories, and the MLA key/value views before a speculative
pass. That copied roughly 475 MB merely to create a rollback point. The current
snapshot records references to the old tensor objects instead:

- KDA recurrence and short convolution return new state storage during
  inference, so the previous objects remain immutable.
- MLA uses a geometrically growing slab. Appending writes beyond the old view,
  leaving its prefix unchanged; the old view also records the rollback length.
- A rejected draft restores the old KDA and convolution objects and restores or
  truncates the MLA views.

This turns the measured snapshot operation from 3.56 ms into 0.001 ms and
removes the large clone. The optimization is fundamentally an ownership proof,
not a faster copy.

The tests retain snapshots across accepted drafts, rejected drafts, partial
accepts, cache growth, replay, and reorder operations, then compare the future
token sequence. Deeper speculative passes can capture the KDA recurrence inputs
and replay only the accepted prefix; the whole-reference restore plus rerun
remains the conservative reference path.

Implementation:
[`tools/kimi_run.py`](tools/kimi_run.py),
[`tools/spec_decode.py`](tools/spec_decode.py), and
[`tools/test_snapshot_refs.py`](tools/test_snapshot_refs.py).

## Packed row-int8 output head

**Status: Capability-gated; automatic on the measured MPS path, forceable
elsewhere, with an exact dense fallback.**

The output head is an unusually large matrix: 163,840 vocabulary rows by 7,168
hidden columns. Materializing it as fp32 occupies about 4.7 GB. Deltafin already
has a per-row int8 checkpoint and fp16 row scales, so the faster path keeps the
weights packed and gives them directly to PyTorch's native weight-only int8
operator. Head residency falls to about 1.17 GB, and the separate full-head
dequantization disappears.

There are two additional data-movement details:

1. `torch.from_file` maps the packed checkpoint without first creating another
   1.17 GB Python byte string.
2. During ordinary prefill, generation needs only the final prompt position's
   logits. Deltafin therefore sends only that hidden row through the head.
   Speculative verification still computes every required position.

The native operator is a private PyTorch implementation detail whose device
registrations can change. Deltafin checks symbol and dispatcher availability,
then runs an analytical exact canary at the real KDA projection width. That is
not a head-shaped canary. The first real head call remains exception-guarded. If
it fails, Deltafin materializes the dense head *before* releasing the packed
representation, then uses the established dense calculation from that point
onward.

On the maintainer's M1 Max, balanced full-model runs measured **+17.3% median
steady decode**, **+23.1% prefill**, and **+26.8% fresh-process wall
throughput**. These numbers describe that MPS configuration, not CPU, CUDA, or
every future PyTorch build.

Implementation:
[`tools/packed_q8.py`](tools/packed_q8.py) and
[`tools/kimi_run.py`](tools/kimi_run.py).

## Six KDA projections in one packed-int8 bundle

**Status: Opt-in, exact sequence-qualified on one real MPS stream; no speed
claim yet.**

A KDA layer applies six first-stage projections to the same hidden input:
Q, K, V, G, F-A, and F-B. The ordinary int8-spine path dequantizes all six into
the fp32 template and launches six dense projections. The experimental bundle
instead lays their row-int8 weights and row scales consecutively and asks the
same weight-only operator to process the combined rows. The output is split
back into the six original, unequal role sizes.

This path has two storage modes:

- **Arena** copies the current layer out of the recyclable upload pack into
  template-owned packed storage. It is simpler because the operator can finish
  after the loader has released its source buffer.
- **Stage** binds the packed operator directly to the current upload arena.
  Generation-stamped leases, device events, and a conservative MPS FIFO
  contract prevent the loader from refilling a buffer while an asynchronous
  operator may still read it. A stale lease, failed fence, changed shape, or
  changed stream contract disables the path.

The controller first checks each projection shape and the unequal-row fusion
contract. If a fused call fails, it can retain separate packed calls; if the
packed backend itself fails, it reconstructs the dense weights and atomically
returns the layer to the normal path. Unsupported CPU, CUDA, MPS, dtype, token
count, shape, or ABI combinations never inherit support from another backend.

One streamed-weight MPS sequence executed 816 logical projections as 136
operator calls and emitted the exact reference tokens. That establishes the
mechanism and rollback behavior, but it is not staged-KDA speed evidence.
CUDA packed execution has not been claimed.

Implementation:
[`tools/spine_fast.py`](tools/spine_fast.py),
[`tools/packed_q8.py`](tools/packed_q8.py), and
[`tools/test_dynamic_q8_qkv.py`](tools/test_dynamic_q8_qkv.py).

## Treating the page cache as part of the engine

**Status: The int8 spine and expert-read shaping are default; explicit
resident-tier partitioning remains opt-in.**

With a model this large, RAM left to the operating-system page cache is often
more valuable than an equally sized Python heap cache. The int8 resident spine
helps twice: it approximately halves the bytes read for the non-expert weights,
and it leaves much more memory for recently used spine pages and routed
experts. This is why its effect can exceed the arithmetic suggestion of “half
the I/O” on a machine whose bf16 spine nearly fills RAM.

The checkpoint uses symmetric int8 rows with fp16 row scales. Deltafin selects
it automatically when it has been built, and retains bf16 as the fallback. In
the original quality checks, the top-five next-token candidates kept their
order and the top logit moved by 0.07%. This is a quantized quality result, not
a claim that int8 and bf16 tensors are bit-identical.

The read policy then separates two very different streams:

- Routed experts are a high-volume stream with little reason to evict useful
  spine pages. On macOS, the proven local path applies `F_NOCACHE` to those
  reads. Linux does not receive a Darwin command number; it stays buffered by
  default and has a separate, explicit best-effort `POSIX_FADV_DONTNEED` path.
- The spine is a cyclic scan. When an explicit resident tier is configured,
  a fixed subset of layers remains cacheable and the rest uses streaming cache
  advice. A fixed subset avoids the pathological LRU pattern in which a
  smaller-than-spine cache repeatedly evicts the page needed next.

On the reference Mac, demand-faulting expert mmaps delivered about 0.87 GB/s,
while the parallel read path delivered about 6.85 GB/s; the corresponding
expert-read slice fell from roughly 40 seconds to 4.3 seconds per token.
Those are historical I/O-path measurements, not promises for another SSD or
filesystem.

Implementation:
[`tools/convert_spine_int8.py`](tools/convert_spine_int8.py),
[`tools/spine_io.py`](tools/spine_io.py),
[`tools/spine_cache.py`](tools/spine_cache.py), and
[`tools/fetch_v2.py`](tools/fetch_v2.py).

## Zero-copy Metal expert weights

**Status: Default when MPS and the Metal library are available; numerically
checked with a CPU fallback.**

Each routed expert is stored as one 17,547,264-byte span containing its six
MXFP4 payload and scale tensors at fixed offsets. On Apple Silicon, the Metal
MoE path can wrap a page-aligned local span with
`newBufferWithBytesNoCopy`. The GPU then reads the same pages that the local
expert loader filled; it does not expand the expert into fp16 or fp32 and does
not make a second weight staging copy.

“Zero-copy” here applies to the expert weights. The current bridge still moves
the small activation and result across its synchronous CPU/Metal boundary.

The difficult part is lifetime. Read slots are recycled, and a later expert can
occupy the same virtual address. Deltafin pins the Python object that owns each
mapping for as long as Metal retains a wrapper. If an address appears with a
different owner, the stale wrapper is dropped before reuse. If the same live
slot receives new bytes, command completion and the host fill establish the
required order. Non-contiguous legacy cache entries are copied into reusable
page-aligned anonymous mappings and then use the same Metal path.

One command buffer performs the selected experts' gate/up projections, SiTU,
down projections, and weighted reduction. If Metal initialization, shader
compilation, shape validation, or execution fails, the runtime reports the
reason and selects the native CPU MXFP4 path. The Metal and CPU implementations
have been checked on real experts with token-oracle coverage; they are not
described as byte-identical floating-point implementations.

Implementation:
[`tools/metal_moe.py`](tools/metal_moe.py),
[`tools/metal_moe.mm`](tools/metal_moe.mm), and
[`tools/metal/moe_mxfp4.metal`](tools/metal/moe_mxfp4.metal).

## Position-major Metal MoE

**Status: Opt-in and exact-token qualified.**

Speculative verification gives the MoE layer several positions at once, but
the original Metal bridge submitted and waited once per position. The
position-major path flattens those routes in their original position and
router order, resolves each unique expert span once, and encodes the positions
in one command buffer. It uses a separate scratch pool so an occasional
multi-position pass cannot permanently enlarge the ordinary one-token buffers.

The reduction order for each position remains the model's original top-k order.
Missing support for the multi-position ABI falls back to the established
one-position loop. Balanced full-model measurements on the reference M1 Max
showed **+2.0% pooled throughput** on accepted speculative passes. It remains
off by default because the crossover depends on the Mac, runtime, and accepted
batch shape.

Implementation:
[`tools/metal_moe.py`](tools/metal_moe.py),
[`tools/metal_moe.mm`](tools/metal_moe.mm), and
[`tools/test_metal_position_batch.py`](tools/test_metal_position_batch.py).

## Direct CPU views into resident-spine read buffers

**Status: Default on the synchronous CPU apply path; exact structural
evidence.**

Packed int8 and mixed-codec spine reads already land in writable pooled host
buffers. The old CPU path copied their weights, scales, and bf16 tails into a
second CPU staging allocation before immediately consuming them. CPU
dequantization is synchronous, so the original read view already has the
necessary lifetime.

The direct-view path consumes that source in place and releases it only after
apply completes. Accelerator paths retain their device staging. The
experimental packed-KDA stage controller also keeps a conservative owned
weight copy where its operator may outlive the ordinary CPU apply call.

For a fully streamed 93-layer int8 pass, the exact accounting removes
**50.71 GiB of redundant host copies**. Source poisoning, retained-buffer,
mixed-codec, device-selection, and fallback tests cover the lifetime boundary.
This is structural evidence only; no complete-token speed percentage is
attached to it.

Implementation:
[`tools/spine_fast.py`](tools/spine_fast.py),
[`tools/mixed_spine.py`](tools/mixed_spine.py), and
[`tools/test_cpu_spine_direct_views.py`](tools/test_cpu_spine_direct_views.py).

## In-place SiTU, scratch reuse, and ordered combine

**Status: Default on the portable CPU MXFP4 batch path; bit-exact structural
evidence.**

The CPU expert path produces disposable gate and up vectors, transforms them
with K3's SiTU activation, runs the down projection, and then multiplies each
expert row by its routing weight before adding it in router order. A direct
NumPy expression allocates several full-size temporaries at each step.

Deltafin keeps NumPy's established fp32 ufunc order but supplies explicit
destinations:

- the gate and up arrays are overwritten after their last original use;
- the not-yet-written prefix of the expert-output allocation serves as SiTU
  scratch when its shape permits; and
- each disposable down-projection row is scaled in place before the same
  ordered add into the output.

Real-expert and focused numerical tests are bit-exact. The known T=1
allocation accounting removes **57.5 MiB per 92-layer pass**. Timing on a quiet
host is still needed before assigning a throughput number.

Implementation:
[`tools/fast_moe_batch.py`](tools/fast_moe_batch.py) and
[`tools/test_situ_inplace.py`](tools/test_situ_inplace.py).

## Reusing the short-convolution source as its cache

**Status: Default in inference; exact structural evidence.**

KDA has three width-four causal depthwise convolutions. Both supported formulas
already create the source consisting of the last three cached values followed
by the new input. The previous implementation then concatenated cache and
input a second time solely to return the last four values as the new cache.

In inference, the first source's final four values are the same values, so the
runtime returns that tail. At T=1 the returned cache can be the source
allocation itself. With gradients enabled, the code deliberately keeps the
old independent allocation: backward may retain the first source, and a caller
is allowed to mutate the returned training cache without invalidating it.

Across 207 T=1 convolution calls, this removes **87.328 MiB of allocation per
model pass**. CPU and MPS real-width tests cover both convolution formulas,
cache identity, immutable retained snapshots, and mutation-before-backward.
That byte figure is not a tokens-per-second claim.

The related decode kernel treats a four-tap depthwise convolution as the small
operation it really is: build the sliding windows once, multiply by the four
weights, and reduce. This avoids asking a generic grouped-convolution engine to
schedule 12,288 tiny groups. It is automatic through the measured T=9 CPU
range and for the established T=1 accelerator path. `conv1d` remains
forceable, and larger accelerator batches retain the conservative crossover
until independently measured.

Implementation:
[`tools/fla/modules/__init__.py`](tools/fla/modules/__init__.py) and
[`tools/test_shortconv_cache_reuse.py`](tools/test_shortconv_cache_reuse.py).

## One routing record for every consumer

**Status: Default and exact; structural synchronization evidence.**

Expert fetching needs selected IDs on the CPU. The CPU and Metal MoE backends,
weighted reduction, trace writer, and lookahead accounting need the same IDs
and weights. Repeated `tolist()` calls are not just Python work when the router
lives on an accelerator: each conversion can introduce a device
synchronization.

The driver now materializes one ordered CPU record, reusing the ID rows it
already created for fetch scheduling and converting the fp32 route weights
once. That read-only record flows through the selected backend and optional
trace. Legacy direct callers can still materialize locally, while
`K3_FAST_MOE=0` with tracing disabled avoids weight-list work entirely.

Tie order, dtype conversion, multi-position routing, backend fallback, and
trace serialization are tested. We do not attach a throughput percentage to
this synchronization removal.

Implementation:
[`tools/routing_record.py`](tools/routing_record.py),
[`tools/kimi_run.py`](tools/kimi_run.py), and
[`tools/test_routing_record_portability.py`](tools/test_routing_record_portability.py).

## CPU value padding to unlock fused SDPA

**Status: Opt-in, default off, fingerprinted per live CPU/runtime shape.**

K3's MLA uses 192-wide query/key heads but 128-wide value heads. On the tested
CPU build, ordinary unequal-width scaled-dot-product attention decomposed into
the math implementation. Appending 64 exact-zero value channels selected
PyTorch's fused CPU FlashAttention operator; slicing those channels from the
output restores the original width.

The production candidate pads per call, so the cache remains 128-wide. This
avoids the persistent alternative's 50% larger value cache and 20% larger
combined key/value cache. Eligibility includes PyTorch version, reported CPU
capability, thread counts, dtype, batch, dimensions, T, context bucket, mask,
and layout. On the first real call for each key, the profiler must observe the
native CPU FlashAttention operator. A missing operator, unsupported shape,
allocation failure, changed mask, training mode, or runtime exception disables
that key and executes eager attention.

Across the measured T=1/2/8/9 and context 32/128/512/1024 cells, the padded
attention call had a **1.23x geometric-mean isolated attention speedup**. This
is not a full-layer or tokens-per-second result. MPS already had a native
unequal-width attention operator in the local tests, and CUDA has not inherited
the CPU result.

Implementation:
[`tools/attn_fast.py`](tools/attn_fast.py),
[`tools/test_mla_cpu_sdpa.py`](tools/test_mla_cpu_sdpa.py).

## Fused MXFP4 dequantization and GEMV

**Status: Default native CPU expert path, with architecture and runtime
dispatch.**

The routed weights store two E2M1 values per byte and one E8M0 scale for every
group of 32 columns. A conventional implementation expands a matrix to fp32
and then multiplies it, creating eight times as much weight traffic before the
GEMV even begins. Deltafin decodes each packed group inside the row loop and
accumulates directly into the output.

An immutable, 64-byte-aligned 8 KiB lookup table covers the complete 256-value
E8M0 scale domain. That replaces repeated exponent-table synthesis while
retaining the synthesis implementation as a build-time correctness oracle.

The same source has guarded implementations for:

- NEON on aarch64;
- a 128-bit x86-64 compatibility path using the required
  AVX/FMA3/SSE3/SSSE3 baseline; and
- an internal target-attributed 256-bit AVX2/FMA path selected only when the
  CPU and operating system report it usable.

The x86 library is therefore one fat binary rather than an AVX2-only artifact.
The direct implementation details are internal, not a promised public native
ABI. The build validates the supported entry points before installing a
library.

MXFP4 columns are groups of 32, and the supported matrices preserve that
contract. The tests include odd output-row counts; they do **not** establish
arbitrary column tails. Full-domain scale/nibble checks, single and
multithreaded results, awkward row counts, and batch paths are compared with
the independent reference. Native Linux AVX2 timing is still wanted, so no
Linux AVX2 speed number is borrowed from translated or other-host evidence.

Implementation:
[`tools/fused_gemv.c`](tools/fused_gemv.c),
[`tools/neon_compat_x86.h`](tools/neon_compat_x86.h),
[`tools/build_native.py`](tools/build_native.py), and
[`tools/test_fused_gemv_portability.py`](tools/test_fused_gemv_portability.py).

## Persistent CPU workers and batch-wide activation preparation

**Status: Default when the batch native library validates; legacy native GEMV
fallback otherwise.**

Running 16 experts requires 48 matrices: gate, up, and down for each expert.
The first native wrapper crossed Python and created/joined worker threads for
each matrix. The batch library keeps one worker ring alive and dispatches the
whole gate/up phase, then the whole down phase.

On the AVX2 implementation, the activation uses a lane order convenient for
the packed weight decode. The batch path prepares each distinct activation
once for the whole dispatch. In phase A, 32 matrices normally share one hidden
vector; permuting it 32 times would turn a kernel-local trick into avoidable
host work.

Workers claim fixed row units, so each row retains the same accumulation order
regardless of which worker executes it. The hot generation, cursor, and
completion counters are each aligned to 128 bytes to avoid false sharing.
That counter padding measured **+0.2% at four threads** and **+3.6% at eight
threads** in the focused worker test. Worker selection respects process
affinity and Linux cgroup CPU quotas, uses a conservative default of up to four
workers on macOS and eight on Linux, and retains explicit overrides plus hard
native bounds. This also closes the older risk of indexing fixed thread
storage with an oversized worker count.

The NumPy SiTU remains outside the native batch for the exact default because
platform `tanhf`/`expf` are not bit-identical to NumPy's established
transcendentals. If batch initialization or symbol/ABI validation fails, the
runtime uses the legacy native path.

Implementation:
[`tools/fused_gemv_batch.c`](tools/fused_gemv_batch.c),
[`tools/fast_moe_batch.py`](tools/fast_moe_batch.py), and
[`tools/runtime_platform.py`](tools/runtime_platform.py).

## Packed spine reads and Metal dequantization

**Status: Packed reads are default; the custom dequantizer is
capability-gated to MPS with a Torch fallback.**

A resident layer contains many int8 payload and scale files. Reading each into
a fresh `bytearray`, transferring it separately, dequantizing into a temporary,
and finally copying into the layer template creates unnecessary allocation,
memset, transfer, and dispatch work.

The packed path reads directly into a small pool of reusable aligned host
buffers, overlaps at most the current and next layer, and coalesces the device
staging. On MPS, a small Metal kernel performs `int8 -> fp32`, fp16 row-scale
conversion, multiplication, and the final write into the template in one pass.
Persistent staging buffers prevent a new hundreds-of-megabytes allocation for
every layer.

On the reference M1 Max, the Torch row-broadcast expression moved the tested
tensor at about 43 GB/s, while the fused Metal kernel reached about 297 GB/s;
the broader per-layer load path moved from 118 ms to 21 ms. The kernel was
bit-exact on every tested tensor and non-zero packed-buffer offset because both
paths evaluate `float(int8) * float(fp16)` in fp32. A missing MPS device,
shader-compilation error, incompatible destination, or non-fp32 mode uses the
Torch expression.

Implementation:
[`tools/spine_fast.py`](tools/spine_fast.py).

## Reused layer templates and one serial arena

**Status: Default; the shared-arena extension is capability-gated.**

Materializing and destroying a complete decoder module 93 times asks the
framework allocator to manage far more state than the computation requires.
K3 has only three relevant resident shape classes: one first-layer dense KDA
shape, one KDA+MoE shape reused by the remaining 68 KDA layers, and one MLA+MoE
shape reused by 24 MLA layers. Deltafin creates one template for each class and
copies the current layer's streamed weights into its stable parameter views.

The templates execute serially. Where the selected device can allocate the
required single buffer, their non-overlapping parameter views can therefore
share one maximum-sized arena rather than retaining the sum of three
allocations. On MPS, the runtime checks Metal's maximum buffer length first.
A missing capability query, a too-small limit, or an allocation failure keeps
the separate template allocations.

Correctness depends on not reusing a view before the previous asynchronous
consumer is ordered. The packed loader, staging lifetimes, speculative state,
and layer index are tested while alternating real shape classes. Stable shapes
and parameter identities also make optional compilation experiments possible,
but no compilation speed is claimed here.

Implementation:
[`tools/kimi_run.py`](tools/kimi_run.py) and
[`tools/apple_silicon.py`](tools/apple_silicon.py).

## Keeping KDA recurrence on the selected device

**Status: Default; the historical CPU hop remains forceable.**

An earlier MPS path copied roughly 240 KB of KDA inputs to the CPU, ran the
small recurrent update, and copied the result back. The arithmetic itself was
small, but every transfer drained the accelerator queue. The current default
runs the recurrence on the device that already owns Q/K/V: MPS, CUDA, or CPU.

In a focused MPS measurement, the recurrence took 1.17 ms on-device versus
3.12 ms with the CPU round trip. We do not transfer that result to CUDA or
another Mac. `K3_KDA_RECUR=cpu` retains the historical MPS comparison path,
and speculative verification explicitly keeps its replay on the same
arithmetic route as the original call.

Implementation:
[`tools/fla/ops/kda/__init__.py`](tools/fla/ops/kda/__init__.py) and
[`tools/attn_fast.py`](tools/attn_fast.py).

## Coalesced expert ranges and raw cache files

**Status: Default for streaming misses and local expert storage.**

All six tensors for a K3 routed expert are contiguous in its original shard.
Deltafin verified this for all 82,432 experts and records the fixed offsets.
A remote miss can therefore use one 17.55 MB HTTP range request rather than
six requests. Connections stay alive, the Hugging Face redirect is cached for
its validity window, and file-adjacent selected experts may share a request
without downloading an unselected expert-sized gap. This measured about 6.4x
faster than the original per-tensor remote fetch under the reference
conditions.

The cache file is the exact shard span, published atomically. A local hit needs
no archive parsing, decompression, or checksum traversal before compute; its
six arrays are views at known offsets. An in-memory presence set also avoids a
filesystem metadata lookup for every routed selection.

This HTTP traffic exists only for Hugging Face downloads in a streaming
installation. A full local installation running `tools/kimi_run.py` does not
use HTTP to communicate between Deltafin components.

Implementation:
[`tools/fetch_v2.py`](tools/fetch_v2.py) and
[`tools/k3loader.py`](tools/k3loader.py).

## Parallel expert reads, layer double-buffering, and router lookahead

**Status: Parallel expert reads and spine double-buffering are default;
lookahead is exact with respect to model output and fails closed.**

Handing file-backed views directly to a GEMV can make the compute thread take
page faults one at a time. Instead, a persistent reader pool fills reusable,
page-aligned slots for the selected expert union before the kernel begins.
Slots are bounded and recycled only after their consumer finishes.

At the resident-spine level, one loader thread reads layer `N+1` while the
selected device computes layer `N`. The driver publishes layers in model order,
and any read exception is surfaced before a partially loaded layer can run.

Expert selection is depth-serial: the real router for layer `N+1` normally
cannot run until layer `N` finishes. The lookahead path uses layer `N`'s
pre-MoE hidden state with a cached copy of the next router to predict the next
expert set and begin those reads. Packed int8 router weights are used only when
the live native operator qualifies; otherwise the predictor can retain a dense
representation.

The prediction never decides the computation. The real next-layer router
remains authoritative, a correct prediction turns a demand read into a wait on
already-running work, and a wrong prediction merely wastes I/O. Failed
predictor initialization disables lookahead. Previous-token selection remains
a simpler fallback experiment, but it is not treated as a correctness source.

Implementation:
[`tools/fetch_v2.py`](tools/fetch_v2.py),
[`tools/pilot.py`](tools/pilot.py), and
[`tools/kimi_run.py`](tools/kimi_run.py).

## Lossless n-gram speculative decoding

**Status: Default, exact-token gated.**

Deltafin searches the generated token history for the longest matching suffix
and uses the following historical token as a free draft. A two-position
forward pass verifies that draft with K3 itself. If it matches, the pass emits
the draft and the next verified token. If it does not, the reference-only
snapshot restores the pre-pass state and the runtime performs the ordinary
one-token step.

This is especially useful in Deltafin because one forward pass rereads the
resident spine once. A verified T=2 pass can share that fixed work across two
emitted tokens. It is less magical for routed experts: the pass must still read
the union of experts selected by both positions.

Every emitted token is chosen by K3's greedy verifier. Rollback tests poison
drafts deliberately, restore each cache family, and compare future logits and
tokens. Deeper drafts and recurrence-input replay are available separately;
their batch/union crossover is not assumed from the default T=2 path.

Implementation:
[`tools/kimi_run.py`](tools/kimi_run.py),
[`tools/spec_decode.py`](tools/spec_decode.py), and
[`tools/test_spec_replay.py`](tools/test_spec_replay.py).

## Runtime and memory auto-selection

**Status: Default, conservative.**

Cross-platform speed depends on selecting what the host can actually execute,
not matching a product name:

- MPS is preferred when available, then the first CUDA device visible to the
  process, then CPU. An explicit `cuda:N`, `mps`, or `cpu` remains
  authoritative.
- Metal MoE is selected only with an MPS resident device and a working Metal
  library. CUDA currently accelerates the resident spine and attention while
  routed MXFP4 experts retain the native CPU path.
- aarch64 selects NEON; x86-64 checks the required compatibility features and
  selects AVX2 only at runtime.
- worker counts respect CPU affinity and Linux cgroup quotas.
- RAM budgets respect host memory and Linux cgroup limits. Optional CUDA
  pinning also reserves accelerator headroom instead of treating total VRAM as
  free memory.
- the int8 spine is selected when present; optional operators still pass their
  own dispatcher, shape, dtype, and first-call gates.

Unknown memory limits and failed capability probes are handled conservatively.
This is what keeps a path measured on an older M1 eligible for a newer Mac
without assuming that every M5, CUDA device, container, or PyTorch build has
the same crossover.

Implementation:
[`tools/runtime_platform.py`](tools/runtime_platform.py),
[`tools/apple_silicon.py`](tools/apple_silicon.py), and
[`tools/kimi_run.py`](tools/kimi_run.py).

## Small fixed-cost cleanups

**Status: Default unless noted.**

Several smaller changes are worth retaining even when they do not justify a
headline:

- Generation runs under `torch.inference_mode()`. Cyclic garbage collection is
  temporarily disabled for the generation and restored in `finally`, avoiding
  periodic object-graph scans while still cleaning up normally afterward.
- Router tracing is off in performance runs. Buffered tracing writes one block
  after a model pass instead of synchronously flushing one JSON record per
  layer.
- Persistent objects are reused for the final tail, packed read pools, device
  staging, pointer/shape arrays, and native workers.
- Cache presence is indexed once and updated at the atomic publication
  boundary rather than rescanning or repeatedly `stat`-ing the 82,432-file
  expert pool.

These changes remove host overhead and allocation noise; they do not carry
separate tokens-per-second claims.

Implementation:
[`tools/kimi_run.py`](tools/kimi_run.py),
[`tools/fetch_v2.py`](tools/fetch_v2.py), and
[`tools/fast_moe_batch.py`](tools/fast_moe_batch.py).

## Optional cache-write overlap

**Status: Opt-in, default off.**

In a streaming installation, a downloaded expert must eventually be written
to its local cache. `K3_ASYNC_CACHE_WRITE=1` lets inference keep using the
immutable downloaded bytes while a bounded background writer publishes the
cache entry.

The bound covers queued *and active* payloads, so a slow disk cannot retain an
unlimited amount of model data. Mutable inputs are snapshotted before enqueue;
immutable `bytes` can be retained directly. Publication writes a temporary
file beside the destination and atomically replaces the final name. Shutdown
stops admission, drains accepted work, joins workers, and exposes any failure.
When the queue is full, producers apply backpressure rather than silently
dropping cache data.

This can affect the cache-miss path only. It does not speed a complete local
installation, and it remains off until the retained-buffer and write-contention
tradeoff is measured on the target storage.

Implementation:
[`tools/cache_writer.py`](tools/cache_writer.py),
[`tools/fetch_v2.py`](tools/fetch_v2.py), and
[`tools/test_async_cache_write.py`](tools/test_async_cache_write.py).

## Exact memoization for repeated API requests

**Status: Default in the optional API server, bounded and disableable.**

The OpenAI-compatible server is greedy and serializes access to one immutable
model instance. The exact tuple of request mode, prompt token IDs, and output
limit therefore identifies a deterministic response. A small LRU can return
the stored token IDs, text, and finish reason for that *identical* request.
Similar prompts, changed limits, and different modes are misses; no prefix is
guessed.

This is not a normal tokens-per-second improvement. A hit bypasses inference
because the same deterministic request was already completed in that server
process. Set `K3_RESPONSE_MEMO_ENTRIES=0` to disable it.

The server itself is optional. Normal CLI inference is in-process and does not
send HTTP requests to another Deltafin component.

Implementation:
[`tools/response_memo.py`](tools/response_memo.py),
[`tools/serve_openai.py`](tools/serve_openai.py), and
[`tools/test_response_memo.py`](tools/test_response_memo.py).

## What the evidence does and does not say

Deltafin uses three different acceptance bars:

1. **Exact paths** must preserve bytes, operation order where required, cache
   lifetime, rollback, and emitted tokens.
2. **Numerical paths** must stay inside a stated error gate and preserve the
   sequence oracle; they are never casually described as bit-exact.
3. **Quantized quality paths** need explicit logit/token or task-quality
   evidence and a clear fallback.

An allocation reduction is not automatically a speedup, a fast isolated
kernel is not automatically a faster token, and a result on MPS does not prove
CUDA or CPU behavior. That is why the default-off KDA staging path, CPU padded
SDPA, and position-major Metal path are documented here without being silently
enabled everywhere.

The public end-to-end baseline and its measurement method remain in
[`README.md`](README.md). This file explains the mechanisms behind it; it does
not combine isolated wins into a synthetic throughput number.
