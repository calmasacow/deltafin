# Native provider gate

This directory supplies both the reusable, versioned C ABI linked into the
Rust `deltafin` executable and its standalone no-Python development gate. They
link directly to the installed LibTorch/ATen libraries and execute analytical
canaries for the provider operations used by K3:

- the ordered fp32 RMS-style operation sequence;
- fp32 matrix multiplication;
- fp32 softmax; and
- `aten::_weight_int8pack_mm` with Deltafin's production fp32 activation and
  scale contract, when the selected provider implements the private operator.

It is intentionally not a Python extension: there is no pybind module, no
`libtorch_python`, and no embedded interpreter. One Rust-owned declarative
graph supplies the reviewed sources, flags, definitions and link libraries for
both production and development tests. It invokes compilers and archivers
directly; there is no CMake or shell harness and no evaluation of LibTorch
package scripts.

The graph compiles a tiny native denial guard and places it under the standard
interpreter names for every compiler and archiver invocation. Each test then
runs as a separate child with a minimal environment, sanitized loader
variables, a deadline and bounded stdout/stderr. Before execution, the xtask
recursively audits that executable's Mach-O or ELF dependencies and refuses
`libtorch_python` or `libpython` at any depth. The production archive is built
once per xtask invocation and reused; only the selected test main and an
explicit test-flavor override are compiled separately. Deliberately hostile
external toolchain overrides remain outside this guard's trust boundary and
are never auto-selected.

From the repository root:

```sh
cargo run --locked --package deltafin-xtask -- native-test all
cargo run --locked --package deltafin-xtask -- list
cargo run --locked --package deltafin-xtask -- native-test provider-schedule-oracle
cargo run --locked --package deltafin-xtask -- native-test gate
cargo run --locked --package deltafin-xtask -- native-test dspark
cargo run --locked --package deltafin-xtask -- native-test cuda-moe
```

`native-test provider-schedule-oracle` is an implementation-independent
call-order/shape oracle for the established public KDA, MLA, residual, dense,
MoE and tail schedules at every T=1..9 on CPU, MPS and CUDA. It intentionally
does not link the provider under test. Its executable can dump a six-column TSV
schedule or compare a provider trace against it, so a migration cannot certify
itself merely by calling the same helper from both sides.

`native-test gate` runs the applicable CPU/MPS capability cases, including the
real 12288×7168 packed-int8 shape, split ownership boundary, spine binding and
KDA tape. `native-test all` covers every declaratively registered independent
provider executable.

By default the harness installs or fully revalidates Deltafin's exact pinned
CPU/MPS toolchain under `.deltafin/toolchains`; it never searches a Python
environment. Set `DELTAFIN_TORCH_ROOT` or `LIBTORCH` only to opt into a trusted
external native PyTorch root. `DELTAFIN_CUDA_MOE=ON` requires that explicit
root until the separate CUDA wheel dependency closure has exact audited pins.

The standalone programs are isolated test artifacts, not second production
applications. Cargo compiles the provider sources into a static archive and
links that archive into the one Rust executable. `deltafin doctor` exercises
the same ABI. Its packed-int8 result qualifies the private operator for the
already-quantized matrix it was given; it does **not** claim that row-int8
weights reproduce the original BF16 checkpoint.

Production macOS builds contain only build-time-compiled metallibs. Runtime
Metal source compilation exists solely in the explicit `metal-source`
development flavor, which tests the embedded source and an explicitly supplied
source path in separate processes. The CUDA residency flavor additionally
contains a one-expert physical canary: first-plan miss, pinned upload/admission,
second-plan hit with zero misses, exact device-to-host byte comparison and
cancel. A CPU-only host validates the fail-closed stub; the full residency arm
requires real Linux/CUDA hardware.

`--spine-binding` exercises the single-call, descriptor-driven resident-spine
boundary with tiny synthetic buffers. The provider validates the whole layer
before inspecting payload bytes, promotes original raw BF16/F32 values into
selected-device FP32 tensors, and preserves explicitly selected row-int8
weights plus provider-owned FP32 row scales. BF16-to-FP32 promotion preserves
each stored checkpoint value exactly; row-int8 remains a separately labelled,
non-weight-exact representation. Mutating the caller buffers after the call is
part of the canary, so a pass also proves that the provider retained no caller
pointer. Replacement is generation-ordered and transactional.

`--kda-tape` qualifies the provider-owned, one-position KDA attention tape
against a separately expressed deterministic fp32 reference. The ABI consumes
the already-bound KDA spine slots and runs all eight projections, three causal
width-four convolutions, the gated delta recurrence, gated RMS normalization,
and the output projection behind one coarse call. Original-BF16 projections
use dense FP32 values and retain the established operation order on every
device. Only the optional row-int8 MPS arm may use the separately qualified
projection bundle. Its three convolution histories and recurrent matrix remain
provider-owned. Decode stages new state in a ticket; an explicit commit
publishes it, while ticket release cancels it. Production entry rejects non-K3
shapes and stale layer generations.

`--split-boundary` additionally qualifies the migration's provider-owned state
contract. Rust receives only opaque integer IDs and a fixed top-16 route
mailbox. `prepare_layer` stages cache state and returns a ticket; it does not
publish that cache mutation. `finish_layer` consumes the ticket and commits the
cache exactly once after the expert result arrives. The complete 93-layer
KDA/MLA/dense/MoE/tail tape is now linked behind the same ABI, but remains a
migration path until original-BF16 real-weight sequence parity and performance
gates pass. Synthetic canaries validate ownership and operation contracts; they
are not evidence by themselves that the complete runtime is production-ready.
