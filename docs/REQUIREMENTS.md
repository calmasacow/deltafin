# Requirements

Deltafin builds from source and installs its own native runtime. Most of what
follows is handled for you — the manual steps are called out below.

## Host

- **macOS 14+ on Apple Silicon**, or **Linux with glibc 2.28+** on x86-64 or
  aarch64. These floors come from the pinned upstream artifacts the bootstrap
  uses (`macosx_14_0`, `manylinux_2_28`) — not from untested older systems.
- **Rust 1.85+ and Cargo.**
- **A native C/C++ compiler and archiver** — Clang on macOS, Clang or GCC on
  Linux.
- **Full Xcode** on macOS, or an ordinary GCC/Clang toolchain on Linux.
- **Disk space** for your chosen install mode: roughly **1.7 TB** for a full
  local expert corpus, or **215 GB** for streaming.

## Installed automatically

- **PyTorch 2.13.0 C++ runtime.** The macOS/CPU build downloads and extracts an
  exact, SHA-256-pinned native artifact and links its headers and libraries
  directly. No interpreter frontend, and `libtorch_python` is never admitted.
- **Metal shaders.** Cargo compiles both reviewed shaders into embedded
  metallibs at build time; inference loads those bytes and never compiles Metal
  while serving a token.

## Manual: macOS Metal toolchain

Current Xcode releases ship the offline Metal compiler as a separate download.
Command Line Tools alone are not enough. Install it once:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Manual: NVIDIA/CUDA builds

Deltafin does **not** install CUDA for you, and its pinned CPU runtime is not
CUDA-capable. You supply:

- A complete, mutually compatible **CUDA-enabled LibTorch tree**.
- A **matching CUDA toolkit with NVCC**.

Then build explicitly:

```bash
DELTAFIN_TORCH_ROOT=/absolute/path/to/libtorch \
DELTAFIN_CUDA_MOE=ON \
cargo build --locked --release
```

- `LIBTORCH` works in place of `DELTAFIN_TORCH_ROOT`.
- `DELTAFIN_CUDA_ARCHITECTURES` is only for a deliberate cross-build; otherwise
  the build bootstrap and NVCC derive the architecture set themselves.

## Manual: AMD/ROCm builds

A ROCm LibTorch tree needs no extra build flag. Point `DELTAFIN_TORCH_ROOT` at
it and build normally: the runtime identifies the HIP dependency, links the
same `torch_cuda`/`c10_cuda` pair, and reports AMD hardware as CUDA devices,
which is how ROCm's PyTorch presents itself.

Leave `DELTAFIN_CUDA_MOE` unset. Routed experts run the exact CPU MXFP4 path,
because the device MXFP4 and original-BF16 kernels assume a 32-lane warp and
would misreduce silently on 64-lane wavefronts; `ON` is rejected against a
ROCm root for that reason. `deltafin doctor` names the detected runtime.

## Distribution

Deltafin ships as one complete Git source tree, not as crates.io packages. Its
internal crates are publish-disabled because the reviewed native source graph
deliberately spans several repository directories.
