# Deltafin direct system `curl-sys`

This directory is a deliberately small local fork of the published
[`curl-sys` 0.4.90+curl-8.21.0 crate](https://crates.io/crates/curl-sys/0.4.90+curl-8.21.0)
from the [curl-rust repository](https://github.com/alexcrichton/curl-rust).
The public Rust bindings in `lib.rs` and the MIT `LICENSE` are preserved
byte-for-byte from the published crate. Alex Crichton and the curl-rust
contributors remain the authors of that code.

Deltafin changes only package metadata and the build path. Upstream's build
script can execute `pkg-config`, `curl-config`, Git, a C compiler, vcpkg, and
other helpers, or compile its bundled curl tree. This fork executes no child
processes. On Apple arm64 it asks rustc to link the SDK's `libcurl`. On glibc
Linux x86-64 and arm64 it searches a fixed list of standard library directories,
canonicalizes the candidate, and accepts it only when it is a regular,
root-owned, non-group/world-writable, bounded little-endian ELF64 shared object
for the target machine. The pure-Rust validator parses the complete ELF header,
program headers, `PT_DYNAMIC`, SysV/GNU dynamic hash metadata, string table, and
dynamic symbols. It requires the exact `libcurl.so.4` SONAME and the core easy,
global, list, and multi exports used by curl 0.4.50. Malformed, ambiguous,
overlapping, or out-of-file table mappings fail closed. A private linker-name
symlink in Cargo's `OUT_DIR` lets a runtime-only `libcurl.so.4` installation work
without searching arbitrary paths; a validated library upgrade replaces a
stale private symlink atomically.

ELF metadata cannot prove the implementation's runtime version, selected TLS
backend, or enabled protocol set. Deltafin therefore checks libcurl's reported
version, TLS feature, and HTTPS protocol capability in its native runtime before
performing a download. The fork's host smoke test exercises the same capability
surface, but that test is evidence for the build host rather than a substitute
for the runtime preflight.

The feature names forwarded by curl 0.4.50 remain declared for dependency
resolution. `ssl` is accepted as a system-TLS marker, and
`force-system-lib-on-osx` is accepted for compatibility. Features which ask
the build to configure a backend/protocol, promise a newer ABI, or build static
code fail closed; this fork cannot truthfully satisfy those requests without
probing or compiling foreign code.

The bounded policy intentionally does not support Windows, musl, Nix-style
nonstandard library stores, custom `LIBRARY_PATH`, Homebrew libcurl, cross-SDK
sysroots, cross-compilation (`HOST` must equal `TARGET`), or non-arm64 macOS.
Those require a separately designed trusted artifact or platform linker policy
rather than widening discovery implicitly.
