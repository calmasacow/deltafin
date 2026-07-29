#!/usr/bin/env python3
"""N16 falsifier: can speculative snapshots retain tensor references?

K3's cache update sites appear functional: recurrence, short convolution, and
MLA append return new tensors rather than mutating the pre-pass state.  If that
contract holds, a snapshot can retain old object references in O(1) time and
restore them by list assignment instead of cloning roughly 0.5 GB.

This test exercises the real KDA and short-convolution implementations at K3
shapes, verifies old storage is unmodified and not aliased by the result, then
benchmarks reference versus clone snapshots over 69 logical KDA layers.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fla.modules import ShortConvolution
from fla.ops.kda import chunk_kda

B, HEADS, K, V, WIDTH = 1, 96, 128, 128, 4
DPROJ = HEADS * K


def ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def sync(dev: str) -> None:
    if dev == "mps":
        torch.mps.synchronize()


def median_ms(fn, dev: str, reps: int) -> float:
    values = []
    for _ in range(reps):
        sync(dev)
        before = time.perf_counter_ns()
        value = fn()
        sync(dev)
        values.append((time.perf_counter_ns() - before) / 1e6)
        del value
    return statistics.median(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"),
                        default="mps" if torch.backends.mps.is_available()
                        else "cpu")
    parser.add_argument("--layers", type=int, default=69)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    dev = args.device
    if dev == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable; run outside the filesystem sandbox")

    torch.manual_seed(20260728)
    with torch.inference_mode():
        initial = torch.randn(B, HEADS, K, V, device=dev)
        initial_bits = initial.clone()
        kwargs = dict(
            q=torch.randn(B, 2, HEADS, K, device=dev),
            k=torch.randn(B, 2, HEADS, K, device=dev),
            v=torch.randn(B, 2, HEADS, V, device=dev),
            g=torch.randn(B, 2, HEADS, K, device=dev),
            beta=torch.randn(B, 2, HEADS, device=dev),
            A_log=torch.randn(K, device=dev),
            dt_bias=torch.randn(HEADS * K, device=dev),
            initial_state=initial,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            transpose_state_layout=True,
        )
        _, advanced = chunk_kda(**kwargs)
        sync(dev)
        recurrence_functional = (
            ptr(advanced) != ptr(initial)
            and torch.equal(initial, initial_bits)
        )

        conv = ShortConvolution(DPROJ, WIDTH, activation="silu").to(dev)
        conv_pre = torch.randn(B, DPROJ, WIDTH, device=dev)
        conv_bits = conv_pre.clone()
        conv_x = torch.randn(B, 2, DPROJ, device=dev)
        _, conv_next = conv(
            x=conv_x, cache=conv_pre, output_final_state=True)
        sync(dev)
        conv_functional = (
            ptr(conv_next) != ptr(conv_pre)
            and torch.equal(conv_pre, conv_bits)
        )

        key_pre = torch.randn(1, 8, 17, 128, device=dev)
        key_bits = key_pre.clone()
        key_next = torch.cat(
            [key_pre, torch.randn(1, 8, 2, 128, device=dev)], dim=2)
        sync(dev)
        mla_functional = (
            ptr(key_next) != ptr(key_pre)
            and torch.equal(key_pre, key_bits)
        )

        cache = SimpleNamespace(
            recurrent_states=[initial] * args.layers,
            conv_states=[(conv_pre, conv_pre, conv_pre)] * args.layers,
            key_cache=[key_pre] * args.layers,
            value_cache=[key_pre] * args.layers,
        )

        def refs():
            return {
                "rec": tuple(cache.recurrent_states),
                "conv": tuple(cache.conv_states),
                "key": tuple(cache.key_cache),
                "value": tuple(cache.value_cache),
            }

        def clones():
            return {
                "rec": tuple(t.clone() for t in cache.recurrent_states),
                "conv": tuple(
                    tuple(t.clone() for t in group)
                    for group in cache.conv_states),
                # Current production only records MLA lengths, not clones.
                "mla_len": tuple(t.shape[2] for t in cache.key_cache),
            }

        ref_ms = median_ms(refs, dev, max(args.reps * 20, 20))
        clone_ms = median_ms(clones, dev, args.reps)
        logical_bytes = args.layers * (
            initial.numel() * initial.element_size()
            + 3 * conv_pre.numel() * conv_pre.element_size())

    print(f"[{dev}] recurrence functional={recurrence_functional} "
          f"old_ptr={ptr(initial)} new_ptr={ptr(advanced)}")
    print(f"[{dev}] shortconv functional={conv_functional} "
          f"old_ptr={ptr(conv_pre)} new_ptr={ptr(conv_next)}")
    print(f"[{dev}] MLA cat functional={mla_functional} "
          f"old_ptr={ptr(key_pre)} new_ptr={ptr(key_next)}")
    print(f"[{dev}] {args.layers}-layer snapshot: refs={ref_ms:.6f} ms "
          f"clones={clone_ms:.3f} ms speedup={clone_ms/ref_ms:.0f}x "
          f"clone_bytes={logical_bytes/1e6:.1f} MB")
    passed = recurrence_functional and conv_functional and mla_functional
    print("N16 REFERENCE SNAPSHOT:", "SURVIVES" if passed else "KILL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
