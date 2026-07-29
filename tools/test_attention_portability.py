#!/usr/bin/env python3
"""Weight-free CPU gates for attention backend-selection semantics."""

import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import attn_fast  # noqa: E402
import fla.modules as fm  # noqa: E402
import fla.ops.kda as fk  # noqa: E402


def _selection_checks():
    assert fm.resolve_shortconv_mode("mps", "auto") == "mulsum"
    assert fm.resolve_shortconv_mode("cpu", "auto") == "mulsum"
    assert fm.resolve_shortconv_mode("cuda", "auto") == "mulsum"
    assert fm.resolve_shortconv_mode("cpu", "mulsum") == "mulsum"
    assert fm.resolve_shortconv_mode("mps", "conv1d") == "conv1d"
    assert fm.shortconv_auto_max_t("cpu") == 9
    assert fm.shortconv_auto_max_t("mps") == 1
    assert fm.shortconv_auto_max_t("cuda") == 1
    try:
        fm.resolve_shortconv_mode("cpu", "unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid K3_SHORTCONV mode was accepted")

    assert fk.normalize_recur_mode("device") == "device"
    assert fk.normalize_recur_mode("mps") == "device"
    assert fk.normalize_recur_mode("cpu") == "cpu"
    try:
        fk.normalize_recur_mode("cuda")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid K3_KDA_RECUR mode was accepted")

    old_recur, old_shortconv = attn_fast.RECUR, attn_fast.SHORTCONV
    try:
        attn_fast.RECUR = "device"
        attn_fast.SHORTCONV = "auto"
        description = attn_fast.describe(torch.device("cuda"))
        assert "recur=device(cuda)" in description
        assert "shortconv=auto->mulsum(cuda)" in description
    finally:
        attn_fast.RECUR = old_recur
        attn_fast.SHORTCONV = old_shortconv


def _cpu_shortconv_parity():
    torch.manual_seed(17)
    convolution = fm.ShortConvolution(
        hidden_size=7,
        kernel_size=4,
        bias=False,
        activation="silu",
    ).eval()
    x = torch.randn(2, 1, 7)
    cache = torch.randn(2, 7, 4)
    original_cache = cache.clone()

    old_mode = fm.SHORTCONV
    try:
        fm.SHORTCONV = "conv1d"
        reference, reference_state = convolution(
            x,
            cache=cache,
            output_final_state=True,
        )

        # CPU auto takes the measured multiply/sum path.
        fm.SHORTCONV = "auto"
        automatic, automatic_state = convolution(
            x,
            cache=cache,
            output_final_state=True,
        )
        fm.SHORTCONV = "mulsum"
        forced, forced_state = convolution(
            x,
            cache=cache,
            output_final_state=True,
        )
        torch.testing.assert_close(automatic, forced, rtol=0, atol=0)
        torch.testing.assert_close(automatic_state, forced_state, rtol=0, atol=0)

        # The force override remains usable for CPU experiments, and its cache
        # transition must be exact even if accumulation differs by a few ulps.
        torch.testing.assert_close(forced, reference, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(
            forced_state, reference_state, rtol=0, atol=0
        )
        torch.testing.assert_close(cache, original_cache, rtol=0, atol=0)
    finally:
        fm.SHORTCONV = old_mode


def _cpu_batched_shortconv_matches_sequential_decode():
    """Speculative T>1 must use the exact ordinary-decode arithmetic path."""
    torch.manual_seed(23)
    convolution = fm.ShortConvolution(
        hidden_size=11,
        kernel_size=4,
        bias=False,
        activation="silu",
    ).eval()
    cache = torch.randn(2, 11, 4)
    original_cache = cache.clone()
    old_mode = fm.SHORTCONV
    try:
        # Deep speculation verifies pending+D positions, so the maximum
        # K3_SPEC_DEPTH=8 boundary is T=9 rather than T=8.
        for tokens in (2, 3, 4, 5, 7, 8, 9):
            x = torch.randn(2, tokens, 11)
            residual = torch.randn_like(x)
            fm.SHORTCONV = "mulsum"
            batched, batched_state = convolution(
                x,
                residual=residual,
                cache=cache,
                output_final_state=True,
            )

            state = cache
            pieces = []
            for index in range(tokens):
                piece, state = convolution(
                    x[:, index:index + 1],
                    residual=residual[:, index:index + 1],
                    cache=state,
                    output_final_state=True,
                )
                pieces.append(piece)
            sequential = torch.cat(pieces, dim=1)
            # Backends can choose a slightly different reduction kernel for a
            # different outer T shape. State is exact; output stays inside the
            # explicit numerical gate and full speculation must additionally
            # preserve emitted token IDs.
            torch.testing.assert_close(
                batched, sequential, rtol=2e-6, atol=2e-6,
                msg=f"T={tokens}",
            )
            torch.testing.assert_close(
                batched_state, state, rtol=0, atol=0, msg=f"T={tokens} state"
            )

            # The explicit compatibility arm remains numerically close even
            # though its grouped-convolution reduction order can differ.
            fm.SHORTCONV = "conv1d"
            reference, reference_state = convolution(
                x,
                residual=residual,
                cache=cache,
                output_final_state=True,
            )
            torch.testing.assert_close(
                batched, reference, rtol=2e-6, atol=2e-6
            )
            torch.testing.assert_close(
                batched_state, reference_state, rtol=0, atol=0
            )
            fm.SHORTCONV = "mulsum"

        # Auto is deliberately bounded: a large cached chunk should retain
        # conv1d rather than materializing the four-wide product tensor.
        tokens = fm.shortconv_auto_max_t("cpu") + 1
        x = torch.randn(2, tokens, 11)
        fm.SHORTCONV = "auto"
        automatic = convolution(
            x, cache=cache, output_final_state=True
        )
        fm.SHORTCONV = "conv1d"
        reference = convolution(
            x, cache=cache, output_final_state=True
        )
        torch.testing.assert_close(
            automatic[0], reference[0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            automatic[1], reference[1], rtol=0, atol=0
        )
        torch.testing.assert_close(cache, original_cache, rtol=0, atol=0)
    finally:
        fm.SHORTCONV = old_mode


def main():
    _selection_checks()
    _cpu_shortconv_parity()
    _cpu_batched_shortconv_matches_sequential_decode()
    print("PASS attention portability (CPU, no model weights)")


if __name__ == "__main__":
    main()
