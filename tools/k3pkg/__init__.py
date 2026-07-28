"""Runtime home for Moonshot AI's Kimi K3 modeling code.

These modules are NOT vendored in this repository -- they are Moonshot's code,
distributed from https://huggingface.co/moonshotai/Kimi-K3 under Moonshot's
license, and are downloaded into this package at setup time.
"""
import os

_here = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(_here, "modeling_kimi_linear.py")):
    raise ImportError(
        "k3pkg is empty: the Kimi K3 modeling files are fetched at setup, not "
        "vendored in this repo.\n"
        "Run:  python tools/setup_k3.py\n"
        "(downloads modeling/configuration/tokenizer files from "
        "huggingface.co/moonshotai/Kimi-K3 and builds the tensor index)"
    )
