"""Opt-in grouped expert-read/Metal orchestration for one-token decode.

``K3_MOE_GROUP_SIZE=0`` (default) preserves the established all-read-then-one-
dispatch path.  A positive value smaller than top-k enables N14 only when all
of these capability gates hold:

* Metal is the selected MoE backend;
* expert reads use fetch_v2's local pread path;
* exactly one token with unique routed experts is being evaluated; and
* every selected expert has a validated raw .bin span.

Every other case returns ``None`` before grouped work starts, so prefill,
speculative multi-token passes, CPU MoE, HTTP/.npz fallback, and future Apple
chips retain the existing path.  Group order is deterministic: descending
router weight, then original router slot for ties.  No expert, weight, or route
is changed.
"""

from __future__ import annotations

import os


def configured_group_size(env=None) -> int:
    env = os.environ if env is None else env
    raw = str(env.get("K3_MOE_GROUP_SIZE", "0")).strip().lower()
    if raw in ("", "0", "off", "false", "none"):
        return 0
    try:
        size = int(raw)
    except ValueError as exc:
        raise ValueError(
            "K3_MOE_GROUP_SIZE must be 0/off or a positive integer") from exc
    if size <= 0:
        raise ValueError(
            "K3_MOE_GROUP_SIZE must be 0/off or a positive integer")
    return size


GROUP_SIZE = configured_group_size()
STATS = {
    "calls": 0,
    "fallback_shape": 0,
    "fallback_group": 0,
    "fallback_storage": 0,
    "fallback_read": 0,
    "groups": 0,
    "experts": 0,
}


def enabled() -> bool:
    return GROUP_SIZE > 0


def describe() -> str:
    return f"group_size={GROUP_SIZE} decode_only=1 priority=router_weight"


def try_infer(x, routing_record, layer, fetch_v2, metal_moe):
    """Return ``(output, timing)`` or ``None`` for the unchanged runtime path."""
    if GROUP_SIZE <= 0:
        return None
    ids_rows = routing_record["ids"]
    weight_rows = routing_record["weights"]
    if (len(ids_rows) != 1 or len(weight_rows) != 1
            or getattr(x, "shape", (0,))[0] != 1):
        STATS["fallback_shape"] += 1
        return None
    ids = [int(e) for e in ids_rows[0]]
    weights = [float(w) for w in weight_rows[0]]
    if (len(ids) != len(weights) or len(set(ids)) != len(ids)
            or GROUP_SIZE >= len(ids)):
        STATS["fallback_group"] += 1
        return None

    # Stable ties are explicit instead of depending on torch.topk(sorted=False)
    # or dict order. The same ordered lists drive both reads and accumulation.
    order = sorted(range(len(ids)), key=lambda i: (-weights[i], i))
    ordered_ids = [ids[i] for i in order]
    ordered_weights = [weights[i] for i in order]
    raw_groups = fetch_v2.begin_layer_groups(
        layer, ordered_ids, GROUP_SIZE)
    if raw_groups is None:
        STATS["fallback_storage"] += 1
        return None

    def groups():
        try:
            for group_ids, raw in raw_groups:
                start = groups.position
                stop = start + len(group_ids)
                expected = ordered_ids[start:stop]
                if list(group_ids) != expected:
                    raise RuntimeError(
                        f"grouped read reordered experts {group_ids} != {expected}")
                groups.position = stop
                STATS["groups"] += 1
                STATS["experts"] += len(group_ids)
                yield group_ids, ordered_weights[start:stop], raw
        finally:
            close = getattr(raw_groups, "close", None)
            if close is not None:
                close()

    groups.position = 0
    timing = {}
    try:
        out = metal_moe.moe_infer_grouped(x, groups(), timing=timing)
    except fetch_v2.GroupReadError:
        # Completed Metal partials are local to moe_infer_grouped and discarded.
        # fetch_v2 has already settled/retired every slot and invalidated the
        # failing .bin, so the caller can safely take npz/HTTP recovery.
        STATS["fallback_read"] += 1
        return None
    STATS["calls"] += 1
    return out, timing
