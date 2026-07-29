"""Shared CPU routing records for fetch, trace, and MoE backends.

The driver already materializes routed expert IDs on the CPU to schedule
expert reads.  Fast backends also need Python rows for deterministic expert
selection and weighted accumulation.  Keeping one record avoids asking an
accelerator tensor to synchronize and convert a second time.

Records intentionally remain ordinary nested lists for compatibility with the
existing Metal bridge and trace serializer.  Consumers must treat them as
read-only.
"""

from __future__ import annotations

import torch


def materialize(topk_ids, topk_weight, *, ids=None):
    """Return one ordered ``{"ids", "weights"}`` CPU routing record.

    ``ids`` may be the exact list already produced for fetch scheduling.  When
    omitted, this function retains legacy standalone behavior and materializes
    both tensors itself.  Weights always follow the established float32
    conversion before ``tolist()``.
    """
    ids_rows = topk_ids.tolist() if ids is None else ids
    return {
        "ids": ids_rows,
        "weights": topk_weight.to(torch.float32).tolist(),
    }


def resolve(topk_ids, topk_weight, routing_record=None):
    """Return ordered ID/weight rows, materializing only for legacy callers."""
    record = (
        materialize(topk_ids, topk_weight)
        if routing_record is None
        else routing_record
    )
    return record["ids"], record["weights"]
