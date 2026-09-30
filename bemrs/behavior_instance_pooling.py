"""Shared cross-instance aggregation for ACO behavior descriptors."""

from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np


INSTANCE_POOLING_ENV = "BEMRS_BEHAVIOR_INSTANCE_POOLING"
_POOLING_ALIASES = {
    "concat": "concatenate",
    "concatenate": "concatenate",
    "mean": "mean",
}


def resolve_instance_pooling(value: Optional[str] = None) -> str:
    """Return the canonical instance-pooling mode.

    Concatenation remains the default so existing methods and historical
    experiments keep their original descriptor dimensions.
    """

    raw = value if value is not None else os.environ.get(INSTANCE_POOLING_ENV, "concatenate")
    mode = str(raw).strip().lower()
    try:
        return _POOLING_ALIASES[mode]
    except KeyError as error:
        choices = ", ".join(sorted(_POOLING_ALIASES))
        raise ValueError(
            f"Unsupported behavior instance pooling {raw!r}; expected one of {choices}."
        ) from error


def instance_pooled_output_dim(
    n_matrices: int,
    features_per_instance: int,
    pooling: str,
) -> int:
    mode = resolve_instance_pooling(pooling)
    if mode == "mean":
        return int(features_per_instance)
    return int(n_matrices) * int(features_per_instance)


def aggregate_instance_features(
    parts: Iterable[np.ndarray],
    pooling: str,
) -> np.ndarray:
    """Aggregate aligned per-instance feature blocks as float32.

    Mean pooling is performed in float64 before the final float32 cast. This
    matches the offline geometry ablations and avoids avoidable rounding drift
    in the strict raw-behavior filter.
    """

    blocks = [np.asarray(part, dtype=np.float64).reshape(-1) for part in parts]
    if not blocks:
        raise ValueError("At least one instance feature block is required.")
    feature_dim = int(blocks[0].size)
    if feature_dim <= 0 or any(block.size != feature_dim for block in blocks):
        raise ValueError("Instance feature blocks must be non-empty and aligned.")

    mode = resolve_instance_pooling(pooling)
    if mode == "mean":
        return np.mean(np.stack(blocks, axis=0), axis=0).astype(np.float32)
    return np.concatenate(blocks).astype(np.float32)
