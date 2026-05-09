"""Linear regression weight fitting for heuristic eval features."""

from __future__ import annotations

import numpy as np


def fit_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return least-squares weights `w` such that `X @ w ≈ y`.

    Uses `np.linalg.lstsq` so collinear / under-determined designs return
    minimum-norm solutions instead of raising. Output is always a 1-D
    array of length `X.shape[1]`."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError(f"X must be 2-D and y 1-D (got {X.shape=}, {y.shape=})")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"row count mismatch: {X.shape[0]} vs {y.shape[0]}")
    weights, *_ = np.linalg.lstsq(X, y, rcond=None)
    return weights
