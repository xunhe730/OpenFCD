"""Streaming aggregation accumulators for the per-frame compute loop."""
from __future__ import annotations

import numpy as np


class StreamingMeanAccumulator:
    """Per-pixel incremental mean over a stream of η frames.

    Preserves the exact numerics of the previous in-line implementation in
    ComputeStage: NaN-aware sum and count, mean = sum / count with NaN where
    no valid sample was observed.
    """

    def __init__(self) -> None:
        self._sum: np.ndarray | None = None
        self._count: np.ndarray | None = None
        self._n_frames: int = 0

    def update(self, eta: np.ndarray) -> None:
        if eta is None:
            return
        if self._sum is None:
            self._sum = np.zeros_like(eta, dtype=np.float64)
            self._count = np.zeros(eta.shape, dtype=np.int64)
        if eta.shape != self._sum.shape:
            return
        valid = np.isfinite(eta)
        self._sum[valid] += eta[valid]
        self._count[valid] += 1
        self._n_frames += 1

    def finalize(self) -> np.ndarray | None:
        if self._sum is None or self._count is None:
            return None
        safe = np.maximum(self._count, 1)
        mean = self._sum / safe
        mean[self._count == 0] = np.nan
        return mean

    @property
    def n_valid_pixels(self) -> int:
        if self._count is None:
            return 0
        return int(np.nansum(self._count > 0))

    @property
    def n_frames(self) -> int:
        return self._n_frames
