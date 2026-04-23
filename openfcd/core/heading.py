"""Compute forward direction from per-frame polygon centroids."""
from __future__ import annotations

import numpy as np

from openfcd.core.mask import Polygon


def detect_heading(by_frame: dict[int, Polygon]) -> tuple[float, float]:
    """Least-squares line fit on polygon centroid sequence.

    Returns (dy, dx) unit vector pointing from earliest centroid to latest.
    Sign disambiguation: sign = sign(dot(fit_direction, centroid_last - centroid_first))
    Convention matches annotation 'forward_direction': [dy, dx] unit vector.
    """
    if not by_frame:
        raise ValueError("by_frame is empty; cannot compute forward direction")

    sorted_ids = sorted(by_frame.keys())
    centroids = np.array([by_frame[fid].centroid() for fid in sorted_ids])

    if len(centroids) < 2:
        raise ValueError("Need at least 2 frames to compute forward direction")

    ys = centroids[:, 0]
    xs = centroids[:, 1]
    t = np.arange(len(centroids), dtype=float)

    A = np.column_stack([t, np.ones(len(t))])
    coef_y, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
    coef_x, _ = np.linalg.lstsq(A, xs, rcond=None)[0]

    direction = np.array([coef_y, coef_x])
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError("Degenerate centroid sequence; all centroids are co-located")

    unit = direction / norm

    global_vec = centroids[-1] - centroids[0]
    if np.dot(unit, global_vec) < 0:
        unit = -unit

    return (float(unit[0]), float(unit[1]))
