"""Profile (cross-section) sampling along a line through an eta field.

sample_along draws an extended line through the frame (from edge to
edge, passing through p0->p1) and samples eta values at evenly spaced
positions. Used by the Profile scene to generate eta(s) curves.
"""
from __future__ import annotations
import numpy as np


def _extend_to_edges(
    p0: tuple[float, float],
    p1: tuple[float, float],
    h: int,
    w: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Extend the line through p0, p1 to the image boundary.

    Returns the clipped (start, end) in (row, col) coords.
    """
    r0, c0 = float(p0[0]), float(p0[1])
    r1, c1 = float(p1[0]), float(p1[1])
    dr, dc = r1 - r0, c1 - c0

    if dr == 0 and dc == 0:
        return (r0, c0), (r1, c1)

    # Parametric: P(t) = (r0 + t*dr, c0 + t*dc)
    # Find t range such that both coords stay in [0,h) x [0,w)
    t_bounds = []
    if dr != 0:
        t_bounds += [(0 - r0) / dr, (h - 1 - r0) / dr]
    if dc != 0:
        t_bounds += [(0 - c0) / dc, (w - 1 - c0) / dc]

    if not t_bounds:
        return (r0, c0), (r1, c1)

    t_min, t_max = min(t_bounds), max(t_bounds)
    return (
        (r0 + t_min * dr, c0 + t_min * dc),
        (r0 + t_max * dr, c0 + t_max * dc),
    )


def sample_along(
    eta: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    n: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample eta along the line through p0->p1, extended to frame edges.

    Args:
        eta: 2-D array (H, W) of eta values (may contain NaN).
        p0:  (row, col) of one point on the line.
        p1:  (row, col) of another point on the line.
        n:   number of sample points; defaults to max(H, W).

    Returns:
        (distances_mm, eta_values) both shape (n,).
        distances are in pixel units (caller converts to mm if needed).
    """
    h, w = eta.shape
    start, end = _extend_to_edges(p0, p1, h, w)

    if n is None:
        n = max(h, w)
    n = max(n, 2)

    rows = np.linspace(start[0], end[0], n)
    cols = np.linspace(start[1], end[1], n)

    # Clamp to valid range
    rows = np.clip(rows, 0, h - 1)
    cols = np.clip(cols, 0, w - 1)

    row_int = np.round(rows).astype(int)
    col_int = np.round(cols).astype(int)

    values = eta[row_int, col_int].astype(float)

    dr = end[0] - start[0]
    dc = end[1] - start[1]
    total_len = float(np.sqrt(dr**2 + dc**2))
    distances = np.linspace(0.0, total_len, n)

    return distances, values
