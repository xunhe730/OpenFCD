"""Shared η display helpers used by both `EtaHeatmapRenderer` and `EtaMap`.

Color-range policy: when both ``viz.eta_vmin_mm`` and ``viz.eta_vmax_mm`` are
``None`` (or absent), the range auto-scales to a symmetric 98th percentile of
finite values. Any explicit non-``None`` value overrides that axis.
"""
from __future__ import annotations

from typing import Any

import numpy as np


_DIVERGING_CMAPS = frozenset({"RdBu_r", "RdBu", "bwr", "seismic", "coolwarm"})


def compute_eta_color_range(
    eta: np.ndarray,
    viz: Any,
    *,
    diverging: bool = True,
    cmap: str | None = None,
) -> tuple[float, float]:
    """Resolve ``(vmin, vmax)`` for an η heatmap.

    Honors explicit ``viz.eta_vmin_mm`` / ``viz.eta_vmax_mm`` when non-``None``;
    otherwise auto-scales to symmetric 98th percentile of finite |η|.
    """
    user_vmin = getattr(viz, "eta_vmin_mm", None) if viz is not None else None
    user_vmax = getattr(viz, "eta_vmax_mm", None) if viz is not None else None
    use_diverging = diverging or (cmap is not None and cmap in _DIVERGING_CMAPS)

    finite = np.asarray(eta)[np.isfinite(eta)] if eta is not None else np.empty(0)
    if finite.size == 0:
        vmin = -1.0 if user_vmin is None else float(user_vmin)
        vmax = 1.0 if user_vmax is None else float(user_vmax)
        return vmin, vmax

    if user_vmax is None:
        auto_vmax = float(np.nanpercentile(np.abs(finite), 98))
    else:
        auto_vmax = float(user_vmax)
    if user_vmin is None:
        auto_vmin = -auto_vmax if use_diverging else float(np.nanpercentile(finite, 2))
    else:
        auto_vmin = float(user_vmin)
    return auto_vmin, auto_vmax


def crop_to_valid(eta: np.ndarray) -> np.ndarray:
    """Crop ``eta`` to the bounding box of finite values plus a 1-px halo."""
    if eta is None:
        return eta
    arr = np.asarray(eta)
    valid = np.isfinite(arr)
    if not valid.any():
        return arr
    rows = np.flatnonzero(valid.any(axis=1))
    cols = np.flatnonzero(valid.any(axis=0))
    r0 = max(0, int(rows[0]) - 1)
    r1 = min(arr.shape[0], int(rows[-1]) + 2)
    c0 = max(0, int(cols[0]) - 1)
    c1 = min(arr.shape[1], int(cols[-1]) + 2)
    return arr[r0:r1, c0:c1]
