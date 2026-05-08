"""Default visualization parameters per scene type."""
from __future__ import annotations

import numpy as np


def eta_map_defaults(eta: np.ndarray | None = None) -> dict:
    """Auto vmin/vmax + bidirectional colormap for η data."""
    params: dict = {
        "cmap": "RdBu_r",
        "vmin": None,
        "vmax": None,
        "colorbar": "right",
        "dpi": 150,
        "alpha": 0.8,
    }
    if eta is not None and eta.size > 0:
        valid = eta[~np.isnan(eta)]
        if valid.size > 0:
            lo = float(np.percentile(valid, 2))
            hi = float(np.percentile(valid, 98))
            # Choose colormap based on whether data is bidirectional
            eps = (hi - lo) * 0.05
            if lo < -eps and hi > eps:
                params["cmap"] = "RdBu_r"
                extreme = max(abs(lo), abs(hi))
                params["vmin"] = -extreme
                params["vmax"] = extreme
            else:
                params["cmap"] = "viridis"
                params["vmin"] = lo
                params["vmax"] = hi
    return params


def rms_defaults(rms: np.ndarray | None = None) -> dict:
    params: dict = {
        "cmap": "magma",
        "vmin": 0.0,
        "vmax": None,
        "colorbar": "right",
        "dpi": 150,
    }
    if rms is not None and rms.size > 0:
        valid = rms[~np.isnan(rms)]
        if valid.size > 0:
            params["vmax"] = float(np.percentile(valid, 98))
    return params


def profile_defaults() -> dict:
    return {
        "cmap": "RdBu_r",
        "vmin": None,
        "vmax": None,
        "colorbar": "bottom",
        "line_color": "red",
        "profile_color": "blue",
        "grid": True,
        "xlabel": "x (mm)",
        "ylabel": "Wave height η (mm)",
        "strip_mm": 2.0,
        "y_range_mm": 30.0,
        "min_roi_width_mm": 160.0,
        "min_roi_height_mm": 50.0,
        "x_padding_mm": 5.0,
        "auto_crop": True,
        "show_measurements": False,
        "x_range_mm": None,
        "title": None,
        "dpi": 150,
    }


def scene_defaults(scene_type: str, data: np.ndarray | None = None) -> dict:
    """Return defaults for any scene type."""
    if scene_type == "eta_map":
        return eta_map_defaults(data)
    if scene_type == "rms":
        return rms_defaults(data)
    if scene_type == "profile":
        return profile_defaults()
    return {"dpi": 150}
