"""AC-B2: ``EtaHeatmapRenderer.render(mode='frame')`` and ``EtaMap.set_data``
share the same color-range and crop helpers, so given the same η array they
produce equivalent rasterized images (same data after crop, same vmin/vmax,
same colormap).
"""
from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from openfcd.gui.renderers._eta_view import compute_eta_color_range, crop_to_valid


class _StubResults:
    """Minimal stand-in for HDF5ResultStore exposing read_frame/read_summary."""

    def __init__(self, frames: dict[int, np.ndarray], mean: np.ndarray) -> None:
        self._frames = frames
        self._mean = mean

    def read_frame(self, _batch: str, frame_id: int) -> np.ndarray:
        return self._frames[int(frame_id)]

    def read_summary(self, _batch: str, kind: str) -> np.ndarray:
        if kind == "eta_mean":
            return self._mean
        raise KeyError(kind)


def _synthetic_eta() -> np.ndarray:
    rng = np.random.default_rng(0)
    eta = rng.normal(0.0, 0.05, size=(64, 80))
    eta[:8, :] = np.nan
    eta[-8:, :] = np.nan
    eta[:, :8] = np.nan
    eta[:, -8:] = np.nan
    return eta


def test_profile_vs_preview_data_and_range_parity() -> None:
    eta = _synthetic_eta()
    results = _StubResults({0: eta}, mean=eta)
    viz = SimpleNamespace(eta_vmin_mm=None, eta_vmax_mm=None, cmap="RdBu_r")

    # --- EtaHeatmapRenderer (frame mode) — render to an Axes we can inspect.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from openfcd.gui.renderers.eta_heatmap import EtaHeatmapRenderer

    fig, ax = plt.subplots()
    EtaHeatmapRenderer().render(results, "default", viz, ax=ax, mode="frame", frame_idx=0)
    images = [im for im in ax.get_images()]
    assert images, "renderer must produce an AxesImage"
    profile_data = images[0].get_array().filled(np.nan) if hasattr(images[0].get_array(), "filled") else np.asarray(images[0].get_array())
    profile_vmin, profile_vmax = images[0].get_clim()
    profile_cmap = images[0].get_cmap().name
    plt.close(fig)

    # --- EtaMap.set_data path: equivalent helper outputs.
    cropped = crop_to_valid(eta)
    map_vmin, map_vmax = compute_eta_color_range(cropped, viz, cmap="RdBu_r")

    # Data array equality (NaN-aware) after shared crop
    assert profile_data.shape == cropped.shape
    finite_a = np.isfinite(profile_data)
    finite_b = np.isfinite(cropped)
    assert np.array_equal(finite_a, finite_b)
    assert np.allclose(profile_data[finite_a], cropped[finite_b], atol=0, rtol=0)

    # Color range equality
    assert profile_vmin == pytest.approx(map_vmin)
    assert profile_vmax == pytest.approx(map_vmax)
    # Symmetric auto-scale by default with diverging cmap
    assert profile_vmin == pytest.approx(-profile_vmax)
    assert profile_cmap == "RdBu_r"


def test_explicit_viz_overrides_autoscale() -> None:
    eta = _synthetic_eta()
    viz = SimpleNamespace(eta_vmin_mm=-1.5, eta_vmax_mm=2.0, cmap="RdBu_r")
    vmin, vmax = compute_eta_color_range(eta, viz)
    assert vmin == -1.5
    assert vmax == 2.0
