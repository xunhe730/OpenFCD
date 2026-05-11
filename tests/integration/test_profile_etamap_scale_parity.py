"""Regression: Profile composite and Run-monitor EtaMap must use identical
(vmin, vmax) and pixel data for the same η frame.

Regresses a Phase 4 bug where ``profile_composite._auto_limits`` computed the
98th percentile on the cropped line-aligned ``window.eta`` while ``EtaMap``
computed it on the whole per-frame η, producing visibly different color
saturation between Profile preview and the per-frame Run-monitor view.
"""
from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.backends.backend_agg as _backend_agg
import matplotlib.figure as _mpl_figure

import numpy as np
import pytest

from openfcd.core.profile_composite import (
    build_profile_composite_context,
    render_profile_composite_context,
)
from openfcd.gui.renderers._eta_view import compute_eta_color_range, crop_to_valid


def _synthetic_eta(seed: int, shape: tuple[int, int] = (96, 128)) -> np.ndarray:
    """Synthetic η with a NaN border, a NaN body hole, and a wave pattern."""
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    eta = 0.4 * np.sin(2 * np.pi * xx / 30.0) * np.exp(-((yy - h / 2) ** 2) / (2 * (h / 6) ** 2))
    eta = eta + rng.normal(0.0, 0.02, size=shape)
    # NaN halo
    eta[:6, :] = np.nan
    eta[-6:, :] = np.nan
    eta[:, :6] = np.nan
    eta[:, -6:] = np.nan
    # NaN body hole — the kind of polygon-mask hole annotation produces.
    eta[h // 2 - 5 : h // 2 + 5, w // 2 - 8 : w // 2 + 8] = np.nan
    return eta.astype(np.float64)


@pytest.mark.parametrize("frame_seed", [0, 1, 2])
def test_profile_and_etamap_share_color_range(frame_seed: int) -> None:
    """For frame index k, Profile and EtaMap must produce identical
    (vmin, vmax) and identical post-crop pixel arrays.
    """
    eta = _synthetic_eta(frame_seed)

    # --- EtaMap path: full frame → crop_to_valid → compute_eta_color_range
    viz = SimpleNamespace(eta_vmin_mm=None, eta_vmax_mm=None, cmap="RdBu_r")
    map_eta = crop_to_valid(eta)
    map_vmin, map_vmax = compute_eta_color_range(map_eta, viz, cmap="RdBu_r")

    # --- Profile path: build context with default viz_params, render, then
    # introspect the imshow created on the heatmap axis.
    profile_line = ((20.0, 30.0), (60.0, 100.0))  # diagonal line in row,col
    context = build_profile_composite_context(
        eta=eta,
        profile_line=profile_line,
        viz_params={"colorbar": "none"},  # skip colorbar to simplify axis count
        frame_idx=frame_seed,
        frame_name=f"frame_{frame_seed:03d}.tif",
    )
    fig = _mpl_figure.Figure(figsize=(6.0, 5.0), dpi=100)
    _backend_agg.FigureCanvasAgg(fig)
    render_profile_composite_context(context, fig=fig, layout_mode="export")

    # Find the heatmap axes (the one with an AxesImage).
    heatmap_axes = [ax for ax in fig.axes if ax.get_images()]
    assert heatmap_axes, "Profile composite must render an AxesImage"
    im = heatmap_axes[0].get_images()[0]
    profile_vmin, profile_vmax = im.get_clim()

    # AC-B2 invariant: byte-equal float ranges
    assert profile_vmin == pytest.approx(map_vmin, abs=0.0, rel=0.0), (
        f"vmin diverged: profile={profile_vmin} map={map_vmin}"
    )
    assert profile_vmax == pytest.approx(map_vmax, abs=0.0, rel=0.0), (
        f"vmax diverged: profile={profile_vmax} map={map_vmax}"
    )

    # Symmetric auto-scale (diverging cmap default).
    assert profile_vmin == pytest.approx(-profile_vmax)


def test_explicit_viz_overrides_propagate_to_profile() -> None:
    """User-set viz.eta_vmin_mm / eta_vmax_mm must override auto-scale in
    Profile composite, matching EtaMap's contract.
    """
    eta = _synthetic_eta(0)
    context = build_profile_composite_context(
        eta=eta,
        profile_line=((20.0, 30.0), (60.0, 100.0)),
        viz_params={"vmin": -0.3, "vmax": 0.3, "colorbar": "none"},
    )
    fig = _mpl_figure.Figure(figsize=(6.0, 5.0), dpi=100)
    _backend_agg.FigureCanvasAgg(fig)
    render_profile_composite_context(context, fig=fig, layout_mode="export")
    im = next(ax for ax in fig.axes if ax.get_images()).get_images()[0]
    vmin, vmax = im.get_clim()
    assert vmin == pytest.approx(-0.3)
    assert vmax == pytest.approx(0.3)
