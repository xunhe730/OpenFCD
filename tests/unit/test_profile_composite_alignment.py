"""Alignment invariants for the Profile Composite renderer.

These tests lock in that the eta heatmap (top axes) and the wave-height plot
(bottom axes) occupy identical horizontal page extents (x0 + width in figure
fraction), so that shared xlim translates to pixel-aligned x in both views.
The previous bug: ax_map.set_aspect('equal', adjustable='box') shrunk the
heatmap axes width while ax_profile kept the full assigned rect width, so the
heatmap looked shorter than the profile and the body mask drifted sideways.
"""
from __future__ import annotations

import matplotlib.figure
import numpy as np
import pytest

from openfcd.core.profile_composite import (
    build_profile_composite_context,
    render_profile_composite_context,
)


def _synthetic_eta(h: int = 80, w: int = 320) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return 0.1 * np.sin(2 * np.pi * xx / 30.0) + 0.0 * yy


def _make_context(
    *,
    body_polygon_rc: list[tuple[float, float]] | None = None,
    px_per_mm: float = 4.0,
):
    eta = _synthetic_eta()
    h, w = eta.shape
    profile_line = ((h / 2.0, 5.0), (h / 2.0, w - 5.0))
    return build_profile_composite_context(
        eta=eta,
        profile_line=profile_line,
        viz_params={
            "px_per_mm": px_per_mm,
            "strip_mm": 1.0,
            "y_range_mm": 20.0,
            "min_roi_width_mm": 60.0,
            "min_roi_height_mm": 18.0,
            "x_padding_mm": 2.0,
            "vmin": -0.2,
            "vmax": 0.2,
            "auto_crop": False,
        },
        frame_idx=0,
        frame_name="f0",
        body_polygon_rc=body_polygon_rc,
        body_source="frame_polygon" if body_polygon_rc else "fallback",
        spatial_calibration={
            "pixel_per_mm": px_per_mm,
            "spatial_calibration_source": "test",
        },
    )


def _render_with_widget(fig_w_in: float, fig_h_in: float, ctx, monkeypatch, dpi: float = 100.0):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = matplotlib.figure.Figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    FigureCanvasAgg(fig)
    import openfcd.core.profile_composite as pc
    monkeypatch.setattr(pc, "_figure_size_from_widget", lambda f, d: (fig_w_in, fig_h_in))
    render_profile_composite_context(ctx, fig=fig, layout_mode="preview")
    fig.canvas.draw()
    return fig


@pytest.mark.parametrize(
    "fig_w,fig_h",
    [
        (10.0, 3.5),  # wide-short — bug case (widget shorter than equal-aspect needs)
        (5.0, 7.0),   # tall-narrow
        (8.0, 6.0),   # near-square
    ],
)
def test_map_and_profile_axes_share_horizontal_extent(fig_w, fig_h, monkeypatch):
    ctx = _make_context()
    fig = _render_with_widget(fig_w, fig_h, ctx, monkeypatch)
    ax_map = fig.axes[0]
    ax_profile = fig.axes[-1]
    pos_map = ax_map.get_position()
    pos_profile = ax_profile.get_position()
    aspect = ax_map.get_aspect()
    assert aspect == "equal" or float(aspect) == 1.0
    assert pos_map.x0 == pytest.approx(pos_profile.x0, abs=2e-3)
    assert pos_map.width == pytest.approx(pos_profile.width, abs=2e-3)


def test_body_mask_pixel_alignment_between_map_and_profile(monkeypatch):
    body_polygon_rc = [(35.0, 140.0), (35.0, 200.0), (45.0, 200.0), (45.0, 140.0)]
    ctx = _make_context(body_polygon_rc=body_polygon_rc)
    fig = _render_with_widget(10.0, 4.0, ctx, monkeypatch)

    ax_map = fig.axes[0]
    ax_profile = fig.axes[-1]
    rects = [p for p in ax_map.patches if p.__class__.__name__ == "Rectangle"]
    assert rects, "expected body Rectangle on heatmap"
    rect = rects[0]
    rx0 = rect.get_x()
    rx1 = rx0 + rect.get_width()

    spans = list(ax_profile.patches)
    assert spans, "expected body axvspan on profile"
    span = spans[0]
    sx0 = float(span.get_x())
    sx1 = sx0 + float(span.get_width())

    map_x0_pix = ax_map.transData.transform((rx0, 0))[0]
    map_x1_pix = ax_map.transData.transform((rx1, 0))[0]
    pro_x0_pix = ax_profile.transData.transform((sx0, 0))[0]
    pro_x1_pix = ax_profile.transData.transform((sx1, 0))[0]
    assert abs(map_x0_pix - pro_x0_pix) <= 1.5
    assert abs(map_x1_pix - pro_x1_pix) <= 1.5
