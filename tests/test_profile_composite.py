import numpy as np


def test_profile_composite_renders_heatmap_colorbar_and_profile():
    from openfcd.core.profile_composite import render_profile_composite

    eta = np.tile(np.linspace(-1.0, 1.0, 32), (16, 1))
    fig = render_profile_composite(
        eta,
        ((8.0, 2.0), (8.0, 30.0)),
        {"px_per_mm": 2.0, "strip_mm": 1.0, "show_measurements": False},
    )

    assert len(fig.axes) >= 3
    assert fig.axes[0].lines[0].get_linestyle() == "--"
    assert fig.axes[0].lines[0].get_color() == "red"
    assert fig.axes[2].lines[0].get_color() == "blue"
    assert fig.axes[2].get_xlabel() == "x (mm)"
    assert fig.axes[2].get_ylabel() == "Wave height η (mm)"
    assert len(fig.axes) == 3
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert not any("lambda =" in t or "h =" in t for t in texts)


def test_profile_composite_hides_colorbar_when_requested():
    from openfcd.core.profile_composite import render_profile_composite

    eta = np.tile(np.linspace(-1.0, 1.0, 32), (16, 1))
    fig = render_profile_composite(
        eta,
        ((8.0, 2.0), (8.0, 30.0)),
        {"px_per_mm": 2.0, "strip_mm": 1.0, "colorbar": "none"},
    )

    assert len(fig.axes) == 2


def test_profile_composite_heatmap_shares_profile_x_axis():
    from openfcd.core.profile_composite import extract_profile_window, render_profile_composite

    eta = np.tile(np.linspace(-1.0, 1.0, 80), (40, 1))
    line = ((20.0, 10.0), (20.0, 70.0))
    window = extract_profile_window(eta, line, px_per_mm=2.0, y_range_mm=12.0)
    fig = render_profile_composite(eta, line, {"px_per_mm": 2.0, "y_range_mm": 12.0})

    assert window.y_mm[-1] - window.y_mm[0] >= 50.0
    assert fig.axes[0].get_xlim() == fig.axes[2].get_xlim()
    assert fig.axes[0].get_ylabel() == "offset y (mm)"


def test_profile_composite_axes_visual_x_aligned():
    from openfcd.core.profile_composite import render_profile_composite

    eta = np.tile(np.linspace(-1.0, 1.0, 80), (40, 1))
    fig = render_profile_composite(eta, ((20.0, 10.0), (20.0, 70.0)), {"px_per_mm": 2.0})
    top = fig.axes[0].get_position()
    bottom = fig.axes[2].get_position()

    assert abs(top.x0 - bottom.x0) < 1e-9
    assert abs(top.x1 - bottom.x1) < 1e-9


def test_profile_composite_heatmap_limits_match_roi_with_aligned_axes():
    from openfcd.core.profile_composite import render_profile_composite

    eta = np.tile(np.linspace(-1.0, 1.0, 120), (60, 1))
    fig = render_profile_composite(
        eta,
        ((30.0, 10.0), (30.0, 110.0)),
        {"px_per_mm": 2.0, "y_range_mm": 12.0},
    )

    assert fig.axes[0].get_xlim() == fig.axes[2].get_xlim()
    ymin, ymax = fig.axes[0].get_ylim()
    assert abs(abs(ymax - ymin) - 12.0) < 1e-9
    assert fig.axes[0].get_aspect() == 1.0


def test_profile_composite_auto_x_window_preserves_drawn_roi_extent():
    from openfcd.core.profile_composite import extract_profile_window, render_profile_composite

    eta = np.ones((80, 300))
    line = ((40.0, 10.0), (40.0, 290.0))
    full_line_mm = 280.0 / 2.0
    fig = render_profile_composite(eta, line, {"px_per_mm": 2.0, "y_range_mm": 12.0})
    x0, x1 = fig.axes[2].get_xlim()

    assert (x1 - x0) >= full_line_mm
    window = extract_profile_window(eta, line, px_per_mm=2.0, y_range_mm=12.0)
    assert window.x_mm[0] <= -full_line_mm / 2.0
    assert window.x_mm[-1] >= full_line_mm / 2.0


def test_profile_composite_min_roi_width_is_extraction_floor():
    from openfcd.core.profile_composite import extract_profile_window

    eta = np.ones((120, 260))
    line = ((60.0, 100.0), (60.0, 110.0))
    window = extract_profile_window(
        eta,
        line,
        px_per_mm=2.0,
        y_range_mm=12.0,
        min_roi_width_mm=80.0,
        min_roi_height_mm=30.0,
        auto_crop=False,
    )

    assert window.x_mm[-1] - window.x_mm[0] == 80.0
    assert window.y_mm[-1] - window.y_mm[0] == 30.0


def test_profile_composite_context_validates_scalars():
    from openfcd.core.profile_composite import build_profile_composite_context

    ctx = build_profile_composite_context(
        eta=np.ones((8, 8)),
        profile_line=((1.0, 1.0), (1.0, 7.0)),
        viz_params={
            "px_per_mm": -1,
            "strip_mm": -2,
            "y_range_mm": "bad",
            "min_roi_width_mm": 0,
            "min_roi_height_mm": -1,
            "x_padding_mm": -3,
        },
    )

    assert ctx.px_per_mm == 1.0
    assert ctx.strip_mm == 2.0
    assert ctx.y_range_mm == 30.0
    assert ctx.min_roi_width_mm == 160.0
    assert ctx.min_roi_height_mm == 50.0
    assert ctx.x_padding_mm == 5.0
    assert ctx.body_source == "fallback"
    assert ctx.degraded_reason is not None


def test_profile_context_uses_spatial_calibration_over_viz_px():
    from openfcd.core.profile_composite import build_profile_composite_context

    ctx = build_profile_composite_context(
        eta=np.ones((8, 80)),
        profile_line=((4.0, 0.0), (4.0, 79.0)),
        viz_params={"px_per_mm": 1.0},
        spatial_calibration={
            "pixel_per_mm": 8.0,
            "spatial_calibration_source": "carrier_detected",
        },
    )

    assert ctx.px_per_mm == 8.0
    assert ctx.spatial_calibration_source == "carrier_detected"


def test_resolve_spatial_calibration_prefers_frame_attrs():
    from types import SimpleNamespace
    from openfcd.core.profile_composite import resolve_spatial_calibration

    class ResultStore:
        def read_frame_attrs(self, batch, frame_idx):
            return {
                "pixel_per_mm": 8.0,
                "spatial_calibration_source": "carrier_detected",
            }

        def read_batch_meta(self, batch):
            return {"pixel_per_mm_median": 4.0}

    calibration = resolve_spatial_calibration(
        ResultStore(),
        "default",
        3,
        {"calibration": {"pixel_per_mm_median": 2.0}},
        SimpleNamespace(profile=SimpleNamespace(px_per_mm=1.0)),
    )

    assert calibration.pixel_per_mm == 8.0
    assert calibration.source == "carrier_detected"


def test_resolve_spatial_calibration_marks_project_fallback_unverified():
    from types import SimpleNamespace
    from openfcd.core.profile_composite import resolve_spatial_calibration

    class ResultStore:
        def read_frame_attrs(self, batch, frame_idx):
            return {}

        def read_batch_meta(self, batch):
            raise KeyError(batch)

    calibration = resolve_spatial_calibration(
        ResultStore(),
        "default",
        3,
        None,
        SimpleNamespace(profile=SimpleNamespace(px_per_mm=7.27)),
    )

    assert calibration.pixel_per_mm == 7.27
    assert calibration.source == "fallback_unverified"
    assert calibration.degraded_reason


def test_profile_composite_auto_crop_keeps_body_region_in_x():
    from matplotlib.patches import Rectangle

    from openfcd.core.profile_composite import build_profile_composite_context, render_profile_composite_context

    eta = np.ones((80, 300))
    eta[30:50, 22:62] = np.nan
    line = ((40.0, 20.0), (40.0, 280.0))
    ctx = build_profile_composite_context(
        eta=eta,
        profile_line=line,
        viz_params={"px_per_mm": 2.0, "y_range_mm": 12.0, "x_padding_mm": 5.0, "auto_crop": True},
        body_polygon_rc=[(30.0, 22.0), (30.0, 62.0), (50.0, 62.0), (50.0, 22.0)],
        body_source="frame_polygon",
    )

    fig = render_profile_composite_context(ctx)
    x0, x1 = fig.axes[2].get_xlim()
    body_rect = next(p for p in fig.axes[0].patches if isinstance(p, Rectangle))

    assert x0 < body_rect.get_x() < x1
    assert x0 < body_rect.get_x() + body_rect.get_width() < x1


def test_profile_composite_manual_x_range_overrides_auto():
    from openfcd.core.profile_composite import render_profile_composite

    eta = np.ones((80, 300))
    fig = render_profile_composite(
        eta,
        ((40.0, 10.0), (40.0, 290.0)),
        {"px_per_mm": 2.0, "y_range_mm": 12.0, "x_range_mm": "10:30"},
    )

    assert fig.axes[2].get_xlim() == (10.0, 30.0)


def test_profile_composite_manual_x_range_is_display_only():
    from openfcd.core.profile_composite import extract_profile_window, render_profile_composite

    eta = np.ones((120, 300))
    line = ((60.0, 120.0), (60.0, 180.0))
    window = extract_profile_window(
        eta,
        line,
        px_per_mm=2.0,
        y_range_mm=20.0,
        min_roi_width_mm=100.0,
        x_range_mm="10:20",
        auto_crop=False,
    )
    fig = render_profile_composite(
        eta,
        line,
        {"px_per_mm": 2.0, "min_roi_width_mm": 100.0, "x_range_mm": "10:20"},
    )

    assert window.x_mm[-1] - window.x_mm[0] == 100.0
    assert fig.axes[2].get_xlim() == (10.0, 20.0)


def test_profile_composite_y_range_controls_final_display_height():
    from openfcd.core.profile_composite import render_profile_composite

    eta = np.ones((120, 240))
    fig = render_profile_composite(
        eta,
        ((60.0, 20.0), (60.0, 220.0)),
        {"px_per_mm": 2.0, "y_range_mm": 40.0},
    )

    ymin, ymax = fig.axes[0].get_ylim()
    assert abs(abs(ymax - ymin) - 40.0) < 1e-9


def test_body_span_shared_between_heatmap_and_profile():
    from matplotlib.patches import Rectangle

    from openfcd.core.profile_composite import build_profile_composite_context, render_profile_composite_context

    eta = np.ones((40, 80))
    eta[16:24, 34:46] = np.nan
    ctx = build_profile_composite_context(
        eta=eta,
        profile_line=((20.0, 5.0), (20.0, 75.0)),
        viz_params={"px_per_mm": 2.0, "y_range_mm": 12.0},
        body_polygon_rc=[(16.0, 34.0), (16.0, 46.0), (24.0, 46.0), (24.0, 34.0)],
        body_source="frame_polygon",
    )
    fig = render_profile_composite_context(ctx)
    body_rect = next(p for p in fig.axes[0].patches if isinstance(p, Rectangle))
    span = fig.axes[2].patches[0]

    assert abs(body_rect.get_x() - span.get_x()) < 1e-9
    assert abs((body_rect.get_x() + body_rect.get_width()) - (span.get_x() + span.get_width())) < 1e-9


def test_profile_composite_empty_without_line():
    from openfcd.core.profile_composite import render_profile_composite

    fig = render_profile_composite(np.ones((4, 4)), None)

    assert len(fig.axes) == 1
    assert "Draw a profile line" in fig.axes[0].texts[0].get_text()


def test_extract_line_profile_uses_user_line_not_midline():
    from openfcd.core.profile_composite import extract_line_profile

    eta = np.zeros((10, 10), dtype=float)
    eta[2, :] = 3.0
    eta[5, :] = 9.0

    _, values = extract_line_profile(eta, ((2.0, 0.0), (2.0, 9.0)))

    assert np.allclose(values, 3.0)


def test_wavelength_profile_renderer_uses_profile_context():
    from types import SimpleNamespace

    from openfcd.gui.renderers.wavelength_profile import ProfileCompositeRenderer

    class Results:
        def read_frame(self, batch, frame_idx):
            assert batch == "default"
            assert frame_idx == 7
            eta = np.zeros((10, 10), dtype=float)
            eta[2, :] = 3.0
            eta[5, :] = 9.0
            return eta

    fig = ProfileCompositeRenderer().render(
        Results(),
        "default",
        SimpleNamespace(
            frame_idx=7,
            profile_line=((2.0, 0.0), (2.0, 9.0)),
            min_roi_width_mm=4.5,
            min_roi_height_mm=1.0,
            y_range_mm=1.0,
            strip_mm=0.0,
        ),
    )
    ydata = fig.axes[2].lines[0].get_ydata()

    assert np.nanmax(ydata) == 3.0
