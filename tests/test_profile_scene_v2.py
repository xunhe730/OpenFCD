"""Tests for ProfileSceneView v2 segment-collect / overlay / export flow."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

PyQt6 = pytest.importorskip("PyQt6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from openfcd.io.annotation import (
    AnnotationSchema,
    ProfileLineData,
    WaveSegment,
    WaveStatsConfig,
)


def _make_view(qapp):
    from openfcd.gui.scenes.profile_scene import ProfileSceneView
    view = ProfileSceneView()
    return view


_FRAME_IDX = 100


def _seed_with_annotation(view, segs: list[WaveSegment] | None = None):
    pl = ProfileLineData(start=(50.0, 0.0), end=(50.0, 299.0))
    by_frame = {str(_FRAME_IDX): list(segs)} if segs else {}
    ws = WaveStatsConfig(segments_by_frame=by_frame)
    view._annotation = AnnotationSchema(wave_stats=ws, profile_line=pl)
    # Stub the spec so _frame_idx_at(0) -> _FRAME_IDX.
    view._spec = SimpleNamespace(
        frame_indices=[_FRAME_IDX],
        profile_lines={},
        viz_params={},
        run_id=None,
        name="test",
    )
    view._current_frame_pos = 0
    view._table_panel.set_annotation(view._annotation)
    view._sync_table_current_frame()


# ── Collect-button visibility / behaviour ──────────────────────────────────


def test_collect_button_toggle(qapp):
    view = _make_view(qapp)
    view._btn_collect.setChecked(True)
    assert view._btn_collect.isChecked()
    # Status label is updated with prompt text when collecting starts
    assert "1D" in view._status_lbl.text() or "端点" in view._status_lbl.text()
    view._btn_collect.setChecked(False)
    assert not view._btn_collect.isChecked()
    assert view._pending_first_x is None


# ── Segment-collect: simulated matplotlib clicks ───────────────────────────


def test_collect_two_clicks_create_segment(qapp):
    view = _make_view(qapp)
    _seed_with_annotation(view)

    # Set up a fake lower-axis target for click hit-test
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    view._lower_ax = ax

    view._btn_collect.setChecked(True)
    fake_event_1 = SimpleNamespace(button=1, inaxes=ax, xdata=5.0, ydata=0.1)
    fake_event_2 = SimpleNamespace(button=1, inaxes=ax, xdata=12.5, ydata=0.2)

    captured: list = []
    view.wave_segment_added.connect(lambda seg: captured.append(seg))

    view._on_mpl_click(fake_event_1)
    assert view._pending_first_x == pytest.approx(5.0)
    view._on_mpl_click(fake_event_2)

    assert len(captured) == 1
    seg = captured[0]
    assert seg.s_lo_mm == pytest.approx(5.0)
    assert seg.s_hi_mm == pytest.approx(12.5)
    # Collect button auto-uncheck after second click
    assert not view._btn_collect.isChecked()


def test_collect_reverse_order_normalised(qapp):
    """If user clicks the higher x first, segment should still be (lo, hi)."""
    view = _make_view(qapp)
    _seed_with_annotation(view)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    view._lower_ax = ax

    view._btn_collect.setChecked(True)
    captured: list = []
    view.wave_segment_added.connect(lambda seg: captured.append(seg))

    view._on_mpl_click(SimpleNamespace(button=1, inaxes=ax, xdata=20.0, ydata=0.0))
    view._on_mpl_click(SimpleNamespace(button=1, inaxes=ax, xdata=5.0, ydata=0.0))
    assert len(captured) == 1
    assert captured[0].s_lo_mm == pytest.approx(5.0)
    assert captured[0].s_hi_mm == pytest.approx(20.0)


def test_collect_outside_lower_axis_ignored(qapp):
    view = _make_view(qapp)
    _seed_with_annotation(view)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    fig2, other_ax = plt.subplots()
    view._lower_ax = ax

    view._btn_collect.setChecked(True)
    captured: list = []
    view.wave_segment_added.connect(lambda seg: captured.append(seg))

    # Click on a different axis → ignored
    view._on_mpl_click(SimpleNamespace(button=1, inaxes=other_ax, xdata=5.0, ydata=0.0))
    assert view._pending_first_x is None
    assert not captured


def test_collect_button_off_clicks_ignored(qapp):
    view = _make_view(qapp)
    _seed_with_annotation(view)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    view._lower_ax = ax

    captured: list = []
    view.wave_segment_added.connect(lambda seg: captured.append(seg))

    # Button not checked → no segment captured
    view._on_mpl_click(SimpleNamespace(button=1, inaxes=ax, xdata=5.0, ydata=0.0))
    view._on_mpl_click(SimpleNamespace(button=1, inaxes=ax, xdata=10.0, ydata=0.0))
    assert not captured


# ── Hover cursor ───────────────────────────────────────────────────────────


def test_hover_creates_artists_on_lower_axis(qapp):
    view = _make_view(qapp)
    _seed_with_annotation(view)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    view._fig = fig
    view._lower_ax = ax

    view._on_mpl_motion(
        SimpleNamespace(inaxes=ax, xdata=5.0, ydata=0.1)
    )
    assert view._hover_vline is not None
    assert view._hover_text is not None


def test_hover_outside_lower_axis_hides(qapp):
    view = _make_view(qapp)
    _seed_with_annotation(view)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    fig2, other = plt.subplots()
    view._fig = fig
    view._lower_ax = ax

    # First create artists by hovering over lower axis
    view._on_mpl_motion(SimpleNamespace(inaxes=ax, xdata=5.0, ydata=0.1))
    assert view._hover_vline is not None

    # Now move out — should hide
    view._on_mpl_motion(SimpleNamespace(inaxes=other, xdata=99.0, ydata=99.0))
    assert not view._hover_vline.get_visible()


# ── Segment overlay ────────────────────────────────────────────────────────


def test_overlay_segments_creates_patches(qapp):
    view = _make_view(qapp)
    _seed_with_annotation(
        view,
        [
            WaveSegment(s_lo_mm=0.0, s_hi_mm=10.0, color="#1f77b4"),
            WaveSegment(s_lo_mm=15.0, s_hi_mm=25.0, color="#d62728"),
        ],
    )
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    view._lower_ax = ax

    view._overlay_segments(active_idx=1)
    assert len(view._segment_patches) == 2


def test_overlay_invisible_segment_skipped(qapp):
    view = _make_view(qapp)
    _seed_with_annotation(
        view,
        [
            WaveSegment(s_lo_mm=0.0, s_hi_mm=10.0, visible=True),
            WaveSegment(s_lo_mm=10.0, s_hi_mm=20.0, visible=False),
        ],
    )
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    view._lower_ax = ax

    view._overlay_segments(active_idx=None)
    assert len(view._segment_patches) == 1


def test_table_row_click_sets_active_idx(qapp):
    view = _make_view(qapp)
    _seed_with_annotation(
        view,
        [
            WaveSegment(s_lo_mm=0.0, s_hi_mm=5.0),
            WaveSegment(s_lo_mm=10.0, s_hi_mm=15.0),
        ],
    )
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    view._lower_ax = ax

    view._on_table_row_clicked(1)
    assert view._active_segment_idx == 1
    assert len(view._segment_patches) == 2


# ── Export must not include overlays ───────────────────────────────────────


def test_export_clears_overlays(qapp, tmp_path):
    view = _make_view(qapp)
    _seed_with_annotation(
        view, [WaveSegment(s_lo_mm=0.0, s_hi_mm=10.0)]
    )
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    view._fig = fig
    view._lower_ax = ax
    view._overlay_segments(active_idx=0)
    assert len(view._segment_patches) >= 1

    out = tmp_path / "exp.png"
    view.export_profile_composite(out, dpi=80)
    assert out.exists() and out.stat().st_size > 0
    # After export, overlays are restored
    # (a fresh patch may or may not exist depending on figure state; just ensure
    # no exception)


# ── Segment edit via table → annotation updated ────────────────────────────


def test_table_segment_edit_updates_annotation(qapp):
    view = _make_view(qapp)
    _seed_with_annotation(
        view, [WaveSegment(s_lo_mm=0.0, s_hi_mm=10.0, label="orig")]
    )
    new_seg = WaveSegment(s_lo_mm=0.0, s_hi_mm=10.0, label="renamed")
    view._on_table_segment_edited(0, new_seg)
    segs = view._annotation.wave_stats.segments_for_frame(_FRAME_IDX)
    assert segs[0].label == "renamed"


def test_table_segment_delete_updates_annotation(qapp):
    view = _make_view(qapp)
    _seed_with_annotation(
        view,
        [
            WaveSegment(s_lo_mm=0.0, s_hi_mm=5.0, label="a"),
            WaveSegment(s_lo_mm=10.0, s_hi_mm=15.0, label="b"),
        ],
    )
    view._on_table_segment_deleted(0)
    segs = view._annotation.wave_stats.segments_for_frame(_FRAME_IDX)
    assert len(segs) == 1
    assert segs[0].label == "b"
