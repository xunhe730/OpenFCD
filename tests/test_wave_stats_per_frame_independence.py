"""Per-frame independence of wave_stats segments (v3).

Adding a segment on frame A must not affect frame B; switching back to A
must show the original segment list.
"""
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


FRAME_A = 100
FRAME_B = 200


def _make_view(qapp):
    from openfcd.gui.scenes.profile_scene import ProfileSceneView
    view = ProfileSceneView()
    pl = ProfileLineData(start=(50.0, 0.0), end=(50.0, 299.0))
    view._annotation = AnnotationSchema(
        wave_stats=WaveStatsConfig(),
        profile_line=pl,
    )
    view._spec = SimpleNamespace(
        frame_indices=[FRAME_A, FRAME_B],
        profile_lines={},
        viz_params={},
        run_id=None,
        name="per-frame-test",
    )
    view._current_frame_pos = 0
    view._table_panel.set_annotation(view._annotation)
    view._sync_table_current_frame()
    return view


def test_segments_are_per_frame(qapp):
    view = _make_view(qapp)

    seg_a = WaveSegment(s_lo_mm=-30.0, s_hi_mm=-5.0, label="A")
    view._append_segment_local(seg_a)

    ws = view._annotation.wave_stats
    assert ws is not None
    a_segs = ws.segments_for_frame(FRAME_A)
    assert len(a_segs) == 1
    assert a_segs[0].label == "A"
    # Frame B must still be empty.
    assert ws.segments_for_frame(FRAME_B) == []

    # Switch to frame B; the table panel must render an empty list.
    view._set_frame_pos(1)
    assert view._table_panel._segments() == []

    seg_b = WaveSegment(s_lo_mm=5.0, s_hi_mm=20.0, label="B")
    view._append_segment_local(seg_b)

    ws = view._annotation.wave_stats
    a_segs = ws.segments_for_frame(FRAME_A)
    b_segs = ws.segments_for_frame(FRAME_B)
    assert len(a_segs) == 1 and a_segs[0].label == "A"
    assert len(b_segs) == 1 and b_segs[0].label == "B"

    # Switch back to A; table panel sees A's segments only.
    view._set_frame_pos(0)
    tbl_segs = view._table_panel._segments()
    assert len(tbl_segs) == 1
    assert tbl_segs[0].label == "A"


def test_collect_button_resets_on_frame_switch(qapp):
    """Switching frames must un-check the segment-collect button."""
    view = _make_view(qapp)
    view._btn_collect.setChecked(True)
    assert view._btn_collect.isChecked()
    view._set_frame_pos(1)
    assert not view._btn_collect.isChecked()


def test_copy_previous_frame(qapp):
    view = _make_view(qapp)

    # Seed frame A with two segments.
    view._set_frame_pos(0)
    view._append_segment_local(WaveSegment(s_lo_mm=-30.0, s_hi_mm=-5.0))
    view._append_segment_local(WaveSegment(s_lo_mm=5.0, s_hi_mm=30.0))

    # Move to frame B (initially empty) and copy from the previous frame.
    view._set_frame_pos(1)
    assert view._previous_frame_has_segments() is True
    view._on_copy_previous_frame_segments()

    ws = view._annotation.wave_stats
    assert ws is not None
    b_segs = ws.segments_for_frame(FRAME_B)
    a_segs = ws.segments_for_frame(FRAME_A)
    assert len(b_segs) == len(a_segs) == 2
    assert [s.s_lo_mm for s in b_segs] == [s.s_lo_mm for s in a_segs]


def test_slider_handler_resyncs_table_rows(qapp):
    """Regression: user-driven slider changes (``_on_slider_moved`` /
    ``_on_slider``) only update ``_current_frame_pos`` and request a debounced
    render. The table panel's row list must end up reflecting the destination
    frame's segments — earlier this only worked through the programmatic
    ``_set_frame_pos`` path, so dragging the slider left the table showing the
    previous frame's row entries while overlays/stats were correctly empty.
    """
    view = _make_view(qapp)

    # Seed segments only on frame A; B stays empty.
    view._append_segment_local(WaveSegment(s_lo_mm=-30.0, s_hi_mm=-5.0))
    assert len(view._table_panel._segments()) == 1

    # Drag to frame B via the user-facing slider handler, then flush the
    # debounced render. The handler must trigger a table resync.
    view._on_slider_moved(1)
    view._render_timer.stop()
    view._do_render()
    assert view._table_panel._segments() == []

    # Drag back to A; rows must reappear.
    view._on_slider(0)
    view._render_timer.stop()
    view._do_render()
    assert len(view._table_panel._segments()) == 1


def test_delete_on_one_frame_does_not_affect_other(qapp):
    view = _make_view(qapp)

    view._set_frame_pos(0)
    view._append_segment_local(WaveSegment(s_lo_mm=-30.0, s_hi_mm=-5.0))
    view._set_frame_pos(1)
    view._append_segment_local(WaveSegment(s_lo_mm=5.0, s_hi_mm=20.0))

    # Delete the segment on frame B; frame A must keep its segment.
    view._on_table_segment_deleted(0)

    ws = view._annotation.wave_stats
    assert len(ws.segments_for_frame(FRAME_A)) == 1
    assert ws.segments_for_frame(FRAME_B) == []
