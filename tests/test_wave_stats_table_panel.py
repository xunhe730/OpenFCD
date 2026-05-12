"""Tests for openfcd/gui/panels/wave_stats_table.py (v2)."""

from __future__ import annotations

import pytest

PyQt6 = pytest.importorskip("PyQt6")

from openfcd.io.annotation import (
    AnnotationSchema,
    WaveSegment,
    WaveStatsConfig,
)
from openfcd.gui.panels.wave_stats_table import (
    COL_LABEL,
    COL_S_HI,
    COL_S_LO,
    WaveStatsTablePanel,
)


def _make_annotation(segments: list[WaveSegment] | None = None) -> AnnotationSchema:
    segs = segments or [
        WaveSegment(s_lo_mm=0.0, s_hi_mm=10.0, label="A"),
        WaveSegment(s_lo_mm=10.0, s_hi_mm=20.0, label="B", color="#ff0000", visible=False),
    ]
    return AnnotationSchema(wave_stats=WaveStatsConfig(segments=segs))


def test_table_populates_from_annotation(qapp):
    panel = WaveStatsTablePanel()
    panel.set_annotation(_make_annotation())
    assert panel._table.rowCount() == 2
    assert panel._table.item(0, COL_LABEL).text() == "A"
    assert panel._table.item(1, COL_LABEL).text() == "B"


def test_table_empty_when_no_wave_stats(qapp):
    panel = WaveStatsTablePanel()
    panel.set_annotation(AnnotationSchema())
    assert panel._table.rowCount() == 0


def test_segment_clicked_emits(qapp):
    panel = WaveStatsTablePanel()
    panel.set_annotation(_make_annotation())
    captured: list = []
    panel.segmentClicked.connect(lambda i: captured.append(i))
    panel._on_cell_clicked(1, 0)
    assert captured == [1]


def test_label_edit_emits_segmentEdited(qapp):
    panel = WaveStatsTablePanel()
    panel.set_annotation(_make_annotation())
    captured: list = []
    panel.segmentEdited.connect(lambda i, s: captured.append((i, s)))
    panel._table.item(0, COL_LABEL).setText("Renamed")
    assert captured
    idx, seg = captured[-1]
    assert idx == 0
    assert seg.label == "Renamed"


def test_s_lo_edit_emits(qapp):
    panel = WaveStatsTablePanel()
    panel.set_annotation(_make_annotation())
    captured: list = []
    panel.segmentEdited.connect(lambda i, s: captured.append((i, s)))
    panel._table.item(0, COL_S_LO).setText("1.5")
    assert captured
    idx, seg = captured[-1]
    assert idx == 0
    assert seg.s_lo_mm == pytest.approx(1.5)


def test_s_hi_invalid_does_not_emit(qapp):
    """s_hi <= s_lo: edit is silently rejected."""
    panel = WaveStatsTablePanel()
    panel.set_annotation(_make_annotation())
    captured: list = []
    panel.segmentEdited.connect(lambda i, s: captured.append((i, s)))
    panel._table.item(0, COL_S_HI).setText("-1.0")
    assert not captured


def test_visibility_toggle_emits(qapp):
    panel = WaveStatsTablePanel()
    panel.set_annotation(_make_annotation())
    captured: list = []
    panel.visibilityToggled.connect(lambda i, v: captured.append((i, v)))
    panel._on_visibility_toggled(0, False)
    assert captured == [(0, False)]


def test_delete_emits(qapp):
    panel = WaveStatsTablePanel()
    panel.set_annotation(_make_annotation())
    captured: list = []
    panel.segmentDeleted.connect(lambda i: captured.append(i))
    panel._on_delete_clicked(1)
    assert captured == [1]


def test_three_segments_render(qapp):
    panel = WaveStatsTablePanel()
    ann = _make_annotation(
        [
            WaveSegment(s_lo_mm=0.0, s_hi_mm=5.0, label="x"),
            WaveSegment(s_lo_mm=5.0, s_hi_mm=10.0, label="y"),
            WaveSegment(s_lo_mm=10.0, s_hi_mm=15.0, label="z"),
        ]
    )
    panel.set_annotation(ann)
    assert panel._table.rowCount() == 3
