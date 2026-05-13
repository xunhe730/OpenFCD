"""Tests for WaveStatsPanel (v2) and SessionController wave-stats methods.

v2 WaveStatsPanel only exposes:
  - peak_prominence_k QDoubleSpinBox (_prom_spin, range 0.05–2.0)
  - Recompute Wave Stats button (_btn_recompute)
  - set_annotation(annotation) / set_has_run(bool) / set_running(bool)
  - signals: prominence_k_changed, recompute_clicked

Covers:
  a.  peak_prominence_k spinbox range [0.05, 2.0] — clamping
  a2. set_annotation with WaveStatsConfig(peak_prominence_k=0.5) → spinbox shows 0.5
  a3. Editing spinbox → prominence_k_changed emitted after debounce timer fires
  b.  Recompute button disabled matrix (no run / wave_stats=None / no segments in any
      frame / running)
  b2. Recompute click → recompute_clicked signal emitted
  c.  SessionController: update_wave_stats_config, set_profile_line
      (per-frame segment mutations now go through ProfileSceneView directly;
      the controller's flat-list convenience methods were removed with the
      ``segments_by_frame`` migration.)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from openfcd.gui.panels.properties import WaveStatsPanel
from openfcd.gui.controllers.session_controller import SessionController
from openfcd.io.annotation import (
    AnnotationSchema,
    ProfileLineData,
    WaveSegment,
    WaveStatsConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def panel(qapp) -> WaveStatsPanel:  # noqa: F811 — qapp from conftest
    return WaveStatsPanel()


@pytest.fixture()
def ann_with_segments() -> AnnotationSchema:
    return AnnotationSchema(
        wave_stats=WaveStatsConfig(
            peak_prominence_k=0.5,
            segments_by_frame={
                "0": [
                    WaveSegment(s_lo_mm=0.0, s_hi_mm=10.0, label="fore"),
                    WaveSegment(s_lo_mm=15.0, s_hi_mm=30.0, label="aft"),
                ]
            },
        )
    )


# ---------------------------------------------------------------------------
# a. Spinbox range clamping
# ---------------------------------------------------------------------------

class TestSpinboxRange:
    def test_clamp_below_minimum(self, panel: WaveStatsPanel) -> None:
        panel._prom_spin.setValue(0.01)
        assert panel._prom_spin.value() == pytest.approx(0.05)

    def test_clamp_above_maximum(self, panel: WaveStatsPanel) -> None:
        panel._prom_spin.setValue(3.0)
        assert panel._prom_spin.value() == pytest.approx(2.0)

    def test_value_in_range_unchanged(self, panel: WaveStatsPanel) -> None:
        panel._prom_spin.setValue(1.0)
        assert panel._prom_spin.value() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# a2. set_annotation populates spinbox
# ---------------------------------------------------------------------------

class TestSetAnnotation:
    def test_prominence_k_populated(self, panel: WaveStatsPanel, ann_with_segments: AnnotationSchema) -> None:
        panel.set_annotation(ann_with_segments)
        assert panel._prom_spin.value() == pytest.approx(0.5)

    def test_none_annotation_does_not_raise(self, panel: WaveStatsPanel) -> None:
        panel.set_annotation(None)  # must not raise

    def test_empty_segments_treats_as_no_wave_stats(self, panel: WaveStatsPanel) -> None:
        # WaveStatsConfig with no segments → recompute stays disabled even if has_run
        ann = AnnotationSchema(
            wave_stats=WaveStatsConfig(peak_prominence_k=0.3, segments_by_frame={})
        )
        panel.set_has_run(True)
        panel.set_annotation(ann)
        assert not panel._btn_recompute.isEnabled()

    def test_wave_stats_none_disables_recompute(self, panel: WaveStatsPanel) -> None:
        ann = AnnotationSchema()  # wave_stats=None
        panel.set_has_run(True)
        panel.set_annotation(ann)
        assert not panel._btn_recompute.isEnabled()


# ---------------------------------------------------------------------------
# a3. Debounce: spinbox change → signal after timer fires
# ---------------------------------------------------------------------------

class TestDebounce:
    def test_prominence_change_not_immediate(self, panel: WaveStatsPanel) -> None:
        received: list[float] = []
        panel.prominence_k_changed.connect(received.append)
        panel._prom_spin.setValue(0.8)
        assert len(received) == 0  # timer has not fired

    def test_prominence_emits_after_timer(self, panel: WaveStatsPanel) -> None:
        received: list[float] = []
        panel.prominence_k_changed.connect(received.append)
        panel._prom_spin.setValue(0.75)
        panel._prom_timer.timeout.emit()  # simulate timer expiry
        assert len(received) == 1
        assert received[0] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# b. Recompute button disabled-state matrix
# ---------------------------------------------------------------------------

class TestRecomputeButtonState:
    def test_disabled_by_default(self, panel: WaveStatsPanel) -> None:
        assert not panel._btn_recompute.isEnabled()

    def test_disabled_without_run(self, panel: WaveStatsPanel, ann_with_segments: AnnotationSchema) -> None:
        panel.set_annotation(ann_with_segments)
        panel.set_has_run(False)
        assert not panel._btn_recompute.isEnabled()

    def test_disabled_when_wave_stats_none(self, panel: WaveStatsPanel) -> None:
        panel.set_has_run(True)
        panel.set_annotation(AnnotationSchema())  # wave_stats=None
        assert not panel._btn_recompute.isEnabled()

    def test_disabled_when_segments_empty(self, panel: WaveStatsPanel) -> None:
        panel.set_has_run(True)
        ann = AnnotationSchema(wave_stats=WaveStatsConfig(segments_by_frame={}))
        panel.set_annotation(ann)
        assert not panel._btn_recompute.isEnabled()

    def test_disabled_while_running(self, panel: WaveStatsPanel, ann_with_segments: AnnotationSchema) -> None:
        panel.set_annotation(ann_with_segments)
        panel.set_has_run(True)
        panel.set_running(True)
        assert not panel._btn_recompute.isEnabled()

    def test_enabled_when_all_conditions_met(self, panel: WaveStatsPanel, ann_with_segments: AnnotationSchema) -> None:
        panel.set_annotation(ann_with_segments)
        panel.set_has_run(True)
        panel.set_running(False)
        assert panel._btn_recompute.isEnabled()


# ---------------------------------------------------------------------------
# b2. Recompute button click emits signal
# ---------------------------------------------------------------------------

class TestRecomputeClick:
    def test_click_emits_recompute_clicked(self, panel: WaveStatsPanel, ann_with_segments: AnnotationSchema) -> None:
        panel.set_annotation(ann_with_segments)
        panel.set_has_run(True)
        called: list[bool] = []
        panel.recompute_clicked.connect(lambda: called.append(True))
        panel._btn_recompute.click()
        assert len(called) == 1


# ---------------------------------------------------------------------------
# c. SessionController unit tests
# ---------------------------------------------------------------------------

class TestSessionControllerUpdateWaveStats:
    def test_writes_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = SessionController()
            ctrl.new_project(name="ws_test", location=tmp)
            seg = WaveSegment(s_lo_mm=1.0, s_hi_mm=5.0)
            config = WaveStatsConfig(
                peak_prominence_k=0.7,
                segments_by_frame={"3": [seg]},
            )
            ctrl.update_wave_stats_config(config)
            ann = ctrl.annotation
            assert ann is not None
            assert ann.wave_stats is not None
            assert ann.wave_stats.peak_prominence_k == pytest.approx(0.7)
            frame_segs = ann.wave_stats.segments_for_frame(3)
            assert len(frame_segs) == 1
            assert frame_segs[0].s_lo_mm == pytest.approx(1.0)

    def test_emits_wave_stats_updated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = SessionController()
            ctrl.new_project(name="ws_sig", location=tmp)
            received: list[bool] = []
            ctrl.wave_stats_updated.connect(lambda: received.append(True))
            ctrl.update_wave_stats_config(WaveStatsConfig(peak_prominence_k=0.4))
            assert len(received) == 1

    def test_noop_without_project(self) -> None:
        ctrl = SessionController()
        ctrl.update_wave_stats_config(WaveStatsConfig(peak_prominence_k=0.4))  # must not raise


class TestSessionControllerSetProfileLine:
    def test_writes_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = SessionController()
            ctrl.new_project(name="pl_test", location=tmp)
            line = ProfileLineData(start=(0.0, 0.0), end=(100.0, 200.0))
            ctrl.set_profile_line(line)
            ann = ctrl.annotation
            assert ann is not None
            assert ann.profile_line is not None
            assert ann.profile_line.end == pytest.approx((100.0, 200.0))

    def test_noop_without_project(self) -> None:
        ctrl = SessionController()
        ctrl.set_profile_line(ProfileLineData(start=(0.0, 0.0), end=(1.0, 1.0)))  # must not raise


class TestSessionControllerRecomputeGuards:
    def test_returns_false_without_project(self) -> None:
        ctrl = SessionController()
        assert ctrl.recompute_wave_stats() is False

    def test_returns_false_without_wave_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = SessionController()
            ctrl.new_project(name="no_ws", location=tmp)
            assert ctrl.recompute_wave_stats() is False

    def test_returns_false_without_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = SessionController()
            ctrl.new_project(name="no_runs", location=tmp)
            ctrl.update_wave_stats_config(WaveStatsConfig(peak_prominence_k=0.3))
            assert ctrl.recompute_wave_stats() is False

    def test_returns_false_for_unknown_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = SessionController()
            ctrl.new_project(name="no_rid", location=tmp)
            ctrl.update_wave_stats_config(WaveStatsConfig(peak_prominence_k=0.3))
            assert ctrl.recompute_wave_stats(run_id="nonexistent") is False
