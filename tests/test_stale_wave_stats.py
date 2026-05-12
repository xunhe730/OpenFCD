"""AC-10 — annotation.wave_stats changes must NOT trigger STALE (v2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from openfcd.io.annotation import (
    AnnotationSchema,
    ProfileLineData,
    WaveSegment,
    WaveStatsConfig,
)
from openfcd.io.store import FileSessionStore


def _minimal_store(tmp_path: Path) -> tuple[FileSessionStore, str]:
    ofcd = tmp_path / "proj.ofcd"
    store = FileSessionStore.new(ofcd, "proj")
    run_id = "run-stale-test-0001"
    store.record_run(run_id, {"status": "ok", "frames": 5})
    return store, run_id


def _ws(segments: list[WaveSegment], k: float = 0.3) -> WaveStatsConfig:
    return WaveStatsConfig(segments=segments, peak_prominence_k=k)


def test_fresh_run_not_stale(tmp_path):
    store, run_id = _minimal_store(tmp_path)
    assert not store.is_stale(run_id)


def test_wave_stats_prominence_change_not_stale(tmp_path):
    store, run_id = _minimal_store(tmp_path)

    ws_v1 = _ws(
        [
            WaveSegment(s_lo_mm=5.0, s_hi_mm=30.0),
            WaveSegment(s_lo_mm=60.0, s_hi_mm=95.0),
        ],
        k=0.3,
    )
    store._annotation = AnnotationSchema(wave_stats=ws_v1)
    assert not store.is_stale(run_id)

    ws_v2 = _ws(
        [
            WaveSegment(s_lo_mm=5.0, s_hi_mm=30.0),
            WaveSegment(s_lo_mm=60.0, s_hi_mm=95.0),
        ],
        k=0.5,
    )
    store._annotation = AnnotationSchema(wave_stats=ws_v2)
    assert not store.is_stale(run_id)


def test_wave_stats_segment_bounds_change_not_stale(tmp_path):
    store, run_id = _minimal_store(tmp_path)

    ws_v1 = _ws(
        [
            WaveSegment(s_lo_mm=5.0, s_hi_mm=30.0),
            WaveSegment(s_lo_mm=60.0, s_hi_mm=95.0),
        ]
    )
    store._annotation = AnnotationSchema(wave_stats=ws_v1)

    ws_v2 = _ws(
        [
            WaveSegment(s_lo_mm=2.0, s_hi_mm=40.0),
            WaveSegment(s_lo_mm=65.0, s_hi_mm=90.0),
        ]
    )
    store._annotation = AnnotationSchema(wave_stats=ws_v2)
    assert not store.is_stale(run_id)


def test_wave_stats_add_segment_not_stale(tmp_path):
    store, run_id = _minimal_store(tmp_path)

    ws_one = _ws([WaveSegment(s_lo_mm=5.0, s_hi_mm=30.0)])
    store._annotation = AnnotationSchema(wave_stats=ws_one)
    assert not store.is_stale(run_id)

    ws_two = _ws(
        [
            WaveSegment(s_lo_mm=5.0, s_hi_mm=30.0),
            WaveSegment(s_lo_mm=60.0, s_hi_mm=95.0),
        ]
    )
    store._annotation = AnnotationSchema(wave_stats=ws_two)
    assert not store.is_stale(run_id)


def test_profile_line_move_not_stale(tmp_path):
    store, run_id = _minimal_store(tmp_path)
    pl_before = ProfileLineData(start=(50.0, 0.0), end=(50.0, 299.0))
    store._annotation = AnnotationSchema(profile_line=pl_before)
    assert not store.is_stale(run_id)
    pl_after = ProfileLineData(start=(60.0, 0.0), end=(60.0, 299.0))
    store._annotation = AnnotationSchema(profile_line=pl_after)
    assert not store.is_stale(run_id)


def test_wave_stats_none_to_configured_not_stale(tmp_path):
    store, run_id = _minimal_store(tmp_path)
    store._annotation = AnnotationSchema()
    assert not store.is_stale(run_id)
    store._annotation = AnnotationSchema(
        wave_stats=_ws([WaveSegment(s_lo_mm=5.0, s_hi_mm=30.0)])
    )
    assert not store.is_stale(run_id)


def test_geometry_pattern_period_change_triggers_stale(tmp_path):
    store, run_id = _minimal_store(tmp_path)
    assert not store.is_stale(run_id)

    from openfcd.io.project import GeometryConfig
    new_geom = GeometryConfig.model_validate(
        {**store._project.geometry.model_dump(), "pattern_period_mm": 2.5}
    )
    store._project = store._project.model_copy(update={"geometry": new_geom})
    assert store.is_stale(run_id)


def test_process_flatfield_change_triggers_stale(tmp_path):
    store, run_id = _minimal_store(tmp_path)
    assert not store.is_stale(run_id)
    new_process = store._project.process.model_copy(update={"flatfield_sigma": 9999.0})
    store._project = store._project.model_copy(update={"process": new_process})
    assert store.is_stale(run_id)


def test_unknown_run_id_reports_stale(tmp_path):
    store, _ = _minimal_store(tmp_path)
    assert store.is_stale("run-that-was-never-recorded")


def test_fingerprint_unchanged_by_annotation(tmp_path):
    store, _ = _minimal_store(tmp_path)
    fp_before = store.config_fingerprint()
    store._annotation = AnnotationSchema(
        wave_stats=_ws([WaveSegment(s_lo_mm=10.0, s_hi_mm=50.0)], k=0.8)
    )
    fp_after = store.config_fingerprint()
    assert fp_before == fp_after
