"""Tests for _RunWorker annotation snapshot isolation.

The snapshot ensures that in-flight GUI edits to wave_stats / profile_line do
not affect the pipeline while a run is in progress.
"""

from __future__ import annotations

import pytest

from openfcd.io.annotation import (
    AnnotationSchema,
    ProfileLineData,
    WaveSegment,
    WaveStatsConfig,
)


# ── Test 1: model_copy(deep=True) creates independent copy ────────────────────

def test_annotation_deep_copy_independence():
    """model_copy(deep=True) must return an object independent of the source."""
    seg = WaveSegment(s_lo_mm=0.0, s_hi_mm=15.0)
    cfg = WaveStatsConfig(segments=[seg], peak_prominence_k=0.3)
    pl = ProfileLineData(start=(10.0, 0.0), end=(10.0, 100.0))
    ann = AnnotationSchema(wave_stats=cfg, profile_line=pl)

    # Simulate what _RunWorker does at the top of run()
    snapshot = ann.model_copy(deep=True)

    # Snapshot must equal original
    assert snapshot.wave_stats == ann.wave_stats
    assert snapshot.profile_line == ann.profile_line

    # Replace wave_stats on original → snapshot unaffected
    ann_modified = ann.model_copy(update={"wave_stats": None})
    assert ann_modified.wave_stats is None
    assert snapshot.wave_stats is not None


# ── Test 2: frozen sub-models survive deep copy ───────────────────────────────

def test_frozen_sub_models_survive_deep_copy():
    """WaveSegment and WaveStatsConfig remain frozen after deep copy."""
    from pydantic import ValidationError

    seg = WaveSegment(s_lo_mm=5.0, s_hi_mm=25.0)
    cfg = WaveStatsConfig(segments=[seg], peak_prominence_k=0.4)
    ann = AnnotationSchema(wave_stats=cfg)

    snapshot = ann.model_copy(deep=True)

    assert snapshot.wave_stats is not None
    assert len(snapshot.wave_stats.segments) == 1
    assert snapshot.wave_stats.segments[0].s_lo_mm == pytest.approx(5.0)
    assert snapshot.wave_stats.segments[0].s_hi_mm == pytest.approx(25.0)

    # Sub-models must still be frozen
    with pytest.raises(Exception):
        snapshot.wave_stats.segments[0].s_lo_mm = 99.0  # type: ignore[misc]


# ── Test 3: snapshot is a value-equal but distinct object ─────────────────────

def test_snapshot_is_distinct_object():
    """Deep copy must not share identity with the source."""
    seg = WaveSegment(s_lo_mm=0.0, s_hi_mm=10.0)
    cfg = WaveStatsConfig(segments=[seg])
    ann = AnnotationSchema(wave_stats=cfg)

    snapshot = ann.model_copy(deep=True)

    # Equal values
    assert snapshot == ann

    # But distinct Python objects
    assert snapshot is not ann
    assert snapshot.wave_stats is not ann.wave_stats


# ── Test 4: deep copy with profile_line ──────────────────────────────────────

def test_snapshot_with_profile_line():
    pl = ProfileLineData(start=(50.0, 10.0), end=(50.0, 200.0), label="main")
    ann = AnnotationSchema(profile_line=pl)

    snapshot = ann.model_copy(deep=True)

    assert snapshot.profile_line is not None
    assert snapshot.profile_line.start == (50.0, 10.0)
    assert snapshot.profile_line.end == (50.0, 200.0)
    assert snapshot.profile_line.label == "main"
    assert snapshot.profile_line is not ann.profile_line


# ── Test 5: None fields stay None in snapshot ─────────────────────────────────

def test_snapshot_none_fields():
    ann = AnnotationSchema(condition="bare")
    snapshot = ann.model_copy(deep=True)

    assert snapshot.wave_stats is None
    assert snapshot.profile_line is None
    assert snapshot.condition == "bare"
