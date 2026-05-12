"""Tests for WaveSegment / WaveStatsConfig / ProfileLineData (v2).

v2 schema: arbitrary N wave segments (no fore/aft binary).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from openfcd.io.annotation import (
    AnnotationSchema,
    ProfileLineData,
    WaveSegment,
    WaveStatsConfig,
    load,
    save,
)


# ── WaveSegment ─────────────────────────────────────────────────────────────


class TestWaveSegment:
    def test_valid_construction(self):
        s = WaveSegment(s_lo_mm=12.5, s_hi_mm=38.0)
        assert s.s_lo_mm == 12.5
        assert s.s_hi_mm == 38.0
        assert s.label == ""
        assert s.color == "#1f77b4"
        assert s.visible is True

    def test_custom_label_color_visible(self):
        s = WaveSegment(
            s_lo_mm=0.0, s_hi_mm=10.0, label="A", color="#ff0000", visible=False
        )
        assert s.label == "A"
        assert s.color == "#ff0000"
        assert s.visible is False

    def test_hi_must_exceed_lo_equal_raises(self):
        with pytest.raises(ValidationError):
            WaveSegment(s_lo_mm=10.0, s_hi_mm=10.0)

    def test_hi_less_than_lo_raises(self):
        with pytest.raises(ValidationError):
            WaveSegment(s_lo_mm=20.0, s_hi_mm=5.0)

    def test_frozen_immutable(self):
        s = WaveSegment(s_lo_mm=0.0, s_hi_mm=10.0)
        with pytest.raises(Exception):
            s.s_lo_mm = 99.0  # type: ignore[misc]

    def test_round_trip_json(self):
        s = WaveSegment(s_lo_mm=1.5, s_hi_mm=99.5, label="seg1", color="#aabbcc")
        data = s.model_dump()
        s2 = WaveSegment.model_validate(data)
        assert s2 == s


# ── WaveStatsConfig ──────────────────────────────────────────────────────────


class TestWaveStatsConfig:
    def test_defaults_empty(self):
        cfg = WaveStatsConfig()
        assert cfg.segments == []
        assert cfg.peak_prominence_k == pytest.approx(0.3)

    def test_with_single_segment(self):
        s = WaveSegment(s_lo_mm=0.0, s_hi_mm=20.0)
        cfg = WaveStatsConfig(segments=[s], peak_prominence_k=0.5)
        assert len(cfg.segments) == 1
        assert cfg.segments[0] == s
        assert cfg.peak_prominence_k == pytest.approx(0.5)

    def test_with_multiple_segments(self):
        segs = [
            WaveSegment(s_lo_mm=0.0, s_hi_mm=15.0, label="a"),
            WaveSegment(s_lo_mm=20.0, s_hi_mm=40.0, label="b"),
            WaveSegment(s_lo_mm=45.0, s_hi_mm=60.0, label="c"),
        ]
        cfg = WaveStatsConfig(segments=segs)
        assert len(cfg.segments) == 3
        assert [s.label for s in cfg.segments] == ["a", "b", "c"]

    def test_peak_prominence_k_lower_bound(self):
        with pytest.raises(ValidationError):
            WaveStatsConfig(peak_prominence_k=0.01)

    def test_peak_prominence_k_upper_bound(self):
        with pytest.raises(ValidationError):
            WaveStatsConfig(peak_prominence_k=3.0)

    def test_peak_prominence_k_boundary_valid(self):
        assert WaveStatsConfig(peak_prominence_k=0.05).peak_prominence_k == pytest.approx(0.05)
        assert WaveStatsConfig(peak_prominence_k=2.0).peak_prominence_k == pytest.approx(2.0)

    def test_frozen_immutable(self):
        cfg = WaveStatsConfig()
        with pytest.raises(Exception):
            cfg.peak_prominence_k = 0.8  # type: ignore[misc]

    def test_round_trip_json(self):
        segs = [
            WaveSegment(s_lo_mm=0.0, s_hi_mm=20.0, label="x", color="#112233"),
            WaveSegment(s_lo_mm=30.0, s_hi_mm=60.0, visible=False),
        ]
        cfg = WaveStatsConfig(segments=segs, peak_prominence_k=0.5)
        data = cfg.model_dump()
        cfg2 = WaveStatsConfig.model_validate(data)
        assert cfg2 == cfg


# ── ProfileLineData (unchanged in v2) ────────────────────────────────────────


class TestProfileLineData:
    def test_basic_construction(self):
        pl = ProfileLineData(start=(10.0, 20.0), end=(30.0, 40.0))
        assert pl.start == (10.0, 20.0)
        assert pl.end == (30.0, 40.0)
        assert pl.label == "profile"

    def test_custom_label(self):
        pl = ProfileLineData(start=(0.0, 0.0), end=(100.0, 200.0), label="fore-aft")
        assert pl.label == "fore-aft"

    def test_list_input_coerced_to_tuple(self):
        pl = ProfileLineData.model_validate({"start": [10.0, 20.0], "end": [30.0, 40.0]})
        assert isinstance(pl.start, tuple)
        assert pl.start == (10.0, 20.0)

    def test_frozen_immutable(self):
        pl = ProfileLineData(start=(0.0, 0.0), end=(1.0, 1.0))
        with pytest.raises(Exception):
            pl.label = "new"  # type: ignore[misc]

    def test_round_trip_json(self):
        pl = ProfileLineData(start=(5.0, 15.0), end=(95.0, 175.0), label="test")
        data = pl.model_dump()
        pl2 = ProfileLineData.model_validate(data)
        assert pl2.start == pl.start
        assert pl2.end == pl.end
        assert pl2.label == pl.label


# ── AnnotationSchema integration ─────────────────────────────────────────────


class TestAnnotationSchemaIntegration:
    def test_wave_stats_defaults_none(self):
        ann = AnnotationSchema()
        assert ann.wave_stats is None

    def test_profile_line_defaults_none(self):
        ann = AnnotationSchema()
        assert ann.profile_line is None

    def test_old_json_no_wave_stats_loads_cleanly(self):
        old_data = {
            "condition": "legacy",
            "polygons": [],
            "roi": {"x": 0, "y": 0, "width": 0, "height": 0},
        }
        ann = AnnotationSchema.model_validate(old_data)
        assert ann.condition == "legacy"
        assert ann.wave_stats is None
        assert ann.profile_line is None

    def test_extra_fields_ignored(self):
        ann = AnnotationSchema.model_validate({"condition": "test", "unknown_key": 99})
        assert ann.condition == "test"
        assert not hasattr(ann, "unknown_key")

    def test_wave_stats_set(self):
        seg = WaveSegment(s_lo_mm=0.0, s_hi_mm=20.0, label="alpha")
        cfg = WaveStatsConfig(segments=[seg], peak_prominence_k=0.4)
        ann = AnnotationSchema(wave_stats=cfg)
        assert ann.wave_stats is not None
        assert len(ann.wave_stats.segments) == 1
        assert ann.wave_stats.segments[0].label == "alpha"

    def test_profile_line_set(self):
        pl = ProfileLineData(start=(10.0, 5.0), end=(90.0, 200.0), label="main")
        ann = AnnotationSchema(profile_line=pl)
        assert ann.profile_line is not None
        assert ann.profile_line.label == "main"

    def test_round_trip_model_dump(self):
        segs = [
            WaveSegment(s_lo_mm=0.0, s_hi_mm=20.0, color="#aa0000"),
            WaveSegment(s_lo_mm=30.0, s_hi_mm=55.0, label="aft-ish"),
        ]
        cfg = WaveStatsConfig(segments=segs, peak_prominence_k=0.6)
        pl = ProfileLineData(start=(0.0, 25.0), end=(100.0, 25.0), label="centerline")
        ann = AnnotationSchema(wave_stats=cfg, profile_line=pl)

        dumped = ann.model_dump()
        ann2 = AnnotationSchema.model_validate(dumped)

        assert ann2.wave_stats is not None
        assert len(ann2.wave_stats.segments) == 2
        assert ann2.wave_stats.segments[0].color == "#aa0000"
        assert ann2.wave_stats.segments[1].label == "aft-ish"
        assert ann2.wave_stats.peak_prominence_k == pytest.approx(0.6)
        assert ann2.profile_line is not None
        assert ann2.profile_line.start == (0.0, 25.0)

    def test_save_load_round_trip(self, tmp_path):
        seg = WaveSegment(s_lo_mm=1.0, s_hi_mm=18.0, label="s1", color="#abcdef")
        cfg = WaveStatsConfig(segments=[seg], peak_prominence_k=0.5)
        pl = ProfileLineData(start=(1.0, 2.0), end=(3.0, 4.0))
        ann = AnnotationSchema(wave_stats=cfg, profile_line=pl, condition="roundtrip")

        path = tmp_path / "ann.json"
        save(path, ann)
        ann2 = load(path)

        assert ann2.condition == "roundtrip"
        assert ann2.wave_stats is not None
        assert len(ann2.wave_stats.segments) == 1
        assert ann2.wave_stats.segments[0].s_lo_mm == pytest.approx(1.0)
        assert ann2.wave_stats.segments[0].s_hi_mm == pytest.approx(18.0)
        assert ann2.wave_stats.segments[0].label == "s1"
        assert ann2.wave_stats.segments[0].color == "#abcdef"
        assert ann2.wave_stats.peak_prominence_k == pytest.approx(0.5)
        assert ann2.profile_line is not None
        assert ann2.profile_line.start == (1.0, 2.0)

    def test_save_load_none_fields(self, tmp_path):
        ann = AnnotationSchema(condition="no-stats")
        path = tmp_path / "ann_none.json"
        save(path, ann)
        ann2 = load(path)
        assert ann2.wave_stats is None
        assert ann2.profile_line is None
