"""Tests for openfcd/cli/cmd_postprocess.py — postprocess_cmd (v2)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import click
import h5py
import numpy as np
import pytest

from openfcd.cli.cmd_postprocess import _copy_skipping_wave_stats, postprocess_cmd
from openfcd.io.annotation import (
    AnnotationSchema,
    ProfileLineData,
    WaveSegment,
    WaveStatsConfig,
    save as save_annotation,
)
from openfcd.io.result import HDF5ResultStore
from openfcd.io.store import FileSessionStore


def _exit_code(exc) -> int:
    val = exc.value
    if isinstance(val, SystemExit):
        return int(val.code)
    return int(val.exit_code)


# ── Helpers ──────────────────────────────────────────────────────────────────


H, W = 60, 200
PX_PER_MM = 10.0
LAMBDA_MM = 10.0
L_MM = (W - 1) / PX_PER_MM


def _make_eta(shift: float = 0.0) -> np.ndarray:
    cols = np.arange(W, dtype=np.float64)
    s_mm = cols / PX_PER_MM
    eta_row = np.exp(-0.03 * s_mm) * np.sin(2.0 * np.pi * (s_mm - shift) / LAMBDA_MM)
    return np.tile(eta_row, (H, 1))


def _sha256_dataset(h5_path: Path, dataset_path: str) -> str:
    with h5py.File(h5_path, "r") as f:
        return hashlib.sha256(f[dataset_path][:].tobytes()).hexdigest()


def _make_wave_stats_config(k: float = 0.15, frame_ids=(0, 1)) -> WaveStatsConfig:
    # s_mm is centered on the line midpoint: fore = left half, aft = right half.
    half = L_MM / 2.0
    segs = [
        WaveSegment(s_lo_mm=-half, s_hi_mm=0.0, label="fore"),
        WaveSegment(s_lo_mm=0.0, s_hi_mm=half, label="aft"),
    ]
    return WaveStatsConfig(
        segments_by_frame={str(fi): list(segs) for fi in frame_ids},
        peak_prominence_k=k,
    )


def _make_annotation(k: float = 0.15, frame_ids=(0, 1)) -> AnnotationSchema:
    pl = ProfileLineData(start=(H / 2.0, 0.0), end=(H / 2.0, float(W - 1)))
    return AnnotationSchema(
        wave_stats=_make_wave_stats_config(k, frame_ids=frame_ids),
        profile_line=pl,
    )


def _build_project(tmp_path: Path, annotation: AnnotationSchema) -> Path:
    ofcd = tmp_path / "test.ofcd"
    session = FileSessionStore.new(ofcd, "test")

    ann_path = ofcd / "annotations" / "default.json"
    ann_path.parent.mkdir(exist_ok=True)
    save_annotation(ann_path, annotation)

    run_id = "run-20260101-120000"
    run_dir = ofcd / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    h5_path = run_dir / "results.h5"
    store = HDF5ResultStore.open(h5_path, "w")
    store.write_frame("test", 0, _make_eta(0.0), {"status": "ok"})
    store.write_frame("test", 1, _make_eta(0.1), {"status": "ok"})
    store.write_batch(
        "test",
        data={"eta_mean": np.zeros((H, W))},
        meta={"pixel_per_mm_median": PX_PER_MM},
    )
    # Seed wave_stats with v2 layout — a single bogus segment to confirm it gets replaced
    store.write_wave_stats(
        "test",
        [
            {
                "segment_idx": 0,
                "s_lo_mm": 0.0,
                "s_hi_mm": 1.0,
                "label": "stale",
                "color": "#000000",
                "visible": True,
                "wavelength_mm": np.array([1.0, 1.0]),
                "wavenumber_per_mm": np.array([1.0, 1.0]),
                "peaks": {},
                "troughs": {},
                "heights": {},
            }
        ],
        attrs={"schema_version": 2, "prominence_k": 0.15},
    )
    store.close()

    manifest = {
        "run_id": run_id,
        "config_fingerprint": session.config_fingerprint(),
        "status": "done",
    }
    (run_dir / "manifest.yaml").write_text(json.dumps(manifest, indent=2))

    return ofcd


# ── Tests ────────────────────────────────────────────────────────────────────


class TestPostprocessCmdEndToEnd:
    def test_results_h5_still_at_original_path(self, tmp_path):
        ann = _make_annotation(k=0.15)
        ofcd = _build_project(tmp_path, ann)
        h5_path = ofcd / "runs" / "run-20260101-120000" / "results.h5"

        postprocess_cmd(ofcd_path=ofcd, run_id=None)
        assert h5_path.exists()

    def test_new_prominence_k_written(self, tmp_path):
        ofcd = _build_project(tmp_path, _make_annotation(k=0.15))
        ann_path = ofcd / "annotations" / "default.json"
        save_annotation(ann_path, _make_annotation(k=0.5))

        postprocess_cmd(ofcd_path=ofcd, run_id=None)

        h5_path = ofcd / "runs" / "run-20260101-120000" / "results.h5"
        with h5py.File(h5_path, "r") as f:
            assert f["batches/test/wave_stats"].attrs["prominence_k"] == pytest.approx(0.5)

    def test_eta_mean_dataset_unchanged(self, tmp_path):
        ofcd = _build_project(tmp_path, _make_annotation())
        h5_path = ofcd / "runs" / "run-20260101-120000" / "results.h5"
        sha_before = _sha256_dataset(h5_path, "batches/test/eta_mean")

        save_annotation(
            ofcd / "annotations" / "default.json",
            _make_annotation(k=0.5),
        )
        postprocess_cmd(ofcd_path=ofcd, run_id=None)

        sha_after = _sha256_dataset(h5_path, "batches/test/eta_mean")
        assert sha_before == sha_after

    def test_frames_preserved(self, tmp_path):
        ofcd = _build_project(tmp_path, _make_annotation())
        h5_path = ofcd / "runs" / "run-20260101-120000" / "results.h5"
        sha0_before = _sha256_dataset(h5_path, "batches/test/frames/0")
        sha1_before = _sha256_dataset(h5_path, "batches/test/frames/1")

        postprocess_cmd(ofcd_path=ofcd, run_id=None)

        assert _sha256_dataset(h5_path, "batches/test/frames/0") == sha0_before
        assert _sha256_dataset(h5_path, "batches/test/frames/1") == sha1_before

    def test_tmp_file_absent_after_success(self, tmp_path):
        ofcd = _build_project(tmp_path, _make_annotation())
        run_dir = ofcd / "runs" / "run-20260101-120000"
        postprocess_cmd(ofcd_path=ofcd, run_id=None)
        assert not (run_dir / "results.h5.tmp").exists()

    def test_explicit_run_id(self, tmp_path):
        ofcd = _build_project(tmp_path, _make_annotation())
        h5_path = ofcd / "runs" / "run-20260101-120000" / "results.h5"
        postprocess_cmd(ofcd_path=ofcd, run_id="run-20260101-120000")
        assert h5_path.exists()
        with h5py.File(h5_path, "r") as f:
            assert "batches/test/wave_stats" in f

    def test_empty_frame_is_skipped(self, tmp_path):
        """Frame without segments must not get a wave_stats group (v3)."""
        # Only frame 0 has segments; frame 1 should be skipped.
        ann = _make_annotation(frame_ids=(0,))
        ofcd = _build_project(tmp_path, ann)
        h5_path = ofcd / "runs" / "run-20260101-120000" / "results.h5"
        postprocess_cmd(ofcd_path=ofcd, run_id=None)
        with h5py.File(h5_path, "r") as f:
            ws = f["batches/test/wave_stats"]
            assert "frame_0" in ws
            assert "frame_1" not in ws


class TestPostprocessCmdNoWaveStats:
    def test_no_wave_stats_exits_0_h5_untouched(self, tmp_path):
        ann_no_ws = AnnotationSchema()
        ofcd = _build_project(tmp_path, ann_no_ws)
        h5_path = ofcd / "runs" / "run-20260101-120000" / "results.h5"
        sha_before = _sha256_dataset(h5_path, "batches/test/eta_mean")

        with pytest.raises((SystemExit, click.exceptions.Exit)) as exc_info:
            postprocess_cmd(ofcd_path=ofcd, run_id=None)

        assert _exit_code(exc_info) == 0
        sha_after = _sha256_dataset(h5_path, "batches/test/eta_mean")
        assert sha_before == sha_after


class TestPostprocessCmdErrorHandling:
    def test_nonexistent_run_id_exits_nonzero(self, tmp_path):
        ofcd = _build_project(tmp_path, _make_annotation())
        with pytest.raises((SystemExit, click.exceptions.Exit)) as exc_info:
            postprocess_cmd(ofcd_path=ofcd, run_id="nonexistent-run")
        assert _exit_code(exc_info) != 0

    def test_copy_error_leaves_original_intact(self, tmp_path):
        ofcd = _build_project(tmp_path, _make_annotation())
        h5_path = ofcd / "runs" / "run-20260101-120000" / "results.h5"
        sha_before = _sha256_dataset(h5_path, "batches/test/eta_mean")

        with patch(
            "openfcd.cli.cmd_postprocess._copy_skipping_wave_stats",
            side_effect=RuntimeError("simulated copy failure"),
        ):
            with pytest.raises((RuntimeError, SystemExit)):
                postprocess_cmd(ofcd_path=ofcd, run_id=None)

        assert h5_path.exists()
        assert _sha256_dataset(h5_path, "batches/test/eta_mean") == sha_before


class TestOpenExistingForAppend:
    def test_append_mode_preserves_existing_data(self, tmp_path):
        h5_path = tmp_path / "test.h5"
        with HDF5ResultStore.open(h5_path, "w") as s:
            s.write_frame("b0", 0, np.ones((4, 4)), {"status": "ok"})

        store = HDF5ResultStore.open_existing_for_append(h5_path)
        seg_payload = {
            "segment_idx": 0,
            "s_lo_mm": 0.0,
            "s_hi_mm": 5.0,
            "label": "x",
            "color": "#1f77b4",
            "visible": True,
            "wavelength_mm": np.array([1.0]),
            "wavenumber_per_mm": np.array([1.0]),
            "peaks": {0: np.array([[1.0, 2.0]])},
            "troughs": {},
            "heights": {},
        }
        store.write_wave_stats("b0", [seg_payload], {})
        store.close()

        with h5py.File(h5_path, "r") as f:
            assert "batches/b0/frames/0" in f
            assert "batches/b0/wave_stats/segments/0000" in f

    def test_close_does_not_rename_or_delete(self, tmp_path):
        h5_path = tmp_path / "original.h5"
        with HDF5ResultStore.open(h5_path, "w") as s:
            s.write_frame("b0", 0, np.ones((3, 3)), {"status": "ok"})

        store = HDF5ResultStore.open_existing_for_append(h5_path)
        store.close()

        assert h5_path.exists()
        assert not (tmp_path / "original.h5.tmp").exists()
