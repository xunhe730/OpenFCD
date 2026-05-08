from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from openfcd.io.store import FileSessionStore
from openfcd.io.result import HDF5ResultStore
from openfcd.io.image import scan_frames, parse_frame_number, load_frame
from openfcd.pipeline.runner import PipelineRunner
from openfcd.pipeline.base import StageEvent


def test_session_store_new_creates_structure(tmp_path: Path) -> None:
    proj_dir = tmp_path / "demo.ofcd"
    FileSessionStore.new(proj_dir, name="demo")
    assert (proj_dir / "project.yaml").exists()
    assert (proj_dir / "session.json").exists()
    for sub in ("images", "annotations", "runs", "autosave"):
        assert (proj_dir / sub).is_dir()


def test_session_store_open_reads_project(tmp_path: Path) -> None:
    proj_dir = tmp_path / "demo.ofcd"
    s1 = FileSessionStore.new(proj_dir, name="demo_proj")
    s1.close()
    s2 = FileSessionStore.open(proj_dir)
    assert s2.project.name == "demo_proj"
    s2.close()


def test_config_fingerprint_changes_with_geometry(tmp_path: Path) -> None:
    proj_dir = tmp_path / "demo.ofcd"
    store = FileSessionStore.new(proj_dir, name="demo")
    fp1 = store.config_fingerprint()
    store.project.geometry.pattern_period_mm = 2.4
    fp2 = store.config_fingerprint()
    assert fp1 != fp2


def test_session_store_is_stale(tmp_path: Path) -> None:
    proj_dir = tmp_path / "demo.ofcd"
    store = FileSessionStore.new(proj_dir, name="demo")
    store.record_run("run_001", {"created": "2026-04-22T00:00:00Z"})
    assert store.is_stale("run_001") is False
    store.project.geometry.pattern_period_mm = 9.9
    assert store.is_stale("run_001") is True


def test_result_store_write_read_batch(tmp_path: Path) -> None:
    h5_path = tmp_path / "results.h5"
    eta = np.random.randn(8, 8).astype(np.float64)
    with HDF5ResultStore.open(h5_path, mode="w") as store:
        store.write_batch("b1", {"eta_mean": eta}, {"frames": 10})
    with HDF5ResultStore.open(h5_path, mode="r") as store:
        out = store.read_summary("b1", "eta_mean")
        assert np.allclose(out, eta)
        meta = store.read_batch_meta("b1")
        assert meta["frames"] == 10


def test_result_store_write_read_frame(tmp_path: Path) -> None:
    h5_path = tmp_path / "results.h5"
    eta = np.random.randn(4, 4).astype(np.float64)
    with HDF5ResultStore.open(h5_path, mode="w") as store:
        store.write_frame("b1", 42, eta, {"status": "ok"})
    with HDF5ResultStore.open(h5_path, mode="r") as store:
        out = store.read_frame("b1", 42)
        assert np.allclose(out, eta)


def test_result_store_write_read_frame_qc_datasets(tmp_path: Path) -> None:
    h5_path = tmp_path / "results.h5"
    eta = np.ones((4, 4), dtype=np.float64)
    qc = {
        "carrier_amplitude": np.full((4, 4), 2.0),
        "valid_mask": np.ones((4, 4), dtype=bool),
        "artifact_mask": np.zeros((4, 4), dtype=bool),
        "phase_residual": np.zeros((4, 4), dtype=np.float64),
        "poisson_residual": np.zeros((4, 4), dtype=np.float64),
        "poisson_residual_x": np.zeros((4, 4), dtype=np.float64),
        "poisson_residual_y": np.zeros((4, 4), dtype=np.float64),
        "curl_inconsistency": np.zeros((4, 4), dtype=np.float64),
    }
    attrs = {
        "status": "ok",
        "saturated_ratio": 0.125,
        "invalid_ratio": 0.0,
        "poisson_residual_rms": 0.01,
        "curl_inconsistency_rms": 0.02,
    }
    with HDF5ResultStore.open(h5_path, mode="w") as store:
        store.write_frame("b1", 7, eta, attrs, qc_datasets=qc)
    with HDF5ResultStore.open(h5_path, mode="r") as store:
        assert set(qc).issubset(store.list_frame_qc("b1", 7))
        np.testing.assert_allclose(store.read_frame_qc("b1", 7, "carrier_amplitude"), qc["carrier_amplitude"])
        np.testing.assert_array_equal(store.read_frame_qc("b1", 7, "valid_mask"), qc["valid_mask"])
        assert store.read_frame_attrs("b1", 7)["saturated_ratio"] == 0.125
        assert store.read_frame_attrs("b1", 7)["poisson_residual_rms"] == 0.01
        assert store.read_frame_attrs("b1", 7)["curl_inconsistency_rms"] == 0.02


def test_result_store_batch_status(tmp_path: Path) -> None:
    h5_path = tmp_path / "results.h5"
    with HDF5ResultStore.open(h5_path, mode="w") as store:
        store.write_frame("b1", 1, np.zeros((2, 2)), {"status": "ok"})
        store.write_frame("b1", 2, np.zeros((2, 2)), {"status": "ok"})
        store.write_frame("b1", 3, np.zeros((2, 2)), {"status": "error"})
        store.write_frame("b1", 4, np.zeros((2, 2)), {"status": "skipped"})
    with HDF5ResultStore.open(h5_path, mode="r") as store:
        st = store.batch_status("b1")
        assert st == {"total": 4, "done": 2, "error": 1, "skipped": 1}


def test_parse_frame_number() -> None:
    assert parse_frame_number("Img2000606.jpg") == 2000606
    assert parse_frame_number("Img001.jpg") == 1


def test_scan_frames(tmp_path: Path) -> None:
    for n in (3, 1, 2):
        Image.new("L", (4, 4), color=n).save(tmp_path / f"Img00{n}.jpg")
    paths = scan_frames(tmp_path)
    assert len(paths) == 3
    assert [parse_frame_number(p.name) for p in paths] == [1, 2, 3]
    arr = load_frame(paths[0])
    assert arr.shape == (4, 4)
    assert arr.dtype == np.float32


def test_pipeline_runner_serial() -> None:
    class MockStage:
        name = "mock"

        def run(self, ctx, cancel=None):
            yield StageEvent(
                kind="start", stage="mock", batch=None, frame_idx=None,
                substage=None, progress=0.0, total=1, completed=0,
            )
            yield StageEvent(
                kind="finish", stage="mock", batch=None, frame_idx=None,
                substage=None, progress=1.0, total=1, completed=1,
            )

        def dry_run(self, ctx):
            return ["mock action"]

    runner = PipelineRunner([MockStage()])
    events = list(runner.iter({}))
    assert len(events) == 2
    assert events[0].kind == "start"
    assert events[1].kind == "finish"
    actions = runner.dry_run({})
    assert actions == [{"stage": "mock", "action": "mock action"}]
