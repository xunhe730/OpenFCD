"""Tests for workers parsing in MainWindow._on_run()."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from skimage.io import imsave

from openfcd.cli.cmd_run import ComputeStage, PreprocessStage
from openfcd.gui.mainwindow import MainWindow, _parse_workers_value
from openfcd.io.result import HDF5ResultStore
from openfcd.io.project import ProjectModel
from openfcd.pipeline.base import CancelledError, CancelToken


def test_parse_workers_auto() -> None:
    """'auto' should resolve to -1 (automatic worker count)."""
    assert _parse_workers_value("auto") == -1


def test_parse_workers_auto_uppercase() -> None:
    """'AUTO' should also resolve to -1 (case-insensitive)."""
    assert _parse_workers_value("AUTO") == -1


def test_parse_workers_empty() -> None:
    """Empty string should resolve to -1 (default auto)."""
    assert _parse_workers_value("") == -1


def test_parse_workers_whitespace() -> None:
    """Whitespace-padded 'auto' should resolve to -1."""
    assert _parse_workers_value("  auto  ") == -1


def test_parse_workers_integer() -> None:
    """Integer string should resolve to that integer."""
    assert _parse_workers_value("4") == 4


def test_parse_workers_integer_padded() -> None:
    """Whitespace-padded integer should resolve correctly."""
    assert _parse_workers_value("  8  ") == 8


def test_parse_workers_zero() -> None:
    """'0' should resolve to 0 (explicit zero workers)."""
    assert _parse_workers_value("0") == 0


def test_parse_workers_negative() -> None:
    """Negative integer should parse (caller decides validity)."""
    assert _parse_workers_value("-2") == -2


def test_parse_workers_invalid_alpha() -> None:
    """Non-numeric string should return None."""
    assert _parse_workers_value("abc") is None


def test_parse_workers_invalid_mixed() -> None:
    """Mixed alpha-numeric string should return None."""
    assert _parse_workers_value("4workers") is None


def test_parse_workers_invalid_float() -> None:
    """Float string should return None (only integers accepted)."""
    assert _parse_workers_value("4.5") is None


def _checkerboard(size: int, shift: float = 0.0) -> np.ndarray:
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    arr = np.sin(2 * np.pi * (x + shift) / 8) * np.sin(2 * np.pi * y / 8)
    return ((arr * 50) + 128).astype(np.uint8)


def _make_project(tmp_path: Path, n_frames: int = 5) -> tuple[ProjectModel, Path, list[Path]]:
    project_dir = tmp_path / "override.ofcd"
    project_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tmp_path / "frames"
    img_dir.mkdir(parents=True, exist_ok=True)

    imsave(str(img_dir / "ref.png"), _checkerboard(64))
    frames: list[Path] = []
    for i in range(n_frames):
        path = img_dir / f"frame_{i:03d}.png"
        imsave(str(path), _checkerboard(64, shift=0.1 * i))
        frames.append(path)

    proj = ProjectModel(
        format_version=0,
        name="override",
        created=datetime.now(timezone.utc).isoformat(),
        geometry={
            "pattern_period_mm": 1.2,
            "optical_stack": {
                "preset": "pattern_below_window",
                "layers": [
                    {"thickness_mm": 3.0, "medium": "glass", "n": 1.5},
                    {"thickness_mm": 12.0, "medium": "water", "n": 1.333},
                ],
            },
        },
        data={"frames_dir": str(img_dir), "pattern": "frame_*.png"},
        reference={"mode": "use_existing", "source": "ref.png"},
    )
    return proj, project_dir, frames


def test_preprocess_stage_uses_frame_paths_override(tmp_path: Path) -> None:
    """GUI Run can pass the currently imported frame list instead of scanning all files."""
    proj, project_dir, frames = _make_project(tmp_path, n_frames=5)
    override = [frames[1], frames[3]]
    ctx = {
        "project": proj,
        "project_dir": project_dir,
        "frames_dir": Path(proj.data.frames_dir),
        "pattern": proj.data.pattern,
        "frames_filter": None,
        "run_id": "run-override",
        "frame_paths_override": override,
    }

    for _event in PreprocessStage().run(ctx, CancelToken()):
        pass

    assert ctx["frame_paths"] == override
    assert ctx["frame_count"] == 2


def test_compute_stage_preserves_gui_frame_ids_for_disabled_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabled GUI frames are skipped without renumbering HDF5 frame IDs."""
    from openfcd.pipeline.frame import FrameResult

    proj, project_dir, frames = _make_project(tmp_path, n_frames=3)
    h5_path = project_dir / "results.h5"

    def _fake_compute_frame(inputs, *, progress_cb=None, cancel=None):
        eta = np.zeros(inputs.ref_shape, dtype=np.float64)
        return FrameResult(eta_mm=eta, qc_datasets=None, diagnostics={})

    monkeypatch.setattr("openfcd.pipeline.frame.compute_frame", _fake_compute_frame)
    result_store = HDF5ResultStore.open(h5_path, "w")
    ctx = {
        "project": proj,
        "project_dir": project_dir,
        "frames_dir": Path(proj.data.frames_dir),
        "pattern": proj.data.pattern,
        "frames_filter": None,
        "run_id": "run-disabled",
        "frame_paths_override": frames,
        "result_store": result_store,
        "batches_processed": [],
        "frame_count": 0,
        "frame_paths": [],
        "annotation": None,
        "disabled_frame_indices": frozenset({1}),
    }
    try:
        for stage in (PreprocessStage(), ComputeStage()):
            for _event in stage.run(ctx, CancelToken()):
                pass
    finally:
        result_store.close()

    with HDF5ResultStore.open(h5_path, "r") as store:
        assert store.list_frames("default") == [0, 2]
        assert store.read_frame_attrs("default", 0)["frame_path"] == "frame_000.png"
        assert store.read_frame_attrs("default", 2)["frame_path"] == "frame_002.png"


def test_compute_stage_propagates_in_frame_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancel during compute_frame should stop the run, not become a frame error."""
    proj, project_dir, frames = _make_project(tmp_path, n_frames=1)

    def _fake_compute_frame(inputs, *, progress_cb=None, cancel=None):
        if cancel is not None:
            cancel.cancel()
            cancel.check()

    monkeypatch.setattr("openfcd.pipeline.frame.compute_frame", _fake_compute_frame)

    ctx = {
        "project": proj,
        "project_dir": project_dir,
        "frames_dir": Path(proj.data.frames_dir),
        "pattern": proj.data.pattern,
        "frames_filter": None,
        "run_id": "run-cancel",
        "frame_paths_override": frames,
        "result_store": None,
        "batches_processed": [],
        "frame_count": 0,
        "frame_paths": [],
        "annotation": None,
    }
    cancel = CancelToken()
    for _event in PreprocessStage().run(ctx, cancel):
        pass

    with pytest.raises(CancelledError):
        for _event in ComputeStage().run(ctx, cancel):
            pass


def test_run_finished_maps_results_by_frame_path(tmp_path: Path) -> None:
    """GUI cache must follow HDF5 frame_path attrs, not numeric dataset order."""
    project_dir = tmp_path / "map.ofcd"
    run_dir = project_dir / "runs" / "run-latest"
    run_dir.mkdir(parents=True)
    h5_path = run_dir / "results.h5"

    arr_a = np.full((4, 4), 1.0)
    arr_ref = np.zeros((4, 4))
    arr_b = np.full((4, 4), 2.0)
    with HDF5ResultStore.open(h5_path, "w") as store:
        store.write_frame("default", 0, arr_b, {"status": "ok", "frame_path": "b.png"})
        store.write_frame("default", 1, arr_a, {"status": "ok", "frame_path": "a.png"})
        store.write_frame("default", 2, arr_ref, {"status": "ok", "frame_path": "ref.png"})

    class _Session:
        project_path = project_dir
        scenes = []

        def refresh_scenes(self, _run_id: str) -> None:
            pass

    class _SimTree:
        def populate_scenes(self, *_args, **_kwargs) -> None:
            pass

    class _RmsView:
        _cached_spec_id = None
        _cached_rms = None

    class _SceneContainer:
        _rms_view = _RmsView()

    class _StatusBar:
        def hide_progress(self) -> None:
            pass

        def set_items(self, *_args, **_kwargs) -> None:
            pass

    class _Toolbar:
        def set_running(self, *_args, **_kwargs) -> None:
            pass

        def set_preview_available(self, *_args, **_kwargs) -> None:
            pass

        def set_preview_on(self, *_args, **_kwargs) -> None:
            pass

        def preview_on(self) -> bool:
            return True

        def overlap_on(self) -> bool:
            return True

    class _Preview:
        def __init__(self) -> None:
            self.shown = None
            self.cleared = False

        def set_preview_toggle_visible(self, *_args, **_kwargs) -> None:
            pass

        def set_preview_mode(self, *_args, **_kwargs) -> None:
            pass

        def show_eta_overlay(self, eta, **_kwargs) -> None:
            self.shown = np.asarray(eta)

        def clear_eta_overlay(self) -> None:
            self.cleared = True

    class _Stack:
        def setCurrentWidget(self, *_args, **_kwargs) -> None:
            pass

    window = MainWindow.__new__(MainWindow)
    window._session = _Session()
    window._sim_tree = _SimTree()
    window._scene_container = _SceneContainer()
    window._status_bar = _StatusBar()
    window._toolbar = _Toolbar()
    window._preview = _Preview()
    window._center_stack = _Stack()
    window._frames = [Path("a.png"), Path("b.png"), Path("ref.png")]
    window._frame_eta_cache = {}
    window._run_eta_frames = None
    window._run_eta_mean = None
    window._current_frame_idx = 2

    window._on_run_finished("run-latest")

    np.testing.assert_array_equal(window._frame_eta_cache[0], arr_a)
    np.testing.assert_array_equal(window._frame_eta_cache[1], arr_b)
    np.testing.assert_array_equal(window._frame_eta_cache[2], arr_ref)
    assert np.nanmax(np.abs(window._preview.shown)) == 0.0
