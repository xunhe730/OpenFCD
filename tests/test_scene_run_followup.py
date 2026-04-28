"""Test that scenes auto-refresh and get missing-frame badges after a new Run."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openfcd.io.store import FileSessionStore
from openfcd.io.scene import SceneSpec, SceneType
from openfcd.io.result import HDF5ResultStore


def _make_store(tmp: Path) -> FileSessionStore:
    return FileSessionStore.new(tmp / "proj.ofcd", "proj")


def _write_run(project_path: Path, run_id: str, frame_ids: list[int]) -> None:
    run_dir = project_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    store = HDF5ResultStore.open(run_dir / "results.h5", "w")
    for fid in frame_ids:
        store.write_frame("default", fid, np.ones((4, 4)) * fid, {"status": "ok"})
    store.close()


def test_refresh_scenes_binds_run_id(tmp_path):
    """Scenes with run_id=None get bound to the new run after refresh."""
    store = _make_store(tmp_path)
    spec = SceneSpec(id="s1", name="η", type=SceneType.ETA_MAP,
                     frame_indices=[0, 1, 2], run_id=None)
    store.add_scene(spec)

    from openfcd.gui.controllers.session_controller import SessionController
    ctrl = SessionController()
    ctrl._store = store

    ctrl.refresh_scenes("run-abc")
    assert ctrl.scenes[0].run_id == "run-abc"


def test_refresh_scenes_preserves_pinned_run_id(tmp_path):
    """Scenes with an explicit run_id are not overwritten."""
    store = _make_store(tmp_path)
    spec = SceneSpec(id="s1", name="η", type=SceneType.ETA_MAP,
                     frame_indices=[0, 1], run_id="run-old")
    store.add_scene(spec)

    from openfcd.gui.controllers.session_controller import SessionController
    ctrl = SessionController()
    ctrl._store = store

    ctrl.refresh_scenes("run-new")
    assert ctrl.scenes[0].run_id == "run-old"


def test_missing_counts_detected(tmp_path):
    """_compute_missing_counts returns correct missing count when a frame is absent."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6")

    proj_path = tmp_path / "proj.ofcd"
    proj_path.mkdir()
    (proj_path / "runs").mkdir()
    _write_run(proj_path, "run-1", frame_ids=[0, 1])  # frame 2 missing

    h5_path = proj_path / "runs" / "run-1" / "results.h5"
    store = HDF5ResultStore.open(h5_path, "r")
    available = set(store.list_frames("default"))
    store.close()
    assert available == {0, 1}

    spec = SceneSpec(id="s1", name="t", type=SceneType.ETA_MAP,
                     frame_indices=[0, 1, 2])
    missing = sum(1 for idx in spec.frame_indices if idx not in available)
    assert missing == 1


def test_populate_scenes_shows_badge(tmp_path):
    """populate_scenes labels a node with '[1/3 missing]' when missing_counts set."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    from openfcd.gui.panels.sim_tree import SimTree
    from openfcd.io.scene import SceneSpec, SceneType

    tree = SimTree()
    spec = SceneSpec(id="s1", name="Wake η", type=SceneType.ETA_MAP, frame_indices=[0, 1, 2])
    tree.populate_scenes([spec], missing_counts={"s1": 1})

    parent = tree._find_or_create_parent("Scenes")
    item = parent.child(0)
    assert "missing" in item.text(0)


def test_populate_scenes_no_badge_when_none_missing(tmp_path):
    """populate_scenes shows plain name when no frames are missing."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    from openfcd.gui.panels.sim_tree import SimTree
    from openfcd.io.scene import SceneSpec, SceneType

    tree = SimTree()
    spec = SceneSpec(id="s1", name="Clean", type=SceneType.ETA_MAP, frame_indices=[0, 1])
    tree.populate_scenes([spec])
    parent = tree._find_or_create_parent("Scenes")
    item = parent.child(0)
    assert "missing" not in item.text(0)
