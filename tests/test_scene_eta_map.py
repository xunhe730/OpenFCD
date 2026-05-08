import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def eta_map_view(qapp):
    from openfcd.gui.scenes.eta_map_scene import EtaMapSceneView
    w = EtaMapSceneView()
    yield w
    w.close()


@pytest.fixture(scope="module")
def scene_container(qapp):
    from openfcd.gui.scenes.scene_container import SceneContainer
    c = SceneContainer()
    yield c
    c.close()


def test_eta_map_scene_view_loads_without_h5(eta_map_view, tmp_path):
    from openfcd.io.scene import SceneSpec, SceneType

    spec = SceneSpec(id="s1", name="test", type=SceneType.ETA_MAP, frame_indices=[0, 1])
    eta_map_view.load(spec, tmp_path)  # no results.h5 → graceful empty state
    assert eta_map_view._slider.maximum() == 0


def test_eta_map_scene_view_slider_range(eta_map_view, tmp_path):
    from openfcd.io.result import HDF5ResultStore
    from openfcd.io.scene import SceneSpec, SceneType

    # Write a small fake results.h5
    run_dir = tmp_path / "runs" / "run-test"
    run_dir.mkdir(parents=True)
    store = HDF5ResultStore.open(run_dir / "results.h5", "w")
    for i in range(3):
        store.write_frame("default", i, np.ones((8, 8)) * i, {"status": "ok"})
    store.close()

    spec = SceneSpec(
        id="s1",
        name="test",
        type=SceneType.ETA_MAP,
        frame_indices=[0, 1, 2],
        run_id="run-test",
    )
    eta_map_view.load(spec, tmp_path)
    assert eta_map_view._slider.maximum() == 2
    # Widget not shown in offscreen tests; check it is not explicitly hidden
    assert not eta_map_view._slider.isHidden()
    assert eta_map_view._title.text() == "test"


def test_eta_map_scene_view_applies_viz_colorbar(eta_map_view, tmp_path):
    from openfcd.io.result import HDF5ResultStore
    from openfcd.io.scene import SceneSpec, SceneType

    run_dir = tmp_path / "runs" / "run-test"
    run_dir.mkdir(parents=True)
    with HDF5ResultStore.open(run_dir / "results.h5", "w") as store:
        arr = np.full((20, 30), np.nan)
        arr[5:15, 8:18] = np.arange(100, dtype=float).reshape(10, 10)
        store.write_frame("default", 0, arr, {"status": "ok"})

    spec = SceneSpec(
        id="s2",
        name="η Map 2",
        type=SceneType.ETA_MAP,
        frame_indices=[0],
        run_id="run-test",
        viz_params={"colorbar": "none", "title": "Wake"},
    )
    eta_map_view.load(spec, tmp_path)

    assert eta_map_view._title.text() == "Wake"
    assert eta_map_view._eta_map._cb is None
    rendered = eta_map_view._eta_map._ax.images[0].get_array()
    assert rendered.shape == (12, 12)


def test_scene_container_routes_eta_map(scene_container, tmp_path):
    from openfcd.gui.scenes.scene_container import SceneContainer
    from openfcd.io.scene import SceneSpec, SceneType

    spec = SceneSpec(id="x", name="η", type=SceneType.ETA_MAP, frame_indices=[0])
    scene_container.show_scene(spec, tmp_path)
    assert scene_container._stack.currentIndex() == SceneContainer._IDX_ETA_MAP


def test_eta_map_empty_clears_previous_image(qapp):
    from openfcd.gui.scenes.eta_map import EtaMap

    w = EtaMap()
    w.set_data(np.ones((4, 4)))
    assert w._im is not None
    w.set_data(np.array([]))
    assert w._im is None
    w.close()
