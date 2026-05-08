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
def rms_view(qapp):
    from openfcd.gui.scenes.rms_scene import RmsSceneView
    w = RmsSceneView()
    yield w
    # Wait for any background worker to finish before closing
    if w._worker is not None:
        w._worker.quit()
        w._worker.wait(3000)
    w.close()


def test_rms_scene_view_loads_without_h5(rms_view, tmp_path):
    from openfcd.io.scene import SceneSpec, SceneType

    spec = SceneSpec(id="r1", name="wake RMS", type=SceneType.RMS, frame_indices=[0, 1, 2])
    rms_view.load(spec, tmp_path)
    # Worker will emit result_ready(None) since no h5 exists; wait for it
    if rms_view._worker is not None:
        rms_view._worker.wait(3000)
    assert "wake RMS" in rms_view._title.text()


def test_rms_worker_computes_correctly(tmp_path):
    from openfcd.gui.scenes.rms_scene import _RmsWorker
    from openfcd.io.result import HDF5ResultStore
    from openfcd.io.scene import SceneSpec, SceneType

    # Write 3 frames of constant value 2.0
    run_dir = tmp_path / "runs" / "run-rms"
    run_dir.mkdir(parents=True)
    store = HDF5ResultStore.open(run_dir / "results.h5", "w")
    for i in range(3):
        store.write_frame("default", i, np.full((8, 8), 2.0), {"status": "ok"})
    store.close()

    spec = SceneSpec(
        id="r2",
        name="t",
        type=SceneType.RMS,
        frame_indices=[0, 1, 2],
        run_id="run-rms",
    )
    received = []
    worker = _RmsWorker(tmp_path, spec, None)
    worker.result_ready.connect(received.append)
    worker.run()  # direct call for testing
    assert len(received) == 1
    rms = received[0]
    assert rms is not None
    np.testing.assert_allclose(rms, 2.0, atol=1e-10)


def test_rms_nan_frames_handled(tmp_path):
    from openfcd.gui.scenes.rms_scene import _RmsWorker
    from openfcd.io.result import HDF5ResultStore
    from openfcd.io.scene import SceneSpec, SceneType

    run_dir = tmp_path / "runs" / "run-nan"
    run_dir.mkdir(parents=True)
    store = HDF5ResultStore.open(run_dir / "results.h5", "w")
    arr = np.ones((4, 4))
    arr[0, 0] = np.nan
    store.write_frame("default", 0, arr, {"status": "ok"})
    store.close()

    spec = SceneSpec(
        id="r3",
        name="t",
        type=SceneType.RMS,
        frame_indices=[0],
        run_id="run-nan",
    )
    received = []
    worker = _RmsWorker(tmp_path, spec, None)
    worker.result_ready.connect(received.append)
    worker.run()
    rms = received[0]
    assert rms is not None
    assert np.isnan(rms[0, 0])
    np.testing.assert_allclose(rms[1:, 1:], 1.0, atol=1e-10)


def test_rms_replaced_worker_does_not_update_current_scene(rms_view):
    rms_view._generation = 2
    rms_view._cached_rms = None
    stale = np.ones((3, 3))
    current = np.ones((3, 3)) * 2
    rms_view._on_result(1, stale)
    assert rms_view._cached_rms is None
    rms_view._on_result(2, current)
    np.testing.assert_allclose(rms_view._cached_rms, current)
