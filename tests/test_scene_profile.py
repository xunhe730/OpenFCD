import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def profile_view(qapp):
    from openfcd.gui.scenes.profile_scene import ProfileSceneView
    w = ProfileSceneView()
    yield w
    w.close()


@pytest.fixture(scope="module")
def line_annotator(qapp):
    from openfcd.gui.scenes.profile_scene import _LineAnnotator
    a = _LineAnnotator()
    a.resize(200, 200)
    yield a
    a.close()


def _flush_render(view) -> None:
    """Synchronously drain the debounce timer and execute the pending render.

    The render state machine schedules full-figure repaints via a 48ms
    ``QTimer`` (``_RENDER_DEBOUNCE_MS``).  In headless pytest the Qt event
    loop never spins, so the timer callback never fires.  Calling this helper
    stops the timer and invokes ``_do_render()`` directly, flushing whatever
    render was queued by the preceding ``load()``/``apply_viz()``/``refresh()``.
    """
    view._render_timer.stop()
    view._do_render()


def test_profile_scene_view_loads(profile_view, tmp_path):
    from openfcd.io.scene import SceneSpec, SceneType

    spec = SceneSpec(id="p1", name="wake profile", type=SceneType.PROFILE, frame_indices=[0, 1])
    profile_view.load(spec, tmp_path)
    assert "wake profile" in profile_view._title.text()
    assert profile_view._slider_lbl.text() == "frame 0/1"


def test_line_annotator_drag(line_annotator):
    # Reset state before test
    line_annotator.clear_line()
    line_annotator.set_image(np.ones((100, 100)))
    start = QPoint(10, 20)
    end = QPoint(100, 80)
    line_annotator._drawing = True
    line_annotator._line_start = start
    line_annotator._line_end = end
    line_annotator.mouseReleaseEvent(
        type("E", (), {"button": lambda s: Qt.MouseButton.LeftButton, "pos": lambda s: end})()
    )
    assert line_annotator._line_committed is not None


def test_line_annotator_esc_clears(line_annotator):
    line_annotator._line_committed = ((0.0, 0.0), (10.0, 10.0))
    line_annotator.keyPressEvent(type("E", (), {"key": lambda s: Qt.Key.Key_Escape})())
    assert line_annotator._line_committed is None


def test_profile_annotator_maps_widget_points_to_eta_row_col(line_annotator):
    line_annotator.clear_line()
    line_annotator.resize(200, 100)
    line_annotator.set_image(np.ones((50, 100)))
    mapped = line_annotator._widget_to_rowcol(QPoint(100, 100))
    assert mapped is not None
    row, col = mapped
    assert 20 <= row <= 30
    assert 45 <= col <= 55


def test_profile_line_change_emits_updated_spec(profile_view, tmp_path):
    from openfcd.io.result import HDF5ResultStore
    from openfcd.io.scene import SceneSpec, SceneType

    run_dir = tmp_path / "runs" / "run-profile"
    run_dir.mkdir(parents=True)
    store = HDF5ResultStore.open(run_dir / "results.h5", "w")
    store.write_frame("default", 0, np.ones((20, 30)), {"status": "ok"})
    store.close()

    spec = SceneSpec(
        id="p1",
        name="profile",
        type=SceneType.PROFILE,
        frame_indices=[0],
        run_id="run-profile",
    )
    received = []
    profile_view.scene_changed.connect(received.append)
    profile_view.load(spec, tmp_path)
    profile_view._annotator.set_image(np.ones((20, 30)))
    profile_view._annotator.set_committed_line(((2.0, 3.0), (4.0, 5.0)))
    profile_view._finish_annotation()
    profile_view.scene_changed.disconnect(received.append)
    assert received
    assert received[-1].profile_lines[0].p0 == (2.0, 3.0)


def test_profile_line_persists_after_save_reload(qapp, tmp_path):
    from openfcd.gui.mainwindow import MainWindow
    from openfcd.gui.controllers.session_controller import SessionController
    from openfcd.io.scene import ProfileLine, SceneSpec, SceneType
    from openfcd.io.store import FileSessionStore

    class _Tree:
        def populate_scenes(self, scenes, missing_counts=None):
            pass

    class _Status:
        def set_items(self, items):
            pass

    w = MainWindow.__new__(MainWindow)
    w._session = SessionController()
    w._sim_tree = _Tree()
    w._status_bar = _Status()
    project_dir = w._session.new_project("profile", tmp_path)
    spec = SceneSpec(id="p1", name="profile", type=SceneType.PROFILE, frame_indices=[0])
    w._session.add_scene(spec)
    w._session.save()
    updated = spec.model_copy(
        update={"profile_lines": {0: ProfileLine(p0=(1.0, 2.0), p1=(3.0, 4.0))}}
    )
    w._on_scene_changed(updated)

    reopened = FileSessionStore.open(project_dir)
    assert reopened.scenes[0].profile_lines[0].p1 == (3.0, 4.0)
    reopened.close()


def test_profile_scene_uses_composite_preview(profile_view, tmp_path):
    from openfcd.io.result import HDF5ResultStore
    from openfcd.io.scene import ProfileLine, SceneSpec, SceneType

    run_dir = tmp_path / "runs" / "run-profile-preview"
    run_dir.mkdir(parents=True)
    store = HDF5ResultStore.open(run_dir / "results.h5", "w")
    store.write_frame("default", 7, np.tile(np.linspace(-1.0, 1.0, 30), (20, 1)), {"status": "ok"})
    store.close()

    spec = SceneSpec(
        id="p1",
        name="profile",
        type=SceneType.PROFILE,
        frame_indices=[7],
        run_id="run-profile-preview",
        profile_lines={7: ProfileLine(p0=(10.0, 0.0), p1=(10.0, 29.0))},
    )
    profile_view.load(spec, tmp_path)
    _flush_render(profile_view)

    assert len(profile_view._fig.axes) >= 3
    assert profile_view._fig.axes[2].get_xlabel() == "x (mm)"


def test_profile_scene_marks_missing_spatial_calibration(profile_view, tmp_path):
    from openfcd.io.result import HDF5ResultStore
    from openfcd.io.scene import ProfileLine, SceneSpec, SceneType

    run_dir = tmp_path / "runs" / "run-profile-fallback"
    run_dir.mkdir(parents=True)
    store = HDF5ResultStore.open(run_dir / "results.h5", "w")
    store.write_frame("default", 7, np.ones((20, 30)), {"status": "ok"})
    store.close()

    spec = SceneSpec(
        id="p1",
        name="profile",
        type=SceneType.PROFILE,
        frame_indices=[7],
        run_id="run-profile-fallback",
        profile_lines={7: ProfileLine(p0=(10.0, 0.0), p1=(10.0, 29.0))},
    )
    profile_view.load(spec, tmp_path)
    _flush_render(profile_view)

    assert profile_view._frame_calibrations[7].source == "fallback_unverified"
    assert any("Unverified spatial calibration" in t.get_text() for t in profile_view._fig.axes[0].texts)


def test_profile_apply_viz_switches_from_annotation_to_chart(profile_view, tmp_path):
    from openfcd.io.result import HDF5ResultStore
    from openfcd.io.scene import SceneSpec, SceneType

    run_dir = tmp_path / "runs" / "run-profile-apply"
    run_dir.mkdir(parents=True)
    store = HDF5ResultStore.open(run_dir / "results.h5", "w")
    store.write_frame("default", 0, np.ones((20, 30)), {"status": "ok"})
    store.close()

    spec = SceneSpec(
        id="p1",
        name="profile",
        type=SceneType.PROFILE,
        frame_indices=[0],
        run_id="run-profile-apply",
    )
    profile_view.load(spec, tmp_path)
    assert profile_view._mode_stack.currentWidget() is profile_view._annotator

    profile_view.apply_viz({"y_range_mm": 40.0})

    assert profile_view._mode_stack.currentWidget() is profile_view._chart_widget


def test_profile_scene_restores_preferred_frame_pos(profile_view, tmp_path):
    from openfcd.io.result import HDF5ResultStore
    from openfcd.io.scene import ProfileLine, SceneSpec, SceneType

    run_dir = tmp_path / "runs" / "run-profile-restore"
    run_dir.mkdir(parents=True)
    store = HDF5ResultStore.open(run_dir / "results.h5", "w")
    for idx in range(3):
        store.write_frame("default", idx, np.ones((20, 30)), {"status": "ok"})
    store.close()

    spec = SceneSpec(
        id="p1",
        name="profile",
        type=SceneType.PROFILE,
        frame_indices=[0, 1, 2],
        run_id="run-profile-restore",
        profile_lines={
            0: ProfileLine(p0=(1.0, 1.0), p1=(1.0, 10.0)),
            1: ProfileLine(p0=(1.0, 1.0), p1=(1.0, 10.0)),
            2: ProfileLine(p0=(1.0, 1.0), p1=(1.0, 10.0)),
        },
    )
    profile_view.set_preferred_frame_pos(2)
    profile_view.load(spec, tmp_path)

    assert profile_view.current_slider_value() == 2


def test_profile_scene_opens_annotation_when_lines_missing(profile_view, tmp_path):
    from openfcd.io.result import HDF5ResultStore
    from openfcd.io.scene import SceneSpec, SceneType

    run_dir = tmp_path / "runs" / "run-profile-annotate"
    run_dir.mkdir(parents=True)
    store = HDF5ResultStore.open(run_dir / "results.h5", "w")
    store.write_frame("default", 0, np.ones((20, 30)), {"status": "ok"})
    store.close()

    spec = SceneSpec(
        id="p1",
        name="profile",
        type=SceneType.PROFILE,
        frame_indices=[0],
        run_id="run-profile-annotate",
    )
    profile_view.load(spec, tmp_path)

    assert profile_view._mode_stack.currentWidget() is profile_view._annotator


def test_profile_load_jumps_to_first_missing_line_frame(profile_view, tmp_path):
    from openfcd.io.result import HDF5ResultStore
    from openfcd.io.scene import ProfileLine, SceneSpec, SceneType

    run_dir = tmp_path / "runs" / "run-profile-missing"
    run_dir.mkdir(parents=True)
    store = HDF5ResultStore.open(run_dir / "results.h5", "w")
    store.write_frame("default", 0, np.ones((20, 30)), {"status": "ok"})
    store.write_frame("default", 1, np.ones((20, 30)) * 2, {"status": "ok"})
    store.close()
    spec = SceneSpec(
        id="p1",
        name="profile",
        type=SceneType.PROFILE,
        frame_indices=[0, 1],
        run_id="run-profile-missing",
        profile_lines={0: ProfileLine(p0=(1.0, 1.0), p1=(1.0, 10.0))},
    )

    profile_view.load(spec, tmp_path)

    assert profile_view._current_frame_pos == 1
    assert profile_view._mode_stack.currentWidget() is profile_view._annotator


def test_profile_finish_annotation_advances_to_next_missing_frame(profile_view, tmp_path):
    from openfcd.io.result import HDF5ResultStore
    from openfcd.io.scene import SceneSpec, SceneType

    run_dir = tmp_path / "runs" / "run-profile-advance"
    run_dir.mkdir(parents=True)
    store = HDF5ResultStore.open(run_dir / "results.h5", "w")
    for idx in range(3):
        store.write_frame("default", idx, np.ones((20, 30)) * idx, {"status": "ok"})
    store.close()
    spec = SceneSpec(
        id="p1",
        name="profile",
        type=SceneType.PROFILE,
        frame_indices=[0, 1, 2],
        run_id="run-profile-advance",
    )
    profile_view.load(spec, tmp_path)
    profile_view._annotator.set_committed_line(((2.0, 3.0), (4.0, 5.0)))
    profile_view._finish_annotation()

    assert profile_view._current_frame_pos == 1
    assert profile_view._mode_stack.currentWidget() is profile_view._annotator


def test_profile_finish_annotation_returns_to_preview_when_all_lines_exist(profile_view, tmp_path):
    from openfcd.io.result import HDF5ResultStore
    from openfcd.io.scene import SceneSpec, SceneType

    run_dir = tmp_path / "runs" / "run-profile-complete"
    run_dir.mkdir(parents=True)
    store = HDF5ResultStore.open(run_dir / "results.h5", "w")
    store.write_frame("default", 0, np.ones((20, 30)), {"status": "ok"})
    store.close()
    spec = SceneSpec(
        id="p1",
        name="profile",
        type=SceneType.PROFILE,
        frame_indices=[0],
        run_id="run-profile-complete",
    )
    profile_view.load(spec, tmp_path)
    profile_view._annotator.set_committed_line(((2.0, 3.0), (4.0, 5.0)))
    profile_view._finish_annotation()

    assert profile_view._mode_stack.currentWidget() is profile_view._chart_widget
