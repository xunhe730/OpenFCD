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


def test_profile_scene_view_loads(profile_view, tmp_path):
    from openfcd.io.scene import SceneSpec, SceneType

    spec = SceneSpec(id="p1", name="wake profile", type=SceneType.PROFILE, frame_indices=[0, 1])
    profile_view.load(spec, tmp_path)
    assert "wake profile" in profile_view._title.text()


def test_line_annotator_drag(line_annotator):
    # Reset state before test
    line_annotator.clear_line()
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
    line_annotator._line_committed = (QPoint(0, 0), QPoint(10, 10))
    line_annotator.keyPressEvent(type("E", (), {"key": lambda s: Qt.Key.Key_Escape})())
    assert line_annotator._line_committed is None


def test_sample_along_used_in_profile(qapp, tmp_path):
    from openfcd.core.profile import sample_along

    eta = np.ones((20, 30)) * 0.5
    dists, vals = sample_along(eta, (10, 0), (10, 29), n=30)
    assert len(vals) == 30
    assert np.allclose(vals, 0.5)
