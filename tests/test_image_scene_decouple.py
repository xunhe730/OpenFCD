import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication

@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])

def test_scene_view_placeholder_renders(qapp):
    from openfcd.gui.scenes.scene_view_placeholder import SceneViewPlaceholder
    w = SceneViewPlaceholder()
    w.set_scene("s1", "eta_map", "Wake η", 42)
    assert "s1" in w._lbl_frames.text()
    assert "42" in w._lbl_frames.text()
    assert "Wake" in w._lbl_name.text()

def test_preview_widget_scene_slider(qapp):
    from openfcd.gui.widgets.preview_widget import PreviewWidget
    pw = PreviewWidget()
    pw.setup_slider(100)
    assert pw._slider.maximum() == 99

    pw.set_scene_slider([10, 20, 30, 40])
    assert pw._slider.maximum() == 3   # 4 items → 0..3

    pw.restore_full_slider(100)
    assert pw._slider.maximum() == 99

def test_preview_widget_scene_slider_empty(qapp):
    from openfcd.gui.widgets.preview_widget import PreviewWidget
    pw = PreviewWidget()
    pw.set_scene_slider([])
    # Should not crash; slider hidden
    assert not pw._slider.isVisible()
