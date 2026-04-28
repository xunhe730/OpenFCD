import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="session")
def main_window(qapp):
    from openfcd.gui.mainwindow import MainWindow
    from openfcd.gui.preferences import UserPrefs
    w = MainWindow()
    w._prefs = UserPrefs(backend={})
    yield w


def test_export_png_creates_file(main_window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    output = tmp_path / "test_export.png"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **kw: (str(output), "PNG files (*.png)")),
    )
    main_window._export_current_view_png()
    assert output.exists()
    assert output.stat().st_size > 0


def test_last_export_dir_persisted(qapp, tmp_path):
    from openfcd.gui.preferences import UserPrefs
    prefs = UserPrefs(backend={})
    prefs.last_export_dir = str(tmp_path)
    assert prefs.last_export_dir == str(tmp_path)


def test_last_export_dir_default_empty():
    from openfcd.gui.preferences import UserPrefs
    prefs = UserPrefs(backend={})
    assert prefs.last_export_dir == ""


def test_export_png_signal_exists(qapp):
    from openfcd.gui.widgets.preview_widget import PreviewWidget
    w = PreviewWidget()
    assert hasattr(w, "export_png_requested")
