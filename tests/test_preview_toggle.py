import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
pytest.importorskip("PyQt6")
import numpy as np
from PyQt6.QtWidgets import QApplication
from openfcd.gui.mainwindow import MainWindow
from openfcd.gui.widgets.toolbar import Toolbar

@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])

def test_preview_disabled_by_default(qapp):
    t = Toolbar()
    assert not t._btn_preview_toggle.isEnabled()
    assert t.preview_on() is False

def test_set_preview_available_enables_button(qapp):
    t = Toolbar()
    t.set_preview_available(True)
    assert t._btn_preview_toggle.isEnabled()
    t.set_preview_available(False)
    assert not t._btn_preview_toggle.isEnabled()
    assert t.preview_on() is False

def test_preview_on_off_toggle(qapp):
    t = Toolbar()
    t.set_preview_available(True)
    assert t.preview_on() is False
    t.set_preview_on(True)
    assert t.preview_on() is True
    t.set_preview_on(False)
    assert t.preview_on() is False

def test_overlap_colorbar_enabled_only_when_preview_on(qapp):
    t = Toolbar()
    t.set_preview_available(True)
    # Initially disabled
    assert not t._overlap_cb.isEnabled()
    assert not t._colorbar_cb.isEnabled()
    # Enable preview → checkboxes become enabled
    t.set_preview_on(True)
    assert t._overlap_cb.isEnabled()
    assert t._colorbar_cb.isEnabled()
    # Disable preview → checkboxes disabled again
    t.set_preview_on(False)
    assert not t._overlap_cb.isEnabled()
    assert not t._colorbar_cb.isEnabled()

def test_overlap_on_default_true(qapp):
    t = Toolbar()
    assert t.overlap_on() is True

def test_colorbar_on_default_false(qapp):
    t = Toolbar()
    assert t.colorbar_on() is False

def test_preview_signals_emitted(qapp):
    t = Toolbar()
    t.set_preview_available(True)
    received = []
    t.preview_changed.connect(received.append)
    t.set_preview_on(True)
    # set_preview_on blocks signals on the toggle, so preview_changed is NOT emitted
    # (programmatic set doesn't fire the signal — only user interaction does)
    assert received == []
    # But toggling via _btn_preview_toggle directly does emit
    t._btn_preview_toggle.toggle()
    assert len(received) == 1


def test_main_window_syncs_toolbar_colorbar_to_preview(qapp):
    class _Toolbar:
        def preview_on(self) -> bool:
            return True

        def overlap_on(self) -> bool:
            return True

        def colorbar_on(self) -> bool:
            return True

    class _Preview:
        def __init__(self) -> None:
            self.colorbar = None
            self.shown = None

        def set_colorbar(self, val: bool) -> None:
            self.colorbar = val

        def show_eta_overlay(self, eta, **kwargs) -> None:
            self.shown = (np.asarray(eta), kwargs)

        def clear_eta_overlay(self) -> None:
            self.shown = None

    window = MainWindow.__new__(MainWindow)
    window._toolbar = _Toolbar()
    window._preview = _Preview()
    window._current_frame_idx = 0
    window._frame_eta_cache = {0: np.ones((2, 2))}
    window._run_eta_frames = None

    window._reapply_run_eta()

    assert window._preview.colorbar is True
    np.testing.assert_array_equal(window._preview.shown[0], np.ones((2, 2)))
