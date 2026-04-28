import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def panel(qapp):
    from openfcd.gui.panels.properties import VizSettingsPanel
    w = VizSettingsPanel()
    yield w
    w.close()


def test_apply_button_dirty_state(panel):
    panel._dirty = False
    panel._btn_apply.setText("Apply")
    panel._cmap.setCurrentIndex((panel._cmap.currentIndex() + 1) % panel._cmap.count())
    assert panel._dirty
    assert "Apply *" in panel._btn_apply.text()


def test_apply_emits_params(panel):
    received = []
    panel.apply_clicked.connect(received.append)
    panel._on_apply()
    assert len(received) == 1
    assert "cmap" in received[0]
    assert not panel._dirty
    panel.apply_clicked.disconnect()


def test_vmin_auto_disables_input(panel):
    panel._vmin_auto.setChecked(True)
    assert not panel._vmin.isEnabled()
    panel._vmin_auto.setChecked(False)
    assert panel._vmin.isEnabled()
    panel._vmin_auto.setChecked(True)  # restore


def test_get_viz_params_structure(panel):
    params = panel.get_viz_params()
    assert "cmap" in params
    assert "vmin" in params
    assert "dpi" in params


def test_colorbar_position_options(panel):
    opts = [panel._colorbar_pos.itemText(i) for i in range(panel._colorbar_pos.count())]
    assert "right" in opts
    assert "bottom" in opts
    assert "none" in opts


def test_set_scene_type_switches_stack(panel):
    panel.set_scene_type("eta_map")
    assert panel._type_stack.currentIndex() == 0
    panel.set_scene_type("profile")
    assert panel._type_stack.currentIndex() == 1
    panel.set_scene_type("rms")
    assert panel._type_stack.currentIndex() == 2


def test_load_viz_params_populates_fields(panel):
    panel.load_viz_params({"cmap": "viridis", "vmin": -1.0, "vmax": 1.0, "dpi": 200})
    assert panel._cmap.currentText() == "viridis"
    assert not panel._vmin_auto.isChecked()
    assert panel._vmin.text() == "-1.0"
    assert not panel._vmax_auto.isChecked()
    assert panel._vmax.text() == "1.0"
    assert panel._dpi.text() == "200"
    assert not panel._dirty
