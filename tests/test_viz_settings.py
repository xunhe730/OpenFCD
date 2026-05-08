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


def test_viz_load_resets_defaults_before_overlay(panel):
    panel.load_viz_params("profile", {"line_color": "green", "strip_mm": 9.0})
    panel.load_viz_params("profile", {})

    assert panel._line_color.text() == "red"
    assert panel._strip_mm.text() == "2.0"


def test_switch_scene_does_not_reuse_previous_profile_params(panel):
    panel.load_viz_params("profile", {"line_color": "green", "y_range_mm": 44.0})
    panel.load_viz_params("eta_map", {})
    panel.load_viz_params("profile", {})

    assert panel._line_color.text() == "red"
    assert panel._y_range_mm.text() == "30.0"
    assert not hasattr(panel, "_show_measurements")


def test_profile_viz_invalid_numeric_values_fall_back_to_defaults(panel):
    panel.load_viz_params("profile", {})
    panel._strip_mm.setText("-2")
    panel._y_range_mm.setText("not-a-number")
    panel._min_roi_width_mm.setText("0")
    panel._min_roi_height_mm.setText("-1")
    panel._x_padding_mm.setText("-3")
    panel._dpi.setText("0")
    params = panel.get_viz_params()

    assert "px_per_mm" not in params
    assert params["strip_mm"] == 2.0
    assert params["y_range_mm"] == 30.0
    assert params["min_roi_width_mm"] == 160.0
    assert params["min_roi_height_mm"] == 50.0
    assert params["x_padding_mm"] == 5.0
    assert params["dpi"] == 150


def test_profile_x_range_is_normalized_on_apply(panel):
    panel.load_viz_params("profile", {})
    panel._x_range_mm.setText("10:30")

    params = panel.get_viz_params()

    assert params["x_range_mm"] == [10.0, 30.0]


def test_profile_invalid_x_range_clears_field(panel):
    panel.load_viz_params("profile", {})
    panel._x_range_mm.setText("bad:value")

    params = panel.get_viz_params()

    assert params["x_range_mm"] is None
    assert panel._x_range_mm.text() == ""


def test_load_viz_params_clears_dirty_button_style(panel):
    panel._mark_dirty()
    panel.load_viz_params("profile", {})

    assert not panel._dirty
    assert panel._btn_apply.text() == "Apply"
    assert "font-weight:600" not in panel._btn_apply.styleSheet()


def test_viz_apply_updates_scene_json_and_refreshes_view(qapp, tmp_path):
    from openfcd.gui.controllers.session_controller import SessionController
    from openfcd.gui.mainwindow import MainWindow
    from openfcd.io.scene import SceneSpec, SceneType

    class _Tree:
        def populate_scenes(self, scenes, missing_counts=None):
            pass

    class _SceneContainer:
        def __init__(self):
            self.params = None

        def apply_viz(self, params):
            self.params = params

    class _Status:
        def set_items(self, items):
            pass

    w = MainWindow.__new__(MainWindow)
    w._session = SessionController()
    w._sim_tree = _Tree()
    w._scene_container = _SceneContainer()
    w._status_bar = _Status()
    project_dir = w._session.new_project("viz", tmp_path)
    spec = SceneSpec(id="s1", name="scene", type=SceneType.ETA_MAP, frame_indices=[0])
    w._session.add_scene(spec)
    w._session.save()
    w._current_scene_id = "s1"
    w._on_viz_apply({"cmap": "viridis", "dpi": 200})
    data = (project_dir / "scenes" / "s1.json").read_text()
    assert '"cmap": "viridis"' in data
    assert w._scene_container.params["cmap"] == "viridis"


def test_persist_scene_reselects_current_scene(qapp, tmp_path):
    from openfcd.gui.controllers.session_controller import SessionController
    from openfcd.gui.mainwindow import MainWindow
    from openfcd.io.scene import SceneSpec, SceneType

    class _Tree:
        def __init__(self):
            self.populated = False
            self.selected = None
            self.blocked = False

        def populate_scenes(self, scenes, missing_counts=None):
            self.populated = True

        def find_scene_item(self, scene_id):
            return {"id": scene_id}

        def blockSignals(self, val):
            self.blocked = val

        def setCurrentItem(self, item):
            self.selected = item

    class _Panel:
        def __init__(self):
            self.loaded = None

        def load_viz_params(self, scene_type, params):
            self.loaded = (scene_type, params)

    class _Props:
        def __init__(self):
            self.viz_settings_panel = _Panel()

    w = MainWindow.__new__(MainWindow)
    w._session = SessionController()
    w._sim_tree = _Tree()
    w._properties = _Props()
    project_dir = w._session.new_project("persist", tmp_path)
    spec = SceneSpec(id="s1", name="scene", type=SceneType.PROFILE, frame_indices=[0])
    w._session.add_scene(spec)
    w._session.save()

    w._persist_scene(spec)

    assert w._sim_tree.populated
    assert w._sim_tree.selected == {"id": "s1"}
    assert w._properties.viz_settings_panel.loaded[0] == "profile"
