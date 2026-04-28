"""Test that NewProjectWizard reads and writes UserPrefs."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from openfcd.gui.dialogs.new_project_wizard import NewProjectWizard
from openfcd.gui.preferences import UserPrefs


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_wizard_uses_prefs_for_initial_values(qapp) -> None:
    backend = {
        "last_project_dir": "/var/projects",
        "last_image_folder": "/data/run42",
        "last_file_pattern": "*.tif",
        "last_optical_preset": "immersed_pattern",
        "last_pattern_period_mm": 1.7,
        "last_glass_thickness_mm": 5.0,
        "last_fluid_depth_mm": 18.0,
    }
    prefs = UserPrefs(backend=backend)
    w = NewProjectWizard(prefs=prefs)
    assert w.location_edit.text() == "/var/projects"
    assert w.image_folder_edit.text() == "/data/run42"
    assert w.pattern_edit.text() == "*.tif"
    assert w.optical_preset == "immersed_pattern"
    assert w.pattern_period_mm == pytest.approx(1.7)
    assert w.glass_thickness_mm == pytest.approx(5.0)
    assert w.fluid_depth_mm == pytest.approx(18.0)
    w.close()


def test_wizard_persists_prefs_on_accept(qapp) -> None:
    backend: dict = {}
    prefs = UserPrefs(backend=backend)
    w = NewProjectWizard(prefs=prefs)
    w.name_edit.setText("flow_run")
    w.location_edit.setText("/tmp/openfcd-tests")
    w.image_folder_edit.setText("/tmp/frames")
    w.pattern_edit.setText("frame_*.png")
    w._optical_preset.setCurrentText("immersed_pattern")
    w._pattern_period.setValue(1.4)
    w._glass_thickness.setValue(2.5)
    w._fluid_depth.setValue(11.0)

    w.persist_to_prefs()

    reread = UserPrefs(backend=backend)
    assert reread.last_project_dir == "/tmp/openfcd-tests"
    assert reread.last_image_folder == "/tmp/frames"
    assert reread.last_file_pattern == "frame_*.png"
    assert reread.last_optical_preset == "immersed_pattern"
    assert reread.last_pattern_period_mm == pytest.approx(1.4)
    assert reread.last_glass_thickness_mm == pytest.approx(2.5)
    assert reread.last_fluid_depth_mm == pytest.approx(11.0)
    w.close()


def test_wizard_falls_back_to_hardcoded_defaults_when_prefs_empty(qapp) -> None:
    prefs = UserPrefs(backend={})
    w = NewProjectWizard(prefs=prefs)
    assert w.location_edit.text().endswith("Documents")
    assert w.pattern_edit.text() == "Img*.jpg"
    assert w.optical_preset == "pattern_below_window"
    assert w.pattern_period_mm == pytest.approx(1.2)
    assert w.glass_thickness_mm == pytest.approx(3.0)
    assert w.fluid_depth_mm == pytest.approx(12.0)
    w.close()
