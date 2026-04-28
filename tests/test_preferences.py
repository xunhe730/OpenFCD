"""Tests for UserPrefs persistence layer."""
from __future__ import annotations

from openfcd.gui.preferences import UserPrefs


def test_defaults_when_backend_empty() -> None:
    prefs = UserPrefs(backend={})
    assert prefs.last_project_dir == ""
    assert prefs.last_image_folder == ""
    assert prefs.last_file_pattern == "Img*.jpg"
    assert prefs.last_optical_preset == "pattern_below_window"
    assert prefs.last_pattern_period_mm == 0.0
    assert prefs.last_glass_thickness_mm == 0.0
    assert prefs.last_fluid_depth_mm == 0.0
    assert prefs.recent_projects == []


def test_round_trip_through_shared_backend() -> None:
    backend: dict = {}
    writer = UserPrefs(backend=backend)
    writer.last_project_dir = "/tmp/projects"
    writer.last_image_folder = "/data/frames"
    writer.last_file_pattern = "*.tif"
    writer.last_optical_preset = "immersed_pattern"
    writer.last_pattern_period_mm = 1.5
    writer.last_glass_thickness_mm = 4.5
    writer.last_fluid_depth_mm = 22.0

    reader = UserPrefs(backend=backend)
    assert reader.last_project_dir == "/tmp/projects"
    assert reader.last_image_folder == "/data/frames"
    assert reader.last_file_pattern == "*.tif"
    assert reader.last_optical_preset == "immersed_pattern"
    assert reader.last_pattern_period_mm == 1.5
    assert reader.last_glass_thickness_mm == 4.5
    assert reader.last_fluid_depth_mm == 22.0


def test_recent_projects_dedup_and_cap() -> None:
    prefs = UserPrefs(backend={})
    for i in range(10):
        prefs.add_recent_project(f"/p/{i}.ofcd")

    recent = prefs.recent_projects
    assert recent[0] == "/p/9.ofcd"
    assert len(recent) == 8

    # Re-adding existing path should move it to the front.
    prefs.add_recent_project("/p/3.ofcd")
    recent = prefs.recent_projects
    assert recent[0] == "/p/3.ofcd"
    assert recent.count("/p/3.ofcd") == 1
    assert len(recent) == 8


def test_clear_recent_projects() -> None:
    prefs = UserPrefs(backend={})
    prefs.add_recent_project("/p/a.ofcd")
    prefs.add_recent_project("/p/b.ofcd")
    assert prefs.recent_projects

    prefs.clear_recent_projects()
    assert prefs.recent_projects == []


def test_floats_coerce_from_string_backend() -> None:
    """QSettings on some platforms returns strings even for numeric writes."""
    prefs = UserPrefs(backend={"last_pattern_period_mm": "1.7"})
    assert prefs.last_pattern_period_mm == 1.7


def test_add_recent_ignores_empty_path() -> None:
    prefs = UserPrefs(backend={})
    prefs.add_recent_project("")
    prefs.add_recent_project("   ")
    assert prefs.recent_projects == []


def test_reset_clears_all_known_keys() -> None:
    backend: dict = {}
    prefs = UserPrefs(backend=backend)
    prefs.last_project_dir = "/x"
    prefs.last_pattern_period_mm = 9.9
    prefs.add_recent_project("/x/a.ofcd")
    prefs.reset()
    assert prefs.last_project_dir == ""
    assert prefs.last_pattern_period_mm == 0.0
    assert prefs.recent_projects == []
