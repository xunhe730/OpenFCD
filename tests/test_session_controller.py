"""Tests for SessionController.new_project() geometry initialization and last_run_id persistence."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from openfcd.gui.controllers.session_controller import SessionController
from openfcd.io.store import FileSessionStore


@pytest.fixture()
def controller() -> SessionController:
    return SessionController()


def test_new_project_geometry_pattern_below_window(controller: SessionController) -> None:
    """Create project with pattern_below_window preset; verify layers populated."""
    with tempfile.TemporaryDirectory() as tmp:
        result = controller.new_project(
            name="test_proj",
            location=tmp,
            optical_preset="pattern_below_window",
            pattern_period_mm=1.5,
            glass_thickness_mm=4.0,
            fluid_depth_mm=20.0,
        )

        assert result.exists()
        proj = controller.project
        assert proj is not None
        assert proj.geometry.pattern_period_mm == 1.5
        assert proj.geometry.optical_stack.preset == "pattern_below_window"
        layers = proj.geometry.optical_stack.layers
        assert len(layers) == 2
        assert layers[0].medium == "glass"
        assert layers[0].thickness_mm == 4.0
        assert layers[1].medium == "water"
        assert layers[1].thickness_mm == 20.0


def test_new_project_geometry_custom(controller: SessionController) -> None:
    """Create project with custom preset; verify layers is empty list."""
    with tempfile.TemporaryDirectory() as tmp:
        result = controller.new_project(
            name="test_custom",
            location=tmp,
            optical_preset="custom",
            pattern_period_mm=2.0,
        )

        assert result.exists()
        proj = controller.project
        assert proj is not None
        assert proj.geometry.pattern_period_mm == 2.0
        assert proj.geometry.optical_stack.preset == "custom"
        assert proj.geometry.optical_stack.layers == []


def test_new_project_custom_pattern_period(controller: SessionController) -> None:
    """Verify pattern_period_mm from parameter is written to geometry."""
    with tempfile.TemporaryDirectory() as tmp:
        result = controller.new_project(
            name="test_period",
            location=tmp,
            pattern_period_mm=0.8,
        )

        assert result.exists()
        proj = controller.project
        assert proj is not None
        assert proj.geometry.pattern_period_mm == 0.8


# ── last_run_id persistence tests ────────────────────────────────────────────

def test_last_run_id_round_trip() -> None:
    """save session with last_run_id, reopen, verify _last_run_id is restored."""
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp) / "roundtrip.ofcd"
        store = FileSessionStore.new(project_dir, "roundtrip")
        store._last_run_id = "abc123"
        store.save()

        reopened = FileSessionStore.open(project_dir)
        assert reopened._last_run_id == "abc123"


def test_record_run_sets_last_run_id() -> None:
    """record_run() updates _last_run_id to the recorded run's id."""
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp) / "rec.ofcd"
        store = FileSessionStore.new(project_dir, "rec")
        assert store._last_run_id is None

        store.record_run("run-xyz", {"status": "ok", "timestamp": "2026-01-01T00:00:00Z"})
        assert store._last_run_id == "run-xyz"


def test_session_controller_last_run_id_property() -> None:
    """SessionController.last_run_id setter+getter round-trips correctly via save/reopen."""
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = SessionController()
        ctrl.new_project(name="sc_test", location=tmp)
        assert ctrl.last_run_id is None

        ctrl.last_run_id = "rid-42"
        ctrl.save()

        ctrl2 = SessionController()
        ctrl2.open_project(str(Path(tmp) / "sc_test.ofcd"))
        assert ctrl2.last_run_id == "rid-42"
