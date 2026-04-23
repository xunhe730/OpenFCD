"""Tests for SessionController.new_project() geometry initialization."""
from __future__ import annotations

import tempfile

import pytest

from openfcd.gui.controllers.session_controller import SessionController


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
