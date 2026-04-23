"""Tests for _repair_optical_config — auto-repair of old projects with missing layers."""
from __future__ import annotations

import tempfile
from pathlib import Path
from datetime import datetime, timezone

import pytest

from openfcd.io.project import OpticalLayer
from openfcd.io.project import ProjectModel
from openfcd.io.store import FileSessionStore
from openfcd.gui.controllers.session_controller import SessionController
from openfcd.gui.mainwindow import (
    _apply_optical_preset,
    _apply_optical_preset_with_dimensions,
    _repair_optical_config,
)


def _make_mock_project(preset: str, layers: list) -> tuple[FileSessionStore, SessionController]:
    """Create a project with specific optical_stack settings for testing."""
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp) / "test.ofcd"
        project_dir.mkdir(parents=True, exist_ok=True)

        store = FileSessionStore.new(project_dir, "test")
        proj = store.project
        proj.geometry.optical_stack.preset = preset  # type: ignore[assignment]
        proj.geometry.optical_stack.layers = layers
        store.save()

        session = SessionController()
        session._store = store
        session._dirty = False

        return store, session


def test_repair_pattern_below_window_with_empty_layers() -> None:
    """preset=pattern_below_window + layers=[] → layers populated, dirty=True."""
    store, session = _make_mock_project("pattern_below_window", [])
    result = _repair_optical_config(store.project, session)

    assert result is True
    assert len(store.project.geometry.optical_stack.layers) == 2
    assert session.is_dirty is True


def test_repair_immersed_pattern_with_empty_layers() -> None:
    """preset=immersed_pattern + layers=[] → layers populated, dirty=True."""
    store, session = _make_mock_project("immersed_pattern", [])
    result = _repair_optical_config(store.project, session)

    assert result is True
    assert len(store.project.geometry.optical_stack.layers) == 1
    assert session.is_dirty is True


def test_no_repair_for_custom_preset() -> None:
    """preset=custom + layers=[] → no repair, dirty=False."""
    store, session = _make_mock_project("custom", [])
    result = _repair_optical_config(store.project, session)

    assert result is False
    assert store.project.geometry.optical_stack.layers == []
    assert session.is_dirty is False


def test_no_repair_for_valid_project() -> None:
    """preset with existing layers → no changes."""
    from openfcd.geometry.optical import get_default_layers

    store, session = _make_mock_project("pattern_below_window", [])
    # Pre-populate layers
    store.project.geometry.optical_stack.layers = get_default_layers("pattern_below_window")
    session._dirty = False

    result = _repair_optical_config(store.project, session)

    assert result is False
    assert session.is_dirty is False


def test_apply_optical_preset_populates_default_layers() -> None:
    """Switching to a preset-backed stack should install matching default layers."""
    store, _session = _make_mock_project("custom", [])

    applied = _apply_optical_preset(store.project, "pattern_below_window")

    assert applied is True
    assert store.project.geometry.optical_stack.preset == "pattern_below_window"
    layers = store.project.geometry.optical_stack.layers
    assert len(layers) == 2
    assert layers[0].medium == "glass"
    assert layers[1].medium == "water"


def test_apply_optical_preset_replaces_layers_when_switching_presets() -> None:
    """Changing between non-custom presets should replace stale layer geometry."""
    store, _session = _make_mock_project(
        "pattern_below_window",
        [
            OpticalLayer(thickness_mm=3.0, medium="glass", n=1.5),
            OpticalLayer(thickness_mm=12.0, medium="water", n=1.333),
        ],
    )

    applied = _apply_optical_preset(store.project, "immersed_pattern")

    assert applied is True
    assert store.project.geometry.optical_stack.preset == "immersed_pattern"
    layers = store.project.geometry.optical_stack.layers
    assert len(layers) == 1
    assert layers[0].medium == "water"
    assert layers[0].thickness_mm == 10.0


def test_apply_optical_preset_with_dimensions_updates_layer_thicknesses() -> None:
    """User-edited glass/fluid thickness should propagate into preset-backed layers."""
    store, _session = _make_mock_project("pattern_below_window", [])

    applied = _apply_optical_preset_with_dimensions(
        store.project,
        "pattern_below_window",
        glass_thickness_mm=4.2,
        fluid_depth_mm=21.5,
    )

    assert applied is True
    layers = store.project.geometry.optical_stack.layers
    assert len(layers) == 2
    assert layers[0].thickness_mm == 4.2
    assert layers[1].thickness_mm == 21.5
