"""Unit tests for DisplayModeRegistry."""
from __future__ import annotations

import numpy as np
import pytest
from openfcd.gui.widgets.display_mode import DisplayMode, DisplayModeRegistry, DisplayModeSpec


def _make_spec(mode: DisplayMode, label: str = "Test") -> DisplayModeSpec:
    return DisplayModeSpec(
        mode=mode,
        label=label,
        tooltip=f"Tooltip for {label}",
        paint=lambda w, eta: None,
    )


def test_register_and_get_spec():
    registry = DisplayModeRegistry()
    spec = _make_spec(DisplayMode.ORIGINAL, "Original")
    registry.register(spec)

    result = registry.get(DisplayMode.ORIGINAL)

    assert result is spec
    assert result.mode == DisplayMode.ORIGINAL
    assert result.label == "Original"
    assert result.tooltip == "Tooltip for Original"


def test_all_modes_returns_registered_specs_in_order():
    registry = DisplayModeRegistry()
    spec_orig = _make_spec(DisplayMode.ORIGINAL, "Original")
    spec_over = _make_spec(DisplayMode.OVERLAY, "Overlay")
    spec_eta = _make_spec(DisplayMode.ETA_ONLY, "Eta Only")

    registry.register(spec_orig)
    registry.register(spec_over)
    registry.register(spec_eta)

    modes = registry.all_modes()
    assert len(modes) == 3
    assert modes[0] is spec_orig
    assert modes[1] is spec_over
    assert modes[2] is spec_eta


def test_get_unknown_mode_returns_none():
    registry = DisplayModeRegistry()
    assert registry.get(DisplayMode.OVERLAY) is None


def test_register_overwrites_existing_mode():
    registry = DisplayModeRegistry()
    spec_v1 = _make_spec(DisplayMode.ORIGINAL, "v1")
    spec_v2 = _make_spec(DisplayMode.ORIGINAL, "v2")
    registry.register(spec_v1)
    registry.register(spec_v2)

    result = registry.get(DisplayMode.ORIGINAL)
    assert result is spec_v2
    assert len(registry.all_modes()) == 1


def test_paint_callable_is_stored():
    called = []
    def my_paint(w, eta):
        called.append((w, eta))

    spec = DisplayModeSpec(
        mode=DisplayMode.OVERLAY,
        label="Overlay",
        tooltip="With paint",
        paint=my_paint,
    )
    registry = DisplayModeRegistry()
    registry.register(spec)

    retrieved = registry.get(DisplayMode.OVERLAY)
    retrieved.paint("widget", np.zeros((3, 3)))
    assert len(called) == 1
    assert called[0][0] == "widget"
