"""Tests for openfcd.geometry.optical — get_default_layers helper."""
from __future__ import annotations

import pytest

from openfcd.geometry.optical import get_default_layers
from openfcd.io.project import OpticalLayer


class TestGetDefaultLayers:
    """Test the get_default_layers preset helper."""

    def test_default_layers_pattern_below_window(self):
        """pattern_below_window returns 2 layers: glass + water."""
        layers = get_default_layers("pattern_below_window")

        assert len(layers) == 2

        # Layer 0: glass
        assert isinstance(layers[0], OpticalLayer)
        assert layers[0].thickness_mm == 3.0
        assert layers[0].medium == "glass"
        assert layers[0].n == 1.5

        # Layer 1: water
        assert isinstance(layers[1], OpticalLayer)
        assert layers[1].thickness_mm == 12.0
        assert layers[1].medium == "water"
        assert layers[1].n == 1.333

    def test_default_layers_immersed_pattern(self):
        """immersed_pattern returns 1 layer: water."""
        layers = get_default_layers("immersed_pattern")

        assert len(layers) == 1
        assert isinstance(layers[0], OpticalLayer)
        assert layers[0].thickness_mm == 10.0
        assert layers[0].medium == "water"
        assert layers[0].n == 1.333

    def test_default_layers_accept_explicit_dimensions(self):
        """Preset-backed helpers should honor user-specified thickness inputs."""
        layers = get_default_layers(
            "pattern_below_window",
            glass_thickness_mm=5.5,
            fluid_depth_mm=18.0,
        )

        assert len(layers) == 2
        assert layers[0].thickness_mm == 5.5
        assert layers[1].thickness_mm == 18.0

    def test_default_layers_custom(self):
        """custom preset returns empty list."""
        layers = get_default_layers("custom")
        assert layers == []

    def test_default_layers_invalid(self):
        """Invalid preset name returns empty list (graceful fallback)."""
        layers = get_default_layers("nonexistent_preset")
        assert layers == []
