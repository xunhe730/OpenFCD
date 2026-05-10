"""Unit tests for ``openfcd.gui.renderers._eta_view`` helpers."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from openfcd.gui.renderers._eta_view import compute_eta_color_range, crop_to_valid


class TestComputeEtaColorRange:
    def test_autoscale_when_both_none(self) -> None:
        rng = np.random.default_rng(0)
        eta = rng.normal(0.0, 0.1, size=(50, 50))
        viz = SimpleNamespace(eta_vmin_mm=None, eta_vmax_mm=None)
        vmin, vmax = compute_eta_color_range(eta, viz)
        # Symmetric (diverging by default) and equal to 98th percentile of |eta|
        expected = float(np.nanpercentile(np.abs(eta), 98))
        assert vmax == pytest.approx(expected)
        assert vmin == pytest.approx(-expected)

    def test_explicit_overrides(self) -> None:
        eta = np.array([[0.0, 1.0], [-1.0, 2.0]])
        viz = SimpleNamespace(eta_vmin_mm=-0.5, eta_vmax_mm=0.7)
        vmin, vmax = compute_eta_color_range(eta, viz)
        assert vmin == -0.5
        assert vmax == 0.7

    def test_partial_override_keeps_auto(self) -> None:
        rng = np.random.default_rng(1)
        eta = rng.normal(0.0, 0.1, size=(20, 20))
        viz = SimpleNamespace(eta_vmin_mm=None, eta_vmax_mm=0.5)
        vmin, vmax = compute_eta_color_range(eta, viz)
        assert vmax == 0.5
        # Diverging default → -vmax
        assert vmin == pytest.approx(-0.5)

    def test_all_nan_returns_finite_defaults(self) -> None:
        eta = np.full((4, 4), np.nan)
        viz = SimpleNamespace(eta_vmin_mm=None, eta_vmax_mm=None)
        vmin, vmax = compute_eta_color_range(eta, viz)
        assert np.isfinite(vmin) and np.isfinite(vmax)
        assert vmin < vmax

    def test_diverging_false_uses_low_percentile(self) -> None:
        eta = np.linspace(-1.0, 1.0, 100).reshape(10, 10)
        viz = SimpleNamespace(eta_vmin_mm=None, eta_vmax_mm=None)
        vmin, vmax = compute_eta_color_range(eta, viz, diverging=False, cmap="viridis")
        assert vmin < 0.0 < vmax
        # Not necessarily symmetric
        assert vmin != pytest.approx(-vmax)


class TestCropToValid:
    def test_crops_to_finite_bbox(self) -> None:
        eta = np.full((10, 12), np.nan)
        eta[3:6, 4:9] = 1.0
        out = crop_to_valid(eta)
        assert out.shape[0] >= 3
        assert out.shape[1] >= 5
        assert np.isfinite(out).any()

    def test_all_nan_returns_input(self) -> None:
        eta = np.full((5, 5), np.nan)
        out = crop_to_valid(eta)
        assert out.shape == eta.shape

    def test_all_finite_no_change(self) -> None:
        eta = np.ones((6, 7))
        out = crop_to_valid(eta)
        assert out.shape == eta.shape
