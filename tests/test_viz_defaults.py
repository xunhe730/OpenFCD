import numpy as np
import pytest
from openfcd.gui.scenes.viz_defaults import (
    eta_map_defaults,
    rms_defaults,
    profile_defaults,
    scene_defaults,
)


def test_eta_map_bidirectional():
    eta = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    d = eta_map_defaults(eta)
    assert d["cmap"] == "RdBu_r"
    assert d["vmin"] < 0 and d["vmax"] > 0
    assert abs(d["vmin"]) == pytest.approx(abs(d["vmax"]), rel=0.01)


def test_eta_map_single_direction():
    eta = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    d = eta_map_defaults(eta)
    assert d["cmap"] == "viridis"
    assert d["vmin"] >= 0


def test_eta_map_no_data():
    d = eta_map_defaults()
    assert d["cmap"] == "RdBu_r"
    assert d["vmin"] is None
    assert d["vmax"] is None


def test_eta_map_nan_only():
    eta = np.full(5, np.nan)
    d = eta_map_defaults(eta)
    # no valid data — keep defaults
    assert d["vmin"] is None
    assert d["vmax"] is None


def test_rms_defaults_vmin_zero():
    d = rms_defaults()
    assert d["cmap"] == "magma"
    assert d["vmin"] == 0.0


def test_rms_defaults_vmax_from_data():
    rms = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    d = rms_defaults(rms)
    assert d["vmax"] is not None
    assert d["vmax"] < 2.0  # 98th percentile, not max


def test_profile_defaults_structure():
    d = profile_defaults()
    assert "line_color" in d
    assert d["grid"] is True
    assert "xlabel" in d
    assert "ylabel" in d


def test_scene_defaults_routing():
    d_eta = scene_defaults("eta_map")
    assert "cmap" in d_eta
    d_rms = scene_defaults("rms")
    assert d_rms["cmap"] == "magma"
    d_prof = scene_defaults("profile")
    assert "line_color" in d_prof


def test_scene_defaults_unknown_type():
    d = scene_defaults("unknown_type")
    assert d == {"dpi": 150}


def test_scene_defaults_with_data():
    eta = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    d = scene_defaults("eta_map", eta)
    assert d["vmin"] is not None
    assert d["vmax"] is not None
