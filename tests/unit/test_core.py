"""Unit tests for openfcd.core."""
import numpy as np
import pytest
import tempfile
import os

from openfcd.core.fcd import fftinvgrad, carrier_wavelength_mm
from openfcd.core.mask import polygon_mask, Polygon
from openfcd.core.temporal import time_mean, time_rms
from openfcd.core.flatfield import flatfield_normalize
from openfcd.pipeline.base import StageEvent


# ---------------------------------------------------------------------------
# 1. fftinvgrad — synthetic gradient field integration accuracy
# ---------------------------------------------------------------------------

def test_fftinvgrad_zero_input():
    """fftinvgrad of zero gradient field returns zero."""
    h, w = 32, 32
    fx = np.zeros((h, w))
    fy = np.zeros((h, w))
    result = fftinvgrad(fx, fy)
    assert result.shape == (h, w)
    assert np.allclose(result, 0.0, atol=1e-12)


def test_fftinvgrad_linearity():
    """fftinvgrad is linear: fftinvgrad(a*fx, a*fy) == a * fftinvgrad(fx, fy)."""
    h, w = 32, 32
    rng = np.random.default_rng(0)
    fx = rng.standard_normal((h, w))
    fy = rng.standard_normal((h, w))
    a = 3.7
    result_scaled = fftinvgrad(a * fx, a * fy)
    result_base = fftinvgrad(fx, fy)
    assert np.allclose(result_scaled, a * result_base, atol=1e-10)


def test_fftinvgrad_output_shape_and_real():
    """fftinvgrad returns real-valued array with same shape as input."""
    h, w = 40, 48
    rng = np.random.default_rng(1)
    fx = rng.standard_normal((h, w))
    fy = rng.standard_normal((h, w))
    result = fftinvgrad(fx, fy)
    assert result.shape == (h, w)
    assert result.dtype in (np.float64, np.float32, float)
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# 2. polygon_mask — shape and coverage
# ---------------------------------------------------------------------------

def test_polygon_mask_shape():
    shape = (100, 120)
    poly = Polygon([(10, 10), (10, 50), (40, 50), (40, 10)])
    m = polygon_mask(shape, poly)
    assert m.shape == shape
    assert m.dtype == bool


def test_polygon_mask_coverage():
    shape = (100, 120)
    # Rectangle from (10,10) to (40,50): (40-10)*(50-10) = 30*40 = 1200 pixels
    poly = Polygon([(10, 10), (10, 50), (40, 50), (40, 10)])
    m = polygon_mask(shape, poly)
    # skimage draw_polygon fills including boundary pixels; count within ±10% of area
    expected = 30 * 40
    assert abs(int(m.sum()) - expected) < expected * 0.10


def test_polygon_mask_zero_outside():
    shape = (50, 50)
    poly = Polygon([(5, 5), (5, 15), (15, 15), (15, 5)])
    m = polygon_mask(shape, poly)
    assert not m[0, 0]
    assert not m[49, 49]


# ---------------------------------------------------------------------------
# 3. time_mean / time_rms — known frame sequence
# ---------------------------------------------------------------------------

def _write_npy_files(arrays):
    """Write arrays to temp npy files, return list of paths."""
    paths = []
    for a in arrays:
        f = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
        np.save(f.name, a)
        paths.append(f.name)
        f.close()
    return paths


def test_time_mean_constant():
    frames = [np.full((4, 4), float(v)) for v in [1, 2, 3]]
    paths = _write_npy_files(frames)
    try:
        result = time_mean(paths)
        assert result.shape == (4, 4)
        assert np.allclose(result, 2.0, atol=1e-10)
    finally:
        for p in paths:
            os.unlink(p)


def test_time_mean_known():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[3.0, 4.0], [5.0, 6.0]])
    paths = _write_npy_files([a, b])
    try:
        result = time_mean(paths)
        expected = (a + b) / 2
        assert np.allclose(result, expected, atol=1e-10)
    finally:
        for p in paths:
            os.unlink(p)


def test_time_rms_zero_for_constant():
    frames = [np.full((3, 3), 5.0)] * 4
    paths = _write_npy_files(frames)
    try:
        result = time_rms(paths)
        assert np.allclose(result, 0.0, atol=1e-10)
    finally:
        for p in paths:
            os.unlink(p)


def test_time_rms_known():
    # Two frames: [0, 0] and [2, 2] => mean=1, deviations=[-1,+1], rms=1
    a = np.zeros((2, 2))
    b = np.full((2, 2), 2.0)
    paths = _write_npy_files([a, b])
    try:
        result = time_rms(paths)
        assert np.allclose(result, 1.0, atol=1e-10)
    finally:
        for p in paths:
            os.unlink(p)


# ---------------------------------------------------------------------------
# 4. flatfield_normalize — output mean close to 1.0
# ---------------------------------------------------------------------------

def test_flatfield_normalize_mean():
    rng = np.random.default_rng(42)
    # Simulate a non-uniform image: base illumination + checkerboard
    h, w = 64, 64
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    illumination = 1.0 + 0.3 * np.sin(2 * np.pi * x / w)
    pattern = 0.1 * np.sin(2 * np.pi * x / 4) * np.sin(2 * np.pi * y / 4)
    img = (illumination + pattern + rng.normal(0, 0.01, (h, w))).astype(np.float64)
    out = flatfield_normalize(img, sigma=20.0)
    # After normalization, mean should be close to 1.0
    assert abs(out.mean() - 1.0) < 0.05, f"mean={out.mean()}"


# ---------------------------------------------------------------------------
# 5. carrier_wavelength_mm
# ---------------------------------------------------------------------------

def test_carrier_wavelength_mm():
    assert carrier_wavelength_mm(1.2) == pytest.approx(0.6)


def test_carrier_wavelength_mm_various():
    assert carrier_wavelength_mm(2.0) == pytest.approx(1.0)
    assert carrier_wavelength_mm(0.5) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 6. StageEvent — field construction
# ---------------------------------------------------------------------------

def test_stage_event_construction():
    ev = StageEvent(
        kind="start",
        stage="compute",
        batch="batch_01",
        frame_idx=0,
        substage="flatfield",
        progress=0.0,
        total=100,
        completed=0,
        metrics={"fps": 30.0},
        run_id="run-001",
        seq=1,
    )
    assert ev.kind == "start"
    assert ev.stage == "compute"
    assert ev.batch == "batch_01"
    assert ev.frame_idx == 0
    assert ev.substage == "flatfield"
    assert ev.progress == 0.0
    assert ev.total == 100
    assert ev.completed == 0
    assert ev.metrics == {"fps": 30.0}
    assert ev.run_id == "run-001"
    assert ev.seq == 1
    assert ev.timestamp > 0


def test_stage_event_defaults():
    ev = StageEvent(
        kind="finish",
        stage="postprocess",
        batch=None,
        frame_idx=None,
        substage=None,
        progress=None,
        total=None,
        completed=None,
    )
    assert ev.metrics == {}
    assert ev.run_id == ""
    assert ev.seq == 0
    assert ev.timestamp > 0


def test_stage_event_frozen():
    ev = StageEvent(
        kind="log",
        stage="preprocess",
        batch=None,
        frame_idx=None,
        substage=None,
        progress=None,
        total=None,
        completed=None,
    )
    with pytest.raises((AttributeError, TypeError)):
        ev.kind = "error"  # type: ignore
