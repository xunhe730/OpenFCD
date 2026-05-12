"""Tests for openfcd/core/wave_stats.py — compute_frame_wave_stats (v2)."""

from __future__ import annotations

import numpy as np
import pytest

from openfcd.core.wave_stats import (
    FrameWaveStats,
    SegmentWaveStats,
    _compute_heights,
    _estimate_lambda0_fft,
    compute_frame_wave_stats,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_sine_eta(
    H: int,
    W: int,
    lambda_mm: float,
    amplitude: float,
    px_per_mm: float,
    x_offset_px: float = 0.0,
) -> np.ndarray:
    cols = np.arange(W, dtype=np.float64)
    s_mm = (cols - x_offset_px) / px_per_mm
    eta_row = amplitude * np.sin(2.0 * np.pi * s_mm / lambda_mm)
    return np.tile(eta_row, (H, 1))


def _horizontal_profile(W: int, row: float = 50.0) -> tuple[tuple[float, float], tuple[float, float]]:
    return (row, 0.0), (row, float(W - 1))


# ── Test 1: pure sine ────────────────────────────────────────────────────────


def test_pure_sine_wavelength():
    H, W = 100, 600
    px_per_mm = 10.0
    lambda_mm = 10.0
    A = 1.0
    eta = _make_sine_eta(H, W, lambda_mm, A, px_per_mm)
    L_mm = (W - 1) / px_per_mm
    mid_mm = L_mm / 2.0

    result = compute_frame_wave_stats(
        eta=eta,
        profile_line=_horizontal_profile(W, row=50.0),
        body_polygon_rc=None,
        px_per_mm=px_per_mm,
        segments=[(0.0, mid_mm), (mid_mm, L_mm)],
        prominence_k=0.15,
    )

    assert isinstance(result, FrameWaveStats)
    assert len(result.segments) == 2
    for i, seg in enumerate(result.segments):
        assert seg.segment_idx == i
        assert np.isfinite(seg.wavelength_mm), f"segment {i} wavelength NaN"
        rel_err = abs(seg.wavelength_mm - lambda_mm) / lambda_mm
        assert rel_err < 0.01, f"segment {i} wavelength error {rel_err:.3%}"
        assert np.isfinite(seg.wavenumber_per_mm)


# ── Test 2: damped sine ──────────────────────────────────────────────────────


def test_damped_sine_wavelength_and_heights():
    H, W = 100, 600
    px_per_mm = 10.0
    lambda_mm = 20.0
    A = 2.0
    alpha = 0.03

    cols = np.arange(W, dtype=np.float64)
    s_mm = cols / px_per_mm
    eta_row = A * np.exp(-alpha * s_mm) * np.sin(2.0 * np.pi * s_mm / lambda_mm)
    eta = np.tile(eta_row, (H, 1))

    L_mm = (W - 1) / px_per_mm
    mid_mm = L_mm / 2.0

    result = compute_frame_wave_stats(
        eta=eta,
        profile_line=_horizontal_profile(W, row=50.0),
        body_polygon_rc=None,
        px_per_mm=px_per_mm,
        segments=[(0.0, mid_mm), (mid_mm, L_mm)],
        prominence_k=0.10,
    )

    fore = result.segments[0]
    if len(fore.peak_to_trough_heights) >= 2:
        h = fore.peak_to_trough_heights
        for i in range(len(h) - 1):
            assert h[i] >= h[i + 1] - 0.05 * h[i]

    side = fore if np.isfinite(fore.wavelength_mm) else result.segments[1]
    if np.isfinite(side.wavelength_mm):
        rel_err = abs(side.wavelength_mm - lambda_mm) / lambda_mm
        assert rel_err < 0.10


# ── Test 3: noisy signal & prominence ────────────────────────────────────────


def test_noise_prominence_filtering():
    H, W = 100, 600
    px_per_mm = 10.0
    lambda_mm = 10.0
    A_signal = 1.0
    A_noise = A_signal / 5.0

    rng = np.random.default_rng(42)
    cols = np.arange(W, dtype=np.float64)
    s_mm = cols / px_per_mm
    signal = A_signal * np.sin(2.0 * np.pi * s_mm / lambda_mm)
    noise = A_noise * rng.standard_normal(W)
    eta = np.tile(signal + noise, (H, 1))

    L_mm = (W - 1) / px_per_mm
    mid_mm = L_mm / 2.0
    segs = [(0.0, mid_mm), (mid_mm, L_mm)]

    strict = compute_frame_wave_stats(
        eta=eta, profile_line=_horizontal_profile(W, 50.0),
        body_polygon_rc=None, px_per_mm=px_per_mm,
        segments=segs, prominence_k=0.3,
    )
    loose = compute_frame_wave_stats(
        eta=eta, profile_line=_horizontal_profile(W, 50.0),
        body_polygon_rc=None, px_per_mm=px_per_mm,
        segments=segs, prominence_k=0.01,
    )
    strict_n = sum(s.n_peaks for s in strict.segments)
    loose_n = sum(s.n_peaks for s in loose.segments)
    assert strict_n <= loose_n

    for r in (strict, loose):
        for seg in r.segments:
            assert isinstance(seg, SegmentWaveStats)
            assert len(seg.peaks_s_mm) == seg.n_peaks
            assert len(seg.troughs_s_mm) == seg.n_troughs


# ── Test 4: short window → NaN wavelength ────────────────────────────────────


def test_short_window_nan_wavelength():
    H, W = 100, 400
    px_per_mm = 10.0
    rng = np.random.default_rng(0)
    eta = rng.standard_normal((H, W)) * 0.01

    result = compute_frame_wave_stats(
        eta=eta, profile_line=_horizontal_profile(W, 50.0),
        body_polygon_rc=None, px_per_mm=px_per_mm,
        segments=[(0.0, 1.5), (38.5, 39.9)],
        prominence_k=0.3,
    )

    for seg in result.segments:
        if seg.n_peaks < 2 and seg.n_troughs < 2:
            assert not np.isfinite(seg.wavelength_mm)
            assert not np.isfinite(seg.wavenumber_per_mm)
            assert len(seg.peak_to_trough_heights) == 0


# ── Test 5: empty segments → empty result ────────────────────────────────────


def test_empty_segments_returns_empty():
    H, W = 100, 400
    eta = np.zeros((H, W))
    result = compute_frame_wave_stats(
        eta=eta, profile_line=_horizontal_profile(W, 50.0),
        body_polygon_rc=None, px_per_mm=10.0,
        segments=[], prominence_k=0.3,
    )
    assert isinstance(result, FrameWaveStats)
    assert result.segments == []


# ── Test 6: body_polygon_rc=None must not crash ──────────────────────────────


def test_none_polygon_no_crash():
    H, W = 100, 400
    px_per_mm = 10.0
    lambda_mm = 20.0
    eta = _make_sine_eta(H, W, lambda_mm, 1.0, px_per_mm)
    L_mm = (W - 1) / px_per_mm

    result = compute_frame_wave_stats(
        eta=eta, profile_line=_horizontal_profile(W, 50.0),
        body_polygon_rc=None, px_per_mm=px_per_mm,
        segments=[(0.0, L_mm / 2.0), (L_mm / 2.0, L_mm)],
        prominence_k=0.15,
    )
    assert len(result.segments) == 2


# ── Test 7: N=3 segments ─────────────────────────────────────────────────────


def test_three_segments():
    H, W = 100, 600
    px_per_mm = 10.0
    lambda_mm = 10.0
    eta = _make_sine_eta(H, W, lambda_mm, 1.0, px_per_mm)
    L_mm = (W - 1) / px_per_mm

    result = compute_frame_wave_stats(
        eta=eta, profile_line=_horizontal_profile(W, 50.0),
        body_polygon_rc=None, px_per_mm=px_per_mm,
        segments=[
            (0.0, L_mm / 3.0),
            (L_mm / 3.0, 2.0 * L_mm / 3.0),
            (2.0 * L_mm / 3.0, L_mm),
        ],
        prominence_k=0.15,
    )
    assert len(result.segments) == 3
    for i, seg in enumerate(result.segments):
        assert seg.segment_idx == i


# ── Test 8: degenerate profile_line ──────────────────────────────────────────


def test_degenerate_profile_line():
    H, W = 100, 400
    eta = np.zeros((H, W))
    result = compute_frame_wave_stats(
        eta=eta, profile_line=((50.0, 100.0), (50.0, 100.0)),
        body_polygon_rc=None, px_per_mm=10.0,
        segments=[(0.0, 10.0)], prominence_k=0.3,
    )
    assert len(result.segments) == 1
    seg = result.segments[0]
    assert seg.n_peaks == 0
    assert not np.isfinite(seg.wavelength_mm)


# ── Helper unit tests ────────────────────────────────────────────────────────


def test_estimate_lambda0_fft():
    ds_mm = 0.1
    n = 200
    lambda_mm = 10.0
    s = np.arange(n) * ds_mm
    y = np.sin(2.0 * np.pi * s / lambda_mm)
    lam = _estimate_lambda0_fft(y, ds_mm)
    assert np.isfinite(lam)
    assert abs(lam - lambda_mm) / lambda_mm < 0.05


def test_estimate_lambda0_fft_short_window():
    lam = _estimate_lambda0_fft(np.array([1.0, 2.0, 1.0]), ds_mm=0.1)
    assert not np.isfinite(lam)


def test_compute_heights_basic():
    peaks_s = np.array([1.0, 3.0, 5.0])
    peaks_eta = np.array([1.0, 1.0, 1.0])
    troughs_s = np.array([2.0, 4.0])
    troughs_eta = np.array([-1.0, -1.0])
    heights = _compute_heights(peaks_s, peaks_eta, troughs_s, troughs_eta)
    assert len(heights) == 4
    assert np.allclose(heights, 2.0)


def test_compute_heights_empty():
    empty = np.empty(0)
    h = _compute_heights(empty, empty, empty, empty)
    assert len(h) == 0
