"""End-to-end wave-stats test on a spatially decaying sine wave (v2).

Two segments (fore-like at low s, aft-like at high s) replace the v1
binary fore/aft model. Same physics, same expectations.
"""

from __future__ import annotations

import numpy as np
import pytest

from openfcd.core.wave_stats import (
    FrameWaveStats,
    SegmentWaveStats,
    compute_frame_wave_stats,
)


# ── Physical constants ───────────────────────────────────────────────────────

LAMBDA_MM = 10.0
ALPHA = 0.05
A0 = 1.0
FREQ_HZ = 20.0
OMEGA = 2.0 * np.pi * FREQ_HZ
N_FRAMES = 30
FRAME_RATE = 30.0

PX_PER_MM = 10.0
H = 100
L_MM = 100.0
W = int(L_MM * PX_PER_MM) + 1

# s_mm is centered on the line midpoint (s=0 at L_MM/2). Originally these
# segments lived on a [0, L_MM] axis as (5, 30) and (60, 95); they are
# translated by -L_MM/2 here to preserve identical pixel coverage.
_HALF = L_MM / 2.0
FORE_LO, FORE_HI = 5.0 - _HALF, 30.0 - _HALF   # = (-45.0, -20.0): low-s end, near-field
AFT_LO, AFT_HI = 60.0 - _HALF, 95.0 - _HALF    # = (10.0, 45.0): high-s end, far-field
SEGMENTS: list[tuple[float, float]] = [(FORE_LO, FORE_HI), (AFT_LO, AFT_HI)]
FORE_IDX, AFT_IDX = 0, 1

PROMINENCE_K = 0.3
WAVELENGTH_TOL_MM = 0.3
WAVELENGTH_STD_MAX = 0.2
HEIGHT_RATIO_MIN = 2.0


def _make_eta_frame(t: float) -> np.ndarray:
    cols = np.arange(W, dtype=np.float64)
    s_mm = cols / PX_PER_MM
    eta_row = A0 * np.exp(-ALPHA * s_mm) * np.sin(
        2.0 * np.pi * s_mm / LAMBDA_MM - OMEGA * t
    )
    return np.tile(eta_row, (H, 1))


def _profile_line():
    return (float(H // 2), 0.0), (float(H // 2), float(W - 1))


def _run_frame(t: float) -> FrameWaveStats:
    return compute_frame_wave_stats(
        eta=_make_eta_frame(t),
        profile_line=_profile_line(),
        body_polygon_rc=None,
        px_per_mm=PX_PER_MM,
        segments=SEGMENTS,
        prominence_k=PROMINENCE_K,
    )


@pytest.fixture(scope="module")
def all_frame_results() -> list[FrameWaveStats]:
    return [_run_frame(i / FRAME_RATE) for i in range(N_FRAMES)]


def test_fore_wavelength_near_lambda(all_frame_results):
    errors = [
        abs(res.segments[FORE_IDX].wavelength_mm - LAMBDA_MM)
        for res in all_frame_results
        if np.isfinite(res.segments[FORE_IDX].wavelength_mm)
    ]
    assert errors
    assert max(errors) < WAVELENGTH_TOL_MM


def test_aft_wavelength_near_lambda(all_frame_results):
    errors = [
        abs(res.segments[AFT_IDX].wavelength_mm - LAMBDA_MM)
        for res in all_frame_results
        if np.isfinite(res.segments[AFT_IDX].wavelength_mm)
    ]
    assert errors
    assert max(errors) < WAVELENGTH_TOL_MM


def test_wavelength_stable_across_frames(all_frame_results):
    wls = [
        res.segments[FORE_IDX].wavelength_mm
        for res in all_frame_results
        if np.isfinite(res.segments[FORE_IDX].wavelength_mm)
    ]
    assert len(wls) >= 2
    assert float(np.std(wls)) < WAVELENGTH_STD_MAX


def test_fore_heights_exceed_aft_heights(all_frame_results):
    fore_means, aft_means = [], []
    for res in all_frame_results:
        if len(res.segments[FORE_IDX].peak_to_trough_heights) > 0:
            fore_means.append(float(np.mean(res.segments[FORE_IDX].peak_to_trough_heights)))
        if len(res.segments[AFT_IDX].peak_to_trough_heights) > 0:
            aft_means.append(float(np.mean(res.segments[AFT_IDX].peak_to_trough_heights)))
    assert fore_means and aft_means
    assert float(np.mean(fore_means)) > float(np.mean(aft_means))


def test_fore_aft_height_ratio(all_frame_results):
    fore_means, aft_means = [], []
    for res in all_frame_results:
        if len(res.segments[FORE_IDX].peak_to_trough_heights) > 0:
            fore_means.append(float(np.mean(res.segments[FORE_IDX].peak_to_trough_heights)))
        if len(res.segments[AFT_IDX].peak_to_trough_heights) > 0:
            aft_means.append(float(np.mean(res.segments[AFT_IDX].peak_to_trough_heights)))
    if not fore_means or not aft_means:
        pytest.skip("insufficient height data")
    ratio = float(np.mean(fore_means)) / float(np.mean(aft_means))
    assert ratio > HEIGHT_RATIO_MIN


def test_fore_heights_monotone_at_t0(all_frame_results):
    first = all_frame_results[0]
    h = first.segments[FORE_IDX].peak_to_trough_heights
    if len(h) < 2:
        pytest.skip("need >=2 height pairs at t=0")
    for i in range(len(h) - 1):
        assert h[i] >= h[i + 1] - 0.20 * h[i]


def test_both_segments_populated(all_frame_results):
    for i, res in enumerate(all_frame_results):
        assert len(res.segments) == 2
        for seg in res.segments:
            assert isinstance(seg, SegmentWaveStats)


def test_peak_trough_arrays_consistent(all_frame_results):
    for i, res in enumerate(all_frame_results):
        for seg in res.segments:
            assert len(seg.peaks_s_mm) == seg.n_peaks
            assert len(seg.peaks_eta) == seg.n_peaks
            assert len(seg.troughs_s_mm) == seg.n_troughs
            assert len(seg.troughs_eta) == seg.n_troughs
