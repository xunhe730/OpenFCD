"""Per-segment wave-field statistics along a spatial profile line (v2).

Pure functions — no IO, no Qt, no h5py dependencies.

The profile line is sampled at 1-pixel resolution via bilinear interpolation.
The arc-length axis ``s_mm`` is **centered on the line midpoint** (s=0), with
``s ∈ [-L/2, +L/2]`` where ``L`` is the line length in millimetres. This
matches the centered ``x_mm`` axis used by ``profile_composite`` for the 1D
sub-plot, so user-drawn segment boundaries read directly off that plot can be
fed in verbatim. Each user-defined segment ``[s_lo_mm, s_hi_mm]`` is analysed
independently using a prominence-gated peak finder whose minimum distance is
estimated from the dominant FFT wavelength.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks


# ── Output dataclasses ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class SegmentWaveStats:
    """Wave statistics for a single user-defined segment on one frame."""
    segment_idx: int                       # 0-based index within the frame's segment list
    s_lo_mm: float
    s_hi_mm: float
    peaks_s_mm: np.ndarray
    peaks_eta: np.ndarray
    troughs_s_mm: np.ndarray
    troughs_eta: np.ndarray
    peak_to_trough_heights: np.ndarray
    wavelength_mm: float
    wavenumber_per_mm: float
    n_peaks: int
    n_troughs: int
    lambda0_fft_mm: float


@dataclass(frozen=True)
class FrameWaveStats:
    """Per-frame wave statistics across all visible segments."""
    segments: list[SegmentWaveStats]


# ── Internal helpers ─────────────────────────────────────────────────────────


def _empty_segment_stats(segment_idx: int, s_lo: float, s_hi: float) -> SegmentWaveStats:
    empty = np.empty(0, dtype=np.float64)
    return SegmentWaveStats(
        segment_idx=segment_idx,
        s_lo_mm=float(s_lo),
        s_hi_mm=float(s_hi),
        peaks_s_mm=empty,
        peaks_eta=empty,
        troughs_s_mm=empty,
        troughs_eta=empty,
        peak_to_trough_heights=empty,
        wavelength_mm=float("nan"),
        wavenumber_per_mm=float("nan"),
        n_peaks=0,
        n_troughs=0,
        lambda0_fft_mm=float("nan"),
    )


def _estimate_lambda0_fft(eta_window: np.ndarray, ds_mm: float) -> float:
    """Estimate dominant wavelength in mm via zero-padded FFT.

    Returns NaN if the window has fewer than 8 finite samples.
    """
    valid = np.isfinite(eta_window)
    if valid.sum() < 8:
        return float("nan")
    n = len(eta_window)
    y = eta_window.copy()
    y[~valid] = 0.0
    y -= float(np.nanmean(y))
    y *= np.hanning(n)
    fft_vals = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(n, d=ds_mm)
    if len(freqs) < 2:
        return float("nan")
    fft_vals[0] = 0.0
    peak_idx = int(np.argmax(fft_vals))
    f_peak = freqs[peak_idx]
    if f_peak <= 0.0:
        return float("nan")
    return float(1.0 / f_peak)


def _compute_heights(
    peaks_s: np.ndarray,
    peaks_eta: np.ndarray,
    troughs_s: np.ndarray,
    troughs_eta: np.ndarray,
) -> np.ndarray:
    """Adjacent peak-trough height magnitudes |eta_peak - eta_trough|."""
    if len(peaks_s) == 0 or len(troughs_s) == 0:
        return np.empty(0, dtype=np.float64)

    kinds: list[str] = (["peak"] * len(peaks_s)) + (["trough"] * len(troughs_s))
    s_all = np.concatenate([peaks_s, troughs_s])
    eta_all = np.concatenate([peaks_eta, troughs_eta])
    order = np.argsort(s_all, kind="stable")
    s_all = s_all[order]
    eta_all = eta_all[order]
    kinds = [kinds[i] for i in order]

    heights: list[float] = []
    for i in range(len(kinds) - 1):
        if kinds[i] != kinds[i + 1]:
            heights.append(abs(float(eta_all[i]) - float(eta_all[i + 1])))

    return np.array(heights, dtype=np.float64)


def _compute_one_segment(
    segment_idx: int,
    eta_profile: np.ndarray,
    s_profile_mm: np.ndarray,
    s_lo: float,
    s_hi: float,
    ds_mm: float,
    prominence_k: float,
) -> SegmentWaveStats:
    """Compute wave statistics for the slice ``s_lo <= s <= s_hi``."""
    s_lo_eff = max(float(s_lo), float(s_profile_mm[0]))
    s_hi_eff = min(float(s_hi), float(s_profile_mm[-1]))

    if s_hi_eff <= s_lo_eff:
        return _empty_segment_stats(segment_idx, s_lo, s_hi)

    mask = (s_profile_mm >= s_lo_eff) & (s_profile_mm <= s_hi_eff)
    eta_window = eta_profile[mask]
    s_window = s_profile_mm[mask]

    if len(eta_window) == 0:
        return _empty_segment_stats(segment_idx, s_lo, s_hi)

    lambda0 = _estimate_lambda0_fft(eta_window, ds_mm)

    n_pts = len(eta_window)
    if np.isfinite(lambda0) and lambda0 > 0.0:
        distance = max(2, int(0.5 * lambda0 / ds_mm))
    else:
        distance = max(2, int(0.05 * n_pts))

    sigma = float(np.nanstd(eta_window))
    prominence = prominence_k * sigma

    eta_clean = np.where(np.isfinite(eta_window), eta_window, 0.0)

    peaks_idx, _ = find_peaks(eta_clean, distance=distance, prominence=prominence)
    troughs_idx, _ = find_peaks(-eta_clean, distance=distance, prominence=prominence)

    peaks_s_mm = s_window[peaks_idx] if len(peaks_idx) else np.empty(0, dtype=np.float64)
    peaks_eta = eta_clean[peaks_idx] if len(peaks_idx) else np.empty(0, dtype=np.float64)
    troughs_s_mm = s_window[troughs_idx] if len(troughs_idx) else np.empty(0, dtype=np.float64)
    troughs_eta = eta_clean[troughs_idx] if len(troughs_idx) else np.empty(0, dtype=np.float64)

    n_peaks = int(len(peaks_idx))
    n_troughs = int(len(troughs_idx))

    lambda_parts: list[float] = []
    if n_peaks >= 2:
        lambda_parts.append(float(np.mean(np.diff(peaks_s_mm))))
    if n_troughs >= 2:
        lambda_parts.append(float(np.mean(np.diff(troughs_s_mm))))

    if lambda_parts:
        wavelength_mm = float(np.mean(lambda_parts))
        wavenumber_per_mm = (1.0 / wavelength_mm) if wavelength_mm != 0.0 else float("nan")
    else:
        wavelength_mm = float("nan")
        wavenumber_per_mm = float("nan")

    heights = _compute_heights(peaks_s_mm, peaks_eta, troughs_s_mm, troughs_eta)

    return SegmentWaveStats(
        segment_idx=segment_idx,
        s_lo_mm=float(s_lo),
        s_hi_mm=float(s_hi),
        peaks_s_mm=peaks_s_mm,
        peaks_eta=peaks_eta,
        troughs_s_mm=troughs_s_mm,
        troughs_eta=troughs_eta,
        peak_to_trough_heights=heights,
        wavelength_mm=wavelength_mm,
        wavenumber_per_mm=wavenumber_per_mm,
        n_peaks=n_peaks,
        n_troughs=n_troughs,
        lambda0_fft_mm=lambda0,
    )


# ── Public API ───────────────────────────────────────────────────────────────


def compute_frame_wave_stats(
    eta: np.ndarray,
    profile_line: tuple[tuple[float, float], tuple[float, float]],
    body_polygon_rc: np.ndarray | None,
    px_per_mm: float,
    segments: list[tuple[float, float]],
    prominence_k: float = 0.3,
) -> FrameWaveStats:
    """Compute per-segment wave statistics along a spatial profile line (v2).

    Parameters
    ----------
    eta : (H, W) float64 eta field for one frame.
    profile_line : ((r0, c0), (r1, c1)) pixel-coordinate endpoints.
    body_polygon_rc : (N, 2) polygon vertices in (row, col) px, or None.
        Reserved for future use; not consulted in v2 (fore/aft split removed).
    px_per_mm : pixels per mm (scalar, > 0).
    segments : list of (s_lo_mm, s_hi_mm) tuples for each segment to analyse.
        Caller passes only segments that should actually be computed
        (e.g. ``visible=True`` ones).
    prominence_k : fraction of sigma used as find_peaks prominence threshold.

    Returns
    -------
    FrameWaveStats with one ``SegmentWaveStats`` entry per input segment, in
    the same order as ``segments``.
    """
    (r0, c0), (r1, c1) = profile_line
    dr = float(r1) - float(r0)
    dc = float(c1) - float(c0)
    L_px = float(np.hypot(dr, dc))

    if not segments:
        return FrameWaveStats(segments=[])

    if L_px == 0.0 or px_per_mm <= 0.0:
        return FrameWaveStats(
            segments=[
                _empty_segment_stats(idx, s_lo, s_hi)
                for idx, (s_lo, s_hi) in enumerate(segments)
            ]
        )

    # Sample eta along the profile line at ~1 px resolution
    ds_px = 1.0
    n_samples = max(2, int(L_px / ds_px) + 1)
    t_vals = np.linspace(0.0, 1.0, n_samples)
    rows = float(r0) + t_vals * dr
    cols = float(c0) + t_vals * dc

    eta_profile = map_coordinates(
        np.asarray(eta, dtype=np.float64),
        [rows, cols],
        order=1,
        mode="constant",
        cval=float("nan"),
    )

    ds_mm = 1.0 / px_per_mm
    L_mm = L_px * ds_mm
    # Centered arc-length axis: s=0 at the line midpoint, matching the
    # centered x_mm axis used by profile_composite's 1D sub-plot.
    s_profile_mm = t_vals * L_mm - 0.5 * L_mm

    out: list[SegmentWaveStats] = []
    for idx, (s_lo, s_hi) in enumerate(segments):
        out.append(
            _compute_one_segment(
                segment_idx=idx,
                eta_profile=eta_profile,
                s_profile_mm=s_profile_mm,
                s_lo=float(s_lo),
                s_hi=float(s_hi),
                ds_mm=ds_mm,
                prominence_k=prominence_k,
            )
        )

    return FrameWaveStats(segments=out)
