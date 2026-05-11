"""End-to-end guard: compute_frame on pure-wave def + clean ref must not NaN wave fronts.

Locks in the call-site contract at openfcd/pipeline/frame.py — that
detect_filament_occluders is invoked on ref_ff, not def_ff. Reverting that
single line MUST cause this test to fail; otherwise the regression guard
is meaningless.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from openfcd.pipeline.compute import GeomParams
from openfcd.pipeline.frame import FrameInputs, ProcessParams, compute_frame


def _checkerboard(size: int = 256) -> np.ndarray:
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    return (np.sin(2 * np.pi * x / 8) * np.sin(2 * np.pi * y / 8) * 50 + 128).astype(float)


def _wave_warp(img: np.ndarray, amp_px: float, lam_px: float) -> np.ndarray:
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dy = amp_px * np.sin(2 * np.pi * r / lam_px)
    return map_coordinates(img, [yy + dy, xx], order=1, mode="reflect")


def test_compute_frame_pure_wave_no_nan_slivers() -> None:
    ref = _checkerboard()
    deformed = _wave_warp(ref, amp_px=2.5, lam_px=40.0)

    process = ProcessParams(
        flatfield_sigma_auto=False,
        flatfield_sigma=60.0,
        taper_alpha=0.08,
        edge_nan_mm=0.0,
        detrend="plane",
        auto_scale_ref=False,
        highpass_sigma_px=0.0,
        small_hole_fill_radius_mm=0.0,
    )
    geom = GeomParams(pattern_period_mm=1.0, alpha=1.0, h_p_eff_mm=10.0)
    inputs = FrameInputs(
        ref_img=ref,
        def_img=deformed,
        geom=geom,
        process=process,
        roi_box=None,
        robot_poly=None,
        fast_preview=False,
        ref_shape=ref.shape,
    )
    result = compute_frame(inputs)
    eta = np.asarray(result.eta_mm, dtype=np.float64)
    nan_count = int(np.isnan(eta).sum())
    # Empirically calibrated discriminator: with the bug (def_ff), this scene
    # produces ~940 NaN pixels (~1.43%); the fix (ref_ff) drives it to 0.
    # The 100-pixel ceiling cleanly separates the two states while leaving
    # headroom for legitimate edge effects on slightly different inputs.
    assert nan_count <= 100, (
        f"Pure-wave compute_frame produced {nan_count} NaN pixels "
        f"({100 * nan_count / eta.size:.3f}% of {eta.size}); expected <= 100. "
        "If this fails, frame.py likely calls detect_filament_occluders on def_ff "
        "rather than ref_ff."
    )
