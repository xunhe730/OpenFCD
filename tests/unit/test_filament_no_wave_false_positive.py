"""Filament detector must not flag wave-induced carrier-amplitude dips.

Companion fix: openfcd/pipeline/frame.py calls detect_filament_occluders on
the reference image (ref_ff), not the deformed image (def_ff). Wave fronts
on def_ff locally compress the checkerboard carrier and produce elongated
low-amplitude bands that pass the filament shape filter (aspect>=2.5,
length>=10, width<=14, area>=12), causing spurious NaN slivers in eta.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from openfcd.core.fcd import calculate_carriers
from openfcd.core.mask import auto_mask, detect_filament_occluders


def _checkerboard(size: int = 256) -> np.ndarray:
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    return (np.sin(2 * np.pi * x / 8) * np.sin(2 * np.pi * y / 8) * 50 + 128).astype(float)


def _wave_warp(img: np.ndarray, amp_px: float, lam_px: float) -> np.ndarray:
    """Apply a radial sinusoidal vertical pixel displacement to mimic surface waves."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dy = amp_px * np.sin(2 * np.pi * r / lam_px)
    return map_coordinates(img, [yy + dy, xx], order=1, mode="reflect")


def test_pure_wave_deformation_does_not_trigger_filament_on_reference() -> None:
    """Reference image (no waves, no filaments) must produce empty filament mask."""
    ref = _checkerboard()
    carriers = calculate_carriers(ref - ref.mean())
    mask_ref = detect_filament_occluders(ref, carriers)
    assert mask_ref.sum() == 0, (
        "Clean reference checkerboard should yield zero filament hits, "
        f"got {int(mask_ref.sum())} flagged pixels."
    )


def test_pure_wave_deformation_would_false_positive_on_deformed() -> None:
    """Lock in the documented bug mechanism.

    Running detect_filament_occluders on def_ff with a pure wave deformation
    and no actual filaments must produce non-zero hits — this is exactly the
    pathology that motivated switching the call site in frame.py to ref_ff.
    If this assertion ever fails, the underlying false-positive mechanism has
    changed and the rationale for the fix should be reconsidered.
    """
    ref = _checkerboard()
    deformed = _wave_warp(ref, amp_px=2.5, lam_px=40.0)
    carriers = calculate_carriers(ref - ref.mean())
    mask_def = detect_filament_occluders(deformed, carriers)
    assert mask_def.sum() > 0, (
        "Pre-fix sanity: pure-wave def is expected to false-positive when "
        "filament detection runs on def. If this passes, the underlying "
        "amplitude-dip mechanism may have changed."
    )


def test_auto_mask_threshold_robust_to_wave_deformation() -> None:
    """The 0.2-threshold auto_mask companion path must not be wave-sensitive.

    auto_mask feeds carrier_loss_mask, which OR-merges into occlusion_mask
    in compute_frame. If the 0.2 threshold ever drops low enough that
    wave-induced dips trigger auto_mask, NaN slivers reappear via that path
    even after the filament-detector fix. This test guards that boundary.
    """
    ref = _checkerboard()
    deformed = _wave_warp(ref, amp_px=2.5, lam_px=40.0)
    carriers = calculate_carriers(ref - ref.mean())
    mask_ref = auto_mask(ref, carriers, threshold_ratio=0.2, dilate_px=4)
    mask_def = auto_mask(deformed, carriers, threshold_ratio=0.2, dilate_px=4)
    diff = abs(int(mask_def.sum()) - int(mask_ref.sum()))
    assert diff <= 20, (
        f"auto_mask diverges by {diff} pixels between ref and def on a "
        "pure-wave scene; the 0.2 threshold may also be wave-sensitive. "
        "Consider extending the ref-vs-def fix to auto_mask."
    )
