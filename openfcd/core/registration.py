"""Image registration utilities for FCD preprocessing.

The main entry point is ``scale_normalize_reference``, which detects a carrier-period
mismatch between the reference and deformed flatfield images and rescales the reference
to match the deformed image's spatial scale.  When ref and def were captured at the
same camera zoom the function is a no-op (returns inputs unchanged).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openfcd.core.fcd import Carrier

log = logging.getLogger(__name__)

_SCALE_TOLERANCE = 0.005  # < 0.5 % → no correction (empirical:
                          #  a 2 % mismatch already randomises FCD phase;
                          #  zoom + carrier-recompute cost is negligible,
                          #  so prefer aggressive normalisation.)
_SCALE_WARN_THRESHOLD = 0.10  # ≥ 10 % → emit warning


def _mean_carrier_period_px(carriers: list[Carrier]) -> float:
    """Return mean carrier spatial period in pixels."""
    return float(np.mean(
        [(2.0 * np.pi) / np.linalg.norm(c.k_loc) for c in carriers]
    ))


def scale_normalize_reference(
    ref_ff: np.ndarray,
    def_ff: np.ndarray,
    carriers_ref: list[Carrier],
    *,
    tolerance: float = _SCALE_TOLERANCE,
) -> tuple[np.ndarray, list[Carrier], float, tuple[int, int, int, int] | None]:
    """Rescale *ref_ff* so its carrier period matches *def_ff*.

    Parameters
    ----------
    ref_ff:
        Flatfield-normalised reference image.
    def_ff:
        Flatfield-normalised deformed image.
    carriers_ref:
        Carriers already detected in *ref_ff*.
    tolerance:
        Fractional scale deviation below which no correction is applied.

    Returns
    -------
    aligned_ref_ff:
        Rescaled reference (or the original if no correction was needed).
        When scale < 1 this image is *smaller* than the input — see *valid_crop*.
    aligned_carriers:
        Carriers recomputed on *aligned_ref_ff* (or *carriers_ref* if unchanged).
    scale:
        Applied scale factor (``period_def / period_ref``).  1.0 means no change.
    valid_crop:
        ``(r0, c0, h, w)`` slice of *def_ff* that spatially corresponds to
        *aligned_ref_ff* when scale < 1.  The caller should crop *def_ff* to
        ``def_ff[r0:r0+h, c0:c0+w]`` before running FCD.  ``None`` when the
        full images are used (scale ≥ 1 or no correction applied).
    """
    from openfcd.core.fcd import calculate_carriers
    from scipy.ndimage import zoom

    # Detect carriers in deformed image
    try:
        carriers_def = calculate_carriers(def_ff - def_ff.mean())
    except RuntimeError:
        log.debug("scale_normalize_reference: carrier detection failed on def — skipping")
        return ref_ff, carriers_ref, 1.0, None

    period_ref = _mean_carrier_period_px(carriers_ref)
    period_def = _mean_carrier_period_px(carriers_def)

    if period_ref <= 0 or period_def <= 0:
        return ref_ff, carriers_ref, 1.0, None

    scale = period_def / period_ref
    deviation = abs(scale - 1.0)

    if deviation <= tolerance:
        log.debug(
            "scale_normalize_reference: scale=%.4f within tolerance — no correction",
            scale,
        )
        return ref_ff, carriers_ref, scale, None

    if deviation >= _SCALE_WARN_THRESHOLD:
        log.warning(
            "scale_normalize_reference: carrier period mismatch %.1f%% "
            "(ref=%.1f px, def=%.1f px) — applying scale correction. "
            "Results may be approximate; recapture reference at the same zoom for best accuracy.",
            deviation * 100,
            period_ref,
            period_def,
        )
    else:
        log.info(
            "scale_normalize_reference: correcting %.1f%% scale mismatch "
            "(ref=%.1f px → def=%.1f px)",
            deviation * 100,
            period_ref,
            period_def,
        )

    h, w = ref_ff.shape
    ref_zoomed = zoom(ref_ff, scale, order=1)
    hz, wz = ref_zoomed.shape

    if scale > 1.0:
        # Zoomed image is larger — take the centre crop; full image remains valid.
        r0 = (hz - h) // 2
        c0 = (wz - w) // 2
        ref_aligned = ref_zoomed[r0: r0 + h, c0: c0 + w]
        carriers_aligned = calculate_carriers(ref_aligned - ref_aligned.mean())
        return ref_aligned, carriers_aligned, scale, None
    else:
        # Zoomed image is smaller. Return it as-is (no padding) and tell the
        # caller which centre slice of def_ff corresponds to this ref.
        # Padding with a constant would leave def's carrier unpaired in the
        # border region, causing checkerboard leakage through FCD integration.
        r0 = (h - hz) // 2
        c0 = (w - wz) // 2
        carriers_aligned = calculate_carriers(ref_zoomed - ref_zoomed.mean())
        return ref_zoomed, carriers_aligned, scale, (r0, c0, hz, wz)
