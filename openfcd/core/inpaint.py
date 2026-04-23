"""Replace masked (occluded) pixels with a synthetic checkerboard.

The synthetic pattern is reconstructed from the two carrier peaks of the
reference image: this preserves both the pattern phase and orientation
exactly, so once we re-blend the intensity to match local statistics the
patch is invisible to FCD.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.fft import fft2, ifft2

from openfcd.core.fcd import Carrier


def synthesize_from_carriers(i_ref: np.ndarray,
                              carriers: Sequence[Carrier]) -> np.ndarray:
    """Rebuild a clean checkerboard with the same phase as the reference."""
    ref_fft = fft2(i_ref)
    keep = np.zeros_like(ref_fft)
    keep[0, 0] = ref_fft[0, 0]                 # DC
    for c in carriers:
        keep[c.mask] = ref_fft[c.mask]         # carrier band + its conjugate
    syn = np.real(ifft2(keep))
    return syn


def intensity_match(target_patch: np.ndarray,
                    syn_patch: np.ndarray,
                    surround_target: np.ndarray,
                    surround_syn: np.ndarray) -> np.ndarray:
    """Linearly remap syn_patch so its statistics match the surround of target."""
    a_t = surround_target.std() + 1e-6
    a_s = surround_syn.std() + 1e-6
    a = a_t / a_s
    b = surround_target.mean() - a * surround_syn.mean()
    return a * syn_patch + b


def inpaint_fft(img: np.ndarray,
                mask: np.ndarray,
                syn: np.ndarray,
                surround_dilate_px: int = 20) -> np.ndarray:
    """Fill ``mask`` region of ``img`` with intensity-matched ``syn``."""
    if not mask.any():
        return img.astype(float).copy()
    from skimage.morphology import dilation, disk
    surround = dilation(mask, disk(surround_dilate_px)) & ~mask
    if not surround.any():
        surround = ~mask
    out = img.astype(float).copy()
    matched = intensity_match(img[mask], syn[mask], img[surround], syn[surround])
    out[mask] = matched
    return out
