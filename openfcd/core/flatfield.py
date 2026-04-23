"""Flat-field correction for non-uniform illumination."""
import numpy as np
from scipy.ndimage import gaussian_filter


def flatfield_normalize(img: np.ndarray, sigma: float = 300.0,
                        bg_src: np.ndarray | None = None) -> np.ndarray:
    """Divide out a low-frequency background to flatten illumination.

    `sigma` should be much larger than the checkerboard period so the
    background estimate does not eat into the carrier signal. For a
    1.2 mm pattern at ~12 px/mm, the period is ~14 px; sigma=80 px is safe.

    `bg_src`: if provided, the illumination envelope is estimated from this
    image instead of from `img` itself.  Pass the reference image for both
    reference and deformed calls so that both are divided by the *same*
    background — illumination-induced contrast differences then cancel exactly
    in the FCD phase difference, eliminating the taper/fftinvgrad saddle
    artifact seen in ref-vs-ref self-tests.
    """
    img = img.astype(np.float64)
    source = bg_src.astype(np.float64) if bg_src is not None else img
    bg = gaussian_filter(source, sigma=sigma)
    out = img / np.maximum(bg, 1e-6)
    out *= bg.mean() / out.mean()
    return out
