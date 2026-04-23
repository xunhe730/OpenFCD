"""Temporal aggregation of per-frame eta fields.

Three useful reductions for video-based denoising:

* `time_mean`   — pixel-wise mean. Cancels random noise; for a periodic motion
                  sampled densely it tends toward zero (Stokes drift remains).
* `time_median` — pixel-wise median. Same as mean for noise but robust against
                  isolated phase-wrap blowups in individual frames.
* `time_rms`    — pixel-wise standard deviation. Excellent visualization of
                  *where* the surface is active without caring about phase
                  (this is the "wave amplitude map").

For phase-locked averaging when the actuation period is known in frames,
use `phase_lock_stack`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def _stack(eta_files: Iterable[Path | str]) -> np.ndarray:
    arrays = []
    shape = None
    for f in eta_files:
        a = np.load(f)
        if shape is None:
            shape = a.shape
        elif a.shape != shape:
            raise ValueError(f"shape mismatch in {f}: {a.shape} vs {shape}")
        arrays.append(a)
    if not arrays:
        raise ValueError("empty stack")
    return np.stack(arrays, axis=0)


def time_mean(eta_files: Iterable[Path | str]) -> np.ndarray:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(_stack(eta_files), axis=0)


def time_median(eta_files: Iterable[Path | str]) -> np.ndarray:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(_stack(eta_files), axis=0)


def time_rms(eta_files: Iterable[Path | str]) -> np.ndarray:
    """RMS amplitude: sqrt(mean((η - <η>)²))."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stack = _stack(eta_files)
        mean = np.nanmean(stack, axis=0, keepdims=True)
        return np.sqrt(np.nanmean((stack - mean) ** 2, axis=0))


def phase_lock_stack(eta_files: Sequence[Path | str],
                     period_frames: int) -> np.ndarray:
    """Average frames separated by `period_frames` to reconstruct one cycle.

    Returns an array of shape (period_frames, H, W) holding the cycle-mean
    surface for each phase bin.
    """
    files = list(eta_files)
    bins = [[] for _ in range(period_frames)]
    for i, f in enumerate(files):
        bins[i % period_frames].append(f)
    out = []
    for bf in bins:
        if bf:
            out.append(time_mean(bf))
        else:
            out.append(np.full_like(np.load(files[0]), np.nan))
    return np.stack(out, axis=0)
