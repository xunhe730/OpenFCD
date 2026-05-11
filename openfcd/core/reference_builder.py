"""Reference-image builder.

Constructs a reference (background) image from a subset of frames in an
image sequence.  Supports three reducers:
  - median  — robust to transient occlusions (default)
  - mean    — less noisy when no occlusion present
  - min     — keeps darkest pixels, useful for back-lit setups

Usage::

    from openfcd.core.reference_builder import build_reference
    ref = build_reference(Path("DATA/20Hz/s2"), "Img*.jpg", n=100, stride=11)
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np


def build_reference(
    frames_dir: Path | str,
    pattern: str = "Img*.jpg",
    n: int = 100,
    stride: int = 1,
    reducer: Literal["median", "mean", "min"] = "median",
) -> np.ndarray:
    """Build a reference image from *n* frames sampled every *stride*.

    Parameters
    ----------
    frames_dir : Path
        Directory containing frame images.
    pattern : str
        Glob pattern for frame files.
    n : int
        Maximum number of frames to use.
    stride : int
        Step size between sampled frames (1 = consecutive).
    reducer : str
        Aggregation method: ``"median"`` (default), ``"mean"``, or ``"min"``.

    Returns
    -------
    np.ndarray
        Float64 grayscale reference image (0–1 range if input was 8-bit).
    """
    from openfcd.io.image import scan_frames
    from openfcd.pipeline.compute import load_gray

    all_frames = scan_frames(frames_dir, pattern)
    if not all_frames:
        raise FileNotFoundError(
            f"No frames matching '{pattern}' in {frames_dir}"
        )

    # Sample frames with stride
    sampled = all_frames[::stride][:n]
    if not sampled:
        raise ValueError(f"Sampling produced 0 frames (stride={stride}, n={n})")

    # Load and stack
    first = load_gray(sampled[0])
    stack = np.empty((len(sampled), *first.shape), dtype=np.float64)
    stack[0] = first
    for i, p in enumerate(sampled[1:], 1):
        stack[i] = load_gray(p)

    # Reduce
    if reducer == "median":
        return np.median(stack, axis=0)
    elif reducer == "mean":
        return np.mean(stack, axis=0)
    elif reducer == "min":
        return np.min(stack, axis=0)
    else:
        raise ValueError(f"Unknown reducer: {reducer!r}")


def build_reference_from_paths(
    frames: list[Path],
    reducer: Literal["median", "mean", "min"] = "mean",
) -> np.ndarray:
    """Build a reference image from an explicit list of frame paths.

    Same as :func:`build_reference` but accepts a pre-filtered list of paths
    instead of re-globbing a directory.  The picker already filtered the
    user's selection; re-globbing the directory here would silently include
    disabled or hidden frames.

    Parameters
    ----------
    frames : list[Path]
        Explicit list of image paths to reduce.
    reducer : str
        Aggregation method: ``"mean"`` (default), ``"median"``, or ``"min"``.

    Returns
    -------
    np.ndarray
        Float64 grayscale reference image.

    Raises
    ------
    ValueError
        If *frames* is empty.
    """
    if not frames:
        raise ValueError("build_reference_from_paths: empty frame list")
    from openfcd.pipeline.compute import load_gray

    first = load_gray(frames[0])
    stack = np.empty((len(frames), *first.shape), dtype=np.float64)
    stack[0] = first
    for i, p in enumerate(frames[1:], 1):
        stack[i] = load_gray(p)

    reducers: dict[str, object] = {
        "median": np.median,
        "mean": np.mean,
        "min": np.min,
    }
    fn = reducers.get(reducer)
    if fn is None:
        raise ValueError(f"Unknown reducer: {reducer!r}")
    return fn(stack, axis=0)  # type: ignore[operator]
