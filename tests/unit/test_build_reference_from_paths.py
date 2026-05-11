"""Unit tests for build_reference_from_paths."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import skimage.io

from openfcd.core.reference_builder import build_reference, build_reference_from_paths


def _make_checkerboard(seed: int, shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Return a uint8 checkerboard image with a slight per-seed variation."""
    rows, cols = np.indices(shape)
    base = ((rows + cols) % 2).astype(np.uint8) * 200
    # Add a small constant so each frame is slightly different
    return np.clip(base + seed * 5, 0, 255).astype(np.uint8)


@pytest.fixture()
def checkerboard_dir(tmp_path: Path) -> tuple[Path, list[Path]]:
    """Write 5 checkerboard PNGs to a temp dir; return (dir, sorted_paths)."""
    paths: list[Path] = []
    for i in range(5):
        img = _make_checkerboard(seed=i)
        p = tmp_path / f"frame_{i:03d}.png"
        skimage.io.imsave(str(p), img)
        paths.append(p)
    return tmp_path, sorted(paths)


class TestBuildReferenceFromPaths:
    def test_mean_matches_build_reference(
        self, checkerboard_dir: tuple[Path, list[Path]]
    ) -> None:
        """Explicit-list mean must equal directory-glob mean."""
        frames_dir, paths = checkerboard_dir
        result = build_reference_from_paths(paths, reducer="mean")
        expected = build_reference(
            frames_dir, "*.png", n=5, stride=1, reducer="mean"
        )
        assert np.allclose(result, expected), (
            f"mean mismatch: max diff = {np.abs(result - expected).max()}"
        )

    def test_median_matches_build_reference(
        self, checkerboard_dir: tuple[Path, list[Path]]
    ) -> None:
        """Explicit-list median must equal directory-glob median."""
        frames_dir, paths = checkerboard_dir
        result = build_reference_from_paths(paths, reducer="median")
        expected = build_reference(
            frames_dir, "*.png", n=5, stride=1, reducer="median"
        )
        assert np.allclose(result, expected)

    def test_min_reducer(
        self, checkerboard_dir: tuple[Path, list[Path]]
    ) -> None:
        """min reducer should return pixel-wise minimum."""
        _, paths = checkerboard_dir
        result = build_reference_from_paths(paths, reducer="min")
        expected = build_reference_from_paths(paths, reducer="mean")
        # min <= mean for all pixels
        assert (result <= expected + 1e-9).all()

    def test_output_dtype_float64(
        self, checkerboard_dir: tuple[Path, list[Path]]
    ) -> None:
        """Output must be float64 to match build_reference."""
        _, paths = checkerboard_dir
        result = build_reference_from_paths(paths)
        assert result.dtype == np.float64

    def test_output_shape(
        self, checkerboard_dir: tuple[Path, list[Path]]
    ) -> None:
        """Output shape must match single-frame shape."""
        _, paths = checkerboard_dir
        result = build_reference_from_paths(paths)
        assert result.shape == (64, 64)

    def test_empty_list_raises(self) -> None:
        """Empty frame list must raise ValueError."""
        with pytest.raises(ValueError, match="empty frame list"):
            build_reference_from_paths([])

    def test_unknown_reducer_raises(
        self, checkerboard_dir: tuple[Path, list[Path]]
    ) -> None:
        """Unknown reducer must raise ValueError."""
        _, paths = checkerboard_dir
        with pytest.raises(ValueError, match="Unknown reducer"):
            build_reference_from_paths(paths, reducer="bogus")  # type: ignore[arg-type]

    def test_single_frame(self, tmp_path: Path) -> None:
        """Single frame: result equals the frame itself (as float64)."""
        img = _make_checkerboard(seed=0)
        p = tmp_path / "only.png"
        skimage.io.imsave(str(p), img)
        result = build_reference_from_paths([p], reducer="mean")
        # load_gray returns float64 without normalising; compare as-is
        expected = img.astype(np.float64)
        assert np.allclose(result, expected, atol=1e-6)
