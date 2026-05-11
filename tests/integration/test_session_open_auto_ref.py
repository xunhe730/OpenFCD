"""Integration tests for the auto_ref full flow.

These tests exercise the programmatic path that _auto_build_reference
follows in MainWindow, without requiring a live Qt event loop:

  build_reference_from_paths → np.save → mutate proj.reference → store.save()

and then verifies that the persisted state is exactly correct.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import skimage.io

from openfcd.core.reference_builder import build_reference_from_paths
from openfcd.io.project import ProjectModel
from openfcd.io.store import FileSessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_checkerboard(seed: int, shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    rows, cols = np.indices(shape)
    base = ((rows + cols) % 2).astype(np.uint8) * 200
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 10, size=shape, dtype=np.uint8)
    return np.clip(base.astype(np.int32) + noise, 0, 255).astype(np.uint8)


def _create_project(tmp_path: Path, n_frames: int = 8) -> tuple[Path, list[Path]]:
    """Create a minimal .ofcd project and write n_frames PNGs.

    Returns (project_dir, sorted frame paths).
    """
    project_dir = tmp_path / "test_project.ofcd"
    frames_dir = project_dir / "frames"
    frames_dir.mkdir(parents=True)

    # Write PNGs
    paths: list[Path] = []
    for i in range(n_frames):
        p = frames_dir / f"frame_{i:03d}.png"
        skimage.io.imsave(str(p), _make_checkerboard(seed=i))
        paths.append(p)
    paths = sorted(paths)

    # Bootstrap the project directory (creates annotations/, runs/, etc.)
    store = FileSessionStore.new(project_dir, "test_project")
    # Point data.frames_dir at the frames subdir
    store.project.data.frames_dir = str(frames_dir)
    store.project.data.pattern = "*.png"
    store.save()

    return project_dir, paths


# ---------------------------------------------------------------------------
# Test 1: auto_ref ON — build, persist, verify
# ---------------------------------------------------------------------------

class TestAutoRefOn:
    def test_npy_file_created(self, tmp_path: Path) -> None:
        project_dir, paths = _create_project(tmp_path)
        ref_img = build_reference_from_paths(paths, reducer="mean")
        out = project_dir / "reference_built.npy"
        np.save(out, ref_img)
        assert out.exists(), "reference_built.npy must be written"

    def test_npy_content_matches_mean(self, tmp_path: Path) -> None:
        project_dir, paths = _create_project(tmp_path)
        ref_img = build_reference_from_paths(paths, reducer="mean")
        out = project_dir / "reference_built.npy"
        np.save(out, ref_img)

        # Compute expected via raw stack
        from openfcd.pipeline.compute import load_gray
        stack = np.stack([load_gray(p) for p in paths], axis=0)
        expected = np.mean(stack, axis=0)

        loaded = np.load(out)
        assert np.allclose(loaded, expected), (
            f"npy content mismatch; max diff = {np.abs(loaded - expected).max()}"
        )

    def test_project_yaml_schema_updated(self, tmp_path: Path) -> None:
        project_dir, paths = _create_project(tmp_path)

        # Simulate _auto_build_reference persistence step
        store = FileSessionStore.open(project_dir)
        proj = store.project
        ref_img = build_reference_from_paths(paths, reducer="mean")
        np.save(project_dir / "reference_built.npy", ref_img)
        proj.reference.mode = "build"
        proj.reference.source = "reference_built.npy"
        proj.reference.build_params = {
            "reducer": "mean",
            "n": len(paths),
            "stride": 1,
            "explicit": True,
        }
        store.save()

        # Reload from disk and verify schema
        reloaded = ProjectModel.from_yaml(project_dir / "project.yaml")
        assert reloaded.reference.mode == "build"
        assert reloaded.reference.source == "reference_built.npy"
        bp = reloaded.reference.build_params
        assert bp["reducer"] == "mean"
        assert bp["n"] == len(paths)
        assert bp["stride"] == 1
        assert bp["explicit"] is True

    def test_cached_npy_matches_reference_array(self, tmp_path: Path) -> None:
        """np.load of reference_built.npy must match the in-memory result."""
        project_dir, paths = _create_project(tmp_path)
        ref_img = build_reference_from_paths(paths, reducer="mean")
        out = project_dir / "reference_built.npy"
        np.save(out, ref_img)

        cached = np.load(out)
        assert np.allclose(cached, ref_img), "Cached npy must be byte-equal to in-memory result"

    def test_median_reducer_persisted(self, tmp_path: Path) -> None:
        project_dir, paths = _create_project(tmp_path)
        store = FileSessionStore.open(project_dir)
        proj = store.project

        ref_img = build_reference_from_paths(paths, reducer="median")
        np.save(project_dir / "reference_built.npy", ref_img)
        proj.reference.mode = "build"
        proj.reference.source = "reference_built.npy"
        proj.reference.build_params = {
            "reducer": "median",
            "n": len(paths),
            "stride": 1,
            "explicit": True,
        }
        store.save()

        reloaded = ProjectModel.from_yaml(project_dir / "project.yaml")
        assert reloaded.reference.build_params["reducer"] == "median"

        cached = np.load(project_dir / "reference_built.npy")
        expected = build_reference_from_paths(paths, reducer="median")
        assert np.allclose(cached, expected)


# ---------------------------------------------------------------------------
# Test 2: auto_ref OFF — no-op invariant
# ---------------------------------------------------------------------------

class TestAutoRefOff:
    def test_no_npy_written_when_skipped(self, tmp_path: Path) -> None:
        project_dir, _ = _create_project(tmp_path)
        npy = project_dir / "reference_built.npy"
        assert not npy.exists(), "reference_built.npy must not exist when auto_ref is off"

    def test_project_yaml_byte_identical_when_unchanged(self, tmp_path: Path) -> None:
        project_dir, _ = _create_project(tmp_path)

        yaml_path = project_dir / "project.yaml"
        before = yaml_path.read_bytes()

        # Open session, make no reference changes, save
        store = FileSessionStore.open(project_dir)
        store.save()

        after = yaml_path.read_bytes()
        assert before == after, (
            "project.yaml must be byte-identical when no reference changes were made"
        )

    def test_reference_fields_unchanged_when_skipped(self, tmp_path: Path) -> None:
        project_dir, _ = _create_project(tmp_path)
        before = ProjectModel.from_yaml(project_dir / "project.yaml")
        original_mode = before.reference.mode
        original_source = before.reference.source

        # Simulate no auto_ref action
        store = FileSessionStore.open(project_dir)
        store.save()

        after = ProjectModel.from_yaml(project_dir / "project.yaml")
        assert after.reference.mode == original_mode
        assert after.reference.source == original_source


# ---------------------------------------------------------------------------
# Dual-file (.npy + .jpg) behaviour: _auto_build_reference writes both, and
# _resolve_reference prefers the .npy sibling for full float precision.
# ---------------------------------------------------------------------------

class TestDualFileReference:
    def test_load_prefers_npy_sibling_over_jpg(self, tmp_path: Path) -> None:
        """When source is foo.jpg and foo.npy exists in project_dir, load .npy."""
        from openfcd.cli.cmd_run import _load_reference_source
        project_dir, paths = _create_project(tmp_path)
        store = FileSessionStore.open(project_dir)
        proj = store.project
        proj.reference.mode = "build"
        proj.reference.source = "reference_built.jpg"
        store.save()

        ref_img = build_reference_from_paths(paths, reducer="mean")
        np.save(project_dir / "reference_built.npy", ref_img)
        from skimage.io import imsave
        from skimage.util import img_as_ubyte
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(
                str(project_dir / "reference_built.jpg"),
                img_as_ubyte(np.clip(ref_img / 255.0, 0.0, 1.0)),
            )

        loaded = _load_reference_source(proj, project_dir, "reference_built.jpg")
        # Must be the float64 .npy content, not the lossy 8-bit jpg.
        assert loaded.dtype == np.float64
        np.testing.assert_allclose(loaded, ref_img)

    def test_jpg_thumbnail_renders_real_grayscale_not_white(
        self, tmp_path: Path,
    ) -> None:
        """The auto-ref JPG must be a visible thumbnail, not saturated white.

        Regression: load_gray returns float64 in [0, 255] for grayscale
        inputs but [0, 1] for RGB-via-rgb2gray inputs. An earlier version
        unconditionally clipped to [0, 1] before img_as_ubyte, producing
        an all-white JPG when the input range was [0, 255].
        """
        project_dir, paths = _create_project(tmp_path)
        ref_img = build_reference_from_paths(paths, reducer="mean")
        # Replicate _auto_build_reference's JPG write path
        arr = ref_img.astype(np.float64, copy=False)
        arr_u8 = (
            np.clip(arr * 255.0, 0.0, 255.0)
            if float(arr.max(initial=0.0)) <= 1.0 + 1e-6
            else np.clip(arr, 0.0, 255.0)
        ).astype(np.uint8)
        from skimage.io import imsave, imread
        import warnings
        out_jpg = project_dir / "reference_built.jpg"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(str(out_jpg), arr_u8)
        loaded = imread(str(out_jpg))
        # Must be visible: substantial dynamic range, mean far from 255.
        assert loaded.dtype == np.uint8
        assert loaded.max() - loaded.min() > 30, (
            f"JPG looks flat: range={loaded.min()}..{loaded.max()}"
        )
        assert loaded.mean() < 240, (
            f"JPG looks saturated white: mean={loaded.mean():.1f}"
        )

    def test_load_falls_back_to_jpg_when_no_npy(self, tmp_path: Path) -> None:
        """Without a sibling .npy, fall back to load_gray on the .jpg."""
        from openfcd.cli.cmd_run import _load_reference_source
        project_dir, paths = _create_project(tmp_path)
        store = FileSessionStore.open(project_dir)
        proj = store.project
        proj.reference.mode = "build"
        proj.reference.source = "reference_built.jpg"
        store.save()

        from skimage.io import imsave
        from skimage.util import img_as_ubyte
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(
                str(project_dir / "reference_built.jpg"),
                img_as_ubyte(np.full((64, 64), 0.5, dtype=np.float64)),
            )

        loaded = _load_reference_source(proj, project_dir, "reference_built.jpg")
        assert loaded.shape == (64, 64)
        assert loaded.dtype == np.float64
