"""Integration tests for the full GUI compute chain: new project -> compute -> run -> results."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import numpy as np

from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QApplication

from openfcd.gui.controllers.session_controller import SessionController
from openfcd.gui.mainwindow import MainWindow, _SingleFrameWorker, _repair_optical_config, _parse_workers_value
from openfcd.cli.cmd_run import _resolve_reference
from openfcd.io.store import FileSessionStore
from openfcd.io.store import FileSessionStore
from openfcd.io.project import ProjectModel
from openfcd.io.annotation import AnnotationSchema


def _capture_frame_done(results: list):
    def _receiver(overlay, eta_mm) -> None:
        results.append((overlay, eta_mm))

    return _receiver


def _write_gray(path: Path, arr: np.ndarray) -> None:
    """Write a 2D numpy array as a grayscale image."""
    from skimage.io import imsave
    imsave(str(path), arr)


def _make_annotation() -> AnnotationSchema:
    """Create a minimal annotation schema."""
    return AnnotationSchema()


class TestNewProjectLayersPopulated:
    """Test 1: New project with optical preset has layers populated."""

    def test_new_project_layers_populated(self) -> None:
        """Create a new project via SessionController.new_project() with
        optical_preset='pattern_below_window' and verify layers are populated."""
        with tempfile.TemporaryDirectory() as tmp:
            controller = SessionController()
            result = controller.new_project(
                name="integration_test",
                location=tmp,
                optical_preset="pattern_below_window",
                pattern_period_mm=1.5,
            )

            # Verify project directory exists
            assert result.exists()
            assert result.name == "integration_test.ofcd"

            # Verify project.yaml has layers populated (2 layers)
            proj = controller.project
            assert proj is not None
            layers = proj.geometry.optical_stack.layers
            assert len(layers) == 2
            assert layers[0].medium == "glass"
            assert layers[0].thickness_mm == 3.0
            assert layers[1].medium == "water"
            assert layers[1].thickness_mm == 12.0

            # Verify pattern_period_mm is set correctly
            assert proj.geometry.pattern_period_mm == 1.5
            assert proj.geometry.optical_stack.preset == "pattern_below_window"

            # Verify project.yaml was written to disk
            yaml_path = result / "project.yaml"
            assert yaml_path.exists()

            # Re-read from disk to confirm persistence
            reloaded = ProjectModel.from_yaml(yaml_path)
            assert len(reloaded.geometry.optical_stack.layers) == 2
            assert reloaded.geometry.pattern_period_mm == 1.5


class TestOpticalRepairOnOpen:
    """Test 2: Optical repair on open for projects with empty layers."""

    def test_optical_repair_on_open(self) -> None:
        """Create a project with preset='pattern_below_window' but empty layers,
        open it via SessionController, call _repair_optical_config, and verify
        layers are populated and session is dirty."""
        with tempfile.TemporaryDirectory() as tmp:
            # Step 1: Create project with preset but empty layers
            project_dir = Path(tmp) / "repair_test.ofcd"
            project_dir.mkdir(parents=True, exist_ok=True)

            store = FileSessionStore.new(project_dir, "repair_test")
            proj = store.project
            proj.geometry.optical_stack.preset = "pattern_below_window"  # type: ignore[assignment]
            proj.geometry.optical_stack.layers = []  # Force empty layers
            proj.geometry.pattern_period_mm = 1.2
            store.save()

            # Step 2: Open the project via SessionController
            session = SessionController()
            session.open_project(project_dir)

            # Verify initial state: empty layers, not dirty
            assert session.project is not None
            assert len(session.project.geometry.optical_stack.layers) == 0
            assert session.is_dirty is False

            # Step 3: Call _repair_optical_config (simulating mainwindow repair)
            repaired = _repair_optical_config(session.project, session)

            # Step 4: Verify repair was performed
            assert repaired is True
            layers = session.project.geometry.optical_stack.layers
            assert len(layers) == 2
            assert layers[0].medium == "glass"
            assert layers[1].medium == "water"
            assert session.is_dirty is True


class TestSingleFrameComputeChain:
    """Test 3: Single-frame compute chain with synthetic images."""

    def test_single_frame_compute_chain(self) -> None:
        """Create a project with valid geometry, synthetic images, run
        _SingleFrameWorker, and verify frame_done emits numpy array."""
        app = QCoreApplication.instance() or QCoreApplication([])

        tmp = Path(tempfile.mkdtemp())
        project_dir = tmp / "compute_test.ofcd"
        project_dir.mkdir(parents=True, exist_ok=True)

        # Create synthetic images
        img_dir = tmp / "frames"
        img_dir.mkdir(parents=True, exist_ok=True)

        size = 200
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        ref_arr = (np.sin(2 * np.pi * x / 8) * np.sin(2 * np.pi * y / 8) * 50 + 128).astype(np.uint8)
        def_arr = (np.sin(2 * np.pi * (x + 0.5) / 8) * np.sin(2 * np.pi * y / 8) * 50 + 128).astype(np.uint8)

        ref_path = img_dir / "ref.png"
        frame_path = img_dir / "frame_000.png"
        _write_gray(ref_path, ref_arr)
        _write_gray(frame_path, def_arr)

        # Create project with valid geometry (pattern_below_window, populated layers)
        proj = ProjectModel(
            format_version=0,
            name="compute_test",
            created="2026-04-23T00:00:00+00:00",
            geometry={
                "pattern_period_mm": 1.2,
                "optical_stack": {
                    "preset": "pattern_below_window",
                    "layers": [
                        {"thickness_mm": 3.0, "medium": "glass", "n": 1.5},
                        {"thickness_mm": 12.0, "medium": "water", "n": 1.333},
                    ],
                },
            },
            data={
                "frames_dir": str(img_dir),
                "pattern": "frame_*.png",
            },
            reference={
                "mode": "use_existing",
                "source": "ref.png",
            },
        )
        proj.to_yaml(project_dir / "project.yaml")

        ann = _make_annotation()

        # Create and run _SingleFrameWorker
        worker = _SingleFrameWorker(
            project=proj,
            project_dir=project_dir,
            frame_path=frame_path,
            annotation=ann,
        )

        results: list = []
        errors: list = []
        worker.frame_done.connect(_capture_frame_done(results))
        worker.frame_failed.connect(errors.append)

        worker.run()

        # Verify frame_done signal emits a numpy array
        assert len(results) == 1, f"Expected frame_done, got errors: {errors}"
        eta_overlay, eta_mm = results[0]
        assert isinstance(eta_overlay, np.ndarray)
        assert eta_overlay.shape == (size, size)
        assert isinstance(eta_mm, np.ndarray)
        assert eta_mm.ndim == 2
        assert eta_mm.shape == (size, size)

        # Verify no HDF5 files in runs/ directory
        runs_dir = project_dir / "runs"
        assert not runs_dir.exists(), "runs/ directory should not be created by single-frame worker"


class TestWorkersParsingIntegration:
    """Test 4: _parse_workers_value function with various inputs."""

    def test_parse_workers_auto(self) -> None:
        """'auto' -> -1"""
        assert _parse_workers_value("auto") == -1

    def test_parse_workers_empty_string(self) -> None:
        """'' -> -1 (treated as auto)"""
        assert _parse_workers_value("") == -1

    def test_parse_workers_numeric(self) -> None:
        """'4' -> 4"""
        assert _parse_workers_value("4") == 4

    def test_parse_workers_invalid(self) -> None:
        """'abc' -> None"""
        assert _parse_workers_value("abc") is None

    def test_parse_workers_whitespace(self) -> None:
        """'  auto  ' -> -1 (whitespace trimmed)"""
        assert _parse_workers_value("  auto  ") == -1

    def test_parse_workers_negative_number(self) -> None:
        """'-1' -> -1 (valid integer)"""
        assert _parse_workers_value("-1") == -1

    def test_parse_workers_large_number(self) -> None:
        """'64' -> 64"""
        assert _parse_workers_value("64") == 64


class TestReferenceSelectionIntegration:
    """Test 5: GUI reference selection updates reference mode and source."""

    def test_set_reference_switches_project_to_use_existing(self) -> None:
        app = QApplication.instance() or QApplication([])

        tmp = Path(tempfile.mkdtemp())
        controller = SessionController()
        project_dir = controller.new_project(
            name="ref_mode_test",
            location=tmp,
            optical_preset="pattern_below_window",
            pattern_period_mm=1.2,
        )

        img_dir = tmp / "frames"
        img_dir.mkdir(parents=True, exist_ok=True)
        arr = np.zeros((16, 16), dtype=np.uint8)
        ref_path = img_dir / "Img0001.png"
        _write_gray(ref_path, arr)

        proj = controller.project
        assert proj is not None
        proj.data.frames_dir = str(img_dir)
        proj.data.pattern = "Img*.png"
        proj.reference.mode = "build"
        proj.reference.source = ""

        window = MainWindow()
        window._session = controller
        window._frames = [ref_path]

        window._on_set_reference(0)

        assert proj.reference.mode == "use_existing"
        assert proj.reference.source == "Img0001.png"

    def test_explicit_reference_source_overrides_mode(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        img_dir = tmp / "frames"
        img_dir.mkdir(parents=True, exist_ok=True)

        ref_a = np.full((8, 8), 25, dtype=np.uint8)
        ref_b = np.full((8, 8), 200, dtype=np.uint8)
        path_a = img_dir / "Img0001.png"
        path_b = img_dir / "Img0002.png"
        _write_gray(path_a, ref_a)
        _write_gray(path_b, ref_b)

        proj = ProjectModel(
            format_version=0,
            name="ref_override_test",
            created="2026-04-23T00:00:00+00:00",
            geometry={
                "pattern_period_mm": 1.2,
                "optical_stack": {
                    "preset": "pattern_below_window",
                    "layers": [
                        {"thickness_mm": 3.0, "medium": "glass", "n": 1.5},
                        {"thickness_mm": 12.0, "medium": "water", "n": 1.333},
                    ],
                },
            },
            data={"frames_dir": str(img_dir), "pattern": "Img*.png"},
            reference={"mode": "build", "source": "Img0002.png"},
        )

        loaded = _resolve_reference(proj, tmp)
        assert loaded.shape == (8, 8)
        assert float(loaded.mean()) > 100.0
