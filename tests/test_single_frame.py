"""TDD tests for _SingleFrameWorker — single-frame compute preview."""
from __future__ import annotations

import tempfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pytest

from PyQt6.QtCore import QCoreApplication

from PyQt6.QtCore import QCoreApplication


from openfcd.io.project import ProjectModel
from openfcd.io.annotation import AnnotationSchema
from openfcd.gui.mainwindow import _SingleFrameWorker


def _make_mock_project(
    preset: str = "pattern_below_window",
    frames_dir: str = "",
    reference_source: str = "ref.png",
) -> tuple[ProjectModel, Path, Path]:
    """Create a temporary project with synthetic images.

    Returns (project, project_dir, frame_path).
    """
    tmp = Path(tempfile.mkdtemp())
    project_dir = tmp / "test.ofcd"
    project_dir.mkdir(parents=True, exist_ok=True)

    # Create frames_dir with reference and one deformed frame
    img_dir = tmp / "frames"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic 128x128 grayscale checkerboard-like images
    size = 128
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    ref_arr = (np.sin(2 * np.pi * x / 8) * np.sin(2 * np.pi * y / 8) * 50 + 128).astype(np.uint8)
    # Deformed: slightly shifted pattern
    def_arr = (np.sin(2 * np.pi * (x + 0.5) / 8) * np.sin(2 * np.pi * y / 8) * 50 + 128).astype(np.uint8)

    ref_path = img_dir / "ref.png"
    frame_path = img_dir / "frame_000.png"
    _write_gray(ref_path, ref_arr)
    _write_gray(frame_path, def_arr)

    proj = ProjectModel(
        format_version=0,
        name="test",
        created=datetime.now(timezone.utc).isoformat(),
        geometry={
            "pattern_period_mm": 1.2,
            "optical_stack": {
                "preset": preset,
                "layers": [
                    {"thickness_mm": 3.0, "medium": "glass", "n": 1.5},
                    {"thickness_mm": 12.0, "medium": "water", "n": 1.333},
                ] if preset != "custom" else [],
            },
        },
        data={
            "frames_dir": str(img_dir),
            "pattern": "frame_*.png",
        },
        reference={
            "mode": "use_existing",
            "source": reference_source,
        },
    )

    return proj, project_dir, frame_path


def _write_gray(path: Path, arr: np.ndarray) -> None:
    from skimage.io import imsave
    imsave(str(path), arr)


def _make_annotation() -> AnnotationSchema:
    """Create a minimal annotation schema (no ROI, no polygons)."""
    return AnnotationSchema()


def _capture_frame_done(results: list):
    def _receiver(overlay, eta_mm) -> None:
        results.append((overlay, eta_mm))

    return _receiver


class TestSingleFrameWorker:
    """Tests for _SingleFrameWorker background compute thread."""

    def test_single_frame_worker_ignores_hidden_global_polygon(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Global polygons are not applied to frames with no visible per-frame mask."""
        from openfcd.pipeline.frame import FrameResult
        app = QCoreApplication.instance() or QCoreApplication([])

        proj, project_dir, frame_path = _make_mock_project()
        ann = AnnotationSchema(
            polygons=[{
                "vertices": [[40, 40], [40, 80], [90, 80], [90, 40]],
                "label": "body",
            }]
        )
        seen: list = []

        def _fake_compute_frame(inputs, *, progress_cb=None, cancel=None):
            seen.append(inputs.robot_poly)
            eta = np.zeros(inputs.ref_shape, dtype=np.float64)
            return FrameResult(eta_mm=eta, qc_datasets=None, diagnostics={})

        monkeypatch.setattr("openfcd.pipeline.frame.compute_frame", _fake_compute_frame)

        worker = _SingleFrameWorker(
            project=proj,
            project_dir=project_dir,
            frame_path=frame_path,
            annotation=ann,
        )
        errors: list = []
        worker.frame_failed.connect(errors.append)
        worker.run()

        assert errors == []
        assert seen == [None]

    def test_single_frame_worker_emits_frame_done(self) -> None:
        """Valid images and geometry → frame_done signal fires with eta_mm array."""
        app = QCoreApplication.instance() or QCoreApplication([])

        proj, project_dir, frame_path = _make_mock_project()
        ann = _make_annotation()

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

        # Call run() synchronously (QThread.run is just a method call)
        worker.run()

        assert len(results) == 1, f"Expected frame_done, got errors: {errors}"
        eta_overlay, eta_mm = results[0]
        assert isinstance(eta_overlay, np.ndarray)
        assert eta_overlay.shape == (128, 128)
        assert isinstance(eta_mm, np.ndarray)
        assert eta_mm.ndim == 2
        assert eta_mm.shape == (128, 128)

    def test_single_frame_worker_passes_cancel_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Single-frame Cancel should interrupt the same compute path as Run."""
        app = QCoreApplication.instance() or QCoreApplication([])

        proj, project_dir, frame_path = _make_mock_project()
        ann = _make_annotation()
        seen_cancel = []

        def _fake_compute_frame(inputs, *, progress_cb=None, cancel=None):
            seen_cancel.append(cancel)
            if cancel is not None:
                cancel.cancel()
                cancel.check()

        monkeypatch.setattr("openfcd.pipeline.frame.compute_frame", _fake_compute_frame)

        worker = _SingleFrameWorker(
            project=proj,
            project_dir=project_dir,
            frame_path=frame_path,
            annotation=ann,
        )
        errors: list = []
        worker.frame_failed.connect(errors.append)

        worker.run()

        assert seen_cancel
        assert errors == ["Compute cancelled"]

    def test_single_frame_worker_emits_frame_failed_on_empty_layers(self) -> None:
        """preset=custom with layers=[] → frame_failed signal fires."""
        app = QCoreApplication.instance() or QCoreApplication([])

        proj, project_dir, frame_path = _make_mock_project(preset="custom")
        ann = _make_annotation()

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

        assert len(errors) == 1, f"Expected frame_failed, got results: {results}"
        assert "layers" in errors[0].lower() or "optical" in errors[0].lower()

    def test_single_frame_worker_no_hdf5_writes(self) -> None:
        """After single-frame compute, no HDF5 files should exist in runs/."""
        app = QCoreApplication.instance() or QCoreApplication([])

        proj, project_dir, frame_path = _make_mock_project()
        ann = _make_annotation()

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

        # Verify no runs/ directory or results.h5 created
        runs_dir = project_dir / "runs"
        assert not runs_dir.exists(), "runs/ directory should not be created by single-frame worker"
        assert len(results) == 1, "Worker should have completed successfully"

    def test_single_frame_worker_applies_roi_annotation(self) -> None:
        """ROI annotation crops compute; result is embedded back in ref_shape for both signals."""
        app = QCoreApplication.instance() or QCoreApplication([])

        proj, project_dir, frame_path = _make_mock_project()
        ann = AnnotationSchema(roi={"x": 12, "y": 20, "width": 40, "height": 30})

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

        assert len(errors) == 0, f"Unexpected worker failure: {errors}"
        assert len(results) == 1
        eta_overlay, eta_mm = results[0]
        # Both signals now carry the full ref_shape-embedded array (new API invariant).
        assert eta_overlay.shape == (128, 128)
        assert eta_mm.shape == (128, 128)
        # ROI region should have some finite values; outside should be NaN.
        assert np.isfinite(eta_overlay[20:50, 12:52]).any()
        assert np.isnan(eta_overlay[:10, :10]).all()

    def test_single_frame_worker_ref_vs_ref_with_mask_stays_near_zero(self) -> None:
        """Self-test with a manual mask should not invent ring artefacts."""
        app = QCoreApplication.instance() or QCoreApplication([])

        proj, project_dir, _frame_path = _make_mock_project()
        ref_path = Path(proj.data.frames_dir) / "ref.png"
        ann = AnnotationSchema(
            frame_polygons={
                "ref.png": [{
                    "vertices": [[40, 40], [40, 80], [90, 80], [90, 40]],
                    "label": "body",
                }]
            }
        )

        worker = _SingleFrameWorker(
            project=proj,
            project_dir=project_dir,
            frame_path=ref_path,
            annotation=ann,
        )

        results: list = []
        errors: list = []
        worker.frame_done.connect(_capture_frame_done(results))
        worker.frame_failed.connect(errors.append)

        worker.run()

        assert len(errors) == 0, f"Unexpected worker failure: {errors}"
        assert len(results) == 1
        eta_overlay, eta_mm = results[0]
        assert eta_overlay.shape == eta_mm.shape
        finite = eta_mm[np.isfinite(eta_mm)]
        assert finite.size > 0
        assert np.max(np.abs(finite)) < 1e-3

    def test_single_frame_worker_auto_masks_thin_wire_occluder(self) -> None:
        """A thin dark filament should be auto-masked instead of turning into a sharp eta jump."""
        app = QCoreApplication.instance() or QCoreApplication([])

        proj, project_dir, frame_path = _make_mock_project()
        ref_path = Path(proj.data.frames_dir) / "ref.png"

        from skimage.io import imread, imsave
        # Paint the same wire on both reference and deformed frames — physical
        # filaments are static occluders present in both. The detector now
        # runs on the reference image (frame.py), so a wire only on the
        # deformed frame would no longer be auto-masked.
        for path in (ref_path, frame_path):
            arr = imread(str(path)).astype(np.uint8)
            arr[20:110, 60:63] = 0
            imsave(str(path), arr)

        ann = _make_annotation()
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

        assert len(errors) == 0, f"Unexpected worker failure: {errors}"
        assert len(results) == 1
        eta_overlay, eta_mm = results[0]
        wire_band = eta_mm[20:110, 60:63]
        assert np.isnan(wire_band).mean() > 0.5
