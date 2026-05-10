"""Regression: ComputeStage produces consistent eta_mm, workers resolver returns >=1."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from skimage.io import imsave

from openfcd.cli.cmd_run import (
    ComputeStage,
    PreprocessStage,
    _resolve_default_workers,
)
from openfcd.io.project import ProjectModel
from openfcd.io.result import HDF5ResultStore
from openfcd.pipeline.base import CancelToken


# ---- helpers --------------------------------------------------------------

def _checkerboard(size: int, shift: float = 0.0) -> np.ndarray:
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    arr = np.sin(2 * np.pi * (x + shift) / 8) * np.sin(2 * np.pi * y / 8)
    return ((arr * 50) + 128).astype(np.uint8)


def _make_project(tmp: Path, n_frames: int = 5) -> tuple[ProjectModel, Path]:
    project_dir = tmp / "test.ofcd"
    project_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tmp / "frames"
    img_dir.mkdir(parents=True, exist_ok=True)

    imsave(str(img_dir / "ref.png"), _checkerboard(192, shift=0.0))
    for i in range(n_frames):
        imsave(str(img_dir / f"frame_{i:03d}.png"),
               _checkerboard(192, shift=0.05 * (i + 1)))

    proj = ProjectModel(
        format_version=0,
        name="speedup_test",
        created=datetime.now(timezone.utc).isoformat(),
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
        data={"frames_dir": str(img_dir), "pattern": "frame_*.png"},
        reference={"mode": "use_existing", "source": "ref.png"},
    )
    return proj, project_dir


def _run_compute(workers: int, tmp: Path) -> list[np.ndarray]:
    """Run Pre+Compute with the given worker count; return ordered eta_list."""
    from openfcd.io.annotation import AnnotationSchema
    proj, project_dir = _make_project(tmp / f"w{workers}", n_frames=5)
    proj.to_yaml(project_dir / "project.yaml")

    run_dir = project_dir / "runs" / f"run-w{workers}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rs = HDF5ResultStore.open(run_dir / "results.h5", mode="w")
    ctx: dict = {
        "project": proj,
        "project_dir": project_dir,
        "frames_dir": Path(proj.data.frames_dir),
        "pattern": proj.data.pattern,
        "frames_filter": None,
        "run_id": f"run-w{workers}",
        "result_store": rs,
        "batches_processed": [],
        "frame_count": 0,
        "frame_paths": [],
        "annotation": AnnotationSchema(),
        "workers": workers,
    }
    cancel = CancelToken()
    for stage in (PreprocessStage(), ComputeStage()):
        for _ev in stage.run(ctx, cancel):
            pass
    rs.close()
    return list(ctx["eta_list"])


# ---- tests ----------------------------------------------------------------

def test_resolve_default_workers_returns_positive() -> None:
    n = _resolve_default_workers()
    assert isinstance(n, int) and n >= 1


def test_resolve_default_workers_force_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENFCD_FORCE_SERIAL", "1")
    assert _resolve_default_workers() == 1


def test_compute_frame_returns_qc_datasets(tmp_path: Path) -> None:
    """compute_frame returns FrameResult with all required QC datasets."""
    from openfcd.cli.cmd_run import _build_geom_params
    from openfcd.pipeline.frame import (
        FrameInputsSnapshot,
        ProcessParams,
        build_frame_inputs,
        compute_frame,
    )

    proj, _ = _make_project(tmp_path, n_frames=1)
    ref = _checkerboard(192, shift=0.0).astype(np.float64)
    def_arr = _checkerboard(192, shift=0.1).astype(np.float64)
    geom = _build_geom_params(proj)

    snapshot = FrameInputsSnapshot(
        process=ProcessParams.from_project(proj),
        geom=geom,
        ref_img=ref,
        ref_shape=ref.shape,
        annotation=None,
        roi_box=None,
    )
    frame_path = tmp_path / "frames" / "ref.png"
    inputs = build_frame_inputs(snapshot, frame_path, fast_preview=False, def_img=def_arr)
    result = compute_frame(inputs)

    required = {
        "carrier_amplitude", "valid_mask", "artifact_mask",
        "phase_residual", "poisson_residual", "poisson_residual_x",
        "poisson_residual_y", "curl_inconsistency",
    }
    assert required.issubset(result.qc_datasets)
    for name in required:
        assert result.qc_datasets[name].shape == result.eta_mm.shape
    assert "saturated_ratio" in result.qc_attrs
    assert "invalid_ratio" in result.qc_attrs
    assert "carrier_amp_median" in result.qc_attrs
    assert "poisson_residual_rms" in result.qc_attrs
    assert "curl_inconsistency_rms" in result.qc_attrs
    assert "phase_residual_rms" in result.qc_attrs
    assert "slope_rms" in result.qc_attrs
    assert result.calibration.get("checker_cell_semantics") == (
        "single checker cell side length, not full black-white cycle"
    )


def test_serial_vs_parallel_bit_equal(tmp_path: Path, monkeypatch) -> None:
    """workers=1 vs workers=4 produce bit-equal eta_mm."""
    monkeypatch.setenv("OPENFCD_BLAS_THREADS", "1")
    eta_serial = _run_compute(1, tmp_path)
    eta_parallel = _run_compute(4, tmp_path)
    assert len(eta_serial) == len(eta_parallel) > 0
    for s, p in zip(eta_serial, eta_parallel):
        assert s.shape == p.shape
        ns, np_ = np.isnan(s), np.isnan(p)
        np.testing.assert_array_equal(ns, np_)
        np.testing.assert_array_equal(s[~ns], p[~np_])


def test_unannotated_frame_uses_per_frame_auto_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unannotated frames should auto-detect masks instead of needing drawings."""
    from openfcd.cli.cmd_run import _build_geom_params
    from openfcd.pipeline.frame import (
        FrameInputsSnapshot,
        ProcessParams,
        build_frame_inputs,
        compute_frame,
    )

    proj, _ = _make_project(tmp_path, n_frames=1)
    ref = _checkerboard(192, shift=0.0).astype(np.float64)
    def_arr = _checkerboard(192, shift=0.4).astype(np.float64)
    geom = _build_geom_params(proj)

    calls = []

    def _auto_object_mask(img, *_args, **_kwargs):
        calls.append(img.shape)
        return np.zeros(img.shape, dtype=bool)

    monkeypatch.setattr("openfcd.core.mask.auto_mask", _auto_object_mask)

    snapshot = FrameInputsSnapshot(
        process=ProcessParams.from_project(proj),
        geom=geom,
        ref_img=ref,
        ref_shape=ref.shape,
        annotation=None,
        roi_box=None,
    )
    frame_path = tmp_path / "frames" / "ref.png"
    inputs = build_frame_inputs(snapshot, frame_path, fast_preview=False, def_img=def_arr)
    result = compute_frame(inputs)
    assert result.pixel_per_mm > 0
    assert result.eta_mm.shape == ref.shape
    assert calls  # auto_mask was called


def test_reference_zero_field_fills_small_mask_holes(tmp_path: Path) -> None:
    """Reference-vs-reference stays zero while small enclosed mask holes are filled."""
    from openfcd.cli.cmd_run import _build_geom_params
    from openfcd.core.mask import Polygon
    from openfcd.pipeline.frame import (
        FrameInputs,
        FrameInputsSnapshot,
        ProcessParams,
        build_frame_inputs,
        compute_frame,
    )

    proj, _ = _make_project(tmp_path, n_frames=1)
    proj.process.small_hole_fill_radius_mm = 2.0
    ref = _checkerboard(192, shift=0.0).astype(np.float64)
    geom = _build_geom_params(proj)
    small_poly = Polygon([(92, 92), (92, 93), (93, 93), (93, 92)])

    snapshot = FrameInputsSnapshot(
        process=ProcessParams.from_project(proj),
        geom=geom,
        ref_img=ref,
        ref_shape=ref.shape,
        annotation=None,
        roi_box=None,
    )
    frame_path = tmp_path / "frames" / "ref.png"
    inputs = build_frame_inputs(snapshot, frame_path, fast_preview=False, def_img=ref.copy())
    # Override robot_poly — build_frame_inputs gives None since no annotation
    inputs = FrameInputs(
        ref_img=inputs.ref_img,
        def_img=inputs.def_img,
        geom=inputs.geom,
        process=inputs.process,
        roi_box=inputs.roi_box,
        robot_poly=small_poly,
        fast_preview=inputs.fast_preview,
        ref_shape=inputs.ref_shape,
    )
    result = compute_frame(inputs)
    assert result.pixel_per_mm > 0
    assert np.isfinite(result.eta_mm[90:96, 90:96]).all()
    assert np.nanmax(np.abs(result.eta_mm)) == 0.0


def test_reference_zero_field_keeps_large_mask_nan(tmp_path: Path) -> None:
    """Small-hole cleanup must not fill the robot-body-sized mask."""
    from openfcd.cli.cmd_run import _build_geom_params
    from openfcd.core.mask import Polygon
    from openfcd.pipeline.frame import (
        FrameInputs,
        FrameInputsSnapshot,
        ProcessParams,
        build_frame_inputs,
        compute_frame,
    )

    proj, _ = _make_project(tmp_path, n_frames=1)
    proj.process.small_hole_fill_radius_mm = 10.0
    ref = _checkerboard(192, shift=0.0).astype(np.float64)
    geom = _build_geom_params(proj)
    large_poly = Polygon([(70, 70), (70, 120), (120, 120), (120, 70)])

    snapshot = FrameInputsSnapshot(
        process=ProcessParams.from_project(proj),
        geom=geom,
        ref_img=ref,
        ref_shape=ref.shape,
        annotation=None,
        roi_box=None,
    )
    frame_path = tmp_path / "frames" / "ref.png"
    inputs = build_frame_inputs(snapshot, frame_path, fast_preview=False, def_img=ref.copy())
    inputs = FrameInputs(
        ref_img=inputs.ref_img,
        def_img=inputs.def_img,
        geom=inputs.geom,
        process=inputs.process,
        roi_box=inputs.roi_box,
        robot_poly=large_poly,
        fast_preview=inputs.fast_preview,
        ref_shape=inputs.ref_shape,
    )
    result = compute_frame(inputs)
    assert np.isnan(result.eta_mm[80:110, 80:110]).all()


def test_compute_stage_ignores_hidden_global_polygon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run should not apply invisible global polygons to unannotated frames."""
    from openfcd.io.annotation import AnnotationSchema
    from openfcd.pipeline.frame import FrameResult

    proj, project_dir = _make_project(tmp_path, n_frames=2)
    seen: list = []

    def _fake_compute_frame(inputs, *, progress_cb=None, cancel=None):
        seen.append(inputs.robot_poly)
        eta = np.zeros(inputs.ref_shape, dtype=np.float64)
        return FrameResult(eta_mm=eta, qc_datasets=None, diagnostics={})

    monkeypatch.setattr("openfcd.pipeline.frame.compute_frame", _fake_compute_frame)

    ctx: dict = {
        "project": proj,
        "project_dir": project_dir,
        "frames_dir": Path(proj.data.frames_dir),
        "pattern": proj.data.pattern,
        "frames_filter": None,
        "run_id": "run-global-hidden",
        "result_store": None,
        "batches_processed": [],
        "frame_count": 0,
        "frame_paths": [],
        "annotation": AnnotationSchema(
            polygons=[{
                "vertices": [[40, 40], [40, 80], [90, 80], [90, 40]],
                "label": "body",
            }]
        ),
    }
    cancel = CancelToken()
    for stage in (PreprocessStage(), ComputeStage()):
        for _event in stage.run(ctx, cancel):
            pass

    assert seen == [None, None]
