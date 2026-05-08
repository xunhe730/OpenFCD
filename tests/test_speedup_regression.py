"""Regression: serial vs parallel ComputeStage produce bit-equal eta_mm,
ref_invariants hoist preserves output, default-workers resolver returns >=1."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from skimage.io import imsave

from openfcd.cli.cmd_run import (
    ComputeStage,
    PreprocessStage,
    _compute_ref_invariants,
    _compute_single_frame,
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

    class _StubStore:
        annotation = AnnotationSchema()

    store = _StubStore()
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
        "annotation": store.annotation,
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


def test_compute_ref_invariants_matches_inline(tmp_path: Path) -> None:
    """Hoisted ref_ff equals what _compute_single_frame would compute inline."""
    proj, _ = _make_project(tmp_path, n_frames=1)
    from openfcd.core.flatfield import flatfield_normalize
    ref = _checkerboard(192, shift=0.0).astype(np.float64)
    inv = _compute_ref_invariants(ref, proj, roi_box=None)
    h, w = ref.shape
    sigma_expected = float(np.clip(max(h, w) * 0.06, 100.0, 2000.0))
    assert inv.sigma == pytest.approx(sigma_expected)
    np.testing.assert_array_equal(inv.ref_ff, flatfield_normalize(ref, sigma=sigma_expected))


def test_compute_single_frame_returns_qc_datasets(tmp_path: Path) -> None:
    proj, _ = _make_project(tmp_path, n_frames=1)
    ref = _checkerboard(192, shift=0.0).astype(np.float64)
    deformed = _checkerboard(192, shift=0.1).astype(np.float64)
    from openfcd.cli.cmd_run import _build_geom_params

    result = _compute_single_frame(ref, deformed, _build_geom_params(proj), proj, fast_preview=False)

    required = {
        "carrier_amplitude",
        "valid_mask",
        "artifact_mask",
        "phase_residual",
        "poisson_residual",
        "poisson_residual_x",
        "poisson_residual_y",
        "curl_inconsistency",
    }
    assert required.issubset(result.qc_datasets)
    for name in required:
        assert result.qc_datasets[name].shape == result.eta_mm.shape
    assert "saturated_ratio" in result.qc_attrs
    assert "invalid_ratio" in result.qc_attrs
    assert "carrier_amp_median" in result.qc_attrs
    assert "poisson_residual_rms" in result.qc_attrs
    assert "curl_inconsistency_rms" in result.qc_attrs
    assert result.calibration["checker_cell_semantics"] == "single checker cell side length, not full black-white cycle"


def test_serial_vs_parallel_bit_equal(tmp_path: Path, monkeypatch) -> None:
    """workers=1 vs workers=4 produce bit-equal eta_mm.

    Pin OPENFCD_BLAS_THREADS=1 so native math libraries don't inject
    nondeterminism (e.g. inpaint_biharmonic's linear solver).
    The default is now "auto"; this test opts back into deterministic mode.
    """
    monkeypatch.setenv("OPENFCD_BLAS_THREADS", "1")
    eta_serial = _run_compute(1, tmp_path)
    eta_parallel = _run_compute(4, tmp_path)
    assert len(eta_serial) == len(eta_parallel) > 0
    for s, p in zip(eta_serial, eta_parallel):
        assert s.shape == p.shape
        ns, np_ = np.isnan(s), np.isnan(p)
        np.testing.assert_array_equal(ns, np_)
        np.testing.assert_array_equal(s[~ns], p[~np_])


def test_ref_invariants_hoist_bit_equal(tmp_path: Path) -> None:
    """_compute_single_frame with ref_invariants matches inline path."""
    proj, _ = _make_project(tmp_path, n_frames=1)
    ref = _checkerboard(192, shift=0.0).astype(np.float64)
    deformed = _checkerboard(192, shift=0.4).astype(np.float64)
    from openfcd.cli.cmd_run import _build_geom_params
    geom = _build_geom_params(proj)

    inline = _compute_single_frame(ref, deformed, geom, proj)
    inv = _compute_ref_invariants(ref, proj, roi_box=None)
    hoisted = _compute_single_frame(ref, deformed, geom, proj, ref_invariants=inv)

    assert inline.pixel_per_mm > 0
    assert inline.calibration["eta_unit"] == "mm"
    assert inline.shape == hoisted.shape
    ni, nh = np.isnan(inline), np.isnan(hoisted)
    np.testing.assert_array_equal(ni, nh)
    np.testing.assert_allclose(inline[~ni], hoisted[~nh], rtol=0, atol=0)


def test_unannotated_frame_uses_per_frame_auto_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unannotated frames should auto-detect masks instead of needing drawings."""
    proj, _ = _make_project(tmp_path, n_frames=1)
    ref = _checkerboard(192, shift=0.0).astype(np.float64)
    deformed = _checkerboard(192, shift=0.4).astype(np.float64)
    from openfcd.cli.cmd_run import _build_geom_params
    geom = _build_geom_params(proj)

    calls = []

    def _auto_object_mask(img, *_args, **_kwargs):
        calls.append(img.shape)
        return np.zeros(img.shape, dtype=bool)

    monkeypatch.setattr("openfcd.core.mask.auto_mask", _auto_object_mask)

    eta = _compute_single_frame(ref, deformed, geom, proj, robot_poly=None)
    assert eta.pixel_per_mm > 0
    assert eta.shape == ref.shape
    assert calls == [ref.shape]


def test_reference_zero_field_fills_small_mask_holes(tmp_path: Path) -> None:
    """Reference-vs-reference stays zero while small enclosed mask holes are filled."""
    from openfcd.cli.cmd_run import _build_geom_params
    from openfcd.core.mask import Polygon

    proj, _ = _make_project(tmp_path, n_frames=1)
    proj.process.small_hole_fill_radius_mm = 2.0
    ref = _checkerboard(192, shift=0.0).astype(np.float64)
    geom = _build_geom_params(proj)
    small_poly = Polygon([(92, 92), (92, 93), (93, 93), (93, 92)])

    eta = _compute_single_frame(ref, ref.copy(), geom, proj, robot_poly=small_poly)

    assert eta.pixel_per_mm > 0
    assert np.isfinite(eta[90:96, 90:96]).all()
    assert np.nanmax(np.abs(eta)) == 0.0


def test_reference_zero_field_keeps_large_mask_nan(tmp_path: Path) -> None:
    """Small-hole cleanup must not fill the robot-body-sized mask."""
    from openfcd.cli.cmd_run import _build_geom_params
    from openfcd.core.mask import Polygon

    proj, _ = _make_project(tmp_path, n_frames=1)
    proj.process.small_hole_fill_radius_mm = 10.0
    ref = _checkerboard(192, shift=0.0).astype(np.float64)
    geom = _build_geom_params(proj)
    large_poly = Polygon([(70, 70), (70, 120), (120, 120), (120, 70)])

    eta = _compute_single_frame(ref, ref.copy(), geom, proj, robot_poly=large_poly)

    assert np.isnan(eta[80:110, 80:110]).all()


def test_compute_stage_ignores_hidden_global_polygon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run should not apply invisible global polygons to unannotated frames."""
    from openfcd.io.annotation import AnnotationSchema
    import openfcd.cli.cmd_run as cmd_run

    proj, project_dir = _make_project(tmp_path, n_frames=2)
    seen: list = []

    def _fake_compute(ref_img, _def_img, *_args, **kwargs):
        seen.append(kwargs.get("robot_poly"))
        return np.zeros(ref_img.shape, dtype=np.float64)

    monkeypatch.setattr(cmd_run, "_compute_single_frame", _fake_compute)

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
