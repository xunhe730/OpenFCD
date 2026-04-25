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


def test_serial_vs_parallel_bit_equal(tmp_path: Path) -> None:
    """workers=1 vs workers=4 produce bit-equal eta_mm.

    The parallel branch wraps the thread pool in ``threadpool_limits(1)`` so
    native math libraries (BLAS / OpenMP) don't oversubscribe and inject
    nondeterminism into routines like ``inpaint_biharmonic``.
    """
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

    assert inline.shape == hoisted.shape
    ni, nh = np.isnan(inline), np.isnan(hoisted)
    np.testing.assert_array_equal(ni, nh)
    np.testing.assert_allclose(inline[~ni], hoisted[~nh], rtol=0, atol=0)
