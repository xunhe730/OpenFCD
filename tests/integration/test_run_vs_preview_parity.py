"""AC-B1: ``compute_frame`` invoked directly produces η byte-equal to the
``ComputeStage`` η written into ``results.h5`` for the same frame."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from skimage.io import imsave

from openfcd.cli.cmd_run import (
    ComputeStage,
    PreprocessStage,
    _build_geom_params,
    _resolve_reference,
)
from openfcd.io.annotation import AnnotationSchema
from openfcd.io.project import ProjectModel
from openfcd.io.result import HDF5ResultStore
from openfcd.pipeline.base import CancelToken
from openfcd.pipeline.frame import (
    FrameInputsSnapshot,
    build_frame_inputs,
    compute_frame,
)


def _checkerboard(size: int, shift: float = 0.0) -> np.ndarray:
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    arr = np.sin(2 * np.pi * (x + shift) / 8) * np.sin(2 * np.pi * y / 8)
    return ((arr * 50) + 128).astype(np.uint8)


def _make_one_frame_project(tmp: Path) -> tuple[ProjectModel, Path, Path]:
    project_dir = tmp / "parity.ofcd"
    project_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tmp / "frames"
    img_dir.mkdir(parents=True, exist_ok=True)
    imsave(str(img_dir / "ref.png"), _checkerboard(192, shift=0.0))
    frame_path = img_dir / "frame_000.png"
    imsave(str(frame_path), _checkerboard(192, shift=0.07))
    proj = ProjectModel(
        format_version=0,
        name="parity",
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
    return proj, project_dir, frame_path


def _nan_aware_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        return False
    finite = ~nan_a
    return bool(np.array_equal(a[finite], b[finite]))


def test_run_vs_preview_byte_equal(tmp_path: Path) -> None:
    proj, project_dir, frame_path = _make_one_frame_project(tmp_path)
    proj.to_yaml(project_dir / "project.yaml")

    # --- preview path: build_frame_inputs + compute_frame, identical to GUI worker
    geom = _build_geom_params(proj)
    ref_img = _resolve_reference(proj, project_dir)
    snapshot = FrameInputsSnapshot.from_session(AnnotationSchema(), proj, geom, ref_img)
    inputs = build_frame_inputs(snapshot, frame_path, fast_preview=True)
    preview = compute_frame(inputs)

    # --- run path: ComputeStage end-to-end
    run_dir = project_dir / "runs" / "run-parity"
    run_dir.mkdir(parents=True, exist_ok=True)
    rs = HDF5ResultStore.open(run_dir / "results.h5", mode="w")
    ctx: dict = {
        "project": proj,
        "project_dir": project_dir,
        "frames_dir": Path(proj.data.frames_dir),
        "pattern": proj.data.pattern,
        "frames_filter": None,
        "run_id": "run-parity",
        "result_store": rs,
        "batches_processed": [],
        "frame_count": 0,
        "frame_paths": [],
        "annotation": AnnotationSchema(),
        "workers": 1,
    }
    cancel = CancelToken()
    for stage in (PreprocessStage(), ComputeStage()):
        for _ev in stage.run(ctx, cancel):
            pass
    rs.close()

    # Read frame back from HDF5
    rs2 = HDF5ResultStore.open(run_dir / "results.h5", mode="r")
    try:
        frame_ids = rs2.list_frames("default")
        assert frame_ids, "ComputeStage should have written at least one frame"
        run_eta = rs2.read_frame("default", frame_ids[0])
    finally:
        rs2.close()

    assert _nan_aware_equal(preview.eta_mm, run_eta), (
        "preview and Run η must be byte-equal for the same inputs"
    )
