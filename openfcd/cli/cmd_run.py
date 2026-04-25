"""`openfcd run` — run the FCD pipeline on a project."""
from __future__ import annotations

import signal
import time
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterator

import numpy as np
import typer

from openfcd.io.image import scan_frames
from openfcd.io.project import ProjectModel
from openfcd.io.result import HDF5ResultStore
from openfcd.io.store import FileSessionStore
from openfcd.pipeline.base import CancelToken, CancelledError, Stage, StageEvent
from openfcd.pipeline.runner import PipelineRunner

from openfcd.cli._output import format_progress, stage_event_to_json


# ---------------------------------------------------------------------------
# Helper: build GeomParams from project config
# ---------------------------------------------------------------------------

def _build_geom_params(project: ProjectModel):
    """Derive optical geometry parameters from project.yaml configuration."""
    from openfcd.geometry.optical import OpticalGeometry
    from openfcd.pipeline.compute import GeomParams

    og = OpticalGeometry(project.geometry)
    return GeomParams(
        pattern_period_mm=og.pattern_period_mm,
        alpha=og.alpha,
        h_p_eff_mm=og.h_p_eff_mm,
    )


# ---------------------------------------------------------------------------
# Helper: resolve reference image
# ---------------------------------------------------------------------------

def _load_reference_source(project: ProjectModel, project_dir: Path, src: str) -> np.ndarray:
    """Resolve and load a reference image from an explicit source path/name."""
    from openfcd.pipeline.compute import load_gray

    ref_path = Path(src)
    # 1) Already absolute and exists?
    if ref_path.is_absolute() and ref_path.exists():
        return load_gray(ref_path)
    # 2) Look in frames_dir first (most common: user set a frame by filename)
    frames_dir = Path(project.data.frames_dir)
    candidate = frames_dir / ref_path.name
    if candidate.exists():
        return load_gray(candidate)
    # 3) Relative to project_dir
    candidate2 = project_dir / ref_path
    if candidate2.exists():
        return load_gray(candidate2)
    raise FileNotFoundError(
        f"Reference image not found: '{src}'\n"
        f"  Tried: {candidate}\n"
        f"  Tried: {candidate2}"
    )


def _resolve_reference(project: ProjectModel, project_dir: Path) -> np.ndarray:
    """Load or build the reference image based on project.reference config.

    GUI semantics are strict: once a specific reference source is chosen,
    that image wins regardless of the stored mode. Mode is only used as a
    fallback when no explicit source exists.
    """
    ref_cfg = project.reference
    mode = ref_cfg.mode

    if ref_cfg.source:
        return _load_reference_source(project, project_dir, ref_cfg.source)

    if mode == "use_existing":
        raise ValueError("reference.mode='use_existing' but reference.source is empty")

    elif mode == "first_frame":
        frames = scan_frames(project.data.frames_dir, project.data.pattern)
        if not frames:
            raise FileNotFoundError("No frames found for first_frame reference mode")
        return load_gray(frames[0])

    elif mode == "build":
        from openfcd.core.reference_builder import build_reference
        params = ref_cfg.build_params
        ref_img = build_reference(
            frames_dir=Path(project.data.frames_dir),
            pattern=project.data.pattern,
            n=params.get("n", 100),
            stride=params.get("stride", 11),
            reducer=params.get("reducer", "median"),
        )
        # Cache the built reference to the project directory
        cache_path = project_dir / "reference_built.npy"
        np.save(cache_path, ref_img)
        return ref_img

    else:
        raise ValueError(f"Unknown reference.mode: {mode!r}")


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

class PreprocessStage:
    """Load frames, resolve reference image, set up geometric calibration."""

    name: str = "preprocess"

    def run(
        self, ctx: dict, cancel: CancelToken | None = None
    ) -> Iterator[StageEvent]:
        run_id: str = ctx.get("run_id", "")
        project: ProjectModel = ctx["project"]
        project_dir: Path = ctx["project_dir"]
        frames_dir = Path(project.data.frames_dir)
        pattern = project.data.pattern
        frames_filter: str | None = ctx.get("frames_filter")

        yield StageEvent(
            kind="start", stage=self.name, batch=None, frame_idx=None,
            substage=None, progress=0.0, total=None, completed=None,
            metrics={"action": "scanning_frames"}, run_id=run_id,
        )

        if cancel and cancel.is_cancelled:
            yield StageEvent(
                kind="cancel", stage=self.name, batch=None, frame_idx=None,
                substage=None, progress=None, total=None, completed=None,
                metrics={}, run_id=run_id,
            )
            return

        # 1. Scan frames
        frames = scan_frames(frames_dir, pattern)
        if frames_filter:
            frames = _filter_frames(frames, frames_filter)
        total = len(frames)

        yield StageEvent(
            kind="progress", stage=self.name, batch=None, frame_idx=None,
            substage="frames_scanned", progress=0.3, total=total,
            completed=total, metrics={"frame_count": total}, run_id=run_id,
        )

        # 2. Resolve reference image
        ref_img = _resolve_reference(project, project_dir)
        ctx["reference_image"] = ref_img

        yield StageEvent(
            kind="progress", stage=self.name, batch=None, frame_idx=None,
            substage="reference_loaded", progress=0.6, total=total,
            completed=total,
            metrics={"ref_mode": project.reference.mode, "ref_shape": list(ref_img.shape)},
            run_id=run_id,
        )

        # 3. Build GeomParams from optical stack
        geom = _build_geom_params(project)
        ctx["geom_params"] = geom

        # 4. Store frame paths and count
        ctx["frame_paths"] = frames
        ctx["frame_count"] = total

        yield StageEvent(
            kind="progress", stage=self.name, batch=None, frame_idx=None,
            substage="calibration_ready", progress=0.9, total=total,
            completed=total,
            metrics={
                "alpha": round(geom.alpha, 4),
                "h_p_eff_mm": round(geom.h_p_eff_mm, 3),
                "pattern_period_mm": geom.pattern_period_mm,
            },
            run_id=run_id,
        )

        yield StageEvent(
            kind="finish", stage=self.name, batch=None, frame_idx=None,
            substage=None, progress=1.0, total=total, completed=total,
            metrics={"frame_count": total}, run_id=run_id,
        )

    def dry_run(self, ctx: dict) -> list[str]:
        return ["scan_frames", "resolve_reference", "build_geom_params"]


class ComputeStage:
    """Process each frame through the FCD pipeline.

    Each frame is paired with the shared reference image. The process()
    function from openfcd.pipeline.compute does the heavy lifting:
    flatfield → carriers → mask → inpaint → FCD → calibrate → eta_mm.
    Results are written per-frame into the HDF5ResultStore.
    """

    name: str = "compute"

    def run(
        self, ctx: dict, cancel: CancelToken | None = None
    ) -> Iterator[StageEvent]:
        from openfcd.pipeline.compute import process, GeomParams, load_gray

        run_id: str = ctx.get("run_id", "")
        frame_paths: list[Path] = ctx.get("frame_paths", [])
        frame_count: int = ctx.get("frame_count", 0)
        ref_img: np.ndarray = ctx["reference_image"]
        geom: GeomParams = ctx["geom_params"]
        result_store: HDF5ResultStore | None = ctx.get("result_store")
        project: ProjectModel = ctx["project"]

        yield StageEvent(
            kind="start", stage=self.name, batch=None, frame_idx=None,
            substage=None, progress=0.0, total=frame_count, completed=0,
            metrics={"action": "starting_compute"}, run_id=run_id,
        )

        if cancel and cancel.is_cancelled:
            yield StageEvent(
                kind="cancel", stage=self.name, batch=None, frame_idx=None,
                substage=None, progress=None, total=None, completed=None,
                metrics={}, run_id=run_id,
            )
            return

        # Resolve annotation ROI and polygon for masking
        annotation = ctx.get("annotation")  # AnnotationSchema | None
        roi_box = None
        polygon_map: dict[str, list] = {}
        if annotation is not None:
            roi = annotation.roi
            if not roi.is_empty:
                from openfcd.core.mask import Box
                roi_box = Box(
                    row0=int(roi.y),
                    col0=int(roi.x),
                    height=int(roi.height),
                    width=int(roi.width),
                )
            polygon_map = annotation.frame_polygons

        # Hoist invariant ref-side work (ROI crop + flatfield) once per run.
        ref_invariants = _compute_ref_invariants(ref_img, project, roi_box=roi_box)

        # Per-frame polygon resolver
        def _resolve_frame_poly(name: str):
            if name not in polygon_map:
                return None
            polys = polygon_map[name]
            if not polys or not polys[0].vertices:
                return None
            from openfcd.core.mask import Polygon
            return Polygon([(float(v[0]), float(v[1])) for v in polys[0].vertices])

        # Single-frame body shared by serial and parallel paths.
        def _process_one(idx: int, frame_path: Path) -> tuple[int, np.ndarray | None, str | None]:
            """Returns (idx, eta_mm, error_msg). eta_mm is None on error."""
            try:
                deformed_img = load_gray(frame_path)
                if deformed_img.shape != ref_img.shape:
                    return idx, None, "shape_mismatch"
                eta_mm = _compute_single_frame(
                    ref_img, deformed_img, geom, project,
                    roi_box=roi_box,
                    robot_poly=_resolve_frame_poly(frame_path.name),
                    ref_invariants=ref_invariants,
                )
                return idx, eta_mm, None
            except Exception as exc:  # noqa: BLE001
                return idx, None, str(exc)

        eta_list: list[np.ndarray | None] = [None] * frame_count
        processed = 0
        errors = 0
        t0 = time.time()

        # Resolve worker count. workers<=0 means "auto" → use P-cores on Darwin
        # else (cpu_count - 1). When workers==1, run serial (no thread overhead).
        workers_req = int(ctx.get("workers", 1) or 1)
        if workers_req <= 0:
            workers_req = _resolve_default_workers()
        # Cap at frame count.
        n_workers = max(1, min(workers_req, frame_count if frame_count > 0 else 1))

        write_lock = __import__("threading").Lock()

        def _emit_progress(idx: int, frame_name: str) -> StageEvent:
            prog = (processed + errors) / max(frame_count, 1)
            elapsed = time.time() - t0
            eta_sec = 0.0
            done = processed + errors
            if done > 0:
                eta_sec = (elapsed / done) * (frame_count - done)
            return StageEvent(
                kind="progress", stage=self.name, batch="default",
                frame_idx=idx, substage="processing_frame", progress=prog,
                total=frame_count, completed=processed,
                metrics={
                    "frame": frame_name,
                    "errors": errors,
                    "eta_seconds": round(eta_sec, 1),
                },
                run_id=run_id,
            )

        def _record(idx: int, frame_path: Path, eta_mm: np.ndarray | None, err: str | None) -> None:
            nonlocal processed, errors
            if eta_mm is not None:
                eta_list[idx] = eta_mm
                if result_store is not None:
                    with write_lock:
                        result_store.write_frame(
                            "default", idx, eta_mm,
                            {"status": "ok", "frame_path": str(frame_path.name)},
                        )
                processed += 1
            else:
                errors += 1
                if result_store is not None:
                    with write_lock:
                        result_store.write_frame(
                            "default", idx,
                            np.zeros((1, 1), dtype=np.float64),
                            {
                                "status": "error",
                                "message": err or "",
                                "frame_path": str(frame_path.name),
                            },
                        )

        if n_workers <= 1:
            for idx, frame_path in enumerate(frame_paths):
                if cancel and cancel.is_cancelled:
                    break
                idx_, eta_mm, err = _process_one(idx, frame_path)
                _record(idx_, frame_path, eta_mm, err)
                yield _emit_progress(idx, frame_path.name)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from threadpoolctl import threadpool_limits

            # Pin BLAS / OpenMP to 1 thread per pipeline worker. Without this,
            # numpy/scipy/scikit-image's native pools fight the Python pool and
            # produce nondeterministic output (especially inpaint_biharmonic's
            # linear solver), violating serial↔parallel equivalence.
            with threadpool_limits(limits=1):
                executor = ThreadPoolExecutor(
                    max_workers=n_workers, thread_name_prefix="openfcd-fcd"
                )
                try:
                    futures = {
                        executor.submit(_process_one, i, fp): (i, fp)
                        for i, fp in enumerate(frame_paths)
                    }
                    for fut in as_completed(futures):
                        if cancel and cancel.is_cancelled:
                            for f in futures:
                                f.cancel()
                            break
                        idx_orig, frame_path = futures[fut]
                        try:
                            idx_, eta_mm, err = fut.result()
                        except Exception as exc:  # noqa: BLE001
                            idx_, eta_mm, err = idx_orig, None, str(exc)
                        _record(idx_, frame_path, eta_mm, err)
                        yield _emit_progress(idx_, frame_path.name)
                finally:
                    executor.shutdown(wait=True, cancel_futures=True)

        # Store eta stack in context for postprocess (drop None slots = errored frames)
        ctx["eta_list"] = [e for e in eta_list if e is not None]
        ctx["batches_processed"] = ["default"]
        ctx["compute_errors"] = errors

        yield StageEvent(
            kind="finish", stage=self.name, batch=None, frame_idx=None,
            substage=None, progress=1.0, total=frame_count,
            completed=processed,
            metrics={"frames_processed": processed, "errors": errors},
            run_id=run_id,
        )

    def dry_run(self, ctx: dict) -> list[str]:
        return ["process_frames", "write_per_frame_hdf5"]


class PostprocessStage:
    """Compute batch summary statistics and finalize results.h5."""

    name: str = "postprocess"

    def run(
        self, ctx: dict, cancel: CancelToken | None = None
    ) -> Iterator[StageEvent]:
        run_id: str = ctx.get("run_id", "")
        result_store: HDF5ResultStore | None = ctx.get("result_store")
        eta_list: list[np.ndarray] = ctx.get("eta_list", [])
        total_frames = ctx.get("frame_count", 0)
        project: ProjectModel = ctx["project"]

        yield StageEvent(
            kind="start", stage=self.name, batch=None, frame_idx=None,
            substage=None, progress=0.0, total=None, completed=None,
            metrics={"action": "finalizing"}, run_id=run_id,
        )

        if cancel and cancel.is_cancelled:
            yield StageEvent(
                kind="cancel", stage=self.name, batch=None, frame_idx=None,
                substage=None, progress=None, total=None, completed=None,
                metrics={}, run_id=run_id,
            )
            return

        if eta_list and result_store is not None:
            # Compute summary statistics
            # Filter to same-size arrays (some might have failed)
            target_shape = eta_list[0].shape
            valid_etas = [e for e in eta_list if e.shape == target_shape]

            if valid_etas:
                stack = np.stack(valid_etas, axis=0)

                # Per-pixel statistics (ignoring NaN)
                eta_mean = np.nanmean(stack, axis=0)
                eta_median = np.nanmedian(stack, axis=0)
                eta_rms = np.sqrt(np.nanmean(stack ** 2, axis=0))

                yield StageEvent(
                    kind="progress", stage=self.name, batch="default",
                    frame_idx=None, substage="writing_summaries",
                    progress=0.5, total=total_frames, completed=total_frames,
                    metrics={"n_valid_frames": len(valid_etas)},
                    run_id=run_id,
                )

                # Write batch data
                result_store.write_batch(
                    "default",
                    data={
                        "eta_mean": eta_mean,
                        "eta_median": eta_median,
                        "eta_rms": eta_rms,
                        "eta_stack": stack,
                    },
                    meta={
                        "n_frames": len(valid_etas),
                        "n_errors": ctx.get("compute_errors", 0),
                        "project_name": project.name,
                        "config_fingerprint": _quick_fingerprint(project),
                    },
                )

        # Write metadata
        if result_store is not None:
            result_store._file.attrs["annotation_fingerprint"] = _quick_fingerprint(project)
            result_store.close()
            ctx["result_store"] = None  # mark as closed

        yield StageEvent(
            kind="finish", stage=self.name, batch=None, frame_idx=None,
            substage=None, progress=1.0, total=total_frames,
            completed=total_frames,
            metrics={"batches": ctx.get("batches_processed", [])},
            run_id=run_id,
        )

    def dry_run(self, ctx: dict) -> list[str]:
        return ["compute_summaries", "write_metadata", "close_store"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _RefInvariants:
    """Frame-invariant ref-side work: ROI-cropped ref + its flatfield.

    Computed once per run and shared across all frames. Skipped when fast_preview
    is on (each preview call has its own downsample factor → different sigma)."""

    __slots__ = ("ref_img", "ref_ff", "sigma")

    def __init__(self, ref_img: np.ndarray, ref_ff: np.ndarray, sigma: float):
        self.ref_img = ref_img
        self.ref_ff = ref_ff
        self.sigma = sigma


def _compute_ref_invariants(
    ref_img: np.ndarray, project: ProjectModel, roi_box=None
) -> _RefInvariants:
    """Hoist ROI crop + flatfield of the reference image (constant across frames)."""
    from openfcd.core.flatfield import flatfield_normalize

    if getattr(project.process, "flatfield_sigma_auto", True):
        h_img, w_img = ref_img.shape
        sigma = float(np.clip(max(h_img, w_img) * 0.06, 100.0, 2000.0))
    else:
        sigma = float(project.process.flatfield_sigma)
    if roi_box is not None:
        r0 = max(0, roi_box.row0)
        c0 = max(0, roi_box.col0)
        r1 = min(ref_img.shape[0], roi_box.row0 + roi_box.height)
        c1 = min(ref_img.shape[1], roi_box.col0 + roi_box.width)
        ref_img = ref_img[r0:r1, c0:c1]
    ref_ff = flatfield_normalize(ref_img, sigma=sigma)
    return _RefInvariants(ref_img=ref_img, ref_ff=ref_ff, sigma=sigma)


def _compute_single_frame(
    ref_img: np.ndarray,
    def_img: np.ndarray,
    geom,
    project: ProjectModel,
    roi_box=None,
    robot_poly=None,
    robot_pad_px: int = 4,
    fast_preview: bool = False,
    progress_cb=None,
    *,
    ref_invariants: "_RefInvariants | None" = None,
) -> np.ndarray:
    """Run the FCD pipeline on a single (reference, deformed) pair.

    Uses core functions directly rather than the file-based process() API,
    since we already have the images loaded in memory.

    Args:
        roi_box:      optional Box for ROI crop (from annotation)
        robot_poly:   optional Polygon for occlusion mask (from annotation mask)
        robot_pad_px: pixels to dilate the occlusion mask
        fast_preview: if True, auto-downsample images >3000px on longest side and
                      skip suppress_nonphysical_eta_filaments (2–4× speedup)
        progress_cb:  optional callable(pct: int, label: str) for progress updates
        ref_invariants: precomputed ROI-cropped ref + flatfield (run-loop hoist).
                        Mutually exclusive with fast_preview.
    """
    def _report(pct: int, label: str) -> None:
        if progress_cb is not None:
            progress_cb(pct, label)
    from openfcd.core.fcd import calculate_carriers, carriers_pixel_per_mm, fcd as fcd_height
    from openfcd.core.flatfield import flatfield_normalize
    from openfcd.core.inpaint import inpaint_fft, synthesize_from_carriers
    from openfcd.core.mask import (
        polygon_mask,
        auto_mask,
        detect_filament_occluders,
        find_oriented_polygon,
        find_largest_interior_blob,
    )
    from openfcd.pipeline.compute import (
        cosine_taper,
        detrend_plane,
        edge_margin_mask,
        suppress_nonphysical_eta_filaments,
    )

    if ref_invariants is not None and fast_preview:
        # fast_preview path performs its own downsample → cached sigma/ref_ff
        # would be wrong. Drop the hoist for safety.
        ref_invariants = None

    if ref_invariants is not None:
        sigma = ref_invariants.sigma
    elif getattr(project.process, "flatfield_sigma_auto", True):
        h_img, w_img = ref_img.shape
        sigma = float(np.clip(max(h_img, w_img) * 0.06, 100.0, 2000.0))
    else:
        sigma = project.process.flatfield_sigma
    taper_alpha = project.process.taper.alpha
    edge_mm = project.process.edge_nan_mm
    same_input = np.array_equal(ref_img, def_img)

    # Fast-preview: auto-downsample images wider/taller than 3000px.
    # Processing at half resolution is ~4× faster; result is upsampled back.
    _original_shape: tuple | None = None
    _downsample_scale = 1.0  # tracks pixel scale for sigma correction
    if fast_preview:
        h0, w0 = ref_img.shape
        # Target ~2500px on longest side (factor=2 for 5000-5999px, etc.)
        # Capped at factor=2 to keep carrier period ≥25px and HP sigma > wave period
        factor = min(2, max(h0, w0) // 2500)
        if factor >= 2:
            from scipy.ndimage import zoom
            scale = 1.0 / factor
            _downsample_scale = scale
            ref_img = zoom(ref_img, scale, order=1)
            def_img = zoom(def_img, scale, order=1)
            sigma = sigma * scale
            if robot_poly is not None:
                from openfcd.core.mask import Polygon as _Polygon
                robot_poly = _Polygon([(v[0] * scale, v[1] * scale) for v in robot_poly.vertices])
            if roi_box is not None:
                from openfcd.core.mask import Box as _Box
                roi_box = _Box(
                    row0=int(roi_box.row0 * scale),
                    col0=int(roi_box.col0 * scale),
                    height=int(roi_box.height * scale),
                    width=int(roi_box.width * scale),
                )
            _original_shape = (h0, w0)

    _report(5, "Setup")

    # Apply ROI crop if specified. When ref_invariants is supplied, ref_img is
    # already ROI-cropped; def_img and robot_poly still need the same crop.
    if roi_box is not None:
        r0 = max(0, roi_box.row0)
        c0 = max(0, roi_box.col0)
        if ref_invariants is None:
            r1 = min(ref_img.shape[0], roi_box.row0 + roi_box.height)
            c1 = min(ref_img.shape[1], roi_box.col0 + roi_box.width)
            ref_img = ref_img[r0:r1, c0:c1]
        r1d = min(def_img.shape[0], roi_box.row0 + roi_box.height)
        c1d = min(def_img.shape[1], roi_box.col0 + roi_box.width)
        def_img = def_img[r0:r1d, c0:c1d]
        if robot_poly is not None:
            robot_poly = robot_poly.shifted(-r0, -c0)

    # Flatfield normalization (ref side hoisted out of the run loop when possible)
    _report(10, "Flatfield")
    if ref_invariants is not None:
        ref_img = ref_invariants.ref_img
        ref_ff = ref_invariants.ref_ff
    else:
        ref_ff = flatfield_normalize(ref_img, sigma=sigma)
    def_ff = flatfield_normalize(def_img, sigma=sigma, bg_src=ref_img)

    # Find carriers in reference
    _report(30, "Carriers")
    carriers0 = calculate_carriers(ref_ff - ref_ff.mean())

    # Scale normalization: rescale ref to match def's carrier period when they differ.
    # For scale < 1 the function returns a *smaller* ref and the crop coordinates
    # of def that correspond to it — we must crop def to avoid carrier leakage in
    # the FCD border region (unpaired def carrier → checkerboard through integration).
    if getattr(project.process, "auto_scale_ref", True):
        from openfcd.core.registration import scale_normalize_reference
        ref_ff, carriers0, _scale, _valid_crop = scale_normalize_reference(ref_ff, def_ff, carriers0)
        if _valid_crop is not None:
            r0v, c0v, hv, wv = _valid_crop
            def_ff = def_ff[r0v:r0v + hv, c0v:c0v + wv]
            if robot_poly is not None:
                robot_poly = robot_poly.shifted(-r0v, -c0v)

    filament_mask = detect_filament_occluders(def_ff, carriers0)

    # Build occlusion mask
    _report(45, "Inpaint")
    if robot_poly is not None:
        # Manual annotation mask
        occlusion_mask = polygon_mask(ref_ff.shape, robot_poly, dilate_px=robot_pad_px)
        occlusion_mask |= filament_mask
        # Inpaint carrier region
        syn = synthesize_from_carriers(ref_ff - ref_ff.mean(), carriers0)
        ref_clean = inpaint_fft(ref_ff, occlusion_mask, syn + ref_ff.mean())
        carriers = calculate_carriers(ref_clean - ref_clean.mean())
        syn2 = synthesize_from_carriers(ref_clean - ref_clean.mean(), carriers)
        def_clean = inpaint_fft(def_ff, occlusion_mask, syn2 + ref_clean.mean())
    else:
        # Auto-detect occlusion (may fail if scene is clean)
        scout = auto_mask(def_ff, carriers0, threshold_ratio=0.2, dilate_px=4)
        auto_poly = find_oriented_polygon(scout, edge_margin=20)
        if auto_poly is None:
            auto_box = find_largest_interior_blob(scout, edge_margin=20)
            if auto_box is not None:
                from openfcd.core.mask import Polygon
                auto_poly = Polygon.from_box(auto_box)
        if auto_poly is not None:
            occlusion_mask = polygon_mask(ref_ff.shape, auto_poly, dilate_px=robot_pad_px)
            occlusion_mask |= filament_mask
            syn = synthesize_from_carriers(ref_ff - ref_ff.mean(), carriers0)
            ref_clean = inpaint_fft(ref_ff, occlusion_mask, syn + ref_ff.mean())
            carriers = calculate_carriers(ref_clean - ref_clean.mean())
            syn2 = synthesize_from_carriers(ref_clean - ref_clean.mean(), carriers)
            def_clean = inpaint_fft(def_ff, occlusion_mask, syn2 + ref_clean.mean())
        else:
            # No occlusion detected — process as clean field
            occlusion_mask = filament_mask.copy()
            ref_clean = ref_ff
            def_clean = def_ff
            carriers = carriers0

    # Edge conditioning: taper OR Moisan periodic decomposition (mutually exclusive).
    # Combining them reintroduces a periodic→zero boundary jump that causes an
    # artifact ring, so exactly one method is applied.
    #
    #  taper_alpha > 0  →  cosine taper (traditional; zeros out edges)
    #  taper_alpha = 0  →  Moisan (2011) periodic+smooth decomposition:
    #                       creates a truly periodic image so FFT integration
    #                       has no boundary artefacts, preserving edge content.
    _report(65, "Taper/PeriodicBC")
    if taper_alpha > 0:
        win = cosine_taper(ref_clean.shape, alpha=taper_alpha)
        ref_mean = ref_clean.mean()
        def_mean = def_clean.mean()
        ref_clean = (ref_clean - ref_mean) * win + ref_mean
        def_clean = (def_clean - def_mean) * win + def_mean
    else:
        from openfcd.core.fcd import periodic_smooth_decompose
        ref_clean, _ = periodic_smooth_decompose(ref_clean)
        def_clean, _ = periodic_smooth_decompose(def_clean)

    # Self-test path: reference against itself should produce a zero field.
    # Returning zeros explicitly is more honest than surfacing algorithmic
    # carrier/integration residuals as if they were physical waves.
    if same_input:
        px_per_mm = carriers_pixel_per_mm(carriers, geom.pattern_period_mm)
        eta_mm = np.zeros(ref_clean.shape, dtype=np.float64)
        eta_mm[occlusion_mask] = np.nan
        if edge_mm > 0:
            em_px = int(edge_mm * px_per_mm)
            eta_mm[edge_margin_mask(eta_mm.shape, em_px)] = np.nan
        _report(100, "Done")
        return eta_mm

    # FCD: demodulate and integrate
    _report(70, "FCD demodulate")
    raw_eta = fcd_height(def_clean - ref_clean.mean(), carriers, unwrap=False)

    # Apply occlusion mask (NaN over masked region)
    raw_eta[occlusion_mask] = np.nan

    # Detrend
    _report(80, "Detrend")
    if project.process.detrend == "plane":
        valid = np.isfinite(raw_eta) & ~edge_margin_mask(raw_eta.shape, 20)
        if valid.sum() > 100:
            raw_eta = detrend_plane(raw_eta, valid=valid)

    # Calibrate: pixel → mm
    _report(85, "Calibrate")
    px_per_mm = carriers_pixel_per_mm(carriers, geom.pattern_period_mm)
    eta_mm = raw_eta / (geom.alpha * geom.h_p_eff_mm * px_per_mm ** 2)

    # Mask edges
    if edge_mm > 0:
        em_px = int(edge_mm * px_per_mm)
        eta_mm[edge_margin_mask(eta_mm.shape, em_px)] = np.nan

    if not fast_preview:
        _report(90, "Filament suppress")
        eta_mm = suppress_nonphysical_eta_filaments(
            eta_mm,
            low_signal_mask=occlusion_mask,
        )

    # Optional spatial high-pass: remove large-scale drift (non-physical waves
    # from ref/def mismatch) while preserving short-wavelength surface waves.
    hp_sigma = float(getattr(project.process, "highpass_sigma_px", 0.0)) * _downsample_scale
    if hp_sigma > 0.0:
        _report(93, "Highpass")
        from scipy.ndimage import gaussian_filter
        finite = np.isfinite(eta_mm)
        filled = np.where(finite, eta_mm, 0.0)
        # Normalize by the filtered validity mask so NaN gaps don't bleed into the trend
        weight = gaussian_filter(finite.astype(np.float32), sigma=hp_sigma)
        trend = gaussian_filter(filled, sigma=hp_sigma) / np.maximum(weight, 1e-6)
        eta_mm = np.where(finite, eta_mm - trend, np.nan)
        # Near the valid-region boundary the Gaussian has truncated support and
        # the trend estimate is unreliable, producing a coloured fringe.  NaN
        # out a margin wider than the kernel reach (~1.5·σ) to hide it.
        hp_margin = int(round(1.5 * hp_sigma))
        if hp_margin > 0:
            eta_mm[edge_margin_mask(eta_mm.shape, hp_margin)] = np.nan

    # Upsample back to ROI-resolution (undo the fast-preview downsample).
    # _original_shape is the full image shape but eta_mm is at cropped+downsampled
    # resolution; using (h0/eta_h, w0/eta_w) as zoom factors would wrongly stretch
    # the ROI content to fill the full image.  Use the exact downscale factor instead.
    if _original_shape is not None:
        _report(97, "Upsample")
        from scipy.ndimage import zoom as _zoom
        valid = np.isfinite(eta_mm)
        filled = np.where(valid, eta_mm, 0.0)
        up = round(1.0 / _downsample_scale)  # invert the fast-preview downsample
        eta_up = _zoom(filled, up, order=1)
        valid_up = _zoom(valid.astype(np.float32), up, order=0) > 0.5
        eta_mm = np.where(valid_up, eta_up, np.nan)

    _report(100, "Done")
    return eta_mm


def _filter_frames(frames: list[Path], filt: str) -> list[Path]:
    """Apply a simple frame range or glob filter."""
    if ":" in filt:
        parts = filt.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else len(frames)
        return frames[start:end]
    return [f for f in frames if fnmatch(f.name, filt)]


def _resolve_default_workers() -> int:
    """Best-effort default worker count.

    Honors OPENFCD_FORCE_SERIAL=1 escape hatch. On Darwin uses physical P-cores
    (`sysctl hw.perflevel0.physicalcpu`) to avoid scheduling onto E-cores. Other
    platforms fall back to ``cpu_count - 1``.
    """
    import os
    if os.environ.get("OPENFCD_FORCE_SERIAL", "") == "1":
        return 1
    import platform
    if platform.system() == "Darwin":
        try:
            import subprocess
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                text=True, timeout=1.0,
            ).strip()
            n = int(out)
            if n > 0:
                return n
        except Exception:
            pass
    return max(1, (os.cpu_count() or 2) - 1)


def _quick_fingerprint(project: ProjectModel) -> str:
    """Quick hash of geometry + process config for STALE detection."""
    import hashlib
    import json
    payload = (
        json.dumps(project.geometry.model_dump(), sort_keys=True)
        + json.dumps(project.process.model_dump(), sort_keys=True)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# run command implementation (called from __init__.py)
# ---------------------------------------------------------------------------

def run_cmd(
    project_path: Path = typer.Argument(..., help="Path to .ofcd project directory"),
    workers: int = typer.Option(-1, help="Parallel workers (-1=auto)"),
    frames: str | None = typer.Option(
        None, help="Frame range filter (e.g. '0:100' or 'Img200*.jpg')"
    ),
    run_id: str | None = typer.Option(
        None, help="Run ID (default: run-YYYYMMDD-HHMMSS)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON lines"),
) -> None:
    """Run the FCD pipeline on a project. Writes runs/{id}/results.h5."""
    store = FileSessionStore.open(project_path)
    project: ProjectModel = store.project

    if run_id is None:
        run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    run_dir = store._dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    result_store = HDF5ResultStore.open(run_dir / "results.h5", mode="w")

    ctx: dict = {
        "project": project,
        "project_dir": store._dir,
        "frames_dir": Path(project.data.frames_dir),
        "pattern": project.data.pattern,
        "frames_filter": frames,
        "run_id": run_id,
        "result_store": result_store,
        "batches_processed": [],
        "frame_count": 0,
        "frame_paths": [],
        "workers": workers,
    }

    stages: list[Stage] = [
        PreprocessStage(),
        ComputeStage(),
        PostprocessStage(),
    ]
    runner = PipelineRunner(stages, workers=workers if workers > 0 else -1)

    cancel_token = CancelToken()
    exit_code: int = 0

    def _sigint_handler(signum: int, frame: object) -> None:
        cancel_token.cancel()

    original_handler = signal.signal(signal.SIGINT, _sigint_handler)

    try:
        for event in runner.iter(ctx, cancel_token):
            if json_output:
                typer.echo(stage_event_to_json(event))
            else:
                typer.echo(format_progress(event))

            if event.kind == "error":
                exit_code = 2
                break
            if event.kind == "cancel":
                exit_code = 1

    except CancelledError:
        exit_code = 1
    except Exception as exc:
        typer.echo(f"Pipeline error: {exc}", err=True)
        exit_code = 2
    finally:
        signal.signal(signal.SIGINT, original_handler)

        if exit_code == 1:
            cancel_token.wait(3.0)

        manifest = {
            "project_name": project.name,
            "run_id": run_id,
            "frames_filter": frames,
            "workers": workers,
            "status": "success" if exit_code == 0
            else ("cancelled" if exit_code == 1 else "error"),
        }
        store.record_run(run_id, manifest)

        # Close stores if still open
        if ctx.get("result_store") is not None:
            try:
                result_store.close()
            except Exception:
                pass
        store.close()

    raise typer.Exit(code=exit_code)
