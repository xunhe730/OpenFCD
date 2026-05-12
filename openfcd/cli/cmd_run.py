"""`openfcd run` — run the FCD pipeline on a project."""
from __future__ import annotations

import signal
import time
import json
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class FrameComputation:
    """Single-frame compute output plus the calibration used to create it."""

    eta_mm: np.ndarray
    pixel_per_mm: float
    calibration: dict
    qc_datasets: dict[str, np.ndarray] = field(default_factory=dict)
    qc_attrs: dict[str, float] = field(default_factory=dict)

    @property
    def shape(self):
        return self.eta_mm.shape

    def __array__(self, dtype=None):
        return np.asarray(self.eta_mm, dtype=dtype)

    def __getitem__(self, key):
        return self.eta_mm[key]


def _frame_calibration(pixel_per_mm: float, geom) -> dict:
    return {
        "pixel_per_mm": float(pixel_per_mm),
        "pattern_period_mm": float(geom.pattern_period_mm),
        "checker_cell_mm": float(geom.pattern_period_mm),
        "checker_cell_semantics": "single checker cell side length, not full black-white cycle",
        "alpha": float(geom.alpha),
        "h_p_eff_mm": float(geom.h_p_eff_mm),
        "eta_unit": "mm",
        "spatial_calibration_source": "carrier_detected",
    }


def _calibration_summary(frame_calibrations: list[dict] | None) -> dict:
    valid = [
        c for c in (frame_calibrations or [])
        if c.get("pixel_per_mm") is not None and float(c.get("pixel_per_mm", 0.0)) > 0
    ]
    if not valid:
        return {"calibration_status": "missing"}
    px = np.asarray([float(c["pixel_per_mm"]) for c in valid], dtype=np.float64)
    first = valid[0]
    calibration = {
        "pixel_per_mm_median": float(np.nanmedian(px)),
        "pixel_per_mm_min": float(np.nanmin(px)),
        "pixel_per_mm_max": float(np.nanmax(px)),
        "pattern_period_mm": float(first["pattern_period_mm"]),
        "checker_cell_mm": float(first.get("checker_cell_mm", first["pattern_period_mm"])),
        "checker_cell_semantics": "single checker cell side length, not full black-white cycle",
        "alpha": float(first["alpha"]),
        "h_p_eff_mm": float(first["h_p_eff_mm"]),
        "eta_unit": "mm",
        "spatial_calibration_source": "carrier_detected",
        "n_calibrated_frames": len(valid),
    }
    return {
        "calibration_status": "ok",
        "pixel_per_mm_median": calibration["pixel_per_mm_median"],
        "calibration": calibration,
    }


def _batch_calibration_meta(frame_calibrations: list[dict] | None) -> dict:
    summary = _calibration_summary(frame_calibrations)
    if summary.get("calibration_status") != "ok":
        return {"calibration_status": "missing"}
    calibration = summary["calibration"]
    return {
        "calibration_status": "ok",
        "pixel_per_mm_median": calibration["pixel_per_mm_median"],
        "pixel_per_mm_min": calibration["pixel_per_mm_min"],
        "pixel_per_mm_max": calibration["pixel_per_mm_max"],
        "calibration_json": json.dumps(calibration, sort_keys=True),
    }


def build_run_manifest(
    project: ProjectModel,
    run_id: str,
    frames_filter,
    workers: int,
    status: str,
    ctx: dict,
) -> dict:
    manifest = {
        "project_name": project.name,
        "run_id": run_id,
        "frames_filter": frames_filter,
        "workers": workers,
        "status": status,
    }
    summary = _calibration_summary(ctx.get("frame_calibrations", []))
    manifest["calibration_status"] = summary["calibration_status"]
    if summary.get("calibration_status") == "ok":
        manifest["pixel_per_mm_median"] = summary["pixel_per_mm_median"]
        manifest["calibration"] = summary["calibration"]
    return manifest


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
    # 0) If source is an image file and a sibling .npy of the same stem
    # exists in project_dir (auto-built reference cache), prefer the .npy
    # — it preserves the full float64 mean/median, the .jpg only exists
    # for SimTree thumbnail display.
    if ref_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
        npy_sibling = project_dir / (ref_path.stem + ".npy")
        if npy_sibling.exists():
            return np.load(npy_sibling)
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
        from openfcd.pipeline.compute import load_gray as _lg
        frames = scan_frames(project.data.frames_dir, project.data.pattern)
        if not frames:
            raise FileNotFoundError("No frames found for first_frame reference mode")
        return _lg(frames[0])

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

        # 1. Resolve frame set. GUI runs pass the currently imported/visible
        # frame list explicitly; CLI runs continue to scan project.data.
        override = ctx.get("frame_paths_override")
        if override is not None:
            frames = [Path(p) for p in override]
        else:
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

        # 4. Manual masks are resolved in ComputeStage. Frames without visible
        #    per-frame polygons fall through to per-frame auto-mask.

        # 5. Store frame paths and count
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
    """Process each frame through the FCD pipeline via `compute_frame`."""

    name: str = "compute"

    def run(
        self, ctx: dict, cancel: CancelToken | None = None
    ) -> Iterator[StageEvent]:
        from openfcd.pipeline.aggregate import StreamingMeanAccumulator
        from openfcd.pipeline.frame import (
            FrameInputsSnapshot,
            build_frame_inputs,
            compute_frame,
        )

        run_id: str = ctx.get("run_id", "")
        frame_paths: list[Path] = ctx.get("frame_paths", [])
        frame_count: int = ctx.get("frame_count", 0)
        result_store: HDF5ResultStore | None = ctx.get("result_store")
        project: ProjectModel = ctx["project"]

        disabled = set(ctx.get("disabled_frame_indices", []))
        frame_items = list(enumerate(frame_paths))
        if disabled:
            frame_items = [(i, fp) for i, fp in frame_items if i not in disabled]
            ctx["frame_paths"] = [fp for _, fp in frame_items]
            frame_count = len(frame_items)
            ctx["frame_count"] = frame_count
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

        snapshot = FrameInputsSnapshot.from_run_context(ctx)
        accumulator = StreamingMeanAccumulator()
        ctx["eta_accumulator"] = accumulator
        ctx["frame_calibrations"] = []
        stream_results = _resolve_stream_results(result_store, snapshot.ref_img, frame_count)

        eta_list: list[np.ndarray | None] = [None] * frame_count
        processed = 0
        errors = 0
        t0 = time.time()

        for ordinal, (frame_idx, frame_path) in enumerate(frame_items):
            if cancel and cancel.is_cancelled:
                break
            yield StageEvent(
                kind="progress", stage=self.name, batch="default",
                frame_idx=frame_idx, substage="loading",
                progress=ordinal / max(frame_count, 1),
                total=frame_count, completed=processed,
                metrics={"frame": frame_path.name,
                         "frame_no": f"{ordinal + 1}/{frame_count}"},
                run_id=run_id,
            )
            sub_events: list[tuple[float, str]] = []
            def _cb(pct: int, lbl: str, _i=ordinal) -> None:
                sub_events.append((_i + pct / 100.0, lbl))

            result = None
            err: str | None = None
            try:
                inputs = build_frame_inputs(snapshot, frame_path, fast_preview=True)
                if inputs.def_img.shape != snapshot.ref_shape:
                    err = "shape_mismatch"
                else:
                    result = compute_frame(inputs, progress_cb=_cb, cancel=cancel)
            except CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                err = str(exc)

            for local_prog, label in sub_events:
                yield StageEvent(
                    kind="progress", stage=self.name, batch="default",
                    frame_idx=frame_idx, substage=label,
                    progress=local_prog / max(frame_count, 1),
                    total=frame_count, completed=processed,
                    metrics={"frame": frame_path.name,
                             "frame_no": f"{ordinal + 1}/{frame_count}"},
                    run_id=run_id,
                )

            if result is not None:
                _write_frame_result(result_store, project, frame_idx, frame_path, result, ctx)
                if not stream_results:
                    eta_list[ordinal] = result.eta_mm
                accumulator.update(result.eta_mm)
                processed += 1
            else:
                errors += 1
                if result_store is not None:
                    result_store.write_frame(
                        "default", frame_idx, np.zeros((1, 1), dtype=np.float64),
                        {"status": "error", "message": err or "",
                         "frame_path": str(frame_path.name)},
                    )
            yield _frame_progress_event(self.name, run_id, frame_idx, frame_path.name,
                                        processed, errors, frame_count, t0)

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


def _resolve_stream_results(
    result_store: HDF5ResultStore | None, ref_img: np.ndarray, frame_count: int
) -> bool:
    import os as _os
    env = _os.environ.get("OPENFCD_STREAM_RESULTS", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return result_store is not None
    if env in ("0", "false", "no", "off"):
        return False
    estimated = ref_img.nbytes * frame_count
    return result_store is not None and (frame_count > 50 or estimated > 400_000_000)


def _write_frame_result(
    result_store: HDF5ResultStore | None,
    project: ProjectModel,
    frame_idx: int,
    frame_path: Path,
    result,
    ctx: dict,
) -> None:
    if result_store is None:
        return
    attrs: dict = {"status": "ok", "frame_path": str(frame_path.name)}
    calibration = result.calibration
    if calibration:
        attrs.update(calibration)
        if "pattern_period_mm" in calibration and "checker_cell_mm" not in attrs:
            attrs["checker_cell_mm"] = float(calibration["pattern_period_mm"])
            attrs["checker_cell_semantics"] = (
                "single checker cell side length, not full black-white cycle"
            )
    attrs.update(result.qc_attrs or {})
    try:
        from openfcd.pipeline.qc import qc_verdict, thresholds_from_config
        verdict = qc_verdict(attrs, thresholds_from_config(getattr(project, "qc", None)))
        attrs["qc_verdict"] = verdict["verdict"]
        attrs["qc_reasons"] = "; ".join(verdict["reasons"])
    except Exception as exc:  # noqa: BLE001
        attrs["qc_verdict"] = "WARN"
        attrs["qc_reasons"] = f"qc verdict unavailable: {exc}"
    result_store.write_frame(
        "default", frame_idx, result.eta_mm, attrs, qc_datasets=result.qc_datasets,
    )
    if calibration:
        ctx["frame_calibrations"].append(dict(calibration))


def _frame_progress_event(
    stage_name: str, run_id: str, frame_idx: int, frame_name: str,
    processed: int, errors: int, frame_count: int, t0: float,
) -> StageEvent:
    prog = (processed + errors) / max(frame_count, 1)
    done = processed + errors
    elapsed = time.time() - t0
    eta_sec = (elapsed / done) * (frame_count - done) if done > 0 else 0.0
    return StageEvent(
        kind="progress", stage=stage_name, batch="default",
        frame_idx=frame_idx, substage="processing_frame", progress=prog,
        total=frame_count, completed=processed,
        metrics={"frame": frame_name, "errors": errors,
                 "eta_seconds": round(eta_sec, 1)},
        run_id=run_id,
    )


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

        accumulator = ctx.get("eta_accumulator")
        precomputed_mean = accumulator.finalize() if accumulator is not None else None
        precomputed_count = accumulator.n_valid_pixels if accumulator is not None else 0
        if result_store is not None:
            summary = _summarize_results(
                eta_list,
                result_store,
                precomputed_mean=precomputed_mean,
                precomputed_count=precomputed_count,
            )
            if summary is not None:
                eta_mean, eta_median, eta_rms, n_valid = summary

                yield StageEvent(
                    kind="progress", stage=self.name, batch="default",
                    frame_idx=None, substage="writing_summaries",
                    progress=0.95, total=total_frames, completed=total_frames,
                    metrics={"n_valid_frames": n_valid},
                    run_id=run_id,
                )

                meta = {
                    "n_frames": n_valid,
                    "n_errors": ctx.get("compute_errors", 0),
                    "project_name": project.name,
                    "config_fingerprint": _quick_fingerprint(project),
                }
                meta.update(_batch_calibration_meta(ctx.get("frame_calibrations", [])))
                result_store.write_batch(
                    "default",
                    data={
                        "eta_mean": eta_mean,
                        "eta_median": eta_median,
                        "eta_rms": eta_rms,
                    },
                    meta=meta,
                )

        # Wave statistics (no-op if annotation.wave_stats is None)
        if result_store is not None:
            try:
                self._inject_wave_stats(result_store, ctx, run_id)
                yield StageEvent(
                    kind="progress", stage=self.name, batch=None, frame_idx=None,
                    substage="wave_stats_done", progress=0.98,
                    total=total_frames, completed=total_frames,
                    metrics={"wave_stats": "ok"}, run_id=run_id,
                )
            except Exception as exc:  # noqa: BLE001
                yield StageEvent(
                    kind="progress", stage=self.name, batch=None, frame_idx=None,
                    substage="wave_stats_warning", progress=0.98,
                    total=total_frames, completed=total_frames,
                    metrics={"wave_stats_warning": str(exc), "level": "WARNING"},
                    run_id=run_id,
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

    def _inject_wave_stats(
        self,
        store: HDF5ResultStore,
        ctx: dict,
        run_id: str = "",
    ) -> None:
        """Compute wave statistics and write them into the open result store.

        No-op if annotation.wave_stats or annotation.profile_line is None.
        Reads η frames directly from the open h5 handle (same process).
        """
        from openfcd.pipeline.wave_stats_pipeline import compute_and_write_wave_stats

        annotation = ctx.get("annotation")
        if annotation is None:
            return
        if annotation.wave_stats is None or annotation.profile_line is None:
            return

        batches: list[str] = ctx.get("batches_processed", ["default"])
        project: ProjectModel = ctx["project"]

        # ── px_per_mm from batch attrs already written by write_batch ──────
        px_per_mm_by_batch: dict[str, float] = {}
        for b in batches:
            try:
                meta = store.read_batch_meta(b)
                px_per_mm_by_batch[b] = float(meta.get("pixel_per_mm_median", 1.0))
            except Exception:  # noqa: BLE001
                px_per_mm_by_batch[b] = 1.0

        # ── Body polygons from annotation.frame_polygons ────────────────────
        # frame_polygons is keyed by frame ID (e.g. "Img0042"), NOT by batch
        # name. Iterate frames recorded in the h5 and look them up directly.
        # Fallback to annotation.polygons[0] (static, batch-wide) if a frame
        # has no per-frame entry; finally None if neither exists.
        body_polygons_by_batch_frame: dict[str, dict[str, np.ndarray | None]] = {}
        for b in batches:
            per_frame: dict[str, np.ndarray | None] = {}
            try:
                frame_ids = store.list_frames(b)
            except Exception:  # noqa: BLE001
                frame_ids = []
            for fid in frame_ids:
                fid_str = str(fid)
                poly_rc: np.ndarray | None = None
                poly_list = annotation.frame_polygons.get(fid_str, [])
                for p in poly_list:
                    if p.label == "body" and p.vertex_count >= 3:
                        poly_rc = p.as_rc_array()
                        break
                if poly_rc is None and annotation.polygons:
                    poly_rc = annotation.polygons[0].as_rc_array()
                per_frame[fid_str] = poly_rc
            body_polygons_by_batch_frame[b] = per_frame

        # ── η loader: read directly from the open h5 handle ────────────────
        def eta_loader(batch: str, frame_id_str: str) -> np.ndarray:
            return store._file[f"batches/{batch}/frames/{frame_id_str}"][:]

        compute_and_write_wave_stats(
            annotation=annotation,
            store=store,
            batches=batches,
            px_per_mm_by_batch=px_per_mm_by_batch,
            body_polygons_by_batch_frame=body_polygons_by_batch_frame,
            eta_loader=eta_loader,
        )

    def dry_run(self, ctx: dict) -> list[str]:
        return ["compute_summaries", "wave_stats", "write_metadata", "close_store"]


# ---------------------------------------------------------------------------


def _filter_frames(frames: list[Path], filt: str) -> list[Path]:
    """Apply a simple frame range or glob filter."""
    if ":" in filt:
        parts = filt.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else len(frames)
        return frames[start:end]
    return [f for f in frames if fnmatch(f.name, filt)]


def _summarize_results(
    eta_list: list,
    result_store: HDF5ResultStore,
    precomputed_mean=None,
    precomputed_count: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Compute (eta_mean, eta_median, eta_rms, n_valid) over a batch.

    Two paths:
      * In-memory: use the list as-is, compute median exactly via np.stack.
      * Streaming (eta_list empty/None): read frames one-by-one from HDF5 and
        accumulate per-pixel sum / sum-of-squares / count. This stays at
        O(target_shape) memory regardless of frame count. Median is set to
        eta_mean in this mode (true streaming median would need a second pass
        or chunked quantile, both expensive — and downstream rarely uses it).

    When precomputed_mean is provided (from ComputeStage incremental accumulator),
    large-image streaming mode returns it directly without re-reading HDF5.
    """
    # Use precomputed mean from ComputeStage if available — avoids re-reading HDF5.
    if precomputed_mean is not None:
        eta_mean = np.asarray(precomputed_mean, dtype=np.float64)
        in_memory = [e for e in eta_list if e is not None]
        if not in_memory:
            # Large image streaming mode: return precomputed mean as mean, rms, median
            return eta_mean, eta_mean.copy(), eta_mean.copy(), precomputed_count
        # else: in_memory path runs normally below for proper rms/median

    in_memory = [e for e in eta_list if e is not None]
    if in_memory:
        target_shape = in_memory[0].shape
        valid_etas = [e for e in in_memory if e.shape == target_shape]
        if not valid_etas:
            return None
        stack = np.stack(valid_etas, axis=0)
        eta_mean = np.nanmean(stack, axis=0)
        eta_median = np.nanmedian(stack, axis=0)
        eta_rms = np.sqrt(np.nanmean(stack ** 2, axis=0))
        return eta_mean, eta_median, eta_rms, len(valid_etas)

    # Streaming path.
    try:
        frame_ids = result_store.list_frames("default")
    except Exception:
        return None

    target_shape = None
    sum_arr: np.ndarray | None = None
    sumsq_arr: np.ndarray | None = None
    count_arr: np.ndarray | None = None
    n_valid = 0
    for fid in frame_ids:
        try:
            arr = result_store.read_frame("default", fid)
        except Exception:
            continue
        if arr.ndim != 2 or arr.size <= 1:
            continue
        if target_shape is None:
            target_shape = arr.shape
            sum_arr = np.zeros(target_shape, dtype=np.float64)
            sumsq_arr = np.zeros(target_shape, dtype=np.float64)
            count_arr = np.zeros(target_shape, dtype=np.int64)
        if arr.shape != target_shape:
            continue
        a = np.asarray(arr, dtype=np.float64)
        valid = ~np.isnan(a)
        sum_arr[valid] += a[valid]
        sumsq_arr[valid] += a[valid] ** 2
        count_arr[valid] += 1
        n_valid += 1

    if not n_valid or sum_arr is None:
        return None

    safe_count = np.maximum(count_arr, 1)
    eta_mean = sum_arr / safe_count
    eta_rms = np.sqrt(sumsq_arr / safe_count)
    eta_mean[count_arr == 0] = np.nan
    eta_rms[count_arr == 0] = np.nan
    # Median omitted in streaming mode (see docstring); reuse mean so downstream
    # consumers always see a valid array of the right shape.
    return eta_mean, eta_mean.copy(), eta_rms, n_valid


def _resolve_blas_threads(n_workers: int) -> int:
    """BLAS threads per worker for the parallel ComputeStage.

    Default (unset / "auto"): max(1, ncores // n_workers) — each worker gets a
    fair share of cores, matching the throughput of single-frame compute × N.

    OPENFCD_BLAS_THREADS:
      - unset / "auto"  → max(1, ncores // n_workers)  [default]
      - "1"             → 1 (deterministic; bit-equal serial↔parallel)
      - "<int>"         → that integer (clamped >=1)
    """
    import os
    raw = os.environ.get("OPENFCD_BLAS_THREADS", "auto").strip().lower()
    if raw == "auto" or not raw:
        ncores = os.cpu_count() or 1
        return max(1, ncores // max(1, n_workers))
    if raw == "1":
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return max(1, (os.cpu_count() or 1) // max(1, n_workers))


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
        "annotation": store.annotation,
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

        if cancel_token.is_cancelled:
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

        manifest = build_run_manifest(
            project,
            run_id,
            frames,
            workers,
            "success" if exit_code == 0 else ("cancelled" if exit_code == 1 else "error"),
            ctx,
        )
        store.record_run(run_id, manifest)

        # Close stores if still open
        if ctx.get("result_store") is not None:
            try:
                result_store.close()
            except Exception:
                pass
        store.close()

    raise typer.Exit(code=exit_code)
