"""`openfcd postprocess` — recompute wave_stats without re-running ComputeStage.

Flow:
  1. Load .ofcd project and annotation.
  2. Resolve target run (default: latest).
  3. Write a tmp copy of results.h5, skipping batches/*/wave_stats/.
  4. Open tmp with HDF5ResultStore.open_existing_for_append().
  5. Call compute_and_write_wave_stats (shared with PostprocessStage).
  6. store.close().
  7. os.replace(tmp, results.h5) — atomic rename.

If this command is interrupted before step 7, a ``results.h5.tmp`` file may
remain in the run directory.  Delete it manually to recover; the original
``results.h5`` is never modified in-place.
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import typer

from openfcd.io.result import HDF5ResultStore
from openfcd.io.store import FileSessionStore
from openfcd.pipeline.wave_stats_pipeline import compute_and_write_wave_stats

postprocess_app = typer.Typer(
    help="Recompute wave_stats for an existing run without re-running compute."
)


# ---------------------------------------------------------------------------
# Public helper (also unit-tested directly)
# ---------------------------------------------------------------------------

def _copy_skipping_wave_stats(src: h5py.File, dst: h5py.File) -> None:
    """Copy all content from *src* into *dst*, skipping ``wave_stats`` sub-groups.

    For every batch under ``batches/``, all sub-keys **except** ``wave_stats``
    are copied verbatim.  All group and file-level attrs are preserved.
    Every dict/group traversal uses ``sorted(keys)`` for deterministic output.

    Parameters
    ----------
    src:
        Source h5py.File opened for reading.
    dst:
        Destination h5py.File opened for writing (must be empty).
    """
    # Top-level file attrs
    for k in sorted(src.attrs.keys()):
        dst.attrs[k] = src.attrs[k]

    for top_key in sorted(src.keys()):
        if top_key != "batches":
            src.copy(top_key, dst)
            continue

        # ``batches`` needs fine-grained handling to skip wave_stats sub-groups
        dst_batches = dst.create_group("batches")
        for k in sorted(src["batches"].attrs.keys()):
            dst_batches.attrs[k] = src["batches"].attrs[k]

        for batch_name in sorted(src["batches"].keys()):
            src_b = src["batches"][batch_name]
            dst_b = dst_batches.create_group(batch_name)
            for k in sorted(src_b.attrs.keys()):
                dst_b.attrs[k] = src_b.attrs[k]
            for sub_name in sorted(src_b.keys()):
                if sub_name == "wave_stats":
                    continue  # skip — will be recomputed
                src_b.copy(sub_name, dst_b)


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

@postprocess_app.callback(invoke_without_command=True)
def postprocess_cmd(
    ofcd_path: Path = typer.Argument(..., help="Path to .ofcd project directory"),
    run_id: str | None = typer.Option(None, "--run", help="Run id (default: latest)"),
) -> None:
    """Recompute wave_stats for an existing run without re-running ComputeStage.

    Uses copy-and-rewrite + os.replace: the original results.h5 is never
    opened for in-place writing.
    """
    # ── Load project + annotation + runs (single open, single close) ────────
    session = FileSessionStore.open(ofcd_path, read_only=True)
    try:
        annotation = session.annotation
        runs = session.list_runs()
    finally:
        session.close()

    if annotation.wave_stats is None:
        typer.echo(
            "No wave_stats config found in annotation; nothing to postprocess."
        )
        raise typer.Exit(0)

    # ── Resolve run ──────────────────────────────────────────────────────────
    if not runs:
        typer.echo("No runs found in project.", err=True)
        raise typer.Exit(1)

    if run_id is None:
        target_run = runs[-1]["run_id"]
    else:
        available = [r["run_id"] for r in runs]
        if run_id not in available:
            typer.echo(
                f"Run '{run_id}' not found. Available: {available}", err=True
            )
            raise typer.Exit(1)
        target_run = run_id

    h5_path = ofcd_path / "runs" / target_run / "results.h5"
    if not h5_path.exists():
        typer.echo(f"results.h5 not found at: {h5_path}", err=True)
        raise typer.Exit(1)

    tmp_path = h5_path.with_name(h5_path.name + ".tmp")

    typer.echo(f"Postprocessing run '{target_run}' …")

    # ── Step 1: copy skipping wave_stats → tmp ───────────────────────────────
    with h5py.File(h5_path, "r") as src, h5py.File(tmp_path, "w") as dst:
        _copy_skipping_wave_stats(src, dst)

    # ── Steps 2–3: open tmp for append, compute, write ───────────────────────
    store = HDF5ResultStore.open_existing_for_append(tmp_path)
    try:
        batches = store.list_batches()

        px_per_mm_by_batch: dict[str, float] = {}
        for b in batches:
            meta = store.read_batch_meta(b)
            px_per_mm_by_batch[b] = float(meta.get("pixel_per_mm_median", 1.0))

        # Build per-batch/frame body polygon map from annotation
        body_polygons_by_batch_frame: dict[str, dict[str, np.ndarray | None]] = {}
        for b in batches:
            frame_poly_map: dict[str, np.ndarray | None] = {}
            for fid in store.list_frames(b):
                fid_str = str(fid)
                if (
                    fid_str in annotation.frame_polygons
                    and annotation.frame_polygons[fid_str]
                ):
                    frame_poly_map[fid_str] = (
                        annotation.frame_polygons[fid_str][0].as_rc_array()
                    )
                elif annotation.polygons:
                    frame_poly_map[fid_str] = annotation.polygons[0].as_rc_array()
                else:
                    frame_poly_map[fid_str] = None
            body_polygons_by_batch_frame[b] = frame_poly_map

        def _eta_loader(batch: str, fid_str: str) -> np.ndarray:
            return store._file[f"batches/{batch}/frames/{fid_str}"][:]  # type: ignore[index]

        ok = compute_and_write_wave_stats(
            annotation=annotation,
            store=store,
            batches=batches,
            px_per_mm_by_batch=px_per_mm_by_batch,
            body_polygons_by_batch_frame=body_polygons_by_batch_frame,
            eta_loader=_eta_loader,
        )
        if not ok:
            typer.echo("wave_stats computation was a no-op (no profile_line?).")
    finally:
        store.close()

    # ── Step 4: atomic replace ───────────────────────────────────────────────
    os.replace(tmp_path, h5_path)
    typer.echo(f"Done — results.h5 updated for run '{target_run}'.")
