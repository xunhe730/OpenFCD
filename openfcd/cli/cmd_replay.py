"""`openfcd replay` — replay a previous run's results."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from openfcd.core.figures import FIGURE_IDS
from openfcd.io.result import HDF5ResultStore
from openfcd.io.store import FileSessionStore


def _profile_replay_context(store, result_store, requested_run: str):
    """Build deterministic ProfileComposite context for CLI replay."""
    from openfcd.core.profile_composite import (
        build_profile_composite_context,
        resolve_spatial_calibration,
        resolve_body_polygon,
    )
    from openfcd.io.scene import SceneType

    batch = "default"
    frames = set(result_store.list_frames(batch))
    run_manifest = next((r for r in store.list_runs() if r.get("run_id") == requested_run), None)
    for scene in store.scenes:
        if scene.type != SceneType.PROFILE:
            continue
        if scene.run_id is not None and scene.run_id != requested_run:
            continue
        lines = scene.profile_lines or {}
        eligible = [idx for idx in scene.frame_indices if idx in lines and idx in frames]
        if not eligible:
            continue
        requested_frame = scene.viz_params.get("frame_idx") if scene.viz_params else None
        try:
            requested_frame = int(requested_frame) if requested_frame is not None else None
        except (TypeError, ValueError):
            requested_frame = None
        frame_idx = requested_frame if requested_frame in eligible else eligible[0]
        eta = result_store.read_frame(batch, frame_idx)
        try:
            attrs = result_store.read_frame_attrs(batch, frame_idx)
            frame_path = attrs.get("frame_path")
            frame_name = Path(str(frame_path)).name if frame_path else None
        except Exception:
            frame_name = None
        body_polygon, body_source = resolve_body_polygon(store.annotation, frame_name)
        line = lines[frame_idx]
        viz_params = dict(scene.viz_params or {})
        viz_params.pop("px_per_mm", None)
        spatial_calibration = resolve_spatial_calibration(
            result_store,
            batch,
            frame_idx,
            run_manifest,
            store.project,
        )
        return build_profile_composite_context(
            eta=eta,
            profile_line=(tuple(line.p0), tuple(line.p1)),
            viz_params=viz_params,
            frame_idx=frame_idx,
            frame_name=frame_name,
            run_id=requested_run,
            batch=batch,
            body_polygon_rc=body_polygon,
            body_source=body_source,
            spatial_calibration=spatial_calibration,
        )
    return build_profile_composite_context(
        eta=None,
        profile_line=None,
        viz_params={},
        run_id=requested_run,
        batch=batch,
        degraded_reason="No compatible Profile scene with a saved line and replayable frame",
    )


def replay_cmd(
    project_path: Path = typer.Argument(..., help="Path to .ofcd project directory"),
    run: str | None = typer.Option(None, help="Run ID (default: latest)"),
    figure: str | None = typer.Option(None, help="Figure ID to render (eta_heatmap, etc.)"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Figure output path"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
) -> None:
    """Replay a previous run's results (no recomputation)."""
    store = FileSessionStore.open(project_path)
    runs = store.list_runs()

    if not runs:
        typer.echo("No runs found.", err=True)
        store.close()
        raise typer.Exit(code=1)

    # Resolve run_id: explicit or latest
    if run is None:
        run_id = runs[-1]["run_id"]
    else:
        run_ids = {r["run_id"] for r in runs}
        if run not in run_ids:
            typer.echo(f"Run '{run}' not found.", err=True)
            store.close()
            raise typer.Exit(code=1)
        run_id = run

    results_path = store._dir / "runs" / run_id / "results.h5"
    if not results_path.exists():
        typer.echo(f"Results file not found: {results_path}", err=True)
        store.close()
        raise typer.Exit(code=1)

    result_store = HDF5ResultStore.open(results_path, mode="r")
    try:
        batches = result_store.list_batches()

        if json_output:
            summary: dict = {
                "run_id": run_id,
                "batches": [],
            }
            for batch in batches:
                batch_info: dict = {
                    "name": batch,
                    "meta": result_store.read_batch_meta(batch),
                    "frames": result_store.list_frames(batch),
                    "status": result_store.batch_status(batch),
                }
                summary["batches"].append(batch_info)
            typer.echo(json.dumps(summary, indent=2))
        elif figure is not None:
            if figure not in FIGURE_IDS:
                typer.echo(
                    f"Unknown figure ID: {figure}. Valid: {sorted(FIGURE_IDS)}",
                    err=True,
                )
                raise typer.Exit(code=1)
            import importlib
            from openfcd.core.figures import RENDERERS

            importlib.import_module("openfcd.gui.renderers")  # registers built-in matplotlib renderers
            renderer = RENDERERS.get(figure)
            if renderer is None:
                typer.echo(f"No renderer registered for figure '{figure}'.", err=True)
                raise typer.Exit(code=1)
            if not batches:
                typer.echo("No batches found in results file.", err=True)
                raise typer.Exit(code=1)
            out_path = output or (store._dir / "exports" / f"{run_id}_{figure}.png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if figure == "wavelength_profile":
                viz = _profile_replay_context(store, result_store, run_id)
                fig = renderer.render(result_store, "default", viz)
            else:
                from types import SimpleNamespace
                fig = renderer.render(result_store, batches[0], SimpleNamespace())
            fig.savefig(out_path, bbox_inches="tight")
            typer.echo(f"[replay] Wrote {out_path}")
        else:
            typer.echo(f"Run: {run_id}")
            typer.echo(f"Batches: {len(batches)}")
            for batch in batches:
                status = result_store.batch_status(batch)
                stale = "STALE" if store.is_stale(run_id) else "OK"
                typer.echo(f"  {batch}: {status['done']}/{status['total']} frames [{stale}]")
    finally:
        result_store.close()
        store.close()
