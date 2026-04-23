"""`openfcd replay` — replay a previous run's results."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from openfcd.core.figures import FIGURE_IDS
from openfcd.io.result import HDF5ResultStore
from openfcd.io.store import FileSessionStore


def replay_cmd(
    project_path: Path = typer.Argument(..., help="Path to .ofcd project directory"),
    run: str | None = typer.Option(None, help="Run ID (default: latest)"),
    figure: str | None = typer.Option(None, help="Figure ID to render (eta_heatmap, etc.)"),
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
            typer.echo(f"[replay] Rendering figure '{figure}' for run '{run_id}'...")
            typer.echo("[replay] Figure rendering not yet available (no renderers registered).")
            raise typer.Exit(code=1)
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