"""`openfcd project` — project management subcommand group."""
from __future__ import annotations

from pathlib import Path

import typer

from openfcd.io.store import FileSessionStore


def project_info_cmd(
    project_path: Path = typer.Argument(...),
) -> None:
    """Print project.yaml summary."""
    store = FileSessionStore.open(project_path)
    project = store.project
    typer.echo(f"Name: {project.name}")
    typer.echo(f"Created: {project.created}")
    typer.echo("Geometry:")
    typer.echo(f"  Checker cell side: {project.geometry.checker_cell_mm} mm")
    typer.echo(f"  Optical stack: {project.geometry.optical_stack.preset}")
    typer.echo("Data:")
    typer.echo(f"  Frames dir: {project.data.frames_dir}")
    typer.echo(f"  Pattern: {project.data.pattern}")
    if project.data.time_step_ms is not None:
        typer.echo(f"  Time step: {project.data.time_step_ms} ms")
    typer.echo("Process:")
    typer.echo(f"  Flatfield sigma: {project.process.flatfield_sigma}")
    typer.echo(f"  Detrend: {project.process.detrend}")
    store.close()


def project_runs_cmd(
    project_path: Path = typer.Argument(...),
) -> None:
    """List all runs with STALE status."""
    store = FileSessionStore.open(project_path)
    runs = store.list_runs()

    if not runs:
        typer.echo("No runs found.")
        store.close()
        return

    for run_info in runs:
        run_id = run_info["run_id"]
        stale = "STALE" if store.is_stale(run_id) else "OK"
        fingerprint = run_info.get("config_fingerprint", "unknown")[:12]
        typer.echo(f"  {run_id} [{stale}] fingerprint={fingerprint}...")

    store.close()


def project_fingerprint_cmd(
    project_path: Path = typer.Argument(...),
) -> None:
    """Print current config_fingerprint."""
    store = FileSessionStore.open(project_path)
    fp = store.config_fingerprint()
    typer.echo(fp)
    store.close()
