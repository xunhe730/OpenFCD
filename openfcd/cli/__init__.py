"""OpenFCD CLI entry point."""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(name="openfcd", help="Fast Checkerboard Demodulation pipeline", no_args_is_help=True)


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

@app.command()
def version() -> None:
    """Print version."""
    import openfcd

    typer.echo(openfcd.__version__)


# ---------------------------------------------------------------------------
# New
# ---------------------------------------------------------------------------

@app.command()
def new(
    name: str = typer.Argument(..., help="Project name"),
    path: Path = typer.Argument(..., help="Directory path for new .ofcd project"),
) -> None:
    """Create a new .ofcd project directory."""
    from openfcd.io.store import FileSessionStore

    FileSessionStore.new(path, name)
    typer.echo(f"Created project '{name}' at {path}")


# ---------------------------------------------------------------------------
# Open
# ---------------------------------------------------------------------------

@app.command()
def open(
    project_path: Path = typer.Argument(..., help="Path to .ofcd project directory"),
) -> None:
    """Load and print project summary."""
    from openfcd.io.store import FileSessionStore

    store = FileSessionStore.open(project_path)
    project = store.project
    typer.echo(f"Name: {project.name}")
    typer.echo(f"Created: {project.created}")
    typer.echo(f"Format version: {project.format_version}")
    typer.echo(f"Geometry preset: {project.geometry.optical_stack.preset}")
    typer.echo(f"Frames dir: {project.data.frames_dir}")
    typer.echo(f"Pattern: {project.data.pattern}")
    store.close()


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

@app.command()
def validate(
    project_path: Path = typer.Argument(..., help="Path to .ofcd project directory"),
) -> None:
    """Validate project.yaml schema. Exit 0 if OK, exit 2 if invalid."""
    from openfcd.io.project import ProjectModel

    try:
        ProjectModel.from_yaml(project_path / "project.yaml")
        typer.echo("Project schema is valid.")
    except Exception as exc:
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

@app.command()
def run(
    project_path: Path = typer.Argument(..., help="Path to .ofcd project directory"),
    workers: int = typer.Option(-1, help="Parallel workers (-1=auto)"),
    frames: str | None = typer.Option(None, help="Frame range filter (e.g. '0:100' or 'Img200*.jpg')"),
    run_id: str | None = typer.Option(None, help="Run ID (default: run-YYYYMMDD-HHMMSS)"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON lines"),
) -> None:
    """Run the FCD pipeline on a project. Writes runs/{id}/results.h5."""
    # Delegate to cmd_run module to avoid circular imports
    from openfcd.cli.cmd_run import run_cmd as _run_impl

    _run_impl(project_path=project_path, workers=workers, frames=frames, run_id=run_id, json_output=json_output)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

@app.command()
def replay(
    project_path: Path = typer.Argument(..., help="Path to .ofcd project directory"),
    run: str | None = typer.Option(None, help="Run ID (default: latest)"),
    figure: str | None = typer.Option(None, help="Figure ID to render (eta_heatmap, etc.)"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Figure output path"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
) -> None:
    """Replay a previous run's results (no recomputation)."""
    from openfcd.cli.cmd_replay import replay_cmd as _replay_impl

    _replay_impl(project_path=project_path, run=run, figure=figure, output=output, json_output=json_output)


# ---------------------------------------------------------------------------
# Manual grid calibration
# ---------------------------------------------------------------------------

@app.command("calibrate-grid")
def calibrate_grid(
    image: Path = typer.Argument(..., help="Checkerboard/reference image to mark"),
    cells: float = typer.Option(..., "--cells", help="Number of checker cells between the two marked endpoints"),
    cell_mm: float = typer.Option(1.2, "--cell-mm", help="Physical checker cell size in mm"),
    points: str | None = typer.Option(None, "--points", help="Non-interactive endpoints: x1,y1,x2,y2"),
    compare_px_per_mm: float | None = typer.Option(
        None,
        "--compare-px-per-mm",
        help="Optional existing px/mm value to compare against",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
) -> None:
    """Measure pixel/mm by marking a known checkerboard span."""
    from openfcd.cli.cmd_calibrate import calibrate_grid_cmd

    try:
        calibrate_grid_cmd(
            image=image,
            cells=cells,
            cell_mm=cell_mm,
            points=points,
            compare_px_per_mm=compare_px_per_mm,
            json_output=json_output,
        )
    except Exception as exc:
        typer.echo(f"Calibration error: {exc}", err=True)
        raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# QC diagnostics
# ---------------------------------------------------------------------------

@app.command("qc-summary")
def qc_summary(
    project_path: Path = typer.Argument(..., help="Path to .ofcd project directory"),
    run: str | None = typer.Option(None, help="Run ID (default: latest)"),
    frame: int | None = typer.Option(None, help="Frame ID (default: first frame)"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Single-frame QC PNG output"),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Directory for --all QC PNGs"),
    all_frames: bool = typer.Option(False, "--all", help="Render QC summary PNGs for all frames"),
) -> None:
    """Render QC summary figures from an existing run."""
    from openfcd.cli.cmd_qc import qc_summary_cmd

    try:
        qc_summary_cmd(project_path, run, frame, output, output_dir, all_frames)
    except Exception as exc:
        typer.echo(f"QC summary error: {exc}", err=True)
        raise typer.Exit(code=2)


@app.command("noise-floor")
def noise_floor(
    project_path: Path = typer.Argument(..., help="Path to flat-water .ofcd project directory"),
    frames: str | None = typer.Option(None, help="Frame range/filter for the flat-water run"),
    run_id: str | None = typer.Option(None, help="Run ID for a new flat-water run"),
    workers: int = typer.Option(-1, help="Parallel workers (-1=auto)"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Noise-floor JSON summary path"),
    hdf5_output: Path | None = typer.Option(None, "--hdf5-output", help="Noise-floor HDF5 summary path"),
    from_run: str | None = typer.Option(None, "--from-run", help="Analyze an existing run instead of recomputing"),
) -> None:
    """Run or analyze a flat-water sequence and write noise-floor summary stats."""
    from openfcd.cli.cmd_qc import noise_floor_cmd

    try:
        noise_floor_cmd(project_path, frames, run_id, workers, output, hdf5_output, from_run)
    except Exception as exc:
        typer.echo(f"Noise-floor error: {exc}", err=True)
        raise typer.Exit(code=2)


@app.command("sensitivity")
def sensitivity(
    project_path: Path = typer.Argument(..., help="Path to .ofcd project directory"),
    run: str | None = typer.Option(None, help="Run ID (default: latest)"),
    frame: int | None = typer.Option(None, help="Frame ID (default: first frame)"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Sensitivity JSON output"),
) -> None:
    """Estimate eta amplitude sensitivity to calibration parameter perturbations."""
    from openfcd.cli.cmd_qc import sensitivity_cmd

    try:
        sensitivity_cmd(project_path, run, frame, output)
    except Exception as exc:
        typer.echo(f"Sensitivity error: {exc}", err=True)
        raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# Project subcommand group
# ---------------------------------------------------------------------------

project_app = typer.Typer(name="project", help="Project management")


@project_app.command("info")
def project_info(project_path: Path = typer.Argument(...)) -> None:
    """Print project.yaml summary."""
    from openfcd.cli.cmd_project import project_info_cmd as _info_impl

    _info_impl(project_path=project_path)


@project_app.command("runs")
def project_runs(project_path: Path = typer.Argument(...)) -> None:
    """List all runs with STALE status."""
    from openfcd.cli.cmd_project import project_runs_cmd as _runs_impl

    _runs_impl(project_path=project_path)


@project_app.command("fingerprint")
def project_fingerprint(project_path: Path = typer.Argument(...)) -> None:
    """Print current config_fingerprint."""
    from openfcd.cli.cmd_project import project_fingerprint_cmd as _fp_impl

    _fp_impl(project_path=project_path)


app.add_typer(project_app)
