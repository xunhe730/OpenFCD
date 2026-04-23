"""Integration tests for the openfcd CLI."""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from openfcd.cli import app
from openfcd.pipeline.base import Stage, StageEvent, CancelToken

runner = CliRunner()


def _write_minimal_project(dir_path: Path, name: str = "test-project") -> Path:
    """Create a minimal valid project.yaml in dir_path."""
    import io
    from ruamel.yaml import YAML

    data = {
        "name": name,
        "created": "2026-04-22T10:00:00+00:00",
        "format_version": 0,
        "geometry": {
            "pattern_period_mm": 1.2,
            "optical_stack": {"preset": "custom", "layers": []},
        },
        "data": {
            "frames_dir": str(dir_path / "images"),
            "pattern": "Img*.jpg",
        },
        "process": {
            "flatfield_sigma": 300.0,
            "detrend": "plane",
            "taper": {"kind": "tukey", "alpha": 0.08},
            "edge_nan_mm": 3.0,
        },
        "viz": {
            "eta_vmin_mm": -0.35,
            "eta_vmax_mm": 0.35,
            "cmap": "RdBu_r",
            "figure_dpi": 200,
        },
    }
    yaml = YAML()
    yaml.default_flow_style = False
    buf = io.StringIO()
    yaml.dump(data, buf)
    (dir_path / "project.yaml").write_text(buf.getvalue())
    return dir_path / "project.yaml"


def _write_invalid_project(dir_path: Path) -> Path:
    """Create an invalid project.yaml (missing required fields)."""
    content = "name: broken\n"  # missing created, geometry, data
    (dir_path / "project.yaml").write_text(content)
    return dir_path / "project.yaml"


# ---------------------------------------------------------------------------
# Test 1: version
# ---------------------------------------------------------------------------

def test_version() -> None:
    """`openfcd version` prints 0.0.1."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.0.1" in result.output


# ---------------------------------------------------------------------------
# Test 2: validate valid project
# ---------------------------------------------------------------------------

def test_validate_valid(tmp_path: Path) -> None:
    """`openfcd validate` exits 0 for a valid project."""
    _write_minimal_project(tmp_path)
    result = runner.invoke(app, ["validate", str(tmp_path)])
    assert result.exit_code == 0
    assert "valid" in result.output.lower()


# ---------------------------------------------------------------------------
# Test 3: validate invalid project
# ---------------------------------------------------------------------------

def test_validate_invalid(tmp_path: Path) -> None:
    """`openfcd validate` exits 2 for an invalid project."""
    _write_invalid_project(tmp_path)
    result = runner.invoke(app, ["validate", str(tmp_path)])
    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# Test 4: project info
# ---------------------------------------------------------------------------

def test_project_info(tmp_path: Path) -> None:
    """`openfcd project info` prints project name."""
    _write_minimal_project(tmp_path, name="my-cool-project")
    result = runner.invoke(app, ["project", "info", str(tmp_path)])
    assert result.exit_code == 0
    assert "my-cool-project" in result.output


# ---------------------------------------------------------------------------
# Test 5: run --json produces valid JSON lines
# ---------------------------------------------------------------------------

def test_run_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`openfcd run --json` produces valid JSON lines with run_id."""
    _write_minimal_project(tmp_path)

    # Create a dummy stage that yields a few events
    class DummyStage(Stage):
        name: str = "dummy"

        def run(
            self, ctx: dict, cancel: CancelToken | None = None
        ) -> Iterator[StageEvent]:
            run_id = ctx.get("run_id", "test-run")
            yield StageEvent(
                kind="start", stage="preprocess", batch=None, frame_idx=None,
                substage=None, progress=0.0, total=None, completed=None,
                metrics={}, run_id=run_id,
            )
            yield StageEvent(
                kind="finish", stage="preprocess", batch=None, frame_idx=None,
                substage=None, progress=1.0, total=0, completed=0,
                metrics={"frame_count": 0}, run_id=run_id,
            )

        def dry_run(self, ctx: dict) -> list[str]:
            return ["dummy"]

    # Patch cmd_run to use our dummy stages
    import openfcd.cli.cmd_run as cmd_run_mod
    original_run = cmd_run_mod.run_cmd

    def patched_run_cmd(
        project_path: Path,
        workers: int = -1,
        frames: str | None = None,
        run_id: str | None = None,
        json_output: bool = False,
    ) -> None:
        from datetime import datetime, timezone
        from openfcd.io.store import FileSessionStore
        from openfcd.io.project import ProjectModel
        from openfcd.io.result import HDF5ResultStore
        from openfcd.pipeline.runner import PipelineRunner
        from openfcd.cli._output import stage_event_to_json
        import typer

        store = FileSessionStore.open(project_path)
        project: ProjectModel = store.project

        if run_id is None:
            run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

        run_dir = store._dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        result_store = HDF5ResultStore.open(run_dir / "results.h5", mode="w")

        ctx: dict = {
            "project": project,
            "frames_dir": Path(project.data.frames_dir),
            "pattern": project.data.pattern,
            "frames_filter": frames,
            "run_id": run_id,
            "result_store": result_store,
            "pattern_period_mm": project.geometry.pattern_period_mm,
            "optical_stack": project.geometry.optical_stack,
            "flatfield_sigma": project.process.flatfield_sigma,
            "detrend": project.process.detrend,
            "taper": project.process.taper,
            "edge_nan_mm": project.process.edge_nan_mm,
            "viz": project.viz,
            "output": project.output,
            "batches_processed": [],
            "frame_count": 0,
            "frame_paths": [],
        }

        runner_obj = PipelineRunner([DummyStage()], workers=1)

        for event in runner_obj.iter(ctx, None):
            if json_output:
                typer.echo(stage_event_to_json(event))

        result_store.close()
        manifest = {
            "project_name": project.name,
            "run_id": run_id,
            "status": "success",
        }
        store.record_run(run_id, manifest)
        store.close()

    monkeypatch.setattr(cmd_run_mod, "run_cmd", patched_run_cmd)

    # Now invoke the CLI
    result = runner.invoke(app, ["run", str(tmp_path), "--json"])

    # Exit code 0 means success; typer.Exit with code=0 still raises
    # but CliRunner catches it. Exit code 1 is from typer.Exit(0) sometimes.
    # Actually, typer.Exit(code=0) gives exit_code 0.
    # Our run_cmd raises typer.Exit(code=0) for success.
    assert result.exit_code == 0

    lines = [line for line in result.output.strip().split("\n") if line.strip()]
    assert len(lines) >= 1

    # Each line should be valid JSON with all 12 fields
    required_fields = {
        "kind", "stage", "batch", "frame_idx", "substage", "progress",
        "total", "completed", "metrics", "run_id", "seq", "timestamp",
    }
    for line in lines:
        parsed = json.loads(line)
        assert required_fields.issubset(set(parsed.keys())), (
            f"Missing fields in: {parsed}"
        )

    # Last line should contain a run_id
    last_parsed = json.loads(lines[-1])
    assert "run_id" in last_parsed
    assert last_parsed["run_id"].startswith("run-")


# ---------------------------------------------------------------------------
# Test 6: project runs with STALE/OK markers
# ---------------------------------------------------------------------------

def test_project_runs_stale(tmp_path: Path) -> None:
    """`openfcd project runs` lists runs with STALE/OK markers."""
    _write_minimal_project(tmp_path)

    # Create a fake run with a manifest
    from datetime import datetime, timezone
    from openfcd.io.store import FileSessionStore

    store = FileSessionStore.open(tmp_path)
    run_id = "run-20260422-100000"
    manifest = {
        "project_name": "test-project",
        "run_id": run_id,
        "config_fingerprint": store.config_fingerprint(),  # same = not stale
        "status": "success",
    }
    store.record_run(run_id, manifest)
    store.close()

    result = runner.invoke(app, ["project", "runs", str(tmp_path)])
    assert result.exit_code == 0
    assert run_id in result.output
    assert "OK" in result.output
