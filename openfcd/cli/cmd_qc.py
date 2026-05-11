from __future__ import annotations

import json
from pathlib import Path

import typer

from openfcd.io.result import HDF5ResultStore
from openfcd.io.store import FileSessionStore
from openfcd.pipeline.qc import (
    eta_noise_summary,
    parameter_sensitivity_report,
    save_qc_summary_figure,
    write_noise_summary,
)


def _latest_run_id(store: FileSessionStore) -> str:
    runs = store.list_runs()
    if not runs:
        raise RuntimeError("no runs found")
    return str(runs[-1]["run_id"])


def _open_results(project_path: Path, run: str | None) -> tuple[FileSessionStore, HDF5ResultStore, str]:
    store = FileSessionStore.open(project_path, read_only=True)
    run_id = run or _latest_run_id(store)
    h5_path = store._dir / "runs" / run_id / "results.h5"
    if not h5_path.exists():
        store.close()
        raise RuntimeError(f"results not found: {h5_path}")
    return store, HDF5ResultStore.open(h5_path, "r"), run_id


def _frame_image(store: FileSessionStore, results: HDF5ResultStore, frame_id: int):
    frame_path = results.read_frame_attrs("default", frame_id).get("frame_path")
    if not frame_path:
        return None
    path = Path(str(frame_path))
    if not path.is_absolute():
        path = Path(store.project.data.frames_dir) / path
    if not path.exists():
        return None
    from openfcd.pipeline.compute import load_gray
    return load_gray(path)


def qc_summary_cmd(
    project_path: Path,
    run: str | None,
    frame: int | None,
    output: Path | None,
    output_dir: Path | None,
    all_frames: bool,
) -> None:
    store, results, run_id = _open_results(project_path, run)
    try:
        frame_ids = results.list_frames("default")
        if not frame_ids:
            raise RuntimeError("no frame results found")
        selected = frame_ids if all_frames else [frame if frame is not None else frame_ids[0]]
        if all_frames:
            out_dir = output_dir or (store._dir / "runs" / run_id / "qc")
            out_dir.mkdir(parents=True, exist_ok=True)
            for fid in selected:
                save_qc_summary_figure(
                    results,
                    "default",
                    fid,
                    out_dir / f"frame_{fid:06d}_qc.png",
                    image=_frame_image(store, results, fid),
                )
            typer.echo(json.dumps({"run_id": run_id, "frames": selected, "output_dir": str(out_dir)}))
        else:
            out = output or (store._dir / "runs" / run_id / f"frame_{selected[0]:06d}_qc.png")
            save_qc_summary_figure(
                results,
                "default",
                selected[0],
                out,
                image=_frame_image(store, results, selected[0]),
            )
            typer.echo(json.dumps({"run_id": run_id, "frame": selected[0], "output": str(out)}))
    finally:
        results.close()
        store.close()


def noise_floor_cmd(
    project_path: Path,
    frames: str | None,
    run_id: str | None,
    workers: int,
    output: Path | None,
    hdf5_output: Path | None,
    from_run: str | None,
) -> None:
    if from_run is None:
        from openfcd.cli.cmd_run import run_cmd
        run_id = run_id or "flat-water-noise"
        try:
            run_cmd(project_path=project_path, workers=workers, frames=frames, run_id=run_id, json_output=False)
        except typer.Exit as exc:
            if exc.exit_code != 0:
                raise
        from_run = run_id

    store, results, resolved = _open_results(project_path, from_run)
    try:
        summary = eta_noise_summary(results, "default")
        summary["run_id"] = resolved
        json_path = output or (store._dir / "runs" / resolved / "noise_floor_summary.json")
        h5_path = hdf5_output or (store._dir / "runs" / resolved / "noise_floor_summary.h5")
        write_noise_summary(summary, json_path, h5_path)
        typer.echo(json.dumps(summary, sort_keys=True))
    finally:
        results.close()
        store.close()


def sensitivity_cmd(
    project_path: Path,
    run: str | None,
    frame: int | None,
    output: Path | None,
) -> None:
    store, results, run_id = _open_results(project_path, run)
    try:
        frame_ids = results.list_frames("default")
        if not frame_ids:
            raise RuntimeError("no frame results found")
        fid = frame if frame is not None else frame_ids[0]
        report = parameter_sensitivity_report(
            results.read_frame("default", fid),
            results.read_frame_attrs("default", fid),
        )
        report["run_id"] = run_id
        report["frame_id"] = fid
        out = output or (store._dir / "runs" / run_id / f"frame_{fid:06d}_sensitivity.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        typer.echo(json.dumps({"run_id": run_id, "frame": fid, "output": str(out)}))
    finally:
        results.close()
        store.close()
