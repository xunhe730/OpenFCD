"""Run controller — wraps subprocess in QThread, emits StageEvent signals."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal


class StageEvent:
    """Lightweight stage event parsed from pipeline JSON output."""

    def __init__(self, stage: str, progress: float, frame: int | None = None) -> None:
        self.stage = stage
        self.progress = progress
        self.frame = frame


class _RunWorker(QThread):
    """Background thread that runs `openfcd run --json`."""

    stage_event = pyqtSignal(object)  # StageEvent
    run_finished = pyqtSignal(str)    # run_id
    run_failed = pyqtSignal(str)      # error message

    def __init__(self, project_path: Path, workers: int = -1, disabled_indices: frozenset[int] = frozenset()) -> None:
        super().__init__()
        self._project_path = project_path
        self._workers = workers
        self._disabled_indices = disabled_indices
        self._process: subprocess.Popen | None = None
        self._cancelled = False

    def run(self) -> None:
        from openfcd.cli.cmd_run import PreprocessStage, ComputeStage, PostprocessStage
        from openfcd.io.store import FileSessionStore
        from openfcd.io.result import HDF5ResultStore
        from openfcd.pipeline.runner import PipelineRunner
        from openfcd.pipeline.base import CancelToken, CancelledError

        # Create cancel token
        self._cancel_token = CancelToken()

        try:
            store = FileSessionStore.open(self._project_path)
            project = store.project
            run_id = "run-latest"

            # Delete the previous run so results don't accumulate.
            import shutil as _shutil
            run_dir = store._dir / "runs" / run_id
            if run_dir.exists():
                _shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            result_store = HDF5ResultStore.open(run_dir / "results.h5", mode="w")

            ctx: dict = {
                "project": project,
                "project_dir": store._dir,
                "frames_dir": Path(project.data.frames_dir),
                "pattern": project.data.pattern,
                "frames_filter": None,
                "run_id": run_id,
                "result_store": result_store,
                "batches_processed": [],
                "frame_count": 0,
                "frame_paths": [],
"annotation": store.annotation,  # pass annotation for ROI/mask
                "workers": self._workers,
                "disabled_frame_indices": self._disabled_indices,
}

            stages = [PreprocessStage(), ComputeStage(), PostprocessStage()]
            runner = PipelineRunner(stages, workers=self._workers if self._workers > 0 else -1)

            exit_code = 0
            for event in runner.iter(ctx, self._cancel_token):
                if self._cancel_token.is_cancelled:
                    exit_code = 1
                
                # Convert Pipeline event to GUI StageEvent
                stage = event.stage or "unknown"
                progress = event.progress if event.progress is not None else 0.0
                frame = event.frame_idx
                self.stage_event.emit(StageEvent(stage, progress, frame))

                if event.kind == "error":
                    exit_code = 2
                    break
                if event.kind == "cancel":
                    exit_code = 1
                    break

        except CancelledError:
            exit_code = 1
        except Exception as exc:
            self.run_failed.emit(str(exc))
            exit_code = 2
        
        try:
            # Finalize run status
            manifest = {
                "project_name": project.name,
                "run_id": run_id,
                "frames_filter": None,
                "workers": self._workers,
                "status": "success" if exit_code == 0 else ("cancelled" if exit_code == 1 else "error"),
            }
            store.record_run(run_id, manifest)

            if ctx.get("result_store") is not None:
                try:
                    result_store.close()
                except Exception:
                    pass
            store.close()

            if exit_code == 0:
                self.run_finished.emit(run_id)
            elif exit_code == 1:
                self.run_failed.emit("Run cancelled")
            else:
                self.run_failed.emit("Run failed with errors")

        except Exception as exc:
            self.run_failed.emit(str(exc))

    def cancel(self) -> None:
        """Signal the pipeline to cancel."""
        self._cancelled = True
        if hasattr(self, "_cancel_token"):
            self._cancel_token.cancel()


class RunController(QObject):
    """High-level controller for pipeline execution."""

    stage_event = pyqtSignal(object)   # StageEvent
    run_finished = pyqtSignal(str)     # run_id
    run_failed = pyqtSignal(str)       # error message

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._worker: _RunWorker | None = None

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def start_run(self, project_path: str | Path, workers: int = -1, disabled_indices: frozenset[int] = frozenset()) -> None:
        """Start a pipeline run in background thread."""
        if self.is_running:
            self.run_failed.emit("Run already in progress")
            return

        path = Path(project_path)
        self._worker = _RunWorker(path, workers, disabled_indices)
        self._worker.stage_event.connect(self.stage_event)
        self._worker.run_finished.connect(self.run_finished)
        self._worker.run_failed.connect(self.run_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def cancel(self) -> None:
        """Cancel the current run."""
        if self._worker is not None:
            self._worker.cancel()

    def _on_worker_finished(self) -> None:
        self._worker = None
