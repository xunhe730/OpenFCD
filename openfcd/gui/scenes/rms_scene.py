"""RMS Scene view: per-pixel sqrt(mean(η²)) over selected frames."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from openfcd.gui import tokens
from openfcd.gui.scenes.eta_map import EtaMap


class _RmsWorker(QThread):
    """Compute RMS in background."""

    result_ready = pyqtSignal(object)  # np.ndarray or None
    failed = pyqtSignal(str)

    def __init__(self, project_path: Path, spec, annotation) -> None:
        super().__init__()
        self._project_path = project_path
        self._spec = spec
        self._annotation = annotation

    def run(self) -> None:
        try:
            spec = self._spec
            run_id = spec.run_id or self._latest_run_id(self._project_path)
            if run_id is None:
                self.result_ready.emit(None)
                return
            h5_path = self._project_path / "runs" / run_id / "results.h5"
            if not h5_path.exists():
                self.result_ready.emit(None)
                return

            from openfcd.io.result import HDF5ResultStore
            store = HDF5ResultStore.open(h5_path, "r")
            available = set(store.list_frames("default"))
            arrays = []
            for idx in spec.frame_indices:
                if idx in available:
                    try:
                        arr = store.read_frame("default", idx)
                        arrays.append(np.asarray(arr, dtype=np.float64))
                    except Exception:
                        pass
            store.close()

            if not arrays:
                self.result_ready.emit(None)
                return

            # Apply ROI mask if annotation present
            roi_box = None
            if self._annotation is not None:
                roi = self._annotation.roi
                if not roi.is_empty:
                    roi_box = (int(roi.y), int(roi.x), int(roi.height), int(roi.width))

            # Compute RMS: sqrt(nanmean(η², axis=0))
            target_shape = arrays[0].shape
            valid = [a for a in arrays if a.shape == target_shape]
            stack = np.stack(valid, axis=0)
            if roi_box is not None:
                r0, c0, h, w = roi_box
                rms = np.full(target_shape, np.nan)
                region = stack[:, r0:r0 + h, c0:c0 + w]
                rms[r0:r0 + h, c0:c0 + w] = np.sqrt(np.nanmean(region ** 2, axis=0))
            else:
                rms = np.sqrt(np.nanmean(stack ** 2, axis=0))

            self.result_ready.emit(rms)
        except Exception as exc:
            self.failed.emit(str(exc))

    @staticmethod
    def _latest_run_id(project_path: Path) -> str | None:
        runs_dir = project_path / "runs"
        if not runs_dir.exists():
            return None
        dirs = sorted(d.name for d in runs_dir.iterdir() if d.is_dir())
        return dirs[-1] if dirs else None


class RmsSceneView(QWidget):
    """Displays per-pixel η RMS for a selected set of frames."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._worker: _RmsWorker | None = None
        self._old_workers: list[_RmsWorker] = []
        self._generation = 0
        self._cached_rms: np.ndarray | None = None
        self._cached_spec_id: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._title = QLabel("RMS")
        self._title.setContentsMargins(12, 6, 12, 0)
        self._title.setStyleSheet(
            f"color:{tokens.TEXT_SECONDARY};font-size:12px;font-weight:600;"
        )
        layout.addWidget(self._title)
        self._status = QLabel("Computing…")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet(
            f"color:{tokens.TEXT_MUTED};font-size:12px;padding:20px;"
        )
        layout.addWidget(self._status)
        self._eta_map = EtaMap(self)
        self._eta_map.setVisible(False)
        layout.addWidget(self._eta_map, 1)

    def load(self, spec, project_path: Path, annotation=None) -> None:
        self._generation += 1
        generation = self._generation
        self._title.setText(f"RMS — {spec.name}  ({len(spec.frame_indices)} frames)")
        self._retire_current_worker()
        # Use cached result if spec hasn't changed
        if self._cached_spec_id == spec.id and self._cached_rms is not None:
            self._show_rms(self._cached_rms)
            return
        self._cached_spec_id = spec.id
        self._cached_rms = None
        self._status.setText("Computing RMS…")
        self._status.setVisible(True)
        self._eta_map.setVisible(False)
        self._worker = _RmsWorker(project_path, spec, annotation)
        self._worker.result_ready.connect(lambda rms, gen=generation: self._on_result(gen, rms))
        self._worker.failed.connect(lambda msg, gen=generation: self._on_failed(gen, msg))
        self._worker.finished.connect(lambda worker=self._worker: self._on_worker_finished(worker))
        self._worker.start()

    def _retire_current_worker(self) -> None:
        if self._worker is None:
            return
        worker = self._worker
        self._worker = None
        worker.requestInterruption()
        self._old_workers.append(worker)

    def _on_worker_finished(self, worker: _RmsWorker) -> None:
        if worker is self._worker:
            self._worker = None
        if worker in self._old_workers:
            self._old_workers.remove(worker)
        worker.deleteLater()

    def _on_result(self, generation: int, rms) -> None:
        if generation != self._generation:
            return
        if rms is None:
            self._status.setText("No data — run a computation first.")
            return
        self._cached_rms = rms
        self._show_rms(rms)

    def _on_failed(self, generation: int, msg: str) -> None:
        if generation != self._generation:
            return
        self._status.setText(f"Failed: {msg[:80]}")

    def _show_rms(self, rms: np.ndarray) -> None:
        self._status.setVisible(False)
        self._eta_map.setVisible(True)
        self._eta_map.set_data(rms, colormap="magma")

    def apply_viz(self, params: dict) -> None:
        cmap = params.get("cmap", "magma")
        if self._cached_rms is not None:
            self._eta_map.set_data(self._cached_rms, colormap=cmap)

    def closeEvent(self, event) -> None:
        self._retire_current_worker()
        for worker in list(self._old_workers):
            worker.requestInterruption()
        super().closeEvent(event)
