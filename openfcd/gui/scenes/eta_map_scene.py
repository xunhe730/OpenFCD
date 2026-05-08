"""η Map Scene view: slider-navigated η heatmap over a frame subset from results.h5."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from openfcd.gui import tokens
from openfcd.gui.scenes.eta_map import EtaMap


class EtaMapSceneView(QWidget):
    """Renders one scene's η frames with a slider over frame_indices."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._spec = None
        self._eta_frames: list[np.ndarray | None] = []
        self._frame_indices: list[int] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Title bar
        self._title = QLabel("η Map")
        self._title.setContentsMargins(12, 6, 12, 6)
        tokens.on_theme_changed(lambda: self._title.setStyleSheet(
            f"color:{tokens.TEXT_SECONDARY};font-size:12px;font-weight:600;"
        ))
        self._title.setStyleSheet(
            f"color:{tokens.TEXT_SECONDARY};font-size:12px;font-weight:600;"
        )
        layout.addWidget(self._title)

        # η map
        self._eta_map = EtaMap(self)
        layout.addWidget(self._eta_map, 1)

        # Slider row
        slider_row = QWidget()
        sr = QHBoxLayout(slider_row)
        sr.setContentsMargins(12, 4, 12, 4)
        self._slider_lbl = QLabel("frame 0/0")
        self._slider_lbl.setStyleSheet(f"color:{tokens.TEXT_MUTED};font-size:11px;")
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        # Debounced slider: prevent reentrant matplotlib draw() calls
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(80)
        self._pending_idx = 0
        self._render_timer.timeout.connect(self._do_render)
        self._slider.valueChanged.connect(self._on_slider_moved)
        sr.addWidget(self._slider_lbl)
        sr.addWidget(self._slider, 1)
        layout.addWidget(slider_row)

    def load(self, spec, project_path: Path) -> None:
        """Load η frames from results.h5 for this spec's frame_indices."""
        self._spec = spec
        self._frame_indices = list(spec.frame_indices)
        self._eta_frames = []
        title = (spec.viz_params or {}).get("title") or spec.name or "η Map"
        self._title.setText(str(title))

        run_id = spec.run_id or self._latest_run_id(project_path)
        if run_id is None or not self._frame_indices:
            self._eta_map.clear("No run selected")
            return

        h5_path = project_path / "runs" / run_id / "results.h5"
        if not h5_path.exists():
            self._eta_map.clear("Results file not found")
            return

        from openfcd.io.result import HDF5ResultStore
        try:
            store = HDF5ResultStore.open(h5_path, "r")
            available = set(store.list_frames("default"))
            for idx in self._frame_indices:
                if idx in available:
                    try:
                        arr = store.read_frame("default", idx)
                        self._eta_frames.append(np.asarray(arr, dtype=np.float64))
                    except Exception:
                        self._eta_frames.append(None)
                else:
                    self._eta_frames.append(None)
            store.close()
        except Exception:
            self._eta_frames = [None] * len(self._frame_indices)

        n = len(self._eta_frames)
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(0)
        self._slider.setVisible(n > 1)
        self._show_frame(0)

    @staticmethod
    def _latest_run_id(project_path: Path) -> str | None:
        runs_dir = project_path / "runs"
        if not runs_dir.exists():
            return None
        dirs = sorted(d.name for d in runs_dir.iterdir() if d.is_dir())
        return dirs[-1] if dirs else None

    def _on_slider_moved(self, idx: int) -> None:
        self._pending_idx = idx
        self._render_timer.start()

    def _do_render(self) -> None:
        self._show_frame(self._pending_idx)

    def _show_frame(self, idx: int) -> None:
        n = len(self._eta_frames)
        self._slider_lbl.setText(f"frame {idx + 1}/{n}")
        if 0 <= idx < n and self._eta_frames[idx] is not None:
            params = self._spec.viz_params or {}
            cmap = params.get("cmap", "RdBu_r")
            self._eta_map.set_data(
                self._eta_frames[idx],
                colormap=cmap,
                vmin=self._parse_float(params.get("vmin")),
                vmax=self._parse_float(params.get("vmax")),
                colorbar=params.get("colorbar", "right"),
            )
        else:
            self._eta_map.clear("No data")

    def apply_viz(self, params: dict) -> None:
        if self._spec is not None:
            self._spec = self._spec.model_copy(update={"viz_params": dict(params)})
            self._title.setText(str(params.get("title") or self._spec.name or "η Map"))
        self._show_frame(self._slider.value())

    @staticmethod
    def _parse_float(value) -> float | None:
        if value in (None, "", "auto"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
