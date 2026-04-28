"""Profile Scene view: per-frame drag-line annotation + cross-section curve."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QPoint, QTimer
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from openfcd.gui import tokens


class _LineAnnotator(QWidget):
    """Image canvas that lets the user drag-draw one profile line."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._image: np.ndarray | None = None
        self._line_start: QPoint | None = None
        self._line_end: QPoint | None = None
        self._drawing = False
        self._line_committed: tuple[QPoint, QPoint] | None = None
        self.setMinimumSize(200, 200)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_image(self, image: np.ndarray | None) -> None:
        self._image = image
        self.update()

    def set_committed_line(self, line: tuple | None) -> None:
        """Show a previously committed line (row,col) coords."""
        if line is None:
            self._line_committed = None
        else:
            p0, p1 = line
            self._line_committed = (
                QPoint(int(p0[1]), int(p0[0])),  # col, row → x, y
                QPoint(int(p1[1]), int(p1[0])),
            )
        self.update()

    def clear_line(self) -> None:
        self._line_committed = None
        self._line_start = None
        self._line_end = None
        self.update()

    def current_line_rowcol(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Return committed line as ((r0,c0),(r1,c1)) or None."""
        if self._line_committed is None:
            return None
        p0, p1 = self._line_committed
        return (float(p0.y()), float(p0.x())), (float(p1.y()), float(p1.x()))

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drawing = True
            self._line_start = ev.pos()
            self._line_end = ev.pos()

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._drawing:
            self._line_end = ev.pos()
            self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        if self._drawing and ev.button() == Qt.MouseButton.LeftButton:
            self._drawing = False
            if self._line_start and self._line_end:
                self._line_committed = (self._line_start, self._line_end)
            self.update()

    def keyPressEvent(self, ev) -> None:
        if ev.key() == Qt.Key.Key_Escape:
            self.clear_line()

    def paintEvent(self, ev) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(tokens.BG_TERTIARY))

        # Draw in-progress or committed line
        line = None
        if self._drawing and self._line_start and self._line_end:
            line = (self._line_start, self._line_end)
        elif self._line_committed:
            line = self._line_committed

        if line:
            pen = QPen(QColor(tokens.ACCENT_CLAY), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawLine(line[0], line[1])
            # Endpoint dots
            painter.setBrush(QColor(tokens.ACCENT_CLAY))
            for p in line:
                painter.drawEllipse(p, 5, 5)


class ProfileSceneView(QWidget):
    """Profile scene: annotation mode + curve view."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._spec = None
        self._project_path: Path | None = None
        self._eta_frames: dict[int, np.ndarray] = {}  # frame_idx → η array
        self._current_frame_pos = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        hl = QHBoxLayout(header)
        hl.setContentsMargins(12, 6, 12, 6)
        self._title = QLabel("Profile")
        hl.addWidget(self._title)
        hl.addStretch()
        self._btn_edit = QPushButton("Edit Lines")
        self._btn_edit.setFixedHeight(24)
        self._btn_edit.clicked.connect(self._enter_annotation)
        hl.addWidget(self._btn_edit)
        self._btn_done = QPushButton("Done")
        self._btn_done.setFixedHeight(24)
        self._btn_done.clicked.connect(self._finish_annotation)
        self._btn_done.setVisible(False)
        hl.addWidget(self._btn_done)
        layout.addWidget(header)

        # Main stack: 0=chart, 1=annotation
        self._mode_stack = QStackedWidget()
        layout.addWidget(self._mode_stack, 1)

        # Chart widget (matplotlib)
        self._chart_widget = self._make_chart_widget()
        self._mode_stack.addWidget(self._chart_widget)  # 0

        # Annotation widget
        self._annotator = _LineAnnotator()
        self._mode_stack.addWidget(self._annotator)  # 1

        # Slider
        slider_row = QWidget()
        sr = QHBoxLayout(slider_row)
        sr.setContentsMargins(12, 4, 12, 4)
        self._slider_lbl = QLabel("frame 0/0")
        self._slider = QSlider(Qt.Orientation.Horizontal)
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

    def _on_slider_moved(self, idx: int) -> None:
        self._pending_idx = idx
        self._render_timer.start()

    def _do_render(self) -> None:
        self._on_slider(self._pending_idx)

    def _make_chart_widget(self) -> QWidget:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            fig = Figure(figsize=(6, 3), tight_layout=True)
            self._ax = fig.add_subplot(111)
            canvas = FigureCanvasQTAgg(fig)
            self._fig = fig
            return canvas
        except ImportError:
            self._ax = None
            self._fig = None
            lbl = QLabel("matplotlib not available")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return lbl

    def load(self, spec, project_path: Path) -> None:
        self._spec = spec
        self._project_path = project_path
        self._title.setText(f"Profile — {spec.name}")
        self._load_eta_frames()
        n = len(spec.frame_indices)
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(0)
        self._slider.setVisible(n > 1)
        self._mode_stack.setCurrentIndex(0)
        self._btn_done.setVisible(False)
        self._btn_edit.setVisible(True)
        self._show_chart(0)

    def _load_eta_frames(self) -> None:
        self._eta_frames = {}
        spec = self._spec
        if spec is None or not self._project_path:
            return
        run_id = spec.run_id or self._latest_run_id(self._project_path)
        if run_id is None:
            return
        h5_path = self._project_path / "runs" / run_id / "results.h5"
        if not h5_path.exists():
            return
        from openfcd.io.result import HDF5ResultStore
        try:
            store = HDF5ResultStore.open(h5_path, "r")
            available = set(store.list_frames("default"))
            for idx in spec.frame_indices:
                if idx in available:
                    try:
                        arr = store.read_frame("default", idx)
                        self._eta_frames[idx] = np.asarray(arr, dtype=np.float64)
                    except Exception:
                        pass
            store.close()
        except Exception:
            pass

    @staticmethod
    def _latest_run_id(project_path: Path) -> str | None:
        runs_dir = project_path / "runs"
        if not runs_dir.exists():
            return None
        dirs = sorted(d.name for d in runs_dir.iterdir() if d.is_dir())
        return dirs[-1] if dirs else None

    def _on_slider(self, pos: int) -> None:
        self._current_frame_pos = pos
        if self._mode_stack.currentIndex() == 0:
            self._show_chart(pos)
        else:
            self._show_annotator_frame(pos)

    def _frame_idx_at(self, pos: int) -> int | None:
        if self._spec and 0 <= pos < len(self._spec.frame_indices):
            return self._spec.frame_indices[pos]
        return None

    def _get_line_for_pos(self, pos: int) -> tuple | None:
        """Return (p0, p1) row-col for this slider position, or default horizontal."""
        spec = self._spec
        if spec is None:
            return None
        frame_idx = self._frame_idx_at(pos)
        if frame_idx is None:
            return None
        plines = spec.profile_lines or {}
        if frame_idx in plines:
            pl = plines[frame_idx]
            return (tuple(pl.p0), tuple(pl.p1))
        # Default: horizontal midline
        eta = self._eta_frames.get(frame_idx)
        if eta is not None:
            h, w = eta.shape
            mid = h // 2
            return (float(mid), 0.0), (float(mid), float(w - 1))
        return None

    def _show_chart(self, pos: int) -> None:
        if self._ax is None:
            return
        self._ax.clear()
        frame_idx = self._frame_idx_at(pos)
        if frame_idx is None:
            self._fig.canvas.draw_idle()
            return
        eta = self._eta_frames.get(frame_idx)
        line = self._get_line_for_pos(pos)
        if eta is None or line is None:
            self._ax.text(
                0.5, 0.5, "No data",
                transform=self._ax.transAxes,
                ha="center", va="center",
            )
            self._fig.canvas.draw_idle()
            return
        from openfcd.core.profile import sample_along
        dists, vals = sample_along(eta, line[0], line[1])
        self._ax.plot(dists, vals, color=tokens.ACCENT_CLAY, linewidth=1.5)
        self._ax.set_xlabel("Distance [px]")
        self._ax.set_ylabel("η [mm]")
        self._ax.set_title(f"Profile — frame {frame_idx}")
        self._ax.grid(True, alpha=0.3)
        self._fig.canvas.draw_idle()

    def _enter_annotation(self) -> None:
        self._mode_stack.setCurrentIndex(1)
        self._btn_edit.setVisible(False)
        self._btn_done.setVisible(True)
        self._show_annotator_frame(self._current_frame_pos)

    def _show_annotator_frame(self, pos: int) -> None:
        line = self._get_line_for_pos(pos)
        self._annotator.set_committed_line(line)

    def _finish_annotation(self) -> None:
        """Save the drawn line back to SceneSpec.profile_lines."""
        pos = self._current_frame_pos
        frame_idx = self._frame_idx_at(pos)
        if frame_idx is not None and self._spec is not None:
            line = self._annotator.current_line_rowcol()
            if line is not None:
                from openfcd.io.scene import ProfileLine
                if self._spec.profile_lines is None:
                    self._spec = self._spec.model_copy(update={"profile_lines": {}})
                updated_lines = dict(self._spec.profile_lines or {})
                updated_lines[frame_idx] = ProfileLine(p0=line[0], p1=line[1])
                self._spec = self._spec.model_copy(update={"profile_lines": updated_lines})
        self._mode_stack.setCurrentIndex(0)
        self._btn_done.setVisible(False)
        self._btn_edit.setVisible(True)
        self._show_chart(self._current_frame_pos)
