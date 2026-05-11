"""Profile Scene view: per-frame drag-line annotation + composite preview."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QPoint, QRectF, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSizePolicy,
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
        self._pixmap: QPixmap | None = None
        self._image_rect = QRectF()
        self._line_start: QPoint | None = None
        self._line_end: QPoint | None = None
        self._drawing = False
        self._line_committed: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.setMinimumSize(200, 200)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_image(self, image: np.ndarray | None) -> None:
        self._image = image
        self._pixmap = self._make_pixmap(image)
        self._update_image_rect()
        self.update()

    def _make_pixmap(self, image: np.ndarray | None) -> QPixmap | None:
        if image is None or image.size == 0:
            return None
        try:
            from matplotlib import colormaps
            from matplotlib.colors import Normalize

            from openfcd.gui.renderers._eta_view import (
                compute_eta_color_range,
                crop_to_valid,
            )

            # Share color-range policy with EtaMap so Profile annotation overlay
            # never saturates differently from the Run-monitor view.
            color_eta = crop_to_valid(np.asarray(image, dtype=float))
            viz_stub = type(
                "_AnnotatorVizStub",
                (),
                {"eta_vmin_mm": None, "eta_vmax_mm": None},
            )()
            vmin, vmax = compute_eta_color_range(color_eta, viz_stub, cmap="RdBu_r")
            norm = Normalize(vmin=vmin, vmax=vmax)
            rgba = (colormaps.get_cmap("RdBu_r")(norm(np.nan_to_num(image))) * 255).astype(np.uint8)
            rgba[~np.isfinite(image), 3] = 0
            h, w, _ = rgba.shape
            qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
            return QPixmap.fromImage(qimg)
        except Exception:
            return None

    def _update_image_rect(self) -> None:
        if self._image is None:
            self._image_rect = QRectF()
            return
        h, w = self._image.shape[:2]
        if h <= 0 or w <= 0 or self.width() <= 0 or self.height() <= 0:
            self._image_rect = QRectF()
            return
        scale = min(self.width() / w, self.height() / h)
        draw_w = w * scale
        draw_h = h * scale
        self._image_rect = QRectF(
            (self.width() - draw_w) / 2,
            (self.height() - draw_h) / 2,
            draw_w,
            draw_h,
        )

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._update_image_rect()

    def _widget_to_rowcol(self, pos: QPoint) -> tuple[float, float] | None:
        if (
            self._image is None
            or self._image_rect.isEmpty()
            or not self._image_rect.contains(float(pos.x()), float(pos.y()))
        ):
            return None
        h, w = self._image.shape[:2]
        col = (pos.x() - self._image_rect.left()) / self._image_rect.width() * (w - 1)
        row = (pos.y() - self._image_rect.top()) / self._image_rect.height() * (h - 1)
        return (
            float(np.clip(row, 0, h - 1)),
            float(np.clip(col, 0, w - 1)),
        )

    def _rowcol_to_widget(self, point: tuple[float, float]) -> QPoint:
        if self._image is None or self._image_rect.isEmpty():
            return QPoint(int(point[1]), int(point[0]))
        h, w = self._image.shape[:2]
        row, col = point
        x = self._image_rect.left() + (col / max(1, w - 1)) * self._image_rect.width()
        y = self._image_rect.top() + (row / max(1, h - 1)) * self._image_rect.height()
        return QPoint(int(round(x)), int(round(y)))

    def set_committed_line(self, line: tuple | None) -> None:
        """Show a previously committed line (row,col) coords."""
        if line is None:
            self._line_committed = None
        else:
            p0, p1 = line
            self._line_committed = ((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])))
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
        return self._line_committed

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.MouseButton.LeftButton and self._widget_to_rowcol(ev.pos()) is not None:
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
                p0 = self._widget_to_rowcol(self._line_start)
                p1 = self._widget_to_rowcol(self._line_end)
                if p0 is not None and p1 is not None:
                    self._line_committed = (p0, p1)
            self.update()

    def keyPressEvent(self, ev) -> None:
        if ev.key() == Qt.Key.Key_Escape:
            self.clear_line()

    def paintEvent(self, ev) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(tokens.BG_TERTIARY))
        if self._pixmap is not None and not self._image_rect.isEmpty():
            painter.drawPixmap(self._image_rect, self._pixmap, QRectF(self._pixmap.rect()))
        else:
            painter.setPen(QColor(tokens.TEXT_MUTED))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No eta frame")

        # Draw in-progress or committed line
        line = None
        if self._drawing and self._line_start and self._line_end:
            line = (self._line_start, self._line_end)
        elif self._line_committed:
            line = (
                self._rowcol_to_widget(self._line_committed[0]),
                self._rowcol_to_widget(self._line_committed[1]),
            )

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

    scene_changed = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._spec = None
        self._project_path: Path | None = None
        self._eta_frames: dict[int, np.ndarray] = {}  # frame_idx → η array
        self._frame_names: dict[int, str] = {}
        self._frame_calibrations: dict[int, object] = {}
        self._annotation = None
        self._project_model = None
        self._run_manifest: dict | None = None
        self._current_frame_pos = 0
        self._preferred_frame_pos: int | None = None
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
            from matplotlib.figure import Figure
            from openfcd.gui.widgets.mpl_canvas import CompactCanvas
            fig = Figure(figsize=(8.6, 5.8))
            self._ax = fig.add_subplot(111)
            canvas = CompactCanvas(fig)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            canvas.updateGeometry()
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
        initial_pos = 0
        if self._preferred_frame_pos is not None:
            initial_pos = max(0, min(self._preferred_frame_pos, max(0, n - 1)))
        self._set_frame_pos(initial_pos)
        self._slider.setVisible(n > 1)
        self._mode_stack.setCurrentIndex(0)
        self._btn_done.setVisible(False)
        self._btn_edit.setVisible(True)
        missing_pos = self._first_missing_line_pos()
        if missing_pos is not None:
            self._set_frame_pos(missing_pos)
            self._enter_annotation()
        else:
            self._show_chart(self._current_frame_pos)

    def _load_eta_frames(self) -> None:
        self._eta_frames = {}
        self._frame_names = {}
        self._frame_calibrations = {}
        self._annotation = None
        self._project_model = None
        self._run_manifest = None
        spec = self._spec
        if spec is None or not self._project_path:
            return
        try:
            from openfcd.io.store import FileSessionStore
            session = FileSessionStore.open(self._project_path, read_only=True)
            self._annotation = session.annotation
            self._project_model = session.project
            run_id_for_manifest = spec.run_id or self._latest_run_id(self._project_path)
            self._run_manifest = next(
                (r for r in session.list_runs() if r.get("run_id") == run_id_for_manifest),
                None,
            )
            session.close()
        except Exception:
            self._annotation = None
            self._project_model = None
            self._run_manifest = None
        run_id = spec.run_id or self._latest_run_id(self._project_path)
        if run_id is None:
            return
        h5_path = self._project_path / "runs" / run_id / "results.h5"
        if not h5_path.exists():
            return
        from openfcd.io.result import HDF5ResultStore
        try:
            store = HDF5ResultStore.open(h5_path, "r")
            from openfcd.core.profile_composite import resolve_spatial_calibration
            available = set(store.list_frames("default"))
            for idx in spec.frame_indices:
                if idx in available:
                    try:
                        arr = store.read_frame("default", idx)
                        self._eta_frames[idx] = np.asarray(arr, dtype=np.float64)
                        attrs = store.read_frame_attrs("default", idx)
                        frame_path = attrs.get("frame_path")
                        if frame_path:
                            self._frame_names[idx] = Path(str(frame_path)).name
                        self._frame_calibrations[idx] = resolve_spatial_calibration(
                            store,
                            "default",
                            idx,
                            self._run_manifest,
                            self._project_model,
                        )
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
        total = len(self._spec.frame_indices) if self._spec is not None else 0
        self._slider_lbl.setText(f"frame {pos}/{max(0, total - 1)}")
        if self._mode_stack.currentIndex() == 0:
            self._show_chart(pos)
        else:
            self._show_annotator_frame(pos)

    def _set_frame_pos(self, pos: int) -> None:
        self._current_frame_pos = pos
        total = len(self._spec.frame_indices) if self._spec is not None else 0
        self._slider_lbl.setText(f"frame {pos}/{max(0, total - 1)}")
        if self._slider.value() != pos:
            self._slider.setValue(pos)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        if getattr(self, "_fig", None) is not None and self._mode_stack.currentIndex() == 0:
            self._show_chart(self._current_frame_pos)

    def _frame_idx_at(self, pos: int) -> int | None:
        if self._spec and 0 <= pos < len(self._spec.frame_indices):
            return self._spec.frame_indices[pos]
        return None

    def _needs_line_annotation(self) -> bool:
        if self._spec is None or not self._spec.frame_indices:
            return False
        lines = self._spec.profile_lines or {}
        return any(idx not in lines for idx in self._spec.frame_indices)

    def _first_missing_line_pos(self, start_pos: int = 0) -> int | None:
        spec = self._spec
        if spec is None:
            return None
        lines = spec.profile_lines or {}
        positions = list(range(max(0, start_pos), len(spec.frame_indices)))
        positions.extend(range(0, min(max(0, start_pos), len(spec.frame_indices))))
        for pos in positions:
            frame_idx = self._frame_idx_at(pos)
            if frame_idx is not None and frame_idx not in lines:
                return pos
        return None

    def _get_line_for_pos(self, pos: int) -> tuple | None:
        """Return saved (p0, p1) row-col for this slider position."""
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
        return None

    def _profile_viz_params(self) -> dict:
        params = dict(self._spec.viz_params or {}) if self._spec is not None else {}
        params.pop("px_per_mm", None)
        params["layout_mode"] = "preview"
        if self._fig is not None:
            dpi = float(params.get("dpi", 150) or 150)
            canvas = self._fig.canvas
            if canvas is not None and canvas.width() > 1 and canvas.height() > 1 and dpi > 0:
                params["figure_width"] = canvas.width() / dpi
                params["figure_height"] = canvas.height() / dpi
        return params

    def _show_chart(self, pos: int) -> None:
        if self._fig is None:
            return
        frame_idx = self._frame_idx_at(pos)
        eta = self._eta_frames.get(frame_idx) if frame_idx is not None else None
        from openfcd.core.profile_composite import (
            build_profile_composite_context,
            render_profile_composite_context,
            resolve_body_polygon,
        )
        frame_name = self._frame_names.get(frame_idx) if frame_idx is not None else None
        body_polygon, body_source = resolve_body_polygon(self._annotation, frame_name)
        context = build_profile_composite_context(
            eta=eta,
            profile_line=self._get_line_for_pos(pos),
            viz_params=self._profile_viz_params(),
            frame_idx=frame_idx,
            frame_name=frame_name,
            run_id=getattr(self._spec, "run_id", None),
            body_polygon_rc=body_polygon,
            body_source=body_source,
            spatial_calibration=self._frame_calibrations.get(frame_idx),
        )
        render_profile_composite_context(
            context,
            fig=self._fig,
            frame_label=f"Profile Composite - frame {frame_idx}" if frame_idx is not None else None,
        )
        self._fig.canvas.draw_idle()

    def _enter_annotation(self) -> None:
        self._mode_stack.setCurrentIndex(1)
        self._btn_edit.setVisible(False)
        self._btn_done.setVisible(True)
        self._show_annotator_frame(self._current_frame_pos)

    def _show_annotator_frame(self, pos: int) -> None:
        frame_idx = self._frame_idx_at(pos)
        self._annotator.set_image(self._eta_frames.get(frame_idx) if frame_idx is not None else None)
        line = self._get_line_for_pos(pos)
        self._annotator.set_committed_line(line)

    def _finish_annotation(self) -> None:
        """Save the drawn line back to SceneSpec.profile_lines."""
        pos = self._current_frame_pos
        frame_idx = self._frame_idx_at(pos)
        saved = False
        if frame_idx is not None and self._spec is not None:
            line = self._annotator.current_line_rowcol()
            if line is not None:
                from openfcd.io.scene import ProfileLine
                if self._spec.profile_lines is None:
                    self._spec = self._spec.model_copy(update={"profile_lines": {}})
                updated_lines = dict(self._spec.profile_lines or {})
                updated_lines[frame_idx] = ProfileLine(p0=line[0], p1=line[1])
                self._spec = self._spec.model_copy(update={"profile_lines": updated_lines})
                self.scene_changed.emit(self._spec)
                saved = True
        if not saved:
            self._enter_annotation()
            return
        next_missing = self._first_missing_line_pos(pos + 1)
        if next_missing is not None:
            self._set_frame_pos(next_missing)
            self._enter_annotation()
            return
        self._mode_stack.setCurrentIndex(0)
        self._btn_done.setVisible(False)
        self._btn_edit.setVisible(True)
        self._show_chart(self._current_frame_pos)

    def apply_viz(self, params: dict) -> None:
        if self._spec is not None:
            self._spec = self._spec.model_copy(update={"viz_params": dict(params)})
        self._mode_stack.setCurrentIndex(0)
        self._btn_done.setVisible(False)
        self._btn_edit.setVisible(True)
        self._show_chart(self._current_frame_pos)

    def current_slider_value(self) -> int:
        return int(self._current_frame_pos)

    def set_preferred_frame_pos(self, pos: int | None) -> None:
        self._preferred_frame_pos = pos

    def export_context(self) -> dict:
        frame_idx = self._frame_idx_at(self._current_frame_pos)
        frame_name = self._frame_names.get(frame_idx) if frame_idx is not None else None
        from openfcd.core.profile_composite import resolve_body_polygon
        body_polygon, body_source = resolve_body_polygon(self._annotation, frame_name)
        return {
            "frame_idx": frame_idx,
            "frame_name": frame_name,
            "spatial_calibration": self._frame_calibrations.get(frame_idx),
            "profile_line": self._get_line_for_pos(self._current_frame_pos),
            "run_id": getattr(self._spec, "run_id", None),
            "body_polygon_rc": body_polygon,
            "body_source": body_source,
            "layout_mode": "export",
        }
