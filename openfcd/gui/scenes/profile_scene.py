"""Profile Scene view: per-frame line annotation + composite preview + wave-segment table (v2).

Render state machine (Tier 1c): the **only** full-figure repaint entry is
``_request_render`` → debounced ``QTimer`` → ``_do_render`` → ``__show_chart``.
``__show_chart`` is name-mangled to discourage direct external calls; a permanent
``assert self._render_state == "rendering"`` at its entry plus the AST-grep
whitelist test (see ``tests/test_overlay_whitelist_ast_grep.py``) enforce the
invariant. Overlay-only paths (``_recompute_live_stats``) must not call
``fig.clear``/``add_axes``/``add_subplot`` nor construct new Artists; they may
only mutate existing artists via ``set_data``/``set_xy``/``set_offsets``/
``set_array``.
"""
from __future__ import annotations

import collections
import logging
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QPoint, QRectF, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from openfcd.gui import tokens

__all__ = ["ProfileSceneView"]

_logger = logging.getLogger(__name__)


# Local color cycle for wave-segment auto-assignment. Kept here (not imported
# from controllers/) to avoid scenes→controllers upward dependency.
_COLOR_CYCLE: tuple[str, ...] = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)


# ── Module-level pure coordinate helpers ────────────────────────────────────


def widget_to_rowcol(
    pos: QPoint,
    image_shape: tuple[int, int],
    image_rect: QRectF,
) -> tuple[float, float] | None:
    """Convert widget pixel pos to (row, col) image coordinates."""
    h, w = image_shape
    if not image_rect.contains(float(pos.x()), float(pos.y())):
        return None
    col = (pos.x() - image_rect.left()) / image_rect.width() * (w - 1)
    row = (pos.y() - image_rect.top()) / image_rect.height() * (h - 1)
    return (
        float(np.clip(row, 0, h - 1)),
        float(np.clip(col, 0, w - 1)),
    )


def rowcol_to_widget(
    point: tuple[float, float],
    image_shape: tuple[int, int],
    image_rect: QRectF,
) -> QPoint:
    """Convert (row, col) image coordinates to widget pixel pos."""
    h, w = image_shape
    row, col = point
    x = image_rect.left() + (col / max(1, w - 1)) * image_rect.width()
    y = image_rect.top() + (row / max(1, h - 1)) * image_rect.height()
    return QPoint(int(round(x)), int(round(y)))


# ── _LineAnnotator (unchanged: profile-line draw widget) ────────────────────


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
        if self._image is None or self._image_rect.isEmpty():
            return None
        return widget_to_rowcol(pos, self._image.shape[:2], self._image_rect)

    def _rowcol_to_widget(self, point: tuple[float, float]) -> QPoint:
        if self._image is None or self._image_rect.isEmpty():
            return QPoint(int(point[1]), int(point[0]))
        return rowcol_to_widget(point, self._image.shape[:2], self._image_rect)

    def set_committed_line(self, line: tuple | None) -> None:
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
            painter.setBrush(QColor(tokens.ACCENT_CLAY))
            for p in line:
                painter.drawEllipse(p, 5, 5)


# ── ProfileSceneView ────────────────────────────────────────────────────────


class ProfileSceneView(QWidget):
    """Profile scene: composite figure + wave-segment table + segment-collect interaction."""

    scene_changed = pyqtSignal(object)
    profile_line_changed = pyqtSignal(object)       # ProfileLineData | None
    wave_stats_config_changed = pyqtSignal(object)  # WaveStatsConfig
    wave_segment_added = pyqtSignal(object)         # WaveSegment
    wave_segment_edited = pyqtSignal(int, object)   # (idx, WaveSegment)
    wave_segment_deleted = pyqtSignal(int)
    live_stats_changed = pyqtSignal(object)         # FrameWaveStats | None

    # Debounce window for the render coalescer. 48 ms = 3 frames @60 Hz /
    # 6 frames @120 Hz ProMotion. See draft-plan.md §Principle 2.
    _RENDER_DEBOUNCE_MS: int = 48

    def __init__(self, parent=None, session_controller=None) -> None:
        super().__init__(parent)
        # When supplied, the scene borrows the live in-memory annotation and
        # ProjectModel from SessionController instead of re-reading them from
        # disk on every load(). This keeps in-memory edits (profile_line,
        # wave_stats segments) alive across scene switches. Test fixtures and
        # legacy callers pass None and fall back to the disk-read path.
        self._session = session_controller
        self._spec = None
        self._project_path: Path | None = None
        self._eta_frames: dict[int, np.ndarray] = {}
        self._frame_names: dict[int, str] = {}
        self._frame_calibrations: dict[int, object] = {}
        self._annotation = None
        self._project_model = None
        self._run_manifest: dict | None = None
        self._current_frame_pos = 0
        self._preferred_frame_pos: int | None = None
        self._annotating_per_frame: bool = False
        self._active_segment_idx: int | None = None

        # Matplotlib state: composite figure references and hover overlay artists
        self._fig = None
        self._composite_axes: dict | None = None  # {"map", "profile"} after render
        self._lower_ax = None                     # the 1D profile axis (segment-collect target)
        self._hover_vline = None
        self._hover_text = None
        self._segment_patches: list = []
        self._peak_artists: list = []                # scatter artists for peaks/troughs
        self._pending_first_x: float | None = None
        self._live_frame_stats = None                # FrameWaveStats | None
        # ── Render state machine (Tier 1c) ──────────────────────────
        # State machine collapses same-frame multi-entry races into a single
        # debounced repaint. Threading contract: ``_request_render``,
        # ``_do_render``, and the ``_render_timer`` QTimer slot are ALL
        # invoked on the Qt GUI main thread (Qt single-threaded GUI
        # invariant); the ``deque`` therefore needs no lock. Background
        # ``_RunWorker`` QThreads must hop back to the main thread via
        # ``Qt.QueuedConnection`` signals before reaching ``_request_render``.
        self._render_state: str = "idle"  # ∈ {"idle", "scheduled", "rendering"}
        self._pending_render_dirty: bool = False
        self._render_dirty_rearm_count: int = 0
        self._render_state_log: collections.deque = collections.deque(maxlen=64)

        # ── Top-level layout ────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header: title + Edit Lines / Done + Collect Segment + status label
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
        self._btn_collect = QPushButton("采集波段")
        self._btn_collect.setFixedHeight(24)
        self._btn_collect.setCheckable(True)
        self._btn_collect.toggled.connect(self._on_collect_toggled)
        hl.addWidget(self._btn_collect)
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(f"color: {tokens.TEXT_SECONDARY}; font-size: 11px;")
        self._status_lbl.setVisible(False)
        hl.addWidget(self._status_lbl)
        layout.addWidget(header)

        # ── Body: QStackedWidget — page 0 = composite+table splitter, page 1 = _LineAnnotator
        self._mode_stack = QStackedWidget()
        layout.addWidget(self._mode_stack, 1)

        # Page 0: splitter (top = composite figure, bottom = wave-segment table).
        # The whole splitter is the "chart widget" page for stack-currentWidget tests.
        self._chart_widget = QSplitter(Qt.Orientation.Vertical)
        self._chart_widget.setChildrenCollapsible(False)
        self._splitter = self._chart_widget   # alias for clarity
        self._fig_panel = self._make_chart_widget()
        self._chart_widget.addWidget(self._fig_panel)

        # Table panel
        from openfcd.gui.panels.wave_stats_table import WaveStatsTablePanel
        self._table_panel = WaveStatsTablePanel()
        self._table_panel.segmentClicked.connect(self._on_table_row_clicked)
        self._table_panel.segmentEdited.connect(self._on_table_segment_edited)
        self._table_panel.segmentDeleted.connect(self._on_table_segment_deleted)
        self._table_panel.visibilityToggled.connect(self._on_table_visibility_toggled)
        self._table_panel.copyPreviousFrameRequested.connect(
            self._on_copy_previous_frame_segments
        )
        self.live_stats_changed.connect(self._table_panel.set_live_stats)
        self._chart_widget.addWidget(self._table_panel)
        self._chart_widget.setStretchFactor(0, 3)
        self._chart_widget.setStretchFactor(1, 1)

        self._mode_stack.addWidget(self._chart_widget)  # 0

        # Page 1: line annotator (Edit Lines mode)
        self._annotator = _LineAnnotator()
        self._mode_stack.addWidget(self._annotator)  # 1

        # Slider row
        slider_row = QWidget()
        sr = QHBoxLayout(slider_row)
        sr.setContentsMargins(12, 4, 12, 4)
        self._slider_lbl = QLabel("frame 0/0")
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(self._RENDER_DEBOUNCE_MS)
        self._pending_idx = 0
        self._render_timer.timeout.connect(self._do_render)
        self._slider.valueChanged.connect(self._on_slider_moved)
        sr.addWidget(self._slider_lbl)
        sr.addWidget(self._slider, 1)
        layout.addWidget(slider_row)

    # ── chart widget ──────────────────────────────────────────────────

    def _make_chart_widget(self) -> QWidget:
        container = QWidget()
        vl = QVBoxLayout(container)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)
        try:
            from matplotlib.figure import Figure
            from openfcd.gui.widgets.mpl_canvas import CompactCanvas

            fig = Figure(figsize=(8.6, 4.2))
            self._ax = fig.add_subplot(111)
            canvas = CompactCanvas(fig)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            canvas.setMinimumHeight(400)
            canvas.updateGeometry()
            self._fig = fig

            # Wire matplotlib event handlers for hover + segment-collect clicks
            try:
                canvas.mpl_connect("motion_notify_event", self._on_mpl_motion)
                canvas.mpl_connect("button_press_event", self._on_mpl_click)
            except Exception:
                pass

            vl.addWidget(canvas, 1)
            return container
        except ImportError:
            self._ax = None
            self._fig = None
            lbl = QLabel("matplotlib not available")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            vl.addWidget(lbl)
            return container

    # ── lifecycle ─────────────────────────────────────────────────────

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
        self._table_panel.set_annotation(self._annotation)
        self._table_panel.set_h5_path(self._h5_path(), batch="default")
        self._sync_table_current_frame()
        missing_pos = self._first_missing_line_pos()
        if missing_pos is not None:
            self._set_frame_pos(missing_pos)
            self._enter_annotation()
        else:
            self._request_render(self._current_frame_pos)

    def _load_eta_frames(self) -> None:
        self._eta_frames = {}
        self._frame_names = {}
        self._frame_calibrations = {}
        spec = self._spec
        if spec is None or not self._project_path:
            # No project bound — clear everything (session_controller may
            # also be None or empty, but binding to its property is still
            # safe since it returns None when its store is None).
            self._annotation = self._session.annotation if self._session else None
            self._project_model = self._session.project if self._session else None
            self._run_manifest = None
            return
        # Annotation + project: prefer the live in-memory copy from
        # SessionController (the canonical owner — see
        # `.omc/plans/wave-stats-scene-switch-rootcause.md`). Falls back to
        # an own read-only FileSessionStore only when no session_controller
        # was injected (test fixtures, legacy callers).
        if self._session is not None:
            self._annotation = self._session.annotation
            self._project_model = self._session.project
        else:
            self._annotation = None
            self._project_model = None
            try:
                from openfcd.io.store import FileSessionStore
                session = FileSessionStore.open(self._project_path, read_only=True)
                try:
                    self._annotation = session.annotation
                    self._project_model = session.project
                finally:
                    session.close()
            except Exception:
                pass
        # Run manifest: runs are immutable snapshots — always disk-read.
        self._run_manifest = None
        try:
            from openfcd.io.store import FileSessionStore
            session = FileSessionStore.open(self._project_path, read_only=True)
            try:
                run_id_for_manifest = spec.run_id or self._latest_run_id(self._project_path)
                self._run_manifest = next(
                    (r for r in session.list_runs() if r.get("run_id") == run_id_for_manifest),
                    None,
                )
            finally:
                session.close()
        except Exception:
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

    def _on_slider_moved(self, idx: int) -> None:
        # Label updates immediately (cheap, no figure work); the heavy
        # full-figure repaint is coalesced through ``_request_render``.
        self._current_frame_pos = idx
        total = len(self._spec.frame_indices) if self._spec is not None else 0
        self._slider_lbl.setText(f"frame {idx}/{max(0, total - 1)}")
        self._request_render(idx)

    def _do_render(self) -> None:
        """QTimer slot: drive ``scheduled → rendering → idle`` transition.

        Page-1 (annotator) takes the lightweight path and does NOT enter
        ``__show_chart``; page-0 transitions to ``rendering`` and calls
        ``__show_chart``, which manages its own ``finally`` state cleanup.
        """
        pending = self._pending_idx
        if self._mode_stack.currentIndex() != 0:
            # Annotator page: no figure work; just refresh the image.
            prev = self._render_state
            self._render_state = "idle"
            self._pending_render_dirty = False
            self._render_dirty_rearm_count = 0
            self._render_state_log.append(
                (time.monotonic(), prev, "idle", "do_render_page1_annotator")
            )
            self._show_annotator_frame(pending)
            return
        # Page-0 chart path: transition into ``rendering`` and dispatch.
        prev = self._render_state
        self._render_state = "rendering"
        self._render_state_log.append(
            (time.monotonic(), prev, "rendering", "do_render_dispatch")
        )
        self.__show_chart(pending)

    def _on_slider(self, pos: int) -> None:
        self._current_frame_pos = pos
        total = len(self._spec.frame_indices) if self._spec is not None else 0
        self._slider_lbl.setText(f"frame {pos}/{max(0, total - 1)}")
        self._request_render(pos)

    def _set_frame_pos(self, pos: int) -> None:
        self._current_frame_pos = pos
        total = len(self._spec.frame_indices) if self._spec is not None else 0
        self._slider_lbl.setText(f"frame {pos}/{max(0, total - 1)}")
        if self._slider.value() != pos:
            self._slider.setValue(pos)
        # Reset segment-collect mode so it does not bleed across frames.
        if hasattr(self, "_btn_collect") and self._btn_collect.isChecked():
            self._btn_collect.setChecked(False)
        # Notify the table panel of the active frame so it renders the
        # segment list for the new frame (per-frame wave-stats, v3).
        self._sync_table_current_frame()

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        # Coalesce rapid resize storms (window drag, splitter move, table-row
        # add/remove that nudges the canvas height) into a single deferred
        # render via the state machine. Page-1 (annotator) is skipped because
        # the chart figure is not visible there.
        if getattr(self, "_fig", None) is not None and self._mode_stack.currentIndex() == 0:
            self._request_render(self._current_frame_pos)

    # ── Render state machine: single full-figure repaint entry ───────

    def _request_render(self, pos: int) -> None:
        """Single full-figure repaint entry (state-machine driver).

        Threading contract: must be called on the Qt GUI main thread.
        Background workers must hop via ``Qt.QueuedConnection`` signal.
        """
        prev = self._render_state
        if self._render_state == "idle":
            self._pending_idx = pos
            self._render_state = "scheduled"
            self._render_state_log.append(
                (time.monotonic(), prev, "scheduled", "request_render_idle")
            )
            self._render_timer.start()
        elif self._render_state == "scheduled":
            # Coalesce: keep the timer running, just refresh the target pos.
            self._pending_idx = pos
            self._render_state_log.append(
                (time.monotonic(), prev, "scheduled", "request_render_coalesce")
            )
        else:  # "rendering"
            self._pending_idx = pos
            self._pending_render_dirty = True
            self._render_state_log.append(
                (time.monotonic(), prev, "rendering", "request_render_dirty")
            )

    def _frame_idx_at(self, pos: int) -> int | None:
        if self._spec and 0 <= pos < len(self._spec.frame_indices):
            return self._spec.frame_indices[pos]
        return None

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
        # In preview mode the renderer reads fig.get_size_inches() (Qt-synced
        # from the canvas widget) and never writes it back, so the agg buffer
        # always matches the on-screen widget rect. Don't inject figure_width/
        # figure_height — that path was the root cause of the noisy uninited
        # buffer leaking through when canvas size and figure size diverged.
        params["layout_mode"] = "preview"
        return params

    def __show_chart(self, pos: int) -> None:
        """Full-figure repaint. **Permanent invariant**: must be entered in
        ``rendering`` state via ``_do_render`` (or the rearm-cap final
        consume); name-mangled (``_ProfileSceneView__show_chart``) so any
        external call site immediately surfaces as ``AttributeError``.

        MUST NOT call any API that spins the Qt event loop — no
        ``QApplication.processEvents``, no ``QEventLoop.exec``, no modal
        dialogs. Spinning the loop here re-enters the state machine while
        the figure is half-built and reintroduces the very race this
        machine exists to suppress.
        """
        # Permanent invariant — kept in release builds (do NOT gate with
        # ``__debug__``); state contract is the last line of defence
        # against bypass routes that escape the AST-grep whitelist.
        assert self._render_state == "rendering"
        # Backbuffer-flush root-cause fix (orthogonal to the state machine):
        # explicitly clear the figure and drain pending agg events so the
        # next add_axes pass renders into a clean buffer. Without this,
        # axes rebuilt after a resize can leak old pixels at the next
        # paintEvent even when entry races are serialised.
        if self._fig is not None:
            self._fig.clear()
            try:
                self._fig.canvas.flush_events()
            except Exception:
                pass
        try:
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
            # Reset overlay refs (matplotlib clears axes on render).
            self._lower_ax = self._detect_lower_axis()
            self._hover_vline = None
            self._hover_text = None
            self._segment_patches = []
            self._peak_artists = []
            # Resync the table panel's current frame so its row list reflects
            # this frame's segments — slider handlers only update
            # ``_current_frame_pos``, the canonical frame-change point is
            # the debounced render.
            self._sync_table_current_frame()
            # Single recompute path: this also redraws the segment/peak overlay
            # on the freshly-built axes and schedules a canvas repaint.
            self._recompute_live_stats()
            # After fig.clear()+add_axes(), force a synchronous draw so the agg
            # buffer is committed at the current canvas size before Qt's next
            # paintEvent — otherwise a stale buffer can leak through as
            # uninited pixels around the new axes.
            self._fig.canvas.draw()
        finally:
            # Transition out of ``rendering``. If a ``_request_render`` came
            # in mid-flight (dirty=True), rearm the timer (state→scheduled)
            # up to ``_render_dirty_rearm_count < 3``; on cap, consume the
            # final pending idx ONCE more and then force idle so we don't
            # silently drop the user's most recent intent.
            if self._pending_render_dirty:
                self._render_dirty_rearm_count += 1
                if self._render_dirty_rearm_count >= 3:
                    final_pos = self._pending_idx
                    self._pending_render_dirty = False
                    self._render_state_log.append(
                        (time.monotonic(), "rendering", "rendering", "rearm_cap_final_consume")
                    )
                    # State remains "rendering" so the recursive call's
                    # entry assert holds. The inner call's ``finally``
                    # sees ``dirty=False`` and falls through to idle; we
                    # then overwrite with our cap-finalised idle below.
                    try:
                        self.__show_chart(final_pos)
                    except Exception:
                        _logger.exception(
                            "rearm cap final consume failed at idx=%s", final_pos
                        )
                    _logger.error(
                        "render rearm cap reached at idx=%s; final frame may lag, "
                        "retrigger via slider/resize",
                        final_pos,
                    )
                    self._render_state = "idle"
                    self._render_dirty_rearm_count = 0
                    self._render_state_log.append(
                        (time.monotonic(), "rendering", "idle", "rearm_cap_finalized")
                    )
                else:
                    self._pending_render_dirty = False
                    self._render_state = "scheduled"
                    self._render_state_log.append(
                        (time.monotonic(), "rendering", "scheduled", "dirty_rearm")
                    )
                    self._render_timer.start()
            else:
                self._render_state = "idle"
                self._render_dirty_rearm_count = 0
                self._render_state_log.append(
                    (time.monotonic(), "rendering", "idle", "render_complete")
                )

    def _detect_lower_axis(self):
        """Detect which axis is the 1D η(x) subplot.

        Uses the bottom-most axis on the figure as a heuristic.
        """
        if self._fig is None:
            return None
        axes = self._fig.get_axes()
        if not axes:
            return None
        # Pick the axis with the smallest y0 (= lowest on screen)
        return min(axes, key=lambda a: a.get_position().y0)

    # ── Edit Lines flow (unchanged behaviour) ─────────────────────────

    def _enter_annotation(self) -> None:
        # Switching to the annotator page hides the chart figure entirely.
        # Cancel any in-flight chart render bookkeeping so a stale rearm
        # cannot fire while page-1 is active.
        self._render_timer.stop()
        prev_state = self._render_state
        self._render_state = "idle"
        self._pending_render_dirty = False
        self._render_dirty_rearm_count = 0
        self._render_state_log.append(
            (time.monotonic(), prev_state, "idle", "enter_annotation")
        )
        self._annotating_per_frame = True
        self._mode_stack.setCurrentIndex(1)
        self._btn_edit.setVisible(False)
        self._btn_done.setVisible(True)
        self._show_annotator_frame(self._current_frame_pos)

    def _show_annotator_frame(self, pos: int) -> None:
        frame_idx = self._frame_idx_at(pos)
        self._annotator.set_image(self._eta_frames.get(frame_idx) if frame_idx is not None else None)
        if self._annotating_per_frame:
            line = self._get_line_for_pos(pos)
            self._annotator.set_committed_line(line)
        else:
            ann_line = None
            if self._annotation is not None and self._annotation.profile_line is not None:
                pl = self._annotation.profile_line
                ann_line = (pl.start, pl.end)
            self._annotator.set_committed_line(ann_line)

    def _finish_annotation(self) -> None:
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
                # Also push to annotation.profile_line so wave_stats picks it up
                from openfcd.io.annotation import ProfileLineData
                new_pl = ProfileLineData(start=line[0], end=line[1])
                self.profile_line_changed.emit(new_pl)
                # Route through SessionController so the canonical
                # in-memory annotation survives scene re-entry. Falls back
                # to a local model_copy for test/CLI paths without a
                # SessionController. The legacy ``parent.set_profile_line``
                # branch was dead — ``self.parent()`` is the QStackedWidget,
                # which has no such method.
                self._commit_profile_line(new_pl)
                saved = True
        if not saved:
            self._enter_annotation()
            return
        next_missing = self._first_missing_line_pos(pos + 1)
        if next_missing is not None:
            self._set_frame_pos(next_missing)
            self._enter_annotation()
            return
        self._annotating_per_frame = False
        self._mode_stack.setCurrentIndex(0)
        self._btn_done.setVisible(False)
        self._btn_edit.setVisible(True)
        self._request_render(self._current_frame_pos)

    def apply_viz(self, params: dict) -> None:
        if self._spec is not None:
            self._spec = self._spec.model_copy(update={"viz_params": dict(params)})
        self._annotating_per_frame = False
        self._mode_stack.setCurrentIndex(0)
        self._btn_done.setVisible(False)
        self._btn_edit.setVisible(True)
        self._request_render(self._current_frame_pos)

    def current_slider_value(self) -> int:
        return int(self._current_frame_pos)

    def set_preferred_frame_pos(self, pos: int | None) -> None:
        self._preferred_frame_pos = pos

    # ── Segment-collect interaction ───────────────────────────────────

    def _on_collect_toggled(self, checked: bool) -> None:
        if checked:
            self._pending_first_x = None
            self._status_lbl.setText("点击 1D 子图选第一个端点")
            self._status_lbl.setVisible(True)
        else:
            self._pending_first_x = None
            self._status_lbl.setVisible(False)

    def _on_mpl_motion(self, event) -> None:
        """Hover cursor: vline + (x, η) text on the 1D subplot only."""
        if self._fig is None:
            return
        if self._lower_ax is None or event.inaxes is not self._lower_ax:
            if self._hover_vline is not None:
                try:
                    self._hover_vline.set_visible(False)
                except Exception:
                    pass
            if self._hover_text is not None:
                try:
                    self._hover_text.set_visible(False)
                except Exception:
                    pass
            self._fig.canvas.draw_idle()
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._hover_vline is None:
            try:
                self._hover_vline = self._lower_ax.axvline(
                    event.xdata, color="0.4", linestyle="--", linewidth=0.8, zorder=10
                )
            except Exception:
                self._hover_vline = None
        else:
            try:
                self._hover_vline.set_xdata([event.xdata, event.xdata])
                self._hover_vline.set_visible(True)
            except Exception:
                pass
        label = f"x={event.xdata:.2f} mm  η={event.ydata:.3f} mm"
        if self._hover_text is None:
            try:
                self._hover_text = self._lower_ax.text(
                    0.02, 0.95, label,
                    transform=self._lower_ax.transAxes,
                    ha="left", va="top",
                    fontsize=8,
                    color="0.2",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.85),
                    zorder=11,
                )
            except Exception:
                self._hover_text = None
        else:
            try:
                self._hover_text.set_text(label)
                self._hover_text.set_visible(True)
            except Exception:
                pass
        self._fig.canvas.draw_idle()

    def _on_mpl_click(self, event) -> None:
        """Two-click segment-collect flow on the 1D subplot."""
        if not self._btn_collect.isChecked():
            return
        if self._lower_ax is None or event.inaxes is not self._lower_ax:
            return
        if event.button != 1 or event.xdata is None:
            return
        if self._pending_first_x is None:
            self._pending_first_x = float(event.xdata)
            self._status_lbl.setText(
                f"已选第一点 x={self._pending_first_x:.2f} mm；再点一次确定第二端点"
            )
            return
        x2 = float(event.xdata)
        x1 = float(self._pending_first_x)
        self._pending_first_x = None
        s_lo, s_hi = (x1, x2) if x1 < x2 else (x2, x1)
        if s_hi - s_lo < 1e-3:
            self._status_lbl.setText("段太短，已忽略")
            self._btn_collect.setChecked(False)
            return
        # Emit new-segment signal; let the parent controller append to annotation
        from openfcd.io.annotation import WaveSegment
        new_seg = WaveSegment(s_lo_mm=s_lo, s_hi_mm=s_hi)
        self.wave_segment_added.emit(new_seg)
        # Local-cache update so the table refresh has the new segment immediately
        self._append_segment_local(new_seg)
        self._btn_collect.setChecked(False)
        self._status_lbl.setText(
            f"已采集 [{s_lo:.2f}, {s_hi:.2f}] mm"
        )

    def _current_frame_key(self) -> str | None:
        """Return ``str(frame_idx)`` for the active slider position, or None."""
        frame_idx = self._frame_idx_at(self._current_frame_pos)
        if frame_idx is None:
            return None
        return str(frame_idx)

    def _sync_table_current_frame(self) -> None:
        """Push the active frame_idx to the table panel."""
        frame_idx = self._frame_idx_at(self._current_frame_pos)
        if hasattr(self._table_panel, "set_current_frame"):
            self._table_panel.set_current_frame(frame_idx)
        # Refresh "copy previous frame" button enabled state.
        if hasattr(self._table_panel, "set_previous_frame_has_segments"):
            self._table_panel.set_previous_frame_has_segments(
                self._previous_frame_has_segments()
            )

    def _previous_frame_has_segments(self) -> bool:
        if self._spec is None or self._annotation is None:
            return False
        ws = self._annotation.wave_stats
        if ws is None:
            return False
        pos = self._current_frame_pos
        if pos <= 0:
            return False
        prev_idx = self._frame_idx_at(pos - 1)
        if prev_idx is None:
            return False
        return bool(ws.segments_for_frame(prev_idx))

    def _commit_wave_stats(self, new_cfg) -> None:
        """Route a WaveStatsConfig update through SessionController when
        available — its in-place ``_store.annotation.wave_stats = config``
        keeps the canonical shared AnnotationSchema in sync across the app
        (table, other scenes, eventual flush to disk). Fallback path keeps
        a local model_copy for tests and CLI replay callers that don't
        wire a SessionController.

        Always re-binds ``self._annotation`` to the canonical live
        annotation so that subsequent scene re-entries borrowing
        ``self._session.annotation`` observe the mutation.
        """
        if self._session is not None:
            self._session.update_wave_stats_config(new_cfg)
            self._annotation = self._session.annotation
        elif self._annotation is not None:
            self._annotation = self._annotation.model_copy(update={"wave_stats": new_cfg})

    def _commit_profile_line(self, new_pl) -> None:
        """Route a ProfileLineData update through SessionController. See
        ``_commit_wave_stats`` for the rationale.
        """
        if self._session is not None:
            self._session.set_profile_line(new_pl)
            self._annotation = self._session.annotation
        elif self._annotation is not None:
            self._annotation = self._annotation.model_copy(update={"profile_line": new_pl})

    def _append_segment_local(self, new_seg) -> None:
        """Append ``new_seg`` to the CURRENT frame's segment list."""
        if self._annotation is None:
            return
        frame_key = self._current_frame_key()
        if frame_key is None:
            return
        from openfcd.io.annotation import WaveStatsConfig
        old_cfg = self._annotation.wave_stats
        existing = list(old_cfg.segments_for_frame(frame_key)) if old_cfg else []
        # Auto-assign next color from cycle if caller passed the default.
        # Pick the first cycle entry not currently in use among existing
        # segments — using len(existing) as the index regressed after a
        # deletion and collided with the surviving segment's color.
        if new_seg.color == "#1f77b4":
            used = {s.color for s in existing}
            chosen = next(
                (c for c in _COLOR_CYCLE if c not in used),
                _COLOR_CYCLE[len(existing) % len(_COLOR_CYCLE)],
            )
            new_seg = new_seg.model_copy(update={"color": chosen})
        if old_cfg is None:
            new_cfg = WaveStatsConfig(segments_by_frame={frame_key: [new_seg]})
        else:
            new_by_frame = dict(old_cfg.segments_by_frame)
            new_by_frame[frame_key] = existing + [new_seg]
            new_cfg = old_cfg.model_copy(update={"segments_by_frame": new_by_frame})
        self._commit_wave_stats(new_cfg)
        self.wave_stats_config_changed.emit(new_cfg)
        self._table_panel.set_annotation(self._annotation)
        self._sync_table_current_frame()
        # _recompute_live_stats handles overlay refresh + canvas redraw.
        self._recompute_live_stats()

    def _on_copy_previous_frame_segments(self) -> None:
        """Copy the previous frame's segments to the current frame."""
        if self._annotation is None or self._annotation.wave_stats is None:
            return
        if self._spec is None:
            return
        pos = self._current_frame_pos
        if pos <= 0:
            return
        prev_idx = self._frame_idx_at(pos - 1)
        cur_idx = self._frame_idx_at(pos)
        if prev_idx is None or cur_idx is None:
            return
        old_cfg = self._annotation.wave_stats
        prev_segs = old_cfg.segments_for_frame(prev_idx)
        if not prev_segs:
            return
        # Re-instantiate frozen WaveSegment objects to keep frames decoupled.
        new_segs = [seg.model_copy() for seg in prev_segs]
        new_by_frame = dict(old_cfg.segments_by_frame)
        new_by_frame[str(cur_idx)] = new_segs
        new_cfg = old_cfg.model_copy(update={"segments_by_frame": new_by_frame})
        self._commit_wave_stats(new_cfg)
        self.wave_stats_config_changed.emit(new_cfg)
        self._table_panel.set_annotation(self._annotation)
        self._sync_table_current_frame()
        self._recompute_live_stats()

    # ── Live wave-stats computation ───────────────────────────────────

    def _resolve_px_per_mm(self) -> float | None:
        """Resolve px_per_mm for the active frame, falling back to spec viz."""
        frame_idx = self._frame_idx_at(self._current_frame_pos)
        cal = self._frame_calibrations.get(frame_idx) if frame_idx is not None else None
        if cal is not None:
            px = getattr(cal, "pixel_per_mm", None)
            if px is not None and float(px) > 0:
                return float(px)
        params = self._spec.viz_params if self._spec is not None else None
        if isinstance(params, dict):
            val = params.get("px_per_mm")
            try:
                if val is not None and float(val) > 0:
                    return float(val)
            except (TypeError, ValueError):
                pass
        return None

    def _resolve_body_polygon_rc(self):
        """Return a (N, 2) body polygon array in (row, col) or None."""
        frame_idx = self._frame_idx_at(self._current_frame_pos)
        frame_name = self._frame_names.get(frame_idx) if frame_idx is not None else None
        ann = self._annotation
        if ann is None:
            return None
        if frame_name and frame_name in getattr(ann, "frame_polygons", {}):
            polys = ann.frame_polygons[frame_name]
            if polys:
                return polys[0].as_rc_array()
        if ann.polygons:
            return ann.polygons[0].as_rc_array()
        return None

    def _compute_live_stats(self):
        """Run compute_frame_wave_stats for the current frame + visible segments.

        Returns the FrameWaveStats result, or ``None`` if any required input is
        missing or the underlying compute fails.
        """
        ann = self._annotation
        # Profile line may live in annotation.profile_line (newly drawn) or in
        # spec.profile_lines per-frame dict (loaded from an existing project).
        # __show_chart sources from _get_line_for_pos; mirror that here so the
        # two pipelines see the same line.
        pl_endpoints = self._get_line_for_pos(self._current_frame_pos)
        if pl_endpoints is None and ann is not None and ann.profile_line is not None:
            pl_endpoints = (tuple(ann.profile_line.start), tuple(ann.profile_line.end))
        if ann is None or ann.wave_stats is None or pl_endpoints is None:
            return None
        frame_idx = self._frame_idx_at(self._current_frame_pos)
        if frame_idx is None:
            return None
        segs = ann.wave_stats.segments_for_frame(frame_idx)
        if not segs:
            return None
        eta = self._eta_frames.get(frame_idx) if frame_idx is not None else None
        if eta is None:
            return None
        try:
            from openfcd.core.wave_stats import compute_frame_wave_stats
            return compute_frame_wave_stats(
                eta=np.asarray(eta, dtype=np.float64),
                profile_line=pl_endpoints,
                body_polygon_rc=self._resolve_body_polygon_rc(),
                px_per_mm=float(self._resolve_px_per_mm() or 1.0),
                segments=[(float(s.s_lo_mm), float(s.s_hi_mm)) for s in segs],
                prominence_k=float(ann.wave_stats.peak_prominence_k),
            )
        except Exception:
            return None

    def _recompute_live_stats(self) -> None:
        """Recompute live stats and synchronise table + 1D-subplot overlay.

        Single entry point for any mutation that affects wave-stats output
        (segment add/edit/delete, visibility toggle, frame switch). Always
        refreshes the peak/trough overlay so it cannot drift out of sync with
        the cached ``self._live_frame_stats``.
        """
        stats = self._compute_live_stats()
        self._live_frame_stats = stats
        if hasattr(self._table_panel, "set_live_stats"):
            self._table_panel.set_live_stats(stats)
        self.live_stats_changed.emit(stats)
        self._overlay_segments(self._active_segment_idx)
        if self._fig is not None:
            self._fig.canvas.draw_idle()

    # ── Segment overlay on 1D subplot ─────────────────────────────────

    def _overlay_segments(self, active_idx: int | None) -> None:
        """Draw semi-transparent rectangle overlays on the 1D η(x) subplot.

        ``active_idx`` is rendered at higher alpha than the others.  Overlays
        are pure UI: ``export_profile_composite`` clears them before saving.
        """
        # Remove existing overlays
        for patch in self._segment_patches:
            try:
                patch.remove()
            except Exception:
                pass
        self._segment_patches = []
        for art in self._peak_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._peak_artists = []

        if self._lower_ax is None:
            return
        if self._annotation is None or self._annotation.wave_stats is None:
            return
        frame_idx = self._frame_idx_at(self._current_frame_pos)
        if frame_idx is None:
            return
        segs = self._annotation.wave_stats.segments_for_frame(frame_idx)
        if not segs:
            return

        y0, y1 = self._lower_ax.get_ylim()
        for idx, seg in enumerate(segs):
            if not seg.visible:
                continue
            alpha = 0.35 if idx == active_idx else 0.15
            try:
                patch = self._lower_ax.axvspan(
                    seg.s_lo_mm, seg.s_hi_mm,
                    facecolor=seg.color, alpha=alpha,
                    edgecolor=seg.color if idx == active_idx else "none",
                    linewidth=1.2 if idx == active_idx else 0.0,
                    zorder=5,
                )
                self._segment_patches.append(patch)
            except Exception:
                pass

        # Overlay peaks (circles) and troughs (triangles) from the live stats
        stats = self._live_frame_stats
        if stats is not None:
            for seg_stats in stats.segments:
                idx = int(seg_stats.segment_idx)
                if idx >= len(segs) or not segs[idx].visible:
                    continue
                color = segs[idx].color
                try:
                    if seg_stats.peaks_s_mm.size:
                        a = self._lower_ax.scatter(
                            seg_stats.peaks_s_mm, seg_stats.peaks_eta,
                            s=36, marker="o",
                            facecolors=color, edgecolors="black",
                            linewidths=0.6, alpha=0.85, zorder=8,
                        )
                        self._peak_artists.append(a)
                    if seg_stats.troughs_s_mm.size:
                        b = self._lower_ax.scatter(
                            seg_stats.troughs_s_mm, seg_stats.troughs_eta,
                            s=36, marker="v",
                            facecolors=color, edgecolors="black",
                            linewidths=0.6, alpha=0.85, zorder=8,
                        )
                        self._peak_artists.append(b)
                except Exception:
                    pass

        # Restore y-limits in case axvspan changed them
        try:
            self._lower_ax.set_ylim(y0, y1)
        except Exception:
            pass

    # ── Table-event handlers ──────────────────────────────────────────

    def _on_table_row_clicked(self, idx: int) -> None:
        self._active_segment_idx = idx
        self._overlay_segments(idx)
        if self._fig is not None:
            self._fig.canvas.draw_idle()

    def _on_table_segment_edited(self, idx: int, new_seg) -> None:
        if self._annotation is None or self._annotation.wave_stats is None:
            return
        frame_key = self._current_frame_key()
        if frame_key is None:
            return
        old_cfg = self._annotation.wave_stats
        cur_segs = list(old_cfg.segments_for_frame(frame_key))
        if idx >= len(cur_segs):
            return
        cur_segs[idx] = new_seg
        new_by_frame = dict(old_cfg.segments_by_frame)
        new_by_frame[frame_key] = cur_segs
        new_cfg = old_cfg.model_copy(update={"segments_by_frame": new_by_frame})
        self._commit_wave_stats(new_cfg)
        self.wave_segment_edited.emit(idx, new_seg)
        self.wave_stats_config_changed.emit(new_cfg)
        self._table_panel.set_annotation(self._annotation)
        self._sync_table_current_frame()
        self._recompute_live_stats()

    def _on_table_segment_deleted(self, idx: int) -> None:
        if self._annotation is None or self._annotation.wave_stats is None:
            return
        frame_key = self._current_frame_key()
        if frame_key is None:
            return
        old_cfg = self._annotation.wave_stats
        cur_segs = list(old_cfg.segments_for_frame(frame_key))
        if idx >= len(cur_segs):
            return
        cur_segs.pop(idx)
        new_by_frame = dict(old_cfg.segments_by_frame)
        new_by_frame[frame_key] = cur_segs
        new_cfg = old_cfg.model_copy(update={"segments_by_frame": new_by_frame})
        self._commit_wave_stats(new_cfg)
        self.wave_segment_deleted.emit(idx)
        self.wave_stats_config_changed.emit(new_cfg)
        if self._active_segment_idx is not None and self._active_segment_idx >= len(cur_segs):
            self._active_segment_idx = None
        self._table_panel.set_annotation(self._annotation)
        self._sync_table_current_frame()
        self._recompute_live_stats()

    def _on_table_visibility_toggled(self, idx: int, visible: bool) -> None:
        # Data mutation arrives separately via wave_stats_table's dual emit
        # (visibilityToggled + segmentEdited → _on_table_segment_edited). Here
        # we just rerun the live recompute so peaks/troughs of the (now-hidden
        # or now-visible) segment are added/removed from the 1D subplot.
        self._recompute_live_stats()

    # ── Misc public API ───────────────────────────────────────────────

    def refresh(self) -> None:
        if self._mode_stack.currentIndex() == 0:
            self._request_render(self._current_frame_pos)
        self._table_panel.set_h5_path(self._h5_path(), batch="default")
        self._table_panel.set_annotation(self._annotation)
        self._sync_table_current_frame()

    def _h5_path(self) -> Path | None:
        if self._project_path is None or self._spec is None:
            return None
        run_id = (
            getattr(self._spec, "run_id", None)
            or self._latest_run_id(self._project_path)
        )
        if run_id is None:
            return None
        return self._project_path / "runs" / run_id / "results.h5"

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

    def export_profile_composite(self, path: Path | str, dpi: int = 150) -> None:
        """Export the composite figure without segment overlays."""
        if self._fig is None:
            return
        # Clear overlays + hover artifacts
        prior_active = self._active_segment_idx
        self._overlay_segments(None)
        if self._hover_vline is not None:
            try:
                self._hover_vline.set_visible(False)
            except Exception:
                pass
        if self._hover_text is not None:
            try:
                self._hover_text.set_visible(False)
            except Exception:
                pass
        try:
            self._fig.canvas.draw()
            self._fig.savefig(str(path), dpi=int(dpi))
        finally:
            self._overlay_segments(prior_active)
            if self._fig is not None:
                self._fig.canvas.draw_idle()
