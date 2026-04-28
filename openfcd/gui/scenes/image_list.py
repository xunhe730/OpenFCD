"""Image List scene — thumbnail grid with selection, filtering, and badges.

Translates the v3 design bundle `image-list.jsx` to PyQt6.
Supports 4 visual states: idle (grid loaded), empty, filtered, running.
"""

from __future__ import annotations

import math
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QSize, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QPainter, QColor, QIcon, QPen, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QScrollArea, QGridLayout, QSizePolicy,
    QCheckBox, QFrame,
)

from openfcd.gui import tokens

if TYPE_CHECKING:
    pass


class ImageListState(Enum):
    IDLE = auto()
    EMPTY = auto()
    FILTERED = auto()
    RUNNING = auto()


from PyQt6.QtCore import QThread
from PyQt6.QtGui import QImageReader

class ThumbWorker(QThread):
    """Background thread to load thumbnails incrementally."""
    thumb_loaded = pyqtSignal(int, object)  # QImage; main thread converts to QPixmap

    def __init__(self, paths: list[Path], parent=None):
        super().__init__(parent)
        self.paths = paths
        self.running = True

    def run(self):
        for idx, path in enumerate(self.paths):
            if not self.running:
                break
            reader = QImageReader(str(path))
            reader.setScaledSize(QSize(140, 100))
            img = reader.read()
            if not img.isNull():
                self.thumb_loaded.emit(idx, img)  # emit QImage, not QPixmap
            self.msleep(1)

    def stop(self):
        self.running = False
        self.wait()


class ThumbCell(QWidget):
    """Single thumbnail cell with checkbox, badges, and frame label."""

    clicked = pyqtSignal(int, bool)  # frame_idx, is_selected

    BADGE_REF = "★"
    BADGE_ANCHOR = "⚑"
    BADGE_COMPUTED = "✓"

    def __init__(
        self,
        frame_idx: int,
        filename: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.frame_idx = frame_idx
        self.filename = filename
        self._selected = False
        self._is_ref = False
        self._is_anchor = False
        self._is_computed = False
        self._pixmap: QPixmap | None = None

        self.setFixedSize(140, 100)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ── State ───────────────────────────────────────────────────────
    @property
    def is_selected(self) -> bool:
        return self._selected

    @is_selected.setter
    def is_selected(self, v: bool) -> None:
        self._selected = v
        self.update()

    def set_badges(
        self, *, ref: bool = False, anchor: bool = False, computed: bool = False
    ) -> None:
        self._is_ref = ref
        self._is_anchor = anchor
        self._is_computed = computed
        self.update()

    def set_pixmap(self, pm: QPixmap) -> None:
        self._pixmap = pm
        self.update()

    # ── Events ──────────────────────────────────────────────────────
    def mousePressEvent(self, event) -> None:
        self._selected = not self._selected
        self.clicked.emit(self.frame_idx, self._selected)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Border
        border_color = QColor(tokens.ACCENT_CLAY) if self._selected else QColor(tokens.BORDER_SUBTLE)
        p.setPen(QPen(border_color, 1.5 if self._selected else 1.0))
        p.setBrush(QColor(tokens.BG_TERTIARY))
        p.drawRoundedRect(0, 0, w - 1, h - 1, 4, 4)

        # Thumbnail image area
        img_h = int(h * 0.65)
        p.save()
        p.setClipRect(1, 1, w - 2, img_h)
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                w - 2, img_h,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            x_off = (w - 2 - scaled.width()) // 2
            y_off = (img_h - scaled.height()) // 2
            p.drawPixmap(1 + x_off, 1 + y_off, scaled)
        else:
            p.fillRect(1, 1, w - 2, img_h, QColor("#000000"))
        p.restore()

        if not self._selected:
            # Dim unselected
            p.fillRect(1, 1, w - 2, img_h, QColor(0, 0, 0, 115))

        # Checkbox (top-left)
        cb_x, cb_y, cb_sz = 5, 5, 14
        if self._selected:
            p.setBrush(QColor(tokens.ACCENT_CLAY))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(cb_x, cb_y, cb_sz, cb_sz, 2, 2)
            p.setPen(QPen(QColor("#fff"), 1.6))
            p.drawLine(cb_x + 3, cb_y + 7, cb_x + 5, cb_y + 10)
            p.drawLine(cb_x + 5, cb_y + 10, cb_x + 10, cb_y + 5)
        else:
            p.setBrush(QColor(255, 255, 255, 190))
            p.setPen(QPen(QColor(tokens.BORDER_STRONG), 1.0))
            p.drawRoundedRect(cb_x, cb_y, cb_sz, cb_sz, 2, 2)

        # Badges (top-right)
        badge_x = w - 6
        badge_font = QFont(tokens.FONT_UI.split(",")[0], 10)
        p.setFont(badge_font)
        for badge, show in [
            (self.BADGE_REF, self._is_ref),
            (self.BADGE_ANCHOR, self._is_anchor),
        ]:
            if show:
                p.setPen(QColor(tokens.ACCENT_CLAY))
                badge_x -= 14
                p.drawText(badge_x, 16, badge)

        if self._is_computed:
            comp_w, comp_h = 16, 13
            comp_x = badge_x - comp_w - 2
            p.setBrush(QColor(tokens.SUCCESS))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(comp_x, 4, comp_w, comp_h, 2, 2)
            p.setPen(QColor("#fff"))
            comp_font = QFont(tokens.FONT_UI.split(",")[0], 7)
            comp_font.setBold(True)
            p.setFont(comp_font)
            p.drawText(comp_x, 4, comp_w, comp_h, Qt.AlignmentFlag.AlignCenter, "✓")

        # Frame label (bottom gradient)
        grad_h = 22
        for i in range(grad_h):
            alpha = int(200 * (i / grad_h))
            p.fillRect(1, h - grad_h + i - 1, w - 2, 1, QColor(0, 0, 0, alpha))

        p.setPen(QColor("#fff"))
        label_font = QFont(tokens.FONT_MONO.split(",")[0], 9)
        p.setFont(label_font)
        p.drawText(6, h - 6, self.filename)

        p.end()


class _ImageListToolbar(QWidget):
    """Top toolbar: breadcrumb + filter + sort + view mode + select-all."""

    filter_changed = pyqtSignal(str)
    sort_changed = pyqtSignal(str)
    select_mode_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(42)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(10)

        self.setObjectName("ImageListToolbar")
        
        # Breadcrumb
        self._breadcrumb = QLabel("No images loaded")
        layout.addWidget(self._breadcrumb)

        layout.addStretch()

        # Filter input
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Img*.jpg")
        self._filter.setFixedWidth(180)
        self._filter.textChanged.connect(self.filter_changed.emit)
        layout.addWidget(self._filter)

        # Sort
        self._sort = QComboBox()
        self._sort.addItems(["name ▾", "name ▴", "date ▾", "frame# ▾"])
        self._sort.setFixedWidth(90)
        self._sort.currentTextChanged.connect(self.sort_changed.emit)
        layout.addWidget(self._sort)

        # Select mode
        self._select_mode = QComboBox()
        self._select_mode.addItems(["All", "None", "Every 5th", "Every 10th", "Every 20th"])
        self._select_mode.setFixedWidth(110)
        self._select_mode.currentTextChanged.connect(self.select_mode_changed.emit)
        layout.addWidget(self._select_mode)
        
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        self.setStyleSheet(f"""
            #ImageListToolbar {{
                background: {tokens.BG_PRIMARY};
                border-bottom: 1px solid {tokens.BORDER_SUBTLE};
            }}
        """)
        self._breadcrumb.setStyleSheet(
            f"font-size: 12px; color: {tokens.TEXT_SECONDARY}; border: none;"
        )
        combo_style = f"""
            QComboBox, QLineEdit {{
                background: {tokens.BG_TERTIARY};
                border: 1px solid {tokens.BORDER_SUBTLE};
                border-radius: 5px;
                padding: 5px 8px;
                font-size: 12px;
                color: {tokens.TEXT_PRIMARY};
            }}
        """
        self._filter.setStyleSheet(combo_style)
        self._sort.setStyleSheet(combo_style)
        self._select_mode.setStyleSheet(combo_style)



    def set_breadcrumb(self, text: str, count: int) -> None:
        self._breadcrumb.setText(f"{text}  <span style='font-family: {tokens.FONT_MONO}; "
                                 f"font-size: 10.5px; color: {tokens.TEXT_MUTED}'>"
                                 f"{count} files</span>")

    def set_filter_text(self, text: str) -> None:
        self._filter.setText(text)


class _ImageListSummary(QWidget):
    """Bottom summary bar: selection count + estimate + actions."""

    invert_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(44)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(14)

        self.setObjectName("ImageListSummary")

        self._info = QLabel("Selection: 0 frames")
        layout.addWidget(self._info)

        layout.addStretch()

        self._invert_btn = QPushButton("Invert")
        self._invert_btn.clicked.connect(self.invert_clicked.emit)
        layout.addWidget(self._invert_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self.clear_clicked.emit)
        layout.addWidget(self._clear_btn)
        
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        self.setStyleSheet(f"""
            #ImageListSummary {{
                background: {tokens.BG_SECONDARY};
                border-top: 1px solid {tokens.BORDER_SUBTLE};
            }}
        """)
        self._info.setStyleSheet(f"font-size: 12px; color: {tokens.TEXT_PRIMARY}; border: none; background: transparent;")
        
        btn_style = (
            f"padding: 6px 12px; border-radius: 4px; font-size: 11.5px; "
            f"border: 1px solid {tokens.BORDER_SUBTLE}; "
            f"background: {tokens.BG_TERTIARY}; color: {tokens.TEXT_PRIMARY};"
        )
        self._invert_btn.setStyleSheet(btn_style)
        self._clear_btn.setStyleSheet(
            f"padding: 6px 12px; border-radius: 4px; font-size: 11.5px; "
            f"color: {tokens.TEXT_SECONDARY}; background: transparent; border: none;"
        )



    def update_info(self, selected: int, total: int, per_frame: float = 0.54) -> None:
        est = f"{selected * per_frame:.0f}s" if selected > 0 else "—"
        self._info.setText(
            f"<b>Selection:</b> "
            f"<span style='font-family: {tokens.FONT_MONO}'>{selected} frames</span>"
            f"<span style='color: {tokens.TEXT_MUTED}'> · </span>"
            f"<span style='font-family: {tokens.FONT_MONO}'>{per_frame}s/frame</span>"
            f"<span style='color: {tokens.TEXT_MUTED}'> · </span>"
            f"<span style='font-family: {tokens.FONT_MONO}'>est. {est} total</span>"
        )


class _EmptyDropZone(QWidget):
    """Central area shown when no images are loaded."""

    new_project_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.container = QFrame()
        self.container.setFixedSize(540, 280)
        
        inner = QVBoxLayout(self.container)
        inner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.setSpacing(14)

        self.title = QLabel("Drop images here")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(self.title)

        self.desc = QLabel("Supports .jpg .png .tif · recursive folder scan")
        self.desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(self.desc)

        btn = QPushButton("Create New Project…")
        btn.clicked.connect(self.new_project_requested.emit)
        inner.addWidget(btn, 0, Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.container)
        
        # Save btn ref for theming
        self.btn = btn
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        self.container.setStyleSheet(f"""
            QFrame {{
                border: 2px dashed {tokens.BORDER_STRONG};
                border-radius: 8px;
                background: {tokens.BG_TERTIARY};
            }}
        """)
        self.title.setStyleSheet(
            f"font-family: {tokens.FONT_SERIF}; font-size: 24px; "
            f"font-weight: 500; color: {tokens.TEXT_PRIMARY}; border: none;"
        )
        self.desc.setStyleSheet(f"font-size: 12px; color: {tokens.TEXT_MUTED}; border: none;")
        self.btn.setStyleSheet(
            f"margin-top: 6px; padding: 8px 16px; border-radius: 5px; "
            f"background: {tokens.ACCENT_CLAY}; color: #fff; "
            f"font-size: 12.5px; font-weight: 500; border: none;"
        )


class ImageList(QWidget):
    """Image List scene — the primary pre-processing view.

    Shows a thumbnail grid of experiment frames with selection, filtering,
    role badges (★ reference, ⚑ anchor, ✓ computed), and batch actions.
    """

    frame_selected = pyqtSignal(int)  # frame index
    selection_changed = pyqtSignal(list)  # list of selected frame indices
    new_project_requested = pyqtSignal()
    set_reference_requested = pyqtSignal(int)
    set_anchor_requested = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frames: list[Path] = []
        self._cells: list[ThumbCell] = []
        self._selected_indices: set[int] = set()
        self._ref_frame: int = -1
        self._anchor_frame: int = -1
        self._computed_until: int = -1
        self._state = ImageListState.EMPTY
        self._cols = 10

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        self._toolbar = _ImageListToolbar(self)
        self._toolbar.filter_changed.connect(self._on_filter)
        self._toolbar.select_mode_changed.connect(self._on_select_mode)
        self._toolbar.sort_changed.connect(self._on_sort)
        layout.addWidget(self._toolbar)

        # Central area — stacked: grid or empty
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidgetResizable(True)
        
        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setContentsMargins(14, 14, 24, 14)
        self._grid_layout.setSpacing(8)
        self._scroll.setWidget(self._grid_container)

        self._empty_zone = _EmptyDropZone(self)
        self._empty_zone.new_project_requested.connect(self.new_project_requested.emit)

        # Use a simple stacking approach
        self._content_stack = QVBoxLayout()
        self._content_stack.setContentsMargins(0, 0, 0, 0)

        content_wrapper = QWidget()
        content_layout = QVBoxLayout(content_wrapper)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self._scroll)
        content_layout.addWidget(self._empty_zone)

        layout.addWidget(content_wrapper, 1)

        # Summary bar
        self._summary = _ImageListSummary(self)
        self._summary.invert_clicked.connect(self._invert_selection)
        self._summary.clear_clicked.connect(self._clear_selection)
        layout.addWidget(self._summary)

        self._thumb_worker = None

        self._update_state(ImageListState.EMPTY)
        
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        self._scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background: {tokens.BG_PRIMARY};
            }}
            QScrollBar:vertical {{
                width: 8px;
                background: {tokens.BG_TERTIARY};
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {tokens.BORDER_STRONG};
                border-radius: 4px;
                min-height: 30px;
            }}
        """)
        # Trigger redraw of thumb cells (they grab color directly in paintEvent)
        for cell in self._cells:
            cell.update()

    # ── Public API ──────────────────────────────────────────────────
    def load_frames(self, frames_dir: str | Path, pattern: str = "Img*.jpg") -> None:
        """Scan a directory and populate the thumbnail grid."""
        from openfcd.io.image import scan_frames
        self._frames = scan_frames(frames_dir, pattern)
        if not self._frames:
            self._update_state(ImageListState.EMPTY)
            return

        # Update breadcrumb
        path = Path(frames_dir)
        self._toolbar.set_breadcrumb(str(path), len(self._frames))
        self._toolbar.set_filter_text(pattern)

        self._rebuild_grid()
        self._update_state(ImageListState.IDLE)

    def set_ref_frame(self, idx: int) -> None:
        self._ref_frame = idx
        self._refresh_badges()

    def set_anchor_frame(self, idx: int) -> None:
        self._anchor_frame = idx
        self._refresh_badges()

    def set_computed_until(self, idx: int) -> None:
        self._computed_until = idx
        self._refresh_badges()

    def set_running(self, running: bool) -> None:
        if running:
            self._update_state(ImageListState.RUNNING)
        else:
            self._update_state(ImageListState.IDLE)

    def selected_indices(self) -> list[int]:
        return sorted(self._selected_indices)

    # ── Internal ────────────────────────────────────────────────────
    def _rebuild_grid(self) -> None:
        # Clear existing
        if getattr(self, "_thumb_worker", None) is not None:
            self._thumb_worker.stop()
            self._thumb_worker = None

        for cell in self._cells:
            cell.setParent(None)
            cell.deleteLater()
        self._cells.clear()

        # Recalculate columns based on available width
        avail_w = self._scroll.viewport().width() - 38  # margins
        self._cols = max(4, avail_w // 148)

        for idx, frame_path in enumerate(self._frames):
            cell = ThumbCell(idx, frame_path.name, self._grid_container)
            cell.clicked.connect(self._on_cell_clicked)
            cell.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            cell.customContextMenuRequested.connect(lambda pos, c=cell: self._on_cell_context_menu(c, pos))

            row = idx // self._cols
            col = idx % self._cols
            self._grid_layout.addWidget(cell, row, col)
            self._cells.append(cell)

        self._refresh_badges()

        if self._frames:
            self._thumb_worker = ThumbWorker(self._frames, self)
            self._thumb_worker.thumb_loaded.connect(self._on_thumb_loaded)
            self._thumb_worker.start()

    def _on_thumb_loaded(self, idx: int, qimage) -> None:
        if 0 <= idx < len(self._cells) and self._cells[idx]:
            self._cells[idx].set_pixmap(QPixmap.fromImage(qimage))

    def _on_cell_context_menu(self, cell: ThumbCell, pos) -> None:
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)
        
        # Style the menu to match dark mode theme
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {tokens.BG_PRIMARY};
                color: {tokens.TEXT_PRIMARY};
                border: 1px solid {tokens.BORDER_SUBTLE};
            }}
            QMenu::item:selected {{ background-color: {tokens.BG_TERTIARY}; }}
        """)
        
        ref_action = menu.addAction("Set as Reference")
        anch_action = menu.addAction("Set as Anchor (GUI)")
        action = menu.exec(cell.mapToGlobal(pos))
        if action == ref_action:
            self.set_ref_frame(cell.frame_idx)
            self.set_reference_requested.emit(cell.frame_idx)
        elif action == anch_action:
            self.set_anchor_frame(cell.frame_idx)
            self.set_anchor_requested.emit(cell.frame_idx)

    def _refresh_badges(self) -> None:
        for cell in self._cells:
            cell.set_badges(
                ref=(cell.frame_idx == self._ref_frame),
                anchor=(cell.frame_idx == self._anchor_frame),
                computed=(cell.frame_idx <= self._computed_until),
            )
            cell.is_selected = cell.frame_idx in self._selected_indices

    def _update_state(self, state: ImageListState) -> None:
        self._state = state
        self._scroll.setVisible(state in (ImageListState.IDLE, ImageListState.RUNNING))
        self._empty_zone.setVisible(state == ImageListState.EMPTY)

    def _on_cell_clicked(self, idx: int, selected: bool) -> None:
        if selected:
            self._selected_indices.add(idx)
        else:
            self._selected_indices.discard(idx)
        self._summary.update_info(len(self._selected_indices), len(self._frames))
        self.frame_selected.emit(idx)
        self.selection_changed.emit(sorted(self._selected_indices))

    def _on_filter(self, text: str) -> None:
        if not text.strip():
            # Show all
            for cell in self._cells:
                cell.setVisible(True)
            if self._cells:
                self._update_state(ImageListState.IDLE)
            return

        import fnmatch
        count = 0
        for cell in self._cells:
            match = fnmatch.fnmatch(cell.filename, text)
            cell.setVisible(match)
            if match:
                count += 1

        if count == 0:
            self._update_state(ImageListState.FILTERED)
        else:
            self._update_state(ImageListState.IDLE)

    def _on_select_mode(self, mode: str) -> None:
        self._selected_indices.clear()
        if mode == "All":
            self._selected_indices = set(range(len(self._frames)))
        elif mode == "None":
            pass
        elif mode.startswith("Every"):
            try:
                n = int(mode.split()[1].rstrip("th").rstrip("st").rstrip("nd").rstrip("rd"))
            except (ValueError, IndexError):
                n = 10
            self._selected_indices = {i for i in range(0, len(self._frames), n)}

        self._refresh_badges()
        self._summary.update_info(len(self._selected_indices), len(self._frames))
        self.selection_changed.emit(sorted(self._selected_indices))

    def _on_sort(self, mode: str) -> None:
        if mode.startswith("name"):
            self._frames.sort(key=lambda p: p.name, reverse=("▴" in mode))
        elif mode.startswith("date"):
            self._frames.sort(key=lambda p: p.stat().st_mtime, reverse=("▴" in mode))
        self._rebuild_grid()

    def _invert_selection(self) -> None:
        all_indices = set(range(len(self._frames)))
        self._selected_indices = all_indices - self._selected_indices
        self._refresh_badges()
        self._summary.update_info(len(self._selected_indices), len(self._frames))
        self.selection_changed.emit(sorted(self._selected_indices))

    def _clear_selection(self) -> None:
        self._selected_indices.clear()
        self._refresh_badges()
        self._summary.update_info(0, len(self._frames))
        self.selection_changed.emit([])

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._frames and self._cells:
            # Rebuild grid on resize for responsive columns
            avail_w = self._scroll.viewport().width() - 38
            new_cols = max(4, avail_w // 148)
            if new_cols != self._cols:
                self._rebuild_grid()
