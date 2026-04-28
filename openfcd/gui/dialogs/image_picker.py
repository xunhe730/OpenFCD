"""Image picker dialog with thumbnail grid for frame import."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QFont, QFontMetrics
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QWidget, QGridLayout, QInputDialog,
)

from openfcd.gui import tokens


class ThumbnailLoader(QThread):
    """Lazy thumbnail loader — emits QImage (safe in non-main thread)."""

    image_ready = pyqtSignal(int, object)  # (index, QImage | None)

    def __init__(self, frames: list[Path], icon_size: QSize) -> None:
        super().__init__()
        self._frames = frames
        self._icon_size = icon_size
        self._cancelled = False

    def run(self) -> None:
        for idx, fpath in enumerate(self._frames):
            if self._cancelled:
                break
            try:
                image = QImage(str(fpath))
                if not image.isNull():
                    scaled = image.scaled(
                        self._icon_size,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    self.image_ready.emit(idx, scaled)
                else:
                    self.image_ready.emit(idx, None)
            except Exception:
                self.image_ready.emit(idx, None)

    def cancel(self) -> None:
        self._cancelled = True


class PickerThumbCell(QWidget):
    """Single thumbnail cell with checkbox, pixmap, and filename label."""

    toggled = pyqtSignal(int, bool)  # (frame_idx, is_checked)

    def __init__(
        self,
        frame_idx: int,
        filename: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.frame_idx = frame_idx
        self.filename = filename
        self._checked = True
        self._pixmap: QPixmap | None = None
        self._icon_w, self._icon_h = 160, 120
        self._cell_w, self._cell_h = 180, 160
        self._bg_color = QColor(40, 40, 40)

        self.setFixedSize(self._cell_w, self._cell_h)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    @property
    def checked(self) -> bool:
        return self._checked

    @checked.setter
    def checked(self, v: bool) -> None:
        self._checked = v
        self.update()

    def set_pixmap(self, qimage: QImage | None) -> None:
        if qimage is None or qimage.isNull():
            return
        # Letterbox on main thread using QPainter
        canvas = QPixmap(self._icon_w, self._icon_h)
        canvas.fill(self._bg_color)
        x = (self._icon_w - qimage.width()) // 2
        y = (self._icon_h - qimage.height()) // 2
        p = QPainter(canvas)
        p.drawImage(x, y, qimage)
        p.end()
        self._pixmap = canvas
        self.update()

    def mousePressEvent(self, event) -> None:
        self._checked = not self._checked
        self.toggled.emit(self.frame_idx, self._checked)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self._cell_w, self._cell_h

        # Cell background + border
        border_color = (
            QColor(tokens.ACCENT_CLAY) if self._checked else QColor(tokens.BORDER_SUBTLE)
        )
        p.setPen(QPen(border_color, 2.0 if self._checked else 1.0))
        p.setBrush(QColor(tokens.BG_TERTIARY))
        p.drawRoundedRect(0, 0, w - 1, h - 1, 4, 4)

        # Thumbnail area (160×120, centred horizontally with 10px margin)
        margin = 10
        thumb_x = margin
        thumb_y = margin
        thumb_w = self._icon_w
        thumb_h = self._icon_h

        # Dark background for thumbnail area
        p.fillRect(thumb_x, thumb_y, thumb_w, thumb_h, self._bg_color)

        # Draw pixmap if available
        if self._pixmap and not self._pixmap.isNull():
            p.drawPixmap(thumb_x, thumb_y, self._pixmap)

        # Checkbox overlay — top-left of thumbnail (hand-painted, always visible)
        cb_x, cb_y, cb_sz = thumb_x + 4, thumb_y + 4, 16
        if self._checked:
            p.setBrush(QColor(tokens.ACCENT_CLAY))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(cb_x, cb_y, cb_sz, cb_sz, 3, 3)
            # Checkmark lines
            p.setPen(QPen(QColor("#ffffff"), 2.0))
            p.drawLine(cb_x + 3, cb_y + 8, cb_x + 6, cb_y + 12)
            p.drawLine(cb_x + 6, cb_y + 12, cb_x + 13, cb_y + 4)
        else:
            p.setBrush(QColor(255, 255, 255, 200))
            p.setPen(QPen(QColor(tokens.BORDER_STRONG), 1.2))
            p.drawRoundedRect(cb_x, cb_y, cb_sz, cb_sz, 3, 3)

        # Filename label — below thumbnail, elided to fit
        label_y = thumb_y + thumb_h + 6
        font = QFont(tokens.FONT_MONO.split(",")[0].strip(), 9)
        p.setFont(font)
        fm = QFontMetrics(font)
        elided = fm.elidedText(
            self.filename,
            Qt.TextElideMode.ElideMiddle,
            self._cell_w - 20,
        )
        p.setPen(QColor(tokens.TEXT_PRIMARY))
        p.drawText(10, label_y + fm.ascent(), elided)

        p.end()


class ImagePickerDialog(QDialog):
    """Dialog with thumbnail grid for selecting frames to import.

    Args:
        parent: Optional parent widget.
        frames: List of image file paths to display.
    """

    def __init__(
        self,
        parent=None,
        frames: list[Path] | None = None,
    ) -> None:
        super().__init__(parent)
        self._frames = list(frames or [])
        self._cells: list[PickerThumbCell] = []
        self.setWindowTitle("Select Images to Import")
        self.setMinimumSize(800, 560)
        self._icon_w, self._icon_h = 160, 120
        self._worker: ThumbnailLoader | None = None
        self._setup_ui()
        self._start_loading()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Toolbar row
        bar = QHBoxLayout()
        for label, slot in [
            ("Select All", self._select_all),
            ("Deselect All", self._deselect_all),
            ("Invert", self._invert),
            ("Every Nth…", self._stride),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(26)
            btn.clicked.connect(slot)
            bar.addWidget(btn)
        bar.addStretch()
        layout.addLayout(bar)

        # Scroll area with grid of cells
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setContentsMargins(12, 8, 16, 8)
        self._grid_layout.setSpacing(8)

        self._scroll.setWidget(self._grid_container)
        layout.addWidget(self._scroll, 1)

        # Build cells — 4 columns
        COLS = 4
        for idx, fpath in enumerate(self._frames):
            cell = PickerThumbCell(idx, fpath.name, self._grid_container)
            cell.toggled.connect(lambda _idx, _checked: self._update_count())
            row, col = divmod(idx, COLS)
            self._grid_layout.addWidget(cell, row, col)
            self._cells.append(cell)

        # Count label
        self._count_lbl = QLabel()
        layout.addWidget(self._count_lbl)

        # OK / Cancel buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        ok = QPushButton("OK")
        ok.setStyleSheet(
            f"background:{tokens.ACCENT_CLAY};color:#fff;border-radius:4px;"
            "padding:5px 16px;font-weight:600;"
        )
        ok.clicked.connect(self.accept)
        btn_row.addWidget(cancel)
        btn_row.addWidget(ok)
        layout.addLayout(btn_row)

        self._update_count()

    def _start_loading(self) -> None:
        if not self._frames:
            return
        self._worker = ThumbnailLoader(self._frames, QSize(self._icon_w, self._icon_h))
        self._worker.image_ready.connect(self._on_thumbnail)
        self._worker.start()

    def _on_thumbnail(self, index: int, qimage: QImage | None) -> None:
        if 0 <= index < len(self._cells):
            self._cells[index].set_pixmap(qimage)

    def _update_count(self) -> None:
        n = len(self.selected_indices())
        self._count_lbl.setText(f"{n} / {len(self._frames)} selected")

    # ── Toolbar actions ──────────────────────────────────────────────
    def _select_all(self) -> None:
        for cell in self._cells:
            cell.checked = True
        self._update_count()

    def _deselect_all(self) -> None:
        for cell in self._cells:
            cell.checked = False
        self._update_count()

    def _invert(self) -> None:
        for cell in self._cells:
            cell.checked = not cell.checked
        self._update_count()

    def _stride(self) -> None:
        n, ok = QInputDialog.getInt(self, "Stride", "Select every N-th frame:", 1, 1, 9999)
        if not ok:
            return
        for cell in self._cells:
            cell.checked = cell.frame_idx % n == 0
        self._update_count()

    # ── Public API ───────────────────────────────────────────────────
    def selected_indices(self) -> list[int]:
        """Return sorted list of checked frame indices."""
        return [cell.frame_idx for cell in self._cells if cell.checked]

    @property
    def selected_frames(self) -> list[Path]:
        """Return list of checked frame paths."""
        return [self._frames[i] for i in self.selected_indices()]

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait()
        super().closeEvent(event)
