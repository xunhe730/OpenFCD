"""Image picker dialog with thumbnail grid for frame import."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QPixmap, QImage
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QInputDialog,
)

from openfcd.gui import tokens


class ThumbnailLoader(QThread):
    """Lazy thumbnail loader — emits QImage (safe in non-main thread)."""

    # Emit QImage; main thread converts to QPixmap/QIcon
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
        self.setWindowTitle("Select Images to Import")
        self.setMinimumSize(640, 480)
        self._worker: ThumbnailLoader | None = None
        self._setup_ui()
        self._start_loading()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Toolbar
        bar = QHBoxLayout()
        for label, slot in [
            ("Select All", self._select_all),
            ("Deselect All", self._deselect_all),
            ("Invert", self._invert),
            ("Every Nth\u2026", self._stride),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(26)
            btn.clicked.connect(slot)
            bar.addWidget(btn)
        bar.addStretch()
        layout.addLayout(bar)

        # Thumbnail grid
        self._list = QListWidget()
        self._list.setViewMode(QListWidget.ViewMode.IconMode)
        self._list.setIconSize(QSize(120, 90))
        self._list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._list.setUniformItemSizes(True)
        self._list.setSpacing(4)

        for fpath in self._frames:
            item = QListWidgetItem(fpath.name)
            item.setCheckState(Qt.CheckState.Checked)
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            )
            self._list.addItem(item)

        self._list.itemChanged.connect(lambda _: self._update_count())
        layout.addWidget(self._list)

        # Count label
        self._count_lbl = QLabel()
        layout.addWidget(self._count_lbl)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        ok = QPushButton("OK")
        ok.setStyleSheet(
            f"background:{tokens.ACCENT_CLAY};color:#fff;border-radius:4px;"
            f"padding:5px 16px;font-weight:600;",
        )
        ok.clicked.connect(self.accept)
        btn_row.addWidget(cancel)
        btn_row.addWidget(ok)
        layout.addLayout(btn_row)

        self._update_count()

    def _start_loading(self) -> None:
        self._worker = ThumbnailLoader(self._frames, QSize(120, 90))
        self._worker.image_ready.connect(self._on_thumbnail)
        self._worker.start()

    def _on_thumbnail(self, index: int, qimage) -> None:
        # QPixmap/QIcon must be created in the main thread — do it here
        if qimage is None:
            return
        item = self._list.item(index)
        if item is not None:
            pixmap = QPixmap.fromImage(qimage)
            item.setIcon(QIcon(pixmap))

    def _update_count(self) -> None:
        n = len(self.selected_indices())
        self._count_lbl.setText(f"{n} / {self._list.count()} selected")

    def _select_all(self) -> None:
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.CheckState.Checked)

    def _deselect_all(self) -> None:
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.CheckState.Unchecked)

    def _invert(self) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            state = (
                Qt.CheckState.Unchecked
                if item.checkState() == Qt.CheckState.Checked
                else Qt.CheckState.Checked
            )
            item.setCheckState(state)

    def _stride(self) -> None:
        n, ok = QInputDialog.getInt(self, "Stride", "Select every N-th frame:", 1, 1, 9999)
        if not ok:
            return
        for i in range(self._list.count()):
            state = Qt.CheckState.Checked if i % n == 0 else Qt.CheckState.Unchecked
            self._list.item(i).setCheckState(state)

    def selected_indices(self) -> list[int]:
        return [
            i
            for i in range(self._list.count())
            if self._list.item(i).checkState() == Qt.CheckState.Checked
        ]

    @property
    def selected_frames(self) -> list[Path]:
        """Return list of checked frame paths."""
        return [self._frames[i] for i in self.selected_indices()]

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait()
        super().closeEvent(event)
