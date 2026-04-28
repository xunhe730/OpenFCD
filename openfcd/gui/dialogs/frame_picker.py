"""Frame selection dialog for Scene creation.

Shows a checkbox list of all available frames. Provides
Select All / Invert / Stride / Range toolbar buttons.
"""
from __future__ import annotations
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QWidget,
)
from PyQt6.QtCore import Qt
from openfcd.gui import tokens


class FramePickerDialog(QDialog):
    def __init__(
        self,
        parent=None,
        frames: list[Path] | list[str] | None = None,
        run_exists: bool = False,
        preselected: list[int] | None = None,
    ) -> None:
        super().__init__(parent)
        self._frames = [str(f) if isinstance(f, Path) else f for f in (frames or [])]
        self._run_exists = run_exists
        self._preselected = set(preselected or [])
        self.setWindowTitle("Select Frames")
        self.setMinimumSize(480, 400)
        self._setup_ui()
        self._update_count()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Toolbar
        bar = QHBoxLayout()
        for label, slot in [
            ("All", self._select_all),
            ("Invert", self._invert),
            ("Stride…", self._stride),
            ("Range…", self._range_select),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(26)
            btn.clicked.connect(slot)
            bar.addWidget(btn)
        bar.addStretch()
        layout.addLayout(bar)

        # List
        self._list = QListWidget()
        for i, name in enumerate(self._frames):
            item = QListWidgetItem(f"{i:4d}  {Path(name).name}")
            item.setCheckState(Qt.CheckState.Checked if i in self._preselected else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, i)
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
            f"padding:5px 16px;font-weight:600;"
        )
        ok.clicked.connect(self.accept)
        btn_row.addWidget(cancel)
        btn_row.addWidget(ok)
        layout.addLayout(btn_row)

    def _update_count(self) -> None:
        n = len(self.selected_indices())
        self._count_lbl.setText(f"{n} / {self._list.count()} selected")

    def _select_all(self) -> None:
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.CheckState.Checked)

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
        from PyQt6.QtWidgets import QInputDialog
        n, ok = QInputDialog.getInt(self, "Stride", "Select every N-th frame:", 5, 1, 9999)
        if not ok:
            return
        for i in range(self._list.count()):
            state = Qt.CheckState.Checked if i % n == 0 else Qt.CheckState.Unchecked
            self._list.item(i).setCheckState(state)

    def _range_select(self) -> None:
        from PyQt6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(self, "Range", "Enter range (e.g. 0-99 or 10-50):")
        if not ok or not text.strip():
            return
        try:
            parts = text.strip().split("-")
            lo, hi = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return
        for i in range(self._list.count()):
            state = Qt.CheckState.Checked if lo <= i <= hi else Qt.CheckState.Unchecked
            self._list.item(i).setCheckState(state)

    def selected_indices(self) -> list[int]:
        return [
            self._list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._list.count())
            if self._list.item(i).checkState() == Qt.CheckState.Checked
        ]
