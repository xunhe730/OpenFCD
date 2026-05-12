"""Dialog showing per-segment peak-trough height bar chart + table.

Triggered from the Wave Stats table when the user double-clicks the
``Heights`` cell. Pure-presentation: read-only, no signals back into the
session.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class WaveStatsDetailDialog(QDialog):
    """Modal dialog with a peak-trough height bar chart and a numeric table."""

    def __init__(
        self,
        segment_label: str,
        heights: np.ndarray | list,
        color: str = "#1f77b4",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Wave Heights — {segment_label}")
        self.resize(520, 520)

        arr = np.asarray(heights, dtype=np.float64).reshape(-1)
        self._heights = arr
        self._color = color or "#1f77b4"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel(
            f"Peak–trough heights ({len(arr)} pairs) for <b>{segment_label}</b>"
        )
        layout.addWidget(title)

        # Bar chart (matplotlib)
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

            fig = Figure(figsize=(5.0, 2.6), tight_layout=True)
            ax = fig.add_subplot(111)
            if arr.size:
                xs = np.arange(1, arr.size + 1)
                ax.bar(xs, arr, color=self._color, edgecolor="0.2", linewidth=0.6)
                ax.set_xticks(xs)
                ax.set_xlabel("Peak index")
                ax.set_ylabel("|η_peak − η_trough| (mm)")
            else:
                ax.text(
                    0.5, 0.5, "No peak–trough pairs detected",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=11,
                )
                ax.set_axis_off()
            canvas = FigureCanvasQTAgg(fig)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            layout.addWidget(canvas, 3)
        except Exception:
            layout.addWidget(QLabel("matplotlib not available"))

        # Table
        self._table = QTableWidget(arr.size, 2, self)
        self._table.setHorizontalHeaderLabels(["#", "height (mm)"])
        self._table.verticalHeader().setVisible(False)
        for row, val in enumerate(arr.tolist()):
            i_item = QTableWidgetItem(str(row + 1))
            i_item.setFlags(i_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, i_item)
            v_item = QTableWidgetItem(f"{val:.4f}")
            v_item.setFlags(v_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 1, v_item)
        layout.addWidget(self._table, 2)

        # Buttons
        buttons = QHBoxLayout()
        self._btn_copy = QPushButton("Copy to Clipboard")
        self._btn_copy.clicked.connect(self._on_copy)
        buttons.addWidget(self._btn_copy)
        buttons.addStretch()
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        box.accepted.connect(self.accept)
        buttons.addWidget(box)
        layout.addLayout(buttons)

    def _on_copy(self) -> None:
        lines = ["index\theight_mm"]
        for i, v in enumerate(self._heights.tolist(), start=1):
            lines.append(f"{i}\t{v:.6f}")
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText("\n".join(lines))
