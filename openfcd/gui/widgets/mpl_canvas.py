"""Matplotlib canvas with a small minimumSizeHint to prevent
figsize × dpi from inflating its parent window's sizeHint."""
from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PyQt6.QtCore import QSize


class CompactCanvas(FigureCanvasQTAgg):
    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(200, 150)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(400, 300)
