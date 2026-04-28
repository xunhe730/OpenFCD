"""Eta Map scene with pyqtgraph primary rendering and matplotlib fallback."""

from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget, QSizePolicy

from openfcd.gui import tokens

try:
    import pyqtgraph as pg

    HAS_PYQTGRAPH = True
except ImportError:
    pg = None
    HAS_PYQTGRAPH = False

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    FigureCanvasQTAgg = None
    Figure = None
    HAS_MATPLOTLIB = False


class EtaMap(QWidget):
    """Scene widget for eta heatmap."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image_item = None
        self._mpl_canvas = None
        self._mpl_axes = None
        self._mpl_colorbar = None
        self._fallback_label = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._title = QLabel("η Map")
        self._title.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {tokens.TEXT_PRIMARY}; "
            f"padding: 8px 16px; background: {tokens.BG_SECONDARY}; "
            f"border-bottom: 1px solid {tokens.BORDER_SUBTLE};"
        )
        layout.addWidget(self._title)

        if HAS_PYQTGRAPH:
            self._view = pg.ImageView()
            self._image_item = self._view.getImageItem()
            layout.addWidget(self._view)
            return

        if HAS_MATPLOTLIB:
            figure = Figure(figsize=(6, 4), tight_layout=True)
            self._mpl_axes = figure.add_subplot(111)
            self._mpl_axes.set_title("η heatmap")
            self._mpl_canvas = FigureCanvasQTAgg(figure)
            layout.addWidget(self._mpl_canvas)
            return

        self._fallback_label = QLabel(
            "No heatmap backend available.\nInstall pyqtgraph or matplotlib."
        )
        self._fallback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fallback_label.setStyleSheet(
            f"color: {tokens.TEXT_MUTED}; font-size: 12px; padding: 40px;"
        )
        layout.addWidget(self._fallback_label)

    def set_data(self, data, colormap: str = "RdBu_r") -> None:
        arr = np.asarray(data)

        if HAS_PYQTGRAPH and hasattr(self, "_view"):
            self._view.setImage(arr.T)
            return

        if HAS_MATPLOTLIB and self._mpl_axes is not None and self._mpl_canvas is not None:
            self._mpl_axes.clear()
            if arr.ndim < 2 or arr.size == 0:
                self._mpl_canvas.draw_idle()
                return
            image = self._mpl_axes.imshow(arr, cmap=colormap, origin="upper")
            self._mpl_axes.set_title("η heatmap")
            self._mpl_axes.set_xlabel("x [px]")
            self._mpl_axes.set_ylabel("y [px]")

            if self._mpl_colorbar is not None:
                self._mpl_colorbar.remove()
            self._mpl_colorbar = self._mpl_canvas.figure.colorbar(
                image, ax=self._mpl_axes, label="η [mm]"
            )
            self._mpl_canvas.draw_idle()
            return

        if self._fallback_label is not None:
            self._fallback_label.setText(
                f"η Map  shape={arr.shape}\n(install pyqtgraph or matplotlib for rendering)"
            )
