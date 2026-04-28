"""η Map display widget — matplotlib FigureCanvasQTAgg."""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from openfcd.gui import tokens


class EtaMap(QWidget):
    """Full-canvas η heatmap with colorbar."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._fig = Figure(tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._canvas, 1)

        self._im = None
        self._cb = None
        self._apply_dark_style()

    def _apply_dark_style(self) -> None:
        try:
            bg = tokens.BG_PRIMARY.lstrip("#")
            r = int(bg[0:2], 16) / 255
            g = int(bg[2:4], 16) / 255
            b = int(bg[4:6], 16) / 255
            self._fig.patch.set_facecolor((r, g, b))
            self._ax.set_facecolor((r, g, b))
        except Exception:
            pass

    def set_data(self, eta: np.ndarray, colormap: str = "RdBu_r") -> None:
        """Render eta array as heatmap with colorbar."""
        if eta is None or eta.size == 0:
            return
        self._ax.cla()
        if self._cb is not None:
            try:
                self._cb.remove()
            except Exception:
                pass
            self._cb = None
        # Symmetric colormap for bidirectional data
        finite = eta[np.isfinite(eta)]
        vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size > 0 else 1.0
        vmin = -vmax if colormap in ("RdBu_r", "RdBu", "bwr", "seismic") else 0.0
        self._im = self._ax.imshow(
            eta, cmap=colormap, aspect="auto", vmin=vmin, vmax=vmax
        )
        self._cb = self._fig.colorbar(self._im, ax=self._ax, fraction=0.03, pad=0.02)
        self._ax.set_axis_off()
        self._apply_dark_style()
        self._canvas.draw_idle()  # non-blocking, no reentrance
