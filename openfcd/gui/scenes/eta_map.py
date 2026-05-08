"""η Map display widget — matplotlib FigureCanvasQTAgg."""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import colormaps

from openfcd.gui import tokens


class EtaMap(QWidget):
    """Full-canvas η heatmap with colorbar."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._fig = Figure()
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

    def clear(self, message: str | None = None) -> None:
        """Clear any previously rendered heatmap."""
        self._fig.clear()
        self._ax = self._fig.add_subplot(111)
        self._cb = None
        self._im = None
        self._ax.set_axis_off()
        if message:
            self._ax.text(
                0.5,
                0.5,
                message,
                transform=self._ax.transAxes,
                ha="center",
                va="center",
                color=tokens.TEXT_MUTED,
                fontsize=11,
            )
        self._apply_dark_style()
        self._canvas.draw_idle()

    def set_data(
        self,
        eta: np.ndarray | None,
        colormap: str = "RdBu_r",
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        colorbar: str = "right",
        crop_to_valid: bool = True,
    ) -> None:
        """Render eta array as heatmap with colorbar."""
        if eta is None or eta.size == 0:
            self.clear("No data")
            return
        data = np.asarray(eta, dtype=np.float64)
        if crop_to_valid:
            data = self._crop_to_valid(data)

        self._fig.clear()
        if colorbar == "bottom":
            gs = self._fig.add_gridspec(
                2, 1,
                height_ratios=[1.0, 0.045],
                hspace=0.08,
                left=0.04,
                right=0.98,
                bottom=0.08,
                top=0.98,
            )
            self._ax = self._fig.add_subplot(gs[0, 0])
            cax = self._fig.add_subplot(gs[1, 0])
        elif colorbar == "right":
            gs = self._fig.add_gridspec(
                1, 2,
                width_ratios=[1.0, 0.045],
                wspace=0.06,
                left=0.04,
                right=0.98,
                bottom=0.04,
                top=0.98,
            )
            self._ax = self._fig.add_subplot(gs[0, 0])
            cax = self._fig.add_subplot(gs[0, 1])
        else:
            self._ax = self._fig.add_axes([0.04, 0.04, 0.94, 0.94])
            cax = None
        self._cb = None

        # Symmetric colormap for bidirectional data
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            if vmax is None:
                vmax = float(np.nanpercentile(np.abs(finite), 98))
            if vmin is None:
                vmin = -float(vmax) if colormap in ("RdBu_r", "RdBu", "bwr", "seismic") else float(np.nanpercentile(finite, 2))
        else:
            vmin = -1.0 if vmin is None else vmin
            vmax = 1.0 if vmax is None else vmax

        cmap = colormaps.get_cmap(colormap).copy()
        cmap.set_bad((0, 0, 0, 0))
        self._im = self._ax.imshow(
            data, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        if cax is not None:
            orientation = "horizontal" if colorbar == "bottom" else "vertical"
            self._cb = self._fig.colorbar(self._im, cax=cax, orientation=orientation)
        self._ax.set_axis_off()
        self._apply_dark_style()
        self._canvas.draw_idle()  # non-blocking, no reentrance

    @staticmethod
    def _crop_to_valid(eta: np.ndarray) -> np.ndarray:
        valid = np.isfinite(eta)
        if not valid.any():
            return eta
        rows = np.flatnonzero(valid.any(axis=1))
        cols = np.flatnonzero(valid.any(axis=0))
        r0 = max(0, int(rows[0]) - 1)
        r1 = min(eta.shape[0], int(rows[-1]) + 2)
        c0 = max(0, int(cols[0]) - 1)
        c1 = min(eta.shape[1], int(cols[-1]) + 2)
        return eta[r0:r1, c0:c1]
