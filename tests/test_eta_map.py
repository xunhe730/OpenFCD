from __future__ import annotations

import numpy as np

from PyQt6.QtWidgets import QApplication

from openfcd.gui.scenes.eta_map import EtaMap


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_eta_map_uses_matplotlib_only() -> None:
    app = _app()
    widget = EtaMap()

    # matplotlib-only: _canvas and _ax must exist, no pyqtgraph
    assert getattr(widget, "_canvas", None) is not None
    assert getattr(widget, "_ax", None) is not None
    assert getattr(widget, "_fig", None) is not None


def test_eta_map_set_data_renders_heatmap() -> None:
    app = _app()
    widget = EtaMap()

    data = np.arange(100, dtype=float).reshape(10, 10)
    widget.set_data(data)

    assert widget._ax is not None
    assert len(widget._ax.images) == 1
    rendered = widget._ax.images[0].get_array()
    assert rendered.shape == (10, 10)


def test_eta_map_crops_full_frame_nan_canvas() -> None:
    app = _app()
    widget = EtaMap()

    data = np.full((20, 30), np.nan)
    data[5:15, 8:18] = 1.0
    widget.set_data(data, colorbar="none")

    rendered = widget._ax.images[0].get_array()
    assert rendered.shape == (12, 12)
    assert widget._cb is None
