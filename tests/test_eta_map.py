from __future__ import annotations

import numpy as np

from PyQt6.QtWidgets import QApplication

from openfcd.gui.scenes.eta_map import EtaMap


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_eta_map_uses_matplotlib_fallback_without_pyqtgraph() -> None:
    app = _app()
    widget = EtaMap()

    assert getattr(widget, "_mpl_canvas", None) is not None
    assert getattr(widget, "_fallback_label", None) is None


def test_eta_map_set_data_draws_on_fallback_canvas() -> None:
    app = _app()
    widget = EtaMap()

    data = np.arange(100, dtype=float).reshape(10, 10)
    widget.set_data(data)

    assert widget._mpl_axes is not None
    assert len(widget._mpl_axes.images) == 1
    rendered = widget._mpl_axes.images[0].get_array()
    assert rendered.shape == (10, 10)
