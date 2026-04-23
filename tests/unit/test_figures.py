"""Unit tests for openfcd.core.figures."""
import pytest
from typing import get_protocol_members, runtime_checkable
from typing import Protocol

from openfcd.core.figures import FigureRenderer, FIGURE_IDS, RENDERERS, COMPOSITE_LAYOUTS


# ---------------------------------------------------------------------------
# 7. FIGURE_IDS contains 5 expected IDs
# ---------------------------------------------------------------------------

def test_figure_ids_count():
    assert len(FIGURE_IDS) == 5


def test_figure_ids_contents():
    expected = {
        "eta_heatmap",
        "wavelength_profile",
        "rms_map",
        "sample_grid",
        "eta_timeseries",
    }
    assert FIGURE_IDS == expected


def test_figure_ids_immutable():
    with pytest.raises((AttributeError, TypeError)):
        FIGURE_IDS.add("new_id")  # type: ignore


# ---------------------------------------------------------------------------
# 8. FigureRenderer is a Protocol
# ---------------------------------------------------------------------------

def test_figure_renderer_is_protocol():
    assert issubclass(FigureRenderer, Protocol)


def test_figure_renderer_has_render_method():
    import inspect
    members = FigureRenderer.__protocol_attrs__ if hasattr(FigureRenderer, '__protocol_attrs__') else dir(FigureRenderer)
    assert "render" in members


def test_figure_renderer_has_figure_id():
    members = FigureRenderer.__protocol_attrs__ if hasattr(FigureRenderer, '__protocol_attrs__') else dir(FigureRenderer)
    assert "figure_id" in members


def test_renderers_is_dict():
    assert isinstance(RENDERERS, dict)


def test_composite_layouts():
    assert "1x2" in COMPOSITE_LAYOUTS
    assert "2x2" in COMPOSITE_LAYOUTS
