from typing import Protocol, TYPE_CHECKING
import matplotlib.figure

if TYPE_CHECKING:
    import matplotlib.axes

FIGURE_IDS = frozenset({
    "eta_heatmap",
    "wavelength_profile",
    "rms_map",
    "sample_grid",
    "eta_timeseries",
})

COMPOSITE_LAYOUTS = frozenset({"1x2", "2x1", "2x2", "free"})


class FigureRenderer(Protocol):
    figure_id: str

    def render(
        self,
        results: "object",
        batch: str,
        viz: "object",
        ax: "matplotlib.axes.Axes | None" = None,
    ) -> matplotlib.figure.Figure: ...


RENDERERS: dict[str, "FigureRenderer"] = {}
