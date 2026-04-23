from __future__ import annotations
import matplotlib.figure
import matplotlib.pyplot as plt


class WavelengthProfileRenderer:
    figure_id = "wavelength_profile"

    def render(self, results, batch, viz, ax=None) -> matplotlib.figure.Figure:
        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots()
        try:
            meta = results.read_batch_meta(batch)
            ax.set_xlabel("Position [mm]")
            ax.set_ylabel("λ [mm]")
            ax.set_title("Wavelength profile")
        except Exception:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig
