from __future__ import annotations
import matplotlib.figure
import matplotlib.pyplot as plt


class RmsMapRenderer:
    figure_id = "rms_map"

    def render(self, results, batch, viz, ax=None) -> matplotlib.figure.Figure:
        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots()
        try:
            eta_rms = results.read_summary(batch, "eta_rms")
            im = ax.imshow(eta_rms, cmap="hot", origin="upper")
            fig.colorbar(im, ax=ax, label="η RMS [mm]")
            ax.set_title("η RMS map")
        except Exception:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig
