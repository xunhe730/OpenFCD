from __future__ import annotations
import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt


class EtaHeatmapRenderer:
    figure_id = "eta_heatmap"

    def render(self, results, batch, viz, ax=None) -> matplotlib.figure.Figure:
        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots()
        try:
            eta = results.read_summary(batch, "eta_mean")
            vmin = getattr(viz, "eta_vmin_mm", -0.35)
            vmax = getattr(viz, "eta_vmax_mm", 0.35)
            cmap = getattr(viz, "cmap", "RdBu_r")
            im = ax.imshow(eta, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
            fig.colorbar(im, ax=ax, label="η [mm]")
            ax.set_title("η heatmap")
        except Exception:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig
