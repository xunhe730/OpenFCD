from __future__ import annotations

from typing import Literal

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from openfcd.gui.renderers._eta_view import compute_eta_color_range, crop_to_valid


class EtaHeatmapRenderer:
    figure_id = "eta_heatmap"

    def render(
        self,
        results,
        batch,
        viz,
        ax=None,
        *,
        mode: Literal["mean", "frame"] = "frame",
        frame_idx: int = 0,
        crop_to_valid_pixels: bool = True,
    ) -> matplotlib.figure.Figure:
        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots()
        try:
            if mode == "mean":
                eta = results.read_summary(batch, "eta_mean")
            else:
                eta = results.read_frame(batch, int(frame_idx))
            eta = np.asarray(eta, dtype=np.float64)
            if crop_to_valid_pixels:
                eta = crop_to_valid(eta)
            cmap = getattr(viz, "cmap", "RdBu_r")
            vmin, vmax = compute_eta_color_range(eta, viz, cmap=cmap)
            im = ax.imshow(eta, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
            fig.colorbar(im, ax=ax, label="η [mm]")
            label = "η heatmap (mean)" if mode == "mean" else f"η heatmap (frame {frame_idx})"
            ax.set_title(label)
        except Exception:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig
