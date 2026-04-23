from __future__ import annotations
import matplotlib.figure
import matplotlib.pyplot as plt


class SampleGridRenderer:
    figure_id = "sample_grid"

    def render(self, results, batch, viz, ax=None) -> matplotlib.figure.Figure:
        frame_ids = results.list_frames(batch)
        n = min(len(frame_ids), 9)
        cols = 3
        rows = max(1, (n + cols - 1) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes_flat = [axes] if n == 1 else (axes.flat if hasattr(axes, "flat") else [axes])
        for i, fid in enumerate(frame_ids[:n]):
            a = list(axes_flat)[i]
            try:
                eta = results.read_frame(batch, fid)
                a.imshow(eta, cmap="RdBu_r", origin="upper")
                a.set_title(f"#{fid}", fontsize=8)
                a.axis("off")
            except Exception:
                a.axis("off")
        for j in range(n, rows * cols):
            list(axes_flat)[j].axis("off")
        fig.suptitle("Sample grid")
        return fig
