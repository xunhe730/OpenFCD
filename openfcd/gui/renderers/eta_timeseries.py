from __future__ import annotations
import matplotlib.figure
import matplotlib.pyplot as plt


class EtaTimeseriesRenderer:
    figure_id = "eta_timeseries"

    def render(self, results, batch, viz, ax=None) -> matplotlib.figure.Figure:
        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots()
        try:
            frame_ids = results.list_frames(batch)
            eta_vals = [results.read_frame(batch, fid).mean() for fid in frame_ids]
            ax.plot(frame_ids, eta_vals)
            ax.set_xlabel("Frame")
            ax.set_ylabel("η mean [mm]")
            ax.set_title("η timeseries")
        except Exception:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig
