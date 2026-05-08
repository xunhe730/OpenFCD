from __future__ import annotations
import matplotlib.figure
import matplotlib.pyplot as plt

from openfcd.core.profile_composite import (
    ProfileCompositeContext,
    build_profile_composite_context,
    render_profile_composite_context,
)


class ProfileCompositeRenderer:
    figure_id = "wavelength_profile"

    def render(self, results, batch, viz, ax=None) -> matplotlib.figure.Figure:
        layout_mode = getattr(viz, "layout_mode", "preview")
        if ax is not None:
            fig = ax.figure
        else:
            fig = plt.figure(figsize=(7.0, 5.0), constrained_layout=True)
        if isinstance(viz, ProfileCompositeContext):
            return render_profile_composite_context(viz, fig=fig, layout_mode=layout_mode)
        frame_idx = getattr(viz, "frame_idx", None)
        profile_line = getattr(viz, "profile_line", None)
        eta = None
        try:
            if frame_idx is not None:
                eta = results.read_frame(batch, int(frame_idx))
        except Exception:
            eta = None
        context = build_profile_composite_context(
            eta=eta,
            profile_line=profile_line,
            viz_params=getattr(viz, "__dict__", {}),
            frame_idx=frame_idx,
            frame_name=getattr(viz, "frame_name", None),
            run_id=getattr(viz, "run_id", None),
            batch=batch,
            body_polygon_rc=getattr(viz, "body_polygon_rc", None),
            body_source=getattr(viz, "body_source", "fallback"),
            degraded_reason=getattr(viz, "degraded_reason", None),
            spatial_calibration=getattr(viz, "spatial_calibration", None),
        )
        return render_profile_composite_context(
            context,
            fig=fig,
            layout_mode=layout_mode,
        )


WavelengthProfileRenderer = ProfileCompositeRenderer
