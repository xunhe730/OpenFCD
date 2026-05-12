"""Shared wave-stats computation layer (v2) — called by CLI PostprocessStage
and GUI _RunWorker / SessionController.recompute_wave_stats.

The public entry point is ``compute_and_write_wave_stats``.  It is a pure
orchestration function: it loads η frames from an already-open HDF5ResultStore,
calls compute_frame_wave_stats per frame (one call per frame covers all
visible segments), and writes the results back via store.write_wave_stats.

HDF5 layout produced (per batch b):

  batches/{b}/wave_stats/
    attrs:
      schema_version : int     (=2)
      prominence_k   : float
      ds_mm          : float   (1.0 / px_per_mm)
      n_frames       : int
      n_segments     : int
      segments_meta  : str     (JSON list of {label, color, visible})
    segments/{idx:04d}/        (one group per visible segment)
      attrs: s_lo_mm, s_hi_mm, label, color, visible (uint8)
      wavelength_mm     (n_frames,) float64
      wavenumber_per_mm (n_frames,) float64
      peaks/{frame_id}      (n, 2) float64  columns: [s_mm, eta]
      troughs/{frame_id}    (n, 2) float64  columns: [s_mm, eta]
      heights/{frame_id}    (n,)   float64
"""
from __future__ import annotations

import json
from typing import Callable

import numpy as np

from openfcd.core.wave_stats import FrameWaveStats, compute_frame_wave_stats
from openfcd.io.annotation import AnnotationSchema
from openfcd.io.result import HDF5ResultStore


def compute_and_write_wave_stats(
    annotation: AnnotationSchema,
    store: HDF5ResultStore,
    batches: list[str],
    px_per_mm_by_batch: dict[str, float],
    body_polygons_by_batch_frame: dict[str, dict[str, np.ndarray | None]],
    eta_loader: Callable[[str, str], np.ndarray],
) -> bool:
    """Compute and write wave statistics (v2) for every batch.

    Parameters
    ----------
    annotation :
        AnnotationSchema carrying wave_stats (WaveStatsConfig with N segments)
        and profile_line (ProfileLineData).  Returns False immediately if
        either is None, or if no segment is visible.
    store :
        Open HDF5ResultStore.  Frames must already be written.
    batches :
        Ordered list of batch names to process (sorted internally).
    px_per_mm_by_batch :
        Pixels-per-mm calibration for each batch.
    body_polygons_by_batch_frame :
        ``{batch: {frame_id_str: polygon_rc | None}}``.  Reserved for future
        use; v2 does not consult polygons for segmentation.
    eta_loader :
        ``(batch, frame_id_str) -> η (H, W) float64``.

    Returns
    -------
    bool
        True if at least one batch was written; False if no-op.
    """
    ws_cfg = annotation.wave_stats
    if ws_cfg is None:
        return False

    pl = annotation.profile_line
    if pl is None:
        return False

    # Filter to only visible segments; remember original indices so output
    # h5 group keys match the schema-level segment index.
    visible_segments: list[tuple[int, float, float]] = [
        (idx, float(seg.s_lo_mm), float(seg.s_hi_mm))
        for idx, seg in enumerate(ws_cfg.segments)
        if seg.visible
    ]
    if not visible_segments:
        return False

    profile_line_arg: tuple[tuple[float, float], tuple[float, float]] = (
        pl.start,
        pl.end,
    )
    prominence_k = float(ws_cfg.peak_prominence_k)
    segments_tuples = [(s_lo, s_hi) for (_idx, s_lo, s_hi) in visible_segments]

    any_written = False

    for batch in sorted(batches):
        px_per_mm = float(px_per_mm_by_batch.get(batch, 1.0))
        ds_mm = 1.0 / max(px_per_mm, 1e-9)
        poly_by_frame = body_polygons_by_batch_frame.get(batch, {})

        frame_ids: list[int] = store.list_frames(batch)
        sorted_fids = sorted(frame_ids)
        n_frames = len(sorted_fids)

        # Accumulators keyed by *visible-list local index* (0..len(visible)-1)
        n_vis = len(visible_segments)
        wavelengths_per_seg: list[list[float]] = [[] for _ in range(n_vis)]
        wavenumbers_per_seg: list[list[float]] = [[] for _ in range(n_vis)]
        peaks_per_seg: list[dict[int, np.ndarray]] = [dict() for _ in range(n_vis)]
        troughs_per_seg: list[dict[int, np.ndarray]] = [dict() for _ in range(n_vis)]
        heights_per_seg: list[dict[int, np.ndarray]] = [dict() for _ in range(n_vis)]

        for fid in sorted_fids:
            fid_str = str(fid)
            try:
                eta = np.asarray(eta_loader(batch, fid_str), dtype=np.float64)
            except Exception:  # noqa: BLE001
                # Append NaN placeholders to keep per-frame array length
                for k in range(n_vis):
                    wavelengths_per_seg[k].append(float("nan"))
                    wavenumbers_per_seg[k].append(float("nan"))
                continue

            poly_rc: np.ndarray | None = poly_by_frame.get(fid_str, None)

            result: FrameWaveStats = compute_frame_wave_stats(
                eta=eta,
                profile_line=profile_line_arg,
                body_polygon_rc=poly_rc,
                px_per_mm=px_per_mm,
                segments=segments_tuples,
                prominence_k=prominence_k,
            )

            for k, seg_stats in enumerate(result.segments):
                wavelengths_per_seg[k].append(float(seg_stats.wavelength_mm))
                wavenumbers_per_seg[k].append(float(seg_stats.wavenumber_per_mm))
                if seg_stats.n_peaks > 0:
                    peaks_per_seg[k][fid] = np.column_stack(
                        [seg_stats.peaks_s_mm, seg_stats.peaks_eta]
                    ).astype(np.float64)
                else:
                    peaks_per_seg[k][fid] = np.empty((0, 2), dtype=np.float64)
                if seg_stats.n_troughs > 0:
                    troughs_per_seg[k][fid] = np.column_stack(
                        [seg_stats.troughs_s_mm, seg_stats.troughs_eta]
                    ).astype(np.float64)
                else:
                    troughs_per_seg[k][fid] = np.empty((0, 2), dtype=np.float64)
                heights_per_seg[k][fid] = np.asarray(
                    seg_stats.peak_to_trough_heights, dtype=np.float64
                )

        # Build the per-segment dicts in the layout expected by write_wave_stats
        segments_payload: list[dict] = []
        segments_meta: list[dict] = []
        for k, (orig_idx, s_lo, s_hi) in enumerate(visible_segments):
            seg = ws_cfg.segments[orig_idx]
            segments_payload.append(
                {
                    "segment_idx": int(orig_idx),
                    "s_lo_mm": s_lo,
                    "s_hi_mm": s_hi,
                    "label": str(seg.label),
                    "color": str(seg.color),
                    "visible": bool(seg.visible),
                    "wavelength_mm": np.asarray(wavelengths_per_seg[k], dtype=np.float64),
                    "wavenumber_per_mm": np.asarray(
                        wavenumbers_per_seg[k], dtype=np.float64
                    ),
                    "peaks": peaks_per_seg[k],
                    "troughs": troughs_per_seg[k],
                    "heights": heights_per_seg[k],
                }
            )
            segments_meta.append(
                {
                    "segment_idx": int(orig_idx),
                    "label": str(seg.label),
                    "color": str(seg.color),
                    "visible": bool(seg.visible),
                    "s_lo_mm": s_lo,
                    "s_hi_mm": s_hi,
                }
            )

        attrs: dict = {
            "schema_version": 2,
            "prominence_k": prominence_k,
            "ds_mm": ds_mm,
            "n_frames": n_frames,
            "n_segments": len(visible_segments),
            "segments_meta": json.dumps(segments_meta, sort_keys=True),
        }

        store.write_wave_stats(batch, segments_payload, attrs)
        any_written = True

    return any_written
