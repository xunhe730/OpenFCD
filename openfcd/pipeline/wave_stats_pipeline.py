"""Shared wave-stats computation layer (v3) — called by CLI PostprocessStage
and GUI _RunWorker / SessionController.recompute_wave_stats.

v3 model: each frame carries its own independent list of WaveSegments.
Frames without segments are skipped — no wave_stats group is written for
them. The single-frame core ``compute_frame_wave_stats`` is unchanged; this
module is pure orchestration that fans out by frame.

HDF5 layout produced (per batch b)::

    batches/{b}/wave_stats/
      attrs:
        schema_version : int     (=3)
        prominence_k   : float
        ds_mm          : float   (1.0 / px_per_mm)
        n_frames       : int     (number of frames that have segments)
      frame_{fid}/
        attrs:
          frame_id      : int
          n_segments    : int
          segments_meta : str  (JSON list of {segment_idx,label,color,visible,s_lo,s_hi})
        segments/{idx:04d}/   (one group per visible segment on that frame)
          attrs: s_lo_mm, s_hi_mm, label, color, visible (uint8)
          wavelength_mm     scalar float64
          wavenumber_per_mm scalar float64
          peaks   (n, 2) float64  columns: [s_mm, eta]
          troughs (n, 2) float64  columns: [s_mm, eta]
          heights (n,)   float64
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
    """Compute and write per-frame wave statistics for every batch.

    Returns ``True`` if at least one frame was written; ``False`` if no-op
    (no wave_stats config, no profile_line, no segments_by_frame entries,
    or every key maps to an empty/all-invisible list).
    """
    ws_cfg = annotation.wave_stats
    if ws_cfg is None:
        return False

    pl = annotation.profile_line
    if pl is None:
        return False

    if not ws_cfg.segments_by_frame:
        return False

    profile_line_arg: tuple[tuple[float, float], tuple[float, float]] = (
        pl.start,
        pl.end,
    )
    prominence_k = float(ws_cfg.peak_prominence_k)

    any_written = False

    for batch in sorted(batches):
        px_per_mm = float(px_per_mm_by_batch.get(batch, 1.0))
        ds_mm = 1.0 / max(px_per_mm, 1e-9)
        poly_by_frame = body_polygons_by_batch_frame.get(batch, {})

        frame_ids: list[int] = sorted(store.list_frames(batch))

        # Build per-frame payloads only for frames that have visible segments
        per_frame_payloads: dict[int, dict] = {}
        for fid in frame_ids:
            fid_str = str(fid)
            seg_list = ws_cfg.segments_for_frame(fid_str)
            visible = [
                (i, seg) for i, seg in enumerate(seg_list) if seg.visible
            ]
            if not visible:
                continue
            try:
                eta = np.asarray(eta_loader(batch, fid_str), dtype=np.float64)
            except Exception:  # noqa: BLE001
                continue

            seg_tuples = [
                (float(seg.s_lo_mm), float(seg.s_hi_mm)) for _, seg in visible
            ]
            poly_rc: np.ndarray | None = poly_by_frame.get(fid_str, None)

            result: FrameWaveStats = compute_frame_wave_stats(
                eta=eta,
                profile_line=profile_line_arg,
                body_polygon_rc=poly_rc,
                px_per_mm=px_per_mm,
                segments=seg_tuples,
                prominence_k=prominence_k,
            )

            seg_payloads: list[dict] = []
            seg_meta: list[dict] = []
            for k, (orig_idx, seg) in enumerate(visible):
                seg_stats = result.segments[k]
                seg_payloads.append(
                    {
                        "segment_idx": int(orig_idx),
                        "s_lo_mm": float(seg.s_lo_mm),
                        "s_hi_mm": float(seg.s_hi_mm),
                        "label": str(seg.label),
                        "color": str(seg.color),
                        "visible": bool(seg.visible),
                        "wavelength_mm": float(seg_stats.wavelength_mm),
                        "wavenumber_per_mm": float(seg_stats.wavenumber_per_mm),
                        "peaks": (
                            np.column_stack(
                                [seg_stats.peaks_s_mm, seg_stats.peaks_eta]
                            ).astype(np.float64)
                            if seg_stats.n_peaks > 0
                            else np.empty((0, 2), dtype=np.float64)
                        ),
                        "troughs": (
                            np.column_stack(
                                [seg_stats.troughs_s_mm, seg_stats.troughs_eta]
                            ).astype(np.float64)
                            if seg_stats.n_troughs > 0
                            else np.empty((0, 2), dtype=np.float64)
                        ),
                        "heights": np.asarray(
                            seg_stats.peak_to_trough_heights, dtype=np.float64
                        ),
                    }
                )
                seg_meta.append(
                    {
                        "segment_idx": int(orig_idx),
                        "label": str(seg.label),
                        "color": str(seg.color),
                        "visible": bool(seg.visible),
                        "s_lo_mm": float(seg.s_lo_mm),
                        "s_hi_mm": float(seg.s_hi_mm),
                    }
                )

            per_frame_payloads[fid] = {
                "segments": seg_payloads,
                "segments_meta_json": json.dumps(seg_meta, sort_keys=True),
            }

        if not per_frame_payloads:
            continue

        attrs: dict = {
            "schema_version": 3,
            "prominence_k": prominence_k,
            "ds_mm": ds_mm,
            "n_frames": len(per_frame_payloads),
        }

        store.write_wave_stats_per_frame(batch, per_frame_payloads, attrs)
        any_written = True

    return any_written
