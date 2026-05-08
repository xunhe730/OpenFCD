"""Compute stage: image pair → calibrated surface height.

Ported from BOS/bos/fcd_run.py. Import paths use openfcd.core.*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence, TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from skimage.io import imread
from skimage.color import rgb2gray

from openfcd.core.fcd import (
    Carrier, calculate_carriers, carrier_amplitude,
    carriers_pixel_per_mm, fcd as fcd_height,
)
from openfcd.core.flatfield import flatfield_normalize
from openfcd.core.inpaint import inpaint_fft, synthesize_from_carriers
from openfcd.core.mask import (
    Box, Polygon, auto_mask, detect_filament_occluders, find_largest_interior_blob,
    find_oriented_polygon, manual_mask, polygon_mask,
    rectangular_mask, union,
)
from openfcd.pipeline.base import Stage, StageEvent, CancelToken

if TYPE_CHECKING:
    from openfcd.core.polygon_source import PolygonSource


def _crop_to_roi(img: np.ndarray, center_rc: tuple,
                  size_yx_px: tuple) -> tuple:
    """Rectangular ROI crop. size_yx_px = (height_px, width_px); 0 = keep full
    along that axis. Returns (cropped, (row0, col0))."""
    h, w = img.shape
    sy, sx = size_yx_px
    if sy <= 0 or sy >= h:
        sy = h
    if sx <= 0 or sx >= w:
        sx = w
    cy, cx = center_rc
    r0 = int(max(0, min(h - sy, cy - sy // 2)))
    c0 = int(max(0, min(w - sx, cx - sx // 2)))
    return img[r0:r0 + sy, c0:c0 + sx], (r0, c0)


def cosine_taper(shape: tuple, alpha: float = 0.08) -> np.ndarray:
    """2D Tukey (cosine-tapered) window. `alpha` is the fraction of each
    axis that is tapered; the central (1-alpha) fraction is exactly 1."""
    from scipy.signal import windows
    wy = windows.tukey(shape[0], alpha=alpha)
    wx = windows.tukey(shape[1], alpha=alpha)
    return np.outer(wy, wx)


def edge_margin_mask(shape: tuple, margin_px: int) -> np.ndarray:
    """Boolean mask: True for outer-ring pixels within `margin_px` of any edge."""
    m = np.zeros(shape, dtype=bool)
    if margin_px <= 0:
        return m
    m[:margin_px, :] = True
    m[-margin_px:, :] = True
    m[:, :margin_px] = True
    m[:, -margin_px:] = True
    return m


def detrend_plane(eta: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Subtract a least-squares linear plane fitted over `valid` pixels.

    FCD recovers eta only up to a linear plane (it sees only gradients), and
    any small rigid translation between ref and def shows up as a constant
    bias in (u, v) that fftinvgrad integrates into a huge ramp. Removing the
    plane is the standard cleanup.
    """
    h, w = eta.shape
    yy, xx = np.mgrid[0:h, 0:w]
    A = np.column_stack([xx[valid].ravel(), yy[valid].ravel(),
                         np.ones(int(valid.sum()))])
    b = eta[valid].ravel()
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    plane = coef[0] * xx + coef[1] * yy + coef[2]
    return eta - plane


def suppress_nonphysical_eta_filaments(
    eta: np.ndarray,
    *,
    low_signal_mask: np.ndarray | None = None,
    residual_percentile: float = 98.5,
    median_window_px: int = 11,
    min_length_px: int = 10,
    max_width_px: int = 6,
    min_aspect: float = 4.0,
    glint_max_area_px: int = 2500,
    glint_max_size_px: int = 96,
) -> np.ndarray:
    """Suppress non-physical finite eta artifacts near low-signal regions.

    The target is not broad wave structure, but narrow discontinuities and
    compact red/blue dipoles caused by local carrier corruption (wires, glints,
    tiny occluders) that leak through the global gradient integration.
    """
    from skimage.draw import line
    from skimage.measure import label, regionprops
    from skimage.morphology import dilation, disk
    from skimage.restoration import inpaint_biharmonic
    from skimage.transform import probabilistic_hough_line

    work = np.asarray(eta, dtype=np.float64)
    finite = np.isfinite(work)
    if finite.sum() < 100:
        return work

    background = median_filter(np.where(finite, work, 0.0), size=median_window_px, mode="nearest")
    residual = np.abs(work - background)
    residual_valid = residual[finite]
    median_res = float(np.median(residual_valid))
    mad_res = float(np.median(np.abs(residual_valid - median_res))) + 1e-12
    percentile_thr = float(np.percentile(residual_valid, residual_percentile))
    robust_thr = median_res + 6.0 * mad_res
    threshold = max(percentile_thr, robust_thr)
    seed = residual > threshold

    anchor_region = None
    if low_signal_mask is not None and low_signal_mask.any():
        anchor_region = dilation(low_signal_mask, disk(16))

    line_mask = np.zeros_like(seed, dtype=bool)
    # Pin RNG so output is deterministic across repeat calls and threads.
    for p0, p1 in probabilistic_hough_line(
        seed,
        threshold=10,
        line_length=min_length_px,
        line_gap=6,
        rng=np.random.default_rng(0),
    ):
        rr, cc = line(p0[1], p0[0], p1[1], p1[0])
        segment = np.zeros_like(seed, dtype=bool)
        segment[rr, cc] = True
        segment = dilation(segment, disk(2))
        touches_edge = (
            rr.min() <= 4 or cc.min() <= 4
            or rr.max() >= work.shape[0] - 4
            or cc.max() >= work.shape[1] - 4
        )
        anchored = anchor_region is None or np.any(segment & anchor_region) or touches_edge
        if anchored:
            line_mask |= segment

    lab = label(seed.astype(np.uint8))
    candidate = np.zeros_like(seed, dtype=bool)
    for region in regionprops(lab):
        minr, minc, maxr, maxc = region.bbox
        h = maxr - minr
        w = maxc - minc
        long_axis = max(h, w)
        short_axis = max(1, min(h, w))
        aspect = long_axis / short_axis
        if long_axis < min_length_px:
            continue
        if short_axis > max_width_px:
            continue
        if aspect < min_aspect:
            continue
        region_mask = lab == region.label
        touches_edge = (
            minr <= 4 or minc <= 4
            or maxr >= work.shape[0] - 4
            or maxc >= work.shape[1] - 4
        )
        anchored = anchor_region is None or np.any(region_mask & anchor_region) or touches_edge
        if not anchored:
            continue
        candidate |= region_mask

    if anchor_region is not None:
        # Reflective pinholes often produce compact red/blue dipoles rather
        # than long line-like jumps. Repair only small residual components that
        # are attached to a low-carrier/occlusion anchor, so real wave bands
        # away from glints are left alone.
        glint_lab = label((seed & anchor_region).astype(np.uint8))
        for region in regionprops(glint_lab):
            minr, minc, maxr, maxc = region.bbox
            h = maxr - minr
            w = maxc - minc
            if region.area > glint_max_area_px:
                continue
            if max(h, w) > glint_max_size_px:
                continue
            candidate |= glint_lab == region.label

    candidate |= line_mask
    candidate &= finite
    if not candidate.any():
        return work

    candidate = dilation(candidate, disk(1))

    fill_value = float(np.nanmedian(work[finite]))
    filled = np.where(finite, work, fill_value)
    out = work.copy()
    repair_lab = label(candidate.astype(np.uint8))
    pad = max(8, median_window_px * 2)
    for region in regionprops(repair_lab):
        minr, minc, maxr, maxc = region.bbox
        r0 = max(0, minr - pad)
        c0 = max(0, minc - pad)
        r1 = min(work.shape[0], maxr + pad)
        c1 = min(work.shape[1], maxc + pad)
        target_crop = repair_lab[r0:r1, c0:c1] == region.label
        finite_crop = finite[r0:r1, c0:c1]
        mask_crop = target_crop | ~finite_crop
        crop = filled[r0:r1, c0:c1]
        repaired = inpaint_biharmonic(crop, mask_crop, channel_axis=None)
        out_crop = out[r0:r1, c0:c1]
        out_crop[target_crop] = repaired[target_crop]
        out[r0:r1, c0:c1] = out_crop
    return out


def carrier_phase_residual(
    img: np.ndarray,
    carriers: Sequence[Carrier],
    *,
    window_px: int = 11,
) -> np.ndarray:
    """Local circular phase inconsistency for carrier-demodulated images.

    Values near zero mean the carrier phase is locally coherent. Large values
    flag phase jumps from glints, saturation, and occluder edges.
    """
    from openfcd.core.fcd import fcd_phases

    size = max(3, int(window_px))
    if size % 2 == 0:
        size += 1
    residual = np.zeros(np.asarray(img).shape, dtype=np.float64)
    for phi in fcd_phases(img, list(carriers), unwrap=False):
        c = uniform_filter(np.cos(phi), size=size, mode="nearest")
        s = uniform_filter(np.sin(phi), size=size, mode="nearest")
        local = np.arctan2(s, c)
        delta = np.angle(np.exp(1j * (phi - local)))
        residual = np.maximum(residual, np.abs(delta))
    return residual


def _robust_high_threshold(values: np.ndarray, percentile: float, sigma: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("inf")
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med))) + 1e-12
    return max(float(np.percentile(finite, percentile)), med + sigma * mad)


def eta_confidence_mask(
    eta: np.ndarray,
    *,
    carrier_amplitude_map: np.ndarray | None = None,
    phase_residual: np.ndarray | None = None,
    anchor_mask: np.ndarray | None = None,
    occlusion_mask: np.ndarray | None = None,
    median_window_px: int = 11,
    residual_percentile: float = 98.5,
    phase_percentile: float = 97.5,
    carrier_low_ratio: float = 0.55,
    anchor_dilate_px: int = 18,
    occlusion_boundary_px: int = 28,
    max_component_area_px: int = 6000,
    max_component_size_px: int = 180,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Detect finite eta artifacts caused by locally unreliable carriers.

    The output mask marks only finite eta pixels that should be locally
    repaired. Broad, unanchored wave structure is left untouched.
    """
    from skimage.measure import label, regionprops
    from skimage.morphology import dilation, disk

    work = np.asarray(eta, dtype=np.float64)
    finite = np.isfinite(work)
    empty = np.zeros(work.shape, dtype=bool)
    if finite.sum() < 100:
        if return_debug:
            return empty, {
                "eta_residual": np.zeros(work.shape, dtype=np.float64),
                "eta_seed": empty,
                "carrier_bad": empty,
                "phase_bad": empty,
                "anchor": empty,
            }
        return empty

    size = max(3, int(median_window_px))
    if size % 2 == 0:
        size += 1
    fill_value = float(np.nanmedian(work[finite]))
    filled = np.where(finite, work, fill_value)
    local = median_filter(filled, size=size, mode="nearest")
    eta_residual = np.abs(work - local)
    eta_residual[~finite] = 0.0
    residual_valid = eta_residual[finite]
    hard_thr = _robust_high_threshold(residual_valid, residual_percentile, 6.0)
    soft_thr = _robust_high_threshold(residual_valid, 95.0, 4.0)
    eta_seed = eta_residual > hard_thr
    eta_seed_soft = eta_residual > soft_thr

    carrier_bad = empty.copy()
    if carrier_amplitude_map is not None and carrier_amplitude_map.shape == work.shape:
        amp = np.asarray(carrier_amplitude_map, dtype=np.float64)
        amp_pos = amp[np.isfinite(amp) & (amp > 0)]
        if amp_pos.size:
            amp_ref = float(np.median(amp_pos))
            carrier_bad = amp < carrier_low_ratio * amp_ref

    phase_bad = empty.copy()
    if phase_residual is not None and phase_residual.shape == work.shape:
        phase = np.asarray(phase_residual, dtype=np.float64)
        phase_values = phase[finite & np.isfinite(phase)]
        if phase_values.size:
            phase_thr = max(_robust_high_threshold(phase_values, phase_percentile, 4.0), 0.45)
            phase_bad = phase > phase_thr

    anchor = carrier_bad | phase_bad
    if anchor_mask is not None and anchor_mask.shape == work.shape:
        anchor |= np.asarray(anchor_mask, dtype=bool)
    anchor &= finite | carrier_bad | phase_bad

    if anchor.any():
        anchor_region = dilation(anchor, footprint=disk(max(1, int(anchor_dilate_px))))
    else:
        anchor_region = empty.copy()

    candidate = eta_seed & anchor_region & finite

    near_occlusion = empty.copy()
    if occlusion_mask is not None and occlusion_mask.shape == work.shape:
        occ = np.asarray(occlusion_mask, dtype=bool)
        if occ.any():
            near_occlusion = dilation(
                occ, footprint=disk(max(1, int(occlusion_boundary_px)))
            ) & ~occ
            candidate |= eta_seed_soft & near_occlusion & finite

    if candidate.any():
        candidate = dilation(candidate, footprint=disk(1)) & finite

    accepted = empty.copy()
    lab = label(candidate.astype(np.uint8), connectivity=1)
    for region in regionprops(lab):
        minr, minc, maxr, maxc = region.bbox
        h = maxr - minr
        w = maxc - minc
        if region.area > max_component_area_px:
            continue
        if max(h, w) > max_component_size_px:
            continue
        accepted |= lab == region.label

    if return_debug:
        return accepted, {
            "eta_residual": eta_residual,
            "eta_seed": eta_seed,
            "eta_seed_soft": eta_seed_soft,
            "carrier_bad": carrier_bad,
            "phase_bad": phase_bad,
            "anchor": anchor_region,
            "near_occlusion": near_occlusion,
        }
    return accepted


def repair_eta_confidence_artifacts(
    eta: np.ndarray,
    artifact_mask: np.ndarray,
    *,
    pad_px: int = 24,
) -> np.ndarray:
    """Locally inpaint finite eta artifact pixels selected by a confidence mask."""
    from skimage.measure import label, regionprops
    from skimage.restoration import inpaint_biharmonic

    work = np.asarray(eta, dtype=np.float64)
    mask = np.asarray(artifact_mask, dtype=bool)
    finite = np.isfinite(work)
    mask &= finite
    if not mask.any() or finite.sum() < 100:
        return work.copy()

    fill_value = float(np.nanmedian(work[finite]))
    filled = np.where(finite, work, fill_value)
    out = work.copy()
    lab = label(mask.astype(np.uint8), connectivity=1)
    for region in regionprops(lab):
        minr, minc, maxr, maxc = region.bbox
        r0 = max(0, minr - pad_px)
        c0 = max(0, minc - pad_px)
        r1 = min(work.shape[0], maxr + pad_px)
        c1 = min(work.shape[1], maxc + pad_px)
        target_crop = lab[r0:r1, c0:c1] == region.label
        finite_crop = finite[r0:r1, c0:c1]
        mask_crop = target_crop | ~finite_crop
        crop = filled[r0:r1, c0:c1]
        repaired = inpaint_biharmonic(crop, mask_crop, channel_axis=None)
        out_crop = out[r0:r1, c0:c1]
        out_crop[target_crop] = repaired[target_crop]
        out[r0:r1, c0:c1] = out_crop
    return out


def fill_small_eta_holes(
    eta: np.ndarray,
    *,
    px_per_mm: float,
    radius_mm: float,
    max_aspect_ratio: float = 2.5,
    hard_max_radius_mm: float = 2.0,
) -> np.ndarray:
    """Fill small enclosed NaN holes in a final eta field.

    This targets small round glint holes accidentally marked as invalid. Large
    occluders, edge/ROI margins, and wire-like NaN regions are intentionally
    left untouched.
    """
    if radius_mm <= 0 or px_per_mm <= 0:
        return np.asarray(eta, dtype=np.float64)

    from scipy.ndimage import binary_dilation
    from skimage.measure import label, regionprops
    from skimage.restoration import inpaint_biharmonic

    work = np.asarray(eta, dtype=np.float64)
    finite = np.isfinite(work)
    if finite.sum() < 100:
        return work.copy()

    invalid = ~finite
    if not invalid.any():
        return work.copy()

    effective_radius_mm = min(float(radius_mm), float(hard_max_radius_mm))
    radius_px = max(1, int(round(effective_radius_mm * float(px_per_mm))))
    max_area = np.pi * float(radius_px ** 2)
    max_bbox = max(1, 2 * radius_px)
    pad = max(8, radius_px)
    out = work.copy()

    lab = label(invalid.astype(np.uint8), connectivity=1)
    h_img, w_img = work.shape
    for region in regionprops(lab):
        minr, minc, maxr, maxc = region.bbox
        if minr == 0 or minc == 0 or maxr == h_img or maxc == w_img:
            continue

        height = maxr - minr
        width = maxc - minc
        long_axis = max(height, width)
        short_axis = max(1, min(height, width))
        if region.area > max_area:
            continue
        if long_axis > max_bbox:
            continue
        if long_axis / short_axis > max_aspect_ratio:
            continue

        component = lab == region.label
        ring = binary_dilation(component, iterations=1) & ~component
        if not ring.any() or not np.all(finite[ring]):
            continue

        r0 = max(0, minr - pad)
        c0 = max(0, minc - pad)
        r1 = min(h_img, maxr + pad)
        c1 = min(w_img, maxc + pad)
        crop = out[r0:r1, c0:c1]
        crop_finite = np.isfinite(crop)
        if crop_finite.sum() < 16:
            continue

        fill_value = float(np.nanmedian(crop[crop_finite]))
        filled = np.where(crop_finite, crop, fill_value)
        mask_crop = component[r0:r1, c0:c1]
        repaired = inpaint_biharmonic(filled, mask_crop, channel_axis=None)
        crop_out = crop.copy()
        crop_out[mask_crop] = repaired[mask_crop]
        out[r0:r1, c0:c1] = crop_out

    return out


def load_gray(path: Path | str) -> np.ndarray:
    img = imread(str(path))
    if img.ndim == 3:
        img = rgb2gray(img)
    return img.astype(np.float64)


@dataclass
class FCDResult:
    eta_mm: np.ndarray            # surface height (mm) over full image (NaN where masked)
    mask: np.ndarray              # union of all occlusion masks applied during processing
    pixel_per_mm: float
    carriers: Sequence[Carrier]
    raw_eta_pix: np.ndarray       # un-calibrated pyfcd output, kept for debugging
    robot_box: Optional[Box] = None
    robot_poly: Optional[Polygon] = None
    roi_offset: tuple = (0, 0)
    polygon_source_kind: str = "fixed"


@dataclass
class GeomParams:
    """Optical geometry parameters needed for calibration."""
    pattern_period_mm: float
    alpha: float           # sensitivity factor
    h_p_eff_mm: float      # effective pattern-to-surface distance (mm)


def process(reference_path: Path | str,
            deformed_path: Path | str,
            geom: GeomParams,
            robot_box: Optional[Box] = None,
            robot_poly: Optional[Polygon] = None,
            polygon_source: Optional["PolygonSource"] = None,
            frame_id: Optional[int] = None,
            auto_orient: bool = True,
            hand_boxes: Optional[Sequence[Box]] = None,
            flatfield_sigma: float = 300.0,
            unwrap: bool = False,
            roi_pixel_box: Optional[Box] = None,
            roi_mm: Optional[float] = None,
            roi_x_mm: Optional[float] = None,
            roi_y_mm: Optional[float] = None,
            roi_center_xy_px: Optional[tuple] = None,
            robot_pad_px: int = 4,
            taper_alpha: float = 0.08,
            edge_margin_mm: float = 3.0) -> FCDResult:
    """Run the FCD pipeline with a fixed rectangular mask."""
    i_ref_raw = load_gray(reference_path)
    i_def_raw = load_gray(deformed_path)
    same_input = np.array_equal(i_ref_raw, i_def_raw)
    if i_ref_raw.shape != i_def_raw.shape:
        raise ValueError("reference and deformed images must have the same shape.")

    roi_offset = (0, 0)
    if roi_pixel_box is not None:
        b = roi_pixel_box
        h_full, w_full = i_ref_raw.shape
        r0 = max(0, b.row0)
        c0 = max(0, b.col0)
        r1 = min(h_full, b.row0 + b.height)
        c1 = min(w_full, b.col0 + b.width)
        i_ref_raw = i_ref_raw[r0:r1, c0:c1]
        i_def_raw = i_def_raw[r0:r1, c0:c1]
        roi_offset = (r0, c0)
        if robot_box is not None:
            robot_box = Box(robot_box.row0 - r0, robot_box.col0 - c0,
                              robot_box.height, robot_box.width)
        if robot_poly is not None:
            robot_poly = robot_poly.shifted(-r0, -c0)
        if hand_boxes:
            hand_boxes = [Box(b2.row0 - r0, b2.col0 - c0,
                               b2.height, b2.width) for b2 in hand_boxes]
        roi_mm = roi_x_mm = roi_y_mm = None

    do_crop = (roi_mm is not None) or (roi_x_mm is not None) or (roi_y_mm is not None)
    sy_mm = roi_y_mm if roi_y_mm is not None else roi_mm
    sx_mm = roi_x_mm if roi_x_mm is not None else roi_mm

    if do_crop:
        center_rc = None
        if roi_center_xy_px is not None:
            center_rc = (roi_center_xy_px[1], roi_center_xy_px[0])
        elif robot_poly is not None:
            center_rc = robot_poly.centroid()
        elif robot_box is not None:
            center_rc = (robot_box.row0 + robot_box.height / 2,
                          robot_box.col0 + robot_box.width / 2)
        else:
            i_ref_ff_pre = flatfield_normalize(i_ref_raw, sigma=flatfield_sigma)
            i_def_ff_pre = flatfield_normalize(i_def_raw, sigma=flatfield_sigma,
                                               bg_src=i_ref_raw)
            c_pre = calculate_carriers(i_ref_ff_pre - i_ref_ff_pre.mean())
            px_per_mm_pre = carriers_pixel_per_mm(c_pre, geom.pattern_period_mm)
            def_mask_pre = auto_mask(i_def_ff_pre, c_pre,
                                     threshold_ratio=0.2, dilate_px=8)
            rbox = find_largest_interior_blob(def_mask_pre, edge_margin=20)
            if rbox is not None:
                center_rc = (rbox.row0 + rbox.height / 2,
                              rbox.col0 + rbox.width / 2)
        if center_rc is None:
            print("[process] ROI requested but object blob not detected; skipping crop.")
        else:
            i_ref_ff_full = flatfield_normalize(i_ref_raw, sigma=flatfield_sigma)
            c_full = calculate_carriers(i_ref_ff_full - i_ref_ff_full.mean())
            px_per_mm0 = carriers_pixel_per_mm(c_full, geom.pattern_period_mm)
            sy_px = int(sy_mm * px_per_mm0) if sy_mm else 0
            sx_px = int(sx_mm * px_per_mm0) if sx_mm else 0
            i_ref_raw, roi_offset = _crop_to_roi(i_ref_raw, center_rc, (sy_px, sx_px))
            i_def_raw, _ = _crop_to_roi(i_def_raw, center_rc, (sy_px, sx_px))

            if robot_box is not None:
                robot_box = Box(robot_box.row0 - roi_offset[0],
                                 robot_box.col0 - roi_offset[1],
                                 robot_box.height, robot_box.width)
            if robot_poly is not None:
                robot_poly = robot_poly.shifted(-roi_offset[0], -roi_offset[1])
            if hand_boxes:
                hand_boxes = [Box(b.row0 - roi_offset[0],
                                    b.col0 - roi_offset[1],
                                    b.height, b.width) for b in hand_boxes]

    i_ref_ff = flatfield_normalize(i_ref_raw, sigma=flatfield_sigma)
    i_def_ff = flatfield_normalize(i_def_raw, sigma=flatfield_sigma, bg_src=i_ref_raw)
    carriers0 = calculate_carriers(i_ref_ff - i_ref_ff.mean())
    filament_mask = detect_filament_occluders(i_def_ff, carriers0)

    polygon_source_kind: str
    if polygon_source is not None:
        poly = polygon_source.get(frame_id)
        if poly is None:
            raise RuntimeError(
                f"polygon_source returned None for frame_id={frame_id}"
            )
        if roi_offset != (0, 0):
            poly = poly.shifted(-roi_offset[0], -roi_offset[1])
        robot_poly = poly
        polygon_source_kind = polygon_source.kind()
    elif robot_poly is not None or robot_box is not None:
        polygon_source_kind = "fixed"
    else:
        polygon_source_kind = "bootstrap"

    if robot_poly is None and robot_box is None:
        scout_mask = auto_mask(i_def_ff, carriers0,
                                threshold_ratio=0.2, dilate_px=4)
        if auto_orient:
            robot_poly = find_oriented_polygon(scout_mask, edge_margin=20)
        if robot_poly is None:
            robot_box = find_largest_interior_blob(scout_mask, edge_margin=20)
        if robot_poly is None and robot_box is None:
            raise RuntimeError("Could not auto-detect an object blob; please "
                                "pass robot_box=Box(...) or robot_poly=Polygon(...).")
    if robot_poly is None and robot_box is not None:
        robot_poly = Polygon.from_box(robot_box)
    if robot_box is None:
        robot_box = robot_poly.bbox()

    occlusion_mask = polygon_mask(i_ref_ff.shape, robot_poly,
                                    dilate_px=robot_pad_px)
    occlusion_mask |= filament_mask
    if hand_boxes:
        occlusion_mask |= manual_mask(i_ref_ff.shape, hand_boxes)

    syn = synthesize_from_carriers(i_ref_ff - i_ref_ff.mean(), carriers0)
    i_ref_clean = inpaint_fft(i_ref_ff, occlusion_mask, syn + i_ref_ff.mean())
    carriers = calculate_carriers(i_ref_clean - i_ref_clean.mean())
    syn2 = synthesize_from_carriers(i_ref_clean - i_ref_clean.mean(), carriers)
    i_def_clean = inpaint_fft(i_def_ff, occlusion_mask, syn2 + i_ref_clean.mean())

    if taper_alpha > 0:
        win = cosine_taper(i_ref_clean.shape, alpha=taper_alpha)
        ref_mean = i_ref_clean.mean()
        def_mean = i_def_clean.mean()
        i_ref_clean = (i_ref_clean - ref_mean) * win + ref_mean
        i_def_clean = (i_def_clean - def_mean) * win + def_mean

    if same_input:
        px_per_mm = carriers_pixel_per_mm(carriers, geom.pattern_period_mm)
        eta_out = np.zeros(i_ref_clean.shape, dtype=np.float64)
        eta_out[occlusion_mask] = np.nan
        if edge_margin_mm > 0:
            em_px = int(edge_margin_mm * px_per_mm)
            eta_out[edge_margin_mask(eta_out.shape, em_px)] = np.nan
        return FCDResult(
            eta_mm=eta_out,
            mask=occlusion_mask,
            pixel_per_mm=px_per_mm,
            carriers=carriers,
            raw_eta_pix=np.zeros_like(i_ref_clean),
            robot_box=robot_box,
            robot_poly=robot_poly,
            roi_offset=roi_offset,
            polygon_source_kind=polygon_source_kind,
        )

    raw_eta = fcd_height(i_def_clean - i_ref_clean.mean(), carriers, unwrap=unwrap)
    raw_eta = detrend_plane(raw_eta, valid=~occlusion_mask)
    px_per_mm = carriers_pixel_per_mm(carriers, geom.pattern_period_mm)
    eta_mm = raw_eta / (geom.alpha * geom.h_p_eff_mm * px_per_mm ** 2)
    eta_out = eta_mm.copy()
    eta_out[occlusion_mask] = np.nan
    if edge_margin_mm > 0:
        em_px = int(edge_margin_mm * px_per_mm)
        eta_out[edge_margin_mask(eta_out.shape, em_px)] = np.nan

    eta_out = suppress_nonphysical_eta_filaments(
        eta_out,
        low_signal_mask=occlusion_mask,
    )

    return FCDResult(
        eta_mm=eta_out,
        mask=occlusion_mask,
        pixel_per_mm=px_per_mm,
        carriers=carriers,
        raw_eta_pix=raw_eta,
        robot_box=robot_box,
        robot_poly=robot_poly,
        roi_offset=roi_offset,
        polygon_source_kind=polygon_source_kind,
    )
