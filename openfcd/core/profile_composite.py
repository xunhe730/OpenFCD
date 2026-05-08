"""Profile Composite extraction and rendering.

The public figure id remains ``wavelength_profile`` for compatibility.  This
module owns the internal ProfileComposite contract used by GUI preview,
GUI export, and CLI replay.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal

import matplotlib.figure
from matplotlib import colormaps
from matplotlib.patches import Rectangle
import numpy as np


ProfileLineLike = tuple[tuple[float, float], tuple[float, float]]
BodySource = Literal["frame_polygon", "global_polygon", "fallback"]

PROFILE_COMPOSITE_DEFAULTS: dict[str, Any] = {
    "cmap": "RdBu_r",
    "vmin": None,
    "vmax": None,
    "line_color": "red",
    "profile_color": "blue",
    "strip_mm": 2.0,
    "px_per_mm": 1.0,
    "y_range_mm": 30.0,
    "min_roi_width_mm": 160.0,
    "min_roi_height_mm": 50.0,
    "x_padding_mm": 5.0,
    "auto_crop": True,
    "show_measurements": True,
    "x_range_mm": None,
    "title": None,
    "dpi": 150,
}

PROFILE_MARGIN_LEFT = 0.12
PROFILE_MARGIN_RIGHT = 0.97
PROFILE_MARGIN_TOP = 0.95
PROFILE_MARGIN_BOTTOM = 0.11
PROFILE_HSPACE = 0.26
PROFILE_CBAR_HEIGHT_IN = 0.24
PROFILE_BOTTOM_HEIGHT_IN = 2.6
PROFILE_MIN_TOP_HEIGHT_IN = 1.25
PROFILE_SERIF_FAMILY = ["Times New Roman", "STIX Two Text", "STIXGeneral", "DejaVu Serif"]
PROFILE_TITLE_SIZE = 18
PROFILE_LABEL_SIZE = 16
PROFILE_TICK_SIZE = 13
PROFILE_CBAR_LABEL_SIZE = 15
PROFILE_CBAR_TICK_SIZE = 12


@dataclass(frozen=True)
class ProfileCompositeContext:
    """Fully resolved ProfileComposite render input."""

    eta: np.ndarray | None
    frame_idx: int | None
    frame_name: str | None
    run_id: str | None
    profile_line: ProfileLineLike | None
    viz_params: dict[str, Any]
    px_per_mm: float
    strip_mm: float
    y_range_mm: float
    min_roi_width_mm: float
    min_roi_height_mm: float
    x_padding_mm: float
    x_range_mm: tuple[float, float] | None
    auto_crop: bool
    line_color: str
    profile_color: str
    show_measurements: bool
    body_polygon_rc: list[tuple[float, float]] | None
    body_source: BodySource
    degraded_reason: str | None
    spatial_calibration_source: str = "fallback_unverified"
    spatial_calibration: dict[str, Any] | None = None
    batch: str = "default"


@dataclass(frozen=True)
class ProfileMeasurement:
    wavelength_mm: float | None = None
    height_mm: float | None = None
    body_span_mm: tuple[float, float] | None = None


@dataclass(frozen=True)
class ProfileWindow:
    eta: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray
    profile_mm: np.ndarray


@dataclass(frozen=True)
class SpatialCalibration:
    pixel_per_mm: float
    source: str
    calibration: dict[str, Any]
    degraded_reason: str | None = None


def _get(viz: Any, name: str, default: Any = None) -> Any:
    if isinstance(viz, dict):
        return viz.get(name, default)
    return getattr(viz, name, default)


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", "auto"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive(value: Any, default: float) -> float:
    parsed = _float_or_none(value)
    return parsed if parsed is not None and parsed > 0 else default


def _nonnegative(value: Any, default: float) -> float:
    parsed = _float_or_none(value)
    return parsed if parsed is not None and parsed >= 0 else default


def _calibration_from_mapping(data: dict[str, Any], source: str) -> SpatialCalibration | None:
    px = _float_or_none(data.get("pixel_per_mm") or data.get("pixel_per_mm_median"))
    if px is None or px <= 0:
        return None
    calibration = dict(data)
    calibration["pixel_per_mm"] = float(px)
    return SpatialCalibration(
        pixel_per_mm=float(px),
        source=str(calibration.get("spatial_calibration_source") or source),
        calibration=calibration,
    )


def _manifest_calibration(run_manifest: dict[str, Any] | None) -> SpatialCalibration | None:
    if not run_manifest:
        return None
    calibration = run_manifest.get("calibration")
    if isinstance(calibration, dict):
        return _calibration_from_mapping(calibration, "manifest")
    return _calibration_from_mapping(run_manifest, "manifest")


def resolve_spatial_calibration(
    result_store: Any,
    batch: str,
    frame_idx: int | None,
    run_manifest: dict[str, Any] | None = None,
    project: Any | None = None,
) -> SpatialCalibration:
    """Resolve spatial calibration for Profile x/y coordinates."""
    if frame_idx is not None:
        try:
            attrs = result_store.read_frame_attrs(batch, int(frame_idx))
            found = _calibration_from_mapping(dict(attrs), "frame_attrs")
            if found is not None:
                return found
        except Exception:
            pass
    try:
        meta = result_store.read_batch_meta(batch)
        data = dict(meta)
        if data.get("calibration_json"):
            try:
                data.update(json.loads(str(data["calibration_json"])))
            except json.JSONDecodeError:
                pass
        found = _calibration_from_mapping(data, "batch_attrs")
        if found is not None:
            return found
    except Exception:
        pass
    found = _manifest_calibration(run_manifest)
    if found is not None:
        return found
    fallback = getattr(getattr(project, "profile", None), "px_per_mm", None)
    px = _float_or_none(fallback)
    if px is not None and px > 0:
        return SpatialCalibration(
            pixel_per_mm=float(px),
            source="fallback_unverified",
            calibration={
                "pixel_per_mm": float(px),
                "spatial_calibration_source": "fallback_unverified",
            },
            degraded_reason="Spatial calibration was not stored with this run; using current project fallback.",
        )
    return SpatialCalibration(
        pixel_per_mm=float(PROFILE_COMPOSITE_DEFAULTS["px_per_mm"]),
        source="fallback_unverified",
        calibration={
            "pixel_per_mm": float(PROFILE_COMPOSITE_DEFAULTS["px_per_mm"]),
            "spatial_calibration_source": "fallback_unverified",
        },
        degraded_reason="Spatial calibration was not stored with this run; using unverified default.",
    )


def _coerce_spatial_calibration(value: Any) -> SpatialCalibration | None:
    if isinstance(value, SpatialCalibration):
        return value
    if isinstance(value, dict):
        return _calibration_from_mapping(value, str(value.get("spatial_calibration_source") or "provided"))
    return None


def _line_points(profile_line: Any) -> ProfileLineLike | None:
    if profile_line is None:
        return None
    try:
        if hasattr(profile_line, "p0") and hasattr(profile_line, "p1"):
            p0 = profile_line.p0
            p1 = profile_line.p1
        else:
            p0, p1 = profile_line
        return (
            (float(p0[0]), float(p0[1])),
            (float(p1[0]), float(p1[1])),
        )
    except (TypeError, ValueError, IndexError):
        return None


def parse_profile_x_range(value: Any) -> tuple[float, float] | None:
    """Parse final display x range in profile x-mm coordinates."""
    if value in (None, "", "auto"):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        xmin = _float_or_none(value[0])
        xmax = _float_or_none(value[1])
        if xmin is not None and xmax is not None and xmin < xmax:
            return xmin, xmax
        return None
    if isinstance(value, str) and ":" in value:
        left, right = value.split(":", 1)
        xmin = _float_or_none(left)
        xmax = _float_or_none(right)
        if xmin is not None and xmax is not None and xmin < xmax:
            return xmin, xmax
        return None
    numeric = _float_or_none(value)
    if numeric is not None and numeric > 0:
        return -numeric / 2.0, numeric / 2.0
    return None


def format_profile_x_range(value: Any) -> str:
    parsed = parse_profile_x_range(value)
    if parsed is None:
        return ""
    xmin, xmax = parsed
    return f"{xmin:g}:{xmax:g}"


def _sample_nearest(eta: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    h, w = eta.shape
    rr = np.clip(np.rint(rows).astype(int), 0, h - 1)
    cc = np.clip(np.rint(cols).astype(int), 0, w - 1)
    return eta[rr, cc].astype(float)


def _sample_nearest_unclipped(eta: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    h, w = eta.shape
    values = np.full(rows.shape, np.nan, dtype=float)
    valid = (rows >= 0) & (rows <= h - 1) & (cols >= 0) & (cols <= w - 1)
    if np.any(valid):
        rr = np.rint(rows[valid]).astype(int)
        cc = np.rint(cols[valid]).astype(int)
        values[valid] = eta[rr, cc].astype(float)
    return values


def _nanmean_no_warning(values: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(values)
    counts = finite.sum(axis=axis)
    sums = np.where(finite, values, 0.0).sum(axis=axis)
    out = np.full(sums.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def resolve_body_polygon(annotation: Any, frame_name: str | None) -> tuple[list[tuple[float, float]] | None, BodySource]:
    """Resolve visible body polygon using the shared precedence contract."""
    if annotation is not None and frame_name:
        for poly in getattr(annotation, "frame_polygons", {}).get(frame_name, []):
            if getattr(poly, "visible", False):
                return [(float(r), float(c)) for r, c in poly.vertices], "frame_polygon"
    if annotation is not None:
        for poly in getattr(annotation, "polygons", []):
            if getattr(poly, "visible", False):
                return [(float(r), float(c)) for r, c in poly.vertices], "global_polygon"
    return None, "fallback"


def build_profile_composite_context(
    *,
    eta: np.ndarray | None,
    profile_line: Any,
    viz_params: dict[str, Any] | None = None,
    frame_idx: int | None = None,
    frame_name: str | None = None,
    run_id: str | None = None,
    batch: str = "default",
    body_polygon_rc: list[tuple[float, float]] | None = None,
    body_source: BodySource = "fallback",
    degraded_reason: str | None = None,
    spatial_calibration: SpatialCalibration | dict[str, Any] | None = None,
) -> ProfileCompositeContext:
    """Validate loose scene/export inputs into the typed render context."""
    defaults = PROFILE_COMPOSITE_DEFAULTS
    params = dict(defaults)
    params.update(dict(viz_params or {}))
    resolved_calibration = _coerce_spatial_calibration(spatial_calibration)
    if resolved_calibration is None:
        legacy_px = _positive(params.get("px_per_mm"), float(defaults.get("px_per_mm", 1.0)))
        resolved_calibration = SpatialCalibration(
            pixel_per_mm=legacy_px,
            source="fallback_unverified",
            calibration={
                "pixel_per_mm": legacy_px,
                "spatial_calibration_source": "fallback_unverified",
            },
            degraded_reason="Spatial calibration was not supplied; using unverified fallback.",
        )
    px_per_mm = resolved_calibration.pixel_per_mm
    strip_mm = _nonnegative(params.get("strip_mm"), float(defaults.get("strip_mm", 2.0)))
    y_range_mm = _positive(params.get("y_range_mm"), float(defaults.get("y_range_mm", 30.0)))
    min_roi_width_mm = _positive(
        params.get("min_roi_width_mm"),
        float(defaults.get("min_roi_width_mm", 160.0)),
    )
    min_roi_height_mm = _positive(
        params.get("min_roi_height_mm"),
        float(defaults.get("min_roi_height_mm", 50.0)),
    )
    x_padding_mm = _nonnegative(
        params.get("x_padding_mm"),
        float(defaults.get("x_padding_mm", 5.0)),
    )
    line = _line_points(profile_line)

    reason = degraded_reason
    if eta is None or np.asarray(eta).size == 0:
        reason = reason or "No eta frame"
    elif line is None:
        reason = reason or "Draw a profile line on the eta frame first"
    elif body_source == "fallback" and body_polygon_rc is None:
        reason = reason or "No annotation geometry; using fallback body detection"
    if resolved_calibration.degraded_reason:
        reason = (
            f"{reason}; {resolved_calibration.degraded_reason}"
            if reason
            else resolved_calibration.degraded_reason
        )

    return ProfileCompositeContext(
        eta=None if eta is None else np.asarray(eta, dtype=float),
        frame_idx=frame_idx,
        frame_name=frame_name,
        run_id=run_id,
        batch=batch,
        profile_line=line,
        viz_params=params,
        px_per_mm=px_per_mm,
        strip_mm=strip_mm,
        y_range_mm=y_range_mm,
        min_roi_width_mm=min_roi_width_mm,
        min_roi_height_mm=min_roi_height_mm,
        x_padding_mm=x_padding_mm,
        x_range_mm=parse_profile_x_range(params.get("x_range_mm")),
        auto_crop=bool(params.get("auto_crop", True)),
        line_color=str(params.get("line_color") or "red"),
        profile_color=str(params.get("profile_color") or "blue"),
        show_measurements=bool(params.get("show_measurements", True)),
        body_polygon_rc=body_polygon_rc,
        body_source=body_source,
        degraded_reason=reason,
        spatial_calibration_source=resolved_calibration.source,
        spatial_calibration=resolved_calibration.calibration,
    )


def extract_line_profile(
    eta: np.ndarray,
    profile_line: Any,
    *,
    px_per_mm: float = 1.0,
    strip_mm: float = 0.0,
    samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    eta = np.asarray(eta, dtype=float)
    if eta.ndim != 2 or eta.size == 0:
        raise ValueError("eta must be a non-empty 2-D array")
    line = _line_points(profile_line)
    if line is None:
        raise ValueError("profile_line is required")
    (r0, c0), (r1, c1) = line
    dr = r1 - r0
    dc = c1 - c0
    length_px = float(np.hypot(dr, dc))
    if length_px <= 0:
        raise ValueError("profile_line has zero length")

    px_per_mm = max(float(px_per_mm or 1.0), 1e-9)
    if samples is None:
        samples = max(2, int(round(length_px)) + 1)
    rows = np.linspace(r0, r1, samples)
    cols = np.linspace(c0, c1, samples)

    strip_px = max(0.0, float(strip_mm or 0.0) * px_per_mm)
    if strip_px <= 1.0:
        values = _sample_nearest(eta, rows, cols)
    else:
        half = strip_px / 2.0
        count = max(3, int(round(strip_px)) + 1)
        offsets = np.linspace(-half, half, count)
        normal_r = -dc / length_px
        normal_c = dr / length_px
        samples_2d = [
            _sample_nearest(eta, rows + offset * normal_r, cols + offset * normal_c)
            for offset in offsets
        ]
        values = _nanmean_no_warning(np.vstack(samples_2d), axis=0)

    x_mm = np.linspace(0.0, length_px / px_per_mm, samples)
    x_mm = x_mm - float(np.nanmean(x_mm))
    return x_mm, values


def _line_basis(line: ProfileLineLike) -> tuple[float, float, float, float, float, float, float]:
    (r0, c0), (r1, c1) = line
    dr = r1 - r0
    dc = c1 - c0
    length_px = float(np.hypot(dr, dc))
    if length_px <= 0:
        raise ValueError("profile_line has zero length")
    center_r = 0.5 * (r0 + r1)
    center_c = 0.5 * (c0 + c1)
    unit_r = dr / length_px
    unit_c = dc / length_px
    return center_r, center_c, unit_r, unit_c, -unit_c, unit_r, length_px


def extract_profile_window(
    eta: np.ndarray,
    profile_line: Any,
    *,
    px_per_mm: float = 1.0,
    strip_mm: float = 2.0,
    y_range_mm: float = 50.0,
    x_range_mm: Any = None,
    x_padding_mm: float = 5.0,
    min_roi_width_mm: float = 160.0,
    min_roi_height_mm: float = 50.0,
    auto_crop: bool = True,
) -> ProfileWindow:
    """Extract a profile-aligned ETA ROI.
    """
    eta = np.asarray(eta, dtype=float)
    if eta.ndim != 2 or eta.size == 0:
        raise ValueError("eta must be a non-empty 2-D array")
    line = _line_points(profile_line)
    if line is None:
        raise ValueError("profile_line is required")

    px_per_mm = max(float(px_per_mm or 1.0), 1e-9)
    center_r, center_c, unit_r, unit_c, normal_r, normal_c, length_px = _line_basis(line)
    length_mm = length_px / px_per_mm
    requested_x_range = parse_profile_x_range(x_range_mm)
    requested_width_mm = 0.0
    if requested_x_range is not None:
        requested_width_mm = max(0.0, float(requested_x_range[1] - requested_x_range[0]))
    width_mm = max(
        length_mm + 2.0 * max(0.0, float(x_padding_mm or 0.0)),
        requested_width_mm,
        max(1.0, float(min_roi_width_mm or 160.0)),
    )
    height_mm = max(
        max(1.0, float(y_range_mm or 50.0)),
        max(1.0, float(min_roi_height_mm or 50.0)),
    )
    x_mm = np.linspace(-width_mm / 2.0, width_mm / 2.0, max(3, int(round(width_mm * px_per_mm)) + 1))
    y_mm = np.linspace(-height_mm / 2.0, height_mm / 2.0, max(3, int(round(height_mm * px_per_mm)) + 1))

    rows = center_r + x_mm[None, :] * px_per_mm * unit_r + y_mm[:, None] * px_per_mm * normal_r
    cols = center_c + x_mm[None, :] * px_per_mm * unit_c + y_mm[:, None] * px_per_mm * normal_c
    window = _sample_nearest_unclipped(eta, rows, cols)

    if auto_crop and window.size:
        n_chk = max(2, window.shape[0] // 8)
        col_ok = (
            np.mean(np.isfinite(window[:n_chk, :]), axis=0) > 0.90
        ) & (
            np.mean(np.isfinite(window[-n_chk:, :]), axis=0) > 0.90
        )
        valid_cols = np.where(col_ok)[0]
        if len(valid_cols) >= 3:
            lo, hi = int(valid_cols[0]), int(valid_cols[-1]) + 1
            window = window[:, lo:hi]
            x_mm = x_mm[lo:hi]
        row_ok = np.mean(np.isfinite(window), axis=1) > 0.5
        valid_rows = np.where(row_ok)[0]
        if len(valid_rows) >= 3:
            lo, hi = int(valid_rows[0]), int(valid_rows[-1]) + 1
            window = window[lo:hi, :]
            y_mm = y_mm[lo:hi]

    strip_half = max(0.0, float(strip_mm or 0.0) / 2.0)
    if strip_half <= 0:
        center_idx = int(np.argmin(np.abs(y_mm)))
        profile = window[center_idx, :].astype(float)
    else:
        strip_mask = np.abs(y_mm) <= strip_half
        if not np.any(strip_mask):
            strip_mask = np.abs(y_mm) == np.min(np.abs(y_mm))
        profile = _nanmean_no_warning(window[strip_mask, :], axis=0)
    return ProfileWindow(eta=window, x_mm=x_mm, y_mm=y_mm, profile_mm=profile)


def _crop_window(
    window: ProfileWindow,
    display_x_range: tuple[float, float] | None,
    display_y_range_mm: float | None,
) -> ProfileWindow:
    x_mask = np.ones(window.x_mm.shape, dtype=bool)
    y_mask = np.ones(window.y_mm.shape, dtype=bool)
    if display_x_range is not None:
        xmin, xmax = display_x_range
        proposed = (window.x_mm >= xmin) & (window.x_mm <= xmax)
        if np.count_nonzero(proposed) >= 2:
            x_mask = proposed
    if display_y_range_mm is not None and display_y_range_mm > 0:
        half = display_y_range_mm / 2.0
        proposed = (window.y_mm >= -half) & (window.y_mm <= half)
        if np.count_nonzero(proposed) >= 2:
            y_mask = proposed
    return ProfileWindow(
        eta=window.eta[np.ix_(y_mask, x_mask)],
        x_mm=window.x_mm[x_mask],
        y_mm=window.y_mm[y_mask],
        profile_mm=window.profile_mm[x_mask],
    )


def _largest_nan_span(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    mask = ~np.isfinite(y)
    if not mask.any():
        return None
    best: tuple[int, int] | None = None
    start: int | None = None
    for idx, val in enumerate(mask):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            if best is None or idx - start > best[1] - best[0]:
                best = (start, idx - 1)
            start = None
    if start is not None and (best is None or len(mask) - start > best[1] - best[0]):
        best = (start, len(mask) - 1)
    if best is None:
        return None
    return float(x[best[0]]), float(x[best[1]])


def _body_bbox_from_polygon(
    context: ProfileCompositeContext,
) -> tuple[float, float, float, float] | None:
    line = context.profile_line
    polygon = context.body_polygon_rc
    if line is None or not polygon:
        return None
    center_r, center_c, unit_r, unit_c, normal_r, normal_c, _ = _line_basis(line)
    points = np.asarray(polygon, dtype=float)
    rel_r = points[:, 0] - center_r
    rel_c = points[:, 1] - center_c
    x = (rel_r * unit_r + rel_c * unit_c) / context.px_per_mm
    y = (rel_r * normal_r + rel_c * normal_c) / context.px_per_mm
    return float(np.nanmin(x)), float(np.nanmax(x)), float(np.nanmin(y)), float(np.nanmax(y))


def _nan_body_bbox(window: ProfileWindow) -> tuple[float, float, float, float] | None:
    nan_mask = ~np.isfinite(window.eta)
    if not nan_mask.any():
        return None
    col_nan = np.mean(nan_mask, axis=0)
    row_nan = np.mean(nan_mask, axis=1)
    cols = np.where(col_nan > 0.20)[0]
    rows = np.where(row_nan > 0.05)[0]
    if len(cols) < 2 or len(rows) < 2:
        return None
    return (
        float(window.x_mm[int(cols[0])]),
        float(window.x_mm[int(cols[-1])]),
        float(window.y_mm[int(rows[0])]),
        float(window.y_mm[int(rows[-1])]),
    )


def measure_profile_sides(
    x_mm: np.ndarray,
    eta_mm: np.ndarray,
    body_span_mm: tuple[float, float] | None = None,
) -> ProfileMeasurement:
    finite = np.isfinite(x_mm) & np.isfinite(eta_mm)
    if finite.sum() < 3:
        return ProfileMeasurement(body_span_mm=body_span_mm or _largest_nan_span(x_mm, eta_mm))
    x = x_mm[finite]
    y = eta_mm[finite]
    height = float(np.nanmax(y) - np.nanmin(y))
    wavelength: float | None = None
    if len(y) >= 3:
        peaks = np.where((y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
        if len(peaks) >= 2:
            wavelength = float(np.nanmedian(np.diff(x[peaks])))
    return ProfileMeasurement(
        wavelength_mm=wavelength,
        height_mm=height if np.isfinite(height) and height > 0 else None,
        body_span_mm=body_span_mm or _largest_nan_span(x_mm, eta_mm),
    )


def _auto_limits(eta: np.ndarray, vmin: Any, vmax: Any) -> tuple[float | None, float | None]:
    lo = _float_or_none(vmin)
    hi = _float_or_none(vmax)
    if lo is not None or hi is not None:
        return lo, hi
    finite = eta[np.isfinite(eta)]
    if finite.size == 0:
        return None, None
    span = float(np.nanpercentile(np.abs(finite), 98))
    if span <= 0:
        span = 1.0
    return -span, span


def _figure_size_from_widget(fig: matplotlib.figure.Figure, dpi: float) -> tuple[float, float] | None:
    canvas = getattr(fig, "canvas", None)
    if canvas is None or not hasattr(canvas, "width") or not hasattr(canvas, "height"):
        return None
    width_px = max(int(canvas.width()), 0)
    height_px = max(int(canvas.height()), 0)
    if width_px <= 1 or height_px <= 1:
        return None
    return width_px / dpi, height_px / dpi


def _ensure_figure_height(
    fig_width: float,
    fig_height: float,
    x_span: float,
    y_span: float,
) -> float:
    width_frac = PROFILE_MARGIN_RIGHT - PROFILE_MARGIN_LEFT
    height_frac = PROFILE_MARGIN_TOP - PROFILE_MARGIN_BOTTOM
    top_height_in = max(
        PROFILE_MIN_TOP_HEIGHT_IN,
        fig_width * width_frac * (y_span / max(x_span, 1e-9)),
    )
    needed_height_in = (
        top_height_in
        + PROFILE_CBAR_HEIGHT_IN
        + PROFILE_BOTTOM_HEIGHT_IN
    ) / max(height_frac * (1.0 - PROFILE_HSPACE * 2), 1e-6)
    return max(fig_height, needed_height_in)


def _layout_rects(
    fig_width: float,
    fig_height: float,
    x_span: float,
    y_span: float,
    *,
    include_colorbar: bool = True,
) -> dict[str, list[float]]:
    width_frac = PROFILE_MARGIN_RIGHT - PROFILE_MARGIN_LEFT
    available_frac = PROFILE_MARGIN_TOP - PROFILE_MARGIN_BOTTOM
    gap_frac = 0.045 if include_colorbar else 0.032
    gap_count = 2 if include_colorbar else 1

    top_height_frac = max(
        PROFILE_MIN_TOP_HEIGHT_IN / max(fig_height, 1e-9),
        width_frac * fig_width * (y_span / max(x_span, 1e-9)) / max(fig_height, 1e-9),
    )
    cbar_height_frac = PROFILE_CBAR_HEIGHT_IN / max(fig_height, 1e-9) if include_colorbar else 0.0
    profile_height_frac = max(PROFILE_BOTTOM_HEIGHT_IN / max(fig_height, 1e-9), 0.18)

    total_height_frac = top_height_frac + cbar_height_frac + profile_height_frac + gap_frac * gap_count
    if total_height_frac > available_frac:
        shrinkable = max(top_height_frac + cbar_height_frac + profile_height_frac, 1e-9)
        scale = max((available_frac - gap_frac * gap_count) / shrinkable, 0.4)
        top_height_frac *= scale
        cbar_height_frac *= scale
        profile_height_frac *= scale

    top_y = PROFILE_MARGIN_TOP - top_height_frac
    if include_colorbar:
        cbar_y = top_y - gap_frac - cbar_height_frac
        profile_y = cbar_y - gap_frac - profile_height_frac
    else:
        cbar_y = 0.0
        profile_y = top_y - gap_frac - profile_height_frac
    return {
        "map": [PROFILE_MARGIN_LEFT, top_y, width_frac, top_height_frac],
        "cbar": [PROFILE_MARGIN_LEFT + width_frac * 0.20, cbar_y, width_frac * 0.40, cbar_height_frac],
        "profile": [PROFILE_MARGIN_LEFT, profile_y, width_frac, profile_height_frac],
    }


def _empty_figure(fig: matplotlib.figure.Figure, message: str) -> matplotlib.figure.Figure:
    ax = fig.add_subplot(111)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontfamily=PROFILE_SERIF_FAMILY,
        fontsize=PROFILE_LABEL_SIZE,
    )
    ax.set_axis_off()
    return fig


def _apply_profile_typography(ax_map, ax_profile, cbar) -> None:
    for ax in (ax_map, ax_profile):
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(PROFILE_SERIF_FAMILY)
            label.set_fontsize(PROFILE_TICK_SIZE)
        ax.xaxis.label.set_fontfamily(PROFILE_SERIF_FAMILY)
        ax.yaxis.label.set_fontfamily(PROFILE_SERIF_FAMILY)
        ax.xaxis.label.set_fontsize(PROFILE_LABEL_SIZE)
        ax.yaxis.label.set_fontsize(PROFILE_LABEL_SIZE)
        ax.title.set_fontfamily(PROFILE_SERIF_FAMILY)
        ax.title.set_fontsize(PROFILE_TITLE_SIZE)
    if cbar is not None:
        cbar.ax.xaxis.label.set_fontfamily(PROFILE_SERIF_FAMILY)
        cbar.ax.xaxis.label.set_fontsize(PROFILE_CBAR_LABEL_SIZE)
        for label in cbar.ax.get_xticklabels():
            label.set_fontfamily(PROFILE_SERIF_FAMILY)
            label.set_fontsize(PROFILE_CBAR_TICK_SIZE)


def render_profile_composite_context(
    context: ProfileCompositeContext,
    *,
    fig: matplotlib.figure.Figure | None = None,
    frame_label: str | None = None,
    layout_mode: str = "preview",
) -> matplotlib.figure.Figure:
    """Render heatmap + profile from an already resolved context."""
    dpi = float(context.viz_params.get("dpi", 150) or 150)
    if fig is None:
        fig = matplotlib.figure.Figure(figsize=(7.0, 5.0), dpi=dpi)
    else:
        fig.clear()

    fallback_w = float(context.viz_params.get("figure_width", 8.6) or 8.6)
    fallback_h = float(context.viz_params.get("figure_height", 5.8) or 5.8)
    colorbar_mode = str(context.viz_params.get("colorbar", "bottom") or "bottom").lower()
    if colorbar_mode == "right":
        colorbar_mode = "bottom"
    if colorbar_mode not in {"bottom", "none"}:
        colorbar_mode = "bottom"
    if context.eta is None or context.eta.size == 0:
        widget_size = _figure_size_from_widget(fig, dpi)
        fig_w, fig_h = widget_size if widget_size is not None else (fallback_w, fallback_h)
        fig.set_size_inches(fig_w, fig_h, forward=True)
        fig.set_dpi(dpi)
        return _empty_figure(fig, context.degraded_reason or "No eta frame")
    if context.profile_line is None:
        widget_size = _figure_size_from_widget(fig, dpi)
        fig_w, fig_h = widget_size if widget_size is not None else (fallback_w, fallback_h)
        fig.set_size_inches(fig_w, fig_h, forward=True)
        fig.set_dpi(dpi)
        return _empty_figure(fig, context.degraded_reason or "Draw a profile line on the eta frame first")

    full_window = extract_profile_window(
        context.eta,
        context.profile_line,
        px_per_mm=context.px_per_mm,
        strip_mm=context.strip_mm,
        y_range_mm=context.y_range_mm,
        x_range_mm=context.x_range_mm,
        x_padding_mm=context.x_padding_mm,
        min_roi_width_mm=context.min_roi_width_mm,
        min_roi_height_mm=context.min_roi_height_mm,
        auto_crop=context.auto_crop,
    )
    body_bbox = _body_bbox_from_polygon(context)
    if body_bbox is None and context.body_source == "fallback":
        body_bbox = _nan_body_bbox(full_window)
    body_span = None if body_bbox is None else (body_bbox[0], body_bbox[1])

    window = _crop_window(full_window, context.x_range_mm, context.y_range_mm)
    x_mm = window.x_mm
    y_mm = window.profile_mm
    measurement = measure_profile_sides(x_mm, y_mm, body_span_mm=body_span)

    cmap = context.viz_params.get("cmap", "RdBu_r") or "RdBu_r"
    vmin, vmax = _auto_limits(window.eta, context.viz_params.get("vmin"), context.viz_params.get("vmax"))
    title = context.viz_params.get("title") or frame_label or (
        f"Profile Composite - frame {context.frame_idx}" if context.frame_idx is not None else "Profile Composite"
    )

    x_span = max(abs(float(window.x_mm[-1] - window.x_mm[0])), 1e-9)
    y_span = max(abs(float(window.y_mm[-1] - window.y_mm[0])), 1e-9)
    widget_size = _figure_size_from_widget(fig, dpi) if layout_mode == "preview" else None
    if widget_size is not None:
        fig_w, fig_h = widget_size
    else:
        fig_w = fallback_w
        fig_h = _ensure_figure_height(fallback_w, fallback_h, x_span, y_span)
    fig.set_size_inches(fig_w, fig_h, forward=True)
    fig.set_dpi(dpi)
    rects = _layout_rects(fig_w, fig_h, x_span, y_span, include_colorbar=colorbar_mode == "bottom")
    ax_map = fig.add_axes(rects["map"])
    ax_cbar = fig.add_axes(rects["cbar"]) if colorbar_mode == "bottom" else None
    ax_profile = fig.add_axes(rects["profile"], sharex=ax_map)

    cmap_obj = colormaps.get_cmap(str(cmap)).copy()
    cmap_obj.set_bad("white")
    extent = (
        float(window.x_mm[0]),
        float(window.x_mm[-1]),
        float(window.y_mm[-1]),
        float(window.y_mm[0]),
    )
    im = ax_map.imshow(
        window.eta,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect="equal",
        origin="upper",
        interpolation="bilinear",
    )
    ax_map.set_aspect("equal", adjustable="box", anchor="C")
    ax_map.set_xlim(extent[0], extent[1])
    ax_map.set_ylim(extent[2], extent[3])
    if body_bbox is not None:
        x0, x1, y0, y1 = body_bbox
        if x1 >= window.x_mm[0] and x0 <= window.x_mm[-1]:
            ax_map.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    facecolor="0.55",
                    edgecolor="white",
                    linewidth=1.0,
                    zorder=4,
                )
            )
    ax_map.axhline(0.0, linestyle="--", color=context.line_color, linewidth=1.2, zorder=6)
    ax_map.set_title(str(title))
    ax_map.set_ylabel("offset y (mm)")
    if context.spatial_calibration_source == "fallback_unverified":
        ax_map.text(
            0.01,
            0.96,
            "Unverified spatial calibration",
            ha="left",
            va="top",
            transform=ax_map.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
            zorder=8,
        )
    ax_map.tick_params(labelbottom=False, labelsize=PROFILE_TICK_SIZE, pad=4)
    ax_map.margins(x=0, y=0)
    cbar = None
    if ax_cbar is not None:
        cbar = fig.colorbar(im, cax=ax_cbar, orientation="horizontal")
        cbar.set_label(r"$\eta$ (mm)", labelpad=6)
        cbar.ax.tick_params(labelsize=PROFILE_CBAR_TICK_SIZE, pad=2)

    ax_profile.plot(x_mm, y_mm, color=context.profile_color, linewidth=1.6)
    ax_profile.axhline(0.0, color="0.55", linewidth=0.8)
    ax_profile.set_xlabel("x (mm)")
    ax_profile.set_ylabel("Wave height η (mm)")
    ax_profile.grid(True, alpha=0.22)
    ax_profile.tick_params(labelsize=PROFILE_TICK_SIZE, pad=4)

    if measurement.body_span_mm is not None:
        ax_profile.axvspan(*measurement.body_span_mm, color="0.55", alpha=0.85, zorder=0)
    ax_profile.set_xlim(float(window.x_mm[0]), float(window.x_mm[-1]))
    _apply_profile_typography(ax_map, ax_profile, cbar)
    return fig


def render_profile_composite(
    eta: np.ndarray | ProfileCompositeContext | None,
    profile_line: Any = None,
    viz: Any | None = None,
    *,
    fig: matplotlib.figure.Figure | None = None,
    frame_label: str | None = None,
) -> matplotlib.figure.Figure:
    """Compatibility wrapper around the typed context renderer."""
    if isinstance(eta, ProfileCompositeContext):
        return render_profile_composite_context(
            eta,
            fig=fig,
            frame_label=frame_label,
            layout_mode=getattr(viz, "layout_mode", "preview") if viz is not None else "preview",
        )
    params = dict(viz or {}) if isinstance(viz, dict) else dict(getattr(viz, "__dict__", {}) if viz is not None else {})
    context = build_profile_composite_context(
        eta=eta,
        profile_line=profile_line,
        viz_params=params,
        frame_idx=getattr(viz, "frame_idx", None) if viz is not None else None,
        frame_name=getattr(viz, "frame_name", None) if viz is not None else None,
        run_id=getattr(viz, "run_id", None) if viz is not None else None,
        body_polygon_rc=getattr(viz, "body_polygon_rc", None) if viz is not None else None,
        body_source=getattr(viz, "body_source", "fallback") if viz is not None else "fallback",
        degraded_reason=getattr(viz, "degraded_reason", None) if viz is not None else None,
        spatial_calibration=getattr(viz, "spatial_calibration", None) if viz is not None else None,
    )
    return render_profile_composite_context(
        context,
        fig=fig,
        frame_label=frame_label,
        layout_mode=getattr(viz, "layout_mode", "preview") if viz is not None else "preview",
    )
