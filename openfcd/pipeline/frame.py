"""Pure per-frame FCD compute.

`compute_frame(FrameInputs) -> FrameResult` is the single algorithmic path
shared by GUI single-frame preview and CLI/GUI multi-frame Run. The function
is pure modulo two documented side-effect channels (``progress_cb`` and
``cancel``) — no Session/Store/HDF5/GUI imports anywhere in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:  # typing-only imports — keep frame.py side-effect free
    from pathlib import Path

    from openfcd.core.mask import Box, Polygon
    from openfcd.io.annotation import AnnotationSchema
    from openfcd.io.project import ProjectModel
    from openfcd.pipeline.base import CancelToken
    from openfcd.pipeline.compute import GeomParams


@dataclass(frozen=True)
class ProcessParams:
    """Narrow projection of ProjectModel.process actually read by compute_frame."""

    flatfield_sigma_auto: bool
    flatfield_sigma: float
    taper_alpha: float
    edge_nan_mm: float
    detrend: str
    auto_scale_ref: bool
    highpass_sigma_px: float
    small_hole_fill_radius_mm: float

    @classmethod
    def from_project(cls, project: "ProjectModel") -> "ProcessParams":
        p = project.process
        return cls(
            flatfield_sigma_auto=bool(getattr(p, "flatfield_sigma_auto", True)),
            flatfield_sigma=float(p.flatfield_sigma),
            taper_alpha=float(p.taper.alpha),
            edge_nan_mm=float(p.edge_nan_mm),
            detrend=str(p.detrend),
            auto_scale_ref=bool(getattr(p, "auto_scale_ref", True)),
            highpass_sigma_px=float(getattr(p, "highpass_sigma_px", 0.0)),
            small_hole_fill_radius_mm=float(
                getattr(p, "small_hole_fill_radius_mm", 1.0)
            ),
        )


@dataclass(frozen=True)
class FrameInputs:
    """Per-frame compute package — one per `compute_frame` call."""

    ref_img: np.ndarray
    def_img: np.ndarray
    geom: "GeomParams"
    process: ProcessParams
    roi_box: "Box | None"
    robot_poly: "Polygon | None"
    fast_preview: bool
    ref_shape: tuple[int, int]


@dataclass(frozen=True)
class FrameResult:
    """Output of compute_frame. eta_mm is always embedded in ref_shape."""

    eta_mm: np.ndarray
    qc_datasets: dict[str, np.ndarray] | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def pixel_per_mm(self) -> float:
        return float(self.diagnostics.get("pixel_per_mm", 0.0))

    @property
    def calibration(self) -> dict:
        return dict(self.diagnostics.get("calibration", {}))

    @property
    def qc_attrs(self) -> dict:
        return dict(self.diagnostics.get("qc_attrs", {}))


@dataclass(frozen=True)
class FrameInputsSnapshot:
    """Frame-invariant context built once per Run / per preview session."""

    process: ProcessParams
    geom: "GeomParams"
    ref_img: np.ndarray
    ref_shape: tuple[int, int]
    annotation: "AnnotationSchema | None"
    roi_box: "Box | None"

    @classmethod
    def from_run_context(cls, ctx: dict) -> "FrameInputsSnapshot":
        project = ctx["project"]
        ref_img = ctx["reference_image"]
        geom = ctx["geom_params"]
        annotation = ctx.get("annotation")
        roi_box = _annotation_roi_box(annotation)
        return cls(
            process=ProcessParams.from_project(project),
            geom=geom,
            ref_img=ref_img,
            ref_shape=tuple(ref_img.shape),  # type: ignore[arg-type]
            annotation=annotation,
            roi_box=roi_box,
        )

    @classmethod
    def from_session(
        cls,
        session_annotation: "AnnotationSchema | None",
        project: "ProjectModel",
        geom: "GeomParams",
        ref_img: np.ndarray,
    ) -> "FrameInputsSnapshot":
        return cls(
            process=ProcessParams.from_project(project),
            geom=geom,
            ref_img=ref_img,
            ref_shape=tuple(ref_img.shape),  # type: ignore[arg-type]
            annotation=session_annotation,
            roi_box=_annotation_roi_box(session_annotation),
        )


def _annotation_roi_box(annotation: "AnnotationSchema | None") -> "Box | None":
    if annotation is None:
        return None
    roi = annotation.roi
    if roi.is_empty:
        return None
    from openfcd.core.mask import Box

    return Box(
        row0=int(roi.y),
        col0=int(roi.x),
        height=int(roi.height),
        width=int(roi.width),
    )


def _resolve_frame_polygon(
    annotation: "AnnotationSchema | None", frame_name: str
) -> "Polygon | None":
    if annotation is None:
        return None
    polys = annotation.frame_polygons.get(frame_name, [])
    if polys and polys[0].vertices:
        from openfcd.core.mask import Polygon

        return Polygon([(float(v[0]), float(v[1])) for v in polys[0].vertices])
    return None


def build_frame_inputs(
    snapshot: FrameInputsSnapshot,
    frame_path: "Path",
    *,
    fast_preview: bool,
    def_img: np.ndarray | None = None,
) -> FrameInputs:
    """Build per-frame inputs. Loads `def_img` from disk when not supplied."""
    from openfcd.pipeline.compute import load_gray

    deformed = def_img if def_img is not None else load_gray(frame_path)
    robot_poly = _resolve_frame_polygon(snapshot.annotation, frame_path.name)
    return FrameInputs(
        ref_img=snapshot.ref_img,
        def_img=deformed,
        geom=snapshot.geom,
        process=snapshot.process,
        roi_box=snapshot.roi_box,
        robot_poly=robot_poly,
        fast_preview=fast_preview,
        ref_shape=snapshot.ref_shape,
    )


def compute_frame(
    inputs: FrameInputs,
    *,
    progress_cb: Callable[[int, str], None] | None = None,
    cancel: "CancelToken | None" = None,
) -> FrameResult:
    """Run the FCD pipeline on a single (ref, def) pair.

    Pure modulo `progress_cb` (UI animation hook) and `cancel` (cooperative
    cancellation). Returns η embedded in `inputs.ref_shape`.
    """
    from openfcd.cli.cmd_run import _frame_calibration
    from openfcd.core.fcd import (
        calculate_carriers,
        carrier_amplitude,
        carriers_pixel_per_mm,
        fcd_displacement,
        fftinvgrad,
        periodic_smooth_decompose,
    )
    from openfcd.core.flatfield import flatfield_normalize
    from openfcd.core.inpaint import inpaint_fft, synthesize_from_carriers
    from openfcd.core.mask import (
        Box,
        Polygon,
        auto_mask,
        detect_filament_occluders,
        find_largest_interior_blob,
        find_oriented_polygon,
        polygon_mask,
    )
    from openfcd.pipeline.compute import (
        carrier_phase_residual,
        cosine_taper,
        detrend_plane,
        edge_margin_mask,
        eta_confidence_mask,
        fill_small_eta_holes,
        poisson_residual_diagnostics,
        repair_eta_confidence_artifacts,
        suppress_nonphysical_eta_filaments,
    )

    ref_img = inputs.ref_img
    def_img = inputs.def_img
    geom = inputs.geom
    process = inputs.process
    roi_box = inputs.roi_box
    robot_poly = inputs.robot_poly
    fast_preview = inputs.fast_preview
    robot_pad_px = 4

    def _report(pct: int, label: str) -> None:
        if cancel is not None:
            cancel.check()
        if progress_cb is not None:
            progress_cb(pct, label)

    if process.flatfield_sigma_auto:
        h_img, w_img = ref_img.shape
        sigma = float(np.clip(max(h_img, w_img) * 0.06, 100.0, 2000.0))
    else:
        sigma = float(process.flatfield_sigma)
    taper_alpha = process.taper_alpha
    edge_mm = process.edge_nan_mm
    small_hole_radius_mm = process.small_hole_fill_radius_mm
    same_input = np.array_equal(ref_img, def_img)

    _original_shape: tuple | None = None
    _downsample_scale = 1.0
    if fast_preview:
        h0, w0 = ref_img.shape
        factor = min(2, max(h0, w0) // 2500)
        if factor >= 2:
            from scipy.ndimage import zoom

            scale = 1.0 / factor
            _downsample_scale = scale
            ref_img = zoom(ref_img, scale, order=1)
            def_img = zoom(def_img, scale, order=1)
            sigma = sigma * scale
            if robot_poly is not None:
                robot_poly = Polygon(
                    [(v[0] * scale, v[1] * scale) for v in robot_poly.vertices]
                )
            if roi_box is not None:
                roi_box = Box(
                    row0=int(roi_box.row0 * scale),
                    col0=int(roi_box.col0 * scale),
                    height=int(roi_box.height * scale),
                    width=int(roi_box.width * scale),
                )
            _original_shape = (h0, w0)

    _report(5, "Setup")

    if roi_box is not None:
        r0 = max(0, roi_box.row0)
        c0 = max(0, roi_box.col0)
        r1 = min(ref_img.shape[0], roi_box.row0 + roi_box.height)
        c1 = min(ref_img.shape[1], roi_box.col0 + roi_box.width)
        ref_img = ref_img[r0:r1, c0:c1]
        r1d = min(def_img.shape[0], roi_box.row0 + roi_box.height)
        c1d = min(def_img.shape[1], roi_box.col0 + roi_box.width)
        def_img = def_img[r0:r1d, c0:c1d]
        if robot_poly is not None:
            robot_poly = robot_poly.shifted(-r0, -c0)

    _report(10, "Flatfield")
    ref_ff = flatfield_normalize(ref_img, sigma=sigma)
    def_ff = flatfield_normalize(def_img, sigma=sigma, bg_src=ref_img)
    finite_def = np.isfinite(def_img)
    if finite_def.any():
        max_level = 255.0 if float(np.nanmax(def_img)) > 1.5 else 1.0
        saturated = finite_def & ((def_img <= 0.0) | (def_img >= 0.995 * max_level))
        saturated_ratio = float(saturated.sum() / finite_def.size)
    else:
        saturated_ratio = 1.0

    _report(30, "Carriers")
    carriers0 = calculate_carriers(ref_ff - ref_ff.mean())

    if process.auto_scale_ref:
        from openfcd.core.registration import scale_normalize_reference

        ref_ff, carriers0, _scale, _valid_crop = scale_normalize_reference(
            ref_ff, def_ff, carriers0
        )
        if _valid_crop is not None:
            r0v, c0v, hv, wv = _valid_crop
            def_ff = def_ff[r0v : r0v + hv, c0v : c0v + wv]
            if robot_poly is not None:
                robot_poly = robot_poly.shifted(-r0v, -c0v)

    filament_mask = detect_filament_occluders(ref_ff, carriers0)
    carrier_loss_mask = auto_mask(def_ff, carriers0, threshold_ratio=0.2, dilate_px=4)
    carrier_amp_map = carrier_amplitude(def_ff, carriers0)

    _report(45, "Inpaint")
    if robot_poly is not None:
        occlusion_mask = polygon_mask(ref_ff.shape, robot_poly, dilate_px=robot_pad_px)
        occlusion_mask |= filament_mask
        syn = synthesize_from_carriers(ref_ff - ref_ff.mean(), carriers0)
        ref_clean = inpaint_fft(ref_ff, occlusion_mask, syn + ref_ff.mean())
        carriers = calculate_carriers(ref_clean - ref_clean.mean())
        syn2 = synthesize_from_carriers(ref_clean - ref_clean.mean(), carriers)
        def_clean = inpaint_fft(def_ff, occlusion_mask, syn2 + ref_clean.mean())
    else:
        auto_poly = find_oriented_polygon(carrier_loss_mask, edge_margin=20)
        if auto_poly is None:
            auto_box = find_largest_interior_blob(carrier_loss_mask, edge_margin=20)
            if auto_box is not None:
                auto_poly = Polygon.from_box(auto_box)
        if auto_poly is not None:
            occlusion_mask = polygon_mask(
                ref_ff.shape, auto_poly, dilate_px=robot_pad_px
            )
            occlusion_mask |= filament_mask
            syn = synthesize_from_carriers(ref_ff - ref_ff.mean(), carriers0)
            ref_clean = inpaint_fft(ref_ff, occlusion_mask, syn + ref_ff.mean())
            carriers = calculate_carriers(ref_clean - ref_clean.mean())
            syn2 = synthesize_from_carriers(ref_clean - ref_clean.mean(), carriers)
            def_clean = inpaint_fft(def_ff, occlusion_mask, syn2 + ref_clean.mean())
        else:
            occlusion_mask = filament_mask.copy()
            ref_clean = ref_ff
            def_clean = def_ff
            carriers = carriers0
    glint_anchor_mask = occlusion_mask | filament_mask | carrier_loss_mask

    def _finalize_eta(eta: np.ndarray, px_per_mm: float) -> tuple[np.ndarray, float]:
        if _original_shape is not None:
            _report(97, "Upsample")
            from scipy.ndimage import zoom as _zoom

            valid = np.isfinite(eta)
            filled = np.where(valid, eta, 0.0)
            up = round(1.0 / _downsample_scale)
            eta_up = _zoom(filled, up, order=1)
            valid_up = _zoom(valid.astype(np.float32), up, order=0) > 0.5
            eta = np.where(valid_up, eta_up, np.nan)
            px_per_mm = px_per_mm / _downsample_scale
        if small_hole_radius_mm > 0:
            _report(98, "Fill holes")
            eta = fill_small_eta_holes(
                eta, px_per_mm=px_per_mm, radius_mm=small_hole_radius_mm
            )
        return eta, float(px_per_mm)

    def _finalize_qc_map(arr: np.ndarray) -> np.ndarray:
        if _original_shape is None:
            return arr
        from scipy.ndimage import zoom as _zoom

        up = round(1.0 / _downsample_scale)
        order = 0 if arr.dtype == bool else 1
        out = _zoom(
            arr.astype(np.float32) if arr.dtype == bool else arr, up, order=order
        )
        if arr.dtype == bool:
            return out > 0.5
        return out

    _report(65, "Taper/PeriodicBC")
    if taper_alpha > 0:
        win = cosine_taper(ref_clean.shape, alpha=taper_alpha)
        ref_mean = ref_clean.mean()
        def_mean = def_clean.mean()
        ref_clean = (ref_clean - ref_mean) * win + ref_mean
        def_clean = (def_clean - def_mean) * win + def_mean
    else:
        ref_clean, _ = periodic_smooth_decompose(ref_clean)
        def_clean, _ = periodic_smooth_decompose(def_clean)

    if same_input:
        px_per_mm = carriers_pixel_per_mm(carriers, geom.pattern_period_mm)
        eta_mm = np.zeros(ref_clean.shape, dtype=np.float64)
        eta_mm[occlusion_mask] = np.nan
        if edge_mm > 0:
            em_px = int(edge_mm * px_per_mm)
            eta_mm[edge_margin_mask(eta_mm.shape, em_px)] = np.nan
        eta_mm, px_per_mm = _finalize_eta(eta_mm, px_per_mm)
        valid_mask = np.isfinite(eta_mm)
        qc_datasets = {
            "carrier_amplitude": _finalize_qc_map(carrier_amp_map),
            "valid_mask": valid_mask,
            "artifact_mask": np.zeros_like(valid_mask, dtype=bool),
            "phase_residual": np.zeros_like(eta_mm, dtype=np.float64),
            "poisson_residual": np.zeros_like(eta_mm, dtype=np.float64),
            "poisson_residual_x": np.zeros_like(eta_mm, dtype=np.float64),
            "poisson_residual_y": np.zeros_like(eta_mm, dtype=np.float64),
            "curl_inconsistency": np.zeros_like(eta_mm, dtype=np.float64),
        }
        qc_attrs = {
            "saturated_ratio": saturated_ratio,
            "invalid_ratio": float((~valid_mask).sum() / valid_mask.size),
            "carrier_amp_median": float(np.nanmedian(qc_datasets["carrier_amplitude"])),
            "poisson_residual_rms": 0.0,
            "curl_inconsistency_rms": 0.0,
            "phase_residual_rms": 0.0,
            "slope_rms": 0.0,
        }
        _report(100, "Done")
        eta_full = _embed_eta(eta_mm, inputs.ref_shape, inputs.roi_box)
        qc_full = {
            name: _embed_qc(value, inputs.ref_shape, inputs.roi_box)
            for name, value in qc_datasets.items()
        }
        return FrameResult(
            eta_mm=eta_full,
            qc_datasets=qc_full,
            diagnostics={
                "pixel_per_mm": float(px_per_mm),
                "calibration": _frame_calibration(px_per_mm, geom),
                "qc_attrs": qc_attrs,
            },
        )

    _report(70, "FCD demodulate")
    disp_u, disp_v = fcd_displacement(def_clean - ref_clean.mean(), carriers, unwrap=False)
    raw_eta = fftinvgrad(-disp_u, -disp_v)
    raw_eta[occlusion_mask] = np.nan

    _report(80, "Detrend")
    if process.detrend == "plane":
        valid = np.isfinite(raw_eta) & ~edge_margin_mask(raw_eta.shape, 20)
        if valid.sum() > 100:
            raw_eta = detrend_plane(raw_eta, valid=valid)

    _report(85, "Calibrate")
    px_per_mm = carriers_pixel_per_mm(carriers, geom.pattern_period_mm)
    eta_mm = raw_eta / (geom.alpha * geom.h_p_eff_mm * px_per_mm**2)

    if edge_mm > 0:
        em_px = int(edge_mm * px_per_mm)
        eta_mm[edge_margin_mask(eta_mm.shape, em_px)] = np.nan

    hp_sigma = process.highpass_sigma_px * _downsample_scale
    if hp_sigma > 0.0:
        _report(93, "Highpass")
        from scipy.ndimage import gaussian_filter

        finite = np.isfinite(eta_mm)
        filled = np.where(finite, eta_mm, 0.0)
        weight = gaussian_filter(finite.astype(np.float32), sigma=hp_sigma)
        trend = gaussian_filter(filled, sigma=hp_sigma) / np.maximum(weight, 1e-6)
        eta_mm = np.where(finite, eta_mm - trend, np.nan)
        hp_margin = int(round(1.5 * hp_sigma))
        if hp_margin > 0:
            eta_mm[edge_margin_mask(eta_mm.shape, hp_margin)] = np.nan

    _report(94, "Confidence mask")
    phase_residual = carrier_phase_residual(def_clean - ref_clean.mean(), carriers)
    artifact_mask = eta_confidence_mask(
        eta_mm,
        carrier_amplitude_map=carrier_amp_map,
        phase_residual=phase_residual,
        anchor_mask=glint_anchor_mask,
        occlusion_mask=occlusion_mask,
    )
    if artifact_mask.any():
        eta_mm = repair_eta_confidence_artifacts(eta_mm, artifact_mask)
        glint_anchor_mask = glint_anchor_mask | artifact_mask

    _report(95, "Glint suppress")
    eta_mm = suppress_nonphysical_eta_filaments(
        eta_mm, low_signal_mask=glint_anchor_mask
    )

    measured_sx = -disp_u / (geom.alpha * geom.h_p_eff_mm * px_per_mm)
    measured_sy = -disp_v / (geom.alpha * geom.h_p_eff_mm * px_per_mm)
    eta_mm, px_per_mm = _finalize_eta(eta_mm, px_per_mm)
    measured_sx = _finalize_qc_map(measured_sx)
    measured_sy = _finalize_qc_map(measured_sy)
    diagnostics = poisson_residual_diagnostics(
        eta_mm, measured_sx, measured_sy, px_per_mm=px_per_mm
    )

    valid_mask = np.isfinite(eta_mm)
    qc_datasets = {
        "carrier_amplitude": _finalize_qc_map(carrier_amp_map),
        "valid_mask": valid_mask,
        "artifact_mask": _finalize_qc_map(artifact_mask),
        "phase_residual": _finalize_qc_map(phase_residual),
        "poisson_residual": _finalize_qc_map(np.asarray(diagnostics["poisson_residual"])),
        "poisson_residual_x": _finalize_qc_map(np.asarray(diagnostics["poisson_residual_x"])),
        "poisson_residual_y": _finalize_qc_map(np.asarray(diagnostics["poisson_residual_y"])),
        "curl_inconsistency": _finalize_qc_map(np.asarray(diagnostics["curl_inconsistency"])),
    }
    curl = qc_datasets["curl_inconsistency"]
    curl_valid = np.isfinite(curl)
    phase_valid = np.isfinite(phase_residual)
    slope_mag = np.sqrt(measured_sx**2 + measured_sy**2)
    slope_valid = np.isfinite(slope_mag)
    qc_attrs = {
        "saturated_ratio": saturated_ratio,
        "invalid_ratio": float((~valid_mask).sum() / valid_mask.size),
        "carrier_amp_median": float(np.nanmedian(qc_datasets["carrier_amplitude"])),
        "poisson_residual_rms": float(diagnostics["poisson_residual_rms"]),
        "curl_inconsistency_rms": (
            float(np.sqrt(np.nanmean(curl[curl_valid] ** 2))) if curl_valid.any() else float("nan")
        ),
        "phase_residual_rms": (
            float(np.sqrt(np.nanmean(phase_residual[phase_valid] ** 2)))
            if phase_valid.any() else float("nan")
        ),
        "slope_rms": (
            float(np.sqrt(np.nanmean(slope_mag[slope_valid] ** 2)))
            if slope_valid.any() else float("nan")
        ),
    }
    _report(100, "Done")
    eta_full = _embed_eta(eta_mm, inputs.ref_shape, inputs.roi_box)
    qc_full = {
        name: _embed_qc(value, inputs.ref_shape, inputs.roi_box)
        for name, value in qc_datasets.items()
    }
    return FrameResult(
        eta_mm=eta_full,
        qc_datasets=qc_full,
        diagnostics={
            "pixel_per_mm": float(px_per_mm),
            "calibration": _frame_calibration(px_per_mm, geom),
            "qc_attrs": qc_attrs,
        },
    )


def _embed_eta(
    eta_mm: np.ndarray,
    ref_shape: tuple[int, int],
    roi_box: "Box | None",
) -> np.ndarray:
    h_ref, w_ref = ref_shape
    if eta_mm.shape == ref_shape:
        return eta_mm
    eta_full = np.full((h_ref, w_ref), np.nan, dtype=np.float64)
    if roi_box is not None:
        r0 = max(0, roi_box.row0)
        c0 = max(0, roi_box.col0)
        r1 = min(h_ref, roi_box.row0 + roi_box.height)
        c1 = min(w_ref, roi_box.col0 + roi_box.width)
    else:
        r0, c0, r1, c1 = 0, 0, h_ref, w_ref
    box_h, box_w = r1 - r0, c1 - c0
    eh, ew = eta_mm.shape
    dr = max(0, (box_h - eh) // 2)
    dc = max(0, (box_w - ew) // 2)
    ef_h = min(eh, box_h - dr)
    ef_w = min(ew, box_w - dc)
    if ef_h > 0 and ef_w > 0:
        eta_full[r0 + dr : r0 + dr + ef_h, c0 + dc : c0 + dc + ef_w] = eta_mm[
            :ef_h, :ef_w
        ]
    return eta_full


def _embed_qc(
    arr: np.ndarray,
    ref_shape: tuple[int, int],
    roi_box: "Box | None",
) -> np.ndarray:
    data = np.asarray(arr)
    if data.shape == ref_shape:
        return data
    h_ref, w_ref = ref_shape
    fill = False if data.dtype == bool else np.nan
    out = np.full(
        (h_ref, w_ref),
        fill,
        dtype=data.dtype if data.dtype == bool else np.float64,
    )
    if roi_box is not None:
        r0 = max(0, roi_box.row0)
        c0 = max(0, roi_box.col0)
        r1 = min(h_ref, roi_box.row0 + roi_box.height)
        c1 = min(w_ref, roi_box.col0 + roi_box.width)
    else:
        r0, c0, r1, c1 = 0, 0, h_ref, w_ref
    box_h, box_w = r1 - r0, c1 - c0
    dh, dw = data.shape
    dr = max(0, (box_h - dh) // 2)
    dc = max(0, (box_w - dw) // 2)
    ef_h = min(dh, box_h - dr)
    ef_w = min(dw, box_w - dc)
    if ef_h > 0 and ef_w > 0:
        out[r0 + dr : r0 + dr + ef_h, c0 + dc : c0 + dc + ef_w] = data[:ef_h, :ef_w]
    return out
