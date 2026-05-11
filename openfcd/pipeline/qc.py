from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from openfcd.io.result import HDF5ResultStore


@dataclass(frozen=True)
class QCThresholds:
    saturated_ratio_warn: float = 0.001
    invalid_ratio_warn: float = 0.20
    carrier_amp_median_min_warn: float = 1e-6
    poisson_norm_warn: float = 0.10
    poisson_norm_fail: float = 0.20
    curl_norm_warn: float = 0.10
    curl_norm_fail: float = 0.20
    max_abs_phase_warn: float = 0.7 * np.pi
    max_abs_phase_fail: float = 0.9 * np.pi


def thresholds_from_config(config: Any | None) -> QCThresholds:
    defaults = QCThresholds()
    if config is None:
        return defaults
    data = config.model_dump() if hasattr(config, "model_dump") else dict(config)
    return QCThresholds(
        saturated_ratio_warn=float(data.get("saturated_ratio_warning", defaults.saturated_ratio_warn)),
        invalid_ratio_warn=float(data.get("invalid_ratio_warning", defaults.invalid_ratio_warn)),
        carrier_amp_median_min_warn=float(
            data.get("carrier_amp_median_min_warning", defaults.carrier_amp_median_min_warn)
        ),
        poisson_norm_warn=float(data.get("poisson_norm_warning", defaults.poisson_norm_warn)),
        poisson_norm_fail=float(data.get("poisson_norm_fail", defaults.poisson_norm_fail)),
        curl_norm_warn=float(data.get("curl_norm_warning", defaults.curl_norm_warn)),
        curl_norm_fail=float(data.get("curl_norm_fail", defaults.curl_norm_fail)),
        max_abs_phase_warn=float(data.get("max_abs_phase_warning", defaults.max_abs_phase_warn)),
        max_abs_phase_fail=float(data.get("max_abs_phase_fail", defaults.max_abs_phase_fail)),
    )


def qc_verdict(stats: dict[str, Any], thresholds: QCThresholds | None = None) -> dict[str, Any]:
    """Return PASS/WARN/FAIL and reasons for frame-level QC stats."""
    t = thresholds or QCThresholds()
    severity = "PASS"
    reasons: list[str] = []

    def value(name: str) -> float | None:
        raw = stats.get(name)
        if raw is None:
            return None
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        return val if np.isfinite(val) else None

    def warn_if(name: str, limit: float, label: str) -> None:
        nonlocal severity
        val = value(name)
        if val is not None and val > limit:
            severity = "WARN" if severity == "PASS" else severity
            reasons.append(f"{label} {val:.4g} > warn {limit:.4g}")

    def fail_warn_norm(name: str, warn: float, fail: float, label: str) -> None:
        nonlocal severity
        val = value(name)
        if val is None:
            return
        if val > fail:
            severity = "FAIL"
            reasons.append(f"{label} {val:.4g} > fail {fail:.4g}")
        elif val > warn:
            severity = "WARN" if severity == "PASS" else severity
            reasons.append(f"{label} {val:.4g} > warn {warn:.4g}")

    warn_if("saturated_ratio", t.saturated_ratio_warn, "saturated_ratio")
    warn_if("invalid_ratio", t.invalid_ratio_warn, "invalid_ratio")
    carrier_amp = value("carrier_amp_median")
    if carrier_amp is not None and carrier_amp <= t.carrier_amp_median_min_warn:
        severity = "WARN" if severity == "PASS" else severity
        reasons.append(
            f"carrier_amp_median {carrier_amp:.4g} <= warn {t.carrier_amp_median_min_warn:.4g}"
        )

    slope_rms = value("slope_rms")
    poisson = value("poisson_residual_rms")
    curl = value("curl_inconsistency_rms")
    if slope_rms is not None and slope_rms > 1e-12:
        if poisson is not None:
            stats["poisson_residual_norm"] = poisson / slope_rms
        if curl is not None:
            stats["curl_inconsistency_norm"] = curl / slope_rms
    else:
        if poisson is not None:
            stats["poisson_residual_norm"] = poisson
        if curl is not None:
            stats["curl_inconsistency_norm"] = curl

    fail_warn_norm("poisson_residual_norm", t.poisson_norm_warn, t.poisson_norm_fail, "poisson_residual_norm")
    fail_warn_norm("curl_inconsistency_norm", t.curl_norm_warn, t.curl_norm_fail, "curl_inconsistency_norm")
    fail_warn_norm("max_abs_phase", t.max_abs_phase_warn, t.max_abs_phase_fail, "max_abs_phase")

    phase_residual_rms = value("phase_residual_rms")
    if phase_residual_rms is not None and phase_residual_rms > t.max_abs_phase_warn:
        severity = "WARN" if severity == "PASS" else severity
        reasons.append(
            f"phase_residual_rms {phase_residual_rms:.4g} > warn {t.max_abs_phase_warn:.4g}"
        )

    return {"verdict": severity, "reasons": reasons}


def eta_noise_summary(store: HDF5ResultStore, batch: str = "default") -> dict[str, Any]:
    frame_ids = store.list_frames(batch)
    eta_rms_values: list[float] = []
    frame_means: list[float] = []
    invalid_ratios: list[float] = []
    verdict_counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for fid in frame_ids:
        attrs = store.read_frame_attrs(batch, fid)
        if attrs.get("status") != "ok":
            continue
        eta = np.asarray(store.read_frame(batch, fid), dtype=np.float64)
        finite = np.isfinite(eta)
        if finite.any():
            vals = eta[finite]
            eta_rms_values.append(float(np.sqrt(np.nanmean(vals ** 2))))
            frame_means.append(float(np.nanmean(vals)))
        invalid_ratios.append(float(attrs.get("invalid_ratio", 1.0 - finite.sum() / finite.size)))
        verdict = str(attrs.get("qc_verdict", "PASS"))
        if verdict in verdict_counts:
            verdict_counts[verdict] += 1

    means = np.asarray(frame_means, dtype=np.float64)
    invalid = np.asarray(invalid_ratios, dtype=np.float64)
    rms = np.asarray(eta_rms_values, dtype=np.float64)
    drift = float(np.nanmax(means) - np.nanmin(means)) if means.size else float("nan")
    return {
        "frame_count": int(len(frame_ids)),
        "valid_frame_count": int(rms.size),
        "eta_noise_rms": float(np.sqrt(np.nanmean(rms ** 2))) if rms.size else float("nan"),
        "eta_noise_p95": float(np.nanpercentile(rms, 95)) if rms.size else float("nan"),
        "temporal_drift": drift,
        "invalid_ratio_mean": float(np.nanmean(invalid)) if invalid.size else float("nan"),
        "invalid_ratio_p95": float(np.nanpercentile(invalid, 95)) if invalid.size else float("nan"),
        "qc_verdict_counts": verdict_counts,
    }


def write_noise_summary(summary: dict[str, Any], json_path: Path, h5_path: Path | None = None) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if h5_path is None:
        return
    import h5py
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as h5:
        for key, val in summary.items():
            if isinstance(val, dict):
                grp = h5.create_group(key)
                for sub_key, sub_val in val.items():
                    grp.attrs[sub_key] = sub_val
            else:
                h5.attrs[key] = val


def eta_amplitude_stats(eta: np.ndarray) -> dict[str, float]:
    vals = np.asarray(eta, dtype=np.float64)
    finite = np.isfinite(vals)
    if not finite.any():
        return {"eta_abs_p95": float("nan"), "eta_rms": float("nan"), "eta_peak_to_peak": float("nan")}
    v = vals[finite]
    return {
        "eta_abs_p95": float(np.nanpercentile(np.abs(v), 95)),
        "eta_rms": float(np.sqrt(np.nanmean(v ** 2))),
        "eta_peak_to_peak": float(np.nanmax(v) - np.nanmin(v)),
    }


def parameter_sensitivity_report(
    eta: np.ndarray,
    attrs: dict[str, Any],
    *,
    perturbations: tuple[float, ...] = (-0.05, 0.05),
) -> dict[str, Any]:
    """Estimate amplitude sensitivity by rescaling an already computed eta field."""
    base = eta_amplitude_stats(eta)
    base_checker = float(attrs.get("checker_cell_mm", attrs.get("pattern_period_mm", 1.0)))
    base_px = float(attrs.get("pixel_per_mm", 1.0))
    base_h = float(attrs.get("h_p_eff_mm", 1.0))
    base_alpha = float(attrs.get("alpha", 1.0))
    params = {
        "checker_cell_mm": base_checker,
        "pixel_per_mm": base_px,
        "h_eff": base_h,
        "alpha": base_alpha,
    }
    rows: list[dict[str, float | str]] = []
    for name, base_value in params.items():
        for delta in perturbations:
            factor = 1.0 + float(delta)
            if name == "checker_cell_mm":
                scale = factor ** 2
            elif name == "pixel_per_mm":
                scale = factor ** -2
            else:
                scale = factor ** -1
            stats = eta_amplitude_stats(np.asarray(eta, dtype=np.float64) * scale)
            row: dict[str, float | str] = {
                "parameter": name,
                "relative_delta": float(delta),
                "scale_factor": float(scale),
            }
            row.update(stats)
            rows.append(row)
    return {
        "baseline": base,
        "baseline_parameters": params,
        "perturbations": rows,
    }


def save_qc_summary_figure(
    store: HDF5ResultStore,
    batch: str,
    frame_id: int,
    output: Path,
    *,
    image: np.ndarray | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    eta = store.read_frame(batch, frame_id)
    attrs = store.read_frame_attrs(batch, frame_id)
    qc_names = set(store.list_frame_qc(batch, frame_id))

    def qc_or_nan(name: str) -> np.ndarray:
        if name in qc_names:
            return store.read_frame_qc(batch, frame_id, name)
        return np.full_like(eta, np.nan, dtype=np.float64)

    panels = [
        ("image", image if image is not None else np.full_like(eta, np.nan, dtype=np.float64), "gray"),
        ("eta", eta, "RdBu_r"),
        ("valid_mask", qc_or_nan("valid_mask"), "gray"),
        ("carrier_amplitude", qc_or_nan("carrier_amplitude"), "viridis"),
        ("phase_residual", qc_or_nan("phase_residual"), "magma"),
        ("poisson_residual", qc_or_nan("poisson_residual"), "magma"),
        ("curl_inconsistency", qc_or_nan("curl_inconsistency"), "RdBu_r"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    for ax, (title, arr, cmap) in zip(axes.ravel()[:7], panels):
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax = axes.ravel()[7]
    ax.axis("off")
    lines = [
        f"QC: {attrs.get('qc_verdict', 'UNKNOWN')}",
        f"saturated: {float(attrs.get('saturated_ratio', np.nan)):.4g}",
        f"invalid: {float(attrs.get('invalid_ratio', np.nan)):.4g}",
        f"carrier median: {float(attrs.get('carrier_amp_median', np.nan)):.4g}",
        f"poisson rms: {float(attrs.get('poisson_residual_rms', np.nan)):.4g}",
        f"curl rms: {float(attrs.get('curl_inconsistency_rms', np.nan)):.4g}",
        f"phase residual rms: {float(attrs.get('phase_residual_rms', np.nan)):.4g}",
    ]
    reasons = str(attrs.get("qc_reasons", ""))
    if reasons:
        lines.append(reasons[:180])
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
