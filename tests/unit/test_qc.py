from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from openfcd.io.result import HDF5ResultStore
from openfcd.pipeline.qc import (
    QCThresholds,
    eta_noise_summary,
    parameter_sensitivity_report,
    qc_verdict,
    save_qc_summary_figure,
    write_noise_summary,
)


def test_qc_verdict_pass_warn_fail() -> None:
    stats = {
        "saturated_ratio": 0.0,
        "invalid_ratio": 0.0,
        "carrier_amp_median": 3.0,
        "poisson_residual_rms": 0.01,
        "curl_inconsistency_rms": 0.01,
        "slope_rms": 1.0,
    }
    assert qc_verdict(dict(stats))["verdict"] == "PASS"

    warn = dict(stats, saturated_ratio=0.002)
    assert qc_verdict(warn)["verdict"] == "WARN"

    low_carrier = dict(stats, carrier_amp_median=0.0)
    assert qc_verdict(low_carrier)["verdict"] == "WARN"

    fail = dict(stats, poisson_residual_rms=0.25)
    result = qc_verdict(fail)
    assert result["verdict"] == "FAIL"
    assert any("poisson_residual_norm" in reason for reason in result["reasons"])


def test_qc_verdict_uses_phase_thresholds_when_available() -> None:
    result = qc_verdict({"max_abs_phase": 0.95 * np.pi}, QCThresholds())
    assert result["verdict"] == "FAIL"


def test_noise_summary_and_writer(tmp_path: Path) -> None:
    h5_path = tmp_path / "results.h5"
    with HDF5ResultStore.open(h5_path, "w") as store:
        store.write_frame(
            "default",
            0,
            np.ones((4, 4)) * 0.1,
            {"status": "ok", "invalid_ratio": 0.0, "qc_verdict": "PASS"},
        )
        eta = np.ones((4, 4)) * 0.2
        eta[0, 0] = np.nan
        store.write_frame(
            "default",
            1,
            eta,
            {"status": "ok", "invalid_ratio": 1 / 16, "qc_verdict": "WARN"},
        )

    with HDF5ResultStore.open(h5_path, "r") as store:
        summary = eta_noise_summary(store)

    assert summary["frame_count"] == 2
    assert summary["valid_frame_count"] == 2
    assert summary["eta_noise_rms"] > 0.0
    assert summary["invalid_ratio_p95"] > 0.0
    assert summary["qc_verdict_counts"]["PASS"] == 1
    assert summary["qc_verdict_counts"]["WARN"] == 1

    json_path = tmp_path / "noise.json"
    out_h5 = tmp_path / "noise.h5"
    write_noise_summary(summary, json_path, out_h5)
    assert json.loads(json_path.read_text())["frame_count"] == 2
    assert out_h5.exists()


def test_parameter_sensitivity_report_scales_amplitude_stats() -> None:
    eta = np.array([[-1.0, 0.0], [0.5, 1.0]])
    report = parameter_sensitivity_report(
        eta,
        {"checker_cell_mm": 1.2, "pixel_per_mm": 8.0, "h_p_eff_mm": 12.0, "alpha": 0.25},
    )
    assert report["baseline"]["eta_peak_to_peak"] == 2.0
    assert {row["parameter"] for row in report["perturbations"]} == {
        "checker_cell_mm",
        "pixel_per_mm",
        "h_eff",
        "alpha",
    }
    checker_plus = [
        row for row in report["perturbations"]
        if row["parameter"] == "checker_cell_mm" and row["relative_delta"] > 0
    ][0]
    assert checker_plus["eta_peak_to_peak"] > report["baseline"]["eta_peak_to_peak"]


def test_qc_summary_figure_writes_png(tmp_path: Path) -> None:
    h5_path = tmp_path / "results.h5"
    qc = {
        "valid_mask": np.ones((8, 8), dtype=bool),
        "carrier_amplitude": np.ones((8, 8)),
        "phase_residual": np.zeros((8, 8)),
        "poisson_residual": np.zeros((8, 8)),
        "curl_inconsistency": np.zeros((8, 8)),
    }
    with HDF5ResultStore.open(h5_path, "w") as store:
        store.write_frame(
            "default",
            0,
            np.ones((8, 8)),
            {"status": "ok", "qc_verdict": "PASS"},
            qc_datasets=qc,
        )
    out = tmp_path / "qc.png"
    with HDF5ResultStore.open(h5_path, "r") as store:
        save_qc_summary_figure(store, "default", 0, out)
    assert out.exists()
    assert out.stat().st_size > 0
