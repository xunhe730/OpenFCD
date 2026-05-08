from __future__ import annotations

import numpy as np

from openfcd.pipeline.compute import (
    eta_confidence_mask,
    fill_small_eta_holes,
    poisson_residual_diagnostics,
    repair_eta_confidence_artifacts,
    suppress_nonphysical_eta_filaments,
)


def test_suppress_nonphysical_eta_filaments_reduces_narrow_jump_line() -> None:
    size = 128
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    eta = np.sin((y - 64) / 7.0) + 0.15 * np.sin((x - 64) / 11.0)
    eta[:, 60:62] += 3.0

    low_signal = np.zeros_like(eta, dtype=bool)
    low_signal[:, 59:63] = True

    fixed = suppress_nonphysical_eta_filaments(eta, low_signal_mask=low_signal)

    before = float(np.mean(np.abs(eta[:, 61] - eta[:, 59])))
    after = float(np.mean(np.abs(fixed[:, 61] - fixed[:, 59])))

    assert after < before * 0.4


def test_suppress_nonphysical_eta_filaments_handles_diagonal_artifact() -> None:
    size = 160
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    eta = 0.8 * np.sin((y - 80) / 9.0) + 0.2 * np.sin((x - 80) / 13.0)

    for offset in range(18, 92):
        row = offset
        col = int(0.7 * offset) + 18
        eta[max(0, row - 1):min(size, row + 2), max(0, col - 1):min(size, col + 2)] += 2.5

    low_signal = np.zeros_like(eta, dtype=bool)
    low_signal[88:120, 62:94] = True

    fixed = suppress_nonphysical_eta_filaments(eta, low_signal_mask=low_signal)

    before = float(np.nanmax(eta) - np.nanmin(eta))
    after = float(np.nanmax(fixed) - np.nanmin(fixed))
    assert after < before * 0.85


def test_suppress_nonphysical_eta_filaments_repairs_anchored_glint_dipole() -> None:
    size = 180
    y, x = np.mgrid[0:size, 0:size]
    wave = np.sin(np.hypot(x - 90, y - 90) / 7.0)
    eta = wave.copy()
    pos = (x - 102) ** 2 + (y - 88) ** 2 <= 3 ** 2
    neg = (x - 110) ** 2 + (y - 92) ** 2 <= 3 ** 2
    eta[pos] += 5.0
    eta[neg] -= 5.0

    low_signal = np.zeros_like(eta, dtype=bool)
    low_signal[82:98, 96:116] = True

    fixed = suppress_nonphysical_eta_filaments(eta, low_signal_mask=low_signal)

    artifact = pos | neg
    before = float(np.mean(np.abs(eta[artifact] - wave[artifact])))
    after = float(np.mean(np.abs(fixed[artifact] - wave[artifact])))
    outside = ~low_signal
    outside_change = float(np.mean(np.abs(fixed[outside] - eta[outside])))

    assert after < before * 0.35
    assert outside_change < 0.02


def test_suppress_nonphysical_eta_filaments_leaves_unanchored_dipole() -> None:
    size = 180
    y, x = np.mgrid[0:size, 0:size]
    eta = np.sin(np.hypot(x - 90, y - 90) / 7.0)
    pos = (x - 102) ** 2 + (y - 88) ** 2 <= 3 ** 2
    neg = (x - 110) ** 2 + (y - 92) ** 2 <= 3 ** 2
    eta[pos] += 5.0
    eta[neg] -= 5.0

    low_signal = np.zeros_like(eta, dtype=bool)
    low_signal[20:36, 20:36] = True

    fixed = suppress_nonphysical_eta_filaments(eta, low_signal_mask=low_signal)

    np.testing.assert_allclose(fixed[pos | neg], eta[pos | neg])


def test_eta_confidence_mask_repairs_finite_phase_glint_dipole() -> None:
    size = 180
    y, x = np.mgrid[0:size, 0:size]
    wave = np.sin(np.hypot(x - 92, y - 88) / 8.0)
    eta = wave.copy()
    pos = (x - 102) ** 2 + (y - 84) ** 2 <= 4 ** 2
    neg = (x - 113) ** 2 + (y - 91) ** 2 <= 4 ** 2
    eta[pos] += 6.0
    eta[neg] -= 6.0

    phase_residual = np.zeros_like(eta)
    phase_residual[pos | neg] = 1.4
    carrier_amp = np.ones_like(eta)

    artifact_mask, debug = eta_confidence_mask(
        eta,
        carrier_amplitude_map=carrier_amp,
        phase_residual=phase_residual,
        return_debug=True,
    )
    fixed = repair_eta_confidence_artifacts(eta, artifact_mask)

    artifact = pos | neg
    before = float(np.mean(np.abs(eta[artifact] - wave[artifact])))
    after = float(np.mean(np.abs(fixed[artifact] - wave[artifact])))

    assert artifact_mask[artifact].mean() > 0.7
    assert debug["phase_bad"][artifact].mean() > 0.7
    assert after < before * 0.35


def test_eta_confidence_mask_leaves_unanchored_physical_waves() -> None:
    size = 220
    y, x = np.mgrid[0:size, 0:size]
    eta = np.sin(np.hypot(x - 110, y - 105) / 7.0)
    eta += 0.3 * np.sin((0.8 * x + 0.2 * y) / 13.0)

    artifact_mask = eta_confidence_mask(
        eta,
        carrier_amplitude_map=np.ones_like(eta),
        phase_residual=np.zeros_like(eta),
    )

    assert artifact_mask.mean() < 0.001


def test_eta_confidence_mask_repairs_robot_boundary_dipole() -> None:
    size = 180
    y, x = np.mgrid[0:size, 0:size]
    wave = np.sin(np.hypot(x - 90, y - 90) / 8.5)
    eta = wave.copy()
    occlusion = np.zeros_like(eta, dtype=bool)
    occlusion[75:115, 78:112] = True
    eta[occlusion] = np.nan

    pos = ((x - 95) ** 2 + (y - 68) ** 2) <= 4 ** 2
    neg = ((x - 106) ** 2 + (y - 70) ** 2) <= 4 ** 2
    eta[pos] += 5.5
    eta[neg] -= 5.5

    artifact_mask = eta_confidence_mask(
        eta,
        carrier_amplitude_map=np.ones_like(eta),
        phase_residual=np.zeros_like(eta),
        occlusion_mask=occlusion,
    )
    fixed = repair_eta_confidence_artifacts(eta, artifact_mask)

    artifact = pos | neg
    before = float(np.mean(np.abs(eta[artifact] - wave[artifact])))
    after = float(np.mean(np.abs(fixed[artifact] - wave[artifact])))

    assert artifact_mask[artifact].mean() > 0.6
    assert after < before * 0.45
    assert np.isnan(fixed[occlusion]).all()


def test_fill_small_eta_holes_repairs_round_holes() -> None:
    size = 96
    y, x = np.mgrid[0:size, 0:size]
    eta = 0.02 * x + 0.03 * y
    hole = (x - 48) ** 2 + (y - 42) ** 2 <= 3 ** 2
    expected = eta.copy()
    eta[hole] = np.nan

    fixed = fill_small_eta_holes(eta, px_per_mm=8.0, radius_mm=1.0)

    assert np.isfinite(fixed[hole]).all()
    assert np.mean(np.abs(fixed[hole] - expected[hole])) < 0.2


def test_fill_small_eta_holes_keeps_large_mask_nan() -> None:
    eta = np.ones((80, 80), dtype=np.float64)
    eta[30:50, 30:50] = np.nan

    fixed = fill_small_eta_holes(eta, px_per_mm=8.0, radius_mm=1.0)

    assert np.isnan(fixed[30:50, 30:50]).all()


def test_fill_small_eta_holes_caps_user_radius_to_protect_robot_mask() -> None:
    eta = np.ones((160, 160), dtype=np.float64)
    eta[60:100, 60:100] = np.nan

    fixed = fill_small_eta_holes(eta, px_per_mm=8.0, radius_mm=10.0)

    assert np.isnan(fixed[60:100, 60:100]).all()


def test_fill_small_eta_holes_keeps_wire_like_nan() -> None:
    eta = np.ones((90, 90), dtype=np.float64)
    eta[42:44, 20:70] = np.nan

    fixed = fill_small_eta_holes(eta, px_per_mm=20.0, radius_mm=1.0)

    assert np.isnan(fixed[42:44, 20:70]).all()


def test_fill_small_eta_holes_keeps_edge_connected_nan() -> None:
    eta = np.ones((80, 80), dtype=np.float64)
    eta[:6, :] = np.nan
    eta[10:14, 10:14] = np.nan

    fixed = fill_small_eta_holes(eta, px_per_mm=8.0, radius_mm=1.0)

    assert np.isnan(fixed[:6, :]).all()
    assert np.isfinite(fixed[10:14, 10:14]).all()


def test_fill_small_eta_holes_zero_radius_is_noop() -> None:
    eta = np.ones((40, 40), dtype=np.float64)
    eta[18:21, 18:21] = np.nan

    fixed = fill_small_eta_holes(eta, px_per_mm=8.0, radius_mm=0.0)

    np.testing.assert_array_equal(np.isnan(fixed), np.isnan(eta))


def test_poisson_residual_diagnostics_reports_zero_for_consistent_slopes() -> None:
    size = 96
    px_per_mm = 8.0
    y, x = np.mgrid[:size, :size]
    eta = 0.2 * np.sin(2 * np.pi * x / 48.0) * np.sin(2 * np.pi * y / 48.0)
    sy, sx = np.gradient(eta, 1.0 / px_per_mm, 1.0 / px_per_mm, edge_order=1)

    diag = poisson_residual_diagnostics(eta, sx, sy, px_per_mm=px_per_mm)

    interior = (slice(2, -2), slice(2, -2))
    assert np.nanmax(diag["poisson_residual"][interior]) < 1e-12
    assert abs(float(diag["poisson_residual_rms"])) < 1e-12
    assert np.nanmax(np.abs(diag["curl_inconsistency"][interior])) < 0.02


def test_poisson_residual_diagnostics_flags_inconsistent_slopes() -> None:
    size = 64
    px_per_mm = 8.0
    y, x = np.mgrid[:size, :size]
    eta = 0.1 * np.sin(2 * np.pi * x / 32.0) * np.sin(2 * np.pi * y / 32.0)
    sy, sx = np.gradient(eta, 1.0 / px_per_mm, 1.0 / px_per_mm, edge_order=1)
    sx_bad = sx + 0.05 * (y / size)

    diag = poisson_residual_diagnostics(eta, sx_bad, sy, px_per_mm=px_per_mm)

    assert float(diag["poisson_residual_rms"]) > 0.02
    assert np.nanmax(np.abs(diag["curl_inconsistency"])) > 0.005
