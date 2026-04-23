from __future__ import annotations

import numpy as np

from openfcd.pipeline.compute import suppress_nonphysical_eta_filaments


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
