from __future__ import annotations

import numpy as np

from openfcd.core.fcd import calculate_carriers
from openfcd.core.mask import detect_filament_occluders


def _checkerboard(size: int = 128) -> np.ndarray:
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    return (np.sin(2 * np.pi * x / 8) * np.sin(2 * np.pi * y / 8) * 50 + 128).astype(float)


def test_detect_filament_occluders_finds_thin_wire() -> None:
    img = _checkerboard()
    img[20:110, 60:63] *= 0.02

    carriers = calculate_carriers(img - img.mean())
    mask = detect_filament_occluders(img, carriers)

    assert mask.dtype == bool
    assert mask[20:110, 60:63].mean() > 0.5
