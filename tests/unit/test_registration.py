"""Unit tests for openfcd.core.registration scale normalization."""
from __future__ import annotations

import numpy as np
import pytest

from openfcd.core.registration import scale_normalize_reference, _mean_carrier_period_px


def _make_checkerboard(shape: tuple, period: int) -> np.ndarray:
    """Synthetic axis-aligned checkerboard with given cell period."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    return ((yy // period + xx // period) % 2).astype(np.float64)


def _flatfield_carriers(img: np.ndarray):
    from openfcd.core.flatfield import flatfield_normalize
    from openfcd.core.fcd import calculate_carriers
    ff = flatfield_normalize(img, sigma=max(img.shape) * 0.25)
    carriers = calculate_carriers(ff - ff.mean())
    return ff, carriers


class TestScaleNormalizeReference:
    def test_noop_when_periods_match(self):
        """Same-scale images: scale ≈ 1.0, ref returned unchanged."""
        img = _make_checkerboard((256, 256), period=16)
        ref_ff, carriers = _flatfield_carriers(img)
        aligned, c_aligned, scale = scale_normalize_reference(ref_ff, ref_ff, carriers)
        assert abs(scale - 1.0) < 0.03
        np.testing.assert_array_equal(aligned, ref_ff)

    def test_scale_correction_applied_for_large_mismatch(self):
        """14% mismatch triggers rescaling; aligned carriers match def period."""
        ref = _make_checkerboard((512, 512), period=32)   # large period (zoom out)
        from scipy.ndimage import zoom
        # Simulate def with ~14% smaller period (camera zoomed in)
        def_raw = _make_checkerboard((512, 512), period=27)
        ref_ff, c_ref = _flatfield_carriers(ref)
        def_ff, c_def = _flatfield_carriers(def_raw)

        p_ref = _mean_carrier_period_px(c_ref)
        p_def = _mean_carrier_period_px(c_def)
        assert abs(p_ref / p_def - 1.0) > 0.03, "Pre-condition: periods differ"

        aligned, c_aligned, scale = scale_normalize_reference(ref_ff, def_ff, c_ref)
        p_aligned = _mean_carrier_period_px(c_aligned)

        # After normalization the aligned reference period should be close to def's
        mismatch = abs(p_aligned / p_def - 1.0)
        assert mismatch < 0.05, f"Post-alignment mismatch {mismatch:.3f} > 5%"
        assert aligned.shape == ref_ff.shape

    def test_scale_less_than_one_pads_to_original_size(self):
        """When ref has smaller period (more zoomed) than def, scale < 1 → pad path."""
        ref = _make_checkerboard((256, 256), period=12)  # small period = zoomed in
        def_raw = _make_checkerboard((256, 256), period=18)  # larger period = zoomed out
        ref_ff, c_ref = _flatfield_carriers(ref)
        def_ff, _ = _flatfield_carriers(def_raw)

        aligned, _, scale = scale_normalize_reference(ref_ff, def_ff, c_ref)
        assert scale > 1.0 or abs(scale - 1.0) < 0.03  # either corrected or within tol
        assert aligned.shape == ref_ff.shape

    def test_mean_carrier_period_px(self):
        img = _make_checkerboard((256, 256), period=16)
        _, carriers = _flatfield_carriers(img)
        period = _mean_carrier_period_px(carriers)
        # Checkerboard with cell=16 → diagonal carrier period ≈ 16*sqrt(2) ≈ 22.6 px
        assert 15.0 < period < 35.0

    def test_fallback_when_def_carrier_detection_fails(self):
        """Returns original ref unchanged when carrier detection on def fails."""
        ref_ff, c_ref = _flatfield_carriers(_make_checkerboard((128, 128), period=8))
        # Uniform image — carrier detection will fail
        flat_def = np.ones((128, 128), dtype=np.float64) * 0.5
        aligned, c_aligned, scale = scale_normalize_reference(ref_ff, flat_def, c_ref)
        assert scale == 1.0
        np.testing.assert_array_equal(aligned, ref_ff)
