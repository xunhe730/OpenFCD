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
        """Same-scale images: scale ≈ 1.0, ref returned unchanged, no crop."""
        img = _make_checkerboard((256, 256), period=16)
        ref_ff, carriers = _flatfield_carriers(img)
        aligned, c_aligned, scale, valid_crop = scale_normalize_reference(ref_ff, ref_ff, carriers)
        assert abs(scale - 1.0) < 0.03
        assert valid_crop is None
        np.testing.assert_array_equal(aligned, ref_ff)

    def test_scale_less_than_one_returns_valid_crop(self):
        """scale < 1: aligned ref is smaller; valid_crop tells caller where to crop def."""
        ref = _make_checkerboard((512, 512), period=32)   # larger period = ref zoomed out
        def_raw = _make_checkerboard((512, 512), period=27)  # smaller period = def more zoomed in
        ref_ff, c_ref = _flatfield_carriers(ref)
        def_ff, c_def = _flatfield_carriers(def_raw)

        p_ref = _mean_carrier_period_px(c_ref)
        p_def = _mean_carrier_period_px(c_def)
        assert abs(p_ref / p_def - 1.0) > 0.03, "Pre-condition: periods differ"

        aligned, c_aligned, scale, valid_crop = scale_normalize_reference(ref_ff, def_ff, c_ref)

        # scale < 1: aligned ref is smaller than original
        assert scale < 1.0
        assert aligned.shape[0] < ref_ff.shape[0]
        assert aligned.shape[1] < ref_ff.shape[1]

        # valid_crop is set and describes the centre slice of def_ff
        assert valid_crop is not None
        r0, c0, hv, wv = valid_crop
        assert aligned.shape == (hv, wv)
        # Crop is centred
        assert abs(r0 - (512 - hv) // 2) <= 1
        assert abs(c0 - (512 - wv) // 2) <= 1

        # Aligned carriers period matches def's period
        p_aligned = _mean_carrier_period_px(c_aligned)
        mismatch = abs(p_aligned / p_def - 1.0)
        assert mismatch < 0.05, f"Post-alignment mismatch {mismatch:.3f} > 5%"

    def test_scale_greater_than_one_no_valid_crop(self):
        """scale > 1: centre-crop path; output is same size as input, no valid_crop."""
        ref = _make_checkerboard((256, 256), period=12)  # small period = zoomed in
        def_raw = _make_checkerboard((256, 256), period=18)  # larger period = zoomed out
        ref_ff, c_ref = _flatfield_carriers(ref)
        def_ff, _ = _flatfield_carriers(def_raw)

        aligned, _, scale, valid_crop = scale_normalize_reference(ref_ff, def_ff, c_ref)
        assert scale > 1.0 or abs(scale - 1.0) < 0.03
        assert aligned.shape == ref_ff.shape
        assert valid_crop is None

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
        aligned, c_aligned, scale, valid_crop = scale_normalize_reference(ref_ff, flat_def, c_ref)
        assert scale == 1.0
        assert valid_crop is None
        np.testing.assert_array_equal(aligned, ref_ff)
