import numpy as np
import pytest
from openfcd.core.profile import sample_along


def test_horizontal_line_matches_row():
    eta = np.arange(100, dtype=float).reshape(10, 10)
    p0, p1 = (5.0, 0.0), (5.0, 9.0)
    dists, vals = sample_along(eta, p0, p1, n=10)
    np.testing.assert_array_equal(vals, eta[5, :])


def test_sample_length():
    eta = np.ones((20, 30))
    _, vals = sample_along(eta, (0, 0), (19, 29))
    assert len(vals) == max(20, 30)


def test_nan_preserved():
    eta = np.ones((10, 10))
    eta[5, 5] = np.nan
    dists, vals = sample_along(eta, (5, 0), (5, 9), n=10)
    assert np.isnan(vals[5])


def test_diagonal_length_approx():
    H, W = 100, 100
    eta = np.zeros((H, W))
    p0, p1 = (0.0, 0.0), (99.0, 99.0)
    dists, _ = sample_along(eta, p0, p1)
    expected = np.sqrt(2) * 99
    assert abs(dists[-1] - expected) < 2.0


def test_single_point_no_crash():
    eta = np.ones((10, 10))
    dists, vals = sample_along(eta, (5, 5), (5, 5), n=5)
    assert len(vals) == 5
