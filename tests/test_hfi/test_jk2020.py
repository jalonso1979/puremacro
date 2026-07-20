"""Tests for puremacro.hfi.jk2020."""
import numpy as np
import pytest

from puremacro.hfi._results import JKResult
from puremacro.hfi.jk2020 import jk_poor_man


def test_jk_poor_man_separates_by_sign():
    rate = np.array([+0.10, -0.05, +0.20, +0.03])
    asset = np.array([-1.0, +0.5, +1.5, -0.4])
    # idx 0: rate>0, asset<0 → MP
    # idx 1: rate<0, asset>0 → MP
    # idx 2: rate>0, asset>0 → info
    # idx 3: rate>0, asset<0 → MP
    res = jk_poor_man(rate, asset)
    assert isinstance(res, JKResult)
    assert res.method == "poor_man"
    np.testing.assert_array_equal(res.mp_shock, [+0.10, -0.05, 0.0, +0.03])
    np.testing.assert_array_equal(res.info_shock, [0.0, 0.0, +0.20, 0.0])


def test_jk_poor_man_zero_rate_zero_shock():
    """Zero rate surprise → no shock attribution either way."""
    rate = np.array([0.0, 0.5, 0.0])
    asset = np.array([1.0, 1.0, -1.0])
    res = jk_poor_man(rate, asset)
    np.testing.assert_array_equal(res.mp_shock, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(res.info_shock, [0.0, 0.5, 0.0])


def test_jk_poor_man_orthogonality_in_support():
    """At every t, exactly one of (mp, info) is non-zero (or both zero)."""
    rng = np.random.default_rng(0)
    rate = rng.standard_normal(50)
    asset = rng.standard_normal(50)
    res = jk_poor_man(rate, asset)
    both_nonzero = (res.mp_shock != 0) & (res.info_shock != 0)
    assert not both_nonzero.any()


from puremacro.hfi.jk2020 import jk_median_target


def test_jk_median_target_returns_orthogonal_rotation():
    rng = np.random.default_rng(0)
    T = 200
    rate = rng.standard_normal(T)
    asset = rng.standard_normal(T)
    res = jk_median_target(rate, asset, n_rotations=2000, seed=0)
    assert res.rotation is not None
    np.testing.assert_allclose(res.rotation @ res.rotation.T, np.eye(2), atol=1e-8)
    assert res.n_admissible > 0


def test_jk_median_target_method_label():
    rng = np.random.default_rng(1)
    res = jk_median_target(rng.standard_normal(100), rng.standard_normal(100),
                           n_rotations=500, seed=0)
    assert res.method == "median_target"


def test_jk_median_target_perfect_negative_correlation_attributes_to_mp():
    """When (rate, asset) are perfectly negatively correlated, the data look
    purely monetary-policy: the info shock should have small variance relative
    to mp."""
    rng = np.random.default_rng(2)
    T = 300
    factor = rng.standard_normal(T)
    rate = factor                # rate up
    asset = -factor              # asset down
    res = jk_median_target(rate, asset, n_rotations=2000, seed=0)
    assert np.var(res.mp_shock) > 5 * np.var(res.info_shock)


def test_jk_median_target_perfect_positive_correlation_attributes_to_info():
    """Symmetric case: rate up + asset up → information shock dominates."""
    rng = np.random.default_rng(3)
    T = 300
    factor = rng.standard_normal(T)
    rate = factor
    asset = factor
    res = jk_median_target(rate, asset, n_rotations=2000, seed=0)
    assert np.var(res.info_shock) > 5 * np.var(res.mp_shock)
