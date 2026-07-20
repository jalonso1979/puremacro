import numpy as np
import pytest

from puremacro.tests.unit_root import (
    adf_test, kpss_test, pp_test, zivot_andrews_test,
)


def test_adf_rejects_stationary_ar1():
    rng = np.random.default_rng(11)
    T = 500
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.3 * y[t-1] + rng.standard_normal()
    out = adf_test(y, regression="c")
    assert out["p_value"] < 0.05
    assert out["decision"] in {"reject", "accept"}


def test_adf_fails_to_reject_random_walk():
    rng = np.random.default_rng(13)
    T = 500
    y = np.cumsum(rng.standard_normal(T))
    out = adf_test(y, regression="c")
    assert out["p_value"] > 0.05


def test_kpss_complementary_decision_random_walk():
    """KPSS H_0 is stationarity → on a random walk, KPSS should reject."""
    rng = np.random.default_rng(17)
    T = 500
    y = np.cumsum(rng.standard_normal(T))
    out = kpss_test(y, regression="c")
    assert out["p_value"] < 0.05


def test_kpss_fails_to_reject_stationary():
    rng = np.random.default_rng(19)
    T = 500
    y = rng.standard_normal(T)  # white noise — clearly stationary
    out = kpss_test(y, regression="c")
    assert out["p_value"] > 0.05


def test_pp_rejects_stationary_ar1():
    rng = np.random.default_rng(23)
    T = 500
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.3 * y[t-1] + rng.standard_normal()
    out = pp_test(y, regression="c")
    assert out["p_value"] < 0.05


def test_zivot_andrews_finds_break():
    """Series with a level shift at t=200 — Zivot-Andrews should find a
    break date close to 200."""
    rng = np.random.default_rng(29)
    T = 400
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.3 * y[t-1] + rng.standard_normal()
    y[200:] += 5.0  # level shift
    out = zivot_andrews_test(y, regression="c")
    # Break should be detected within ±20 of true break point
    assert abs(out["break_date"] - 200) < 30
