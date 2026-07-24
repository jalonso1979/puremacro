import numpy as np
import pytest

from puremacro.tests.unit_root import (
    adf_test, kpss_test, pp_test, zivot_andrews_test,
    dfgls_test, ng_perron_test,
)


def _ar1(phi, T, seed, const=0.0, slope=0.0):
    """AR(1) u_t = phi u_{t-1} + e_t plus an optional deterministic const+trend."""
    rng = np.random.default_rng(seed)
    u = np.zeros(T)
    for t in range(1, T):
        u[t] = phi * u[t - 1] + rng.standard_normal()
    return const + slope * np.arange(T) + u


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


# --------------------------------------------------------------------------- #
# DF-GLS (Elliott-Rothenberg-Stock 1996)                                      #
# --------------------------------------------------------------------------- #

def test_dfgls_fails_to_reject_random_walk():
    """A strongly persistent random walk must retain the unit root."""
    rng = np.random.default_rng(101)
    y = np.cumsum(rng.standard_normal(500))
    out = dfgls_test(y, regression="c")
    assert out["p_value"] > 0.05
    assert out["decision"] == "accept"


def test_dfgls_rejects_stationary_ar1():
    """A stationary AR(1) with small phi is rejected at 5%."""
    y = _ar1(0.2, 500, seed=103)
    out = dfgls_test(y, regression="c")
    assert out["p_value"] < 0.05
    assert out["decision"] == "reject"


def test_dfgls_rejects_white_noise_strongly():
    """White noise is the extreme stationary case — reject hard (p at floor)."""
    rng = np.random.default_rng(107)
    out = dfgls_test(rng.standard_normal(400), regression="c")
    assert out["p_value"] <= 0.01
    assert out["stat"] < out["crit_values"]["1%"]


def test_dfgls_higher_power_than_adf_near_unit_root():
    """The canonical selling point: on a near-unit-root series with a trend,
    DF-GLS rejects the unit root at 5% where ADF cannot."""
    y = _ar1(0.85, 200, seed=10, const=5.0, slope=0.3)
    dg = dfgls_test(y, regression="ct")
    ad = adf_test(y, regression="ct")
    assert dg["p_value"] < 0.05 <= ad["p_value"], (dg["p_value"], ad["p_value"])
    assert dg["decision"] == "reject" and ad["decision"] == "accept"


def test_dfgls_gls_detrend_recovers_trend_exactly():
    """GLS detrending of a purely deterministic trend recovers its coefficients
    exactly (planted-truth identity: y = a + b t -> gls_coef = [a, b])."""
    T = 200
    y = 2.0 + 0.5 * np.arange(1, T + 1)
    out = dfgls_test(y, regression="ct")
    assert out["gls_coef"] == pytest.approx([2.0, 0.5], abs=1e-8)


# --------------------------------------------------------------------------- #
# Ng-Perron (2001) M-tests                                                    #
# --------------------------------------------------------------------------- #

def test_ng_perron_fails_to_reject_random_walk():
    rng = np.random.default_rng(211)
    y = np.cumsum(rng.standard_normal(500))
    out = ng_perron_test(y, regression="c")
    assert out["decision"] == "accept"
    # All four M-tests agree the unit root is retained.
    assert all(d == "accept" for d in out["decisions"].values())


def test_ng_perron_rejects_stationary_ar1():
    y = _ar1(0.2, 500, seed=213)
    out = ng_perron_test(y, regression="c")
    assert out["decision"] == "reject"
    assert out["statistics"]["MZt"] < out["crit_values_all"]["MZt"]["5%"]


def test_ng_perron_all_four_statistics_present():
    rng = np.random.default_rng(217)
    out = ng_perron_test(rng.standard_normal(300), regression="ct")
    assert set(out["statistics"]) == {"MZa", "MZt", "MSB", "MPT"}
    assert set(out["crit_values_all"]) == {"MZa", "MZt", "MSB", "MPT"}
    # MZt == MZa * MSB by construction.
    s = out["statistics"]
    assert s["MZt"] == pytest.approx(s["MZa"] * s["MSB"], rel=1e-10)


# --------------------------------------------------------------------------- #
# Reproducibility + input validation                                          #
# --------------------------------------------------------------------------- #

def test_dfgls_reproducible():
    y = _ar1(0.5, 300, seed=301)
    a = dfgls_test(y, regression="c")
    b = dfgls_test(y, regression="c")
    assert a["stat"] == b["stat"] and a["p_value"] == b["p_value"]


@pytest.mark.parametrize("fn", [dfgls_test, ng_perron_test])
def test_invalid_regression_raises(fn):
    y = _ar1(0.5, 100, seed=307)
    with pytest.raises(ValueError):
        fn(y, regression="n")  # DF-GLS/Ng-Perron only defined for 'c'/'ct'


@pytest.mark.parametrize("fn", [dfgls_test, ng_perron_test])
def test_manual_lag_honoured(fn):
    y = _ar1(0.5, 300, seed=311)
    assert fn(y, lags=3, regression="c")["lag_used"] == 3
