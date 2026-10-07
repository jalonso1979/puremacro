"""FIXB: Bartlett fixed-b critical values follow Kiefer & Vogelsang (2005) Table I.

Published values: Kiefer, N.M. & Vogelsang, T.J. (2005), "A New Asymptotic
Theory for Heteroskedasticity-Autocorrelation Robust Tests", Econometric
Theory 21, 1130-1164; read as Cornell CAE Working Paper 05-08 (updated
January 2005), https://cae.economics.cornell.edu/05-08.pdf, Table I
("Asymptotic Critical Values for t*_b Using Bartlett Kernel"), printed p. 24
(PDF p. 26).  The table gives right-tail quantiles of a symmetric law, so a
two-sided test at level alpha uses the 1 - alpha/2 row.  The b = 1 entries are
analytical (table notes).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from puremacro.inference.hac_fixed_b import hac_fixed_b, llsw_critical_value


# --------------------------------------------------------------------------- #
# Table lookups against the published constants                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "b, alpha, expected",
    [
        (1.00, 0.05, 4.771),  # two-sided 5% -> 97.5% row (analytical at b = 1)
        (1.00, 0.10, 3.764),  # two-sided 10% -> 95% row
        (1.00, 0.20, 2.740),  # two-sided 20% -> 90% row
        (1.00, 0.02, 6.090),  # two-sided 2% -> 99% row
        (0.50, 0.10, 2.781),
        (0.50, 0.05, 3.514),
        (0.20, 0.05, 2.553),
        (0.10, 0.10, 1.861),
        (0.10, 0.05, 2.235),
        (0.02, 0.05, 2.018),
        (0.30, 0.02, 3.580),
        (0.44, 0.02, 4.212),
        (0.66, 0.20, 2.272),
        (0.88, 0.05, 4.524),
        (0.98, 0.10, 3.764),
    ],
)
def test_two_sided_matches_kv2005_table1(b, alpha, expected):
    assert llsw_critical_value(b, alpha) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "b, alpha, expected",
    [
        (1.00, 0.05, 3.764),   # one-sided 5% -> 95% row
        (1.00, 0.025, 4.771),  # one-sided 2.5% -> 97.5% row
        (1.00, 0.10, 2.740),
        (1.00, 0.01, 6.090),
        (0.30, 0.05, 2.324),
        (0.06, 0.10, 1.368),
    ],
)
def test_one_sided_uses_one_minus_alpha_row(b, alpha, expected):
    assert llsw_critical_value(b, alpha, two_sided=False) == pytest.approx(expected, abs=1e-12)


def test_b1_two_sided_5pct_is_4771():
    """The headline defect: the library returned 3.96 here."""
    assert llsw_critical_value(1.0, alpha=0.05) == pytest.approx(4.771, abs=1e-12)


def test_linear_interpolation_between_grid_points():
    # KV 2005 section 3.3: "Critical values can be interpolated for values of
    # b that fall between the values on the grid."  95% row: 1.944 (0.14), 1.988 (0.16).
    assert llsw_critical_value(0.15, 0.10) == pytest.approx(0.5 * (1.944 + 1.988), abs=1e-12)


def test_below_grid_interpolates_towards_normal_quantile():
    # b -> 0 recovers N(0,1); between 0 and 0.02 interpolate to the b = 0.02 entry.
    z = 1.959963984540054
    assert llsw_critical_value(0.01, 0.05) == pytest.approx(0.5 * (z + 2.018), abs=1e-9)
    assert z < llsw_critical_value(0.001, 0.05) < 2.018


@pytest.mark.parametrize("alpha", [0.07, 0.5, 0.0, -0.05])
def test_unsupported_alpha_raises(alpha):
    with pytest.raises(ValueError, match="alpha"):
        llsw_critical_value(0.1, alpha)


@pytest.mark.parametrize("b", [0.0, -0.1, 1.2, float("nan")])
def test_b_outside_unit_interval_raises(b):
    with pytest.raises(ValueError, match="b"):
        llsw_critical_value(b, 0.05)


def test_monotone_in_b_and_in_level():
    bs = np.arange(1, 51) / 50.0
    for alpha in (0.20, 0.10, 0.05, 0.02, 0.01):
        cv = np.array([llsw_critical_value(b, alpha) for b in bs])
        # KV's simulated rows carry Monte Carlo noise (b = 0.98 and 1.0 share
        # 3.764 at 95%), so require non-decreasing, not strictly increasing.
        assert np.all(np.diff(cv) >= 0.0), alpha
    for b in bs:
        levels = [llsw_critical_value(b, a) for a in (0.20, 0.10, 0.05, 0.02, 0.01)]
        assert np.all(np.diff(levels) > 0.0), b


# --------------------------------------------------------------------------- #
# The two-sided 1% row (99.5% quantile) is not in Table I; it is computed.     #
# Recompute it here independently (Imhof inversion of the discretised limit)   #
# and check the method against KV's analytical b = 1 values.                   #
# --------------------------------------------------------------------------- #
def _bartlett_bridge_eigs(b: float, T: int) -> np.ndarray:
    M = b * T
    idx = np.arange(T)
    K = np.clip(1.0 - np.abs(idx[:, None] - idx[None, :]) / M, 0.0, None)
    K = K - K.mean(axis=0, keepdims=True)
    K = K - K.mean(axis=1, keepdims=True)
    lam = np.linalg.eigvalsh(K / T)
    return lam[lam > 1e-14 * lam.max()]


def _two_sided_tail(c: float, lam: np.ndarray) -> float:
    """P(|W(1)| > c sqrt(Q)) = P(Z^2 - c^2 sum lam_j chi2_j > 0), Imhof (1961)."""
    from scipy import integrate

    w = np.concatenate([[1.0], -c * c * lam])

    def f(u):
        if u == 0.0:
            return 0.5 * float(np.sum(w))
        theta = 0.5 * np.sum(np.arctan(w * u))
        logrho = 0.25 * np.sum(np.log1p((w * u) ** 2))
        return math.sin(theta) * math.exp(-logrho) / u  # exp(-x) underflows safely

    val, _ = integrate.quad(f, 0.0, np.inf, limit=2000, epsabs=1e-12, epsrel=1e-10)
    return 0.5 + val / math.pi


def _quantile(level: float, lam: np.ndarray) -> float:
    from scipy import optimize

    return optimize.brentq(lambda c: _two_sided_tail(c, lam) - 2.0 * (1.0 - level), 1.0, 20.0,
                           xtol=1e-9)


def test_exact_method_reproduces_kv_analytical_b1_values():
    lam = _bartlett_bridge_eigs(1.0, 500)
    for level, published in ((0.90, 2.740), (0.95, 3.764), (0.975, 4.771), (0.99, 6.090)):
        assert _quantile(level, lam) == pytest.approx(published, abs=1e-3)


@pytest.mark.parametrize("b", [0.1, 0.5, 1.0])
def test_two_sided_1pct_row_matches_independent_computation(b):
    lam = _bartlett_bridge_eigs(b, 500)
    assert llsw_critical_value(b, 0.01) == pytest.approx(_quantile(0.995, lam), abs=2e-3)
    assert llsw_critical_value(b, 0.005, two_sided=False) == llsw_critical_value(b, 0.01)


# --------------------------------------------------------------------------- #
# hac_fixed_b wiring                                                          #
# --------------------------------------------------------------------------- #
def _location_data(T, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(T), np.ones((T, 1))


def test_hac_fixed_b_reports_effective_b_and_its_critical_values():
    y, X = _location_data(200)
    out = hac_fixed_b(y, X, b=0.10)
    assert out["lags"] == 20
    # Bartlett weights 1 - l/(L+1): bandwidth M = L + 1 in KV's k(j/M) notation.
    assert out["bandwidth"] == 21
    assert out["b_eff"] == pytest.approx(21 / 200)
    assert out["fixed_b_cv_90"] == pytest.approx(llsw_critical_value(21 / 200, 0.10))
    assert out["fixed_b_cv_95"] == pytest.approx(llsw_critical_value(21 / 200, 0.05))
    assert out["fixed_b_cv_99"] == pytest.approx(llsw_critical_value(21 / 200, 0.01))
    assert out["llsw_cv_90"] == out["fixed_b_cv_90"]  # legacy key kept


def test_hac_fixed_b_b1_uses_b1_row():
    y, X = _location_data(200)
    out = hac_fixed_b(y, X, b=1.0)
    assert out["b_eff"] == 1.0
    assert out["fixed_b_cv_95"] == pytest.approx(4.771, abs=1e-12)


def test_hac_fixed_b_lag_count_is_robust_to_float_rounding():
    y, X = _location_data(100)
    assert hac_fixed_b(y, X, b=0.29)["lags"] == 29  # 0.29 * 100 = 28.999999999999996


@pytest.mark.parametrize("b", [0.0, -0.2, 1.5, float("nan")])
def test_hac_fixed_b_rejects_b_outside_unit_interval(b):
    y, X = _location_data(50)
    with pytest.raises(ValueError, match="b"):
        hac_fixed_b(y, X, b=b)


# --------------------------------------------------------------------------- #
# Monte Carlo size under H0                                                   #
# --------------------------------------------------------------------------- #
def test_null_rejection_rate_near_nominal_end_to_end():
    """iid N(0,1) location model through the public API; nominal 5%.

    Before the fix (3.96 at b=1, 2.83 at b=0.5) these rejected about 9%.
    With 3000 replications the Monte Carlo s.e. is 0.004.
    """
    rng = np.random.default_rng(20260930)
    T, R = 100, 3000
    X = np.ones((T, 1))
    for b in (0.5, 1.0):
        rej = 0
        for _ in range(R):
            out = hac_fixed_b(rng.standard_normal(T), X, b=b)
            rej += abs(out["t"][0]) > out["fixed_b_cv_95"]
        assert abs(rej / R - 0.05) < 0.015, (b, rej / R)


def _vectorised_location_t(E: np.ndarray, b: float) -> np.ndarray:
    """t-statistics of hac_fixed_b(y, ones) for each row of E (same weights)."""
    R, T = E.shape
    L = max(1, int(math.floor(b * T + 1e-9)))
    lag = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :])
    K = np.where(lag <= L, 1.0 - lag / (L + 1.0), 0.0)
    ybar = E.mean(axis=1)
    U = E - ybar[:, None]
    S = np.einsum("ij,ij->i", U @ K, U)  # = T * Omega_hat
    return ybar / np.sqrt(S / T**2)


def test_vectorised_replica_matches_hac_fixed_b():
    rng = np.random.default_rng(1)
    E = rng.standard_normal((5, 120))
    for b in (0.1, 0.5, 1.0):
        t_vec = _vectorised_location_t(E, b)
        t_lib = [hac_fixed_b(e, np.ones((120, 1)), b=b)["t"][0] for e in E]
        np.testing.assert_allclose(t_vec, t_lib, rtol=1e-10)


@pytest.mark.parametrize("b", [0.2, 0.5, 1.0])
@pytest.mark.parametrize("alpha, tol", [(0.10, 0.012), (0.05, 0.008), (0.01, 0.004)])
def test_null_rejection_rate_near_nominal_all_levels(b, alpha, tol):
    """40,000 replications (s.e. 0.0015 at 5%, 0.0005 at 1%) via the verified replica."""
    rng = np.random.default_rng(int(1000 * b) + int(1000 * alpha))
    T, R = 150, 40_000
    L = max(1, int(math.floor(b * T + 1e-9)))
    cv = llsw_critical_value(min(1.0, (L + 1) / T), alpha)
    t = _vectorised_location_t(rng.standard_normal((R, T)), b)
    rate = float(np.mean(np.abs(t) > cv))
    assert abs(rate - alpha) < tol, (b, alpha, cv, rate)
