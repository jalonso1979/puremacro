"""HP-filtered theoretical moments must weight the spectrum by the *squared* HP gain.

Regression for the 4.0.1 bug in ``dsge._moments.spectral_moments``: it weighted the
spectral density by the cyclical HP transfer function H(w) instead of |H(w)|^2, so every
``theoretical_moments(hp_filter=...)`` variance and autocovariance was overstated (22% for
the variance of an AR(1) with rho = 0.9 at lambda = 1600).

The oracle does not touch the frequency domain. The HP cycle of a long sample is
``(I - (I + lambda K'K)^{-1}) y``; the middle row of that matrix is the two-sided filter
``c``, so for a stationary input with autocovariance ``gamma`` the filtered autocovariance
at lag ``k`` is ``sum_ij c_i c_j gamma(i - j + k)``. A test built on the frequency-domain
formula would share any mistake in it, which is how the bug survived.
"""
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from puremacro.dsge import build
from puremacro.dsge._moments import spectral_moments

T_ORACLE = 6001


def _hp_cycle_weights(lam: float, T: int = T_ORACLE) -> np.ndarray:
    """Two-sided HP cycle filter: the middle row of I - (I + lam K'K)^{-1}."""
    ones = np.ones(T - 2)
    K = sp.diags([ones, -2.0 * ones, ones], [0, 1, 2], shape=(T - 2, T))
    A = (sp.eye(T) + lam * (K.T @ K)).tocsc()
    mid = T // 2
    unit = np.zeros(T)
    unit[mid] = 1.0
    c = -spl.spsolve(A, unit)  # A is symmetric, so this column is also the middle row
    c[mid] += 1.0
    return c


def _filtered_acov(c: np.ndarray, acov, lag: int) -> float:
    """sum_ij c_i c_j gamma(i - j + lag) for a symmetric autocovariance function gamma(h)."""
    n = len(c)
    h = np.arange(-(n - 1), n)
    cc = np.convolve(c, c[::-1])  # cc[m] = sum_i c_i c_{i-m}, indexed by h
    return float(np.dot(cc, acov(h + lag)))


@pytest.mark.parametrize("lam", [100.0, 1600.0, 14400.0])
@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_hp_filtered_ar1_moments_match_time_domain_filter(lam, rho):
    c = _hp_cycle_weights(lam)
    acov = lambda h: rho ** np.abs(h) / (1.0 - rho**2)  # noqa: E731  unit innovation variance

    one = np.eye(1)
    _, g0, gammas = spectral_moments(
        np.array([[rho]]), one, np.array([[rho]]), one, one, lags=2, filter_type="hp", hp_lambda=lam
    )
    for k, got in enumerate([g0[0, 0], gammas[0][0, 0], gammas[1][0, 0]]):
        want = _filtered_acov(c, acov, k)
        assert got == pytest.approx(want, rel=1e-7), f"lag {k}: {got:.10f} vs time-domain {want:.10f}"


def test_hp_filtered_white_noise_static_model():
    lam = 1600.0
    c = _hp_cycle_weights(lam)
    _, g0, _ = spectral_moments(
        np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)), np.eye(1), np.eye(1),
        lags=1, filter_type="hp", hp_lambda=lam,
    )
    assert g0[0, 0] == pytest.approx(float(np.sum(c**2)), rel=1e-9)


def test_theoretical_moments_hp_std_through_public_api():
    """The user-facing path: an AR(1) built as a model, std of its HP(1600) cycle."""
    rho, lam = 0.95, 1600.0

    def eqs(xp, x, e, p):
        return [xp.a - p.rho * x.a - e.eps]

    model = build(eqs, variables=["a"], states=["a"], shocks=["eps"],
                  params=dict(rho=rho), steady_state=dict(a=0.0))
    res = model.theoretical_moments(hp_filter=lam)

    c = _hp_cycle_weights(lam)
    sigma2 = float(model._shock_covariance(None)[0, 0])
    want_var = sigma2 * _filtered_acov(c, lambda h: rho ** np.abs(h) / (1.0 - rho**2), 0)
    got_var = float(res.moments.loc["a", "Variance"])
    assert got_var == pytest.approx(want_var, rel=1e-6)
