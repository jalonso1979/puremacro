"""Tests for puremacro.cointegration_modern result objects."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from puremacro.cointegration_modern import (
    fm_ols, dols, phillips_ouliaris,
    FMOLSResult, DOLSResult, PhillipsOuliarisResult,
)


def _simulate_cointegrated(T=200, beta=2.0, rho=0.5, seed=0):
    rng = np.random.default_rng(seed)
    nu = rng.standard_normal(T)
    xi = rng.standard_normal(T) * 0.7
    u = rho * nu + xi
    x = np.cumsum(nu)
    y = beta * x + u
    return y, x


# --------------------------------------------------------------------------
# fm_ols
# --------------------------------------------------------------------------
def test_fm_ols_returns_FMOLSResult():
    y, x = _simulate_cointegrated()
    res = fm_ols(y, x)
    assert isinstance(res, FMOLSResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.alpha = 0.0


def test_fm_ols_has_documented_fields():
    y, x = _simulate_cointegrated()
    res = fm_ols(y, x)
    assert isinstance(res.beta, np.ndarray) and res.beta.shape == (1,)
    assert isinstance(res.alpha, float)
    assert isinstance(res.se, np.ndarray) and res.se.shape == (1,)
    assert isinstance(res.residuals, np.ndarray)
    assert res.Omega.shape == (2, 2)
    assert res.Sigma.shape == (2, 2)
    assert res.Lambda.shape == (2, 2)


def test_fm_ols_recovers_beta_under_endogeneity():
    """FM-OLS recovers beta=2 within 0.05 in a sample with correlated regressor innovations."""
    y, x = _simulate_cointegrated(T=400, beta=2.0, rho=0.6, seed=7)
    b_fm = float(fm_ols(y, x).beta[0])
    assert abs(b_fm - 2.0) < 0.05


def test_fm_ols_summary_runs():
    y, x = _simulate_cointegrated()
    s = fm_ols(y, x).summary()
    assert isinstance(s, str)
    assert "FM-OLS" in s


# --------------------------------------------------------------------------
# dols
# --------------------------------------------------------------------------
def test_dols_returns_DOLSResult():
    y, x = _simulate_cointegrated()
    res = dols(y, x, leads=2, lags=2)
    assert isinstance(res, DOLSResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.alpha = 0.0


def test_dols_has_documented_fields():
    y, x = _simulate_cointegrated()
    res = dols(y, x, leads=2, lags=2)
    assert isinstance(res.alpha, float)
    assert isinstance(res.beta, np.ndarray) and res.beta.shape == (1,)
    assert isinstance(res.se, np.ndarray) and res.se.shape == (1,)
    assert isinstance(res.alpha_se, float)
    assert res.n_obs > 0


def test_dols_recovers_beta_under_endogeneity():
    y, x = _simulate_cointegrated(T=400, beta=2.0, rho=0.6, seed=11)
    b_dols = float(dols(y, x, leads=2, lags=2).beta[0])
    assert abs(b_dols - 2.0) < 0.05


def test_dols_summary_runs():
    y, x = _simulate_cointegrated()
    s = dols(y, x, leads=2, lags=2).summary()
    assert isinstance(s, str)
    assert "DOLS" in s


# --------------------------------------------------------------------------
# phillips_ouliaris
# --------------------------------------------------------------------------
def test_phillips_ouliaris_returns_PhillipsOuliarisResult():
    y, x = _simulate_cointegrated()
    res = phillips_ouliaris(y, x)
    assert isinstance(res, PhillipsOuliarisResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.z_t = 0.0


def test_phillips_ouliaris_rejects_unit_root_when_cointegrated():
    """On a cointegrated DGP Z_t should be more negative than -3.37 (5% CV, k=1)."""
    y, x = _simulate_cointegrated(T=400, beta=2.0, rho=0.5, seed=3)
    res = phillips_ouliaris(y, x)
    assert res.z_t < -3.37


def test_phillips_ouliaris_summary_runs():
    y, x = _simulate_cointegrated()
    s = phillips_ouliaris(y, x).summary()
    assert isinstance(s, str)
    assert "Phillips-Ouliaris" in s

# --------------------------------------------------------------------------
# _long_run_cov
# --------------------------------------------------------------------------
from puremacro.cointegration_modern import _long_run_cov

def test_long_run_cov_multivariate():
    """Verify _long_run_cov against manual calculation for 2D inputs."""
    u = np.array([[1.0, -1.0], [2.0, 0.0], [3.0, 1.0]])
    Omega, Sigma, Lambda = _long_run_cov(u, lags=1)

    expected_Sigma = np.array([[4.66666667, 0.66666667],
                               [0.66666667, 0.66666667]])
    expected_Lambda = np.array([[1.33333333, -0.33333333],
                                [0.33333333, 0.0]])
    expected_Omega = np.array([[7.33333333, 0.66666667],
                               [0.66666667, 0.66666667]])

    np.testing.assert_allclose(Sigma, expected_Sigma, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(Lambda, expected_Lambda, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(Omega, expected_Omega, rtol=1e-5, atol=1e-8)

def test_long_run_cov_1d():
    """Verify _long_run_cov handles 1D arrays correctly."""
    u = np.array([1.0, 2.0, 3.0])
    Omega, Sigma, Lambda = _long_run_cov(u, lags=1)

    expected_Sigma = np.array([[4.66666667]])
    expected_Lambda = np.array([[1.33333333]])
    expected_Omega = np.array([[7.33333333]])

    np.testing.assert_allclose(Sigma, expected_Sigma, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(Lambda, expected_Lambda, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(Omega, expected_Omega, rtol=1e-5, atol=1e-8)

def test_long_run_cov_default_lags():
    """Verify _long_run_cov calculates correctly when lags is None."""
    u = np.array([[1.0, -1.0], [2.0, 0.0], [3.0, 1.0]])
    Omega, Sigma, Lambda = _long_run_cov(u, lags=None)
    # Just checking it returns arrays of correct shapes
    assert Omega.shape == (2, 2)
    assert Sigma.shape == (2, 2)
    assert Lambda.shape == (2, 2)


# ---------------------------------------------------------------------------
# Regression tests for the 1.9.1 DOLS / FM-OLS corrections.
#
# The pre-existing accuracy tests assert `abs(beta - 2.0) < 0.05` on a fixture
# whose endogeneity is purely contemporaneous and whose innovations are i.i.d.
# Plain OLS passes both of those thresholds (its residual bias on that design is
# ~0.009 at T=400), so neither test could distinguish a working estimator from
# one that does nothing. These tests compare against OLS on a design where the
# correction is supposed to bite, which is the only comparison that can fail.
# ---------------------------------------------------------------------------
def _contemporaneous_endog(T, rho=0.6, seed=0):
    """u_t correlated with dx_t: the channel the DOLS s=0 term handles."""
    rng = np.random.default_rng(seed)
    nu = rng.standard_normal(T)
    xi = rng.standard_normal(T)
    x = np.cumsum(nu)
    return 2.0 + 2.0 * x + rho * nu + xi, x[:, None]


def _serially_correlated_errors(T, seed=0):
    """u_t = 0.8 v_{t-1} + w_t: the channel FM-OLS's one-sided term handles."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(T + 1)
    w = rng.standard_normal(T)
    x = np.cumsum(v[1:])
    return 2.0 * x + 0.8 * v[:-1] + w, x[:, None]


def _ols_slope(y, X):
    Z = np.column_stack([np.ones(len(y)), X])
    return float(np.linalg.lstsq(Z, y, rcond=None)[0][1])


def test_dols_is_actually_bias_corrected_and_not_just_ols():
    """DOLS must beat OLS on contemporaneous endogeneity, not merely match it.

    `dols` used to omit the s=0 term of sum_j gamma_j dX_{t-j}, which is the
    one that removes the second-order bias. Without it the estimator IS OLS
    with extra regressors: measured mean bias over 200 replications at T=200
    was +0.0176 against OLS's +0.0160, versus +0.0010 once the term is
    restored. The docstring promises "long-run-bias-free", so this is the
    property under test.
    """
    bias_ols, bias_dols = [], []
    for s in range(200):
        y, X = _contemporaneous_endog(200, seed=5000 + s)
        bias_ols.append(_ols_slope(y, X) - 2.0)
        bias_dols.append(float(dols(y, X, leads=2, lags=2).beta[0]) - 2.0)
    mean_ols = abs(np.mean(bias_ols))
    mean_dols = abs(np.mean(bias_dols))
    assert mean_dols < 0.4 * mean_ols, (
        f"DOLS bias {mean_dols:.5f} is not meaningfully below OLS bias "
        f"{mean_ols:.5f} — the correction is not doing anything."
    )


def test_dols_uses_no_zero_padded_rows():
    """Every retained row must have all its leads and lags observed.

    The window used to start at t=lags, where the s=-lags term needs dX[-1];
    that row was kept with the missing block zero-filled, which fabricates a
    regressor value. `CONTRIBUTING.md` forbids substituting a plausible value
    for a missing one.
    """
    T, leads, lags = 60, 2, 2
    y, X = _contemporaneous_endog(T, seed=1)
    res = dols(y, X, leads=leads, lags=lags)
    assert res.n_obs == T - leads - lags - 1, (
        f"n_obs={res.n_obs} implies a row whose dX lags are not all observed; "
        f"expected {T - leads - lags - 1}"
    )


def test_fm_ols_beats_ols_under_serially_correlated_errors():
    """FM-OLS must improve on OLS, not move further from the truth.

    The one-sided long-run covariance is Delta = Sigma + Lambda. Using Lambda
    alone (j>=1) over-corrected: measured RMSE at T=200 over 200 replications
    was 0.0374 against OLS's 0.0320 — an FM-OLS worse than doing nothing —
    versus 0.0190 once the lag-0 block is included. The error is identically
    zero when the innovations are serially uncorrelated, which is exactly what
    the i.i.d. fixture above imposes, so only a serially correlated design can
    catch it.
    """
    err_ols, err_fm = [], []
    for s in range(200):
        y, X = _serially_correlated_errors(200, seed=s)
        err_ols.append(_ols_slope(y, X) - 2.0)
        err_fm.append(float(fm_ols(y, X).beta[0]) - 2.0)
    rmse_ols = float(np.sqrt(np.mean(np.square(err_ols))))
    rmse_fm = float(np.sqrt(np.mean(np.square(err_fm))))
    assert rmse_fm < rmse_ols, (
        f"FM-OLS RMSE {rmse_fm:.5f} is not below OLS RMSE {rmse_ols:.5f} — "
        f"the Phillips-Hansen correction is making the estimate worse."
    )
