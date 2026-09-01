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
