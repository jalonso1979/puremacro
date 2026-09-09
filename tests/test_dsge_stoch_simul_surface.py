"""Unit tests for stoch_simul surface area, spectral filtering parity, and empirical simulation moments."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import build, one_sided_hp_filter, TheoreticalMomentsResult, StochSimulResult
from puremacro.dsge._moments import spectral_moments


@pytest.fixture
def rbc_model():
    """Canonical calibrated RBC model solved via puremacro.dsge.build."""
    def eqs(xp, x, e, p):
        return [
            x.c**-p.sigma - p.beta * xp.c**-p.sigma * (p.alpha * xp.z * xp.k**(p.alpha - 1) + 1 - p.delta),
            x.c + xp.k - x.z * x.k**p.alpha - (1 - p.delta) * x.k,
            xp.z - (1.0 - p.rho) - p.rho * x.z - e.eps,
        ]

    beta, delta, alpha = 0.99, 0.025, 0.33
    r_ss = 1.0 / beta - 1.0
    k_ss = (alpha / (r_ss + delta)) ** (1.0 / (1.0 - alpha))
    y_ss = k_ss**alpha
    c_ss = y_ss - delta * k_ss

    return build(
        eqs,
        variables=["c", "k", "z"],
        states=["k", "z"],
        shocks=["eps"],
        params=dict(alpha=alpha, beta=beta, delta=delta, sigma=1.0, rho=0.95),
        steady_state=dict(c=c_ss, k=k_ss, z=1.0),
    )


@pytest.fixture
def ar1_model():
    """Simple calibrated AR(1) state-space model."""
    def eqs(xp, x, e, p):
        return [xp.a - p.rho * x.a - e.eps]

    return build(
        eqs,
        variables=["a"],
        states=["a"],
        shocks=["eps"],
        params=dict(rho=0.8),
        steady_state=dict(a=0.0),
    )


def test_hp_filter_theoretical_variance_quadrature(ar1_model):
    """Verify hp_filter theoretical variance matches high-precision Gauss-Legendre quadrature to < 1e-8."""
    res = ar1_model.theoretical_moments(hp_filter=1600.0)
    var_hp = float(res.moments.loc["a", "Variance"])

    # High-precision 512-point Gauss-Legendre quadrature reference
    G, N = ar1_model.solution.G, ar1_model.solution.N
    M_x, M_u = ar1_model._reported_loadings()
    sigma_u = ar1_model._shock_covariance(None)
    _, g0_ref, _ = spectral_moments(
        G, N, M_x, M_u, sigma_u, lags=1, filter_type="hp", hp_lambda=1600.0, n_quad=512
    )
    var_ref = float(g0_ref[0, 0])

    rel_err = abs(var_hp - var_ref) / var_ref
    assert rel_err < 1e-8, f"HP theoretical variance relative error {rel_err:.3e} exceeds 1e-8"
    assert var_hp > 0.0
    # High-pass filter removes mean: theoretical filtered mean is zero
    assert res.moments.loc["a", "Mean"] == 0.0


def test_simul_replic_mc_confidence_bounds():
    """Verify simul_replic=1000 empirical moments match theoretical ergodic moments within 3-sigma bounds."""
    def eqs(xp, x, e, p):
        return [xp.a - p.rho * x.a - e.eps]

    m = build(
        eqs,
        variables=["a"],
        states=["a"],
        shocks=["eps"],
        params=dict(rho=0.5),
        steady_state=dict(a=0.0),
    )

    res = m.stoch_simul(periods=1000, simul_replic=1000, seed=42)

    assert isinstance(res, StochSimulResult)
    assert res.simulated_moments is not None
    expected_cols = ["Mean", "Std.Dev.", "Variance", "Skewness", "Kurtosis", "MC Std.Err."]
    assert list(res.simulated_moments.columns) == expected_cols
    assert res.simulated_moments.attrs["simul_replic"] == 1000

    theo_mean = float(res.theoretical_moments.moments.loc["a", "Mean"])
    sim_mean = float(res.simulated_moments.loc["a", "Mean"])
    se_mean = float(res.simulated_moments.loc["a", "MC Std.Err."])
    diff_mean = abs(sim_mean - theo_mean)
    assert diff_mean <= 3.0 * se_mean, f"Mean diff {diff_mean} exceeds 3 * SE ({3.0 * se_mean})"

    theo_var = float(res.theoretical_moments.moments.loc["a", "Variance"])
    sim_var = float(res.simulated_moments.loc["a", "Variance"])
    se_var = float(res.simulated_moments.attrs["mc_se_var"]["a"])
    diff_var = abs(sim_var - theo_var)
    assert diff_var <= 3.0 * se_var, f"Variance diff {diff_var} exceeds 3 * SE ({3.0 * se_var})"


def test_one_sided_hp_filter_guard(rbc_model):
    """Verify exact Dynare error guard on one-sided HP filter with theoretical moments and stoch_simul."""
    exact_err = "disp_th_moments:: theoretical moments incompatible with one-sided HP filter. Use simulated moments instead."

    with pytest.raises(ValueError, match=exact_err):
        rbc_model.theoretical_moments(one_sided_hp_filter=True)

    with pytest.raises(ValueError, match=exact_err):
        rbc_model.stoch_simul(periods=0, one_sided_hp_filter=True)

    # Valid execution on simulation (periods > 0)
    res = rbc_model.stoch_simul(periods=200, one_sided_hp_filter=True, seed=42)
    assert res.simulated_moments is not None
    assert res.theoretical_moments is None
    assert res.contemporaneous_correlation is not None
    assert res.contemporaneous_correlation.shape == (3, 3)


def test_autocorrelation_table_and_matrices(rbc_model):
    """Verify ar=4 creates full cross-variable autocorrelation matrices and formatted table."""
    res = rbc_model.theoretical_moments(ar=4)
    assert isinstance(res, TheoreticalMomentsResult)

    assert list(res.autocorr.columns) == ["Lag 1", "Lag 2", "Lag 3", "Lag 4"]
    assert len(res.autocorr_matrices) == 4
    assert len(res.autocorrelation_matrices) == 4

    vars_list = list(rbc_model.variables)
    for k in range(1, 5):
        mat = res.autocorr_matrices[k - 1]
        assert isinstance(mat, pd.DataFrame)
        assert mat.shape == (len(vars_list), len(vars_list))
        assert list(mat.index) == vars_list
        assert list(mat.columns) == vars_list
        # Diagonal matches 1D autocorrelation table
        np.testing.assert_allclose(np.diag(mat.to_numpy()), res.autocorr[f"Lag {k}"].to_numpy())
        np.testing.assert_allclose(mat.to_numpy(), res.autocorr_matrix(k).to_numpy())

    # ar=0 returns empty structures cleanly
    res0 = rbc_model.theoretical_moments(ar=0)
    assert len(res0.autocorr_matrices) == 0
    assert res0.autocorr.empty


def test_contemporaneous_correlation_matrix(rbc_model):
    """Verify contemporaneous correlation matrix is symmetric with unitary diagonal and bounded in [-1, 1]."""
    res = rbc_model.theoretical_moments(contemporaneous_correlation=True)
    corr = res.correlation
    assert corr is not None
    assert corr.shape == (3, 3)
    np.testing.assert_allclose(np.diag(corr.to_numpy()), 1.0, atol=1e-12)
    np.testing.assert_allclose(corr.to_numpy(), corr.to_numpy().T, atol=1e-12)
    assert np.all(corr.to_numpy() >= -1.0)
    assert np.all(corr.to_numpy() <= 1.0)
    assert "MATRIX OF CORRELATIONS" in res.summary()

    # Optional deactivation
    res_no = rbc_model.theoretical_moments(contemporaneous_correlation=False)
    assert res_no.correlation is None
    assert "MATRIX OF CORRELATIONS" not in res_no.summary()


def test_simul_replic_with_filters(rbc_model):
    """Verify simul_replic works seamlessly with trajectory filters (HP and Baxter-King)."""
    res_hp = rbc_model.stoch_simul(periods=200, simul_replic=30, hp_filter=1600.0, seed=10)
    assert res_hp.simulated_moments is not None
    assert not res_hp.simulated_moments.isna().any().any()
    assert (res_hp.simulated_moments["MC Std.Err."] > 0).all()

    res_bp = rbc_model.stoch_simul(periods=200, simul_replic=30, bandpass_filter=(6, 32), seed=10)
    assert res_bp.simulated_moments is not None
    assert not res_bp.simulated_moments.isna().any().any()
    assert (res_bp.simulated_moments["MC Std.Err."] > 0).all()

    res_1s = rbc_model.stoch_simul(periods=200, simul_replic=30, one_sided_hp_filter=True, seed=10)
    assert res_1s.simulated_moments is not None
    assert not res_1s.simulated_moments.isna().any().any()
    assert (res_1s.simulated_moments["MC Std.Err."] > 0).all()


def test_filter_parameter_guards(rbc_model):
    """Verify parameter guards for simul_replic, filters, and lags."""
    with pytest.raises(ValueError, match="simul_replic must be non-negative"):
        rbc_model.stoch_simul(periods=100, simul_replic=-5)

    with pytest.raises(ValueError, match="simul_replic > 0 requires periods > 0"):
        rbc_model.stoch_simul(periods=0, simul_replic=100)

    with pytest.raises(ValueError, match="Only one filter can be specified"):
        rbc_model.stoch_simul(periods=100, hp_filter=1600.0, bandpass_filter=(6, 32))

    with pytest.raises(ValueError, match="hp_filter parameter lambda must be positive"):
        rbc_model.theoretical_moments(hp_filter=-10.0)

    with pytest.raises(ValueError, match="bandpass_filter requires 0 < low < high"):
        rbc_model.theoretical_moments(bandpass_filter=(32, 6))

    with pytest.raises(ValueError, match="one_sided_hp_filter requires at least 4 observations"):
        one_sided_hp_filter(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="one_sided_hp_filter parameter lambda must be positive"):
        one_sided_hp_filter(np.array([1.0, 2.0, 3.0, 4.0]), lamb=-10.0)

    with pytest.raises(ValueError, match="ar must be a non-negative integer"):
        rbc_model.theoretical_moments(ar=-2)
