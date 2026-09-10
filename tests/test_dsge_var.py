"""Unit test suite for Del Negro & Schorfheide (2004) DSGE-VAR Hybrid Modeling (Milestone 2).

Validates:
1. OLS convergence: as lambda -> 0 (or lambda -> lambda_min relaxed), tilde{Phi} -> Phi_ols
   and tilde{Sigma} -> Sigma_ols with tolerance < 1e-6.
2. DSGE convergence: as lambda -> infty (1e5), tilde{Phi} -> Phi* and tilde{Sigma} -> Sigma*,
   and structural DSGE-VAR IRFs converge continuously to pure DSGE IRFs.
3. Log Marginal Data Density (MDD): concavity over grid lambda in [0.2, 5.0] with a
   well-defined interior optimum hat{lambda}.
4. Prior admissibility bound: lambda < lambda_min raises ValueError when check_bounds=True,
   and lambda_min equals (k + n) / T.
5. Structural shock identification & FEVD:
   - Orthonormality of rotation matrix Q* (Q* Q*' = I_n).
   - Exact covariance decomposition B0 B0' = tilde{Sigma}.
   - FEVD shares sum to 1.0 at every horizon.
6. Forecast generation:
   - Returns DSGEForecastResult with upper > mean > lower.
   - Forecast error bands widen monotonically with horizon.
7. Full 6-method presentation contract:
   - .summary(), .plot(kind='irf'), .plot(kind='mdd'), .plot(kind='forecast'),
     .to_frame(), .to_markdown(), .to_latex(), .to_typst().
8. LinearModel.dsge_var() convenience method parity with estimate_dsge_var().
9. Robust error handling for non-stationary models, unknown observables, and shock dimension mismatch.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.build import LinearModel
from puremacro.dsge.dsge_var import DSGEVARResult, estimate_dsge_var
from puremacro.dsge._results import DSGEForecastResult


# ==============================================================================
# Model Fixtures
# ==============================================================================

@pytest.fixture
def nk_model_3shocks() -> LinearModel:
    """Canonical 3-equation New Keynesian model with 3 shocks (technology, cost-push, monetary)."""
    mod = """
    var y pi r a u;
    varexo eps_a eps_u eps_r;
    parameters beta sigma kappa phi_pi phi_y rho_a rho_u;
    beta = 0.99;
    sigma = 1.0;
    kappa = 0.5;
    phi_pi = 1.5;
    phi_y = 0.5;
    rho_a = 0.75;
    rho_u = 0.65;

    model;
    y = y(+1) - (1/sigma)*(r - pi(+1)) + (a(+1) - a);
    pi = beta*pi(+1) + kappa*y + u;
    r = phi_pi*pi + phi_y*y + eps_r;
    a = rho_a*a(-1) + eps_a;
    u = rho_u*u(-1) + eps_u;
    end;

    shocks;
    var eps_a; stderr 0.01;
    var eps_u; stderr 0.01;
    var eps_r; stderr 0.005;
    end;
    """
    return build_dynare(mod)


@pytest.fixture
def simulated_data(nk_model_3shocks: LinearModel) -> pd.DataFrame:
    """Simulate 300 observations of observables ['y', 'pi', 'r']."""
    sim = nk_model_3shocks.simulate(300, seed=42)
    return sim[["y", "pi", "r"]]


# ==============================================================================
# 1. Asymptotic Limit: OLS Convergence as lambda -> 0
# ==============================================================================

def test_ols_convergence_as_lambda_approaches_zero(
    nk_model_3shocks: LinearModel, simulated_data: pd.DataFrame
):
    """Verify that as prior weight lambda -> 0, DSGE-VAR reproduces OLS estimates."""
    p = 2
    res_ols_approx = estimate_dsge_var(
        nk_model_3shocks,
        simulated_data,
        p=p,
        lamb=1e-8,
        check_bounds=False,
    )

    # Directly compare against sample OLS moments stored on the result
    np.testing.assert_allclose(
        res_ols_approx.to_frame().to_numpy(),
        res_ols_approx.Phi_ols,
        rtol=1e-5,
        atol=1e-6,
        err_msg="tilde{Phi} did not converge to Phi_ols as lambda -> 0",
    )
    np.testing.assert_allclose(
        res_ols_approx.Sigma,
        res_ols_approx.Sigma_ols,
        rtol=1e-5,
        atol=1e-6,
        err_msg="tilde{Sigma} did not converge to Sigma_ols as lambda -> 0",
    )


# ==============================================================================
# 2. Asymptotic Limit: DSGE Convergence as lambda -> infty
# ==============================================================================

def test_dsge_convergence_as_lambda_approaches_infinity(
    nk_model_3shocks: LinearModel, simulated_data: pd.DataFrame
):
    """Verify that as lambda -> infty, posterior moments converge to prior moments Phi* and Sigma*,
    and DSGE-VAR structural IRFs converge to the pure DSGE model IRFs."""
    p = 2
    res_inf = estimate_dsge_var(
        nk_model_3shocks,
        simulated_data,
        p=p,
        lamb=1e8,
        identification="dsge",
    )

    # 1. Parameter convergence: tilde{Phi} -> Phi*, tilde{Sigma} -> Sigma*
    np.testing.assert_allclose(
        res_inf.to_frame().to_numpy(),
        res_inf.Phi_star,
        rtol=1e-5,
        atol=1e-6,
        err_msg="tilde{Phi} did not converge to theoretical Phi* as lambda -> infty",
    )
    np.testing.assert_allclose(
        res_inf.Sigma,
        res_inf.Sigma_star,
        rtol=1e-5,
        atol=1e-6,
        err_msg="tilde{Sigma} did not converge to theoretical Sigma* as lambda -> infty",
    )

    # 2. Continuous convergence of structural impact matrix B0 to DSGE impact matrix A0_dsge
    dr = nk_model_3shocks.decision_rules()
    sd_shocks = nk_model_3shocks._shock_sd(None)
    obs = ["y", "pi", "r"]
    shocks = ["eps_a", "eps_u", "eps_r"]
    shock_idx = [nk_model_3shocks.shocks.index(s) for s in shocks]
    A0_dsge = dr.ghu.loc[obs, shocks].to_numpy() @ np.diag(sd_shocks[shock_idx])

    np.testing.assert_allclose(
        res_inf.B0,
        A0_dsge,
        rtol=1e-4,
        atol=1e-5,
        err_msg="DSGE-VAR impact matrix B0 did not converge to DSGE impact matrix A0_dsge",
    )

    # 3. Impulse response convergence on impact (h=0) and short horizons
    for s in shocks:
        sd_s = float(sd_shocks[nk_model_3shocks.shocks.index(s)])
        dsge_irf = nk_model_3shocks.irf(s, horizon=5, size=sd_s)
        var_irf = res_inf.irf(horizon=5, shock=s)
        # Impact response at h=0 must match closely
        np.testing.assert_allclose(
            var_irf.loc[0, obs].to_numpy(),
            dsge_irf.loc[0, obs].to_numpy(),
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"Impact IRF for shock {s} did not match DSGE response",
        )


# ==============================================================================
# 3. Marginal Data Density Concavity & Optimal lambda
# ==============================================================================

def test_marginal_data_density_concavity_and_interior_optimum(
    nk_model_3shocks: LinearModel
):
    """Verify that log MDD over lambda in [0.2, 5.0] is concave with an interior optimum hat{lambda}."""
    # Build prior model with slight parameter offset (misspecification)
    mod_prior = """
    var y pi r a u;
    varexo eps_a eps_u eps_r;
    parameters beta sigma kappa phi_pi phi_y rho_a rho_u;
    beta = 0.99;
    sigma = 1.0;
    kappa = 0.5;
    phi_pi = 1.5;
    phi_y = 0.5;
    rho_a = 0.70;
    rho_u = 0.70;

    model;
    y = y(+1) - (1/sigma)*(r - pi(+1)) + (a(+1) - a);
    pi = beta*pi(+1) + kappa*y + u;
    r = phi_pi*pi + phi_y*y + eps_r;
    a = rho_a*a(-1) + eps_a;
    u = rho_u*u(-1) + eps_u;
    end;

    shocks;
    var eps_a; stderr 0.01;
    var eps_u; stderr 0.01;
    var eps_r; stderr 0.005;
    end;
    """
    m_prior = build_dynare(mod_prior)

    # Simulate from true model (rho_a=0.75, rho_u=0.65)
    sim = nk_model_3shocks.simulate(300, seed=42)
    data = sim[["y", "pi", "r"]]

    grid = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    res = estimate_dsge_var(
        m_prior,
        data,
        p=1,
        lamb="optimal",
        lambda_grid=grid,
    )

    assert res.hat_lambda is not None
    assert 0.2 < res.hat_lambda < 5.0, f"Expected interior optimum in (0.2, 5.0), got {res.hat_lambda}"
    assert res.log_mdd_grid is not None

    mdds = res.log_mdd_grid["log_mdd"].to_numpy()
    lambdas = res.log_mdd_grid["lambda"].to_numpy()

    # The maximum on the grid must be strictly interior (not at endpoints 0.2 or 5.0)
    max_idx = int(np.argmax(mdds))
    assert 0 < max_idx < len(grid) - 1, (
        f"Expected interior grid maximum, got maximum at index {max_idx} (lambda={lambdas[max_idx]})"
    )

    # Verify concavity around the peak: second difference < 0
    second_diff = mdds[max_idx + 1] - 2 * mdds[max_idx] + mdds[max_idx - 1]
    assert second_diff < 0, f"Log MDD is not strictly concave around peak; second_diff={second_diff}"


# ==============================================================================
# 4. Admissibility Bound lambda_min Enforcement
# ==============================================================================

def test_proper_prior_bound_enforcement(
    nk_model_3shocks: LinearModel, simulated_data: pd.DataFrame
):
    """Verify enforcement of the Inverted-Wishart proper prior bound lambda >= lambda_min = (k + n) / T."""
    p = 2
    n = 3
    k = 1 + n * p  # 7
    T = len(simulated_data) - p  # 298
    expected_lambda_min = (k + n) / T  # 10 / 298 ~= 0.03356

    # 1. Estimation with lambda < lambda_min must raise ValueError when check_bounds=True
    with pytest.raises(ValueError, match="below admissibility bound"):
        estimate_dsge_var(
            nk_model_3shocks,
            simulated_data,
            p=p,
            lamb=expected_lambda_min * 0.5,
            check_bounds=True,
        )

    # 2. Estimation at or above lambda_min succeeds
    res_valid = estimate_dsge_var(
        nk_model_3shocks,
        simulated_data,
        p=p,
        lamb=expected_lambda_min * 1.5,
        check_bounds=True,
    )
    np.testing.assert_allclose(res_valid.lambda_min, expected_lambda_min, atol=1e-6)

    # 3. Setting check_bounds=False permits evaluation below lambda_min without exception
    res_relaxed = estimate_dsge_var(
        nk_model_3shocks,
        simulated_data,
        p=p,
        lamb=expected_lambda_min * 0.5,
        check_bounds=False,
    )
    assert res_relaxed.lamb < expected_lambda_min


# ==============================================================================
# 5. Structural Identification & FEVD
# ==============================================================================

def test_structural_identification_and_fevd(
    nk_model_3shocks: LinearModel, simulated_data: pd.DataFrame
):
    """Verify structural rotation Q*, covariance reconstruction B0 B0' = tilde{Sigma}, and FEVD sum-to-1."""
    p = 2
    res = estimate_dsge_var(
        nk_model_3shocks,
        simulated_data,
        p=p,
        lamb=1.2,
        identification="dsge",
    )

    # 1. Exact covariance reconstruction: B0 @ B0.T == tilde{Sigma}
    np.testing.assert_allclose(
        res.B0 @ res.B0.T,
        res.Sigma,
        rtol=1e-5,
        atol=1e-7,
        err_msg="Structural impact matrix B0 does not satisfy B0 B0' = Sigma",
    )

    # 2. Orthonormality of rotation matrix Q* = Sigma_{chol}^{-1} B0
    Sigma_chol = np.linalg.cholesky(res.Sigma)
    Q_star = np.linalg.solve(Sigma_chol, res.B0)
    np.testing.assert_allclose(
        Q_star @ Q_star.T,
        np.eye(3),
        rtol=1e-5,
        atol=1e-7,
        err_msg="Rotation matrix Q* is not orthonormal (Q* Q*' != I)",
    )
    np.testing.assert_allclose(
        Q_star.T @ Q_star,
        np.eye(3),
        rtol=1e-5,
        atol=1e-7,
        err_msg="Rotation matrix Q* is not orthonormal (Q*' Q* != I)",
    )

    # 3. FEVD decomposition: shares sum to 1.0 for each variable at all horizons
    horizon = 15
    fevd_arr = res.fevd(horizon=horizon)
    assert fevd_arr.shape == (horizon + 1, 3, 3)

    # Sum across shocks (axis 2) must equal 1.0 for every horizon and variable
    sums_across_shocks = fevd_arr.sum(axis=2)
    np.testing.assert_allclose(
        sums_across_shocks,
        np.ones((horizon + 1, 3)),
        rtol=1e-5,
        atol=1e-6,
        err_msg="FEVD shares across structural shocks do not sum to 1.0",
    )


# ==============================================================================
# 6. Out-of-Sample Forecast Generation
# ==============================================================================

def test_forecast_generation_and_confidence_bands(
    nk_model_3shocks: LinearModel, simulated_data: pd.DataFrame
):
    """Verify forecast recursion, confidence bands upper > mean > lower, and error widening."""
    p = 2
    res = estimate_dsge_var(nk_model_3shocks, simulated_data, p=p, lamb=1.0)
    horizon = 8
    ci = 0.95

    fc = res.forecast(horizon=horizon, ci=ci)
    assert isinstance(fc, DSGEForecastResult)
    assert fc.horizon == horizon
    assert fc.ci == ci
    assert fc.mean.shape == (horizon, 3)
    assert fc.lower.shape == (horizon, 3)
    assert fc.upper.shape == (horizon, 3)

    # Band consistency: lower < mean < upper
    for v in res.names:
        assert np.all(fc.upper[v] > fc.mean[v]), f"Upper band not strictly greater than mean for {v}"
        assert np.all(fc.mean[v] > fc.lower[v]), f"Mean not strictly greater than lower band for {v}"

        # Error width upper - lower must widen with horizon
        band_width = (fc.upper[v] - fc.lower[v]).to_numpy()
        assert np.all(np.diff(band_width) >= -1e-8), f"Forecast band for {v} shrank with horizon"


# ==============================================================================
# 7. Presentation Contract
# ==============================================================================

def test_presentation_contract_six_methods(
    nk_model_3shocks: LinearModel, simulated_data: pd.DataFrame
):
    """Verify full presentation contract: .summary(), .plot(), .to_frame(), .to_markdown(), .to_latex(), .to_typst()."""
    p = 2
    res = estimate_dsge_var(
        nk_model_3shocks,
        simulated_data,
        p=p,
        lamb="optimal",
        lambda_grid=[0.5, 1.0, 2.0],
    )

    # 1. .summary()
    summary_text = res.summary()
    assert isinstance(summary_text, str)
    assert "DSGE-VAR Estimation" in summary_text
    assert "Prior weight (lambda)" in summary_text
    assert "Log Marginal Data Density" in summary_text
    assert "Optimal lambda (hat)" in summary_text

    # 2. .to_frame()
    df_coeff = res.to_frame()
    assert isinstance(df_coeff, pd.DataFrame)
    assert list(df_coeff.columns) == list(res.names)
    assert "const" in df_coeff.index
    assert "L1.y" in df_coeff.index
    assert "L2.r" in df_coeff.index

    # 3. .to_markdown(), .to_latex(), .to_typst()
    md_str = res.to_markdown()
    assert isinstance(md_str, str)
    assert "|" in md_str

    latex_str = res.to_latex()
    assert isinstance(latex_str, str)
    assert r"\begin{tabular}" in latex_str

    typst_str = res.to_typst()
    assert isinstance(typst_str, str)
    assert "#table" in typst_str

    # 4. .plot(kind='irf', 'mdd', 'forecast')
    fig_irf, ax_irf = res.plot(kind="irf", shock="eps_u", target="y", horizon=10)
    assert isinstance(fig_irf, plt.Figure)
    plt.close(fig_irf)

    fig_mdd, ax_mdd = res.plot(kind="mdd")
    assert isinstance(fig_mdd, plt.Figure)
    plt.close(fig_mdd)

    fig_fc, ax_fc = res.plot(kind="forecast", target="y", horizon=6)
    assert isinstance(fig_fc, plt.Figure)
    plt.close(fig_fc)


# ==============================================================================
# 8. LinearModel.dsge_var() Convenience Method
# ==============================================================================

def test_linearmodel_dsge_var_convenience_method(
    nk_model_3shocks: LinearModel, simulated_data: pd.DataFrame
):
    """Verify LinearModel.dsge_var delegates seamlessly to estimate_dsge_var."""
    p = 2
    res_method = nk_model_3shocks.dsge_var(simulated_data, p=p, lamb=1.5)
    res_direct = estimate_dsge_var(nk_model_3shocks, simulated_data, p=p, lamb=1.5)

    assert isinstance(res_method, DSGEVARResult)
    np.testing.assert_allclose(res_method.Sigma, res_direct.Sigma, atol=1e-10)
    np.testing.assert_allclose(res_method.B0, res_direct.B0, atol=1e-10)
    np.testing.assert_allclose(res_method.log_mdd, res_direct.log_mdd, atol=1e-10)


# ==============================================================================
# 9. Robust Error Handling
# ==============================================================================

def test_robust_error_handling(
    nk_model_3shocks: LinearModel, simulated_data: pd.DataFrame
):
    """Verify informative error handling for invalid arguments and non-stationary models."""
    # 1. Invalid lag order p <= 0
    with pytest.raises(ValueError, match="Lag order p must be positive"):
        estimate_dsge_var(nk_model_3shocks, simulated_data, p=0)

    # 2. Unknown identification scheme
    with pytest.raises(ValueError, match="Unknown identification"):
        estimate_dsge_var(nk_model_3shocks, simulated_data, p=2, identification="unknown")

    # 3. Unknown observable variable
    with pytest.raises(ValueError, match="is not declared in model.variables"):
        estimate_dsge_var(nk_model_3shocks, simulated_data, p=2, varobs=["y", "non_existent"])

    # 4. Unknown shock name
    with pytest.raises(ValueError, match="not declared in model.shocks"):
        estimate_dsge_var(
            nk_model_3shocks,
            simulated_data,
            p=2,
            shocks=["eps_a", "eps_u", "eps_fake"],
        )

    # 5. Dimension mismatch for structural DSGE identification
    with pytest.raises(ValueError, match="equal number of shocks"):
        estimate_dsge_var(
            nk_model_3shocks,
            simulated_data,
            p=2,
            varobs=["y", "pi", "r"],
            shocks=["eps_a", "eps_u"],  # 2 shocks for 3 observables
            identification="dsge",
        )


def test_dsge_var_intercept_false(
    nk_model_3shocks: LinearModel, simulated_data: pd.DataFrame
):
    """Verify estimation when intercept=False (no constant term, k = n * p)."""
    p = 2
    res = estimate_dsge_var(
        nk_model_3shocks,
        simulated_data,
        p=p,
        lamb=1.0,
        intercept=False,
    )
    assert np.allclose(res.intercept, 0.0)
    assert len(res.A_list) == p
    df = res.to_frame()
    assert "const" not in df.index
    assert len(df) == 3 * p


def test_dsge_var_cholesky_identification(
    nk_model_3shocks: LinearModel, simulated_data: pd.DataFrame
):
    """Verify identification='cholesky' returns lower triangular impact matrix B0."""
    res = estimate_dsge_var(
        nk_model_3shocks,
        simulated_data,
        p=2,
        lamb=1.0,
        identification="cholesky",
    )
    # B0 must be lower triangular
    assert np.allclose(np.triu(res.B0, k=1), 0.0)
    # B0 B0' must equal Sigma
    np.testing.assert_allclose(res.B0 @ res.B0.T, res.Sigma, atol=1e-7)


def test_dsge_var_multishock_irf_multiindex(
    nk_model_3shocks: LinearModel, simulated_data: pd.DataFrame
):
    """Verify irf(shock=None) returns MultiIndex DataFrame containing all shocks."""
    res = estimate_dsge_var(nk_model_3shocks, simulated_data, p=2, lamb=1.0)
    irf_all = res.irf(horizon=10, shock=None)
    assert isinstance(irf_all, pd.DataFrame)
    assert isinstance(irf_all.columns, pd.MultiIndex)
    assert set(irf_all.columns.levels[0]) == set(res.shock_names)
    assert set(irf_all.columns.levels[1]) == set(res.names)


def test_nonstationary_model_raises_value_error(
    nk_model_3shocks: LinearModel, simulated_data: pd.DataFrame
):
    """Verify that a model with non-stationary state eigenvalues raises ValueError."""
    import dataclasses
    bad_G = nk_model_3shocks.solution.G.copy()
    bad_G[0, 0] = 1.05
    bad_solution = dataclasses.replace(nk_model_3shocks.solution, G=bad_G)
    m_bad = dataclasses.replace(nk_model_3shocks, solution=bad_solution)
    with pytest.raises(ValueError, match="non-stationary eigenvalues"):
        estimate_dsge_var(m_bad, simulated_data, p=1, lamb=1.0)

