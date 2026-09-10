"""Empirical end-to-end stress testing across integrated Phase C capabilities.

Validates the full Phase C frontier in puremacro:
1. Optimal Discretionary vs Commitment Policy on 3-equation New Keynesian models:
   - Dennis (2007) Riccati policy iteration convergence (||F_{k+1} - F_k||_infty < 1e-9).
   - Bellman optimality equation and PSD of Riccati value matrix V.
   - Inflation bias quantification (E[pi^disc] - E[pi^comm] > 0 when y* > 0, matching theoretical formula).
   - Stabilization bias quantification (Loss^disc > Loss^comm).
2. DSGE-VAR Hybrid Modeling (Del Negro & Schorfheide 2004):
   - Prior admissibility bound enforcement lambda >= lambda_min = (k + n) / T.
   - Closed-form Marginal Data Density (MDD) optimization with interior hat{lambda} and concavity.
   - Structural shock identification via DSGE rotation Q* (Q* Q*' = I).
   - Exact covariance decomposition B0 B0' = tilde{Sigma}.
   - Forecast error variance decomposition (FEVD) shares summing to 1.0 at every horizon.
   - Asymptotic limits: OLS convergence as lambda -> lambda_min and DSGE convergence as lambda -> infty.
3. News & Anticipated Shocks Engine:
   - State-space companion augmentation preserving Blanchard-Kahn determinacy (H zero eigenvalues).
   - News shock simulation with lead k=4: zero pre-realization revision for predetermined states (t < 4).
   - Immediate impact jump on announcement (t=0) for forward-looking jump variables.
   - Realization at t=k and smooth decay for t > k.
   - Forecast error variance decomposition across surprise and news leads summing to 1.0.
4. Integrated End-to-End Pipeline combining Discretion/Commitment, DSGE-VAR, and News Shocks.
5. Adversarial stress testing on extreme numerical boundaries (deep leads, high persistence, near-unity discount).
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
from puremacro.dsge.policy import (
    discretionary_policy,
    lq_commitment,
    optimal_policy,
)
from puremacro.dsge._results import DiscretionaryPolicyResult, PolicyResult
from puremacro.dsge.dsge_var import DSGEVARResult, estimate_dsge_var
from puremacro.dsge.news import (
    news_irf,
    decompose_news,
    augment_news_state_space,
    NewsIRFResult,
    NewsDecompositionResult,
)


# ==============================================================================
# Model Fixtures
# ==============================================================================

@pytest.fixture
def nk_policy_model() -> LinearModel:
    """Canonical 3-equation New Keynesian model with persistent cost-push shock."""
    mod = """
    var y pi r u;
    varexo eps_u;
    parameters beta sigma kappa phi_pi phi_y rho_u;
    beta = 0.99;
    sigma = 1.0;
    kappa = 0.5;
    phi_pi = 1.5;
    phi_y = 0.5;
    rho_u = 0.5;

    model;
    y = y(+1) - (1/sigma)*(r - pi(+1));
    pi = beta*pi(+1) + kappa*y + u;
    r = phi_pi*pi + phi_y*y;
    u = rho_u*u(-1) + eps_u;
    end;

    shocks;
    var eps_u; stderr 1.0;
    end;
    """
    return build_dynare(mod)


@pytest.fixture
def nk_3shocks_model() -> LinearModel:
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
def rbc_model() -> LinearModel:
    """Standard RBC model under Dynare canonical form."""
    mod = """
    var c k z;
    varexo eps_z;
    parameters alpha beta delta rho sigma;
    alpha = 0.33;
    beta = 0.99;
    delta = 0.025;
    rho = 0.95;
    sigma = 1.0;

    initval;
    c = 2.0;
    k = 25.0;
    z = 1.0;
    end;

    model;
    c^(-sigma) = beta * c(+1)^(-sigma) * (alpha * z(+1) * k^(alpha-1) + 1 - delta);
    c + k - (1-delta)*k(-1) = z * k(-1)^alpha;
    z = (1 - rho) + rho * z(-1) + eps_z;
    end;

    shocks;
    var eps_z; stderr 0.01;
    end;
    """
    return build_dynare(mod)


# ==============================================================================
# 1. Discretion vs Commitment Policy Solution on 3-Equation NK Model
# ==============================================================================

def test_empirical_discretion_vs_commitment_policy(nk_policy_model: LinearModel):
    """Verify Discretion vs Commitment policy solution on a 3-equation NK model."""
    m = nk_policy_model
    beta = 0.99
    kappa = 0.5
    lambda_y = 0.25
    y_star = 0.05
    weights = {"pi": 1.0, "y": lambda_y}

    # 1. Solve under Discretion
    res_disc = optimal_policy(
        m,
        loss=weights,
        rule="discretion",
        instruments="r",
        beta=beta,
        y_star=y_star,
        tol=1e-9,
        max_iter=2000,
    )

    assert isinstance(res_disc, DiscretionaryPolicyResult)
    assert res_disc.converged is True
    assert res_disc.diff < 1e-9, f"Convergence diff {res_disc.diff} exceeded 1e-9"
    assert 0 < res_disc.iterations < 100, f"Unexpected iterations: {res_disc.iterations}"

    # 2. Riccati Matrix V properties
    V = res_disc.V
    assert isinstance(V, np.ndarray)
    assert V.shape == (len(m.variables), len(m.variables))
    # Symmetry
    np.testing.assert_allclose(V, V.T, atol=1e-10, err_msg="Riccati matrix V must be symmetric")
    # Positive semi-definiteness: all eigenvalues >= -1e-12
    eigs = np.linalg.eigvalsh(V)
    assert np.all(eigs >= -1e-12), f"Min eigenvalue {np.min(eigs)} < 0"

    # Bellman optimality equation: V = G' W G + beta * G' V G
    G = res_disc.transition
    variables = list(m.variables)
    W_full = np.zeros_like(V)
    for v, w in weights.items():
        idx = variables.index(v)
        W_full[idx, idx] = float(w)

    bellman_rhs = G.T @ W_full @ G + beta * G.T @ V @ G
    bellman_error = float(np.max(np.abs(V - bellman_rhs)))
    assert bellman_error < 1e-6, f"Bellman residual {bellman_error:.2e} exceeded 1e-6"

    # 3. Policy feedback matrix F
    F = res_disc.F
    assert "r" in F.index
    assert "u" in F.columns
    # Instrument r responds positively to cost-push shock u (leaning against the wind)
    assert F.loc["r", "u"] > 0.40, f"Expected strong monetary tightening to cost-push shock, got {F.loc['r', 'u']}"

    # 4. Inflation bias quantification against theoretical Clarida-Gali-Gertler formula
    # Theoretical inflation bias = kappa * lambda_y / (lambda_y * (1 - beta) + kappa^2) * y*
    expected_inflation_bias = (kappa * lambda_y) / (lambda_y * (1 - beta) + kappa**2) * y_star
    assert res_disc.inflation_bias > 0.0
    np.testing.assert_allclose(
        res_disc.inflation_bias,
        expected_inflation_bias,
        rtol=1e-4,
        err_msg="Discretionary inflation bias does not match theoretical formula",
    )

    # When y* = 0, inflation bias must be exactly zero
    res_disc_zero = optimal_policy(
        m,
        loss=weights,
        rule="discretion",
        instruments="r",
        beta=beta,
        y_star=0.0,
        tol=1e-9,
    )
    assert res_disc_zero.inflation_bias == 0.0

    # 5. Solve under Commitment
    res_comm = optimal_policy(
        m,
        loss=weights,
        rule="commitment",
        instruments="r",
        beta=beta,
    )
    assert isinstance(res_comm, PolicyResult)
    assert res_comm.regime == "commitment"
    assert len(res_comm.multipliers) > 0, "Commitment solution must contain Lagrange multipliers"

    # 6. Stabilization bias: Loss^disc > Loss^comm under persistent cost-push shocks
    assert res_disc.loss > 0.0
    assert res_comm.loss > 0.0
    assert res_disc.stabilization_bias > 0.0, (
        f"Expected Loss^disc ({res_disc.loss}) > Loss^comm ({res_comm.loss})"
    )


# ==============================================================================
# 2. DSGE-VAR Estimation with Simulated Data
# ==============================================================================

def test_empirical_dsge_var_optimization_and_identification(nk_3shocks_model: LinearModel):
    """Verify DSGE-VAR estimation with simulated data, log MDD optimization and structural identification."""
    m_true = nk_3shocks_model
    obs = ["y", "pi", "r"]
    shocks = ["eps_a", "eps_u", "eps_r"]

    # Simulate 300 observations from true DGP
    sim = m_true.simulate(300, seed=42)
    data = sim[obs]

    # Prior model with slight parameter offset (misspecification)
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

    # 1. Prior admissibility bound check
    p = 1
    n = len(obs)
    k = 1 + n * p
    T = len(data) - p
    lambda_min = (k + n) / T

    with pytest.raises(ValueError, match="below admissibility bound"):
        estimate_dsge_var(m_prior, data, p=p, lamb=lambda_min * 0.5, check_bounds=True)

    # 2. Log MDD grid optimization
    grid = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    res_opt = estimate_dsge_var(
        m_prior,
        data,
        p=p,
        lamb="optimal",
        lambda_grid=grid,
        identification="dsge",
    )

    assert isinstance(res_opt, DSGEVARResult)
    assert res_opt.hat_lambda is not None
    # Verify optimal lambda is interior
    assert 0.2 < res_opt.hat_lambda < 5.0, f"Expected interior optimum in (0.2, 5.0), got {res_opt.hat_lambda}"

    # Verify concavity around the peak
    assert res_opt.log_mdd_grid is not None
    mdds = res_opt.log_mdd_grid["log_mdd"].to_numpy()
    lambdas = res_opt.log_mdd_grid["lambda"].to_numpy()
    max_idx = int(np.argmax(mdds))
    assert 0 < max_idx < len(grid) - 1, f"Optimum at boundary index {max_idx}"
    second_diff = mdds[max_idx + 1] - 2 * mdds[max_idx] + mdds[max_idx - 1]
    assert second_diff < 0, f"Log MDD not concave around peak; second_diff={second_diff}"

    # 3. Structural Shock Identification
    # Orthonormality of rotation matrix Q* = Sigma_{chol}^{-1} B0
    Sigma_chol = np.linalg.cholesky(res_opt.Sigma)
    Q_star = np.linalg.solve(Sigma_chol, res_opt.B0)
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

    # Exact covariance reconstruction B0 B0' = tilde{Sigma}
    np.testing.assert_allclose(
        res_opt.B0 @ res_opt.B0.T,
        res_opt.Sigma,
        rtol=1e-5,
        atol=1e-7,
        err_msg="Structural impact matrix B0 does not reconstruct Sigma",
    )

    # 4. FEVD variance shares sum to 1.0 at every horizon
    horizon = 20
    fevd_arr = res_opt.fevd(horizon=horizon)
    assert fevd_arr.shape == (horizon + 1, 3, 3)
    fevd_sums = fevd_arr.sum(axis=2)
    np.testing.assert_allclose(
        fevd_sums,
        np.ones((horizon + 1, 3)),
        rtol=1e-5,
        atol=1e-6,
        err_msg="FEVD shares across structural shocks do not sum to 1.0",
    )
    assert (fevd_arr >= -1e-6).all(), "Negative FEVD share detected"
    assert (fevd_arr <= 1.0 + 1e-6).all(), "FEVD share > 1 detected"

    # 5. Asymptotic convergence checks
    # A. OLS limit as lambda -> 0 (relaxed bound)
    res_ols = estimate_dsge_var(m_prior, data, p=p, lamb=1e-8, check_bounds=False)
    np.testing.assert_allclose(res_ols.to_frame().to_numpy(), res_ols.Phi_ols, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(res_ols.Sigma, res_ols.Sigma_ols, rtol=1e-5, atol=1e-6)

    # B. DSGE limit as lambda -> infty
    res_inf = estimate_dsge_var(m_prior, data, p=p, lamb=1e8, identification="dsge")
    np.testing.assert_allclose(res_inf.to_frame().to_numpy(), res_inf.Phi_star, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(res_inf.Sigma, res_inf.Sigma_star, rtol=1e-5, atol=1e-6)


# ==============================================================================
# 3. News Shock Simulation with Lead k=4 & Variance Decomposition
# ==============================================================================

def test_empirical_news_shocks_lead_4_and_fevd(nk_3shocks_model: LinearModel):
    """Verify news shock simulation with lead k=4: zero pre-realization revision, jump on announcement, and FEVD = 1.0."""
    m = nk_3shocks_model
    shock = "eps_a"
    lead = 4
    size = 1.0
    horizon = 30

    # 1. State-space companion augmentation BK determinacy preservation
    m_aug = augment_news_state_space(m, shock=shock, max_lead=lead)
    assert m_aug.is_determinate, "Blanchard-Kahn determinacy failed on augmented news system"
    # Augmentation adds exactly lead zero eigenvalues
    evals = np.sort(np.abs(m_aug.eigenvalues))
    np.testing.assert_allclose(evals[:lead], 0.0, atol=1e-12)

    # 2. News IRF with lead k=4
    res_news = news_irf(m, shock=shock, lead=lead, horizon=horizon, size=size)
    assert isinstance(res_news, NewsIRFResult)
    df = res_news.irf

    # Zero pre-realization revision for predetermined state 'a' for t < 4
    for t in range(lead):
        assert abs(df.loc[t, "a"]) < 1e-12, (
            f"Exogenous state 'a' showed non-zero revision at t={t} (< lead {lead}): {df.loc[t, 'a']}"
        )

    # Jump on announcement at t=0 for forward-looking jump variables ('y', 'pi')
    y_0 = df.loc[0, "y"]
    assert abs(y_0) > 1e-4, f"Forward-looking output 'y' did not jump on impact at t=0: {y_0}"

    # Realization at t = 4: state 'a' equals size (1.0)
    np.testing.assert_allclose(
        df.loc[lead, "a"],
        size,
        rtol=1e-6,
        atol=1e-8,
        err_msg=f"State 'a' did not realize at size {size} at t={lead}",
    )

    # Smooth decay for t > lead: a_t = size * rho_a^(t - lead)
    rho_a = float(m._params["rho_a"])
    for t in range(lead + 1, lead + 6):
        expected_a = size * (rho_a ** (t - lead))
        np.testing.assert_allclose(
            df.loc[t, "a"],
            expected_a,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Discrepancy in shock decay at t={t}",
        )

    # 3. Variance decomposition (decompose_news)
    res_decomp = decompose_news(m, shock=shock, horizon=horizon, max_lead=lead)
    assert isinstance(res_decomp, NewsDecompositionResult)

    shares_df = res_decomp.variance_shares
    assert shares_df.shape == (len(m.variables), 1 + lead)
    expected_cols = ["surprise"] + [f"news_{k}" for k in range(1, lead + 1)]
    assert list(shares_df.columns) == expected_cols

    # Forecast error variance shares sum to 1.0 across surprise and news leads
    row_sums = shares_df.sum(axis=1).to_numpy()
    np.testing.assert_allclose(
        row_sums,
        1.0,
        atol=1e-12,
        err_msg="FEVD variance shares do not sum to 1.0 across surprise and news channels",
    )
    assert (shares_df.to_numpy() >= -1e-12).all(), "Negative news variance share found"
    assert (shares_df.to_numpy() <= 1.0 + 1e-12).all(), "News variance share > 1 found"

    # Total news property: total_news + surprise == 1.0
    np.testing.assert_allclose(
        res_decomp.total_news + shares_df["surprise"],
        1.0,
        atol=1e-12,
    )

    # Dynamic shares sum to 1.0 at every single forecast step
    for var in m.variables:
        dyn_df = res_decomp.dynamic_shares[var]
        assert len(dyn_df) == horizon + 1
        np.testing.assert_allclose(
            dyn_df.sum(axis=1).to_numpy(),
            1.0,
            atol=1e-12,
            err_msg=f"Dynamic FEVD shares for variable {var} do not sum to 1.0 at all horizons",
        )


# ==============================================================================
# 4. Integrated End-to-End Pipeline
# ==============================================================================

def test_integrated_phase_c_end_to_end_pipeline(nk_3shocks_model: LinearModel):
    """Integrated empirical pipeline executing Discretion/Commitment, DSGE-VAR, and News shocks together."""
    m = nk_3shocks_model

    # Step A: Optimal Monetary Policy under Discretion vs Commitment
    res_disc = m.optimal_policy(
        loss={"pi": 1.0, "y": 0.25},
        rule="discretion",
        instruments="r",
        y_star=0.02,
        tol=1e-9,
    )
    res_comm = m.optimal_policy(
        loss={"pi": 1.0, "y": 0.25},
        rule="commitment",
        instruments="r",
    )
    assert res_disc.converged is True
    assert res_disc.inflation_bias > 0.0
    assert res_disc.stabilization_bias > 0.0
    assert len(res_comm.multipliers) > 0

    # Step B: Data Generation & DSGE-VAR Hybrid Estimation
    sim_data = m.simulate(250, seed=123)[["y", "pi", "r"]]
    res_var = m.dsge_var(
        sim_data,
        p=1,
        lamb="optimal",
        lambda_grid=[0.3, 0.6, 1.0, 1.5, 2.5, 4.0],
        identification="dsge",
    )
    assert res_var.hat_lambda is not None
    assert 0.3 <= res_var.hat_lambda <= 5.0
    # FEVD sum to 1
    fevd = res_var.fevd(horizon=12)
    np.testing.assert_allclose(fevd.sum(axis=2), 1.0, atol=1e-6)

    # Step C: News Shock Analysis (lead k=4)
    res_news = m.news_irf(shock="eps_a", lead=4, horizon=25, size=1.0)
    assert abs(res_news.irf.loc[0, "a"]) < 1e-12
    assert abs(res_news.irf.loc[3, "a"]) < 1e-12
    np.testing.assert_allclose(res_news.irf.loc[4, "a"], 1.0, atol=1e-8)

    # News decomposition
    res_decomp = decompose_news(m, shock="eps_a", horizon=25, max_lead=4)
    np.testing.assert_allclose(res_decomp.variance_shares.sum(axis=1).to_numpy(), 1.0, atol=1e-12)


# ==============================================================================
# 5. Adversarial Stress Testing on Numerical Boundaries
# ==============================================================================

def test_adversarial_stress_numerical_boundaries(nk_policy_model: LinearModel, nk_3shocks_model: LinearModel):
    """Stress test boundary cases: deep leads, near-unity discount factors, and extreme persistences."""
    # 1. Discretion policy under extreme discount factor beta = 0.999 and high persistence rho = 0.90
    mod_extreme = """
    var y pi r u;
    varexo eps_u;
    parameters beta sigma kappa phi_pi phi_y rho_u;
    beta = 0.999;
    sigma = 1.0;
    kappa = 0.15;
    phi_pi = 1.5;
    phi_y = 0.5;
    rho_u = 0.90;

    model;
    y = y(+1) - (1/sigma)*(r - pi(+1));
    pi = beta*pi(+1) + kappa*y + u;
    r = phi_pi*pi + phi_y*y;
    u = rho_u*u(-1) + eps_u;
    end;

    shocks;
    var eps_u; stderr 1.0;
    end;
    """
    m_ext = build_dynare(mod_extreme)
    res_disc_ext = optimal_policy(
        m_ext,
        loss={"pi": 1.0, "y": 0.5},
        rule="discretion",
        instruments="r",
        beta=0.999,
        tol=1e-9,
        max_iter=3000,
    )
    assert res_disc_ext.converged is True
    assert res_disc_ext.diff < 1e-9

    # 2. News shock with deep lead k=12 on RBC model
    m_rbc = nk_3shocks_model
    res_news_deep = news_irf(m_rbc, shock="eps_a", lead=12, horizon=40, size=1.0)
    # Exogenous state a must be 0 for t = 0 .. 11
    for t in range(12):
        assert abs(res_news_deep.irf.loc[t, "a"]) < 1e-12
    # At t=12, realization
    np.testing.assert_allclose(res_news_deep.irf.loc[12, "a"], 1.0, atol=1e-8)

    # 3. Variance decomposition with deep lead k=8
    res_decomp_deep = decompose_news(m_rbc, shock="eps_a", horizon=40, max_lead=8)
    np.testing.assert_allclose(
        res_decomp_deep.variance_shares.sum(axis=1).to_numpy(),
        1.0,
        atol=1e-12,
    )


# ==============================================================================
# 6. Advanced Adversarial Stress: News Linearity, Forecasts, and Loss Parser
# ==============================================================================

def test_adversarial_news_linearity_and_surprise_parity(nk_3shocks_model: LinearModel):
    """Verify that news shocks satisfy exact linearity and lead=0 matches surprise IRF to machine precision."""
    m = nk_3shocks_model
    shock = "eps_a"
    lead = 4
    size_pos = 1.5
    size_neg = -1.5

    irf_pos = news_irf(m, shock=shock, lead=lead, horizon=30, size=size_pos).irf
    irf_neg = news_irf(m, shock=shock, lead=lead, horizon=30, size=size_neg).irf

    # Exact linearity: IRF(-size) == -IRF(size)
    np.testing.assert_allclose(
        irf_pos.to_numpy(),
        -irf_neg.to_numpy(),
        atol=1e-12,
        err_msg="News IRF violates linearity/symmetry under negative innovation size",
    )

    # Lead 0 parity with standard surprise IRF within 1e-12
    irf_lead0 = news_irf(m, shock=shock, lead=0, horizon=30, size=1.0).irf
    irf_surp = m.irf(shock, horizon=30, size=1.0)
    np.testing.assert_allclose(
        irf_lead0.to_numpy(),
        irf_surp.to_numpy(),
        atol=1e-12,
        err_msg="News IRF at lead=0 does not match model.irf within 1e-12",
    )


def test_adversarial_dsge_var_forecast_and_ci_bounds(nk_3shocks_model: LinearModel):
    """Verify out-of-sample forecast generation and monotonically widening error bands."""
    m = nk_3shocks_model
    sim = m.simulate(200, seed=99)[["y", "pi", "r"]]
    res_var = estimate_dsge_var(m, sim, p=2, lamb=1.5, identification="dsge")

    fc = res_var.forecast(horizon=12, ci=0.90)
    assert fc.mean.shape == (12, 3)
    assert fc.lower.shape == (12, 3)
    assert fc.upper.shape == (12, 3)

    # Upper > Mean > Lower strictly for all horizons
    assert (fc.upper.to_numpy() > fc.mean.to_numpy()).all()
    assert (fc.mean.to_numpy() > fc.lower.to_numpy()).all()

    # Band width (upper - lower) should widen monotonically or weakly monotonically over short horizon
    band_width = (fc.upper - fc.lower).to_numpy()
    for j in range(3):
        assert band_width[-1, j] >= band_width[0, j], f"Forecast bands did not widen for variable {j}"


def test_adversarial_string_loss_parsing_parity(nk_policy_model: LinearModel):
    """Verify that string loss expressions achieve identical numerical results to dict losses."""
    m = nk_policy_model
    res_dict = optimal_policy(
        m,
        loss={"pi": 1.0, "y": 0.25},
        rule="discretion",
        instruments="r",
        y_star=0.05,
        tol=1e-9,
    )
    res_str = optimal_policy(
        m,
        loss="pi^2 + 0.25 * (y - 0.05)^2",
        rule="discretion",
        instruments="r",
        tol=1e-9,
    )
    np.testing.assert_allclose(
        res_dict.F.to_numpy(),
        res_str.F.to_numpy(),
        atol=1e-8,
        err_msg="String loss parsing produced different feedback matrix F than dictionary loss",
    )
    np.testing.assert_allclose(
        res_dict.inflation_bias,
        res_str.inflation_bias,
        atol=1e-6,
        err_msg="String loss parsing produced different inflation bias than dictionary loss",
    )

