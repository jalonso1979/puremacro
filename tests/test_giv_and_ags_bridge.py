"""Tests for Track 4a (Granular IV) and Track 5 (AGS Sensitivity & Sovereign Empirical Bridges)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.inference.giv import GIVResult, estimate_giv
from puremacro.sovereign.dsa import (
    FiscalReactionEstimateResult,
    estimate_fiscal_reaction_function,
    fit_stochastic_dsa_from_data,
)
from puremacro.sovereign.smm import SovereignSMMResult, calibrate_fiscal_sovereign_smm
from puremacro.structural.bridge import (
    AGSSensitivityResult,
    MomentTargets,
    ags_sensitivity,
    fit_structural,
)


def _assert_presentation_standard(res: object) -> None:
    """Verify all 7 Result-Object Presentation Standard 2.0+ methods."""
    s = res.summary()  # type: ignore[attr-defined]
    assert isinstance(s, str) and len(s) > 20
    df1 = res.to_dataframe()  # type: ignore[attr-defined]
    df2 = res.to_frame()  # type: ignore[attr-defined]
    assert isinstance(df1, pd.DataFrame) and not df1.empty
    assert isinstance(df2, pd.DataFrame) and df1.shape == df2.shape
    md = res.to_markdown()  # type: ignore[attr-defined]
    lx = res.to_latex()  # type: ignore[attr-defined]
    ty = res.to_typst()  # type: ignore[attr-defined]
    assert isinstance(md, str) and "|" in md
    assert isinstance(lx, str) and "\\begin{tabular}" in lx
    assert isinstance(ty, str) and "table(" in ty
    fig = res.plot()  # type: ignore[attr-defined]
    assert fig is not None
    plt.close("all")


def test_estimate_giv_recovers_multiplier_and_presentation() -> None:
    """Verify Gabaix & Koijen (2024) Granular IV recovers the true structural multiplier."""
    rng = np.random.default_rng(2026)
    T, N = 160, 20
    # Pareto-like skewed size shares (fat-tailed firm/country sizes)
    raw_sizes = (np.arange(1, N + 1, dtype=float)) ** (-1.15)
    shares_vec = raw_sizes / np.sum(raw_sizes)
    shares = np.tile(shares_vec, (T, 1))

    # Common unobserved factor f_t correlated with aggregate macro outcome
    f_t = rng.normal(0.0, 1.2, size=(T, 1))
    loadings = rng.uniform(0.5, 1.5, size=(1, N))
    # Heteroskedastic idiosyncratic shocks u_{i,t}
    idio_scales = np.linspace(0.5, 1.5, N)
    u_idio = rng.normal(0.0, 1.0, size=(T, N)) * idio_scales[None, :]
    shocks = f_t @ loadings + u_idio

    # Granular aggregate X_t = sum_i s_i * shocks_{i,t} + 0.5 * f_t (endogenous common shock)
    X_t = np.sum(shares * shocks, axis=1) + 0.8 * f_t[:, 0]
    true_multiplier = 1.75
    # Outcome Y_t = alpha + true_multiplier * X_t + 1.1 * f_t + eps_t (so OLS is biased upward!)
    Y_t = 0.5 + true_multiplier * X_t + 1.1 * f_t[:, 0] + rng.normal(0.0, 0.15, size=T)

    res = estimate_giv(
        shocks,
        shares,
        Y_t,
        endog_macro=X_t,
        n_factors=1,
        precision_weights=True,
        hac_lags=4,
        leave_one_out_top_k=3,
    )
    assert isinstance(res, GIVResult)
    # GIV estimate should be close to true_multiplier = 1.75, while naive OLS is biased upward
    assert abs(res.multiplier - true_multiplier) < 0.25
    assert res.ols_multiplier > res.multiplier + 0.15
    assert res.first_stage_f > 10.0
    assert res.herfindahl > 1.0 / np.sqrt(N)
    assert len(res.leave_one_out) == 3
    assert np.isfinite(res.overid_j_stat)
    assert 0.0 <= res.overid_j_pvalue <= 1.0
    _assert_presentation_standard(res)


def test_ags_sensitivity_exact_linear_and_presentation() -> None:
    """Verify Andrews, Gentzkow & Shapiro (2017) Lambda and standardized Lambda_tilde."""
    # Known linear model m(theta) = G @ theta where G is (3, 2)
    G = np.array(
        [
            [2.0, 0.5],
            [0.2, 1.5],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    theta_true = np.array([1.2, -0.8])
    cov_m = np.diag([0.04, 0.09, 0.16])
    m_hat = G @ theta_true + np.array([0.02, -0.03, 0.01])

    targets = MomentTargets(
        values=m_hat,
        covariance=cov_m,
        labels=("m1", "m2", "m3"),
        units=("pct", "pct", "pct"),
    )
    fit = fit_structural(
        lambda th: G @ th,
        targets,
        theta0=[0.0, 0.0],
        parameter_names=("alpha", "beta"),
        jac=lambda th: G,
        assume_correct_specification=True,
    )
    assert fit.success and fit.inference_valid

    sens = ags_sensitivity(fit)
    assert isinstance(sens, AGSSensitivityResult)
    W = np.linalg.inv(cov_m)
    expected_dtheta_dm = np.linalg.inv(G.T @ W @ G) @ G.T @ W
    expected_lambda = -expected_dtheta_dm
    np.testing.assert_allclose(sens.dtheta_dm, expected_dtheta_dm, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sens.lambda_matrix, expected_lambda, rtol=1e-10, atol=1e-12)

    # Check standardized sensitivity: Lambda_tilde_{j,k} = Lambda_{j,k} * SE(m_k) / SE(theta_j)
    se_m = np.sqrt(np.diag(cov_m))
    se_th = fit.standard_errors
    expected_std = expected_lambda * (se_m[None, :] / se_th[:, None])
    np.testing.assert_allclose(sens.standardized_sensitivity, expected_std, rtol=1e-10, atol=1e-12)
    # Row sum of squares of standardized sensitivity under optimal W equals 1.0 when moments are orthogonal!
    row_ss = np.sum(sens.standardized_sensitivity ** 2, axis=1)
    np.testing.assert_allclose(row_ss, np.ones(2), rtol=1e-10, atol=1e-10)
    _assert_presentation_standard(sens)


def test_estimate_fiscal_reaction_and_empirical_dsa() -> None:
    """Verify Bohn (1998) / Ghosh et al. (2013) fiscal reaction estimation and empirical DSA."""
    rng = np.random.default_rng(1998)
    T = 120
    debt = np.zeros(T)
    pb = np.zeros(T)
    g = rng.normal(2.5, 0.8, size=T)
    r = rng.normal(3.0, 0.6, size=T)
    y_gap = rng.normal(0.0, 1.2, size=T)

    # True nonlinear reaction with positive Bohn slope at moderate debt and cubic fatigue around d* = 0.95
    debt[0] = 0.45
    pb[0] = 0.5
    for t in range(1, T):
        # Slowly varying debt path spanning 0.35 to 1.15
        debt[t] = 0.75 + 0.35 * np.sin(2.0 * np.pi * t / 60.0) + rng.normal(0.0, 0.02)
        d_prev = debt[t - 1]
        # Marginal response d(pb)/dd = 4.5 * d * (0.95 - d) -> positive for d < 0.95, zero at d = 0.95
        poly_d = 4.5 * (0.475 * (d_prev ** 2) - (d_prev ** 3) / 3.0)
        pb[t] = -0.8 + 0.55 * pb[t - 1] + poly_d + 0.15 * y_gap[t] + rng.normal(0.0, 0.04)

    df = pd.DataFrame({"g": g, "r": r, "pb": pb, "debt": debt, "y_gap": y_gap})
    est = estimate_fiscal_reaction_function(
        df,
        pb_col="pb",
        debt_col="debt",
        y_gap_col="y_gap",
        cubic_fatigue=True,
        hac_lags=4,
    )
    assert isinstance(est, FiscalReactionEstimateResult)
    assert est.bohn_t_stat > 2.0
    assert est.bohn_solvency_holds
    assert est.fatigue_threshold is not None
    assert 0.75 <= est.fatigue_threshold <= 1.15
    assert est.r2 > 0.70
    _assert_presentation_standard(est)

    # End-to-end empirical stochastic DSA from DataFrame
    dsa_res = fit_stochastic_dsa_from_data(
        df,
        g_col="g",
        r_col="r",
        pb_col="pb",
        debt_col="debt",
        y_gap_col="y_gap",
        horizon=16,
        n_sims=200,
        seed=42,
        threshold=0.85,
        use_bohn_reaction=True,
    )
    assert dsa_res.paths.shape == (200, 17)
    assert "fiscal_reaction_estimate" in dsa_res.metadata
    assert "var_A" in dsa_res.metadata


def test_calibrate_fiscal_sovereign_smm() -> None:
    """Verify SMM / minimum-distance calibration of fiscal sovereign default and AGS sensitivity."""
    # Generate synthetic target moments at known parameters on a small fast grid
    targets = MomentTargets(
        values=np.array([4.2, 1.8, 4.5]),
        covariance=np.diag([0.25, 0.16, 0.36]),
        labels=("debt_pct", "spread_mean_pct", "spending_pct"),
        units=("pct", "pct", "pct"),
    )
    smm_res = calibrate_fiscal_sovereign_smm(
        targets,
        param_names=("beta", "d0"),
        theta0=(0.94, -0.18),
        bounds=((0.85, 0.985), (-0.35, -0.05)),
        n_b=31,
        n_y=5,
        sigma_eps=0.005,
        max_nfev=6,
    )
    assert isinstance(smm_res, SovereignSMMResult)
    assert smm_res.sensitivity.lambda_matrix.shape == (2, 3)
    assert set(smm_res.parameter_dict.keys()) == {"beta", "d0"}
    _assert_presentation_standard(smm_res)
