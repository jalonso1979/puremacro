"""Empirical Adversarial Stress Test Suite for Milestone 3 (Structural & Econometric Frontiers).

Authored by orch20_challenger_m3_1 to empirically falsify and stress-test:
1. Double Machine Learning (DML / Chernozhukov et al. 2018):
   - Non-linear partially linear DGP with collinear controls, trigonometric, exponential, and interaction confounding.
   - Naive OLS vs DML debiased recovery of true parameter theta_0.
   - Empirical coverage of nominal 95% confidence intervals across Monte Carlo replications.
   - Boundary sample sizes (N=50, N=1000) and cross-fitting fold counts (K=2, 5, 10).
2. Montiel Olea & Pflueger (2013) Weak IV in LP-IV and LA-LP:
   - Severely weak instruments (pi_z -> 0) yielding F_eff -> 0.
   - Anderson-Rubin confidence sets under weak IV (widening to unbounded rays / all real without NaN or crash).
   - Strong instruments design yielding F_eff >> critical value and tight bounded AR sets containing true parameter.
   - Multi-instrument weak identification edge cases.
3. Multi-Constraint OccBin (Guerrieri & Iacoviello 2015 extension):
   - Simultaneous binding of 2 constraints (regime 3).
   - Simultaneous binding of 3 constraints (regime 7).
   - Terminal boundary violation: constraint binds at t=T, verifying converged=False and UserWarning emission.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import pytest

from puremacro.causal import (
    DoubleMLPLR,
    DMLResult,
    LassoCoordinateDescent,
    RidgeGCV,
    dml_plr,
)
from puremacro.lp.iv import (
    lp_iv,
    compute_mop_effective_f,
    mop_critical_values,
)
from puremacro.lp.la_lp import (
    la_lp,
    la_lp_iv,
)
from puremacro.dsge import (
    build_dynare,
    LinearModel,
    OccBinConstraint,
    OccBinResult,
    solve_occbin,
    solve_multiconstraint_occbin,
)


# ============================================================================
# 1. Adversarial Statistical Testing of Double Machine Learning (DML)
# ============================================================================

def test_dml_nonlinear_dgp_bias_reduction():
    """Stress-test DML on non-linear partially linear DGP with collinear controls.

    DGP:
        Y = D * theta_0 + g(X) + U
        D = m(X) + V
    where:
        g(X) has non-linear terms: sin(X_0), exp(X_1 / 3), X_0 * X_1,
        m(X) has non-linear terms: sin(X_0) + 0.5 * X_1,
        and X has high collinearity among 20 covariates.
    """
    rng = np.random.default_rng(101)
    N, p = 600, 20
    theta_0 = 2.5

    # Generate collinear controls via Toeplitz correlation matrix (rho = 0.6)
    rho = 0.6
    sigma = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    L = np.linalg.cholesky(sigma)
    Z = rng.standard_normal((N, p)) @ L.T

    # Dictionary of features including non-linear transformations
    X_dict = np.column_stack([
        Z,
        np.sin(Z[:, 0]),
        np.exp(Z[:, 1] / 3.0),
        Z[:, 0] * Z[:, 1],
    ])

    m_Z = np.sin(Z[:, 0]) + 0.5 * Z[:, 1]
    g_Z = 1.2 * np.sin(Z[:, 0]) + 0.6 * np.exp(Z[:, 1] / 3.0) + 0.8 * (Z[:, 0] * Z[:, 1])

    V = 0.8 * rng.standard_normal(N)
    D = m_Z + V

    U = 0.8 * rng.standard_normal(N)
    Y = D * theta_0 + g_Z + U

    # 1. Naive OLS (regressing Y on D without debiased orthogonalization)
    b_naive = float(np.linalg.lstsq(D[:, None], Y, rcond=None)[0][0])
    naive_bias = abs(b_naive - theta_0)
    # Naive OLS must be heavily biased due to confounding
    assert naive_bias > 0.30, f"Expected large naive OLS bias, got {naive_bias:.4f}"

    # 2. DML with Lasso learner
    res_lasso = dml_plr(Y, D, X_dict, n_folds=5, learner="lasso", random_state=42)
    assert isinstance(res_lasso, DMLResult)
    lasso_bias = abs(res_lasso.theta - theta_0)
    assert lasso_bias < 0.15, f"Lasso DML failed to debias: theta={res_lasso.theta:.4f}, bias={lasso_bias:.4f}"
    assert res_lasso.ci_lower <= theta_0 <= res_lasso.ci_upper, "95% CI does not contain theta_0"

    # 3. DML with Ridge GCV learner
    res_ridge = dml_plr(Y, D, X_dict, n_folds=5, learner="ridge", random_state=42)
    assert isinstance(res_ridge, DMLResult)
    ridge_bias = abs(res_ridge.theta - theta_0)
    assert ridge_bias < 0.15, f"Ridge DML failed to debias: theta={res_ridge.theta:.4f}, bias={ridge_bias:.4f}"
    assert res_ridge.ci_lower <= theta_0 <= res_ridge.ci_upper, "95% CI does not contain theta_0"


def test_dml_coverage_monte_carlo():
    """Empirically test nominal 95% confidence interval coverage over 60 replications.

    Empirical coverage must fall within 88% and 99% of the replications: the lower
    bound is nominal 95% - 6%/-7%, the upper bound rejects an inflated standard error
    (a doubled SE covers all 60 replications).
    """
    n_reps = 60
    theta_0 = 1.80
    n, p = 400, 15
    rho = 0.5
    sigma = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    L = np.linalg.cholesky(sigma)

    covered_count = 0
    estimates = []

    for rep in range(n_reps):
        rng = np.random.default_rng(rep * 23 + 555)
        Z = rng.standard_normal((n, p)) @ L.T
        X = np.column_stack([
            Z,
            np.sin(Z[:, 0]),
            np.exp(Z[:, 1] / 4.0),
            Z[:, 0] * Z[:, 1],
        ])
        m_X = 0.5 * np.sin(Z[:, 0]) + 0.5 * Z[:, 2]
        g_X = 0.7 * np.sin(Z[:, 0]) + 0.4 * np.exp(Z[:, 1] / 4.0)

        D = m_X + 0.7 * rng.standard_normal(n)
        Y = D * theta_0 + g_X + 0.7 * rng.standard_normal(n)

        res = dml_plr(Y, D, X, n_folds=4, learner="ridge", random_state=rep)
        estimates.append(res.theta)
        if res.ci_lower <= theta_0 <= res.ci_upper:
            covered_count += 1

    emp_coverage = covered_count / n_reps
    mean_theta = float(np.mean(estimates))
    assert abs(mean_theta - theta_0) < 0.05, f"Monte Carlo mean {mean_theta:.4f} biased relative to {theta_0}"
    assert emp_coverage >= 0.88, f"Empirical coverage {emp_coverage * 100:.1f}% below acceptable threshold"
    assert emp_coverage <= 0.99, f"Empirical coverage {emp_coverage * 100:.1f}% suggests inflated standard errors"


def test_dml_extreme_sample_sizes():
    """Verify DML behavior under boundary sample sizes: N=50 (small) and N=1000 (large)."""
    theta_0 = 2.0
    p = 10

    # 1. Very small sample size N=50 with K=2
    rng_s = np.random.default_rng(12)
    X_s = rng_s.standard_normal((50, p))
    D_s = 0.5 * X_s[:, 0] + rng_s.standard_normal(50)
    Y_s = D_s * theta_0 + 0.5 * X_s[:, 1] + rng_s.standard_normal(50)

    res_small = dml_plr(Y_s, D_s, X_s, n_folds=2, learner="ridge", random_state=12)
    assert np.isfinite(res_small.theta)
    assert np.isfinite(res_small.se)
    assert res_small.se > 0
    assert res_small.ci_lower < res_small.ci_upper

    # 2. Large sample size N=1000 with K=10
    rng_l = np.random.default_rng(34)
    X_l = rng_l.standard_normal((1000, p))
    D_l = 0.5 * X_l[:, 0] + rng_l.standard_normal(1000)
    Y_l = D_l * theta_0 + 0.5 * X_l[:, 1] + rng_l.standard_normal(1000)

    res_large = dml_plr(Y_l, D_l, X_l, n_folds=10, learner="ridge", random_state=34)
    assert abs(res_large.theta - theta_0) < 0.10
    # Standard error should be substantially smaller in N=1000 than N=50
    assert res_large.se < res_small.se / 3.0


def test_dml_varying_k_folds():
    """Verify stability across different cross-fitting fold counts: K=2, K=5, K=10."""
    rng = np.random.default_rng(77)
    N, p = 300, 12
    theta_0 = 1.25

    X = rng.standard_normal((N, p))
    D = 0.6 * X[:, 0] - 0.4 * X[:, 1] + rng.standard_normal(N)
    Y = D * theta_0 + 0.8 * X[:, 0] + 0.5 * X[:, 2] + rng.standard_normal(N)

    results = {}
    for k in [2, 5, 10]:
        res = dml_plr(Y, D, X, n_folds=k, learner="ridge", random_state=42)
        results[k] = res
        assert abs(res.theta - theta_0) < 2.0 * res.se
        assert res.ci_lower <= theta_0 <= res.ci_upper

    # Consistency across K folds: all estimates should be close to each other
    estimates = [results[k].theta for k in [2, 5, 10]]
    assert max(estimates) - min(estimates) < 0.15


# ============================================================================
# 2. Adversarial Testing of Weak Instruments (MOP Effective F & AR Sets)
# ============================================================================

def test_lp_iv_severely_weak_instrument():
    """Stress-test lp_iv under severely weak instruments (pi_z -> 0).

    Verifies:
    1. Montiel Olea & Pflueger effective F is near zero (F_eff < 1.0 << 11.52).
    2. Anderson-Rubin confidence sets correctly widen or become unbounded without crashing.
    """
    rng = np.random.default_rng(202)
    T = 250
    # Instrument has near-zero relevance
    z = rng.standard_normal(T)
    x = 0.005 * z + rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.4 * y[t - 1] + 1.0 * x[t - 1] + rng.standard_normal()

    df = pd.DataFrame({"x": x, "y": y, "z": z})
    out = lp_iv(
        df,
        y="y",
        x="x",
        z="z",
        horizons=[1],
        n_lags=1,
        anderson_rubin=True,
    )

    mop_f = out["mop_f"].iloc[0]
    cv10 = out["mop_cv_10"].iloc[0]
    ar_type = out["ar_set_type"].iloc[0]

    # Effective F must be tiny
    assert mop_f < 2.0, f"Expected weak IV F_eff < 2.0, got {mop_f}"
    assert mop_f < cv10

    # Under very weak IV, AR set should either be unbounded or widely spanning
    assert ar_type in ("unbounded_rays", "all_real", "bounded")
    if ar_type == "bounded":
        ar_width = out["ar_hi"].iloc[0] - out["ar_lo"].iloc[0]
        # Bounded set under weak IV must be wide
        assert ar_width > 2.0, f"Expected wide AR set under weak IV, got width {ar_width}"


def test_la_lp_iv_zero_relevance_instrument():
    """Stress-test la_lp_iv with pure noise instrument (zero relevance)."""
    rng = np.random.default_rng(303)
    T = 300
    z = rng.standard_normal(T)
    x = rng.standard_normal(T)  # completely independent of z
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.3 * y[t - 1] + 0.8 * x[t - 1] + rng.standard_normal()

    df = pd.DataFrame({"x": x, "y": y, "z": z})

    out = la_lp_iv(
        df,
        y="y",
        x="x",
        z="z",
        horizons=[1],
        n_lags=2,
        extra_lags=1,
        anderson_rubin=True,
    )

    mop_f = out["mop_f"].iloc[0]
    cv10 = out["mop_cv_10"].iloc[0]
    cv20 = out["mop_cv_20"].iloc[0]
    ar_type = out["ar_set_type"].iloc[0]

    assert mop_f < cv20, f"Expected MOP F < cv20 ({cv20}) for irrelevant instrument, got {mop_f}"
    assert mop_f < cv10
    # Anderson-Rubin set should handle uninformative instrument safely
    assert ar_type in ("unbounded_rays", "all_real", "bounded")
    if ar_type == "bounded":
        assert out["ar_hi"].iloc[0] - out["ar_lo"].iloc[0] > 2.0


def test_lp_iv_ar_set_inversion_unbounded_rays():
    """Verify exact Anderson-Rubin inversion into unbounded rays (A <= 0, disc >= 0)."""
    # Seed 3 with T=150 and near-zero instrument creates A <= 0 and disc >= 0
    rng = np.random.default_rng(3)
    T = 150
    z = rng.standard_normal(T)
    x = 0.05 * z + rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.4 * y[t - 1] + 1.0 * x[t - 1] + rng.standard_normal()
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    out = lp_iv(df, y="y", x="x", z="z", horizons=[1], n_lags=1, anderson_rubin=True)

    ar_type = out["ar_set_type"].iloc[0]
    lo = out["ar_lo"].iloc[0]
    hi = out["ar_hi"].iloc[0]
    assert ar_type == "unbounded_rays", f"Expected unbounded_rays, got {ar_type}"
    # For unbounded_rays, confidence set is (-inf, hi] U [lo, inf) where lo > hi
    assert lo > hi, f"Expected lo > hi for inverted rays, got lo={lo}, hi={hi}"


def test_lp_iv_ar_set_inversion_all_real():
    """Verify exact Anderson-Rubin inversion into the entire real line (-inf, inf)."""
    # Seed 0 with tiny instrument creates A <= 0 and disc < 0
    rng = np.random.default_rng(0)
    T = 150
    z = rng.standard_normal(T)
    x = 0.0001 * z + rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.4 * y[t - 1] + 1.0 * x[t - 1] + rng.standard_normal()
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    out = lp_iv(df, y="y", x="x", z="z", horizons=[1], n_lags=1, anderson_rubin=True)

    ar_type = out["ar_set_type"].iloc[0]
    lo = out["ar_lo"].iloc[0]
    hi = out["ar_hi"].iloc[0]
    assert ar_type == "all_real", f"Expected all_real, got {ar_type}"
    assert np.isneginf(lo)
    assert np.isposinf(hi)


def test_lp_iv_and_la_lp_strong_instrument_recovery():
    """Verify strong instrument design yields high MOP F and tight bounded AR sets containing true effect."""
    rng = np.random.default_rng(404)
    T = 400
    z = rng.standard_normal(T)
    x = 1.2 * z + 0.3 * rng.standard_normal(T)  # strong instrument
    true_beta = -0.75
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.3 * y[t - 1] + true_beta * x[t - 1] + rng.standard_normal()

    df = pd.DataFrame({"x": x, "y": y, "z": z})

    # Test in lp_iv
    out_lp = lp_iv(df, y="y", x="x", z="z", horizons=[1], n_lags=1, anderson_rubin=True)
    assert out_lp["mop_f"].iloc[0] > 50.0
    assert out_lp["mop_f"].iloc[0] > out_lp["mop_cv_10"].iloc[0]
    assert out_lp["ar_set_type"].iloc[0] == "bounded"
    lo_lp = out_lp["ar_lo"].iloc[0]
    hi_lp = out_lp["ar_hi"].iloc[0]
    assert lo_lp <= true_beta <= hi_lp
    assert hi_lp - lo_lp < 0.50  # tight interval

    # Test in la_lp_iv
    out_la = la_lp_iv(df, y="y", x="x", z="z", horizons=[1], n_lags=2, extra_lags=1, anderson_rubin=True)
    assert out_la["mop_f"].iloc[0] > 40.0
    assert out_la["ar_set_type"].iloc[0] == "bounded"
    lo_la = out_la["ar_lo"].iloc[0]
    hi_la = out_la["ar_hi"].iloc[0]
    assert lo_la <= true_beta <= hi_la
    assert hi_la - lo_la < 0.60


def test_lp_iv_multi_instrument_weak_robustness():
    """Verify multiple weak instruments (kz=3) do not crash or produce division-by-zero."""
    rng = np.random.default_rng(505)
    T = 250
    z1 = rng.standard_normal(T)
    z2 = rng.standard_normal(T)
    z3 = rng.standard_normal(T)
    x = 0.02 * z1 + 0.01 * z2 - 0.02 * z3 + rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.2 * y[t - 1] + 0.5 * x[t - 1] + rng.standard_normal()

    df = pd.DataFrame({"x": x, "y": y, "z1": z1, "z2": z2, "z3": z3})
    out = lp_iv(df, y="y", x="x", z=["z1", "z2", "z3"], horizons=[1], n_lags=1, anderson_rubin=True)

    assert out["mop_f"].iloc[0] < out["mop_cv_10"].iloc[0]
    assert np.isfinite(out["mop_f"].iloc[0])
    ar_type = out["ar_set_type"].iloc[0]
    assert ar_type in ("bounded", "empty", "all_real", "unbounded_rays")


# ============================================================================
# 3. Multi-Constraint OccBin Stress Testing
# ============================================================================

@pytest.fixture
def triple_constraint_models():
    """Setup a New Keynesian DSGE model with 3 simultaneous occasionally binding constraints:
    - Constraint 1 (ZLB): r_t >= -r_ss (r_t pegged to -r_ss)
    - Constraint 2 (Borrowing cap): b_t <= b_bar (b_t pegged to b_bar)
    - Constraint 3 (Wage floor): w_t >= -w_bar (w_t pegged to -w_bar)
    """
    params = {
        "beta": 0.99,
        "sigma": 1.0,
        "kappa": 0.15,
        "phi_pi": 1.5,
        "phi_y": 0.25,
        "rho_r": 0.6,
        "rho_b": 0.5,
        "rho_w": 0.5,
        "rho_g": 0.7,
        "gamma_y": 0.2,
        "gamma_w": 0.3,
        "chi": 0.1,
        "r_ss": 0.015,
        "b_bar": 0.02,
        "w_bar": 0.025,
    }

    variables = ["y", "pi", "r", "b", "w", "g"]
    shocks = ["eps_g", "eps_r", "eps_b", "eps_w"]

    # Regime 0: unconstrained
    def ref_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_r),
            curr.b - (p.rho_b * lag.b + p.gamma_y * curr.y + shocks_v.eps_b),
            curr.w - (p.rho_w * lag.w + p.gamma_w * curr.y + shocks_v.eps_w),
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    # Regime 1: ZLB
    def zlb_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (-p.r_ss),
            curr.b - (p.rho_b * lag.b + p.gamma_y * curr.y + shocks_v.eps_b),
            curr.w - (p.rho_w * lag.w + p.gamma_w * curr.y + shocks_v.eps_w),
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    # Regime 2: Borrowing cap
    def borr_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_r),
            curr.b - p.b_bar,
            curr.w - (p.rho_w * lag.w + p.gamma_w * curr.y + shocks_v.eps_w),
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    # Regime 3: Wage floor
    def wage_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_r),
            curr.b - (p.rho_b * lag.b + p.gamma_y * curr.y + shocks_v.eps_b),
            curr.w - (-p.w_bar),
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    steady_state = {v: 0.0 for v in variables}
    m_ref = build_dynare(ref_eqs, variables=variables, shocks=shocks, params=params, steady_state=steady_state)
    m_zlb = build_dynare(zlb_eqs, variables=variables, shocks=shocks, params=params, steady_state=steady_state, check_steady_state=False, strict=False)
    m_borr = build_dynare(borr_eqs, variables=variables, shocks=shocks, params=params, steady_state=steady_state, check_steady_state=False, strict=False)
    m_wage = build_dynare(wage_eqs, variables=variables, shocks=shocks, params=params, steady_state=steady_state, check_steady_state=False, strict=False)

    c_zlb = OccBinConstraint(variable="r", threshold=-params["r_ss"], operator="<")
    c_borr = OccBinConstraint(variable="b", threshold=params["b_bar"], operator=">")
    c_wage = OccBinConstraint(variable="w", threshold=-params["w_bar"], operator="<")

    return m_ref, {"zlb": m_zlb, "borrowing": m_borr, "wage": m_wage}, {"zlb": c_zlb, "borrowing": c_borr, "wage": c_wage}


def test_occbin_simultaneous_triple_constraint_binding(triple_constraint_models):
    """Stress-test 3 simultaneous occasionally binding constraints (ZLB, Borrowing cap, Wage floor).

    Verifies:
    1. Regime 7 = (1 | 2 | 4) binds simultaneously.
    2. Economy returns to reference regime at terminal period (regime = 0).
    3. Solver reports converged = True.
    """
    m_ref, m_dict, c_dict = triple_constraint_models

    shocks = np.zeros((45, 4))
    shocks[0, 0] = -0.07  # eps_g: triggers ZLB
    shocks[0, 2] = 0.06   # eps_b: triggers borrowing cap
    shocks[0, 3] = -0.06  # eps_w: triggers wage floor

    res = solve_multiconstraint_occbin(
        m_unconstrained=m_ref,
        m_constrained_dict=m_dict,
        shock_seq=shocks,
        constraints=c_dict,
        horizon=45,
    )

    assert isinstance(res, OccBinResult)
    assert res.converged
    regimes = np.asarray(res.regimes)

    # Verify simultaneous 3-constraint binding (regime 7 = 1 | 2 | 4)
    has_triple_binding = np.any(regimes == 7)
    assert has_triple_binding, f"Expected simultaneous triple-constraint binding (regime 7), regimes={regimes[:10]}"

    # Terminal period must be slack
    assert regimes[-1] == 0


def test_occbin_terminal_slack_violation_and_warning(triple_constraint_models):
    """Adversarial stress-test: persistent shock keeping constraints binding at terminal period T.

    Verifies:
    1. Warning emitted (UserWarning: constraint still binds at terminal period).
    2. converged is strictly False.
    3. res.regimes[-1] != 0.
    """
    m_ref, m_dict, c_dict = triple_constraint_models

    # Horizon of 3 periods with severe shock so constraints cannot relax by t=3
    shocks = np.zeros((3, 4))
    shocks[0, 0] = -0.09
    shocks[0, 2] = 0.08
    shocks[0, 3] = -0.08

    with pytest.warns(UserWarning, match="constraint still binds at terminal period"):
        res = solve_multiconstraint_occbin(
            m_unconstrained=m_ref,
            m_constrained_dict=m_dict,
            shock_seq=shocks,
            constraints=c_dict,
            horizon=3,
        )

    assert not res.converged, "Expected converged=False when terminal slack condition fails"
    assert res.regimes[-1] != 0, f"Expected terminal period to be binding, got {res.regimes[-1]}"


def test_occbin_multiconstraint_input_validation_guards(triple_constraint_models):
    """Adversarial stress-test: input validation guards for solve_multiconstraint_occbin."""
    m_ref, m_dict, c_dict = triple_constraint_models
    shocks = np.zeros((10, 4))

    # Horizon <= 0
    with pytest.raises(ValueError, match="horizon must be an integer >= 1"):
        solve_multiconstraint_occbin(m_ref, m_dict, shocks, c_dict, horizon=0)

    # max_iter <= 0
    with pytest.raises(ValueError, match="max_iter must be an integer >= 1"):
        solve_multiconstraint_occbin(m_ref, m_dict, shocks, c_dict, max_iter=0)

    # Empty constrained models
    with pytest.raises(ValueError, match="requires at least one constrained model"):
        solve_multiconstraint_occbin(m_ref, {}, shocks, c_dict)

    # Incompatible constraints modifying the same equation row
    m_duplicate = {"zlb_1": m_dict["zlb"], "zlb_2": m_dict["zlb"]}
    with pytest.raises(ValueError, match="Incompatible constraint regimes"):
        solve_multiconstraint_occbin(m_ref, m_duplicate, shocks)


def test_dml_input_validation_guards():
    """Adversarial stress-test: input validation guards for DoubleMLPLR."""
    X = np.random.randn(50, 4)
    D = np.random.randn(50)
    Y = np.random.randn(50)

    # n_folds < 2
    with pytest.raises(ValueError, match="n_folds must be an integer >= 2"):
        DoubleMLPLR(n_folds=1)

    # Sample size mismatch
    with pytest.raises(ValueError, match="Sample size mismatch"):
        DoubleMLPLR().fit(Y[:40], D, X)

    with pytest.raises(ValueError, match="Sample size mismatch"):
        DoubleMLPLR().fit(Y, D[:30], X)

    # Unknown learner
    with pytest.raises(ValueError, match="Unknown learner"):
        DoubleMLPLR(learner="deep_neural_network").fit(Y, D, X)

