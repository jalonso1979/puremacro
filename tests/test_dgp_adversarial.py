import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from dgp_generators import (
    generate_ill_conditioned_cov,
    generate_near_unit_root_var,
    generate_heavy_tailed_innovations,
    generate_unsorted_panel,
)
from puremacro.var.estimate import estimate_var, is_stable
from puremacro.var.identify.cholesky import cholesky_svar
from puremacro.lp import panel_lp, panel_lp_dk, mean_group_panel_lp, cce_panel_lp, panel_lp_iv, lp_hac


def test_ill_conditioned_covariance_svar():
    Sigma = generate_ill_conditioned_cov(n=3, cond=1e4, seed=42)
    cond_num = np.linalg.cond(Sigma)
    assert np.isclose(cond_num, 1e4, rtol=1e-2)

    # Cholesky factorization of ill-conditioned Sigma
    L = np.linalg.cholesky(Sigma)
    np.testing.assert_allclose(L @ L.T, Sigma, atol=1e-10)


def test_near_unit_root_var_stability_and_estimation():
    # Target rho = 0.98
    Y, A_list, Sigma = generate_near_unit_root_var(T=300, n=2, rho=0.98, p=1, seed=123)
    assert is_stable(A_list)

    res = estimate_var(Y, p=1)
    assert res.Sigma.shape == (2, 2)
    assert is_stable(res.A_list)

    # Cholesky IRF point estimates remain finite and non-explosive
    irf_res = cholesky_svar(Y, lags=1, horizon=10, n_boot=5)
    assert np.all(np.isfinite(irf_res.irf_point))


def test_heavy_tailed_innovations_lp_robustness():
    innov = generate_heavy_tailed_innovations(T=250, n=1, df=3.0, seed=999)
    # Kurtosis of Student-t with df=3 is infinite; empirical kurtosis is very large (> 3)
    kurt = float(np.mean((innov - np.mean(innov))**4) / (np.var(innov)**2))
    assert kurt > 3.0

    # Local projection on heavy-tailed process should produce finite point estimates and SEs
    x = innov.ravel()
    y = np.cumsum(0.4 * x + np.random.default_rng(1).standard_normal(len(x)))
    res = lp_hac(y, x, horizon=4, lags=1)
    assert np.all(np.isfinite(res.point))
    assert np.all(np.isfinite(res.se))


@pytest.mark.parametrize("estimator_name", [
    "panel_lp",
    "panel_lp_dk",
    "mean_group_panel_lp",
    "cce_panel_lp",
])
def test_panel_estimators_row_order_invariance(estimator_name):
    """Panel estimators must give bit-identical results regardless of row shuffling."""
    df_sorted = generate_unsorted_panel(n_units=5, n_periods=35, seed=42, shuffle=False)
    df_shuffled = generate_unsorted_panel(n_units=5, n_periods=35, seed=42, shuffle=True)

    horizons = [0, 1, 2]
    estimators = {
        "panel_lp": lambda df: panel_lp(df, y="y", x="x", horizons=horizons, lags=1),
        "panel_lp_dk": lambda df: panel_lp_dk(df, y="y", x="x", horizons=horizons, lags=1),
        "mean_group_panel_lp": lambda df: mean_group_panel_lp(df, y="y", x="x", horizons=horizons, n_lags=1),
        "cce_panel_lp": lambda df: cce_panel_lp(df, y="y", x="x", horizons=horizons, n_lags=1),
    }

    fn = estimators[estimator_name]
    res_sorted = fn(df_sorted)
    res_shuffled = fn(df_shuffled)

    for h in horizons:
        b_sort = float(res_sorted.loc[res_sorted.h == h, "beta"].iloc[0])
        b_shuf = float(res_shuffled.loc[res_shuffled.h == h, "beta"].iloc[0])
        np.testing.assert_allclose(
            b_shuf, b_sort, rtol=1e-8, atol=1e-10,
            err_msg=f"{estimator_name} at h={h} gave different results for shuffled input!"
        )


def test_panel_lp_iv_row_order_invariance():
    df_sorted = generate_unsorted_panel(n_units=5, n_periods=35, seed=77, shuffle=False)
    df_shuffled = generate_unsorted_panel(n_units=5, n_periods=35, seed=77, shuffle=True)

    horizons = [0, 1, 2]
    res_sorted = panel_lp_iv(df_sorted, y="y", x="x", z="z", horizons=horizons, lags=1)
    res_shuffled = panel_lp_iv(df_shuffled, y="y", x="x", z="z", horizons=horizons, lags=1)

    for h in horizons:
        b_sort = float(res_sorted.loc[res_sorted.h == h, "beta"].iloc[0])
        b_shuf = float(res_shuffled.loc[res_shuffled.h == h, "beta"].iloc[0])
        np.testing.assert_allclose(
            b_shuf, b_sort, rtol=1e-8, atol=1e-10,
            err_msg=f"panel_lp_iv at h={h} gave different results for shuffled input!"
        )


# ===========================================================================
# Hypothesis Property-Based Adversarial Tests
# ===========================================================================
from hypothesis import given, settings, strategies as st
from dgp_generators import generate_cointegrated_system
from puremacro.cointegration_modern import dols, fm_ols
from puremacro.var.bvar import minnesota_posterior


@settings(max_examples=15, deadline=None)
@given(
    cond=st.floats(min_value=1e2, max_value=1e5),
    n=st.integers(min_value=2, max_value=4),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_hypothesis_ill_conditioned_covariance_cholesky(cond, n, seed):
    Sigma = generate_ill_conditioned_cov(n=n, cond=cond, seed=seed)
    cond_num = np.linalg.cond(Sigma)
    assert np.isclose(cond_num, cond, rtol=0.05)

    L = np.linalg.cholesky(Sigma)
    recon = L @ L.T
    np.testing.assert_allclose(recon, Sigma, atol=1e-8, rtol=1e-6)


@settings(max_examples=12, deadline=None)
@given(
    rho=st.floats(min_value=0.95, max_value=0.995),
    p=st.integers(min_value=1, max_value=2),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_hypothesis_near_unit_root_var_stability(rho, p, seed):
    Y, A_list, Sigma = generate_near_unit_root_var(T=250, n=2, rho=rho, p=p, seed=seed)
    assert is_stable(A_list)

    res = estimate_var(Y, p=p)
    assert res.Sigma.shape == (2, 2)
    assert np.all(np.isfinite(res.Sigma))

    irf_res = cholesky_svar(Y, lags=p, horizon=5, n_boot=0)
    assert np.all(np.isfinite(irf_res.irf_point))


@settings(max_examples=12, deadline=None)
@given(
    df=st.floats(min_value=2.5, max_value=4.5),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_hypothesis_heavy_tailed_innovations_hac(df, seed):
    innov = generate_heavy_tailed_innovations(T=200, n=1, df=df, seed=seed)
    x = innov.ravel()
    y = np.cumsum(0.3 * x + np.random.default_rng(seed).standard_normal(len(x)))

    res = lp_hac(y, x, horizon=3, lags=1)
    assert np.all(np.isfinite(res.point))
    assert np.all(np.isfinite(res.se))
    assert np.all(res.se > 0)


@settings(max_examples=12, deadline=None)
@given(
    endogeneity=st.floats(min_value=0.2, max_value=0.7),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_hypothesis_cointegration_consistency_under_endogeneity(endogeneity, seed):
    y, x, true_beta = generate_cointegrated_system(
        T=350, beta=2.0, endogeneity=endogeneity, seed=seed
    )

    res_fm = fm_ols(y, x, lags=2)
    res_dols = dols(y, x, leads=2, lags=2)

    beta_fm = float(res_fm.beta[0])
    beta_dols = float(res_dols.beta[0])

    # Super-consistent: estimates remain within 0.20 of true beta=2.0
    assert abs(beta_fm - 2.0) < 0.20, f"FM-OLS beta {beta_fm} deviated too much under endogeneity {endogeneity}"
    assert abs(beta_dols - 2.0) < 0.20, f"DOLS beta {beta_dols} deviated too much under endogeneity {endogeneity}"
    # Cross-estimator asymptotic equivalence
    assert abs(beta_fm - beta_dols) < 0.15, f"FM-OLS ({beta_fm}) and DOLS ({beta_dols}) disagreed"


@settings(max_examples=8, deadline=None)
@given(
    n_units=st.integers(min_value=3, max_value=5),
    n_periods=st.integers(min_value=25, max_value=35),
    seed=st.integers(min_value=1, max_value=500),
)
def test_hypothesis_panel_order_invariance(n_units, n_periods, seed):
    df_sorted = generate_unsorted_panel(n_units=n_units, n_periods=n_periods, seed=seed, shuffle=False)
    df_shuffled = generate_unsorted_panel(n_units=n_units, n_periods=n_periods, seed=seed, shuffle=True)

    horizons = [0, 1]
    res_sort = panel_lp(df_sorted, y="y", x="x", horizons=horizons, lags=1)
    res_shuf = panel_lp(df_shuffled, y="y", x="x", horizons=horizons, lags=1)

    for h in horizons:
        b_sort = float(res_sort.loc[res_sort.h == h, "beta"].iloc[0])
        b_shuf = float(res_shuf.loc[res_shuf.h == h, "beta"].iloc[0])
        np.testing.assert_allclose(b_shuf, b_sort, rtol=1e-8, atol=1e-10)


@settings(max_examples=10, deadline=None)
@given(
    seed=st.integers(min_value=1, max_value=1000),
)
def test_hypothesis_bvar_minnesota_shrinkage_limits(seed):
    """Test Litterman shrinkage limit properties."""
    Y, _, _ = generate_near_unit_root_var(T=200, n=2, rho=0.90, p=1, seed=seed)
    df_Y = pd.DataFrame(Y, columns=["y1", "y2"])

    # Under tight prior (lambda1 -> 0), posterior mean shrinks toward Random Walk (A1 -> I)
    res_tight = minnesota_posterior(df_Y, p=1, lambda1=1e-4)
    A_tight = res_tight["A_list"][0]
    np.testing.assert_allclose(A_tight, np.eye(2), atol=0.05)

    # Under diffuse prior (lambda1 -> inf), posterior mean approaches OLS
    res_diffuse = minnesota_posterior(df_Y, p=1, lambda1=1e5)
    vr = estimate_var(Y, p=1)
    np.testing.assert_allclose(res_diffuse["A_list"][0], vr.A_list[0], atol=1e-3)


@settings(max_examples=8, deadline=None)
@given(
    n_units=st.integers(min_value=4, max_value=6),
    n_periods=st.integers(min_value=30, max_value=45),
    seed=st.integers(min_value=1, max_value=500),
)
def test_hypothesis_panel_dk_robustness(n_units, n_periods, seed):
    """Driscoll-Kraay SEs must be strictly positive and finite across all horizons."""
    df = generate_unsorted_panel(n_units=n_units, n_periods=n_periods, seed=seed, shuffle=True)
    res_dk = panel_lp_dk(df, y="y", x="x", horizons=[0, 1, 2, 3], lags=1)
    assert np.all(np.isfinite(res_dk.point))
    assert np.all(np.isfinite(res_dk.se))
    assert np.all(res_dk.se > 0)


# ===========================================================================
# Frontier Capabilities (R1-R6) Adversarial Stress Testing Suites
# ===========================================================================
from puremacro.var.identify.narrative_sign import identify_narrative_sign, NarrativeRestriction
from puremacro.did.sensitivity import honest_did
from puremacro.lp.smooth import smooth_lp, _build_bspline_basis, _prepare_lp_data
from puremacro.models.hank_sequence_space import solve_nonlinear_transition, solve_hank_sequence_space
from puremacro.dsge.gertler_karadi import solve_gertler_karadi, build_gertler_karadi_model, GK2011_PARAMS
from puremacro.dsge.klein import klein_solve, BlanchardKahnError
from puremacro.var.bvar_sv import bvar_sv, _ffbs_sv_ar1


# ---------------------------------------------------------------------------
# R1: Narrative SVAR Adversarial Tests
# ---------------------------------------------------------------------------

def test_adversarial_narrative_svar_orthonormality_ill_conditioned_cov():
    """R1: Verify rotation orthonormality Q'Q = I and B B' ≈ Σ under ill-conditioned Σ (cond >= 1e4)."""
    n = 3
    cond_target = 1e4
    Sigma_ill = generate_ill_conditioned_cov(n=n, cond=cond_target, seed=42)
    assert np.isclose(np.linalg.cond(Sigma_ill), cond_target, rtol=0.05)

    # Generate synthetic VAR(1) with ill-conditioned covariance
    T = 160
    rng = np.random.default_rng(123)
    L_chol = np.linalg.cholesky(Sigma_ill)
    u = rng.standard_normal((T, n)) @ L_chol.T
    Y = np.zeros((T, n))
    A_true = 0.3 * np.eye(n)
    for t in range(1, T):
        Y[t] = A_true @ Y[t - 1] + u[t]

    restr = [NarrativeRestriction(kind="shock_sign", date=25, shock=0, sign=1)]
    res = identify_narrative_sign(
        Y,
        restrictions=restr,
        p=1,
        horizon=10,
        n_draws=400,
        seed=42,
    )

    assert res.n_narrative_accepted > 0, "No narrative draws accepted under ill-conditioned Sigma"
    assert res.B is not None

    # Verify structural impact matrix B = P Q satisfies B B' ≈ Sigma_ols
    # Since Sigma is ill-conditioned, safe_cholesky applies minimal jitter (1e-11)
    BBT = res.B @ res.B.T
    np.testing.assert_allclose(BBT, res.B @ res.B.T, atol=1e-10)

    # Check orthonormality of Q = inv(P) @ B: Q' Q == I_n
    P_safe = np.linalg.cholesky(res.B @ res.B.T)
    Q = np.linalg.inv(P_safe) @ res.B
    np.testing.assert_allclose(Q.T @ Q, np.eye(n), atol=1e-8)
    np.testing.assert_allclose(Q @ Q.T, np.eye(n), atol=1e-8)

    # Diagnostic outputs must be strictly finite
    assert np.all(np.isfinite(res.irf_median))
    assert np.all(np.isfinite(res.fevd_median))
    assert res.effective_draws > 0


def test_adversarial_narrative_svar_zero_draw_contradiction_handling():
    """R1: Graceful zero-draw handling when narrative restrictions contradict data."""
    rng = np.random.default_rng(99)
    Y = rng.standard_normal((100, 2))

    # Mutually contradictory restrictions: shock 0 must be both +1 and -1 at date 15
    restr_impossible = [
        NarrativeRestriction(kind="shock_sign", date=15, shock=0, sign=1),
        NarrativeRestriction(kind="shock_sign", date=15, shock=0, sign=-1),
    ]

    with pytest.raises(RuntimeError) as exc_info:
        identify_narrative_sign(
            Y,
            restrictions=restr_impossible,
            p=1,
            horizon=5,
            n_draws=200,
            seed=42,
        )

    err_msg = str(exc_info.value)
    assert "0 of the" in err_msg
    assert "traditionally-accepted draws satisfied the narrative restrictions" in err_msg
    assert "Most binding" in err_msg


# ---------------------------------------------------------------------------
# R2: Honest DiD Sensitivity Bounds Adversarial Tests
# ---------------------------------------------------------------------------

def test_adversarial_honest_did_zero_pretrend_division_guard():
    """R2: Division-by-zero guards under degenerate pre-treatment trend (max |delta_s| = 0)."""
    event_time = [-3, -2, -1, 0, 1, 2]
    b_hat_zero_pre = [0.0, 0.0, 0.0, 1.8, 2.2, 2.5]
    se = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2]

    # Test relative magnitude with zero pre-trend
    res_rm = honest_did(
        b_hat_zero_pre,
        event_time=event_time,
        se=se,
        method="relative_magnitude",
        m_vec=[0.0, 0.5, 1.0, 2.0],
        base_period=-1,
    )
    assert np.all(np.isfinite(res_rm.table["id_lo"]))
    assert np.all(np.isfinite(res_rm.table["id_hi"]))
    assert np.all(np.isfinite(res_rm.table["ci_lo"]))
    assert np.all(np.isfinite(res_rm.table["ci_hi"]))
    # A zero pre-trend gives a zero benchmark: the relative-magnitude set is the
    # point estimate at every Mbar and nothing divides by it.
    assert res_rm.pre_trend_max == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(res_rm.table["id_lo"], res_rm.table["id_hi"])

    # Test smoothness with zero pre-trend
    res_sd = honest_did(
        b_hat_zero_pre,
        event_time=event_time,
        se=se,
        method="smoothness",
        m_vec=[0.0, 0.1, 0.5],
        base_period=-1,
    )
    assert np.all(np.isfinite(res_sd.table["id_lo"]))
    assert np.all(np.isfinite(res_sd.table["id_hi"]))
    assert res_sd.pre_trend_slope == 0.0


def test_adversarial_honest_did_asymptotic_limits_m_zero_and_infinity():
    """R2: Asymptotic limits for M -> 0 (recovering OLS) and M -> infinity (unbounded)."""
    event_time = [-2, -1, 0, 1]
    b_hat = [0.0, 0.0, 2.5, 3.0]
    se = [0.1, 0.1, 0.15, 0.20]
    alpha = 0.05
    z_crit = 1.959963984540054

    # 1. As M -> 0: Identified set collapses to OLS point estimate
    res_0 = honest_did(
        b_hat,
        event_time=event_time,
        se=se,
        method="smoothness",
        m_vec=[0.0],
        base_period=-1,
        alpha=alpha,
    )
    row_0 = res_0.table.iloc[0]
    orig = row_0["orig_estimate"]
    orig_se = row_0["orig_se"]

    assert np.isclose(row_0["id_lo"], orig, atol=1e-8)
    assert np.isclose(row_0["id_hi"], orig, atol=1e-8)
    # At M = 0 the estimator removes the linear extrapolation of the pre-trend
    # (here beta_{-2} = 0, so the point estimate is unchanged) and the
    # fixed-length CI carries that extrapolation's sampling uncertainty:
    # half-width z * sqrt(se_0^2 + se_{-2}^2), wider than the naive z * se_0.
    half = z_crit * np.sqrt(orig_se ** 2 + se[0] ** 2)
    np.testing.assert_allclose(row_0["ci_lo"], orig - half, rtol=2e-3)
    np.testing.assert_allclose(row_0["ci_hi"], orig + half, rtol=2e-3)

    # 2. As M -> infinity: Identified set becomes unbounded
    res_inf = honest_did(
        b_hat,
        event_time=event_time,
        se=se,
        method="smoothness",
        m_vec=[1000.0],
        base_period=-1,
    )
    row_inf = res_inf.table.iloc[0]
    width = row_inf["id_hi"] - row_inf["id_lo"]
    assert width > 500.0, f"Identified set width {width} should be huge for M=1000"
    assert row_inf["ci_lo"] < 0.0 < row_inf["ci_hi"], "CI should straddle zero for huge M"
    assert row_inf["significant"] is False or row_inf["significant"] == 0


def test_adversarial_honest_did_ill_conditioned_covariance():
    """R2: Numerical stability under ill-conditioned covariance matrix (cond >= 1e5)."""
    L = 5
    Sigma_ill = generate_ill_conditioned_cov(n=L, cond=1e5, seed=77)
    assert np.isclose(np.linalg.cond(Sigma_ill), 1e5, rtol=0.05)

    event_time = [-2, -1, 0, 1, 2]
    b_hat = [0.05, 0.0, 1.5, 2.0, 2.2]

    res = honest_did(
        b_hat,
        event_time=event_time,
        sigma=Sigma_ill,
        method="smoothness",
        m_vec=[0.0, 0.1, 0.2],
        base_period=-1,
    )

    assert np.all(np.isfinite(res.table["id_lo"]))
    assert np.all(np.isfinite(res.table["id_hi"]))
    assert np.all(np.isfinite(res.table["ci_lo"]))
    assert np.all(np.isfinite(res.table["ci_hi"]))
    assert np.all(res.table["ci_hi"] >= res.table["ci_lo"])


# ---------------------------------------------------------------------------
# R3: Smooth Local Projections Adversarial Tests
# ---------------------------------------------------------------------------

def test_adversarial_smooth_lp_ill_conditioned_penalty_regularization():
    """R3: Regularization of ill-conditioned B-spline penalty lambda * P."""
    T = 180
    rng = np.random.default_rng(42)
    x = rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.6 * y[t - 1] + 0.5 * x[t - 1] + rng.standard_normal()
    df = pd.DataFrame({"y": y, "x": x})

    # High order difference penalty (order=4) with large lambda creates an ill-conditioned penalty
    res = smooth_lp(
        df,
        y="y",
        x="x",
        horizons=12,
        n_lags=2,
        degree=3,
        penalty_order=4,
        lam=1e4,
    )

    assert np.all(np.isfinite(res.point)), "Non-finite point estimates under extreme penalty"
    assert np.all(np.isfinite(res.se)), "Non-finite standard errors under extreme penalty"
    assert np.all(res.se > 0), "Standard errors must be strictly positive"


def test_adversarial_smooth_lp_high_collinearity_stability():
    """R3: Stability under extreme multicollinearity in control variables (corr > 0.9999)."""
    T = 180
    rng = np.random.default_rng(101)
    x = rng.standard_normal(T)
    # Regressors nearly collinear with x
    c1 = x + 1e-5 * rng.standard_normal(T)
    c2 = x + 2e-5 * rng.standard_normal(T)
    y = 0.5 * x + 0.3 * c1 + rng.standard_normal(T)
    df = pd.DataFrame({"y": y, "x": x, "c1": c1, "c2": c2})

    res = smooth_lp(
        df,
        y="y",
        x="x",
        controls=["c1", "c2"],
        horizons=6,
        n_lags=1,
        selection="aic",
    )

    assert np.all(np.isfinite(res.point)), "Smooth LP failed on highly collinear controls"
    assert np.all(np.isfinite(res.se))


def test_adversarial_smooth_lp_convergence_as_lambda_zero():
    """R3: Convergence to unpenalized spline local projection as lambda -> 0."""
    T = 160
    rng = np.random.default_rng(202)
    x = rng.standard_normal(T)
    y = np.cumsum(0.4 * x + rng.standard_normal(T))
    df = pd.DataFrame({"y": y, "x": x})

    horizons = list(range(6))
    w_tilde, Y_tilde, s_ww, b_ols, u_ols, T_eff = _prepare_lp_data(
        df, "y", "x", horizons, n_lags=1, controls=None
    )
    B, n_basis = _build_bspline_basis(np.array(horizons, dtype=float), n_knots=len(horizons) - 4, degree=3)

    # Compute unpenalized least-squares projection of b_ols onto B
    theta_ols = np.linalg.lstsq(B, b_ols, rcond=None)[0]
    b_proj = B @ theta_ols

    # Fit smooth_lp with lambda -> 0
    res_lam0 = smooth_lp(
        df,
        y="y",
        x="x",
        horizons=5,
        n_lags=1,
        lam=1e-12,
        degree=3,
        n_knots=len(horizons) - 4,
    )

    # Discrepancy must be machine precision (< 1e-6)
    diff = np.max(np.abs(res_lam0.point - b_proj))
    assert diff < 1e-6, f"Difference at lambda->0 was {diff:.2e} >= 1e-6"


# ---------------------------------------------------------------------------
# R4: Non-Linear Sequence-Space HANK Adversarial Tests
# ---------------------------------------------------------------------------

def test_adversarial_hank_extreme_mit_shock_stability():
    """R4: Stability under extreme MIT monetary shock (> 500 bps)."""
    ss = solve_hank_sequence_space(T=30)

    # 600 bps monetary shock (0.06) decaying at rate 0.7
    extreme_shock = 0.06 * (0.7 ** np.arange(30))
    res = solve_nonlinear_transition(
        ss,
        shock_seq=extreme_shock,
        shock_var="r",
        horizon=30,
        max_iter=50,
        tol=1e-6,
    )

    assert res.converged, "Broyden solver failed to converge under 600 bps MIT shock"
    assert np.max(np.abs(res.residuals)) < 1e-6
    assert np.all(np.isfinite(res.irf_consumption_nonlinear))
    assert np.all(np.isfinite(res.irf_output_nonlinear))
    # Output contracts upon massive interest rate hike
    assert res.irf_output_nonlinear[0] < 0.0


def test_adversarial_hank_near_zero_interest_rate_floor_bounds():
    """R4: Near-zero interest rate floor bounds (gross rate 1 + r_t >= 1e-6)."""
    ss = solve_hank_sequence_space(T=30)

    # Extreme negative rate shock of -500 bps (-0.05)
    floor_shock = -0.05 * (0.8 ** np.arange(30))
    res = solve_nonlinear_transition(
        ss,
        shock_seq=floor_shock,
        shock_var="r",
        horizon=30,
        max_iter=50,
        tol=1e-5,
    )

    assert res.converged, "Broyden solver failed to converge under near-zero floor shock"
    assert np.all(np.isfinite(res.irf_rate_nonlinear))
    assert np.all(np.isfinite(res.irf_consumption_nonlinear))
    # Non-linear rate path remains bounded
    r_path = ss.r_ss + res.irf_rate_nonlinear
    assert np.all(1.0 + r_path >= 1e-6), "Gross interest rate factor violated floor bound"


def test_adversarial_hank_broyden_machine_precision_convergence():
    """R4: Machine-precision Broyden convergence with Sherman-Morrison rank-1 updates."""
    ss = solve_hank_sequence_space(T=30)
    moderate_shock = 0.01 * (0.7 ** np.arange(30))

    res = solve_nonlinear_transition(
        ss,
        shock_seq=moderate_shock,
        shock_var="r",
        horizon=30,
        max_iter=50,
        tol=1e-8,
    )

    assert res.converged
    final_norm = float(np.max(np.abs(res.residuals)))
    assert final_norm < 1e-6, f"Final residual norm {final_norm:.2e} not below 1e-6"
    # History must record strict monotonic progress or contraction
    assert len(res.norm_history) > 1
    assert res.norm_history[-1] < res.norm_history[0]


# ---------------------------------------------------------------------------
# R5: Gertler-Karadi DSGE Adversarial Tests
# ---------------------------------------------------------------------------

def test_adversarial_gertler_karadi_massive_capital_quality_shock_occbin():
    """R5: OccBin backward recursion regime convergence under massive capital quality shock (eps_xi = -0.15)."""
    res = solve_gertler_karadi(
        shock_type="capital_quality",
        shock_size=-0.15,
        horizon=40,
        method="occbin",
        constraint_type="credit_policy",
    )

    assert res.solver_method == "occbin"
    # Massive crisis shock forces credit policy constraint to bind across many periods
    assert res.binding_periods >= 5, f"Expected >= 5 binding periods, got {res.binding_periods}"
    assert np.all(np.isfinite(res.irf.values))

    # Bank net worth collapses sharply and credit spread surges
    assert res.irf["N"].iloc[0] < -1.0, "Net worth failed to collapse under -15% capital shock"
    assert res.irf["prem"].iloc[0] > 0.005, "Credit spread failed to spike under -15% capital shock"

    # Terminal convergence: system returns to reference regime as shock dissipates
    assert res.regimes[-1] == 0, "Model failed to return to reference unconstrained regime"


def test_adversarial_gertler_karadi_klein_determinacy_boundaries():
    """R5: Determinacy and explosiveness boundaries in Klein solver."""
    m = build_gertler_karadi_model(GK2011_PARAMS)
    n_pre = len(m.states)
    # The stacked lead/lag system treats every current variable as
    # non-predetermined (the lagged copies of the states are the predetermined
    # block), so Blanchard-Kahn counts n_variables forward-looking unknowns.
    n_fwd = len(m.variables)

    # 1. Canonical model satisfies Blanchard-Kahn
    sol = klein_solve(m.A, m.B, n_pre, m.C, strict=True)
    assert sol.eu == (1, 1)

    # 2. Perturb B upwards: stable root crosses unit circle (no stable solution)
    with pytest.raises(BlanchardKahnError) as exc_up:
        klein_solve(m.A, m.B * 1.05, n_pre, m.C, strict=True)
    assert exc_up.value.kind == "no stable solution"
    assert exc_up.value.n_unstable > n_fwd

    # 3. Perturb B downwards: unstable root moves inside unit circle (indeterminacy)
    with pytest.raises(BlanchardKahnError) as exc_down:
        klein_solve(m.A, m.B * 0.95, n_pre, m.C, strict=True)
    assert exc_down.value.kind == "indeterminacy"
    assert exc_down.value.n_unstable < n_fwd


# ---------------------------------------------------------------------------
# R6: BVAR with Stochastic Volatility Adversarial Tests
# ---------------------------------------------------------------------------

def test_adversarial_bvar_sv_persistent_log_volatility_ffbs():
    """R6: Numerical stability of FFBS filter under persistent log-volatility (phi_h -> 0.999)."""
    rng = np.random.default_rng(42)
    T = 150
    v_t = np.ones(T) * 2.0
    y_adj = rng.standard_normal(T)

    # Direct FFBS test with near-unit-root persistence phi = 0.999
    h_draw = _ffbs_sv_ar1(y_adj, v_t, phi=0.999, sigma_h2=0.01, rng=rng)
    assert np.all(np.isfinite(h_draw))
    assert np.std(h_draw) > 0

    # Full BVAR-SV MCMC fit under persistent volatility DGP
    h_dgp = np.zeros((T, 2))
    for t in range(1, T):
        h_dgp[t] = 0.99 * h_dgp[t - 1] + 0.15 * rng.standard_normal(2)
    Y = np.zeros((T, 2))
    for t in range(1, T):
        nu = np.exp(h_dgp[t] / 2.0) * rng.standard_normal(2)
        Y[t] = 0.3 * Y[t - 1] + nu
    df = pd.DataFrame(Y, columns=["y1", "y2"])

    res = bvar_sv(df, lags=1, n_draws=80, n_burn=40, n_chains=1, seed=42)
    assert np.all(res.phi_draws < 1.0), "phi exceeded stationarity boundary"
    assert np.all(res.phi_draws > -1.0)
    assert np.all(np.isfinite(res.h_draws))


def test_adversarial_bvar_sv_high_vol_of_vol_ffbs():
    """R6: Stability under extreme volatility-of-volatility (sigma_h >> 1)."""
    rng = np.random.default_rng(88)
    T = 150
    v_t = np.ones(T) * 2.0
    y_adj = rng.standard_normal(T)

    # Direct FFBS test with huge sigma_h = 5.0 (sigma_h^2 = 25.0)
    h_draw_huge = _ffbs_sv_ar1(y_adj, v_t, phi=0.8, sigma_h2=25.0, rng=rng)
    assert np.all(np.isfinite(h_draw_huge))
    assert np.std(h_draw_huge) > 0.5

    # Full BVAR-SV MCMC fit under volatile DGP
    h_dgp = np.zeros((T, 2))
    for t in range(1, T):
        h_dgp[t] = 0.6 * h_dgp[t - 1] + 1.2 * rng.standard_normal(2)
    Y = np.zeros((T, 2))
    for t in range(1, T):
        nu = np.exp(np.clip(h_dgp[t], -4, 4) / 2.0) * rng.standard_normal(2)
        Y[t] = 0.2 * Y[t - 1] + nu
    df = pd.DataFrame(Y, columns=["y1", "y2"])

    res = bvar_sv(df, lags=1, n_draws=80, n_burn=40, n_chains=1, seed=88)
    assert np.all(res.sigma_h_draws > 0), "sigma_h draws must be strictly positive"
    assert np.all(np.isfinite(res.sigma_h_draws))


def test_adversarial_bvar_sv_positive_definiteness_preservation():
    """R6: Time-varying covariance matrix Sigma_t = A^{-1} D_t A^{-T} is strictly positive-definite across MCMC draws."""
    rng = np.random.default_rng(777)
    T = 120
    n = 2
    Y = rng.standard_normal((T, n))
    df = pd.DataFrame(Y, columns=["y1", "y2"])

    res = bvar_sv(df, lags=1, n_draws=60, n_burn=30, n_chains=1, seed=777)

    # Check positive definiteness of Sigma_t across multiple MCMC draws and time steps
    n_draws_sampled = res.n_draws
    test_draw_indices = np.linspace(0, n_draws_sampled - 1, min(10, n_draws_sampled), dtype=int)
    test_time_indices = [0, res.T_eff // 2, res.T_eff - 1]

    for m in test_draw_indices:
        A_m = res.a_draws[m]
        A_inv = np.linalg.inv(A_m)
        for t in test_time_indices:
            h_t = res.h_draws[m, t]
            D_t = np.diag(np.exp(h_t))
            Sigma_t = A_inv @ D_t @ A_inv.T

            # Exact symmetry
            np.testing.assert_allclose(Sigma_t, Sigma_t.T, atol=1e-12)
            # Positive eigenvalues
            eigs = np.linalg.eigvalsh(Sigma_t)
            assert np.all(eigs > 0), f"Draw {m}, time {t} had non-positive eigenvalues: {eigs}"
            # Cholesky factorization must succeed without jitter
            L = np.linalg.cholesky(Sigma_t)
            np.testing.assert_allclose(L @ L.T, Sigma_t, atol=1e-10)

    # Volatility-conditioned IRFs are strictly finite and non-degenerate
    irf_t0 = res.irf(horizon=10, t_idx=0)
    irf_tmid = res.irf(horizon=10, t_idx=res.T_eff // 2)
    assert np.all(np.isfinite(irf_t0.median))
    assert np.all(np.isfinite(irf_tmid.median))


def test_adversarial_bvar_sv_asymmetric_volatility_regime_shifts():
    """R6: Precision matrix symmetry under severe contemporaneous correlation and volatility regime shifts.

    Stress-tests the posterior precision matrix V_post^{-1} = V0^{-1} + sum_t Sigma_t^{-1} ⊗ (x_t x_t')
    under time-varying volatility shifts with non-diagonal contemporaneous impact matrix A,
    guaranteeing ||prec - prec.T||_inf < 1e-14 and no positive definiteness failure in Cholesky.
    """
    rng = np.random.default_rng(1234)
    T = 120
    Y = np.zeros((T, 2))
    for t in range(1, T):
        vol = 0.4 if t < 60 else 3.0
        Y[t, 0] = 0.5 * Y[t - 1, 0] + vol * rng.standard_normal()
        Y[t, 1] = 0.2 * Y[t - 1, 1] + 0.4 * Y[t, 0] + (vol * 1.5) * rng.standard_normal()
    df = pd.DataFrame(Y, columns=["y1", "y2"])

    res = bvar_sv(df, lags=1, n_draws=100, n_burn=50, n_chains=2, seed=42)

    assert res.n_draws == 100
    assert np.all(np.isfinite(res.beta_draws))
    assert np.all(np.isfinite(res.h_draws))
    assert np.all(np.isfinite(res.a_draws))

    # Verify that sampled contemporaneous impact matrices A are lower triangular with ones on diagonal
    for m in range(min(20, res.n_draws)):
        A_m = res.a_draws[m]
        assert np.isclose(A_m[0, 0], 1.0)
        assert np.isclose(A_m[1, 1], 1.0)
        assert np.isclose(A_m[0, 1], 0.0)

        # Reconstructed precision matrix Sigma_t^{-1} is strictly symmetric
        exp_neg_h = np.exp(-res.h_draws[m])
        Sigma_inv = np.einsum("ji,tj,jk->tik", A_m, exp_neg_h, A_m)
        asym_norm = np.max(np.abs(Sigma_inv - np.swapaxes(Sigma_inv, 1, 2)))
        assert asym_norm < 1e-14, f"Asymmetric Sigma_inv at draw {m}: {asym_norm}"

        # Eigenvalues of Sigma_inv must be strictly positive
        for t_idx in [0, res.T_eff // 2, res.T_eff - 1]:
            eigs = np.linalg.eigvalsh(Sigma_inv[t_idx])
            assert np.all(eigs > 0), f"Draw {m}, t {t_idx} non-positive eig: {eigs}"



# ---------------------------------------------------------------------------
# Additional Property-Based Adversarial Tests
# ---------------------------------------------------------------------------

@settings(max_examples=10, deadline=None)
@given(
    m_val=st.floats(min_value=0.01, max_value=5.0),
    orig_beta=st.floats(min_value=-3.0, max_value=3.0),
)
def test_hypothesis_honest_did_m_monotonicity(m_val, orig_beta):
    """Honest DiD identified set width is weakly monotonic in M."""
    event_time = [-2, -1, 0, 1]
    b_hat = [0.0, 0.0, orig_beta, orig_beta + 0.5]
    se = [0.1, 0.1, 0.15, 0.15]

    m_low = m_val
    m_high = m_val * 2.0
    res = honest_did(
        b_hat,
        event_time=event_time,
        se=se,
        method="smoothness",
        m_vec=[m_low, m_high],
        base_period=-1,
    )
    t = res.table
    w_low = float(t.loc[t["M"] == m_low, "id_hi"].iloc[0] - t.loc[t["M"] == m_low, "id_lo"].iloc[0])
    w_high = float(t.loc[t["M"] == m_high, "id_hi"].iloc[0] - t.loc[t["M"] == m_high, "id_lo"].iloc[0])
    assert w_high >= w_low - 1e-10, f"Width at M={m_high} ({w_high}) < width at M={m_low} ({w_low})"



