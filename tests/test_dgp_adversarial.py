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


