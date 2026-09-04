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
