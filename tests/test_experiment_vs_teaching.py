"""Structural parity: puremacro.experiment vs src.teaching.experiment.

This is a *structural* parity test, not a numerical one — bootstrap RNGs
differ and LP specifications differ slightly between the two stacks.
We assert: same result-dict shape, identical Cholesky impact at h=0
(both use the same OLS+Cholesky math), and signs/correlations agree
for LP across horizons.
"""
import numpy as np
import pandas as pd
import pytest

# Skip cleanly if the teaching stack's deps aren't available.
pytest.importorskip("statsmodels")
pytest.importorskip("linearmodels")

try:
    from src.teaching.experiment import run_experiment as teaching_run
except Exception:  # noqa: BLE001
    pytest.skip("src.teaching.experiment not importable", allow_module_level=True)

from puremacro.experiment import run_experiment as pm_run


def _toy_panel(seed=0, T=80, codes=("USA", "MEX")):
    rng = np.random.default_rng(seed)
    rows = []
    for code in codes:
        eps_x = rng.standard_normal(T)
        eps_y = rng.standard_normal(T)
        x = np.zeros(T)
        y = np.zeros(T)
        for t in range(1, T):
            x[t] = 0.6 * x[t - 1] + eps_x[t]
            y[t] = 0.5 * y[t - 1] + 0.4 * x[t - 1] + eps_y[t]
        idx = pd.MultiIndex.from_product(
            [[code], pd.date_range("2000-01-01", periods=T, freq="QE")],
            names=["code", "date"],
        )
        rows.append(pd.DataFrame({"x": x, "y": y}, index=idx))
    return pd.concat(rows).sort_index()


@pytest.fixture
def wide():
    return _toy_panel()


def test_var_cholesky_impact_parity(wide):
    horizons = range(0, 9)
    pm = pm_run(wide, outcome="y", proxy="x", method="var_cholesky",
                country="USA", horizons=horizons, n_lags=2,
                n_boot=20, seed=0, normalize="raw")
    th = teaching_run(wide, outcome="y", proxy="x", method="var_cholesky",
                     country="USA", horizons=horizons, n_lags=2,
                     n_boot=20, seed=0, normalize="raw")

    # Same shape / keys
    assert pm["method"] == th["method"]
    assert pm["n_obs"] == th["n_obs"]
    assert pm["alpha"] == th["alpha"]
    assert set(pm.keys()) >= {"method", "irf_point", "irf_lower",
                               "irf_upper", "n_obs", "alpha"}

    # Cholesky impact at h=0: identical math on both sides.
    pm_h0 = pm["irf_point"].iloc[0]
    th_h0 = th["irf_point"].iloc[0]
    common = sorted(set(pm_h0.index) & set(th_h0.index))
    assert common  # at least "x", "y"
    np.testing.assert_allclose(
        pm_h0.loc[common].values, th_h0.loc[common].values,
        atol=1e-6,
    )


def test_lp_structural_parity(wide):
    # Structural-only check: the puremacro lp_hac uses the
    # Plagborg-Møller-Wolf (cumulative-vs-level) specification while
    # src.teaching.lp_sm.lp_ols_hac may use a different spec. We expect
    # the same sign at h=1 and a high correlation across horizons.
    horizons = range(0, 9)
    pm = pm_run(wide, outcome="y", proxy="x", method="lp",
                country="USA", horizons=horizons, n_lags=2,
                normalize="raw")
    th = teaching_run(wide, outcome="y", proxy="x", method="lp",
                     country="USA", horizons=horizons, n_lags=2,
                     normalize="raw")

    assert pm["method"] == th["method"]
    assert pm["alpha"] == th["alpha"]

    common = sorted(set(pm["irf_point"].index) & set(th["irf_point"].index))
    pm_vals = pm["irf_point"].loc[common].values
    th_vals = th["irf_point"].loc[common].values

    # Sign at h=1 should agree (positive feedthrough x -> y by DGP).
    if 1 in common:
        assert np.sign(pm_vals[common.index(1)]) == np.sign(
            th_vals[common.index(1)]
        )

    # Pearson correlation across horizons > 0.9.
    corr = np.corrcoef(pm_vals, th_vals)[0, 1]
    assert corr > 0.9
