"""Tests for compare_irfs, IRFComparisonResult, and lp_sensitivity_grid (PR 1)."""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge._results import BayesianIRFResult
from puremacro.inference._results import SupTBandResult
from puremacro.lp import (
    la_lp,
    lp_did,
    lp_hac,
    lp_iv,
    lp_quantile,
    lp_sensitivity_grid,
    lp_state_dep,
    panel_lp_dk,
    smooth_lp,
)
from puremacro.reports import (
    IRFComparisonResult,
    IRFResult as ReportsIRFResult,
    LPResult as ReportsLPResult,
    compare_irfs,
)
from puremacro.var import bootstrap_bands, fit_var, svar_proxy
from puremacro.var.identify._results import ProxySVARResult


def _make_ts_df(n: int = 120, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-01-01", periods=n, freq="QE")
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.5 * y[t - 1] + 0.8 * x[t] + 0.4 * x[t - 1] + 0.3 * z[t] + 0.4 * rng.standard_normal()
    return pd.DataFrame({"y": y, "x": x, "z": z}, index=idx)


def _make_panel_did_df(n_units: int = 12, n_periods: int = 16, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        adopt = 6 + (u % 3) * 2 if u < 8 else 999
        fe_u = 0.2 * u
        for t in range(n_periods):
            d = 1 if t >= adopt else 0
            effect = 1.2 if (t >= adopt and t - adopt <= 4) else 0.0
            rows.append(
                {
                    "unit": f"U{u:02d}",
                    "time": t,
                    "y": fe_u + 0.05 * t + effect + 0.15 * rng.standard_normal(),
                    "D": d,
                }
            )
    return pd.DataFrame(rows)


def test_compare_irfs_mixed_lp_and_band_compat():
    df = _make_ts_df()
    r_hac = lp_hac(df, "y", "x", horizons=range(5), lags=2, ci=0.90)
    r_la = la_lp(df, "y", "x", horizons=range(5), lags=2, ci=0.90)
    r_smooth = smooth_lp(df, "y", "x", horizons=range(5), lags=2)  # default alpha=0.05 -> 95%

    # Default band_compat="warn" handles 90% vs 95% with explicit warning & columns
    with pytest.warns(UserWarning, match="Confidence bands differ"):
        cmp = compare_irfs({"LP-HAC": r_hac, "LA-LP": r_la, "Smooth-LP": r_smooth})

    assert isinstance(cmp, IRFComparisonResult)
    assert cmp.baseline == "LP-HAC"
    assert cmp.horizons == (0, 1, 2, 3, 4)
    assert set(["spec", "h", "point", "se", "lower", "upper", "diff_vs_baseline", "ci_level", "band_kind"]).issubset(
        cmp.table.columns
    )
    assert len(cmp.metrics) == 3

    with pytest.raises(ValueError, match="Confidence bands differ"):
        compare_irfs({"LP-HAC": r_hac, "Smooth-LP": r_smooth}, band_compat="raise")

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cmp_ig = compare_irfs({"LP-HAC": r_hac, "Smooth-LP": r_smooth}, band_compat="ignore")
    assert not any("Confidence bands differ" in str(w.message) for w in rec)
    assert cmp_ig.baseline == "LP-HAC"


def test_compare_irfs_horizon_alignment_exact_and_intersection():
    df = _make_ts_df()
    r4 = lp_hac(df, "y", "x", horizons=range(5), lags=2)
    r7 = lp_hac(df, "y", "x", horizons=range(8), lags=2)

    with pytest.raises(ValueError, match="intersection"):
        compare_irfs({"Short": r4, "Long": r7}, horizon_align="exact")

    with pytest.warns(UserWarning, match="intersection"):
        aligned = compare_irfs({"Short": r4, "Long": r7}, horizon_align="intersection")
    assert aligned.horizons == (0, 1, 2, 3, 4)
    assert len(aligned.table) == 10


def test_compare_irfs_cumulative_and_vcov_guards():
    df = _make_ts_df()
    r_hac = lp_hac(df, "y", "x", horizons=range(5), lags=2)
    r_smooth = smooth_lp(df, "y", "x", horizons=range(5), lags=2, ci=0.90)
    r_hac_cum = lp_iv(df, "y", "x", "z", horizons=range(5), lags=2, cumulative=True)
    r_gap = lp_hac(df, "y", "x", horizons=[0, 1, 3, 4], lags=2)

    with pytest.warns(UserWarning, match="cross-horizon vcov"):
        cum_cmp = compare_irfs({"Smooth": r_smooth, "HAC": r_hac}, cumulative=True)

    sm_rows = cum_cmp.table[cum_cmp.table["spec"] == "Smooth"].sort_values("h")
    hac_rows = cum_cmp.table[cum_cmp.table["spec"] == "HAC"].sort_values("h")
    # SmoothLPResult has cross-horizon vcov -> finite cumulative SEs & exact cumsum
    assert np.all(np.isfinite(sm_rows["se"]))
    assert np.all(np.isfinite(sm_rows["lower"]))
    assert np.allclose(sm_rows["point"].to_numpy(), np.cumsum(r_smooth["beta"].to_numpy()))
    # Plain lp_hac lacks cross-horizon vcov -> omitted SEs/intervals (NaN)
    assert np.all(np.isnan(hac_rows["se"]))
    assert np.all(np.isnan(hac_rows["lower"]))

    # Verify SmoothLPResult cumulative SE formula: sqrt(diag(L @ vcov @ L.T))
    H = len(sm_rows)
    tril = np.tril(np.ones((H, H)))
    expected_se = np.sqrt(np.diag(tril @ r_smooth.vcov @ tril.T))
    assert np.allclose(sm_rows["se"].to_numpy(), expected_se)

    # Double cumulation and mixed cumulative guards
    with pytest.raises(ValueError, match="already cumulative"):
        compare_irfs({"CumHAC": r_hac_cum, "HAC": r_hac}, cumulative=True)
    with pytest.raises(ValueError, match="allow_mixed_cumulative"):
        compare_irfs({"CumHAC": r_hac_cum, "HAC": r_hac}, cumulative=False)
    mix_ok = compare_irfs({"CumHAC": r_hac_cum, "HAC": r_hac}, allow_mixed_cumulative=True)
    assert len(mix_ok.metrics) == 2

    # Non-consecutive horizons rejected under cumulative=True
    with pytest.raises(ValueError, match="consecutive unit-step horizons"):
        compare_irfs({"Gap": r_gap}, cumulative=True)


def test_compare_irfs_state_dep_and_quantile_selectors():
    df = _make_ts_df()
    r_hac = lp_hac(df, "y", "x", horizons=range(4), lags=2)
    r_sd = lp_state_dep(df, "y", "x", state="z", horizons=range(4), lags=2)
    r_q = lp_quantile(df, "y", "x", quantiles=(0.25, 0.50, 0.75), horizons=range(4), lags=2, n_boot=25)

    # Mixed single-coef + per-spec state/quantile selectors ignore selectors on lp_hac
    cmp = compare_irfs(
        {"HAC": r_hac, "StateDep": r_sd, "Quantile": r_q},
        regime={"StateDep": "H"},
        quantile={"Quantile": 0.25},
        band_compat="ignore",
    )
    assert set(cmp.table["spec"]) == {"HAC", "StateDep", "Quantile"}
    q_sub = cmp.table[cmp.table["spec"] == "Quantile"].sort_values("h")
    expected_q = r_q[np.isclose(r_q["tau"], 0.25)].sort_values("h")["beta"].to_numpy()
    assert np.allclose(q_sub["point"].to_numpy(), expected_q)
    assert np.all(np.isnan(q_sub["se"]))  # quantile LP has no analytical SE col

    # expand_groups=True expands regimes and quantiles cleanly
    expanded = compare_irfs({"StateDep": r_sd, "Quantile": r_q}, expand_groups=True, band_compat="ignore")
    specs = set(expanded.table["spec"])
    assert {"StateDep [H]", "StateDep [L]", "Quantile [tau=0.25]", "Quantile [tau=0.50]", "Quantile [tau=0.75]"} == specs


def test_compare_irfs_var_svar_supt_bayesian_and_report_containers():
    df = _make_ts_df(n=100)
    var_res = fit_var(df[["y", "x"]], lags=2)
    svar_res = svar_proxy(df[["y", "x"]], df["z"].to_numpy(), p=2, horizon=4, n_boot=30, alpha=0.10, seed=1)
    boot_supt = bootstrap_bands(
        df[["y", "x"]].to_numpy(),
        p=2,
        identify_fn=lambda A, S: np.linalg.cholesky(S),
        horizon=4,
        n_boot=30,
        band="sup-t",
        rng=np.random.default_rng(1),
    )

    supt_1d = SupTBandResult(
        lower=svar_res.irf[:, 0, 1] - 0.3,
        upper=svar_res.irf[:, 0, 1] + 0.3,
        center=svar_res.irf[:, 0, 1],
        scale=np.full(5, 0.1),
        crit_value=2.45,
        method="sup-t",
        alpha=0.10,
        n_draws=100,
    )
    rep_lp = ReportsLPResult(
        df=pd.DataFrame({"h": range(5), "beta": np.linspace(0.5, 0.1, 5), "se": np.full(5, 0.1)}),
        method="Custom-LP",
        n_obs=95,
        alpha=0.10,
    )
    rep_irf = ReportsIRFResult(
        point=np.linspace(0.4, 0.0, 5),
        lower=np.linspace(0.2, -0.2, 5),
        upper=np.linspace(0.6, 0.2, 5),
        var_names=["y"],
        method="Reports-IRF",
        alpha=0.10,
    )
    proxy_res = ProxySVARResult(
        irf_point=np.broadcast_to(np.linspace(1.0, 0.2, 5)[:, None, None], (5, 2, 2)).copy(),
        irf_lower=np.broadcast_to((np.linspace(1.0, 0.2, 5) - 0.2)[:, None, None], (5, 2, 2)).copy(),
        irf_upper=np.broadcast_to((np.linspace(1.0, 0.2, 5) + 0.2)[:, None, None], (5, 2, 2)).copy(),
        B=np.eye(2),
        first_stage_F=18.0,
        n_boot=20,
        ci=0.90,
    )

    h_idx = pd.Index(range(5), name="h")
    med_b = pd.DataFrame({"y": [0.5, 0.4, 0.3, 0.2, 0.1], "x": [1.0, 0.5, 0.2, 0.1, 0.0]}, index=h_idx)
    bayes = BayesianIRFResult(
        median={"eps_x": med_b},
        bands={
            "eps_x": {
                0.68: (med_b - 0.1, med_b + 0.1),
                0.90: (med_b - 0.2, med_b + 0.2),
            }
        },
        quantiles={},
        draws=None,
        variables=("y", "x"),
        shocks=("eps_x",),
        horizon=4,
        n_draws=10,
        determinacy_rate=1.0,
    )

    cmp = compare_irfs(
        {
            "VAR": var_res,
            "SVAR": svar_res,
            "BootSupT": boot_supt,
            "SupT": supt_1d,
            "RepLP": rep_lp,
            "RepIRF": rep_irf,
            "Proxy": proxy_res,
            "Bayes": bayes,
        },
        horizon_align="intersection",
        target={"VAR": "y", "SVAR": "y", "BootSupT": 0, "Proxy": 0, "Bayes": "y"},
        shock={"VAR": "x", "SVAR": "x", "BootSupT": 1, "Proxy": 1, "Bayes": "eps_x"},
        ci=0.68,
        band_compat="ignore",
    )
    assert len(cmp.metrics) == 8
    b_row = cmp.metrics.loc[cmp.metrics["spec"] == "Bayes"].iloc[0]
    assert np.isclose(b_row["ci_level"], 0.68)
    assert b_row["band_kind"] == "credible"
    supt_row = cmp.metrics.loc[cmp.metrics["spec"] == "BootSupT"].iloc[0]
    assert supt_row["band_kind"] == "sup-t"

    # ci= without any multi-band object warns transparently
    r_hac = lp_hac(df, "y", "x", horizons=range(5), lags=2)
    with pytest.warns(UserWarning, match="credible bands"):
        compare_irfs({"A": r_hac, "B": rep_irf}, ci=0.68, band_compat="ignore")


def test_compare_irfs_exports_and_plot_returns_ax():
    df = _make_ts_df()
    r1 = lp_hac(df, "y", "x", horizons=range(4), lags=1)
    r2 = lp_hac(df, "y", "x", horizons=range(4), lags=2)
    cmp = compare_irfs({"L1": r1, "L2": r2})

    wide = cmp.to_frame("wide")
    assert list(wide.columns) == [
        "h",
        "L1_point", "L1_se", "L1_lower", "L1_upper",
        "L2_point", "L2_se", "L2_lower", "L2_upper",
    ]
    coef_se = cmp.to_manuscript_dataframe()
    assert len(coef_se) == 8  # 2 rows per horizon
    assert list(cmp.to_frame("metrics")["spec"]) == ["L1", "L2"]

    md = cmp.to_markdown()
    ltx = cmp.to_latex()
    typ = cmp.to_typst()
    assert "|" in md and "L1" in md
    assert "\\begin{tabular}" in ltx
    assert "#table(" in typ
    assert "IRF Comparison" in cmp.summary()

    try:
        fig = cmp.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        fig2, ax = plt.subplots()
        out_ax = cmp.plot(ax=ax)
        assert out_ax is ax
    finally:
        plt.close("all")


def test_lp_sensitivity_grid_multi_estimator_lp_did_and_skip():
    df = _make_ts_df()
    cmp = lp_sensitivity_grid(
        df,
        "y",
        "x",
        grid={"estimator": [lp_hac, la_lp, smooth_lp], "lags": [1, 3]},
        baseline_spec={"estimator": lp_hac, "lags": 1},
        base_kwargs={"horizons": range(5), "horizon": None},
    )
    assert len(cmp.metrics) == 6
    assert cmp.baseline == "lp_hac | lags=1"
    assert "resolved_params" in cmp.metrics.columns

    # Override alias conflict cleanly (base_kwargs has horizons & n_lags, grid has horizon & lags)
    cmp_alias = lp_sensitivity_grid(
        df,
        "y",
        "x",
        grid={"lags": [1, 2], "horizon": [3]},
        base_kwargs={"n_lags": 2, "horizons": range(6)},
    )
    assert cmp_alias.horizons == (0, 1, 2, 3)
    assert len(cmp_alias.metrics) == 2

    # Custom callable without ci/lags/horizon aliases exercises inspect.signature conversion
    seen_calls: list[dict[str, object]] = []

    def strict_custom_lp(data, y, x, *, horizons=(0, 1, 2), n_lags=1, alpha=0.10):
        seen_calls.append({"horizons": list(horizons), "n_lags": n_lags, "alpha": alpha})
        return lp_hac(data, y, x, horizons=horizons, n_lags=n_lags, alpha=alpha)

    cmp_custom = lp_sensitivity_grid(
        df,
        "y",
        "x",
        estimator=strict_custom_lp,
        grid={"lags": [1, 2], "horizon": [2], "ci": [0.90]},
    )
    assert len(cmp_custom.metrics) == 2
    assert np.isclose(seen_calls[0]["alpha"], 0.10)  # type: ignore[arg-type]
    assert seen_calls[0]["n_lags"] == 1
    assert seen_calls[0]["horizons"] == [0, 1, 2]

    # lp_did integration (x -> treat, preserves lags= and ci= via base_kwargs unit/time)
    pan = _make_panel_did_df()
    did_cmp = lp_sensitivity_grid(
        pan,
        "y",
        "D",
        estimator=lp_did,
        grid={"lags": [0, 1], "ci": [0.90, 0.95]},
        base_kwargs={"unit": "unit", "time": "time", "horizons": range(4), "pre_window": 2},
    )
    assert len(did_cmp.metrics) == 4

    # panel_lp_dk also works alongside lp_did in compare_irfs cleanly
    dk = panel_lp_dk(pan, "y", "D", horizons=range(4), unit_col="unit", time_col="time", lags=0)
    did_pre0 = lp_did(pan, "y", "D", "unit", "time", horizons=range(4), pre_window=0)
    pan_cmp = compare_irfs({"DK": dk, "LP-DiD": did_pre0})
    assert len(pan_cmp.metrics) == 2
    did_pre2 = lp_did(pan, "y", "D", "unit", "time", horizons=range(4), pre_window=2)
    with pytest.warns(UserWarning, match="intersection"):
        pan_cmp_inter = compare_irfs({"DK": dk, "LP-DiD": did_pre2}, horizon_align="intersection")
    assert pan_cmp_inter.horizons == (0, 1, 2, 3)

    # on_error="skip" keeps surviving specs and fails cleanly if <2 survive
    def flaky_est(data, y, x, *, lags=1, horizons=range(4)):
        if lags == 2:
            raise ValueError("synthetic failure for lag=2")
        return lp_hac(data, y, x, horizons=horizons, lags=lags)

    cmp_skip = lp_sensitivity_grid(
        df,
        "y",
        "x",
        estimator=flaky_est,
        grid={"lags": [1, 2, 3]},
        base_kwargs={"horizons": range(4)},
        on_error="skip",
    )
    assert len(cmp_skip.metrics) == 2
    assert "lags=2" in cmp_skip.errors

    with pytest.raises(RuntimeError, match="at least 2"):
        lp_sensitivity_grid(
            df,
            "y",
            "x",
            estimator=flaky_est,
            grid={"lags": [1, 2]},
            base_kwargs={"horizons": range(4)},
            on_error="skip",
        )
