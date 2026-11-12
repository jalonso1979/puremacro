"""Numerical reference conformance and metamorphic tests for puremacro.lp."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from scipy.stats import chi2, norm

from puremacro.lp import lp_hac, lp_iv, panel_lp, panel_lp_iv


def test_lp_hac_with_controls_and_missing_values_matches_statsmodels() -> None:
    rng = np.random.default_rng(20261111)
    n = 140
    idx = pd.period_range("1985Q1", periods=n, freq="Q")
    x = rng.standard_normal(n)
    c1 = rng.standard_normal(n)
    c2 = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.6 * y[t - 1] + 0.5 * x[t] + 0.25 * x[t - 1] - 0.15 * c1[t] + 0.1 * c2[t - 1] + 0.4 * rng.standard_normal()

    df = pd.DataFrame({"y": y, "x": x, "c1": c1, "c2": c2}, index=idx)
    # Inject interior missing values while keeping regular PeriodIndex rows
    df.iloc[44, df.columns.get_loc("x")] = np.nan
    df.iloc[77, df.columns.get_loc("y")] = np.nan

    horizons = [0, 1, 3, 6]
    n_lags = 2
    alpha = 0.05
    pm = lp_hac(df, y="y", x="x", controls=["c1", "c2"], horizons=horizons, n_lags=n_lags, alpha=alpha)
    z_crit = float(norm.ppf(1.0 - alpha / 2.0))

    for h in horizons:
        sub = pd.DataFrame(index=df.index)
        sub["dy"] = df["y"].shift(-h) - df["y"].shift(1)
        sub["x"] = df["x"]
        rhs = ["x"]
        for lag in range(1, n_lags + 1):
            for col in ("x", "y", "c1", "c2"):
                name = f"{col}_L{lag}"
                sub[name] = df[col].shift(lag)
                rhs.append(name)
        for col in ("c1", "c2"):
            sub[col] = df[col]
            rhs.append(col)
        sub = sub[["dy", *rhs]].assign(y_t=df["y"]).dropna()
        X = np.column_stack([np.ones(len(sub)), sub[rhs].to_numpy()])
        ref = sm.OLS(sub["dy"].to_numpy(), X).fit(
            cov_type="HAC", cov_kwds={"maxlags": h + 1, "use_correction": False}
        )
        b_ref = float(ref.params[1])
        se_ref = float(ref.bse[1])
        assert float(pm.loc[h, "beta"]) == pytest.approx(b_ref, rel=1e-12, abs=1e-13)
        assert float(pm.loc[h, "se"]) == pytest.approx(se_ref, rel=1e-12, abs=1e-13)
        assert float(pm.loc[h, "lo"]) == pytest.approx(b_ref - z_crit * se_ref, rel=1e-12, abs=1e-13)
        assert float(pm.loc[h, "hi"]) == pytest.approx(b_ref + z_crit * se_ref, rel=1e-12, abs=1e-13)
        assert pm.attrs["specification"]["n_obs_by_horizon"][h] == len(sub)


def test_lp_iv_single_and_overidentified_match_linearmodels_and_ar_inversion() -> None:
    rng = np.random.default_rng(9090)
    n = 180
    idx = pd.period_range("1975Q1", periods=n, freq="Q")
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    w = rng.standard_normal(n)
    u = rng.standard_normal(n) * 0.6
    x = 0.7 * z1 - 0.45 * z2 + 0.25 * w + u
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + 0.8 * x[t] + 0.2 * x[t - 1] - 0.3 * w[t] + 0.5 * u[t] + 0.4 * rng.standard_normal()
    df = pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2, "w": w}, index=idx)

    horizons = [0, 1, 2, 4]
    n_lags = 2
    alpha = 0.10
    pm = lp_iv(
        df,
        y="y",
        x="x",
        z=["z1", "z2"],
        controls=["w"],
        horizons=horizons,
        n_lags=n_lags,
        alpha=alpha,
        anderson_rubin=True,
    )

    crit_ar = float(chi2.ppf(1.0 - alpha, df=2))
    for h in horizons:
        sub = pd.DataFrame(index=df.index)
        sub["dy"] = df["y"].shift(-h) - df["y"].shift(1)
        sub["x"] = df["x"]
        sub["z1"] = df["z1"]
        sub["z2"] = df["z2"]
        exog_cols = ["const"]
        sub["const"] = 1.0
        for lag in range(1, n_lags + 1):
            for col in ("x", "y", "w"):
                cname = f"{col}_L{lag}"
                sub[cname] = df[col].shift(lag)
                exog_cols.append(cname)
        sub["w"] = df["w"]
        exog_cols.append("w")
        s = sub[["dy", "x", "z1", "z2", *exog_cols]].dropna()

        ref = IV2SLS(
            dependent=s["dy"],
            exog=s[exog_cols],
            endog=s[["x"]],
            instruments=s[["z1", "z2"]],
        ).fit(cov_type="kernel", kernel="bartlett", bandwidth=h + 1, debiased=False)

        assert float(pm.loc[h, "beta"]) == pytest.approx(float(ref.params["x"]), rel=1e-11, abs=1e-12)
        assert float(pm.loc[h, "se"]) == pytest.approx(float(ref.std_errors["x"]), rel=1e-11, abs=1e-12)
        assert pm.loc[h, "ar_set_type"] == "bounded"

        # Verify exact AR boundary root equation AR(beta_bound) == chi2_{K, 1-alpha}
        # via independent statsmodels Newey-West Wald statistic on [z1, z2]
        rhs_ar = np.column_stack([s[["z1", "z2"]].to_numpy(), s[exog_cols].to_numpy()])
        for bound_col in ("ar_lo", "ar_hi"):
            b0 = float(pm.loc[h, bound_col])
            u0 = s["dy"].to_numpy() - b0 * s["x"].to_numpy()
            ar_fit = sm.OLS(u0, rhs_ar).fit(
                cov_type="HAC", cov_kwds={"maxlags": h + 1, "use_correction": False}
            )
            delta_z = np.asarray(ar_fit.params[:2], dtype=float)
            cov_zz = np.asarray(ar_fit.cov_params()[:2, :2], dtype=float)
            ar_wald = float(delta_z @ np.linalg.solve(cov_zz, delta_z))
            assert ar_wald == pytest.approx(crit_ar, rel=1e-8, abs=1e-8)

    # Pure noise instrument triggers weak-IV unbounded_rays / all_real and summary warning
    df_weak = df.copy()
    df_weak["z_noise"] = rng.standard_normal(n) * 1e-3
    weak_res = lp_iv(
        df_weak, y="y", x="x", z="z_noise", horizons=[0, 1], n_lags=1, anderson_rubin=True
    )
    assert (weak_res["mop_f"] < weak_res["mop_cv_10"]).all()
    assert set(weak_res["ar_set_type"]) <= {"unbounded_rays", "all_real"}
    assert "Weak-IV warning:" in weak_res.summary()


def test_panel_lp_iv_balanced_and_unbalanced_match_linearmodels_iv2sls() -> None:
    rng = np.random.default_rng(5150)
    n_units, n_periods = 9, 45
    codes = [f"U{i}" for i in range(n_units)]
    dates = pd.date_range("2002-01-01", periods=n_periods, freq="QS")
    fe_u = rng.standard_normal(n_units) * 0.6
    fe_t = rng.standard_normal(n_periods) * 0.35

    rows = []
    for i, code in enumerate(codes):
        y_prev = x_prev = 0.0
        for t in range(n_periods):
            z1 = rng.standard_normal()
            z2 = rng.standard_normal()
            w = rng.standard_normal()
            nu = rng.standard_normal() * 0.55
            x = 0.75 * z1 - 0.35 * z2 + 0.25 * x_prev + 0.2 * w + nu
            y = (
                fe_u[i]
                + fe_t[t]
                + 0.45 * y_prev
                + 0.7 * x
                + 0.15 * x_prev
                - 0.2 * w
                + 0.45 * nu
                + 0.35 * rng.standard_normal()
            )
            rows.append({"code": code, "date": dates[t], "y": y, "x": x, "z1": z1, "z2": z2, "w": w})
            y_prev, x_prev = y, x

    df_pan = pd.DataFrame(rows).set_index(["code", "date"]).sort_index()
    # Make unbalanced via interior missing values in one unit
    df_unbal = df_pan.copy()
    df_unbal.iloc[12:15, df_unbal.columns.get_loc("y")] = np.nan

    for panel_data in (df_pan, df_unbal):
        pm = panel_lp_iv(
            panel_data,
            y="y",
            x="x",
            z=["z1", "z2"],
            controls=["w"],
            horizons=[0, 1, 3],
            n_lags=2,
            alpha=0.05,
        )
        work = panel_data.copy()
        grp = work.groupby(level="code")
        exog_names: list[str] = []
        for lag in (1, 2):
            for col in ("x", "y", "w"):
                cname = f"{col}_L{lag}"
                work[cname] = grp[col].shift(lag)
                exog_names.append(cname)
        exog_names.append("w")

        for h in (0, 1, 3):
            work_h = work.copy()
            work_h["dy"] = grp["y"].shift(-h) - grp["y"].shift(1)
            s = work_h[["dy", "x", "z1", "z2", *exog_names]].dropna()
            ent = s.index.get_level_values("code")
            tim = s.index.get_level_values("date")
            fe_cols = pd.concat(
                [
                    pd.get_dummies(ent, prefix="e", drop_first=False, dtype=float).set_index(s.index),
                    pd.get_dummies(tim, prefix="t", drop_first=True, dtype=float).set_index(s.index),
                ],
                axis=1,
            )
            ref = IV2SLS(
                dependent=s["dy"],
                exog=pd.concat([s[exog_names], fe_cols], axis=1),
                endog=s[["x"]],
                instruments=s[["z1", "z2"]],
            ).fit(cov_type="clustered", clusters=pd.Series(ent, index=s.index), debiased=False)

            assert float(pm.loc[h, "beta"]) == pytest.approx(float(ref.params["x"]), rel=1e-9, abs=1e-10)
            assert float(pm.loc[h, "se"]) == pytest.approx(float(ref.std_errors["x"]), rel=1e-9, abs=1e-10)

    # Collapse check: panel_lp_iv(z=x) == panel_lp
    ols_pan = panel_lp(df_unbal, y="y", x="x", controls=["w"], horizons=[0, 1, 3], n_lags=2)
    iv_self = panel_lp_iv(df_unbal, y="y", x="x", z="x", controls=["w"], horizons=[0, 1, 3], n_lags=2)
    np.testing.assert_allclose(iv_self["beta"], ols_pan["beta"], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(iv_self["se"], ols_pan["se"], rtol=1e-10, atol=1e-11)


def test_lp_metamorphic_invariances_and_index_representation_parity() -> None:
    rng = np.random.default_rng(314159)
    n = 120
    z = rng.standard_normal(n)
    u = rng.standard_normal(n) * 0.4
    x = 0.85 * z + u
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + 0.6 * x[t] + 0.3 * u[t] + 0.3 * rng.standard_normal()

    p_idx = pd.period_range("1990Q1", periods=n, freq="Q")
    d_idx = pd.date_range("1990-01-01", periods=n, freq="QS")
    df_p = pd.DataFrame({"y": y, "x": x, "z": z}, index=p_idx)
    df_d = pd.DataFrame({"y": y, "x": x, "z": z}, index=d_idx)

    # 1. DatetimeIndex vs PeriodIndex representation equivalence
    fit_p = lp_iv(df_p, y="y", x="x", z="z", horizons=[0, 1, 3], n_lags=2, anderson_rubin=True)
    fit_d = lp_iv(df_d, y="y", x="x", z="z", horizons=[0, 1, 3], n_lags=2, anderson_rubin=True)
    for col in ("beta", "se", "t", "lo", "hi", "first_stage_f", "mop_f", "ar_lo", "ar_hi"):
        np.testing.assert_allclose(fit_p[col], fit_d[col], rtol=1e-13, atol=1e-14)

    # 2. Nonzero instrument rescaling invariance: z -> -7.3 * z
    df_zscale = df_p.copy()
    df_zscale["z"] = -7.3 * df_zscale["z"]
    fit_zscale = lp_iv(df_zscale, y="y", x="x", z="z", horizons=[0, 1, 3], n_lags=2, anderson_rubin=True)
    for col in ("beta", "se", "t", "lo", "hi", "first_stage_f", "mop_f", "ar_lo", "ar_hi"):
        np.testing.assert_allclose(fit_zscale[col], fit_p[col], rtol=1e-12, atol=1e-12)

    # 3. Outcome scaling: y -> 3.5 * y scales beta, se, lo, hi by 3.5 and leaves t, mop_f invariant
    c_y = 3.5
    df_yscale = df_p.copy()
    df_yscale["y"] = c_y * df_yscale["y"]
    fit_yscale = lp_iv(df_yscale, y="y", x="x", z="z", horizons=[0, 1, 3], n_lags=2)
    np.testing.assert_allclose(fit_yscale["beta"], c_y * fit_p["beta"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fit_yscale["se"], c_y * fit_p["se"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fit_yscale["t"], fit_p["t"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fit_yscale["mop_f"], fit_p["mop_f"], rtol=1e-12, atol=1e-12)

    # 4. Shock scaling: x -> -2.5 * x scales beta by -0.4, se by 0.4, flips t, leaves mop_f invariant
    c_x = -2.5
    df_xscale = df_p.copy()
    df_xscale["x"] = c_x * df_xscale["x"]
    fit_xscale = lp_iv(df_xscale, y="y", x="x", z="z", horizons=[0, 1, 3], n_lags=2)
    np.testing.assert_allclose(fit_xscale["beta"], fit_p["beta"] / c_x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fit_xscale["se"], fit_p["se"] / abs(c_x), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fit_xscale["t"], -fit_p["t"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fit_xscale["mop_f"], fit_p["mop_f"], rtol=1e-12, atol=1e-12)

    # 5. Zero-response deterministic outcome independent of shock -> beta == 0
    df_zero = pd.DataFrame(
        {"y": 2.0 + 0.5 * np.arange(n, dtype=float), "x": x, "z": z},
        index=p_idx,
    )
    fit_zero_ols = lp_hac(df_zero, y="y", x="x", horizons=[0, 1, 4], n_lags=0)
    fit_zero_iv = lp_iv(df_zero, y="y", x="x", z="z", horizons=[0, 1, 4], n_lags=0)
    np.testing.assert_allclose(fit_zero_ols["beta"], 0.0, atol=1e-13)
    np.testing.assert_allclose(fit_zero_iv["beta"], 0.0, atol=1e-13)
