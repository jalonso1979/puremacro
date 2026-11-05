"""Follow-up regression tests for GitHub Issue #217 (weak_iv HAC/diagnostics & state_dep)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.examples.ramey_zubairy_2018_multipliers import RZ_CONTROLS, load_rz_data
from puremacro.inference import instrument_influence
from puremacro.inference.weak_iv import (
    anderson_rubin_test,
    msw_bands,
    olea_pflueger_f,
)
from puremacro.lp import lp_garch_state, lp_iv, lp_state_dep, lp_state_dep_iv
from puremacro.lp.state_dep import lp_smooth_transition_irf


def test_olea_pflueger_f_hac_matches_lp_iv_mop_f():
    """olea_pflueger_f(..., lags=h+1, controls=W) matches lp_iv's mop_f on RZ 2018."""
    df = load_rz_data()
    h = 7
    res_lp = lp_iv(
        df, y="y", x="g", z="newsy", horizons=[h], n_lags=0,
        controls=RZ_CONTROLS, alpha=0.10, cumulative=True,
    )
    mop_f_expected = float(res_lp.loc[h, "mop_f"])

    # Build the exact h=7 cumulative sample
    sub = df[["y", "g", "newsy"] + RZ_CONTROLS].copy()
    sub["__dy__"] = sum(sub["y"].shift(-j) - sub["y"].shift(1) for j in range(h + 1))
    sub["__dx__"] = sum(sub["g"].shift(-j) for j in range(h + 1))
    sub = sub.dropna()

    x = sub["__dx__"].to_numpy(dtype=float)
    z = sub["newsy"].to_numpy(dtype=float)
    W_no_const = sub[RZ_CONTROLS].to_numpy(dtype=float)

    # 1. Passing controls= directly to olea_pflueger_f (prepends constant internally)
    f_ctl = olea_pflueger_f(x, z, controls=W_no_const, lags=h + 1)
    assert np.isclose(f_ctl, mop_f_expected, rtol=1e-10, atol=1e-10)

    # 2. Pre-residualized x and z with lags=h+1
    W = np.column_stack([np.ones(len(sub)), W_no_const])
    Q, _ = np.linalg.qr(W, mode="reduced")
    xr = x - Q @ (Q.T @ x)
    zr = z - Q @ (Q.T @ z)
    f_res = olea_pflueger_f(xr, zr, lags=h + 1)
    assert np.isclose(f_res, mop_f_expected, rtol=1e-10, atol=1e-10)


def test_anderson_rubin_and_msw_bands_hac_match_lp_iv():
    """anderson_rubin_test and msw_bands with lags=h+1 match lp_iv's AR confidence set."""
    df = load_rz_data()
    h = 7
    res_lp = lp_iv(
        df, y="y", x="g", z="newsy", horizons=[h], n_lags=0,
        controls=RZ_CONTROLS, alpha=0.10, cumulative=True, anderson_rubin=True,
    )
    ar_lo = float(res_lp.loc[h, "ar_lo"])
    ar_hi = float(res_lp.loc[h, "ar_hi"])

    sub = df[["y", "g", "newsy"] + RZ_CONTROLS].copy()
    sub["__dy__"] = sum(sub["y"].shift(-j) - sub["y"].shift(1) for j in range(h + 1))
    sub["__dx__"] = sum(sub["g"].shift(-j) for j in range(h + 1))
    sub = sub.dropna()

    yv = sub["__dy__"].to_numpy(dtype=float)
    xv = sub["__dx__"].to_numpy(dtype=float)
    zv = sub["newsy"].to_numpy(dtype=float)
    W = sub[RZ_CONTROLS].to_numpy(dtype=float)

    # At the exact AR endpoints from lp_iv, the HAC AR test p-value is alpha = 0.10
    ar_at_lo = anderson_rubin_test(ar_lo, yv, xv, zv, controls=W, lags=h + 1)
    ar_at_hi = anderson_rubin_test(ar_hi, yv, xv, zv, controls=W, lags=h + 1)
    assert np.isclose(ar_at_lo.p_value, 0.10, atol=1e-6)
    assert np.isclose(ar_at_hi.p_value, 0.10, atol=1e-6)

    # msw_bands with lags=h+1 on a fine grid brackets the same endpoints
    grid = np.linspace(ar_lo - 0.1, ar_hi + 0.1, 2001)
    lo_grid, hi_grid = msw_bands(
        grid, yv, xv, zv, alpha=0.10, controls=W, lags=h + 1,
    )
    assert np.isclose(lo_grid, ar_lo, atol=2e-3)
    assert np.isclose(hi_grid, ar_hi, atol=2e-3)


def test_msw_bands_warns_on_empty_unbounded_and_disconnected_sets():
    """msw_bands warns and returns (nan, nan) on empty set, and warns on boundary/disconnected sets."""
    rng = np.random.default_rng(42)
    n = 120
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    Z = np.column_stack([z1, z2])
    # Overidentified incompatible instruments -> empty AR confidence set everywhere
    x = z1 + 0.1 * rng.standard_normal(n)
    y = 2.0 * z1 - 8.0 * z2 + 0.1 * rng.standard_normal(n)

    with pytest.warns(RuntimeWarning, match="confidence set is empty"):
        lo, hi = msw_bands(np.linspace(-5.0, 5.0, 201), y, x, Z, alpha=0.10)
    assert np.isnan(lo) and np.isnan(hi)

    # Weak instrument -> unbounded set touching grid boundaries
    z_weak = rng.standard_normal(n)
    x_weak = 0.01 * z_weak + rng.standard_normal(n)
    y_weak = rng.standard_normal(n)
    with pytest.warns(RuntimeWarning, match="touches the grid boundary"):
        lo_w, hi_w = msw_bands(np.linspace(-2.0, 2.0, 101), y_weak, x_weak, z_weak, alpha=0.10)
    assert np.isfinite(lo_w) and np.isfinite(hi_w)

    # Disconnected set (two rays inside a wide grid when AR(infty) < cv but middle rejects)
    t = np.linspace(0, 1, n)
    z_ray = np.sin(2 * np.pi * t)
    u_orth = np.cos(2 * np.pi * t)
    x_ray = 0.05 * z_ray + u_orth
    y_ray = 1.5 * z_ray + 0.2 * u_orth
    with pytest.warns(RuntimeWarning, match="disconnected"):
        msw_bands(np.linspace(-200.0, 200.0, 401), y_ray, x_ray, z_ray, alpha=0.10)


def test_instrument_influence_identifies_rz_outliers():
    """instrument_influence flags 1941Q4 in full-sample slack and 1940Q2 without WWII."""
    df = load_rz_data()
    # Full-sample slack regime (slack_l == 1)
    slack = df[df["slack_l"] == 1.0].dropna(subset=["newsy"] + RZ_CONTROLS)
    diag_full = instrument_influence(
        slack["newsy"],
        controls=slack[RZ_CONTROLS],
    )
    top_date_full = diag_full.index[0]
    max_share_full = float(diag_full["share"].iloc[0])
    n_eff_full = float(1.0 / np.sum(diag_full["share"].to_numpy() ** 2))
    assert top_date_full == pd.Timestamp("1941-10-01")
    assert 0.45 < max_share_full < 0.60
    assert n_eff_full < 10.0

    # Slack regime excluding WWII (1941Q3..1945Q4 masked as in RZ)
    df_nowwii = df.copy()
    wwii = (df_nowwii.index >= "1941-07-01") & (df_nowwii.index <= "1945-12-31")
    df_nowwii.loc[wwii, ["y", "g", "newsy"]] = np.nan
    for k in range(1, 5):
        for v in ("y", "g", "newsy"):
            df_nowwii[f"{v}_l{k}"] = df_nowwii[v].shift(k)
    slack_nowwii = df_nowwii[df_nowwii["slack_l"] == 1.0].dropna(subset=["newsy"] + RZ_CONTROLS)
    diag_nowwii = instrument_influence(
        slack_nowwii["newsy"],
        controls=slack_nowwii[RZ_CONTROLS],
    )
    top_date_nowwii = diag_nowwii.index[0]
    max_share_nowwii = float(diag_nowwii["share"].iloc[0])
    assert top_date_nowwii == pd.Timestamp("1940-04-01")
    assert max_share_nowwii > 0.75


def test_state_lag_matches_manual_lagged_state():
    """state_lag=1 on raw state matches state_lag=0 on manually shifted state."""
    rng = np.random.default_rng(123)
    T = 140
    unemp = 6.5 + 1.5 * rng.standard_normal(T)
    z = rng.standard_normal(T)
    g = 0.8 * z + 0.3 * rng.standard_normal(T)
    y = np.cumsum(0.6 * g + 0.4 * rng.standard_normal(T))
    df = pd.DataFrame({"y": y, "g": g, "z": z, "unemp": unemp})
    df["unemp_l1"] = df["unemp"].shift(1)

    res_lag1 = lp_state_dep(
        df, y="y", x="g", state="unemp", state_lag=1,
        threshold=6.5, transition="threshold", horizon=4, lags=2,
    )
    res_manual = lp_state_dep(
        df, y="y", x="g", state="unemp_l1", state_lag=0,
        threshold=6.5, transition="threshold", horizon=4, lags=2,
    )
    pd.testing.assert_frame_equal(res_lag1, res_manual)

    iv_lag1 = lp_state_dep_iv(
        df, y="y", x="g", z="z", state="unemp", state_lag=1,
        threshold=6.5, transition="logistic", gamma=2.0, horizon=4, lags=2,
    )
    iv_manual = lp_state_dep_iv(
        df, y="y", x="g", z="z", state="unemp_l1", state_lag=0,
        threshold=6.5, transition="logistic", gamma=2.0, horizon=4, lags=2,
    )
    pd.testing.assert_frame_equal(iv_lag1, iv_manual)

    garch_lag1 = lp_garch_state(
        df, y="y", x="g", sigma_col="unemp", state_lag=1,
        threshold=6.5, transition="logistic", gamma=2.0, horizon=3, lags=1,
    )
    garch_manual = lp_garch_state(
        df, y="y", x="g", sigma_col="unemp_l1", state_lag=0,
        threshold=6.5, transition="logistic", gamma=2.0, horizon=3, lags=1,
    )
    pd.testing.assert_frame_equal(garch_lag1, garch_manual)

    with pytest.raises(ValueError, match="state_lag"):
        lp_state_dep(df, y="y", x="g", state="unemp", state_lag=-1)


def test_sanderson_windmeijer_cond_f_in_lp_state_dep_iv():
    """cond_f_H and cond_f_L equal within-regime F under threshold and populate logistic F."""
    df = load_rz_data()
    res_thr = lp_state_dep_iv(
        df, y="y", x="g", z="newsy", state="slack_l",
        horizons=[0, 7, 15], n_lags=0, transition="threshold",
        threshold=0.5, controls=RZ_CONTROLS, cumulative=True,
    )
    assert "cond_f_H" in res_thr.columns
    assert "cond_f_L" in res_thr.columns
    np.testing.assert_allclose(
        res_thr["cond_f_H"].to_numpy(),
        res_thr["first_stage_f_H"].to_numpy(),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        res_thr["cond_f_L"].to_numpy(),
        res_thr["first_stage_f_L"].to_numpy(),
        rtol=1e-10,
    )

    # Under smooth logistic transition, first_stage_f_H/L equal cond_f_H/L
    d_raw = load_rz_data()
    # Create continuous state from y_l1
    res_log = lp_state_dep_iv(
        d_raw, y="y", x="g", z="newsy", state="y_l1",
        horizons=[0, 4], n_lags=0, transition="logistic", gamma=1.5,
        controls=RZ_CONTROLS, cumulative=True,
    )
    np.testing.assert_allclose(
        res_log["cond_f_H"].to_numpy(),
        res_log["first_stage_f_H"].to_numpy(),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        res_log["cond_f_L"].to_numpy(),
        res_log["first_stage_f_L"].to_numpy(),
        rtol=1e-12,
    )
    assert np.all(res_log["cond_f_H"].to_numpy() > 0)
    assert np.all(res_log["cond_f_L"].to_numpy() > 0)


def test_lp_smooth_transition_irf_deprecation_and_runtime_warning():
    """lp_smooth_transition_irf emits DeprecationWarning and warns on horizon failure."""
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x": [1.0, 1.0, 1.0, 1.0, 1.0],  # constant x -> collinear with intercept
        "s": [-1.0, -0.5, 0.0, 0.5, 1.0],
    })
    with pytest.deprecated_call(match="lp_state_dep"):
        out = lp_smooth_transition_irf(df, y="y", x="x", state_var="s", horizons=[0, 1], n_lags=1)
    assert isinstance(out, pd.DataFrame)
