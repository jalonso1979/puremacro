"""Tests for puremacro.regress.lp panel local projection estimator."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


def _make_synthetic_panel(n_units=20, n_periods=80, beta=0.5, sigma=1.0, seed=0):
    """y_{i,t} = alpha_i + beta * shock_t + e_{i,t},  shock ~ N(0,1) iid."""
    rng = np.random.default_rng(seed)
    units = list(range(n_units))
    dates = pd.period_range("2006Q1", periods=n_periods, freq="Q").to_timestamp(how="end")
    shock = rng.standard_normal(n_periods)
    rows = []
    for i, u in enumerate(units):
        alpha_i = rng.standard_normal()
        e = rng.standard_normal(n_periods) * sigma
        y = alpha_i + beta * shock + e
        for t, d in enumerate(dates):
            rows.append({"unit": u, "date": d, "y": y[t], "shock": shock[t]})
    return pd.DataFrame(rows)


def test_lp_panel_recovers_known_beta_at_h0():
    """y_{i,t+0} = alpha_i + beta*shock_t + e: lp_panel should recover beta at h=0."""
    from puremacro.regress.lp import lp_panel
    df = _make_synthetic_panel(n_units=30, n_periods=200, beta=0.7, seed=1)
    out = lp_panel(df, y="y", shock="shock", horizons=range(0, 1),
                   unit="unit", date="date", se="driscoll_kraay")
    row = out.iloc[0]
    assert abs(row["beta"] - 0.7) < 0.05
    assert row["horizon"] == 0
    assert row["se"] > 0
    assert row["n_obs"] > 0


def test_lp_panel_horizon_decay_with_white_noise_shock():
    """For iid shock, response at h>=1 should be ~0 (within CI)."""
    from puremacro.regress.lp import lp_panel
    df = _make_synthetic_panel(n_units=30, n_periods=200, beta=0.5, seed=2)
    out = lp_panel(df, y="y", shock="shock", horizons=range(0, 5),
                   unit="unit", date="date", se="driscoll_kraay")
    # h=0 recovers beta
    assert abs(out.loc[out["horizon"] == 0, "beta"].iloc[0] - 0.5) < 0.05
    # h>=1: shock is iid, expected response is 0 (within ~3 SE).
    for h in range(1, 5):
        row = out.loc[out["horizon"] == h].iloc[0]
        assert abs(row["beta"]) < 0.10  # tolerance reflects sample noise


def test_lp_panel_unit_fe_absorbs_unit_means():
    """Adding a constant per unit shouldn't change estimated beta when unit_fe=True."""
    from puremacro.regress.lp import lp_panel
    df = _make_synthetic_panel(n_units=20, n_periods=100, beta=0.6, seed=3)
    df_shifted = df.copy()
    rng = np.random.default_rng(7)
    shifts = {u: 10 * rng.standard_normal() for u in df["unit"].unique()}
    df_shifted["y"] = df_shifted.apply(lambda r: r["y"] + shifts[r["unit"]], axis=1)
    out_a = lp_panel(df, y="y", shock="shock", horizons=range(0, 1),
                     unit="unit", date="date", unit_fe=True, se="driscoll_kraay")
    out_b = lp_panel(df_shifted, y="y", shock="shock", horizons=range(0, 1),
                     unit="unit", date="date", unit_fe=True, se="driscoll_kraay")
    assert abs(out_a["beta"].iloc[0] - out_b["beta"].iloc[0]) < 1e-8


def test_lp_panel_returns_required_columns():
    from puremacro.regress.lp import lp_panel
    df = _make_synthetic_panel(n_units=10, n_periods=40, seed=4)
    out = lp_panel(df, y="y", shock="shock", horizons=range(0, 3),
                   unit="unit", date="date", se="driscoll_kraay")
    for col in ("horizon", "beta", "se", "t", "p", "ci_lo", "ci_hi", "n_obs"):
        assert col in out.columns, f"missing column {col}"
    assert len(out) == 3


def test_lp_panel_dk_lag_auto_picks_h_plus_one():
    """When dk_lag=None, lp_panel uses h+1 as the truncation parameter."""
    from puremacro.regress.lp import _dk_default_lag
    assert _dk_default_lag(0) == 1
    assert _dk_default_lag(4) == 5
    assert _dk_default_lag(12) == 13


def test_lp_panel_with_controls():
    """Adding a control variable shouldn't blow up the estimator."""
    from puremacro.regress.lp import lp_panel
    df = _make_synthetic_panel(n_units=15, n_periods=80, beta=0.5, seed=5)
    rng = np.random.default_rng(11)
    df["x1"] = rng.standard_normal(len(df))
    out = lp_panel(df, y="y", shock="shock", horizons=range(0, 2),
                   unit="unit", date="date", controls=["x1"],
                   se="driscoll_kraay")
    assert abs(out.loc[out["horizon"] == 0, "beta"].iloc[0] - 0.5) < 0.10
    assert (out["se"] > 0).all()
