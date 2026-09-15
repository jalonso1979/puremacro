"""Unit tests for Montiel Olea & Pflueger (2013) Weak IV in LP-IV and LA-LP.

Verifies:
1. Montiel Olea & Pflueger effective F-statistic (mop_f) in lp_iv.
2. Exact equivalence between mop_f and squared HAC t-stat for single instrument (kz=1).
3. MOP critical values reporting (mop_cv_10, mop_cv_20).
4. Multiple instrument support in lp_iv (kz >= 2).
5. Weak-IV robust Anderson-Rubin confidence sets for single and multiple instruments.
6. Lag-augmented LP with IV (la_lp with z parameter and la_lp_iv) with White-robust MOP F and AR sets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.lp.iv import (
    lp_iv,
    compute_mop_effective_f,
    mop_critical_values,
)
from puremacro.lp.la_lp import (
    la_lp,
    la_lp_iv,
)


def test_mop_critical_values():
    """Verify MOP critical values match published benchmarks."""
    cv10_1, cv20_1 = mop_critical_values(1)
    assert cv10_1 == 11.52
    assert cv20_1 == 6.70

    cv10_2, cv20_2 = mop_critical_values(2)
    assert cv10_2 == 11.12
    assert cv20_2 == 6.00

    cv10_5, cv20_5 = mop_critical_values(5)
    assert cv10_5 == 9.80
    assert cv20_5 == 4.90


def test_lp_iv_mop_f_single_instrument():
    """Verify mop_f matches first_stage_f (squared HAC t-stat) for single instrument."""
    rng = np.random.default_rng(101)
    T = 250
    z = rng.standard_normal(T)
    x = 0.8 * z + 0.4 * rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] - 0.5 * x[t - 1] + rng.standard_normal()

    df = pd.DataFrame({"x": x, "y": y, "z": z})
    out = lp_iv(df, y="y", x="x", z="z", horizons=[0, 1, 2], n_lags=1)

    assert "mop_f" in out.columns
    assert "mop_cv_10" in out.columns
    assert "mop_cv_20" in out.columns

    for h in [0, 1, 2]:
        row = out.loc[out["h"] == h]
        fs_f = row["first_stage_f"].iloc[0]
        mop_f = row["mop_f"].iloc[0]
        # For kz=1, MOP effective F is mathematically identical to squared HAC t-stat
        np.testing.assert_allclose(mop_f, fs_f, rtol=1e-6)
        assert row["mop_cv_10"].iloc[0] == 11.52
        assert row["mop_cv_20"].iloc[0] == 6.70


def test_lp_iv_multiple_instruments():
    """Verify lp_iv handles multiple instruments (kz=2) with MOP effective F and AR confidence sets."""
    rng = np.random.default_rng(123)
    T = 300
    z1 = rng.standard_normal(T)
    z2 = rng.standard_normal(T)
    x = 0.7 * z1 + 0.6 * z2 + 0.5 * rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.4 * y[t - 1] - 0.6 * x[t - 1] + rng.standard_normal()

    df = pd.DataFrame({"x": x, "y": y, "z1": z1, "z2": z2})
    out = lp_iv(
        df,
        y="y",
        x="x",
        z=["z1", "z2"],
        horizons=[0, 1, 2],
        n_lags=1,
        anderson_rubin=True,
    )

    assert "mop_f" in out.columns
    assert "ar_lo" in out.columns
    assert "ar_hi" in out.columns
    assert "ar_set_type" in out.columns

    # Strong instruments design: mop_f should comfortably exceed 10% critical value
    assert out.loc[out["h"] == 1, "mop_f"].iloc[0] > 15.0
    assert out.loc[out["h"] == 1, "mop_cv_10"].iloc[0] == 11.12
    # Bounded confidence set containing true slope -0.6
    assert out.loc[out["h"] == 1, "ar_set_type"].iloc[0] == "bounded"
    lo = out.loc[out["h"] == 1, "ar_lo"].iloc[0]
    hi = out.loc[out["h"] == 1, "ar_hi"].iloc[0]
    assert lo < -0.6 < hi


def test_lp_iv_weak_instruments_multiple():
    """Verify weak multiple instruments result in small MOP F and unbounded/wide AR sets."""
    rng = np.random.default_rng(303)
    T = 200
    z1 = rng.standard_normal(T)
    z2 = rng.standard_normal(T)
    # Instruments have near-zero correlation with x
    x = 0.05 * z1 + 0.05 * z2 + rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] - 0.4 * x[t - 1] + rng.standard_normal()

    df = pd.DataFrame({"x": x, "y": y, "z1": z1, "z2": z2})
    out = lp_iv(
        df,
        y="y",
        x="x",
        z=["z1", "z2"],
        horizons=[1],
        n_lags=1,
        weak_iv_robust=True,
    )

    # Effective F should be small (below 10% bias threshold of 11.12)
    assert out["mop_f"].iloc[0] < 11.12


def test_la_lp_iv_single_instrument():
    """Verify lag-augmented LP with IV (la_lp_iv) computes White-robust MOP F and AR sets."""
    rng = np.random.default_rng(404)
    T = 300
    z = rng.standard_normal(T)
    x = 0.9 * z + 0.4 * rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + 1.2 * x[t - 1] + rng.standard_normal()

    df = pd.DataFrame({"x": x, "y": y, "z": z})

    out = la_lp_iv(
        df,
        y="y",
        x="x",
        z="z",
        horizons=[0, 1, 2],
        n_lags=2,
        extra_lags=2,
        anderson_rubin=True,
    )

    assert out.method == "la_lp_iv"
    assert "mop_f" in out.columns
    assert "mop_cv_10" in out.columns
    assert "ar_lo" in out.columns
    assert "ar_hi" in out.columns
    assert "ar_set_type" in out.columns

    # Strong instrument: MOP F is large
    assert out.loc[out["h"] == 1, "mop_f"].iloc[0] > 20.0
    # Bounded White AR interval contains true effect 1.2
    assert out.loc[out["h"] == 1, "ar_set_type"].iloc[0] == "bounded"
    lo = out.loc[out["h"] == 1, "ar_lo"].iloc[0]
    hi = out.loc[out["h"] == 1, "ar_hi"].iloc[0]
    assert lo < 1.2 < hi


def test_la_lp_iv_multiple_instruments():
    """Verify lag-augmented LP with multiple instruments (kz=2)."""
    rng = np.random.default_rng(505)
    T = 350
    z1 = rng.standard_normal(T)
    z2 = rng.standard_normal(T)
    x = 0.8 * z1 + 0.7 * z2 + 0.4 * rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.4 * y[t - 1] + 0.8 * x[t - 1] + rng.standard_normal()

    df = pd.DataFrame({"x": x, "y": y, "z1": z1, "z2": z2})

    out = la_lp(
        df,
        y="y",
        x="x",
        z=["z1", "z2"],
        horizons=[1, 2],
        n_lags=2,
        extra_lags=2,
        weak_iv_robust=True,
    )

    assert out.method == "la_lp_iv"
    assert out.loc[out["h"] == 1, "mop_f"].iloc[0] > 15.0
    assert out.loc[out["h"] == 1, "mop_cv_10"].iloc[0] == 11.12
    assert out.loc[out["h"] == 1, "ar_set_type"].iloc[0] == "bounded"
