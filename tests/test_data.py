"""Tests for puremacro.data: panel reshaping and trend/cycle filters."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.data import (
    align_yx,
    bk_filter,
    hamilton_filter,
    hp_filter,
    long_to_wide,
    slice_country,
)


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------
def _build_long_panel() -> pd.DataFrame:
    dates = pd.date_range("2000-01-01", periods=4, freq="QS")
    rows = []
    for code in ("USA", "MEX"):
        for d in dates:
            rows.append({"code": code, "date": d, "variable": "y", "value": np.random.randn()})
            rows.append({"code": code, "date": d, "variable": "x", "value": np.random.randn()})
    return pd.DataFrame(rows)


def test_long_to_wide_round_trip():
    np.random.seed(0)
    long_df = _build_long_panel()
    wide = long_to_wide(long_df)
    assert isinstance(wide.index, pd.MultiIndex)
    assert wide.index.names == ["code", "date"]
    assert set(wide.columns) == {"y", "x"}
    assert wide.shape == (8, 2)


def test_slice_country():
    np.random.seed(1)
    wide = long_to_wide(_build_long_panel())
    sub = slice_country(wide, "USA")
    assert isinstance(sub.index, pd.DatetimeIndex)
    assert set(sub.columns) == {"y", "x"}
    assert len(sub) == 4


def test_slice_country_keyerror():
    np.random.seed(2)
    wide = long_to_wide(_build_long_panel())
    with pytest.raises(KeyError):
        slice_country(wide, "ZZZ")


def test_align_yx_basic():
    idx = pd.date_range("2000-01-01", periods=8, freq="QS")
    df = pd.DataFrame({
        "y": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
        "x": [0.1, 0.2, 0.3, np.nan, 0.5, 0.6, 0.7, 0.8],
    }, index=idx)
    y, x = align_yx(df, "y", "x", start="2000Q1", end="2001Q4")
    assert (y.index == x.index).all()
    assert y.notna().all() and x.notna().all()
    # 8 total quarters; 2000Q3 dropped (y NaN), 2000Q4 dropped (x NaN). 6 left.
    assert len(y) == 6
    # Restrict the window: keep 2001Q1..2001Q4 only.
    y2, x2 = align_yx(df, "y", "x", start="2001Q1", end="2001Q4")
    assert len(y2) == 4
    assert y2.notna().all() and x2.notna().all()


# ---------------------------------------------------------------------------
# HP filter
# ---------------------------------------------------------------------------
def test_hp_filter_decomposition():
    t = np.arange(200)
    y = 0.1 * t + np.sin(2 * np.pi * t / 8.0)
    s = pd.Series(y, index=pd.RangeIndex(200))
    cycle, trend = hp_filter(s, lamb=1600.0)
    assert np.allclose(cycle.values + trend.values, y, atol=1e-8)
    assert cycle.index.equals(s.index)


def test_hp_filter_lambda_scaling():
    rng = np.random.default_rng(0)
    t = np.arange(120)
    y = 0.05 * t + rng.normal(scale=0.3, size=120)
    s = pd.Series(y)
    # Very large lambda -> trend ~ linear fit.
    cycle_big, trend_big = hp_filter(s, lamb=1e10)
    coef = np.polyfit(t, y, 1)
    linear = np.polyval(coef, t)
    assert np.allclose(trend_big.values, linear, atol=5e-2)
    # Very small lambda -> cycle ~ 0 (trend hugs y).
    cycle_small, _ = hp_filter(s, lamb=1e-6)
    assert np.max(np.abs(cycle_small.values)) < 1e-3


# ---------------------------------------------------------------------------
# Hamilton filter
# ---------------------------------------------------------------------------
def test_hamilton_filter_shape():
    rng = np.random.default_rng(42)
    y = pd.Series(rng.normal(size=100))
    cycle, trend = hamilton_filter(y, h=8, p=4)
    assert len(cycle) == 100
    assert cycle.iloc[: 8 + 4 - 1].isna().all()
    assert trend.iloc[: 8 + 4 - 1].isna().all()
    finite = cycle.notna() & trend.notna()
    s_orig = y.values
    s_sum = (cycle + trend).values
    assert np.allclose(s_sum[finite.values], s_orig[finite.values], atol=1e-8)


# ---------------------------------------------------------------------------
# Baxter-King filter
# ---------------------------------------------------------------------------
def test_bk_filter_basic():
    rng = np.random.default_rng(7)
    y = pd.Series(rng.normal(size=200))
    K = 12
    cycle = bk_filter(y, low=6, high=32, K=K)
    assert len(cycle) == 200
    assert cycle.iloc[:K].isna().all()
    assert cycle.iloc[-K:].isna().all()
    assert cycle.iloc[K:-K].notna().all()
