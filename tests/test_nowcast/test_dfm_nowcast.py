"""Tests for Mixed-Frequency Dynamic Factor Model GDP Nowcasting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.nowcast.dfm_nowcast import NowcastResult, nowcast_gdp


@pytest.fixture
def synthetic_nowcast_data():
    rng = np.random.default_rng(42)
    n_months = 60
    n_series = 10
    
    dates_m = pd.date_range("2019-01-01", periods=n_months, freq="MS")
    # True factor
    F_true = np.zeros(n_months)
    for t in range(1, n_months):
        F_true[t] = 0.8 * F_true[t-1] + rng.normal()

    # Monthly indicators
    X = np.zeros((n_months, n_series))
    for i in range(n_series):
        load = rng.uniform(0.5, 1.5)
        X[:, i] = load * F_true + rng.normal(scale=0.3, size=n_months)

    df_X = pd.DataFrame(X, index=dates_m, columns=[f"Series_{i+1}" for i in range(n_series)])
    
    # Introduce ragged edges in the last month: series 5 to 9 are missing
    df_X.iloc[-1, 5:] = np.nan

    # Quarterly GDP
    dates_q = pd.date_range("2019-01-01", periods=n_months // 3, freq="QS")
    # Quarterly average factor
    F_q = df_X.resample("QE").mean().to_numpy().mean(axis=1)[:len(dates_q)]
    gdp = 2.0 + 1.2 * F_q + rng.normal(scale=0.2, size=len(dates_q))
    s_gdp = pd.Series(gdp, index=dates_q.to_period("Q").astype(str), name="GDP")

    return df_X, s_gdp


def test_nowcast_gdp_basic(synthetic_nowcast_data):
    df_X, s_gdp = synthetic_nowcast_data
    res = nowcast_gdp(df_X, s_gdp, n_factors=2)

    assert isinstance(res, NowcastResult)
    assert isinstance(res.nowcast, float)
    assert not np.isnan(res.nowcast)
    assert res.factors.shape == (60, 2)
    assert res.loadings.shape == (10, 2)
    assert 0.0 <= res.model_r2 <= 1.0
    assert not res.news_decomposition.empty

    s = res.summary()
    assert "Dynamic Factor Model GDP Nowcasting" in s
    assert "GDP Growth Nowcast" in s
    assert "Quarterly Bridge Regression R²" in s
