"""Tests for High-Dimensional Penalized Macro Forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.forecast.penalized import (
    PenalizedForecastResult,
    forecast_penalized,
)


@pytest.fixture
def synthetic_sparse_macro():
    rng = np.random.default_rng(123)
    T = 120
    P = 20
    X = rng.normal(size=(T, P))
    
    # Predictive DGP for horizon h=1: y_{t+1} depends on X_t[:, 0] and X_t[:, 2]
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 1.5 + 2.0 * X[t-1, 0] - 1.5 * X[t-1, 2] + rng.normal(scale=0.3)
    y[0] = 1.5 + rng.normal(scale=0.3)
    
    df_X = pd.DataFrame(X, columns=[f"Pred_{i+1}" for i in range(P)])
    s_y = pd.Series(y, name="Inflation")
    return df_X, s_y


def test_forecast_penalized_basic(synthetic_sparse_macro):
    df_X, s_y = synthetic_sparse_macro
    res = forecast_penalized(df_X, s_y, horizon=1, alpha=0.5, adaptive=True)

    assert isinstance(res, PenalizedForecastResult)
    assert isinstance(res.forecast, float)
    assert not np.isnan(res.forecast)
    assert len(res.coefficients) == 20
    assert 0.0 <= res.in_sample_r2 <= 1.0
    assert res.horizon == 1
    
    # Check that true sparse features are recovered
    assert "Pred_1" in res.selected_features
    assert "Pred_3" in res.selected_features
    # Check sparsity (less than half features selected)
    assert len(res.selected_features) <= 10

    s = res.summary()
    assert "Penalized Macro Forecasting" in s
    assert "Selected Predictors" in s
    assert "Optimal Penalty" in s
