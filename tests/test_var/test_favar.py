"""Tests for Factor-Augmented Vector Autoregression (favar)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.var.favar import FAVARResult, favar


@pytest.fixture
def synthetic_panel():
    rng = np.random.default_rng(123)
    T = 150
    N = 25
    # True latent factors: 2 factors
    F_true = np.zeros((T, 2))
    for t in range(1, T):
        F_true[t, 0] = 0.8 * F_true[t-1, 0] + rng.normal()
        F_true[t, 1] = 0.5 * F_true[t-1, 1] + rng.normal()

    # Policy variable: Fed funds rate
    Y_policy = np.zeros(T)
    for t in range(1, T):
        Y_policy[t] = 0.7 * Y_policy[t-1] + 0.3 * F_true[t-1, 0] + rng.normal()

    # Panel variables driven by factors + policy + idiosyncratic noise
    Lambda = rng.uniform(-1.0, 1.0, size=(N, 3))
    Z_true = np.column_stack([Y_policy, F_true])
    X = Z_true @ Lambda.T + rng.normal(scale=0.5, size=(T, N))
    
    col_names = [f"Series_{i+1}" for i in range(N)]
    return pd.DataFrame(X, columns=col_names), pd.Series(Y_policy, name="FFR")


def test_favar_basic_estimation(synthetic_panel):
    panel, policy = synthetic_panel
    res = favar(panel, policy, n_factors=2, p=1, horizon=12, n_boot=50, seed=42)

    assert isinstance(res, FAVARResult)
    assert res.irf_panel.shape == (13, 25)
    assert res.irf_lower_panel.shape == (13, 25)
    assert res.irf_upper_panel.shape == (13, 25)
    assert len(res.irf_policy) == 13
    assert res.irf_factors.shape == (13, 2)
    assert res.factors.shape == (150, 2)
    assert res.loadings.shape == (25, 3)
    assert len(res.explained_variance_ratio) == 2
    assert res.horizon == 12

    # Verify summary string
    s = res.summary()
    assert "Factor-Augmented VAR (FAVAR)" in s
    assert "Panel Dimensions" in s
    assert "Total Panel Variance Explained" in s


def test_favar_dimension_mismatch(synthetic_panel):
    panel, policy = synthetic_panel
    with pytest.raises(ValueError, match="favar: length of policy_var"):
        favar(panel, policy.iloc[:-10], n_factors=2)


def test_favar_exports_and_plot(synthetic_panel):
    panel, policy = synthetic_panel
    res = favar(panel, policy, n_factors=2, p=1, horizon=4, n_boot=20, seed=42)

    df = res.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (5, 25)

    md = res.to_markdown()
    assert "|" in md

    ltx = res.to_latex()
    assert "\\begin{tabular}" in ltx

    typ = res.to_typst()
    assert "#table(" in typ

    # Plot smoke test
    fig = res.plot(variables=["Series_1", "Series_2"])
    assert fig is not None

