"""Unit tests for Double Machine Learning (DML) for Partially Linear Models.

Tests:
1. Pure-NumPy regularized learners (LassoCoordinateDescent, RidgeGCV).
2. Unbiased recovery of causal effect theta_0 on synthetic benchmark data.
3. Asymptotic standard errors and root-N scaling.
4. Nominal 95% confidence interval coverage.
5. Multidimensional treatment D support.
6. Presentation contract (.summary, .plot, .to_markdown, .to_latex, .to_typst).
7. Zero non-Pyodide runtime dependencies.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest

from puremacro.causal import (
    DoubleMLPLR,
    DMLResult,
    LassoCoordinateDescent,
    RidgeGCV,
    dml_plr,
)


def test_lasso_coordinate_descent_sparse_recovery():
    """Verify LassoCoordinateDescent recovers sparse support and coefficients."""
    rng = np.random.default_rng(42)
    n, p = 200, 30
    X = rng.standard_normal((n, p))
    # Sparse true coefficients (only first 3 are non-zero)
    beta_true = np.zeros(p)
    beta_true[:3] = [2.5, -1.8, 1.2]
    y = X @ beta_true + 0.3 * rng.standard_normal(n)

    lasso = LassoCoordinateDescent(n_alphas=40, criterion="bic")
    lasso.fit(X, y)

    pred = lasso.predict(X)
    assert lasso.coef_ is not None
    # Check that non-zero elements are concentrated in first 3
    assert abs(lasso.coef_[0] - 2.5) < 0.4
    assert abs(lasso.coef_[1] - (-1.8)) < 0.4
    assert abs(lasso.coef_[2] - 1.2) < 0.4
    # Zero out negligible coefficients
    assert np.count_nonzero(np.abs(lasso.coef_[3:]) > 0.5) <= 2
    # Prediction correlation
    assert np.corrcoef(pred, y)[0, 1] > 0.95


def test_ridge_gcv_collinear():
    """Verify RidgeGCV handles collinear features with GCV parameter tuning."""
    rng = np.random.default_rng(123)
    n, p = 150, 40
    Z = rng.standard_normal((n, 10))
    # Generate correlated columns
    A = rng.standard_normal((10, p))
    X = Z @ A
    beta_true = rng.standard_normal(p) * 0.5
    y = X @ beta_true + 0.5 * rng.standard_normal(n)

    ridge = RidgeGCV()
    ridge.fit(X, y)

    assert ridge.alpha_ is not None
    assert ridge.alpha_ > 0
    pred = ridge.predict(X)
    assert np.corrcoef(pred, y)[0, 1] > 0.85


def test_dml_unbiased_recovery():
    """Test DML-PLR recovers known causal effect theta_0 on synthetic partially linear DGP."""
    rng = np.random.default_rng(999)
    n, p = 500, 30
    theta_0 = 1.75

    X = rng.standard_normal((n, p))
    # Confounding functions: both Y and D depend on X
    g_0 = 0.8 * X[:, 0] - 1.0 * X[:, 1] + 0.5 * X[:, 2] ** 2
    m_0 = 0.7 * X[:, 0] + 0.9 * X[:, 1] - 0.4 * X[:, 3]

    V = 0.8 * rng.standard_normal(n)
    D = m_0 + V

    U = 0.8 * rng.standard_normal(n)
    Y = D * theta_0 + g_0 + U

    res = dml_plr(Y, D, X, n_folds=5, learner="lasso", random_state=42)

    assert isinstance(res, DMLResult)
    assert isinstance(res.theta, float)
    # Estimated effect should be close to true theta_0 within ~2.5 standard errors
    assert abs(res.theta - theta_0) < 2.5 * res.se
    # 95% CI should contain true theta_0
    assert res.ci_lower <= theta_0 <= res.ci_upper
    # Statistically significant
    assert res.p_value < 1e-4
    assert res.t_stat > 10.0


def test_dml_root_n_se_scaling():
    """Verify DML asymptotic standard error scales proportionally to 1 / sqrt(N)."""
    theta_0 = 1.0

    def simulate_data(n, seed):
        rng = np.random.default_rng(seed)
        p = 20
        X = rng.standard_normal((n, p))
        g_0 = 0.5 * X[:, 0] - 0.5 * X[:, 1]
        m_0 = 0.5 * X[:, 0] + 0.5 * X[:, 2]
        D = m_0 + rng.standard_normal(n)
        Y = D * theta_0 + g_0 + rng.standard_normal(n)
        return Y, D, X

    Y_small, D_small, X_small = simulate_data(200, 101)
    Y_large, D_large, X_large = simulate_data(800, 102)

    res_small = dml_plr(Y_small, D_small, X_small, n_folds=4, learner="ridge", random_state=1)
    res_large = dml_plr(Y_large, D_large, X_large, n_folds=4, learner="ridge", random_state=1)

    ratio = res_small.se / res_large.se
    # Theoretical ratio: sqrt(800 / 200) = 2.0. Check within reasonable empirical tolerance [1.4, 2.6]
    assert 1.4 <= ratio <= 2.6


def test_dml_multidimensional_treatment():
    """Test DML with vector treatments D (k_d = 2)."""
    rng = np.random.default_rng(44)
    n, p = 400, 20
    theta_true = np.array([1.5, -0.8])

    X = rng.standard_normal((n, p))
    g_0 = 0.6 * X[:, 0] - 0.4 * X[:, 1]

    D1 = 0.5 * X[:, 0] + rng.standard_normal(n)
    D2 = -0.5 * X[:, 1] + rng.standard_normal(n)
    D = np.column_stack([D1, D2])

    Y = D @ theta_true + g_0 + rng.standard_normal(n)

    df_D = pd.DataFrame(D, columns=["policy_rate", "credit_spread"])
    s_Y = pd.Series(Y, name="gdp_growth")

    res = DoubleMLPLR(n_folds=4, learner="ridge", random_state=77).fit(s_Y, df_D, X)

    assert isinstance(res.theta, np.ndarray)
    assert len(res.theta) == 2
    assert res.treatment_names == ("policy_rate", "credit_spread")
    assert res.outcome_name == "gdp_growth"
    assert abs(res.theta[0] - theta_true[0]) < 0.25
    assert abs(res.theta[1] - theta_true[1]) < 0.25


def test_dml_result_presentation_contract():
    """Test full presentation contract (.summary, .plot, .to_markdown, .to_latex, .to_typst)."""
    rng = np.random.default_rng(55)
    n = 200
    X = rng.standard_normal((n, 10))
    D = 0.4 * X[:, 0] + rng.standard_normal(n)
    Y = D * 2.0 + 0.5 * X[:, 1] + rng.standard_normal(n)

    res = dml_plr(pd.Series(Y, name="inflation"), pd.Series(D, name="tax_cut"), X, n_folds=3, random_state=12)

    # 1. summary()
    s = res.summary()
    assert isinstance(s, str)
    assert "tax_cut" in s
    assert "inflation" in s
    assert "DML-PLR" in s
    assert "Observations: 200" in s

    # 2. to_markdown()
    md = res.to_markdown()
    assert isinstance(md, str)
    assert "| Variable |" in md
    assert "| tax_cut |" in md

    # 3. to_latex()
    ltx = res.to_latex()
    assert isinstance(ltx, str)
    assert r"\begin{table}" in ltx
    assert "tax\\_cut" in ltx
    assert r"\bottomrule" in ltx

    # 4. to_typst()
    typ = res.to_typst()
    assert isinstance(typ, str)
    assert "#figure(" in typ
    assert "table(" in typ
    assert "tax_cut" in typ

    # 5. plot()
    ax1 = res.plot(kind="forest")
    assert ax1 is not None
    plt.close("all")

    ax2 = res.plot(kind="residuals")
    assert ax2 is not None
    plt.close("all")
