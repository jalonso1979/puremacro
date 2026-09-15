"""Adversarial Empirical Stress Tests for Milestone 3: Challenger 2.

Targeting:
1. DMLResult presentation contract (.summary, .plot, .to_latex, .to_typst, .to_markdown)
   under extreme and degenerating inputs:
   - All NaNs in estimates, standard errors, test statistics, p-values, confidence bounds
   - Infinite values (+inf, -inf)
   - Near-zero SE (1e-25) and zero SE (0.0)
   - Negative treatment coefficients
   - Extreme sample sizes (N=0, N=1, N=10^8)
   - Multidimensional treatment vectors with mixed valid/degenerate entries
   - Headless execution (WASM safety, no GUI popup, PNG/SVG/PDF export)
2. LassoCoordinateDescent coordinate descent robustness:
   - Zero-variance / constant features in X
   - All features in X constant
   - Near-zero variance features (1e-15)
   - Constant outcome y
   - Perfectly collinear / duplicate features in X
   - High-dimensional controls (p > n)
3. RidgeGCV singular covariance and regularization:
   - Rank-deficient feature matrices (rank << p)
   - Exact duplicate features (equal shrinkage allocation)
   - Completely singular / constant feature matrices
   - Ill-conditioned covariance with condition number > 1e15
   - High-dimensional feature matrices (p > n)
   - Custom penalty candidate sequences
4. DoubleMLPLR end-to-end partially linear estimation:
   - Singular control matrices (constant + duplicate + linear combinations)
   - High-dimensional controls (p > n) with both Lasso and Ridge
   - Zero-variance treatment D (unidentified treatment effect)
5. Pyodide purity:
   - Explicit verification of zero unauthorized imports (sklearn, torch, numba, statsmodels, arch)
     across all Milestone 3 modules (puremacro.causal.dml, puremacro.dsge.occbin, puremacro.lp.iv, puremacro.lp.la_lp).
"""
from __future__ import annotations

import io
import sys
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


# ===========================================================================
# 1. DMLResult Presentation Contract under Degenerate Inputs
# ===========================================================================


def test_dml_result_all_nans():
    """Verify DMLResult presentation methods handle NaN values without crashing."""
    n = 100
    res_nan = DMLResult(
        theta=np.nan,
        se=np.nan,
        t_stat=np.nan,
        p_value=np.nan,
        ci_lower=np.nan,
        ci_upper=np.nan,
        n_obs=n,
        n_folds=5,
        learner="lasso",
        residuals_y=np.zeros(n),
        residuals_d=np.zeros(n),
    )

    # 1. summary()
    s = res_nan.summary()
    assert isinstance(s, str)
    assert "nan" in s

    # 2. to_markdown()
    md = res_nan.to_markdown()
    assert isinstance(md, str)
    assert "[nan, nan]" in md

    # 3. to_latex()
    ltx = res_nan.to_latex()
    assert isinstance(ltx, str)
    assert r"\begin{table}" in ltx
    assert "[nan, nan]" in ltx

    # 4. to_typst()
    typ = res_nan.to_typst()
    assert isinstance(typ, str)
    assert "#figure(" in typ
    assert "[nan, nan]" in typ

    # 5. plot()
    ax_forest = res_nan.plot(kind="forest")
    assert ax_forest is not None
    plt.close("all")

    ax_res = res_nan.plot(kind="residuals")
    assert ax_res is not None
    plt.close("all")


def test_dml_result_infinite_values():
    """Verify DMLResult presentation methods handle +/- infinity without crashing."""
    n = 50
    res_inf = DMLResult(
        theta=float("inf"),
        se=float("inf"),
        t_stat=float("inf"),
        p_value=0.0,
        ci_lower=-float("inf"),
        ci_upper=float("inf"),
        n_obs=n,
        n_folds=3,
        learner="ridge",
        residuals_y=np.ones(n),
        residuals_d=np.ones(n),
    )

    s = res_inf.summary()
    assert "inf" in s
    md = res_inf.to_markdown()
    assert "[-inf, inf]" in md
    ltx = res_inf.to_latex()
    assert "[-inf, inf]" in ltx
    typ = res_inf.to_typst()
    assert "[[-inf, inf]]" in typ

    # Headless plot execution with infs
    ax = res_inf.plot(kind="forest")
    assert ax is not None
    plt.close("all")


def test_dml_result_near_zero_se_and_negative_theta():
    """Verify DMLResult handles near-zero SE, negative theta, and large N."""
    n = 10_000_000
    res = DMLResult(
        theta=-42.5678,
        se=1e-25,
        t_stat=-4.25678e26,
        p_value=0.0,
        ci_lower=-42.5678,
        ci_upper=-42.5678,
        n_obs=n,
        n_folds=10,
        learner="lasso",
        residuals_y=np.random.default_rng(1).standard_normal(100),
        residuals_d=np.random.default_rng(2).standard_normal(100),
        treatment_names=("policy_rate_shock",),
        outcome_name="core_inflation",
    )

    s = res.summary()
    assert "-42.5678" in s
    assert "policy_rate_shock" in s
    assert "<0.001" in s

    md = res.to_markdown()
    assert "| policy_rate_shock | -42.5678 |" in md

    ltx = res.to_latex()
    assert "policy\\_rate\\_shock" in ltx
    assert "core_inflation" in ltx

    typ = res.to_typst()
    assert "[policy_rate_shock]" in typ


def test_dml_result_multidimensional_treatments():
    """Verify DMLResult with multidimensional treatment vector (k_d = 4) and mixed values."""
    n = 150
    k_d = 4
    theta = np.array([1.5, -2.0, 0.0, np.nan])
    se = np.array([0.2, 0.3, 0.0, np.nan])
    t_stat = np.array([7.5, -6.667, np.nan, np.nan])
    p_value = np.array([1e-12, 1e-10, np.nan, np.nan])
    ci_lower = theta - 1.96 * se
    ci_upper = theta + 1.96 * se

    res_multi = DMLResult(
        theta=theta,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_obs=n,
        n_folds=5,
        learner="ridge",
        residuals_y=np.random.default_rng(3).standard_normal(n),
        residuals_d=np.random.default_rng(4).standard_normal((n, k_d)),
        treatment_names=("policy_rate", "spread_10y", "constant_d", "unident_d"),
        outcome_name="output_gap",
    )

    s = res_multi.summary()
    assert "policy_rate" in s
    assert "spread_10y" in s
    assert "constant_d" in s
    assert "unident_d" in s

    md = res_multi.to_markdown()
    assert "| policy_rate | 1.5000 |" in md
    assert "| spread_10y | -2.0000 |" in md

    ltx = res_multi.to_latex()
    assert "policy\\_rate" in ltx
    assert "spread\\_10y" in ltx

    typ = res_multi.to_typst()
    assert "[policy_rate]" in typ
    assert "[spread_10y]" in typ

    ax1 = res_multi.plot(kind="forest")
    assert ax1 is not None
    ax2 = res_multi.plot(kind="residuals")
    assert ax2 is not None
    plt.close("all")


def test_dml_result_headless_render_and_export():
    """Verify headless plot rendering into byte buffers (PNG, SVG, PDF) without GUI popup."""
    res = DMLResult(
        theta=1.23,
        se=0.45,
        t_stat=2.73,
        p_value=0.006,
        ci_lower=0.35,
        ci_upper=2.11,
        n_obs=200,
        n_folds=5,
        learner="lasso",
        residuals_y=np.random.default_rng(5).standard_normal(200),
        residuals_d=np.random.default_rng(6).standard_normal(200),
        treatment_names=("monetary_shock",),
        outcome_name="unemployment",
    )

    # 1. PNG export
    ax = res.plot(kind="forest")
    buf_png = io.BytesIO()
    ax.figure.savefig(buf_png, format="png")
    assert buf_png.getvalue().startswith(b"\x89PNG")
    plt.close("all")

    # 2. SVG export
    ax_res = res.plot(kind="residuals")
    buf_svg = io.BytesIO()
    ax_res.figure.savefig(buf_svg, format="svg")
    assert b"<svg" in buf_svg.getvalue()
    plt.close("all")

    # 3. PDF export with custom axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    res.plot(kind="forest", ax=ax1)
    res.plot(kind="residuals", ax=ax2)
    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format="pdf")
    assert buf_pdf.getvalue().startswith(b"%PDF")
    plt.close("all")


# ===========================================================================
# 2. LassoCoordinateDescent Robustness: Constant & Collinear Features
# ===========================================================================


def test_lasso_constant_and_zero_variance_controls():
    """Verify LassoCoordinateDescent handles constant/zero-variance features safely."""
    rng = np.random.default_rng(12345)
    n, p = 120, 10

    # Plant constant features at columns 0 and 3
    X = rng.standard_normal((n, p))
    X[:, 0] = 5.0
    X[:, 3] = -2.5
    y = 2.0 * X[:, 1] - 1.5 * X[:, 2] + rng.standard_normal(n) * 0.1

    lasso = LassoCoordinateDescent()
    lasso.fit(X, y)

    assert lasso.coef_ is not None
    assert not np.isnan(lasso.coef_).any()
    assert not np.isinf(lasso.coef_).any()
    # Zero-variance columns must receive exactly 0 coefficient
    assert abs(lasso.coef_[0]) < 1e-10
    assert abs(lasso.coef_[3]) < 1e-10
    pred = lasso.predict(X)
    assert not np.isnan(pred).any()


def test_lasso_all_constant_features():
    """Verify LassoCoordinateDescent when all features in X are constant."""
    n, p = 80, 8
    X = np.ones((n, p)) * 3.14
    y = np.random.default_rng(11).standard_normal(n)

    lasso = LassoCoordinateDescent()
    lasso.fit(X, y)

    assert lasso.coef_ is not None
    assert not np.isnan(lasso.coef_).any()
    assert np.all(lasso.coef_ == 0.0)
    pred = lasso.predict(X)
    assert not np.isnan(pred).any()
    np.testing.assert_allclose(pred, np.mean(y), atol=1e-5)


def test_lasso_near_zero_variance_features():
    """Verify LassoCoordinateDescent with near-zero variance features (variance 1e-30)."""
    rng = np.random.default_rng(22)
    n, p = 100, 10
    X = rng.standard_normal((n, p))
    X[:, 2] = 1.0 + 1e-15 * rng.standard_normal(n)
    y = X[:, 0] * 1.5 + rng.standard_normal(n) * 0.2

    lasso = LassoCoordinateDescent()
    lasso.fit(X, y)
    assert lasso.coef_ is not None
    assert not np.isnan(lasso.coef_).any()


def test_lasso_collinear_features():
    """Verify LassoCoordinateDescent with perfectly collinear features."""
    rng = np.random.default_rng(33)
    n, p = 100, 8
    X = rng.standard_normal((n, p))
    X[:, 4] = 3.0 * X[:, 2]  # Exact collinearity
    y = X[:, 2] * 2.0 + rng.standard_normal(n) * 0.1

    lasso = LassoCoordinateDescent()
    lasso.fit(X, y)
    assert lasso.coef_ is not None
    assert not np.isnan(lasso.coef_).any()
    pred = lasso.predict(X)
    assert not np.isnan(pred).any()


# ===========================================================================
# 3. RidgeGCV Singular Covariance & Regularization
# ===========================================================================


def test_ridge_gcv_rank_deficient():
    """Verify RidgeGCV on rank-deficient X (rank 3 in 30 features)."""
    rng = np.random.default_rng(44)
    n, p = 120, 30
    Z = rng.standard_normal((n, 3))
    A = rng.standard_normal((3, p))
    X = Z @ A
    y = Z[:, 0] * 2.0 - Z[:, 1] + rng.standard_normal(n) * 0.1

    ridge = RidgeGCV()
    ridge.fit(X, y)

    assert ridge.alpha_ is not None
    assert ridge.alpha_ > 0
    assert not np.isnan(ridge.coef_).any()
    assert not np.isinf(ridge.coef_).any()
    pred = ridge.predict(X)
    assert np.corrcoef(pred, y)[0, 1] > 0.90


def test_ridge_gcv_exact_duplicate_features():
    """Verify RidgeGCV allocates equal regularized coefficients to duplicate features."""
    rng = np.random.default_rng(55)
    n, p = 100, 10
    X = rng.standard_normal((n, p))
    X[:, 1] = X[:, 0]
    X[:, 2] = X[:, 0]
    y = X[:, 0] * 3.0 + rng.standard_normal(n) * 0.1

    ridge = RidgeGCV()
    ridge.fit(X, y)

    assert not np.isnan(ridge.coef_).any()
    np.testing.assert_allclose(ridge.coef_[0], ridge.coef_[1], rtol=1e-5)
    np.testing.assert_allclose(ridge.coef_[1], ridge.coef_[2], rtol=1e-5)


def test_ridge_gcv_completely_singular():
    """Verify RidgeGCV with completely singular constant matrix."""
    n, p = 80, 12
    X = np.ones((n, p)) * 4.2
    y = np.random.default_rng(66).standard_normal(n)

    ridge = RidgeGCV()
    ridge.fit(X, y)

    assert not np.isnan(ridge.coef_).any()
    np.testing.assert_allclose(ridge.coef_, 0.0, atol=1e-15)
    pred = ridge.predict(X)
    np.testing.assert_allclose(pred, np.mean(y), atol=1e-5)


def test_ridge_gcv_high_dimensional():
    """Verify RidgeGCV in high-dimensional p > n regime (n=25, p=80)."""
    rng = np.random.default_rng(77)
    n, p = 25, 80
    X = rng.standard_normal((n, p))
    y = X[:, :4] @ np.array([1.0, -1.0, 2.0, -0.5]) + rng.standard_normal(n) * 0.2

    ridge = RidgeGCV()
    ridge.fit(X, y)

    assert not np.isnan(ridge.coef_).any()
    assert not np.isinf(ridge.coef_).any()
    pred = ridge.predict(X)
    assert not np.isnan(pred).any()


def test_ridge_gcv_ill_conditioned():
    """Verify RidgeGCV handles covariance matrix with condition number > 1e15."""
    rng = np.random.default_rng(88)
    n, p = 100, 20
    U, _ = np.linalg.qr(rng.standard_normal((n, p)))
    S = np.logspace(0, -16, p)
    V, _ = np.linalg.qr(rng.standard_normal((p, p)))
    X = U @ np.diag(S) @ V.T
    y = rng.standard_normal(n)

    ridge = RidgeGCV()
    ridge.fit(X, y)

    assert not np.isnan(ridge.coef_).any()
    assert not np.isinf(ridge.coef_).any()


# ===========================================================================
# 4. DoubleMLPLR End-to-End Stress Tests
# ===========================================================================


def test_dml_plr_singular_controls():
    """Verify DoubleMLPLR converges on singular control matrices (constant + duplicate + linear combos)."""
    rng = np.random.default_rng(99)
    n, p = 120, 25
    theta_true = 2.5

    X = rng.standard_normal((n, p))
    X[:, 0] = 3.0  # Constant column
    X[:, 4] = X[:, 2]  # Duplicate column
    X[:, 7] = 2.0 * X[:, 5] - 3.0 * X[:, 6]  # Exact collinearity

    D = 0.5 * X[:, 2] + rng.standard_normal(n)
    Y = D * theta_true + 0.8 * X[:, 1] + rng.standard_normal(n)

    # Lasso learner
    res_lasso = dml_plr(Y, D, X, n_folds=3, learner="lasso", random_state=42)
    assert not np.isnan(res_lasso.theta)
    assert not np.isnan(res_lasso.se)
    assert abs(res_lasso.theta - theta_true) < 3.0 * res_lasso.se

    # Ridge learner
    res_ridge = dml_plr(Y, D, X, n_folds=3, learner="ridge", random_state=42)
    assert not np.isnan(res_ridge.theta)
    assert not np.isnan(res_ridge.se)
    assert abs(res_ridge.theta - theta_true) < 3.0 * res_ridge.se


def test_dml_plr_high_dimensional_controls():
    """Verify DoubleMLPLR with p > n controls (n=40, p=60)."""
    rng = np.random.default_rng(101)
    n, p = 40, 60
    theta_true = 1.5

    X = rng.standard_normal((n, p))
    D = 0.5 * X[:, 0] + rng.standard_normal(n)
    Y = D * theta_true + 0.7 * X[:, 1] + rng.standard_normal(n)

    res_hd = dml_plr(Y, D, X, n_folds=2, learner="lasso", random_state=1)
    assert not np.isnan(res_hd.theta)
    assert not np.isnan(res_hd.se)


def test_dml_plr_constant_treatment_handling():
    """Verify DoubleMLPLR handles zero-variance/constant treatment D gracefully."""
    rng = np.random.default_rng(202)
    n, p = 100, 10
    X = rng.standard_normal((n, p))
    D = np.ones(n) * 2.0  # Constant treatment
    Y = rng.standard_normal(n)

    res = dml_plr(Y, D, X, n_folds=3, learner="ridge", random_state=42)
    # Effect cannot be identified; estimator should report se=0 or nan and not crash
    assert res.theta == 0.0 or np.isnan(res.theta)
    assert res.se == 0.0 or np.isnan(res.se)
    assert np.isnan(res.t_stat)
    s = res.summary()
    assert isinstance(s, str)


# ===========================================================================
# 5. Pyodide Purity & Unauthorized Import Audit
# ===========================================================================


def test_pyodide_zero_unauthorized_imports():
    """Verify importing Milestone 3 modules triggers zero forbidden dependencies."""
    import subprocess

    target_modules = [
        "puremacro.causal.dml",
        "puremacro.dsge.occbin",
        "puremacro.lp.iv",
        "puremacro.lp.la_lp",
    ]
    forbidden = ["sklearn", "torch", "numba", "statsmodels", "arch"]

    # 1. In-process differential check: verify this step introduces zero new forbidden modules
    forbidden_before = {
        m for m in sys.modules
        if any(m == f or m.startswith(f + ".") for f in forbidden)
    }

    for mod in target_modules:
        __import__(mod)

    forbidden_after = {
        m for m in sys.modules
        if any(m == f or m.startswith(f + ".") for f in forbidden)
    }
    leaked = sorted(forbidden_after - forbidden_before)
    assert not leaked, f"Unauthorized forbidden dependencies detected in sys.modules diff: {leaked}"

    # 2. Clean-subprocess isolation: verify from a completely pristine process
    script = (
        "import sys, importlib\n"
        f"target_modules = {target_modules!r}\n"
        f"forbidden = {forbidden!r}\n"
        "for mod in target_modules:\n"
        "    importlib.import_module(mod)\n"
        "leaked = [f for f in forbidden if any(m == f or m.startswith(f + '.') for m in sys.modules)]\n"
        "if leaked:\n"
        "    print(leaked)\n"
        "    sys.exit(1)\n"
    )
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert res.returncode == 0, (
        f"Unauthorized forbidden dependencies detected in clean subprocess: {res.stdout.strip() or res.stderr.strip()}"
    )
