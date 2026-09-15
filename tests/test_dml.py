"""Unit tests for Double Machine Learning (DML) for Partially Linear Models.

Tests:
1. Pure-NumPy regularized learners (LassoCoordinateDescent, RidgeGCV): sparse
   recovery, the ``fit_intercept=False`` coordinate update (KKT optimality and an
   sklearn cross-check), ``criterion`` validation, input validation, and the
   absolute (sklearn-style) units of the ridge penalty.
2. Unbiased recovery of causal effect theta_0 on synthetic benchmark data.
3. Asymptotic standard errors: root-N scaling, and bit-level agreement of theta,
   the out-of-fold residuals, the Chernozhukov et al. (2018) plug-in variance and
   the z-interval with an independent Robinson two-step on the same folds.
4. Nominal 95% confidence interval coverage (two-sided 91-99% band over 200
   replications, which rejects a doubled/halved SE, a 90% critical value and an
   Omega without u_hat).
5. Multidimensional treatment D support, pandas inputs and constant control columns.
6. Presentation contract (.summary, .plot, .to_markdown, .to_latex, .to_typst):
   GFM-parseable Markdown and LaTeX that compiles with ``booktabs`` alone.
7. Input validation (NaN/inf, 1-D X, n_folds > N, learner instances / classes)
   and the high-dimensional ``n_train <= p + 1`` warning.
8. Zero non-Pyodide runtime dependencies (the module imports only
   numpy/scipy/pandas/matplotlib and the standard library).
"""
from __future__ import annotations

import ast
import re
import shutil
import subprocess
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from puremacro.causal import (
    DoubleMLPLR,
    DMLResult,
    LassoCoordinateDescent,
    RidgeGCV,
    dml_plr,
)


def _plr_dgp(n: int, p: int, seed: int, theta0: float = 1.3):
    """Small linear PLR design used by the validation tests."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    D = 0.6 * X[:, 0] - 0.4 * X[:, 1] + rng.standard_normal(n)
    Y = theta0 * D + 0.8 * X[:, 1] + 0.5 * X[:, 2] + rng.standard_normal(n)
    return Y, D, X


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


# ---------------------------------------------------------------------------
# 3.4.0 regression tests (adversarial pre-release review, group "causal")
# ---------------------------------------------------------------------------


def test_lasso_no_intercept_uses_column_norms():
    """``fit_intercept=False`` on columns with variance > 2 used to diverge to NaN and
    silently return ``coef_ = 0``; the exact coordinate minimiser divides by X_j'X_j / n."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 3)) * 3.0
    beta = np.array([1.0, -1.0, 0.5])
    y = X @ beta

    # Noise-free, alpha = 0: coordinate descent must reach the exact solution.
    m0 = LassoCoordinateDescent(fit_intercept=False, alpha=0.0, tol=1e-10).fit(X, y)
    assert m0.intercept_ == 0.0
    np.testing.assert_allclose(m0.coef_, beta, atol=1e-6)

    # Default BIC path: support and values recovered (was [0, 0, 0]).
    mb = LassoCoordinateDescent(fit_intercept=False).fit(X, y)
    np.testing.assert_allclose(mb.coef_, beta, atol=5e-3)

    # Noisy y, positive penalty: KKT conditions of (1/2n)||y - Xb||^2 + a||b||_1.
    yn = y + rng.standard_normal(200)
    a = 0.5
    m = LassoCoordinateDescent(fit_intercept=False, alpha=a, tol=1e-10).fit(X, yn)
    grad = X.T @ (yn - X @ m.coef_) / len(yn)
    active = m.coef_ != 0
    assert active.any()
    np.testing.assert_allclose(grad[active], a * np.sign(m.coef_[active]), atol=1e-8)
    assert np.all(np.abs(grad[~active]) <= a + 1e-8)

    # An all-zero raw column is unidentified: it stays at 0 instead of producing NaN.
    Xz = np.column_stack([X, np.zeros(200)])
    mz = LassoCoordinateDescent(fit_intercept=False, alpha=0.1).fit(Xz, y)
    assert np.isfinite(mz.coef_).all()
    assert mz.coef_[-1] == 0.0
    np.testing.assert_allclose(mz.coef_[:3], LassoCoordinateDescent(fit_intercept=False, alpha=0.1).fit(X, y).coef_)


def test_lasso_no_intercept_matches_sklearn():
    """Cross-check the raw-column solution against scikit-learn at a fixed penalty."""
    sklearn_linear = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(2)
    X = rng.standard_normal((150, 4)) * np.array([3.0, 0.2, 1.0, 5.0])
    y = X @ np.array([1.0, -2.0, 0.0, 0.3]) + rng.standard_normal(150)
    ours = LassoCoordinateDescent(fit_intercept=False, alpha=0.2, tol=1e-10).fit(X, y).coef_
    ref = sklearn_linear.Lasso(alpha=0.2, fit_intercept=False, tol=1e-12, max_iter=100_000).fit(X, y).coef_
    np.testing.assert_allclose(ours, ref, atol=1e-6)


def test_lasso_criterion_validated():
    """Unknown ``criterion`` strings used to fall through silently to BIC."""
    with pytest.raises(ValueError, match="criterion must be 'aic' or 'bic'"):
        LassoCoordinateDescent(criterion="cv")
    assert LassoCoordinateDescent(criterion="AIC").criterion == "aic"
    assert LassoCoordinateDescent().criterion == "bic"


def test_learners_reject_nonfinite_and_accept_1d_x():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 2))
    y = X[:, 0] + rng.standard_normal(60)
    Xb = X.copy()
    Xb[4, 1] = np.nan
    for cls in (LassoCoordinateDescent, RidgeGCV):
        with pytest.raises(ValueError, match="NaN or inf"):
            cls().fit(Xb, y)
        with pytest.raises(ValueError, match="NaN or inf"):
            cls().fit(X, np.where(np.arange(60) == 2, np.inf, y))
        with pytest.raises(ValueError, match="rows but y has"):
            cls().fit(X, y[:-1])
        m1 = cls().fit(X[:, 0], y)
        m2 = cls().fit(X[:, [0]], y)
        np.testing.assert_allclose(m1.coef_, m2.coef_)
        np.testing.assert_allclose(m1.predict(X[:, 0]), m2.predict(X[:, [0]]))


def test_ridge_gcv_penalty_is_absolute_on_standardized_design():
    """``alpha_`` is in ``sklearn.linear_model.Ridge`` units on the centred, unit-variance
    design: beta = (Xs'Xs + alpha I)^-1 Xs'yc, not divided by N."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((100, 4)) * np.array([1.0, 2.0, 0.5, 3.0])
    y = X @ np.array([1.0, 2.0, 3.0, 4.0]) + rng.standard_normal(100)
    a = 3.0
    m = RidgeGCV(alphas=[a]).fit(X, y)
    assert m.alpha_ == a
    Xs = (X - X.mean(0)) / X.std(0)
    yc = y - y.mean()
    beta_s = np.linalg.solve(Xs.T @ Xs + a * np.eye(4), Xs.T @ yc)
    np.testing.assert_allclose(m.coef_, beta_s / X.std(0), rtol=1e-10)
    np.testing.assert_allclose(m.intercept_, y.mean() - X.mean(0) @ m.coef_, rtol=1e-10)
    # The N-scaled reading of the old docstring is a different estimator.
    beta_n = np.linalg.solve(Xs.T @ Xs + 100 * a * np.eye(4), Xs.T @ yc)
    assert not np.allclose(m.coef_, beta_n / X.std(0), rtol=1e-3)


def test_dml_matches_manual_cross_fitting_reference():
    """Pin the estimator to an independent Robinson two-step on the same folds.

    Reconstructs the shuffled K-fold split, residualises Y and D out-of-fold with
    ``RidgeGCV``, and checks theta, the residuals, the Chernozhukov et al. (2018)
    variance ``mean(psi^2) / mean(V^2)^2 / N`` with ``psi = V (Y_tilde - V theta)``,
    the z-statistic, the p-value and the ``z_{0.975}`` interval to 1e-10. Also
    checks that cross-fitting really happened (the residuals differ from an
    in-sample fit on the full sample) and, for two treatments, the matrix form
    ``J^-1 Omega J^-1 / N``.
    """
    n, p, K, seed = 240, 8, 4, 3
    Y, D, X = _plr_dgp(n, p, 7)
    res = dml_plr(Y, D, X, n_folds=K, learner="ridge", random_state=seed)

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    ry = np.zeros(n)
    rd = np.zeros(n)
    for test_idx in np.array_split(idx, K):
        train_idx = np.setdiff1d(idx, test_idx)
        ry[test_idx] = Y[test_idx] - RidgeGCV().fit(X[train_idx], Y[train_idx]).predict(X[test_idx])
        rd[test_idx] = D[test_idx] - RidgeGCV().fit(X[train_idx], D[train_idx]).predict(X[test_idx])
    theta_ref = float(rd @ ry / (rd @ rd))
    psi = rd * (ry - rd * theta_ref)
    se_ref = float(np.sqrt(np.mean(psi ** 2) / np.mean(rd ** 2) ** 2 / n))
    z = norm.ppf(0.975)

    np.testing.assert_allclose(res.residuals_y, ry, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(res.residuals_d, rd, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(res.theta, theta_ref, rtol=1e-10)
    np.testing.assert_allclose(res.se, se_ref, rtol=1e-10)
    np.testing.assert_allclose(res.t_stat, theta_ref / se_ref, rtol=1e-10)
    np.testing.assert_allclose(res.p_value, 2.0 * norm.sf(abs(theta_ref / se_ref)), rtol=1e-8)
    np.testing.assert_allclose(
        [res.ci_lower, res.ci_upper],
        [theta_ref - z * se_ref, theta_ref + z * se_ref],
        rtol=1e-10,
    )
    assert res.n_obs == n and res.n_folds == K and res.ci_level == 0.95

    # Cross-fitting is real: out-of-fold residuals are not the in-sample ones.
    in_sample = Y - RidgeGCV().fit(X, Y).predict(X)
    assert np.max(np.abs(in_sample - res.residuals_y)) > 0.1

    # Two treatments: J^-1 Omega J^-1 / N with J = V'V/N, Omega = psi'psi/N.
    rng = np.random.default_rng(8)
    D2 = np.column_stack([D, -0.3 * X[:, 2] + rng.standard_normal(n)])
    Y2 = Y + 0.5 * D2[:, 1]
    res2 = dml_plr(Y2, D2, X, n_folds=K, learner="ridge", random_state=seed)
    V = res2.residuals_d
    theta2 = np.linalg.solve(V.T @ V, V.T @ res2.residuals_y)
    psi2 = V * (res2.residuals_y - V @ theta2)[:, None]
    J_inv = np.linalg.inv(V.T @ V / n)
    vcov = J_inv @ (psi2.T @ psi2 / n) @ J_inv / n
    np.testing.assert_allclose(res2.theta, theta2, rtol=1e-10)
    np.testing.assert_allclose(res2.se, np.sqrt(np.diag(vcov)), rtol=1e-10)


def test_dml_confidence_interval_coverage_two_sided():
    """200-replication Monte Carlo: the nominal 95% interval must cover 91-99% of the time.

    The two-sided band is what pins the variance: with this design a doubled SE
    covers 100%, a halved SE 64%, the 90% critical value 88% and an Omega built
    from V alone (no u_hat; the error sd is 2, not 1) 63%.
    """
    theta0 = 1.0
    reps = 200
    covered = 0
    ests = []
    for rep in range(reps):
        rng = np.random.default_rng(1000 + rep)
        n, p = 300, 10
        X = rng.standard_normal((n, p))
        D = 0.5 * X[:, 0] - 0.5 * X[:, 1] + rng.standard_normal(n)
        Y = theta0 * D + 0.7 * X[:, 0] + 0.4 * X[:, 2] + 2.0 * rng.standard_normal(n)
        res = dml_plr(Y, D, X, n_folds=4, learner="ridge", random_state=rep)
        ests.append(res.theta)
        covered += int(res.ci_lower <= theta0 <= res.ci_upper)
    coverage = covered / reps
    assert 0.91 <= coverage <= 0.99, f"empirical coverage {coverage:.3f} outside [0.91, 0.99]"
    assert abs(float(np.mean(ests)) - theta0) < 0.03


def test_dml_rejects_nonfinite_inputs():
    """NaN in Y used to return theta = se = nan silently; NaN in X crashed inside pinv."""
    Y, D, X = _plr_dgp(120, 5, 11)
    Yn = Y.copy()
    Yn[3] = np.nan
    with pytest.raises(ValueError, match="Y contains NaN or inf"):
        dml_plr(Yn, D, X)
    Dn = D.copy()
    Dn[3] = np.inf
    with pytest.raises(ValueError, match="D contains NaN or inf"):
        dml_plr(Y, Dn, X)
    Xn = X.copy()
    Xn[3, 1] = np.nan
    with pytest.raises(ValueError, match="X contains NaN or inf"):
        dml_plr(Y, D, Xn, learner="ridge")


def test_dml_one_dimensional_controls_and_shape_guards():
    Y, D, X = _plr_dgp(120, 5, 12)
    r1 = dml_plr(Y, D, X[:, 0], learner="ridge")
    r2 = dml_plr(Y, D, X[:, [0]], learner="ridge")
    assert r1.theta == r2.theta and r1.se == r2.se
    with pytest.raises(ValueError, match="n_folds must not exceed the number of observations"):
        dml_plr(Y[:4], D[:4], X[:4], n_folds=10)
    with pytest.raises(ValueError, match="D must be 1-D or 2-D"):
        dml_plr(Y, np.zeros((120, 1, 1)), X)
    with pytest.raises(ValueError, match="X must be 2-D"):
        dml_plr(Y, D, np.zeros((120, 2, 2)))
    # n_folds == N (leave-one-out) is allowed.
    assert np.isfinite(dml_plr(Y[:30], D[:30], X[:30], n_folds=30, learner="ridge").theta)


def test_dml_pandas_inputs_and_constant_control_column():
    Y, D, X = _plr_dgp(150, 4, 13)
    Xdf = pd.DataFrame(X, columns=list("abcd"))
    Xdf["const"] = 3.0
    for learner in ("lasso", "ridge"):
        ra = dml_plr(pd.Series(Y, name="gdp"), pd.Series(D, name="rate"), Xdf, learner=learner)
        rb = dml_plr(Y, D, X, learner=learner)
        assert ra.treatment_names == ("rate",)
        assert ra.outcome_name == "gdp"
        np.testing.assert_allclose(ra.theta, rb.theta, rtol=1e-10)
        np.testing.assert_allclose(ra.se, rb.se, rtol=1e-10)


def test_dml_learner_instance_is_cloned_and_class_name_reported():
    """A learner instance is deep-copied per nuisance model (never fitted in place),
    ``learner_kwargs`` cannot silently override it, and a class reports its own name."""
    Y, D, X = _plr_dgp(150, 5, 14)
    inst = RidgeGCV(alphas=[0.5, 5.0])
    res = dml_plr(Y, D, X, learner=inst)
    assert inst.coef_ is None and inst.alpha_ is None
    assert res.learner == "RidgeGCV"
    ref = dml_plr(Y, D, X, learner="ridge", alphas=[0.5, 5.0])
    assert res.theta == ref.theta and res.se == ref.se
    with pytest.raises(ValueError, match="learner_kwargs cannot be combined with a learner instance"):
        dml_plr(Y, D, X, learner=inst, alphas=[1.0])
    with pytest.raises(ValueError, match="learner_kwargs cannot be combined with a learner instance"):
        DoubleMLPLR(learner=inst, learner_kwargs={"alphas": [1.0]})
    assert dml_plr(Y, D, X, learner=RidgeGCV).learner == "RidgeGCV"
    assert dml_plr(Y, D, X, learner=LassoCoordinateDescent, n_alphas=10).learner == "LassoCoordinateDescent"

    class my_ols:
        def fit(self, X, y):
            Xa = np.column_stack([np.ones(len(X)), X])
            self.b = np.linalg.lstsq(Xa, y, rcond=None)[0]
            return self

        def predict(self, X):
            return np.column_stack([np.ones(len(X)), X]) @ self.b

    assert dml_plr(Y, D, X, learner=my_ols).learner == "my_ols"
    assert dml_plr(Y, D, X, learner=my_ols()).learner == "my_ols"
    with pytest.raises(TypeError, match="has no fit/predict methods"):
        dml_plr(Y, D, X, learner=lambda: object())
    with pytest.raises(TypeError, match="learner must be a string alias"):
        DoubleMLPLR(learner=3.0).fit(Y, D, X)


def test_dml_warns_when_training_fold_not_larger_than_controls():
    """In the p >= n_train regime RidgeGCV interpolates (alpha_ collapses to the grid
    minimum) and DML inference under-covers; ``fit`` must say so."""
    rng = np.random.default_rng(101)
    n, p = 40, 60
    X = rng.standard_normal((n, p))
    D = 0.5 * X[:, 0] + rng.standard_normal(n)
    Y = 1.5 * D + 0.7 * X[:, 1] + rng.standard_normal(n)
    with pytest.warns(UserWarning, match=r"n_train <= p \+ 1"):
        dml_plr(Y, D, X, n_folds=2, learner="lasso", random_state=1)

    # The centred design has rank n_train - 1, so interpolation starts at p = n_train - 1.
    y = rng.standard_normal(20)
    assert RidgeGCV().fit(X[:20, :19], y).alpha_ == pytest.approx(1e-4)
    assert RidgeGCV().fit(X[:20, :18], y).alpha_ > 1e-4

    # Boundary: n_train = p + 1 warns, n_train = p + 2 does not.
    Y2, D2, X2 = _plr_dgp(52, 24, 5)
    with pytest.warns(UserWarning, match=r"n_train <= p \+ 1"):
        dml_plr(Y2[:50], D2[:50], X2[:50], n_folds=2, learner="ridge")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dml_plr(Y2, D2, X2, n_folds=2, learner="ridge")


def _presentation_result() -> DMLResult:
    rng = np.random.default_rng(55)
    n = 200
    X = rng.standard_normal((n, 10))
    D = 0.4 * X[:, 0] + rng.standard_normal(n)
    Y = D * 2.0 + 0.5 * X[:, 1] + rng.standard_normal(n)
    return dml_plr(pd.Series(Y, name="inflation"), pd.Series(D, name="tax_cut"), X, n_folds=3, random_state=12)


def test_dml_markdown_table_is_gfm_parseable():
    """The ``P>|z|`` header used to split into 8 cells against a 6-cell delimiter row."""
    res = _presentation_result()
    md = res.to_markdown()
    table_lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    header, delim, *rows = table_lines

    def cells(line: str) -> list[str]:
        return re.split(r"(?<!\\)\|", line.strip())[1:-1]

    assert len(cells(header)) == len(cells(delim)) == 6
    assert all(len(cells(r)) == 6 for r in rows)
    assert "P>\\|z\\|" in header
    assert "| tax_cut |" in md

    markdown = pytest.importorskip("markdown")
    html = markdown.markdown(md, extensions=["tables"])
    assert "<table>" in html
    assert len(re.findall(r"<th[\s>]", html)) == 6
    assert "P&gt;|z|" in html or "P>|z|" in html


def test_dml_latex_uses_standard_packages_only():
    """``\\subcaption`` needs the subcaption package and a bare ``<`` prints as an
    inverted exclamation mark in OT1 text mode."""
    res = _presentation_result()
    assert res.p_value < 0.001
    ltx = res.to_latex()
    assert "\\subcaption" not in ltx
    assert "$<0.001$" in ltx
    assert "<0.001" not in ltx.replace("$<0.001$", "")
    assert "\\multicolumn{6}{l}{\\footnotesize Observations: 200; Folds: 3; Learner: lasso.} \\\\" in ltx
    assert "tax\\_cut" in ltx
    assert "\\caption{Double Machine Learning Estimates for inflation}" in ltx
    body = ltx.split("\\midrule")[1].split("\\bottomrule")[0]
    data_rows = [r for r in body.strip().splitlines() if r.strip()]
    assert len(data_rows) == 1
    assert all(r.count("&") == 5 for r in data_rows)
    # Special characters in names are escaped the same way as the other exporters.
    weird = DMLResult(
        theta=1.0, se=0.1, t_stat=10.0, p_value=0.0, ci_lower=0.8, ci_upper=1.2,
        n_obs=10, n_folds=2, learner="my_learner", residuals_y=np.zeros(10),
        residuals_d=np.zeros(10), treatment_names=("rate_%",), outcome_name="gdp_&_more",
    )
    wl = weird.to_latex()
    assert "rate\\_\\%" in wl and "gdp\\_\\&\\_more" in wl and "Learner: my\\_learner." in wl


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not available on system")
def test_dml_latex_compiles_with_booktabs_only(tmp_path):
    res = _presentation_result()
    (tmp_path / "table.tex").write_text(res.to_latex(), encoding="utf-8")
    (tmp_path / "doc.tex").write_text(
        "\\documentclass{article}\n\\usepackage{booktabs}\n\\begin{document}\n"
        "\\input{table.tex}\n\\end{document}\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
        cwd=tmp_path, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert not [ln for ln in proc.stdout.splitlines() if ln.startswith("!")]
    assert (tmp_path / "doc.pdf").exists()


def test_dml_module_has_no_extra_runtime_dependencies():
    """Pyodide-core contract: the module imports only numpy/scipy/pandas/matplotlib
    and the standard library (checked on the source, so lazy imports count too)."""
    import puremacro.causal.dml as dml_module

    tree = ast.parse(Path(dml_module.__file__).read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    allowed = {
        "__future__", "copy", "warnings", "dataclasses", "typing",
        "numpy", "pandas", "scipy", "matplotlib", "puremacro",
    }
    assert roots <= allowed, roots - allowed
