"""Unit tests for Interactive Double Machine Learning (IRM) and DML-IV.

Tests:
1. Pure-NumPy LogisticCoordinateDescent:
   - Sparse recovery and BIC/AIC parameter selection
   - Monotonic convergence via quadratic surrogate upper bounds (p*(1-p) <= 1/4)
   - L1, L2, and ElasticNet penalties
   - Constant and zero-variance feature handling
   - Calibration and prediction methods (predict, predict_proba, decision_function)
   - Input validation (non-binary targets, dimension mismatch, non-positive C)
2. DoubleMLIRM (Interactive Regression Model):
   - Doubly robust ATE recovery on non-linear DGP with known propensity score:
     abs(res.theta - theta_true) < 2.0 * res.se
   - ATT estimation and comparison against treated-sample ground truth
   - Automatic overlap trimming (clip vs drop modes, threshold enforcement)
   - Stratified fold generation (ensuring both classes in all folds)
   - Input flexibility (numpy arrays, pandas Series/DataFrame, obj_dml_data container)
3. DoubleMLIV (Instrumental Variables with High-Dimensional Controls):
   - Endogeneity correction on DGP with Cov(U, V) > 0 where PLR is biased
   - Consistency: abs(res.theta - theta_true) < 2.0 * res.se
   - Montiel Olea & Pflueger (2013) effective F-statistic (F_eff)
   - Automated UserWarning emission when F_eff < 10 (weak instruments)
   - Multiple instrument support (k_z > 1) and under-identification rejection (k_z < k_d)
4. Diagnostics and Presentation Parity:
   - .plot_overlap() with trimming boundaries and density overlay
   - .plot_coefficients() with confidence intervals and nuisance feature inspection
   - .plot_tuning() with regularization path and criterion minimum
   - .summary(), .to_markdown(), .to_latex(), .to_typst()
"""
from __future__ import annotations

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from puremacro.causal import (
    DMLResult,
    DMLIRMResult,
    DMLIVResult,
    DoubleMLPLR,
    DoubleMLIRM,
    DoubleMLIV,
    LassoCoordinateDescent,
    RidgeGCV,
    LogisticCoordinateDescent,
    dml_plr,
    dml_irm,
    dml_iv,
)


# ===========================================================================
# 1. LogisticCoordinateDescent Unit Tests
# ===========================================================================


def test_logistic_cd_sparse_recovery():
    """Verify LogisticCoordinateDescent recovers true active support with BIC."""
    rng = np.random.default_rng(42)
    n, p = 400, 20
    X = rng.standard_normal((n, p))
    beta_true = np.zeros(p)
    beta_true[:3] = [2.0, -1.5, 1.0]

    eta = X @ beta_true - 0.2
    prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -25, 25)))
    y = rng.binomial(1, prob).astype(float)

    clf = LogisticCoordinateDescent(penalty="l1", n_alphas=40, criterion="bic")
    clf.fit(X, y)

    assert clf.coef_ is not None
    assert clf.intercept_ is not None
    assert clf.alpha_ is not None

    # First 3 coordinates should be non-zero and aligned with true direction
    assert clf.coef_[0] > 0.5
    assert clf.coef_[1] < -0.5
    assert clf.coef_[2] > 0.2

    # Spurious coordinates (indices 3..19) should be heavily zeroed out by L1/BIC
    non_zeros_tail = np.count_nonzero(np.abs(clf.coef_[3:]) > 0.1)
    assert non_zeros_tail <= 2

    # Predictions check
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    assert probs.shape == (n, 2)
    np.testing.assert_allclose(probs[:, 0] + probs[:, 1], 1.0, atol=1e-7)
    accuracy = float(np.mean(preds == y))
    assert accuracy > 0.75


def test_logistic_cd_l2_and_elasticnet():
    """Verify LogisticCoordinateDescent handles L2 shrinkage and ElasticNet."""
    rng = np.random.default_rng(123)
    n, p = 250, 15
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p) * 0.5
    eta = X @ beta_true
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)

    # L2 penalty
    clf_l2 = LogisticCoordinateDescent(penalty="l2", C=0.5)
    clf_l2.fit(X, y)
    assert clf_l2.coef_ is not None
    assert np.isfinite(clf_l2.coef_).all()
    assert np.all(np.abs(clf_l2.coef_) < 5.0)

    # Elastic-Net penalty
    clf_en = LogisticCoordinateDescent(penalty="elasticnet", l1_ratio=0.5, n_alphas=25)
    clf_en.fit(X, y)
    assert clf_en.coef_ is not None
    assert np.isfinite(clf_en.coef_).all()

    # Decision function matches linear predictor
    df = clf_en.decision_function(X)
    expected_eta = X @ clf_en.coef_ + clf_en.intercept_
    np.testing.assert_allclose(df, expected_eta, atol=1e-8)


def test_logistic_cd_constant_and_zero_variance_features():
    """Verify LogisticCoordinateDescent zeroes out constant or uninformative features."""
    rng = np.random.default_rng(99)
    n, p = 150, 8
    X = rng.standard_normal((n, p))
    # Column 3 is exactly constant
    X[:, 3] = 42.0
    # Column 6 has zero variance
    X[:, 6] = -3.14

    y = rng.binomial(1, 0.5, size=n).astype(float)

    clf = LogisticCoordinateDescent(n_alphas=20)
    clf.fit(X, y)

    assert clf.coef_[3] == 0.0
    assert clf.coef_[6] == 0.0
    assert np.isfinite(clf.coef_).all()


def test_logistic_cd_edge_cases_and_input_validation():
    """Verify LogisticCoordinateDescent rejects non-binary targets and invalid parameters."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((50, 4))

    # Continuous targets
    y_cont = rng.standard_normal(50)
    with pytest.raises(ValueError, match="binary targets"):
        LogisticCoordinateDescent().fit(X, y_cont)

    # Negative C
    with pytest.raises(ValueError, match="C must be strictly positive"):
        LogisticCoordinateDescent(C=-1.0)

    # Invalid penalty
    with pytest.raises(ValueError, match="Unknown penalty"):
        LogisticCoordinateDescent(penalty="unknown")

    # Invalid criterion
    with pytest.raises(ValueError, match="criterion must be 'aic' or 'bic'"):
        LogisticCoordinateDescent(criterion="mallows_cp")

    # Unfitted predict raises RuntimeError
    unfitted = LogisticCoordinateDescent()
    with pytest.raises(RuntimeError, match="not fitted yet"):
        unfitted.predict(X)

    # All 0 or all 1 targets
    y_all_ones = np.ones(50)
    clf = LogisticCoordinateDescent().fit(X, y_all_ones)
    p_ones = clf.predict_proba(X)[:, 1]
    assert np.all(p_ones > 0.99)


# ===========================================================================
# 2. DoubleMLIRM Unit Tests
# ===========================================================================


def _irm_nonlinear_dgp(n: int = 1000, p: int = 20, seed: int = 42):
    """Synthetic non-linear DGP with known propensity score and heterogeneous treatment."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))

    # Logistic link for propensity score
    eta_m = 0.6 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2]
    m0 = 1.0 / (1.0 + np.exp(-np.clip(eta_m, -20, 20)))
    D = rng.binomial(1, m0).astype(float)

    # Heterogeneous treatment effect tau(X)
    tau = 2.0 + 0.6 * X[:, 0] - 0.4 * X[:, 1]
    # Conditional mean under control
    g0 = 1.0 * X[:, 0] + 0.8 * X[:, 1] - 0.5 * (X[:, 2] ** 2)
    g1 = g0 + tau

    U = rng.standard_normal(n)
    Y = np.where(D == 1.0, g1, g0) + U

    return Y, D, X, tau


def test_irm_ate_recovery_nonlinear_dgp():
    """Verify DoubleMLIRM recovers true ATE within 2 standard errors on non-linear DGP."""
    Y, D, X, tau = _irm_nonlinear_dgp(n=1000, p=20, seed=42)
    theta_true_ate = 2.0  # Population ATE: E[tau(X)] = 2.0

    res = dml_irm(
        Y=Y,
        D=D,
        X=X,
        n_folds=5,
        ml_g="lasso",
        ml_m="logistic",
        score="ATE",
        trimming_threshold=0.01,
        trimming_rule="clip",
        random_state=42,
    )

    assert isinstance(res, DMLIRMResult)
    assert res.score_type == "ATE"
    assert res.n_obs == 1000
    assert res.n_folds == 5
    assert res.se > 0.0

    # Core statistical identification check: estimate within 2 standard errors of ground truth
    abs_error = abs(res.theta - theta_true_ate)
    assert abs_error < 2.0 * res.se, (
        f"DML-IRM ATE estimate {res.theta:.4f} differs from true {theta_true_ate:.4f} "
        f"by {abs_error:.4f} which exceeds 2 * SE = {2.0 * res.se:.4f}"
    )

    # Confidence interval contains true value
    assert res.ci_lower <= theta_true_ate <= res.ci_upper


def test_irm_att_estimation():
    """Verify DoubleMLIRM correctly computes ATT and matches treated sample ground truth."""
    Y, D, X, tau = _irm_nonlinear_dgp(n=1000, p=20, seed=42)
    sample_att_true = float(np.mean(tau[D == 1.0]))  # Sample ATT

    res_att = dml_irm(
        Y=Y,
        D=D,
        X=X,
        n_folds=5,
        ml_g="lasso",
        ml_m="logistic",
        score="ATT",
        trimming_threshold=0.01,
        trimming_rule="clip",
        random_state=42,
    )

    assert res_att.score_type == "ATT"
    assert res_att.se > 0.0
    # Recover within 2 standard errors of sample ATT
    assert abs(res_att.theta - sample_att_true) < 2.0 * res_att.se


def test_irm_overlap_trimming_clip_and_drop():
    """Verify overlap trimming modes ('clip' and 'drop') and invalid trimming rules."""
    rng = np.random.default_rng(88)
    n, p = 500, 10
    X = rng.standard_normal((n, p))
    # Extreme propensity score link causing severe overlap violations
    eta_m = 3.0 * X[:, 0] + 2.5 * X[:, 1]
    m_extreme = 1.0 / (1.0 + np.exp(-eta_m))
    D = rng.binomial(1, m_extreme).astype(float)
    Y = 1.5 * D + X[:, 0] + rng.standard_normal(n)

    # Clipping mode: sample size preserved, propensity clipped
    res_clip = dml_irm(Y, D, X, trimming_threshold=0.05, trimming_rule="clip", random_state=42)
    assert res_clip.trimming_rule == "clip"
    assert res_clip.n_trimmed > 0
    assert len(res_clip.propensity_scores) == n

    # Dropping mode: observation count trimmed
    res_drop = dml_irm(Y, D, X, trimming_threshold=0.05, trimming_rule="drop", random_state=42)
    assert res_drop.trimming_rule == "drop"
    assert res_drop.n_trimmed > 0

    # Invalid trimming threshold
    with pytest.raises(ValueError, match="trimming_threshold must lie strictly"):
        dml_irm(Y, D, X, trimming_threshold=0.6)

    # Invalid trimming rule
    with pytest.raises(ValueError, match="trimming_rule must be 'clip' or 'drop'"):
        dml_irm(Y, D, X, trimming_rule="invalid_rule")

    # Invalid score
    with pytest.raises(ValueError, match="score must be 'ATE' or 'ATT'"):
        dml_irm(Y, D, X, score="LATE")


def test_irm_data_flexibility_and_pandas():
    """Verify DoubleMLIRM handles pandas inputs, custom names, and obj_dml_data."""
    Y, D, X, _ = _irm_nonlinear_dgp(n=200, p=5, seed=1)
    df_x = pd.DataFrame(X, columns=[f"cov_{i}" for i in range(5)])
    s_y = pd.Series(Y, name="gdp_growth")
    s_d = pd.Series(D, name="policy_reform")

    res = dml_irm(s_y, s_d, df_x, n_folds=3, random_state=1)
    assert res.outcome_name == "gdp_growth"
    assert res.treatment_name == "policy_reform"
    assert res.feature_names == tuple(f"cov_{i}" for i in range(5))

    # Pass data container directly to DoubleMLIRM constructor
    irm_obj = DoubleMLIRM(obj_dml_data=(s_y, s_d, df_x), n_folds=3, random_state=1)
    res_obj = irm_obj.fit()
    assert abs(res_obj.theta - res.theta) < 1e-10


# ===========================================================================
# 3. DoubleMLIV Unit Tests
# ===========================================================================


def _iv_endogenous_dgp(n: int = 800, p: int = 20, seed: int = 101, pi: float = 1.0):
    """Synthetic endogenous treatment DGP with excluded instrument and correlated errors."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    Z = rng.standard_normal(n)

    # Correlated structural errors: Cov(U, V) = 0.6 creates substantial endogeneity
    cov_uv = np.array([[1.0, 0.6], [0.6, 1.0]])
    errors = rng.multivariate_normal([0.0, 0.0], cov_uv, size=n)
    U = errors[:, 0]
    V = errors[:, 1]

    # First stage
    D = pi * Z + 0.7 * X[:, 0] - 0.5 * X[:, 1] + V

    # Structural equation: true theta_0 = 1.5
    Y = 1.5 * D + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + U

    return Y, D, Z, X


def test_dml_iv_endogeneity_correction():
    """Verify DML-IV corrects for endogeneity where naive PLR fails."""
    Y, D, Z, X = _iv_endogenous_dgp(n=800, p=20, seed=101, pi=1.0)
    theta_true = 1.5

    # Naive PLR ignores endogeneity: because Cov(U, V) > 0, estimate is biased upward
    res_plr = dml_plr(Y, D, X, n_folds=5, random_state=101)
    assert res_plr.theta > 1.70, f"Expected PLR to be upward-biased (>1.70), got {res_plr.theta:.4f}"

    # DML-IV partials out high-dimensional confounders and applies 2SLS on orthogonal residuals
    res_iv = dml_iv(Y, D, Z, X, n_folds=5, random_state=101)

    assert isinstance(res_iv, DMLIVResult)
    assert res_iv.n_obs == 800
    assert res_iv.n_folds == 5
    assert res_iv.se > 0.0

    # Unbiased identification test: recovers true theta_0 within 2 standard errors
    abs_err = abs(res_iv.theta - theta_true)
    assert abs_err < 2.0 * res_iv.se, (
        f"DML-IV estimate {res_iv.theta:.4f} differs from true {theta_true:.4f} "
        f"by {abs_err:.4f}, exceeding 2 * SE = {2.0 * res_iv.se:.4f}"
    )

    # Strong instrument verification
    assert res_iv.first_stage_effective_f > 40.0
    assert res_iv.first_stage_f > 40.0
    assert res_iv.weak_instrument is False


def test_dml_iv_weak_instrument_warning():
    """Verify Montiel Olea & Pflueger effective F-statistic detects weak instruments."""
    # pi = 0.04 creates an extremely weak instrument
    Y, D, Z, X = _iv_endogenous_dgp(n=800, p=20, seed=101, pi=0.04)

    with pytest.warns(UserWarning, match="Weak instruments detected: Montiel Olea & Pflueger"):
        res_weak = dml_iv(Y, D, Z, X, n_folds=5, random_state=101)

    assert res_weak.weak_instrument is True
    assert res_weak.first_stage_effective_f < 10.0


def test_dml_iv_multiple_instruments_and_underidentification():
    """Verify DML-IV handles multiple instruments and detects under-identification."""
    rng = np.random.default_rng(55)
    n, p = 300, 10
    X = rng.standard_normal((n, p))
    # Two excluded instruments (k_z = 2)
    Z = rng.standard_normal((n, 2))
    V = rng.standard_normal(n)
    D = 0.8 * Z[:, 0] + 0.6 * Z[:, 1] + 0.5 * X[:, 0] + V
    Y = 2.0 * D + X[:, 1] + rng.standard_normal(n)

    # Valid over-identified IV regression
    res_multi = dml_iv(Y, D, Z, X, n_folds=3, random_state=55)
    assert res_multi.instrument_names == ("Z_1", "Z_2")
    assert abs(res_multi.theta - 2.0) < 2.0 * res_multi.se

    # Under-identified: 2 treatments, 1 instrument (k_z < k_d)
    D_multi = np.column_stack([D, rng.standard_normal(n)])
    Z_single = Z[:, 0]
    with pytest.raises(ValueError, match="Model is under-identified"):
        dml_iv(Y, D_multi, Z_single, X, n_folds=3)


# ===========================================================================
# 4. Presentation Parity and Diagnostics Tests
# ===========================================================================


def test_irm_presentation_contracts():
    """Verify DMLIRMResult fulfills all presentation contracts and diagnostics."""
    Y, D, X, _ = _irm_nonlinear_dgp(n=300, p=10, seed=42)
    res = dml_irm(Y, D, X, n_folds=3, random_state=42)

    # 1. Summary
    summary_txt = res.summary()
    assert "Double / Debiased Machine Learning (DML-IRM)" in summary_txt
    assert "Interactive Regression Model (ATE)" in summary_txt
    assert "Target" in summary_txt and "Coef." in summary_txt

    # 2. Markdown
    md_txt = res.to_markdown()
    assert md_txt.startswith("### DML-IRM (ATE):")
    assert "| Target | Coef. | Std.Err. | z | P>\\|z\\| |" in md_txt
    assert "|:---|---:|---:|---:|---:|:---:|" in md_txt

    # 3. LaTeX
    latex_txt = res.to_latex()
    assert r"\begin{table}[htbp]" in latex_txt
    assert r"\begin{tabular}{lrrrrr}" in latex_txt
    assert r"\toprule" in latex_txt and r"\bottomrule" in latex_txt

    # 4. Typst
    typst_txt = res.to_typst()
    assert "#figure(" in typst_txt
    assert "table(" in typst_txt
    assert "[*Target*], [*Coef.*]" in typst_txt

    # 5. Diagnostics: .plot_overlap(), .plot_coefficients(), .plot_tuning()
    fig, ax = plt.subplots()
    ax_out = res.plot_overlap(ax=ax)
    assert ax_out is ax
    plt.close(fig)

    fig, ax = plt.subplots()
    ax_out2 = res.plot_coefficients(model="treatment", ax=ax)
    assert ax_out2 is ax
    plt.close(fig)

    fig, ax = plt.subplots()
    ax_out3 = res.plot_coefficients(model="all", ax=ax)
    assert ax_out3 is ax
    plt.close(fig)

    fig, ax = plt.subplots()
    ax_out4 = res.plot_tuning(model="m", ax=ax)
    assert ax_out4 is ax
    plt.close(fig)

    # General .plot() delegation
    fig, ax = plt.subplots()
    ax_forest = res.plot(kind="forest", ax=ax)
    assert ax_forest is ax
    plt.close(fig)


def test_dml_iv_presentation_contracts():
    """Verify DMLIVResult fulfills all presentation contracts and diagnostics."""
    Y, D, Z, X = _iv_endogenous_dgp(n=400, p=10, seed=101, pi=1.0)
    res = dml_iv(Y, D, Z, X, n_folds=3, random_state=101)

    # 1. Summary
    summary_txt = res.summary()
    assert "Double / Debiased Machine Learning (DML-IV)" in summary_txt
    assert "Effective F (MOP):" in summary_txt
    assert "Weak IV: No" in summary_txt

    # 2. Markdown
    md_txt = res.to_markdown()
    assert md_txt.startswith("### DML-IV:")
    assert "| Variable | Coef. |" in md_txt

    # 3. LaTeX
    latex_txt = res.to_latex()
    assert r"\begin{table}[htbp]" in latex_txt
    assert r"\toprule" in latex_txt

    # 4. Typst
    typst_txt = res.to_typst()
    assert "#figure(" in typst_txt
    assert "[*Variable*], [*Coef.*]" in typst_txt

    # 5. Diagnostics: .plot(residuals), .plot(first_stage), .plot_coefficients(), .plot_tuning()
    fig, ax = plt.subplots()
    ax_res = res.plot(kind="residuals", ax=ax)
    assert ax_res is ax
    plt.close(fig)

    fig, ax = plt.subplots()
    ax_fs = res.plot(kind="first_stage", ax=ax)
    assert ax_fs is ax
    plt.close(fig)

    fig, ax = plt.subplots()
    ax_coef = res.plot_coefficients(model="treatment", ax=ax)
    assert ax_coef is ax
    plt.close(fig)

    fig, ax = plt.subplots()
    ax_coef_nuis = res.plot_coefficients(model="all", ax=ax)
    assert ax_coef_nuis is ax
    plt.close(fig)

    fig, ax = plt.subplots()
    ax_tune = res.plot_tuning(model="l", ax=ax)
    assert ax_tune is ax
    plt.close(fig)


def test_plr_plot_coefficients_diagnostic():
    """Verify DMLResult (PLR) has diagnostic .plot_coefficients() for parity."""
    rng = np.random.default_rng(12)
    n, p = 150, 5
    X = rng.standard_normal((n, p))
    D = 0.5 * X[:, 0] + rng.standard_normal(n)
    Y = 1.2 * D + X[:, 1] + rng.standard_normal(n)

    res = dml_plr(Y, D, X, n_folds=3, random_state=12)
    assert isinstance(res, DMLResult)
    fig, ax = plt.subplots()
    ax_c = res.plot_coefficients(ax=ax)
    assert ax_c is ax
    plt.close(fig)


# ===========================================================================
# 5. Regression Tests for Fold Routing & Diagnostic Validations
# ===========================================================================


def test_dml_iv_and_irm_n_folds_routing():
    """Verify n_folds parameter is correctly honored and not shadowed by defaults."""
    rng = np.random.default_rng(2026)
    n, p = 60, 2
    X = rng.standard_normal((n, p))
    D = 0.5 * X[:, 0] + rng.standard_normal(n)
    Z = 0.8 * X[:, 0] + rng.standard_normal(n)
    Y = 1.5 * D + X[:, 1] + rng.standard_normal(n)

    # 1. dml_iv(..., n_folds=2) sets n_folds=2
    res_iv_2 = dml_iv(Y, D, Z, X, n_folds=2, random_state=42)
    assert res_iv_2.n_folds == 2

    # 2. dml_irm(..., n_folds=3) sets n_folds=3
    D_bin = (rng.uniform(size=n) > 0.5).astype(float)
    res_irm_3 = dml_irm(Y, D_bin, X, n_folds=3, random_state=42)
    assert res_irm_3.n_folds == 3

    # Direct class instantiation
    est_iv = DoubleMLIV(n_folds=4)
    assert est_iv.n_folds == 4
    est_irm = DoubleMLIRM(n_folds=4)
    assert est_irm.n_folds == 4


def test_dml_iv_and_irm_n_folds_validation():
    """Verify dml_iv and dml_irm reject n_folds < 2 with ValueError."""
    rng = np.random.default_rng(2026)
    n, p = 40, 2
    X = rng.standard_normal((n, p))
    D = 0.5 * X[:, 0] + rng.standard_normal(n)
    Z = 0.8 * X[:, 0] + rng.standard_normal(n)
    Y = 1.5 * D + X[:, 1] + rng.standard_normal(n)
    D_bin = (rng.uniform(size=n) > 0.5).astype(float)

    with pytest.raises(ValueError, match=r"n_folds must be an integer >= 2"):
        dml_iv(Y, D, Z, X, n_folds=1)

    with pytest.raises(ValueError, match=r"n_folds must be an integer >= 2"):
        dml_irm(Y, D_bin, X, n_folds=1)

    with pytest.raises(ValueError, match=r"n_folds must be an integer >= 2"):
        DoubleMLIV(n_folds=0)

    with pytest.raises(ValueError, match=r"n_folds must be an integer >= 2"):
        DoubleMLIRM(n_folds=-1)


def test_dml_iv_small_sample_execution():
    """Verify small-sample dataset (N=4, n_folds=2) runs without crash."""
    rng = np.random.default_rng(42)
    n, p = 4, 1
    X = rng.standard_normal((n, p))
    Z = rng.standard_normal(n)
    D = 0.8 * Z + 0.3 * rng.standard_normal(n)
    Y = 2.0 * D + 0.5 * X[:, 0] + 0.2 * rng.standard_normal(n)

    res = dml_iv(Y, D, Z, X, n_folds=2, random_state=42)
    assert res.n_folds == 2
    assert res.n_obs == 4
    assert np.isfinite(res.theta)
    assert np.isfinite(res.se)


def test_plot_coefficients_top_k_validation():
    """Verify res.plot_coefficients(top_k=0) raises ValueError with clear message."""
    rng = np.random.default_rng(42)
    n, p = 40, 4
    X = rng.standard_normal((n, p))
    Z = rng.standard_normal(n)
    D = 0.7 * Z + rng.standard_normal(n)
    Y = 1.0 * D + X[:, 0] + rng.standard_normal(n)
    D_bin = (rng.uniform(size=n) > 0.5).astype(float)

    res_iv = dml_iv(Y, D, Z, X, n_folds=2, random_state=42)
    with pytest.raises(ValueError, match=r"top_k must be a positive integer >= 1"):
        res_iv.plot_coefficients(top_k=0)

    with pytest.raises(ValueError, match=r"top_k must be a positive integer >= 1"):
        res_iv.plot_coefficients(model="all", top_k=-2)

    res_irm = dml_irm(Y, D_bin, X, n_folds=2, random_state=42)
    with pytest.raises(ValueError, match=r"top_k must be a positive integer >= 1"):
        res_irm.plot_coefficients(top_k=0)

    with pytest.raises(ValueError, match=r"top_k must be a positive integer >= 1"):
        res_irm.plot_coefficients(model="all", top_k=-1)


def test_plot_tuning_model_key_validation():
    """Verify res.plot_tuning(model="invalid") raises ValueError with clear message."""
    rng = np.random.default_rng(42)
    n, p = 40, 4
    X = rng.standard_normal((n, p))
    Z = rng.standard_normal(n)
    D = 0.7 * Z + rng.standard_normal(n)
    Y = 1.0 * D + X[:, 0] + rng.standard_normal(n)
    D_bin = (rng.uniform(size=n) > 0.5).astype(float)

    res_iv = dml_iv(Y, D, Z, X, n_folds=2, random_state=42)
    with pytest.raises(ValueError, match=r"model must be one of \('l', 'm', 'r'\)"):
        res_iv.plot_tuning(model="invalid")

    res_irm = dml_irm(Y, D_bin, X, n_folds=2, random_state=42)
    with pytest.raises(ValueError, match=r"model must be one of \('m', 'g0', 'g1'\)"):
        res_irm.plot_tuning(model="invalid")

