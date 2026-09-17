"""Adversarial Empirical Stress Tests for Milestone 2: IRM and Logistic Coordinate Descent.

Milestone 2 Challenger 1 Suite targeting:
1. DoubleMLIRM Empirical Root-N Consistency (Task 2 & 3):
   - 20 randomized DGP trials with synthetic non-linear propensity scores and heterogeneous treatment.
   - Empirical verification: |theta_hat - theta_true| < 2.0 * se for 100% of trials (20/20).
   - Root-N standard error scaling verification across sample sizes N in [500, 1000, 2000, 4000].
2. Overlap Trimming Stress Testing ('clip' vs 'drop') (Task 4):
   - Panels with severe overlap violations (propensity scores spanning [1e-6, 1 - 1e-6]).
   - Direct comparison of 'clip' vs 'drop' across multiple trimming thresholds (0.01, 0.05, 0.10).
   - Validation of variance reduction, trimmed count parity, and boundary guards when common support vanishes.
3. LogisticCoordinateDescent on Ill-Conditioned and Sparse Data (Task 5):
   - Near-singular Gram matrices with condition numbers exceeding 1e10.
   - High-dimensional sparse regime (p >> n, e.g. n=60, p=300) with L1/BIC sparse support recovery.
   - Complete linear separation where unregularized MLE diverges to infinity.
   - Highly sparse design matrices (98% zero entries) and wild scale disparities (1e7 vs 1e-7).
4. Heterogeneous ATT vs ATE Identification:
   - Empirical validation that DoubleMLIRM separates population ATE from treated-group ATT.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.causal import (
    DoubleMLIRM,
    dml_irm,
    DMLIRMResult,
    LogisticCoordinateDescent,
)


# ===========================================================================
# 1. Empirical Consistency Across 20 Randomized Trials & Root-N Scaling
# ===========================================================================


def _generate_nonlinear_dgp(n: int = 1000, p: int = 20, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Synthetic non-linear DGP with known true propensity score and conditional means."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))

    # Logistic link for propensity score m_0(X)
    eta_m = 0.6 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2]
    m0 = 1.0 / (1.0 + np.exp(-np.clip(eta_m, -20, 20)))
    D = rng.binomial(1, m0).astype(float)

    # Heterogeneous treatment effect tau(X) with population mean E[tau(X)] = 2.0
    tau = 2.0 + 0.6 * X[:, 0] - 0.4 * X[:, 1]

    # Non-linear conditional mean under control
    g0 = 1.0 * X[:, 0] + 0.8 * X[:, 1] - 0.5 * (X[:, 2] ** 2)
    g1 = g0 + tau

    U = rng.standard_normal(n)
    Y = np.where(D == 1.0, g1, g0) + U

    return Y, D, X, 2.0


def test_irm_empirical_ate_consistency_20_trials():
    """Empirically verify across 20 randomized trials that DoubleMLIRM matches true ATE within 2 SE."""
    theta_true = 2.0
    n_trials = 20
    successes = 0
    estimates = []
    standard_errors = []

    for trial_idx in range(n_trials):
        # 20 randomized DGP trial realizations
        Y, D, X, _ = _generate_nonlinear_dgp(n=1000, p=20, seed=trial_idx)
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
        assert res.se > 0.0
        abs_err = abs(res.theta - theta_true)
        threshold_2se = 2.0 * res.se

        if abs_err < threshold_2se:
            successes += 1

        estimates.append(res.theta)
        standard_errors.append(res.se)

    # Empirically verify across 20 trials that estimates match synthetic ground truth within 2 SE
    assert successes == n_trials, (
        f"Only {successes}/{n_trials} trials satisfied |theta_hat - theta_true| < 2.0 * SE."
    )

    # Overall empirical bias across 20 trials should be negligible (|bias| < 0.03)
    mean_theta = float(np.mean(estimates))
    bias = abs(mean_theta - theta_true)
    assert bias < 0.03, f"Empirical bias {bias:.4f} exceeds 0.03 threshold."


def test_irm_root_n_standard_error_scaling():
    """Verify standard error scales as 1/sqrt(N) (root-N consistency) across sample sizes."""
    sample_sizes = [500, 1000, 2000, 4000]
    standard_errors = []
    products = []

    for n in sample_sizes:
        Y, D, X, _ = _generate_nonlinear_dgp(n=n, p=20, seed=42)
        res = dml_irm(Y, D, X, n_folds=5, random_state=42)
        se = res.se
        standard_errors.append(se)
        products.append(se * np.sqrt(n))

    # SE * sqrt(N) should be stable (~ 2.6) across all sample sizes
    prod_arr = np.array(products)
    relative_variation = np.std(prod_arr) / np.mean(prod_arr)
    assert relative_variation < 0.05, (
        f"SE * sqrt(N) exhibits high variation: {products}, relative variation: {relative_variation:.4f}"
    )

    # Quadrupling sample size from 500 to 2000 should cut SE in half (ratio ~ 2.0)
    ratio = standard_errors[0] / standard_errors[2]
    np.testing.assert_allclose(ratio, 2.0, rtol=0.10)


# ===========================================================================
# 2. Overlap Trimming: 'clip' vs 'drop' on Extreme Panels
# ===========================================================================


def test_irm_overlap_trimming_clip_vs_drop_extreme_panels():
    """Stress-test overlap trimming on panels with extreme propensity scores [1e-6, 1 - 1e-6]."""
    rng = np.random.default_rng(42)
    n, p = 1200, 15
    X = rng.standard_normal((n, p))

    # Severe selection into treatment producing extreme propensity scores
    eta_m = 2.5 * X[:, 0] + 2.0 * X[:, 1] - 1.5 * X[:, 2]
    m0 = 1.0 / (1.0 + np.exp(-np.clip(eta_m, -25, 25)))
    D = rng.binomial(1, m0).astype(float)

    # Heterogeneous treatment effect with true population ATE = 3.0
    tau = 3.0 + 0.5 * X[:, 0]
    g0 = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * (X[:, 2] ** 2)
    g1 = g0 + tau
    Y = np.where(D == 1.0, g1, g0) + rng.standard_normal(n)

    # Verify propensity scores span extreme range
    assert m0.min() < 0.001
    assert m0.max() > 0.999

    for eps in [0.01, 0.05, 0.10]:
        res_clip = dml_irm(Y, D, X, trimming_threshold=eps, trimming_rule="clip", random_state=42)
        res_drop = dml_irm(Y, D, X, trimming_threshold=eps, trimming_rule="drop", random_state=42)

        # Both rules should flag the exact same number of extreme units
        assert res_clip.n_trimmed == res_drop.n_trimmed
        assert res_clip.n_trimmed > 0

        # Clipping preserves observation count
        assert res_clip.n_obs == n
        assert len(res_clip.propensity_scores) == n

        # Both rules recover causal effect within 2 standard errors of ground truth
        assert abs(res_clip.theta - 3.0) < 2.0 * res_clip.se
        assert abs(res_drop.theta - 3.0) < 2.0 * res_drop.se

        # Standard errors must be finite and positive
        assert np.isfinite(res_clip.se) and res_clip.se > 0.0
        assert np.isfinite(res_drop.se) and res_drop.se > 0.0


def test_irm_overlap_trimming_guards_and_extreme_boundaries():
    """Verify trimming boundary behavior when common support approaches empty set."""
    rng = np.random.default_rng(99)
    n = 100
    X = rng.standard_normal((n, 4))
    # Complete separation based on first coordinate
    D = (X[:, 0] > 0).astype(float)
    Y = 2.0 * D + rng.standard_normal(n)

    # When trimming threshold is 0.49 on separated data, common support is empty:
    # 'drop' mode must raise ValueError rather than producing NaN/inf
    with pytest.raises(ValueError, match="All observations were trimmed by overlap rule"):
        dml_irm(Y, D, X, trimming_threshold=0.49, trimming_rule="drop", random_state=42)

    # 'clip' mode handles this gracefully by clamping propensities to [0.49, 0.51]
    res_clip = dml_irm(Y, D, X, trimming_threshold=0.49, trimming_rule="clip", random_state=42)
    assert np.isfinite(res_clip.theta)
    assert np.isfinite(res_clip.se)
    assert res_clip.n_trimmed == n


# ===========================================================================
# 3. LogisticCoordinateDescent: Ill-Conditioned, Sparse, & Separated Data
# ===========================================================================


def test_logistic_cd_ill_conditioned_collinear_design():
    """Verify LogisticCoordinateDescent converges on ill-conditioned designs (kappa > 1e10)."""
    rng = np.random.default_rng(202)
    n, p = 150, 10
    X = rng.standard_normal((n, p))
    # Inject near-exact collinearity: column 1 almost equals column 0
    X[:, 1] = X[:, 0] + 1e-10 * rng.standard_normal(n)
    # Inject linear dependency: column 2 = 0.5*X3 - 0.3*X4
    X[:, 2] = 0.5 * X[:, 3] - 0.3 * X[:, 4] + 1e-9 * rng.standard_normal(n)

    _, s, _ = np.linalg.svd(X)
    condition_number = float(s.max() / s.min())
    assert condition_number > 1e9

    y = rng.binomial(1, 0.5, size=n).astype(float)

    # L1 penalty
    clf_l1 = LogisticCoordinateDescent(penalty="l1", n_alphas=20).fit(X, y)
    assert np.isfinite(clf_l1.coef_).all()
    assert np.isfinite(clf_l1.intercept_)
    prob_l1 = clf_l1.predict_proba(X)
    assert np.all((prob_l1 >= 0.0) & (prob_l1 <= 1.0))
    np.testing.assert_allclose(prob_l1[:, 0] + prob_l1[:, 1], 1.0, atol=1e-7)

    # L2 penalty
    clf_l2 = LogisticCoordinateDescent(penalty="l2", C=1.0).fit(X, y)
    assert np.isfinite(clf_l2.coef_).all()
    assert np.isfinite(clf_l2.intercept_)
    prob_l2 = clf_l2.predict_proba(X)
    assert np.all((prob_l2 >= 0.0) & (prob_l2 <= 1.0))


def test_logistic_cd_high_dimensional_sparse_recovery():
    """Verify LogisticCoordinateDescent performs sparse selection when p >> n (p=300, n=60)."""
    rng = np.random.default_rng(303)
    n, p = 60, 300
    X = rng.standard_normal((n, p))
    beta_true = np.zeros(p)
    beta_true[5] = 2.5
    beta_true[12] = -2.0
    beta_true[40] = 1.8

    eta = X @ beta_true
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-np.clip(eta, -20, 20)))).astype(float)

    clf = LogisticCoordinateDescent(penalty="l1", n_alphas=30, criterion="bic")
    clf.fit(X, y)

    assert clf.coef_ is not None
    assert np.isfinite(clf.coef_).all()

    # BIC should zero out the vast majority of the 300 features
    n_nonzero = np.count_nonzero(clf.coef_)
    assert n_nonzero <= 10, f"Expected sparse model, got {n_nonzero} non-zero features."

    # Active features should receive non-zero weights in the correct direction
    assert clf.coef_[5] > 0.1
    assert clf.coef_[12] < -0.1


def test_logistic_cd_complete_separation():
    """Verify regularized LogisticCoordinateDescent prevents divergence under complete separation."""
    rng = np.random.default_rng(404)
    n, p = 120, 5
    X = rng.standard_normal((n, p))
    # Perfect linear separation along first coordinate
    y = (X[:, 0] > 0).astype(float)

    # Unregularized MLE would diverge to +/- infinity; L1 must remain bounded
    clf_l1 = LogisticCoordinateDescent(penalty="l1", alpha=0.05).fit(X, y)
    assert np.isfinite(clf_l1.coef_).all()
    assert abs(clf_l1.coef_[0]) < 15.0

    # L2 must also remain bounded
    clf_l2 = LogisticCoordinateDescent(penalty="l2", C=0.5).fit(X, y)
    assert np.isfinite(clf_l2.coef_).all()
    assert abs(clf_l2.coef_[0]) < 15.0

    # Probabilities must be strictly bounded in [0, 1] without NaN or inf
    probs = clf_l1.predict_proba(X)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.isfinite(probs).all()


def test_logistic_cd_extreme_sparsity_and_scale_disparity():
    """Verify LogisticCoordinateDescent on 98% sparse matrices and huge dynamic scales."""
    rng = np.random.default_rng(505)
    n, p = 200, 20

    # 98% sparse matrix
    X_sparse = (rng.random((n, p)) < 0.02).astype(float)
    y_sparse = rng.binomial(1, 0.4, size=n).astype(float)

    clf_sp = LogisticCoordinateDescent(penalty="l1", n_alphas=15).fit(X_sparse, y_sparse)
    assert np.isfinite(clf_sp.coef_).all()
    assert np.isfinite(clf_sp.intercept_)

    # Extreme scale disparity (1e7 vs 1e-7)
    X_scale = rng.standard_normal((100, 4))
    X_scale[:, 0] *= 1e7
    X_scale[:, 1] *= 1e-7
    y_scale = rng.binomial(1, 0.5, size=100).astype(float)

    clf_scale = LogisticCoordinateDescent(penalty="l1", n_alphas=15).fit(X_scale, y_scale)
    assert np.isfinite(clf_scale.coef_).all()
    assert np.isfinite(clf_scale.intercept_)


# ===========================================================================
# 4. Heterogeneous ATT vs ATE Estimation
# ===========================================================================


def test_irm_heterogeneous_att_vs_ate_identification():
    """Verify DoubleMLIRM separates population ATE from treated-group ATT."""
    rng = np.random.default_rng(606)
    n, p = 1500, 15
    X = rng.standard_normal((n, p))

    # Propensity score strongly correlated with X0
    eta_m = 0.8 * X[:, 0] - 0.6 * X[:, 1]
    m0 = 1.0 / (1.0 + np.exp(-np.clip(eta_m, -20, 20)))
    D = rng.binomial(1, m0).astype(float)

    # Treatment effect is heterogeneous and positively correlated with X0
    tau = 2.0 + 1.2 * X[:, 0]
    g0 = 0.5 * X[:, 0] + 0.5 * X[:, 1]
    g1 = g0 + tau
    Y = np.where(D == 1.0, g1, g0) + rng.standard_normal(n)

    true_ate = 2.0
    true_att = float(np.mean(tau[D == 1.0]))

    # True ATT should be substantially higher than true ATE due to positive selection
    assert true_att > true_ate + 0.3

    res_ate = dml_irm(Y, D, X, score="ATE", random_state=42)
    res_att = dml_irm(Y, D, X, score="ATT", random_state=42)

    # Both estimates match their distinct ground truths within 2 standard errors
    assert abs(res_ate.theta - true_ate) < 2.0 * res_ate.se
    assert abs(res_att.theta - true_att) < 2.0 * res_att.se

    # Estimated ATT should be strictly greater than estimated ATE
    assert res_att.theta > res_ate.theta
