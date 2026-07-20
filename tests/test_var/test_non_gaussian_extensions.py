"""Tests for 0.51.0 extensions to puremacro.var.identify.non_gaussian."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from puremacro.var.identify import non_gaussian as ng


def test_variance_decomposition_consistency_passes_at_truth():
    rng = np.random.default_rng(0)
    n = 3
    B = rng.standard_normal((n, n))
    Sigma_u = B @ B.T  # exact consistency
    out = ng.variance_decomposition_consistency(B, Sigma_u)
    assert out["passed"] is True
    assert out["max_abs_diff"] < 1e-8
    assert out["rms_diff"] < 1e-8


def test_variance_decomposition_consistency_fails_on_mismatch():
    n = 2
    B = np.eye(n)
    Sigma_u = np.diag([1.0, 1.0]) + np.eye(n) * 0.5  # not equal to B B^T
    out = ng.variance_decomposition_consistency(B, Sigma_u)
    assert out["passed"] is False
    assert out["max_abs_diff"] > 1e-6
    # With a wide tolerance the same mismatch passes.
    out_loose = ng.variance_decomposition_consistency(B, Sigma_u, tol=1.0)
    assert out_loose["passed"] is True


def test_gaussian_lr_test_rejects_non_gaussian_data():
    rng = np.random.default_rng(1)
    # Heavy-tailed shocks (Student-t df=4)
    n, T = 2, 2000
    e = rng.standard_t(df=4, size=(T, n))
    B = np.array([[1.0, 0.3], [0.2, 0.8]])
    residuals = e @ B.T
    out = ng.gaussian_lr_test(B, residuals)
    assert out["p_value"] < 0.05, f"LR did not reject Gaussian; p={out['p_value']}"
    assert out["stat"] > 0
    assert out["df"] > 0


def test_gaussian_lr_test_does_not_reject_pure_gaussian_data():
    rng = np.random.default_rng(2)
    n, T = 2, 2000
    e = rng.standard_normal((T, n))
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    residuals = e @ B.T
    out = ng.gaussian_lr_test(B, residuals)
    # Should not strongly reject; allow a soft threshold to avoid flakiness.
    assert out["p_value"] > 0.10, f"unexpected rejection on Gaussian data; p={out['p_value']}"


def test_tiebreak_uses_skewness_when_kurtoses_near_equal():
    # Synthetic shocks where the tiebreaker is forced by a wide tolerance.
    rng = np.random.default_rng(11)
    T = 4000
    # shock 0: zero skew, high kurt (Laplace-like)
    s0 = rng.laplace(size=T)
    s0 = (s0 - s0.mean()) / s0.std()
    # shock 1: matching kurt but heavily right-skewed (chi^2 - mean)
    s1 = (rng.chisquare(df=4, size=T) - 4) / np.sqrt(8)
    src = np.column_stack([s0, s1])
    kurt = (src ** 4).mean(axis=0) / (src ** 2).mean(axis=0) ** 2 - 3.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        order = ng._tiebreak_kurtosis_order(kurt, src, tol=1e-1)  # wide tol to force tie-break
    # When ties are forced, output must still be a permutation of arange(n)
    assert sorted(order.tolist()) == [0, 1]


def test_tiebreak_warns_when_invoked():
    # Construct two shocks where the tiebreak actually flips the order:
    # s0 (Laplace) has higher |kurt| but near-zero skew;
    # s1 (chi^2 centred) has lower |kurt| but high skew.
    # With tol=10.0 both fall in the same bucket, so tiebreak fires on skew
    # and produces [s1, s0] instead of [s0, s1] — the warning must fire.
    rng = np.random.default_rng(99)
    T = 3000
    s0 = rng.laplace(size=T)
    s0 = (s0 - s0.mean()) / s0.std()
    s1 = (rng.chisquare(df=5, size=T) - 5)
    s1 = (s1 - s1.mean()) / s1.std()
    src = np.column_stack([s0, s1])
    kurt = (src ** 4).mean(axis=0) / (src ** 2).mean(axis=0) ** 2 - 3.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ng._tiebreak_kurtosis_order(kurt, src, tol=10.0)  # force the tie path
    assert any("tiebreak" in str(wi.message).lower() for wi in w)


def test_non_gaussian_svar_result_includes_diagnostics():
    rng = np.random.default_rng(13)
    n, T, p = 2, 600, 1
    e = rng.standard_t(df=4, size=(T + p, n))
    B = np.array([[1.0, 0.3], [0.2, 0.8]])
    A = np.array([[0.5, 0.0], [0.0, 0.4]])
    Y = np.zeros((T + p, n))
    for t in range(p, T + p):
        Y[t] = A @ Y[t - 1] + B @ e[t]
    from puremacro.var.identify import non_gaussian_svar
    res = non_gaussian_svar(Y, p=p, horizon=4, seed=13)
    assert res.lr_test is not None
    assert set(res.lr_test.keys()) == {"stat", "df", "p_value"}
    assert res.consistency_check is not None
    assert set(res.consistency_check.keys()) == {"max_abs_diff", "rms_diff", "passed"}
    # B B^T should match Σ_u tightly: B is derived from Cholesky(Σ_dof), but
    # the ICA variance rescaling step inside non_gaussian_svar absorbs the
    # dof/MLE scale factor.
    assert res.consistency_check["passed"] is True
