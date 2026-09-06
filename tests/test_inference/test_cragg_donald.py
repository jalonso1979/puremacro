"""Regression tests for puremacro.inference.weak_iv.cragg_donald_f (audit C29).

Before the fix the statistic divided the minimum eigenvalue of the
concentration matrix by the number of endogenous regressors ``k`` instead of
the number of instruments ``l``, and replaced the first-stage residual
covariance matrix Sigma_VV by the scalar mean of the residual variances. With
``l = 1`` that happens to coincide with the first-stage F, so the only value
check that existed passed; for ``l = 3, k = 1`` it returned exactly 3x the
first-stage F (309.66 vs 103.22) and for ``k = 2, l = 4`` 148.04 against the
Stock-Yogo definition's 65.14 -- so weak instruments read as strong against
the Stock-Yogo tables, which only start at l = 3.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.inference.weak_iv import cragg_donald_f


def _first_stage_f(x, Z):
    """Homoskedastic first-stage F for the excluded instruments (no constant)."""
    n, l = Z.shape
    Pi = np.linalg.lstsq(Z, x, rcond=None)[0]
    xh = Z @ Pi
    xt = x - xh
    return (xh @ xh / l) / (xt @ xt / (n - l))


def _stock_yogo_cd(X, Z, W=None):
    """Direct Stock-Yogo (2005) computation of the Cragg-Donald statistic."""
    n, l = Z.shape
    m = 0
    if W is not None:
        m = W.shape[1]
        Mw = np.eye(n) - W @ np.linalg.solve(W.T @ W, W.T)
        X, Z = Mw @ X, Mw @ Z
    Pi = np.linalg.lstsq(Z, X, rcond=None)[0]
    Xh = Z @ Pi
    V = X - Xh
    Svv = V.T @ V / (n - l - m)
    w, U = np.linalg.eigh(Svv)
    Sm12 = U @ np.diag(w ** -0.5) @ U.T
    G = Sm12 @ (Xh.T @ Xh) @ Sm12 / l
    return float(np.min(np.linalg.eigvalsh(G)))


def _design(l, n=400, seed=5):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, l))
    v = rng.standard_normal(n)
    x = Z @ (0.5 * np.ones(l)) + v
    y = x * 1.0 + 0.5 * v + rng.standard_normal(n)
    return y, x, Z


def test_just_identified_equals_first_stage_f():
    y, x, Z = _design(l=1)
    assert cragg_donald_f(y, x[:, None], Z) == pytest.approx(_first_stage_f(x, Z), rel=1e-10)
    # 1-D X and Z accepted too.
    assert cragg_donald_f(y, x, Z[:, 0]) == pytest.approx(_first_stage_f(x, Z), rel=1e-10)


@pytest.mark.parametrize("l", [2, 3, 5])
def test_over_identified_single_endogenous_equals_first_stage_f(l):
    """k = 1: the statistic IS the first-stage F. It used to be l times it."""
    y, x, Z = _design(l=l)
    got = cragg_donald_f(y, x[:, None], Z)
    F = _first_stage_f(x, Z)
    assert got == pytest.approx(F, rel=1e-10), f"ratio {got / F:.3f} (was exactly {l})"


def test_two_endogenous_four_instruments_matches_stock_yogo_definition():
    """k = 2, l = 4 against min eig(Sigma_VV^-1/2 X'PzX Sigma_VV^-1/2) / l.

    The audit's direct computation gives 65.139 on this design; the old
    code returned 148.04.
    """
    rng = np.random.default_rng(9)
    n, l, k = 400, 4, 2
    Z = rng.standard_normal((n, l))
    V = rng.standard_normal((n, k)) @ np.array([[1, 0.5], [0, 1]])
    X = Z @ rng.standard_normal((l, k)) * 0.4 + V
    y = X @ np.ones(k) + rng.standard_normal(n)
    got = cragg_donald_f(y, X, Z)
    ref = _stock_yogo_cd(X, Z)
    assert got == pytest.approx(ref, rel=1e-10)
    assert got == pytest.approx(65.139, abs=5e-3)
    assert got < 100.0  # the old 148.04 is ruled out


def test_correlated_first_stage_errors_use_the_matrix_not_a_scalar():
    """With strongly correlated first-stage errors the scalar-variance
    shortcut and the matrix form disagree; the matrix form is the definition."""
    rng = np.random.default_rng(11)
    n, l, k = 500, 3, 2
    Z = rng.standard_normal((n, l))
    e = rng.standard_normal(n)
    V = np.column_stack([e + 0.1 * rng.standard_normal(n),
                         e + 0.1 * rng.standard_normal(n)])
    X = Z @ np.array([[0.5, 0.2], [0.1, 0.6], [0.3, -0.4]]) + V
    y = X @ np.ones(k) + rng.standard_normal(n)
    got = cragg_donald_f(y, X, Z)
    assert got == pytest.approx(_stock_yogo_cd(X, Z), rel=1e-10)
    # Scalar shortcut (old code without the /k error) would differ materially.
    Pi = np.linalg.lstsq(Z, X, rcond=None)[0]
    Xh = Z @ Pi
    Vh = X - Xh
    s2 = np.mean(np.sum(Vh ** 2, axis=0)) / (n - l)
    scalar_version = np.min(np.linalg.eigvalsh(Xh.T @ Xh / s2)) / l
    assert abs(got - scalar_version) / scalar_version > 0.05


def test_included_controls_are_partialled_and_cost_degrees_of_freedom():
    rng = np.random.default_rng(13)
    n, l = 300, 2
    Z = rng.standard_normal((n, l))
    w = rng.standard_normal(n)
    x = 1.0 + 0.4 * Z[:, 0] + 0.3 * Z[:, 1] + 0.5 * w + rng.standard_normal(n)
    y = x + rng.standard_normal(n)
    W = np.column_stack([np.ones(n), w])
    got = cragg_donald_f(y, x, Z, W=W)
    assert got == pytest.approx(_stock_yogo_cd(x[:, None], Z, W), rel=1e-10)
    assert got != pytest.approx(cragg_donald_f(y, x, Z), rel=1e-3)


def test_errors_on_underidentification_and_shape_mismatch():
    y, x, Z = _design(l=1)
    with pytest.raises(ValueError, match="at least as many instruments"):
        cragg_donald_f(y, np.column_stack([x, x ** 2]), Z)
    with pytest.raises(ValueError, match="rows"):
        cragg_donald_f(y, x, Z[:-1])
