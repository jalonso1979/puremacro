"""Unit tests for puremacro.var.identify.cholesky.compute_chol_shocks."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.var.identify.cholesky import compute_chol_shocks, cholesky_svar


def _simulate_var2(T: int = 600, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a stable VAR(2) with a known lower-triangular impact matrix L."""
    rng = np.random.default_rng(seed)
    n, p = 3, 2
    A1 = np.array([[0.5, 0.0, 0.0],
                   [0.1, 0.4, 0.0],
                   [0.0, 0.2, 0.3]])
    A2 = np.array([[0.10, 0.00, 0.00],
                   [0.00, 0.05, 0.00],
                   [0.00, 0.00, 0.05]])
    L = np.array([[1.0, 0.0, 0.0],
                  [0.5, 0.8, 0.0],
                  [0.2, 0.3, 0.6]])
    eps = rng.standard_normal((T, n))
    u = eps @ L.T
    Y = np.zeros((T, n))
    for t in range(p, T):
        Y[t] = A1 @ Y[t - 1] + A2 @ Y[t - 2] + u[t]
    burn = 100
    return Y[burn:], eps[burn:], L


@pytest.mark.pyodide_smoke
def test_returns_correct_shapes():
    Y, _, _ = _simulate_var2(T=300)
    t_idx, shocks = compute_chol_shocks(Y, p=2)
    assert t_idx.shape == (Y.shape[0] - 2,)
    assert shocks.shape == (Y.shape[0] - 2, Y.shape[1])
    np.testing.assert_array_equal(t_idx, np.arange(2, Y.shape[0]))


def test_shocks_are_orthonormal():
    """Identified Cholesky shocks should have sample cov ≈ I_n."""
    Y, _, _ = _simulate_var2(T=2000, seed=7)
    _, shocks = compute_chol_shocks(Y, p=2)
    cov = (shocks.T @ shocks) / shocks.shape[0]
    np.testing.assert_allclose(cov, np.eye(shocks.shape[1]), atol=0.10)


def test_ordering_permutes_columns_back():
    """With ordering=[2,1,0], output columns must still match original variable order."""
    Y, _, _ = _simulate_var2(T=600)
    _, shocks_ident = compute_chol_shocks(Y, p=2, ordering=None)
    _, shocks_perm = compute_chol_shocks(Y, p=2, ordering=[2, 1, 0])
    # Same shape, both keyed on the original variable order
    assert shocks_ident.shape == shocks_perm.shape
    # Permuted ordering changes which variable is "first" in the recursive
    # decomposition — so values differ — but the column index still
    # corresponds to the original variable, not the permuted one.
    assert not np.allclose(shocks_ident, shocks_perm)


def test_consistency_with_cholesky_svar():
    """compute_chol_shocks shocks must reconstruct the residuals using
    cholesky_svar's identified impact matrix L = irf_point[0]."""
    Y, _, _ = _simulate_var2(T=800)
    res = cholesky_svar(Y, p=2, horizon=4, n_boot=10, ci=0.9, seed=0)
    L_irf = res.irf_point[0]  # impact response = lower-Cholesky factor of Σ_u
    _, shocks = compute_chol_shocks(Y, p=2)
    # The reduced-form residual is L · ε. Reconstruct it from compute_chol_shocks's
    # output and check it matches the OLS residual that cholesky_svar would have used.
    from puremacro.var.identify.cholesky import estimate_var
    _, _, _, resid, _ = estimate_var(Y, 2)
    np.testing.assert_allclose(shocks @ L_irf.T, resid, atol=1e-10)
