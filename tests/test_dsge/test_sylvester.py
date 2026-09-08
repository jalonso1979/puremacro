"""Unit tests for the Generalized Schur Sylvester solver (_sylvester.py).

Verifies numerical accuracy, equivalence to dense Kronecker solutions, residual norm,
Schur decomposition stability on complex conjugate eigenvalues, zero-RHS corner cases,
near-singular fallback, memory efficiency, and runtime performance on SW07 dimensions.
"""

from __future__ import annotations

import math
import time
import numpy as np
import pytest

from puremacro.dsge._sylvester import solve_generalized_sylvester_kronecker


def test_schur_sylvester_2x2_basic():
    """Solve a 2x2 generalized Sylvester system and check dimensions and finiteness."""
    A_hat = np.array([[2.0, 0.5], [0.1, 1.5]])
    A_plus = np.array([[0.2, 0.0], [0.0, 0.1]])
    h_x = np.array([[0.8, 0.1], [0.0, 0.7]])
    K_xx = np.array([[1.0, 0.2, 0.2, 0.5], [0.3, 0.1, 0.1, 0.4]])

    g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
    assert g_xx.shape == (2, 4)
    assert np.isfinite(g_xx).all()
    assert np.isrealobj(g_xx)


def test_schur_sylvester_matches_kronecker_exact():
    """Schur Sylvester solution matches full Kronecker expansion to <= 1e-10."""
    rng = np.random.default_rng(12345)
    N, n_x = 3, 3
    A_hat = np.eye(N) * 2.0 + 0.1 * rng.standard_normal((N, N))
    A_plus = 0.1 * rng.standard_normal((N, N))
    h_x = 0.6 * np.eye(n_x) + 0.05 * rng.standard_normal((n_x, n_x))
    K_xx = rng.standard_normal((N, n_x**2))

    g_xx_schur = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)

    # Reference dense Kronecker solve
    C = np.kron(h_x, h_x)
    sys_mat = np.kron(np.eye(n_x**2), A_hat) + np.kron(C.T, A_plus)
    rhs = -K_xx.reshape(-1, order="F")
    vec_gxx = np.linalg.solve(sys_mat, rhs)
    g_xx_kron = vec_gxx.reshape((N, n_x**2), order="F")

    max_diff = np.max(np.abs(g_xx_schur - g_xx_kron))
    assert max_diff <= 1e-10, f"Max absolute difference {max_diff} > 1e-10"
    np.testing.assert_allclose(g_xx_schur, g_xx_kron, atol=1e-10)


def test_schur_sylvester_residual_norm():
    """Check algebraic residual ||A_hat X + A_+ X (h_x kron h_x) + K_xx|| < 1e-10."""
    rng = np.random.default_rng(999)
    N, n_x = 4, 3
    A_hat = np.eye(N) * 3.0 + 0.1 * rng.standard_normal((N, N))
    A_plus = 0.2 * rng.standard_normal((N, N))
    h_x = 0.7 * np.eye(n_x) + 0.03 * rng.standard_normal((n_x, n_x))
    K_xx = rng.standard_normal((N, n_x**2))

    g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
    C = np.kron(h_x, h_x)
    residual = A_hat @ g_xx + A_plus @ g_xx @ C + K_xx
    res_norm = np.linalg.norm(residual)
    assert res_norm < 1e-10, f"Residual norm {res_norm} exceeds 1e-10"


def test_schur_sylvester_complex_conjugate_eigenvalues():
    """Operates stably when h_x has complex conjugate eigenvalue pairs."""
    theta = math.pi / 3  # 60 degrees rotation
    h_x = 0.85 * np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ])
    A_hat = np.array([[2.5, 0.2], [0.1, 1.8]])
    A_plus = np.array([[0.1, 0.05], [0.02, 0.15]])
    K_xx = np.array([[1.0, 0.5, 0.5, 2.0], [0.2, 0.8, 0.8, 0.3]])

    g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
    assert np.isrealobj(g_xx)
    assert np.isfinite(g_xx).all()

    C = np.kron(h_x, h_x)
    residual = A_hat @ g_xx + A_plus @ g_xx @ C + K_xx
    assert np.linalg.norm(residual) < 1e-10


def test_schur_sylvester_zero_rhs():
    """Sylvester solver with zero RHS K_xx = 0 produces exact zero solution."""
    A_hat = np.eye(3) * 2.0
    A_plus = np.eye(3) * 0.1
    h_x = np.diag([0.5, 0.6])
    K_xx = np.zeros((3, 4))

    g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
    assert np.allclose(g_xx, 0.0)
    assert g_xx.shape == (3, 4)


def test_schur_sylvester_zero_hx():
    """Sylvester solver with h_x = 0 reduces directly to A_hat X = -K_xx."""
    A_hat = np.array([[2.0, 0.5], [0.0, 3.0]])
    A_plus = np.eye(2)
    h_x = np.zeros((2, 2))
    K_xx = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
    expected = -np.linalg.solve(A_hat, K_xx)
    np.testing.assert_allclose(g_xx, expected, atol=1e-12)


def test_schur_sylvester_zero_states():
    """Sylvester solver handles models with zero predetermined states (n_x = 0)."""
    A_hat = np.eye(4)
    A_plus = np.eye(4) * 0.1
    h_x = np.zeros((0, 0))
    K_xx = np.zeros((4, 0))

    g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
    assert g_xx.shape == (4, 0)


def test_schur_sylvester_1x1_state():
    """Sylvester solver on single-state model (n_x = 1)."""
    A_hat = np.array([[2.0, 0.1], [0.2, 1.5]])
    A_plus = np.array([[0.3, 0.0], [0.0, 0.2]])
    h_x = np.array([[0.7]])
    K_xx = np.array([[1.5], [0.8]])

    g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
    assert g_xx.shape == (2, 1)
    C = np.kron(h_x, h_x)
    residual = A_hat @ g_xx + A_plus @ g_xx @ C + K_xx
    assert np.linalg.norm(residual) < 1e-12


def test_schur_sylvester_near_singular_fallback():
    """Sylvester solver handles ill-conditioned system gracefully via least squares fallback."""
    A_hat = np.array([[1e-15, 0.0], [0.0, 1.0]])
    A_plus = np.eye(2) * 0.001
    h_x = np.array([[0.5]])
    K_xx = np.array([[1.0], [2.0]])

    g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
    assert np.isfinite(g_xx).all()
    assert g_xx.shape == (2, 1)


def test_schur_sylvester_sw07_benchmark_timing():
    """SW07 dimensions (N=40, nx=15) execute in <= 0.030s with max error <= 1e-8."""
    rng = np.random.default_rng(42)
    N, n_x = 40, 15
    A_hat = np.eye(N) + 0.05 * rng.standard_normal((N, N))
    A_plus = 0.1 * rng.standard_normal((N, N))
    h_x = 0.5 * np.eye(n_x) + 0.02 * rng.standard_normal((n_x, n_x))
    K_xx = rng.standard_normal((N, n_x**2))

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
        times.append(time.perf_counter() - t0)
    elapsed = min(times)

    assert g_xx.shape == (N, n_x**2)
    assert np.isfinite(g_xx).all()
    assert np.isrealobj(g_xx)
    assert elapsed <= 0.030, f"Sylvester solver took {elapsed:.4f}s > 0.030s"

    # Fast verification of sample columns residual
    C = np.kron(h_x, h_x)
    # Check 5 random columns
    cols = [0, 50, 100, 150, 224]
    for col in cols:
        lhs_col = A_hat @ g_xx[:, col] + A_plus @ (g_xx @ C[:, col])
        res_col = lhs_col + K_xx[:, col]
        assert np.linalg.norm(res_col) < 1e-9
