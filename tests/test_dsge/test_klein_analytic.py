"""Klein solver vs. closed-form solutions.

``klein_solve`` returns four matrices — G, F, N, L — and until 1.2.0 only
G was checked against an analytic benchmark. The other three were
validated indirectly, through SW07, whose equation ordering happens to
mask two errors:

* ``F`` was computed as ``-inv(Z22) @ Z21``, which is not the partner of
  the ``G = Z11 inv(S11) T11 inv(Z11)`` the same function returns. The
  consistent formula reads the solution off the same Z11-parameterised
  stable subspace: ``F = Z21 @ inv(Z11)``.
* The guard that was supposed to catch a bad ``F`` — and the Sylvester
  fallback behind it — partitioned A and B by *row* at ``n_pre``. Rows
  are equations and the split is over variables, so in any model whose
  equations are not incidentally ordered to match, the guard inspects
  the wrong equations and the fallback solves an underdetermined system.
* ``L`` came from a Klein (2000) eq.-(33) expression that returns zero
  whenever a shock enters a control equation contemporaneously.

The two models below are small enough to have exact solutions, so they
pin all four matrices rather than one.
"""
import numpy as np
import pytest

from puremacro.dsge.klein import klein_solve


# --- model 1: neoclassical growth, full depreciation, log utility -------
# The one textbook case with a closed form:
#     k_{t+1} = alpha*beta*z_t*k_t^alpha,  c_t = (1-alpha*beta)*z_t*k_t^alpha
# so in log deviations   k^_{t+1} = alpha*k^_t + z^_t   and   c^_t = alpha*k^_t + z^_t.
ALPHA, BETA, RHO = 0.33, 0.98, 0.9


def _growth_model():
    """Log-linear Klein form with variables ordered [k, z, c]."""
    k = (ALPHA * BETA) ** (1 / (1 - ALPHA))
    y = k ** ALPHA
    c = y - k
    # Rows: Euler, resource constraint, AR(1).
    Fp = np.array([[-(ALPHA - 1), -1.0, 1.0],
                   [k, 0.0, 0.0],
                   [0.0, 1.0, 0.0]])
    Fc = np.array([[0.0, 0.0, -1.0],
                   [-y * ALPHA, -y, c],
                   [0.0, -RHO, 0.0]])
    Fu = np.array([[0.0], [0.0], [-1.0]])
    return Fp, -Fc, -Fu


def test_growth_model_matches_closed_form():
    A, B, C = _growth_model()
    sol = klein_solve(A, B, n_pre=2, C=C, strict=True)

    assert sol.eu == (1, 1)
    np.testing.assert_allclose(sol.G, [[ALPHA, 1.0], [0.0, RHO]], atol=1e-12)
    np.testing.assert_allclose(sol.F, [[ALPHA, 1.0]], atol=1e-12)
    np.testing.assert_allclose(sol.N.ravel(), [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(sol.L.ravel(), [0.0], atol=1e-12)


def test_growth_model_solution_satisfies_its_own_equations():
    """The equilibrium condition, checked directly rather than via a formula.

    Any correct solution satisfies ``(A1 + A2 F) G == B1 + B2 F`` for the
    state block and ``(A1 + A2 F) N - B2 L == C`` for the shock block.
    The pre-1.2.0 F failed the first by ~0.8.
    """
    A, B, C = _growth_model()
    n_pre = 2
    sol = klein_solve(A, B, n_pre=n_pre, C=C, strict=True)
    A1, A2 = A[:, :n_pre], A[:, n_pre:]
    B1, B2 = B[:, :n_pre], B[:, n_pre:]

    state_resid = (A1 + A2 @ sol.F) @ sol.G - B1 - B2 @ sol.F
    shock_resid = (A1 + A2 @ sol.F) @ sol.N - B2 @ sol.L - C
    assert np.max(np.abs(state_resid)) < 1e-10
    assert np.max(np.abs(shock_resid)) < 1e-10


# --- model 2: a shock that hits a control contemporaneously -------------
# z = [s, c]; s is predetermined, c is static:
#     s_{t+1} = rho*s_t + u_t      =>  G = [rho],  N = [1]
#     c_t     = theta*s_t + u_t    =>  F = [theta], L = [1]
RHO_S, THETA = 0.8, 2.0


def test_contemporaneous_shock_loads_on_the_control():
    A = np.array([[1.0, 0.0], [0.0, 0.0]])
    B = np.array([[RHO_S, 0.0], [THETA, -1.0]])
    C = np.array([[1.0], [1.0]])

    sol = klein_solve(A, B, n_pre=1, C=C, strict=True)

    np.testing.assert_allclose(sol.G, [[RHO_S]], atol=1e-12)
    np.testing.assert_allclose(sol.F, [[THETA]], atol=1e-12)
    np.testing.assert_allclose(sol.N.ravel(), [1.0], atol=1e-12)
    # This is the assertion that failed before 1.2.0: L was 0.
    np.testing.assert_allclose(sol.L.ravel(), [1.0], atol=1e-12)


def test_no_shock_block_returns_empty_loadings():
    A = np.array([[1.0, 0.0], [0.0, 0.0]])
    B = np.array([[RHO_S, 0.0], [THETA, -1.0]])
    sol = klein_solve(A, B, n_pre=1, strict=True)
    assert sol.N.shape == (1, 0)
    assert sol.L.shape == (1, 0)
