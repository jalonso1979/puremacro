from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi.kernels import bellman_step
from puremacro.vfi.solve import solve_vfi


def test_bellman_step_picks_max_and_index():
    # (n_d=1, n_a'=2, n_a=1, n_z=1): a'=0 -> 0.0 ; a'=1 -> 5.0
    R = np.array([[[[0.0]], [[5.0]]]])
    EV = np.zeros((2, 1))
    V, idx = bellman_step(R, EV, beta=0.9)
    assert V[0, 0] == 5.0
    assert idx[0, 0] == 1


def test_solve_reward_on_aprime_constant_value():
    # reward = a' (independent of a); one exo state.
    # V = max_{a'}[a' + beta V(a')] = a'_max + beta V  => V = a'_max/(1-beta).
    a = np.array([1.0, 3.0])
    R = np.empty((1, 2, 2, 1))
    for ap in range(2):
        for ia in range(2):
            R[0, ap, ia, 0] = a[ap]
    P = np.array([[1.0]])
    V, idx, n_ap, n_it, sup = solve_vfi(R, P, beta=0.5, tol=1e-12)
    np.testing.assert_allclose(V[:, 0], np.array([6.0, 6.0]), atol=1e-8)
    assert idx[0, 0] == 1 and idx[1, 0] == 1  # always choose a'=3 (index 1)


def test_howard_matches_pure_iteration():
    rng = np.random.default_rng(0)
    n_a, n_z = 5, 3
    R = rng.standard_normal((1, n_a, n_a, n_z))
    P = rng.random((n_z, n_z))
    P = P / P.sum(axis=1, keepdims=True)
    Vp, *_ = solve_vfi(R, P, beta=0.9, howard=False, tol=1e-12)
    Vh, *_ = solve_vfi(R, P, beta=0.9, howard=True, n_howard=30, tol=1e-12)
    np.testing.assert_allclose(Vp, Vh, atol=1e-7)


def test_nonconvergence_raises():
    # all-ones reward => V* = 1/(1-beta) != 0, so zeros-init does NOT reach the
    # fixed point in 3 iterations (an all-zero R would converge immediately).
    R = np.ones((1, 2, 2, 1))
    P = np.array([[1.0]])
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_vfi(R, P, beta=0.99, howard=False, tol=1e-30, max_iter=3)


import importlib.util


numba_only = pytest.mark.skipif(
    importlib.util.find_spec("numba") is None, reason="numba not installed"
)


@numba_only
def test_numba_twin_matches_numpy_solve():
    rng = np.random.default_rng(7)
    n_a, n_z = 6, 4
    R = np.ascontiguousarray(rng.standard_normal((1, n_a, n_a, n_z)))
    P = rng.random((n_z, n_z))
    P = P / P.sum(axis=1, keepdims=True)
    V_o, idx_o, n_ap, n_it_o, sup_o = solve_vfi(R, P, 0.9, tol=1e-12)

    from puremacro.vfi import kernels_numba as KN

    V_n, idx_n, n_it_n, sup_n = KN.solve_vfi_numba(
        R, P, 0.9, True, 20, 1e-12, 10_000
    )
    np.testing.assert_allclose(V_n, V_o, rtol=1e-9, atol=1e-10)
    np.testing.assert_array_equal(idx_n, idx_o)


def test_no_feasible_action_raises():
    # current-state a=0 has -inf for every a' -> clear error, not NaN-spin
    R = np.zeros((1, 2, 2, 1))
    R[0, :, 0, 0] = -np.inf
    P = np.array([[1.0]])
    with pytest.raises(RuntimeError, match="no feasible action"):
        solve_vfi(R, P, beta=0.9, tol=1e-10)


@numba_only
def test_numba_no_feasible_action_raises():
    from puremacro.vfi import kernels_numba as KN

    R = np.zeros((1, 2, 2, 1))
    R[0, :, 0, 0] = -np.inf
    P = np.array([[1.0]])
    with pytest.raises(Exception, match="no feasible action"):
        KN.solve_vfi_numba(np.ascontiguousarray(R), P, 0.9, True, 20, 1e-10, 10_000)
