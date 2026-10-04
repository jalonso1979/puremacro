"""Tests for ``puremacro.dsge.stacked_newton`` (matrix-free stacked-time Newton-Krylov).

Fixtures are analytic: the Ramsey model of ``tests/test_dynare_perfect_foresight.py``
and a 20-sector Cobb-Douglas growth transition (the assessment prototype; no such
fixture existed under ``tests/`` before this file). Parity tests against the IO
research engine on ``/Volumes/BIGDATA/Research/IO`` skip when the volume is absent.
Tolerances are stated at every assertion.
"""
from __future__ import annotations

import copy
import importlib
import inspect
import json
import os
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from puremacro.dsge.perfect_foresight import PerfectForesightResult, solve_perfect_foresight
from puremacro.dsge.stacked_newton import (
    BlockTridiagonalPreconditioner,
    HorizonComparison,
    StackedNewtonError,
    StackedNewtonResult,
    StackedProblem,
    StructuredBlockTridiagonalPreconditioner,
    compare_horizons,
    compare_stacked_horizons,
    preconditioned_lgmres,
    solve_stacked_newton_krylov,
)

# The IO research workspace; parity tests skip when it is absent. Override with PUREMACRO_IO_ROOT.
IO_ROOT = pathlib.Path(os.environ.get("PUREMACRO_IO_ROOT", "/Volumes/BIGDATA/Research/IO"))


def _io_module(subpath, name):
    """Import an IO module from the research volume without writing bytecode there.

    The IO ``dynamic_model`` package sets BLAS thread variables with
    ``os.environ.setdefault`` on import; variables it adds are removed again so the
    pytest process environment is unchanged.
    """
    if not (IO_ROOT / subpath).exists():
        pytest.skip("IO research volume not mounted")
    sys.path.insert(0, str(IO_ROOT / subpath))
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    env_before = set(os.environ)
    try:
        return importlib.import_module(name)
    finally:
        for key in set(os.environ) - env_before:
            os.environ.pop(key, None)
        sys.dont_write_bytecode = old
        sys.path.pop(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def ramsey():
    """Cass-Koopmans model of tests/test_dynare_perfect_foresight.py plus its analytic Jacobian."""
    alpha, beta, delta, sigma = 0.33, 0.96, 0.10, 1.0
    r_ss = 1.0 / beta - (1.0 - delta)
    k_ss = (alpha / r_ss) ** (1.0 / (1.0 - alpha))
    c_ss = k_ss ** alpha - delta * k_ss

    def _A(eps):
        return float(eps) if np.ndim(eps) == 0 else float(eps[0])

    def equations_fn(y_plus, y_curr, y_lag, eps):
        c_p, k_p = y_plus
        c, k = y_curr
        c_m, k_m = y_lag
        A = _A(eps)
        euler = c ** (-sigma) - beta * c_p ** (-sigma) * (alpha * A * k ** (alpha - 1.0) + 1.0 - delta)
        res_c = k - (A * k_m ** alpha + (1.0 - delta) * k_m - c)
        return [euler, res_c]

    def blocks_fn(y_plus, y_curr, y_lag, eps):
        c_p, _ = y_plus
        c, k = y_curr
        _, k_m = y_lag
        A = _A(eps)
        gross = alpha * A * k ** (alpha - 1.0) + 1.0 - delta
        Bm = np.array([[-sigma * c ** (-sigma - 1.0), -beta * c_p ** (-sigma) * alpha * (alpha - 1.0) * A * k ** (alpha - 2.0)],
                       [1.0, 1.0]])
        Am = np.array([[sigma * beta * c_p ** (-sigma - 1.0) * gross, 0.0], [0.0, 0.0]])
        Cm = np.array([[0.0, 0.0], [0.0, -(alpha * A * k_m ** (alpha - 1.0) + 1.0 - delta)]])
        return Am, Bm, Cm

    def jvp_fn(y_plus, y_curr, y_lag, eps, v_plus, v_curr, v_lag):
        Am, Bm, Cm = blocks_fn(y_plus, y_curr, y_lag, eps)
        return Am @ v_plus + Bm @ v_curr + Cm @ v_lag

    def steady_state(A):
        k = (alpha * A / r_ss) ** (1.0 / (1.0 - alpha))
        return np.array([A * k ** alpha - delta * k, k])

    return dict(equations_fn=equations_fn, blocks_fn=blocks_fn, jvp_fn=jvp_fn,
                y_ss=np.array([c_ss, k_ss]), k_ss=k_ss, c_ss=c_ss, steady_state=steady_state,
                names=["c", "k"])


def _growth20():
    """20-sector Cobb-Douglas growth transition (n_vars = 40, T = 100, seed 11)."""
    rng = np.random.default_rng(11)
    n_sec, T = 20, 100
    alpha = rng.uniform(0.25, 0.45, size=n_sec)
    beta, delta, sigma = 0.96, 0.08, 2.0
    k_ss = (alpha / (1.0 / beta - 1.0 + delta)) ** (1.0 / (1.0 - alpha))
    c_ss = k_ss ** alpha - delta * k_ss
    y_ss = np.r_[k_ss, c_ss]

    def eq(yp, yc, yl, eps):
        k_next, c = yc[:n_sec], yc[n_sec:]
        k = yl[:n_sec]
        c_next = yp[n_sec:]
        resource = k_next - k ** alpha - (1.0 - delta) * k + c
        euler = (c ** (-sigma) - beta * c_next ** (-sigma) * (alpha * k_next ** (alpha - 1.0) + 1.0 - delta)
                 - eps[:n_sec])
        return np.r_[resource, euler]

    y_init = np.r_[0.7 * k_ss, c_ss]
    exo = np.zeros((T, 2 * n_sec))
    return dict(eq=eq, y_init=y_init, y_ss=y_ss, exo=exo, T=T, n=2 * n_sec)


@pytest.fixture(scope="module")
def growth20():
    return _growth20()


@pytest.fixture(scope="module")
def growth20_reference(growth20):
    g = growth20
    ref = solve_perfect_foresight(equations_fn=g["eq"], y_init=g["y_init"], y_ss=g["y_ss"],
                                  exogenous_path=g["exo"], n_periods=g["T"], tol=1e-9, max_iter=60)
    assert ref.converged
    return ref


def _random_blocks(T=7, n=5, seed=0, per_date=False):
    rng = np.random.default_rng(seed)
    shape = (T, n, n) if per_date else (n, n)
    A = 0.3 * rng.normal(size=shape)
    C = 0.3 * rng.normal(size=shape)
    B = rng.normal(size=shape)
    B = B + np.eye(n) * (n + 2.0)
    return A, B, C


def _dense_stacked(A, B, C, T, mask_block=None):
    n = B.shape[-1]
    J = np.zeros((T * n, T * n))
    for t in range(T):
        Bt = B[t] if B.ndim == 3 else B
        J[t * n:(t + 1) * n, t * n:(t + 1) * n] = Bt
        if t + 1 < T and (mask_block is None or t // mask_block == (t + 1) // mask_block):
            At = A[t] if A.ndim == 3 else A
            Ct = C[t + 1] if C.ndim == 3 else C
            J[t * n:(t + 1) * n, (t + 1) * n:(t + 2) * n] = At
            J[(t + 1) * n:(t + 2) * n, t * n:(t + 1) * n] = Ct
    return J


def _pf_fd_blocks(eq, yp, yc, yl, eps, n):
    """Central-difference blocks with the step rule of solve_perfect_foresight (1e-7 * max(1, |y_j|))."""
    out = []
    for which, base in enumerate((yp, yc, yl)):
        M = np.zeros((n, n))
        for j in range(n):
            h = 1e-7 * max(1.0, abs(base[j]))
            args_p = [np.array(yp, dtype=float), np.array(yc, dtype=float), np.array(yl, dtype=float)]
            args_m = [a.copy() for a in args_p]
            args_p[which][j] += h
            args_m[which][j] -= h
            fp = np.asarray(eq(*args_p, eps), dtype=float).ravel()
            fm = np.asarray(eq(*args_m, eps), dtype=float).ravel()
            M[:, j] = (fp - fm) / (2.0 * h)
        out.append(M)
    return out


# ---------------------------------------------------------------------------
# Linear-algebra oracles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["thomas", "splu"])
@pytest.mark.parametrize("per_date", [False, True])
def test_block_inverse_matches_dense_solve(method, per_date):
    T, n = 7, 5
    A, B, C = _random_blocks(T, n, seed=1, per_date=per_date)
    J = _dense_stacked(A, B, C, T)
    pre = BlockTridiagonalPreconditioner(A, B, C, T, method=method)
    rng = np.random.default_rng(2)
    r = rng.normal(size=T * n)
    x = np.linalg.solve(J, r)
    np.testing.assert_allclose(pre.apply(r), x, rtol=0.0, atol=1e-12 * np.max(np.abs(x)))
    np.testing.assert_allclose(pre.as_linear_operator() @ r, x, rtol=0.0, atol=1e-12 * np.max(np.abs(x)))
    np.testing.assert_allclose(pre.to_sparse().toarray(), J, rtol=0.0, atol=1e-15)
    assert pre.time_block == T and pre.method == method and pre.memory_bytes > 0
    assert pre.shape == (T * n, T * n)
    # Accepts a (T, n) right-hand side and returns a flat vector.
    assert pre.apply(r.reshape(T, n)).shape == (T * n,)


@pytest.mark.parametrize("method", ["thomas", "splu"])
@pytest.mark.parametrize("H", [1, 3, 7])
@pytest.mark.parametrize("per_date", [False, True])
def test_time_block_jacobi_matches_masked_dense_solve(method, H, per_date):
    """labels = repeat(arange(T)//H, n): links across blocks are dropped (IO block Jacobi)."""
    T, n = 7, 5
    A, B, C = _random_blocks(T, n, seed=3, per_date=per_date)
    masked = _dense_stacked(A, B, C, T, mask_block=H)
    labels = np.repeat(np.arange(T) // H, n)
    assert np.all((masked != 0) <= (labels[:, None] == labels[None, :]))
    pre = BlockTridiagonalPreconditioner(A, B, C, T, method=method, time_block=H)
    r = np.random.default_rng(4).normal(size=T * n)
    x = np.linalg.solve(masked, r)
    np.testing.assert_allclose(pre.apply(r), x, rtol=0.0, atol=1e-12 * np.max(np.abs(x)))
    np.testing.assert_allclose(pre.to_sparse().toarray(), masked, rtol=0.0, atol=1e-15)
    assert pre.time_block == H
    if H == T:
        assert "exact" in pre.name
    else:
        assert f"time_block={H}" in pre.name


def test_preconditioner_is_linear_and_scale_invariant():
    T, n = 6, 4
    A, B, C = _random_blocks(T, n, seed=5)
    pre = BlockTridiagonalPreconditioner(A, B, C, T)
    rng = np.random.default_rng(6)
    r, s = rng.normal(size=T * n), rng.normal(size=T * n)
    np.testing.assert_allclose(pre.apply(r + 0.3 * s), pre.apply(r) + 0.3 * pre.apply(s), rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(pre.apply(1e6 * r), 1e6 * pre.apply(r), rtol=1e-13, atol=0.0)


def test_block_preconditioner_validation():
    T, n = 5, 3
    A, B, C = _random_blocks(T, n, seed=7)
    with pytest.raises(ValueError, match="unknown method"):
        BlockTridiagonalPreconditioner(A, B, C, T, method="cholesky")
    with pytest.raises(ValueError, match="time_block"):
        BlockTridiagonalPreconditioner(A, B, C, T, time_block=T + 1)
    with pytest.raises(ValueError, match="time_block"):
        BlockTridiagonalPreconditioner(A, B, C, T, time_block=0)
    with pytest.raises(ValueError, match="n_periods"):
        BlockTridiagonalPreconditioner(A, B, C, 0)
    with pytest.raises(ValueError, match="shape"):
        BlockTridiagonalPreconditioner(A[:2, :2], B, C, T)
    with pytest.raises(ValueError, match="finite"):
        BlockTridiagonalPreconditioner(A, B * np.nan, C, T)
    with pytest.raises(ValueError, match="pivot"):
        BlockTridiagonalPreconditioner(A, np.zeros((n, n)), C, T)
    # zero diagonal blocks with T odd: the stacked matrix itself is singular
    with pytest.raises(ValueError, match="SuperLU.*singular"):
        BlockTridiagonalPreconditioner(A, np.zeros((n, n)), C, T, method="splu")
    with pytest.raises(ValueError, match="right-hand side"):
        BlockTridiagonalPreconditioner(A, B, C, T).apply(np.ones(3))
    with pytest.raises(ValueError, match="per-date"):
        BlockTridiagonalPreconditioner(np.zeros((T + 1, n, n)), np.zeros((T + 1, n, n)), np.zeros((T + 1, n, n)), T)


@pytest.mark.parametrize("per_date", [False, True])
def test_block_preconditioner_copies_its_inputs(per_date):
    """Mutating A, B, C after construction changes neither apply() nor to_sparse() (read-only copies)."""
    T, n = 6, 4
    A, B, C = _random_blocks(T, n, seed=1, per_date=per_date)
    pre = BlockTridiagonalPreconditioner(A, B, C, T)
    r = np.random.default_rng(2).normal(size=T * n)
    before = pre.apply(r)
    sparse_before = pre.to_sparse().toarray()
    C *= 2.0
    A *= 0.5
    B += 1.0
    np.testing.assert_array_equal(pre.apply(r), before)
    np.testing.assert_array_equal(pre.to_sparse().toarray(), sparse_before)
    np.testing.assert_allclose(pre.apply(r), np.linalg.solve(sparse_before, r), rtol=0.0,
                               atol=1e-12 * np.max(np.abs(before)))


@pytest.mark.parametrize("sweep", ["jacobi", "forward", "backward", "symmetric"])
def test_structured_sweeps_match_dense_formulas(sweep):
    T, n = 6, 4
    A, B, C = _random_blocks(T, n, seed=8, per_date=True)
    J = _dense_stacked(A, B, C, T)
    D = _dense_stacked(A, B, C, T, mask_block=1)
    L = np.zeros_like(J)
    U = np.zeros_like(J)
    for t in range(T - 1):
        U[t * n:(t + 1) * n, (t + 1) * n:(t + 2) * n] = A[t]
        L[(t + 1) * n:(t + 2) * n, t * n:(t + 1) * n] = C[t + 1]
    np.testing.assert_allclose(D + L + U, J, rtol=0.0, atol=0.0)
    lus = [scipy.linalg.lu_factor(B[t]) for t in range(T)]
    pre = StructuredBlockTridiagonalPreconditioner(
        T, n, diag_solve=lambda t, r: scipy.linalg.lu_solve(lus[t], r),
        lower_apply=lambda t, x: C[t] @ x, upper_apply=lambda t, x: A[t] @ x, sweep=sweep)
    r = np.random.default_rng(9).normal(size=T * n)
    expected = {
        "jacobi": np.linalg.solve(D, r),
        "forward": np.linalg.solve(D + L, r),
        "backward": np.linalg.solve(D + U, r),
        "symmetric": np.linalg.solve(D + U, D @ np.linalg.solve(D + L, r)),
    }[sweep]
    np.testing.assert_allclose(pre.apply(r), expected, rtol=0.0, atol=1e-12 * np.max(np.abs(expected)))
    np.testing.assert_allclose(pre.as_linear_operator() @ r, expected, rtol=0.0, atol=1e-12 * np.max(np.abs(expected)))
    assert pre.sweep == sweep and pre.name == f"structured[{sweep}]"


def test_structured_protocol_subclass_and_validation():
    T, n = 4, 3

    class Exact(StructuredBlockTridiagonalPreconditioner):
        def __init__(self, A, B, C, T):
            super().__init__(T, B.shape[-1], name="exact-subclass")
            self.inner = BlockTridiagonalPreconditioner(A, B, C, T)

        def apply(self, r):
            return self.inner.apply(r)

    A, B, C = _random_blocks(T, n, seed=10)
    pre = Exact(A, B, C, T)
    r = np.random.default_rng(11).normal(size=T * n)
    x = np.linalg.solve(_dense_stacked(A, B, C, T), r)
    np.testing.assert_allclose(pre.as_linear_operator() @ r, x, rtol=0.0, atol=1e-12 * np.max(np.abs(x)))
    assert pre.name == "exact-subclass"
    with pytest.raises(ValueError, match="diag_solve"):
        StructuredBlockTridiagonalPreconditioner(T, n)
    with pytest.raises(ValueError, match="lower_apply"):
        StructuredBlockTridiagonalPreconditioner(T, n, diag_solve=lambda t, r: r, sweep="forward")
    with pytest.raises(ValueError, match="upper_apply"):
        StructuredBlockTridiagonalPreconditioner(T, n, diag_solve=lambda t, r: r, sweep="backward")
    with pytest.raises(ValueError, match="unknown sweep"):
        StructuredBlockTridiagonalPreconditioner(T, n, diag_solve=lambda t, r: r, sweep="zigzag")
    bad = StructuredBlockTridiagonalPreconditioner(T, n, diag_solve=lambda t, r: r * np.nan, sweep="jacobi")
    with pytest.raises(StackedNewtonError, match="Nonfinite"):
        bad.apply(np.ones(T * n))


# ---------------------------------------------------------------------------
# StackedProblem
# ---------------------------------------------------------------------------
def test_stacked_problem_validation(ramsey):
    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]
    with pytest.raises(ValueError, match="n_periods is required"):
        StackedProblem(eq, y_ss, y_ss)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        StackedProblem(eq, y_ss, y_ss[:1], n_periods=3)
    with pytest.raises(ValueError, match="unknown jvp_method"):
        StackedProblem(eq, y_ss, y_ss, n_periods=3, jvp_method="forward")
    with pytest.raises(ValueError, match="variable_names"):
        StackedProblem(eq, y_ss, y_ss, n_periods=3, variable_names=["c"])
    with pytest.raises(ValueError, match="at least n_periods"):
        StackedProblem(eq, y_ss, y_ss, np.ones(3), n_periods=5)
    with pytest.raises(TypeError, match="callable"):
        StackedProblem("not callable", y_ss, y_ss, n_periods=3)
    # With n_periods, T + 2 rows are trimmed to the interior dates and longer paths truncated;
    # without it every row is an interior date (the documented contract).
    p = StackedProblem(eq, y_ss, y_ss, np.arange(7.0), n_periods=5)
    np.testing.assert_array_equal(p.exogenous_path, np.arange(1.0, 6.0))
    assert StackedProblem(eq, y_ss, y_ss, np.arange(7.0)).n_periods == 7
    p = StackedProblem(eq, y_ss, y_ss, np.arange(9.0), n_periods=5)
    np.testing.assert_array_equal(p.exogenous_path, np.arange(5.0))
    assert p.n == 10 and p.n_vars == 2 and p.n_periods == 5 and p.jvp_method == "central"
    assert p.variable_names == ("y_0", "y_1")
    assert not p.y_init.flags.writeable and not p.exogenous_path.flags.writeable
    with pytest.raises(ValueError, match="entries"):
        p.residual(np.zeros(9))
    with pytest.raises(ValueError, match="date index"):
        p.date_blocks(5, np.tile(y_ss, (5, 1)))


def test_residual_matches_solve_perfect_foresight_convention(ramsey):
    """Same stacking as solve_perfect_foresight: its converged path has residual <= its tolerance."""
    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]
    y_init = np.array([ramsey["c_ss"], 0.5 * ramsey["k_ss"]])
    T = 30
    ref = solve_perfect_foresight(eq, y_init=y_init, y_ss=y_ss, exogenous_path=np.ones(T), n_periods=T,
                                  tol=1e-10, variable_names=ramsey["names"])
    prob = StackedProblem(eq, y_init, y_ss, np.ones(T), variable_names=ramsey["names"])
    R = prob.residual_matrix(ref.path.to_numpy())
    assert np.max(np.abs(R)) <= max(ref.residual_norm, 1e-13) * 1.0001
    # Date t occupies rows t*n_vars:(t+1)*n_vars and uses y_init / y_end at the boundaries.
    Y = prob.linear_initial_path()
    flat = prob.residual(Y)
    yp = Y[1]
    row0 = np.asarray(eq(yp, Y[0], y_init, 1.0), dtype=float)
    rowT = np.asarray(eq(y_ss, Y[-1], Y[-2], 1.0), dtype=float)
    np.testing.assert_allclose(flat[:2], row0, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(flat[-2:], rowT, rtol=0.0, atol=1e-15)
    # The cache returns copies; mutating them does not corrupt later evaluations.
    flat[:] = 1.0
    np.testing.assert_allclose(prob.residual(Y)[:2], row0, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("mode", ["central", "complex", "analytic"])
def test_stacked_jvp_matches_dense_jacobian_ramsey(ramsey, mode):
    """J v equals the column-wise dense Jacobian built with solve_perfect_foresight's FD rule."""
    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]
    y_init = np.array([ramsey["c_ss"], 0.5 * ramsey["k_ss"]])
    T, n = 12, 2
    exo = np.ones(T)
    exo[3] = 1.05
    kwargs = {"jvp_method": "complex"} if mode == "complex" else {}
    if mode == "analytic":
        kwargs = {"jvp_fn": ramsey["jvp_fn"], "blocks_fn": ramsey["blocks_fn"]}
    prob = StackedProblem(eq, y_init, y_ss, exo, **kwargs)
    Y = prob.linear_initial_path() * (1.0 + 0.05 * np.sin(np.arange(T * n)).reshape(T, n))
    J_pf = np.zeros((T * n, T * n))
    for t in range(T):
        yl = y_init if t == 0 else Y[t - 1]
        yp = y_ss if t == T - 1 else Y[t + 1]
        A_t, B_t, C_t = _pf_fd_blocks(eq, yp, Y[t], yl, exo[t], n)
        J_pf[t * n:(t + 1) * n, t * n:(t + 1) * n] = B_t
        if t < T - 1:
            J_pf[t * n:(t + 1) * n, (t + 1) * n:(t + 2) * n] = A_t
        if t > 0:
            J_pf[t * n:(t + 1) * n, (t - 1) * n:t * n] = C_t
    matvec = prob.jvp(Y)
    J_ours = np.column_stack([matvec(np.eye(T * n)[:, j]) for j in range(T * n)])
    # the finite-difference reference itself is accurate to about 2e-9, so exact modes are
    # held to 1e-7 against it and to 1e-12 against each other below
    atol = 1e-6 if mode == "central" else 1e-7
    np.testing.assert_allclose(J_ours, J_pf, rtol=0.0, atol=atol)
    np.testing.assert_allclose(prob.stacked_jacobian(Y).toarray(), J_pf, rtol=0.0, atol=atol)
    if mode != "central":
        # complex step and the analytic Jacobian agree to machine precision with each other
        exact = StackedProblem(eq, y_init, y_ss, exo, jvp_fn=ramsey["jvp_fn"], blocks_fn=ramsey["blocks_fn"])
        J_exact = exact.stacked_jacobian(Y).toarray()
        np.testing.assert_allclose(J_ours, J_exact, rtol=0.0, atol=1e-12)
    # date_blocks vs blocks_fn
    A_t, B_t, C_t = prob.date_blocks(4, Y)
    Ae, Be, Ce = ramsey["blocks_fn"](Y[5], Y[4], Y[3], exo[4])
    tol_blocks = 1e-8 if mode == "central" else 1e-13
    np.testing.assert_allclose(A_t, Ae, rtol=0.0, atol=tol_blocks)
    np.testing.assert_allclose(B_t, Be, rtol=0.0, atol=tol_blocks)
    np.testing.assert_allclose(C_t, Ce, rtol=0.0, atol=tol_blocks)
    # steady blocks are cached, read-only and evaluated at (y_end, y_end, y_end, last exogenous row)
    A0, B0, C0 = prob.steady_blocks()
    assert prob.steady_blocks()[1] is B0 and not B0.flags.writeable
    Ae, Be, Ce = ramsey["blocks_fn"](y_ss, y_ss, y_ss, exo[-1])
    np.testing.assert_allclose(B0, Be, rtol=0.0, atol=tol_blocks)
    assert prob.jvp_method == mode


def test_stacked_jvp_growth_central_vs_complex(growth20):
    g = growth20
    prob_c = StackedProblem(g["eq"], g["y_init"], g["y_ss"], g["exo"])
    prob_z = StackedProblem(g["eq"], g["y_init"], g["y_ss"], g["exo"], jvp_method="complex")
    Y = prob_c.linear_initial_path()
    v = np.random.default_rng(12).normal(size=prob_c.n)
    jc, jz = prob_c.jvp(Y)(v), prob_z.jvp(Y)(v)
    np.testing.assert_allclose(jc, jz, rtol=0.0, atol=1e-6 * np.max(np.abs(jz)))
    # zero direction -> zero product without evaluating anything
    assert np.all(prob_c.jvp(Y)(np.zeros(prob_c.n)) == 0.0)
    assert np.all(prob_z.jvp(Y)(np.zeros(prob_c.n)) == 0.0)


def test_stacked_jvp_factory_override(ramsey):
    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]
    calls = []

    def factory(Y_flat):
        calls.append(Y_flat.copy())
        return lambda v: 2.0 * np.asarray(v)

    prob = StackedProblem(eq, y_ss, y_ss, np.ones(4), stacked_jvp_factory=factory)
    assert prob.jvp_method == "stacked_factory"
    v = np.arange(8.0)
    np.testing.assert_array_equal(prob.jvp(np.tile(y_ss, 4))(v), 2.0 * v)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="wrong size"):
        StackedProblem(eq, y_ss, y_ss, np.ones(4), stacked_jvp_factory=lambda Y: (lambda v: v[:3])).jvp(np.tile(y_ss, 4))(v)


def test_complex_jvp_requires_complex_safe_equations():
    def eq(yp, yc, yl, eps):
        if np.iscomplexobj(yc):
            raise TypeError("real inputs only")
        return [float(yc[0]) - 1.0]

    prob = StackedProblem(eq, np.array([0.5]), np.array([1.0]), n_periods=3, jvp_method="complex")
    with pytest.raises(TypeError, match="complex-safe"):
        prob.jvp(np.full(3, 0.7))(np.ones(3))

    def discards(yp, yc, yl, eps):
        return np.real(np.array([yc[0] ** 2 - 1.0]))     # silently drops the imaginary part

    prob = StackedProblem(discards, np.array([0.5]), np.array([1.0]), n_periods=3, jvp_method="complex")
    with pytest.raises(TypeError, match="discards the imaginary part"):
        prob.jvp(np.full(3, 0.7))(np.ones(3))
    with pytest.raises(TypeError, match="discards the imaginary part"):
        prob.steady_blocks()
    # the same function is fine with central differences
    central = StackedProblem(discards, np.array([0.5]), np.array([1.0]), n_periods=3)
    np.testing.assert_allclose(central.jvp(np.full(3, 0.7))(np.ones(3)), np.full(3, 1.4), rtol=0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# Krylov adapter
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["lgmres", "gmres"])
@pytest.mark.parametrize("side", ["right", "left"])
def test_exact_inverse_identity_one_outer_iteration(ramsey, method, side):
    """A preconditioner built from the CURRENT date blocks makes the Krylov solve converge
    to rtol 1e-12 in a single inner cycle (IO test_finite_block_inverts_original_jacobian analogue)."""
    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]
    y_init = np.array([ramsey["c_ss"], 0.5 * ramsey["k_ss"]])
    T = 40
    prob = StackedProblem(eq, y_init, y_ss, np.ones(T), jvp_method="complex")
    Y = prob.linear_initial_path()
    A, B, C = prob.current_blocks(Y)
    pre = BlockTridiagonalPreconditioner(A, B, C, T)
    op = spla.LinearOperator((prob.n, prob.n), matvec=prob.jvp(Y), dtype=float)
    rng = np.random.default_rng(13)
    for _ in range(3):
        rhs = rng.normal(size=prob.n)
        count = [0]
        d, info, true_res = preconditioned_lgmres(op, rhs, pre, side=side, method=method, rtol=1e-12,
                                                  callback=lambda _: count.__setitem__(0, count[0] + 1))
        assert info == 0
        assert true_res <= 1e-12 * np.linalg.norm(rhs)
        np.testing.assert_allclose(true_res, np.linalg.norm(op @ d - rhs), rtol=1e-12, atol=0.0)
        # one inner cycle: SciPy's callback fires before every outer residual check,
        # so a converged single cycle shows exactly two callbacks (start and final check)
        assert count[0] <= 2
        # the direction is the exact Newton direction of the FD-free Jacobian
        np.testing.assert_allclose(d, pre.apply(rhs), rtol=0.0, atol=1e-10 * np.max(np.abs(d)))


@pytest.mark.parametrize("side", ["left", "right"])
def test_true_residual_of_returned_direction_when_cap_exhausted(side):
    """Port of IO test_final_linear_log_uses_returned_original_direction_when_cap_exhausted."""
    matrix = np.array([[4., .5, 0., .2], [-.3, 2., .4, 0.], [.2, 0., 1., .3], [0., .1, -.4, 3.]])
    weights = np.array([.2, .4, 1.3, .6])
    operator = spla.LinearOperator(matrix.shape, matvec=lambda x: matrix @ x, dtype=float)
    M = spla.LinearOperator(matrix.shape, matvec=lambda x: weights * x, dtype=float)
    rhs = np.array([1., 2., -1., .7])
    seen = []

    def callback(point):
        candidate = weights * point if side == "right" else point
        seen.append(float(np.linalg.norm(matrix @ candidate - rhs)))

    direction, info, true_res = preconditioned_lgmres(operator, rhs, M, side=side, inner_m=1, outer_k=0,
                                                      maxiter=1, rtol=1e-12, callback=callback)
    assert info == 1
    assert true_res == pytest.approx(np.linalg.norm(matrix @ direction - rhs), rel=1e-14)
    assert len(seen) == 1
    assert abs(true_res - seen[-1]) > 0.01
    # a converged solve reports the true residual below the requested tolerance
    d2, info2, res2 = preconditioned_lgmres(operator, rhs, M, side=side, inner_m=4, outer_k=2, maxiter=10, rtol=1e-12)
    assert info2 == 0 and res2 <= max(1e-10, 1e-12 * np.linalg.norm(rhs))


def test_preconditioned_lgmres_validation():
    A = np.eye(3)
    with pytest.raises(ValueError, match="side"):
        preconditioned_lgmres(A, np.ones(3), side="middle")
    with pytest.raises(ValueError, match="method"):
        preconditioned_lgmres(A, np.ones(3), method="bicgstab")
    with pytest.raises(ValueError, match="inner_m"):
        preconditioned_lgmres(A, np.ones(3), inner_m=0)
    with pytest.raises(ValueError, match="rhs"):
        preconditioned_lgmres(A, np.ones(4))
    with pytest.raises(ValueError, match="zero Krylov guess"):
        preconditioned_lgmres(A, np.ones(3), np.eye(3), side="right", x0=np.ones(3))
    d, info, res = preconditioned_lgmres(A, np.ones(3), M=None, side="right", rtol=1e-12)
    assert info == 0 and res < 1e-12 and np.allclose(d, 1.0)


@pytest.mark.parametrize("method", ["lgmres", "gmres"])
def test_preconditioned_lgmres_supports_the_pre_1_12_scipy_signature(monkeypatch, method):
    """SciPy 1.10 and 1.11 (admitted by pyproject) take ``tol`` instead of ``rtol``; the adapter
    reads the installed signature and passes the relative tolerance under the accepted name."""
    import scipy.sparse.linalg as spla_module

    real = getattr(spla_module, method)
    seen = []
    if method == "lgmres":
        def legacy(A, b, x0=None, tol=1e-5, maxiter=1000, M=None, callback=None, inner_m=30, outer_k=3,
                   outer_v=None, store_outer_Av=True, prepend_outer_v=False, atol=None):
            seen.append(tol)
            return real(A, b, x0=x0, rtol=tol, maxiter=maxiter, M=M, callback=callback, inner_m=inner_m,
                        outer_k=outer_k, atol=atol)
    else:
        def legacy(A, b, x0=None, tol=1e-5, restart=None, maxiter=None, M=None, callback=None,
                   atol=None, callback_type=None):
            seen.append(tol)
            return real(A, b, x0=x0, rtol=tol, restart=restart, maxiter=maxiter, M=M, callback=callback,
                        atol=atol, callback_type=callback_type)
    assert "rtol" not in inspect.signature(legacy).parameters
    monkeypatch.setattr(spla_module, method, legacy)
    matrix = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
    d, info, res = preconditioned_lgmres(matrix, np.ones(3), np.eye(3), method=method, rtol=1e-11,
                                         callback=lambda _: None)
    assert seen and seen[0] == 1e-11
    assert info == 0 and res <= 1e-10 and np.allclose(matrix @ d, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Solver parity with solve_perfect_foresight
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["transition", "anticipated_shock_t5", "histval_endval"])
def test_parity_with_solve_perfect_foresight_ramsey(ramsey, case):
    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]
    T = 60
    if case == "transition":
        y_init, y_end, exo = np.array([ramsey["c_ss"], 0.5 * ramsey["k_ss"]]), y_ss, np.ones(T)
    elif case == "anticipated_shock_t5":
        y_init, y_end, exo = y_ss, y_ss, np.ones(T)
        exo[4] = 1.05
    else:
        y_init, y_end, exo = y_ss, ramsey["steady_state"](1.05), np.full(T, 1.05)
    ref = solve_perfect_foresight(eq, y_init=y_init, y_end=y_end, exogenous_path=exo, n_periods=T,
                                  tol=1e-8, variable_names=ramsey["names"])
    assert ref.converged
    prob = StackedProblem(eq, y_init, y_end, exo, variable_names=ramsey["names"])
    res = solve_stacked_newton_krylov(prob, tol=1e-8)
    assert res.converged and res.iterations >= 1
    np.testing.assert_allclose(res.to_numpy(), ref.path.to_numpy(), rtol=0.0, atol=1e-8)
    assert res.residual_norm <= 1e-8 * res.metadata["tolerance_scale"]
    assert res.terminal_error == pytest.approx(ref.terminal_error, abs=1e-8)
    assert res.terminal_error == pytest.approx(float(np.max(np.abs(res.to_numpy()[-1] - y_end))), abs=0.0)
    assert res.initial_residual_norm == pytest.approx(ref.initial_residual_norm, rel=1e-12)
    assert list(res.path.columns) == ramsey["names"] and res.path.index.name == "t" and res.path.index[0] == 1
    pf = res.to_perfect_foresight_result()
    assert isinstance(pf, PerfectForesightResult)
    assert pf.converged and pf.iterations == res.iterations and pf.residual_norm == res.residual_norm
    assert pf.terminal_error == res.terminal_error and pf.variable_names == tuple(ramsey["names"])
    pd.testing.assert_frame_equal(pf.path, res.path)


def test_parity_growth_fixture(growth20, growth20_reference):
    """20-sector transition: same path as the dense-FD SuperLU solver to 1e-8 (observed 3.5e-11)."""
    g = growth20
    prob = StackedProblem(g["eq"], g["y_init"], g["y_ss"], g["exo"])
    res = solve_stacked_newton_krylov(prob, tol=1e-9, krylov_maxiter=40)
    assert res.converged
    np.testing.assert_allclose(res.to_numpy(), growth20_reference.path.to_numpy(), rtol=0.0, atol=1e-8)
    outer = res.krylov_outer_iterations()
    assert len(outer) == res.iterations + 1 and max(outer) <= 4
    assert all(h["krylov_status"] == 0 for h in res.linear_history)
    assert all(h["true_linear_residual"] <= h["requested_tolerance"] * (1 + 1e-9) for h in res.linear_history)
    assert res.preconditioner == "block_tridiagonal[thomas, exact]"
    splu = solve_stacked_newton_krylov(prob, preconditioner="steady_splu", tol=1e-9)
    np.testing.assert_allclose(splu.to_numpy(), res.to_numpy(), rtol=0.0, atol=1e-9)
    assert splu.preconditioner == "block_tridiagonal[splu, exact]"


def test_right_left_and_gmres_agree(growth20, growth20_reference):
    g = growth20
    prob = StackedProblem(g["eq"], g["y_init"], g["y_ss"], g["exo"])
    right = solve_stacked_newton_krylov(prob, tol=1e-9, side="right")
    left = solve_stacked_newton_krylov(prob, tol=1e-9, side="left")
    gmres = solve_stacked_newton_krylov(prob, tol=1e-9, krylov_method="gmres")
    ref = growth20_reference.path.to_numpy()
    for r in (right, left, gmres):
        assert r.converged
        np.testing.assert_allclose(r.to_numpy(), ref, rtol=0.0, atol=1e-9)
    assert left.metadata["side"] == "left" and gmres.metadata["krylov_method"] == "gmres"


def test_analytic_jvp_and_blocks_on_ramsey_solver(ramsey):
    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]
    y_init = np.array([ramsey["c_ss"], 0.5 * ramsey["k_ss"]])
    T = 80
    ref = solve_perfect_foresight(eq, y_init=y_init, y_ss=y_ss, exogenous_path=np.ones(T), n_periods=T,
                                  tol=1e-10, variable_names=ramsey["names"])
    prob = StackedProblem(eq, y_init, y_ss, np.ones(T), jvp_fn=ramsey["jvp_fn"], blocks_fn=ramsey["blocks_fn"],
                          variable_names=ramsey["names"])
    res = solve_stacked_newton_krylov(prob, tol=1e-10)
    assert res.converged and res.metadata["jvp_method"] == "analytic"
    np.testing.assert_allclose(res.to_numpy(), ref.path.to_numpy(), rtol=0.0, atol=1e-9)
    # steady blocks come from the analytic jvp when only jvp_fn is given
    only_jvp = StackedProblem(eq, y_init, y_ss, np.ones(T), jvp_fn=ramsey["jvp_fn"])
    for ours, exact in zip(only_jvp.steady_blocks(), ramsey["blocks_fn"](y_ss, y_ss, y_ss, 1.0)):
        np.testing.assert_allclose(ours, exact, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("scale", [1e-12, 1e-9, 1e-4, 1.0, 1e4, 1e8])
def test_equation_scaling_invariance(ramsey, scale):
    """Rescaling every equation must not change the answer (tests/test_dynare_perfect_foresight port)."""
    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]
    y_init = np.array([ramsey["c_ss"], 0.5 * ramsey["k_ss"]])
    T = 60

    def scaled(yp, yc, yl, eps):
        return [scale * r for r in eq(yp, yc, yl, eps)]

    truth = solve_stacked_newton_krylov(StackedProblem(eq, y_init, y_ss, np.ones(T)), tol=1e-8)
    res = solve_stacked_newton_krylov(StackedProblem(scaled, y_init, y_ss, np.ones(T)), tol=1e-8)
    assert truth.converged and res.converged and res.iterations >= 1
    # the step test accepts a final direction up to 1e-10 * max(1, |Y|_inf) = 3.5e-10 here (the
    # solve_perfect_foresight contract), so two converged solves may differ by that much
    np.testing.assert_allclose(res.to_numpy(), truth.to_numpy(), rtol=0.0, atol=1e-9)
    assert res.metadata["tolerance_scale"] >= 1.0


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3, 1e4])
def test_variable_scaling_invariance(ramsey, scale):
    """The Ramsey model in units of `scale` converges with default options in at most ten steps to
    the unit-scale path (path / scale) to 1e-9 relative: the step bound is in path units."""
    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]
    y_init = np.array([ramsey["c_ss"], 0.5 * ramsey["k_ss"]])
    T = 60

    def scaled(yp, yc, yl, eps):
        return eq(np.asarray(yp) / scale, np.asarray(yc) / scale, np.asarray(yl) / scale, eps)

    base = solve_stacked_newton_krylov(StackedProblem(eq, y_init, y_ss, np.ones(T)), tol=1e-8)
    res = solve_stacked_newton_krylov(StackedProblem(scaled, scale * y_init, scale * y_ss, np.ones(T)), tol=1e-8)
    assert base.converged and res.converged
    assert 1 <= res.iterations <= 10
    reference = base.to_numpy()
    np.testing.assert_allclose(res.to_numpy() / scale, reference, rtol=0.0, atol=1e-9 * max(1.0, np.max(np.abs(reference))))
    if scale >= 1.0:
        # the bound never binds differently: same Newton and Krylov counts as the unit-scale solve
        assert res.iterations == base.iterations
        assert res.krylov_outer_iterations() == base.krylov_outer_iterations()


def test_singular_leading_pivot_falls_back_to_superlu():
    """Steady B of rank 1 with a nonsingular stacked matrix (cond about 491): block-Thomas cannot
    factor it, the default preconditioner falls back to SuperLU and converges, the fallback is
    recorded in the name, and a matrix SuperLU cannot factor raises StackedNewtonError with the
    initial state attached."""
    def eq(yp, yc, yl, eps):
        return [yc[0] + yc[1] - 0.5 * yl[0] - eps, yc[0] + yc[1] - 0.5 * yp[1] - 0.3 * yl[0]]

    T = 8
    prob = StackedProblem(eq, np.array([1.0, 0.5]), np.zeros(2), np.zeros(T))
    A, B, C = prob.steady_blocks()
    assert np.linalg.matrix_rank(B) == 1
    J = prob.stacked_jacobian(prob.linear_initial_path()).toarray()
    assert np.linalg.cond(J) < 1e3
    with pytest.raises(ValueError, match="pivot D_1 is singular"):
        BlockTridiagonalPreconditioner(A, B, C, T, method="thomas")
    splu = BlockTridiagonalPreconditioner(A, B, C, T, method="splu")
    r = np.random.default_rng(3).normal(size=2 * T)
    np.testing.assert_allclose(splu.apply(r), np.linalg.solve(_dense_stacked(A, B, C, T), r), rtol=0.0, atol=1e-12)
    pf = solve_perfect_foresight(eq, y_init=np.array([1.0, 0.5]), y_ss=np.zeros(2), exogenous_path=np.zeros(T),
                                 n_periods=T, tol=1e-10)
    assert pf.converged
    for name in ("steady_block_tridiagonal", "steady_splu", "current_block_tridiagonal"):
        res = solve_stacked_newton_krylov(prob, preconditioner=name, tol=1e-10)
        assert res.converged and res.residual_norm <= 1e-10 * res.metadata["tolerance_scale"]
        np.testing.assert_allclose(res.to_numpy(), pf.path.to_numpy(), rtol=0.0, atol=1e-9)
    default = solve_stacked_newton_krylov(prob, tol=1e-10)
    assert default.preconditioner == "block_tridiagonal[splu, exact, thomas_pivot_fallback]"
    assert solve_stacked_newton_krylov(prob, preconditioner="steady_splu", tol=1e-10).preconditioner == "block_tridiagonal[splu, exact]"
    # block Jacobi in time with H = 1 inverts B alone, which is singular for both methods
    with pytest.raises(StackedNewtonError, match="cannot be built") as excinfo:
        solve_stacked_newton_krylov(prob, preconditioner="time_block_jacobi", tol=1e-10)
    failed = excinfo.value.result
    assert failed.converged is False and failed.iterations == 0
    assert failed.metadata["converged_by"] == "preconditioner_failure" and failed.preconditioner == "time_block_jacobi"
    np.testing.assert_array_equal(failed.to_numpy(), prob.linear_initial_path())


def _rank_deficient_linear_problem(seed, n=5, T=100):
    """Linear system A y_{t+1} + B y_t + C y_{t-1} = b (1 + eps_t) with B = U V of rank n - 1.

    With exact blocks (from jvp_fn) the LU pivot of B is rounding-level (about 1e-16),
    not exactly zero, while the stacked matrix is nonsingular (cond 616 for seed 0 and
    941 for seed 3). The block-Thomas elimination would then return a wrong 'exact
    inverse' (relative residual 0.23 and 1.02 before the conditioning test existed).
    """
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, n - 1)) @ rng.normal(size=(n - 1, n))
    A = 0.3 * rng.normal(size=(n, n))
    C = 0.3 * rng.normal(size=(n, n))
    b = np.linspace(1.0, 2.0, n)
    ystar = np.linalg.solve(A + B + C, b)

    def eq(yp, yc, yl, eps):
        return A @ yp + B @ yc + C @ yl - b * (1.0 + float(eps))

    def jvp(yp, yc, yl, eps, vp, vc, vl):
        return A @ vp + B @ vc + C @ vl

    exo = np.zeros(T)
    exo[2] = 0.3
    return A, B, C, T, eq, jvp, ystar, exo


@pytest.mark.parametrize("seed", [0, 3])
def test_rounding_level_pivot_is_rejected_and_falls_back(seed):
    """A rank-deficient B whose LU pivot is not exactly zero: block-Thomas raises instead of
    returning a wrong inverse, SuperLU inverts the stacked matrix to 1e-12, and the default
    solver falls back to SuperLU and matches solve_perfect_foresight to 1e-8 (observed ~1e-14)."""
    A, B, C, T, eq, jvp, ystar, exo = _rank_deficient_linear_problem(seed)
    assert np.linalg.matrix_rank(B) == B.shape[0] - 1
    pivots = np.abs(np.diag(scipy.linalg.lu_factor(B)[0]))
    # Rounding level on Accelerate/MKL; OpenBLAS returns an exact zero for seed 0.
    # Either way block-Thomas must reject it, which is what the rest checks.
    assert pivots.min() < 1e-14
    for H in (None, 4):
        with pytest.raises(ValueError, match="pivot D_1 is singular"):
            BlockTridiagonalPreconditioner(A, B, C, T, method="thomas", time_block=H)
    splu = BlockTridiagonalPreconditioner(A, B, C, T, method="splu")
    J = splu.to_sparse().toarray()
    assert np.linalg.cond(J) < 1e3
    r = np.random.default_rng(seed + 7).normal(size=T * B.shape[0])
    x = np.linalg.solve(J, r)
    np.testing.assert_allclose(splu.apply(r), x, rtol=0.0, atol=1e-12 * np.max(np.abs(x)))
    prob = StackedProblem(eq, ystar, ystar, exo, jvp_fn=jvp)
    res = solve_stacked_newton_krylov(prob, tol=1e-10)
    assert res.converged and res.preconditioner == "block_tridiagonal[splu, exact, thomas_pivot_fallback]"
    assert max(res.krylov_outer_iterations()) <= 2
    pf = solve_perfect_foresight(eq, y_init=ystar, y_ss=ystar, exogenous_path=exo, n_periods=T, tol=1e-10)
    assert pf.converged
    np.testing.assert_allclose(res.to_numpy(), pf.path.to_numpy(), rtol=0.0, atol=1e-8)


def test_ill_conditioned_pivot_is_caught_by_the_backward_error_check():
    """B = [[1, 1], [1, 1 + 1e-12]] passes the pivot conditioning test (rcond about 2.5e-13 above
    2 eps), but the unpivoted block elimination loses about ten digits on a stacked matrix with
    cond 1.3e3 (normwise backward error about 2e-6). The one-vector accuracy check rejects the
    factorization, and the default solver falls back to SuperLU and solves the system."""
    B = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-12]])
    A = np.array([[0.3, 0.1], [-0.2, 0.4]])
    C = np.array([[0.2, -0.1], [0.5, 0.1]])
    T = 8
    with pytest.raises(ValueError, match="backward error"):
        BlockTridiagonalPreconditioner(A, B, C, T)
    splu = BlockTridiagonalPreconditioner(A, B, C, T, method="splu")
    J = splu.to_sparse().toarray()
    assert np.linalg.cond(J) < 1e4
    b = np.array([1.0, 0.5])
    ystar = np.linalg.solve(A + B + C, b)

    def eq(yp, yc, yl, eps):
        return A @ yp + B @ yc + C @ yl - b * (1.0 + float(eps))

    exo = np.zeros(T)
    exo[1] = 0.2
    prob = StackedProblem(eq, ystar, ystar, exo, jvp_fn=lambda yp, yc, yl, e, vp, vc, vl: A @ vp + B @ vc + C @ vl)
    res = solve_stacked_newton_krylov(prob, tol=1e-10)
    assert res.converged and res.preconditioner.endswith("thomas_pivot_fallback]")
    rhs = np.concatenate([b * (1.0 + exo[t]) - (C @ ystar if t == 0 else 0.0) - (A @ ystar if t == T - 1 else 0.0)
                          for t in range(T)])
    np.testing.assert_allclose(res.to_numpy().ravel(), np.linalg.solve(J, rhs), rtol=0.0, atol=1e-9)


def test_block_thomas_accuracy_check_passes_on_regular_blocks():
    """The backward-error check never fires on the regular fixtures: the operator it accepts
    satisfies |J0 x - r| / (|J0| |x| + |r|) <= 1e-14 (observed <= 2e-16)."""
    for seed in range(5):
        for per_date in (False, True):
            for H in (None, 1, 3):
                A, B, C = _random_blocks(7, 5, seed=seed, per_date=per_date)
                pre = BlockTridiagonalPreconditioner(A, B, C, 7, time_block=H)
                J = pre.to_sparse().toarray()
                r = np.random.default_rng(seed).normal(size=35)
                x = pre.apply(r)
                backward = np.max(np.abs(J @ x - r)) / (np.max(np.abs(J).sum(axis=1)) * np.max(np.abs(x)) + np.max(np.abs(r)))
                assert backward <= 1e-14


def test_never_returns_the_untouched_initial_guess(ramsey):
    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]
    y_init = np.array([ramsey["c_ss"], 0.5 * ramsey["k_ss"]])
    T = 60

    def tiny(yp, yc, yl, eps):
        return [1e-9 * r for r in eq(yp, yc, yl, eps)]

    prob = StackedProblem(tiny, y_init, y_ss, np.ones(T))
    res = solve_stacked_newton_krylov(prob, tol=1e-8)
    assert res.initial_residual_norm < 1e-8      # below the raw absolute tolerance ...
    assert res.iterations >= 1                     # ... yet the solver did work
    assert np.max(np.abs(res.to_numpy() - prob.linear_initial_path())) > 0.1


def test_converged_start_takes_zero_steps(ramsey):
    """A path that already solves the system is accepted after one step check, never 'improved'."""
    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]
    prob = StackedProblem(eq, y_ss, y_ss, np.ones(10))
    res = solve_stacked_newton_krylov(prob, tol=1e-8)
    assert res.converged and res.iterations == 0 and len(res.linear_history) == 1
    assert res.linear_history[0]["convergence_check"] is True and res.linear_history[0]["step"] == 0.0
    assert res.metadata["converged_by"] == "residual_and_step"
    np.testing.assert_allclose(res.to_numpy(), np.tile(y_ss, (10, 1)), rtol=0.0, atol=1e-12)
    # step_tol=None (or inf) is the IO residual-only rule: accepted without any linear solve
    for residual_only in (None, float("inf")):
        io_rule = solve_stacked_newton_krylov(prob, np.tile(y_ss, (10, 1)), tol=1e-8, step_tol=residual_only)
        assert io_rule.converged and io_rule.iterations == 0 and io_rule.linear_history == ()
        assert io_rule.metadata["converged_by"] == "residual" and io_rule.metadata["step_tol"] is None


# ---------------------------------------------------------------------------
# Failure contract
# ---------------------------------------------------------------------------
def test_failure_contract_when_krylov_cap_is_exhausted(growth20):
    """Unpreconditioned with a tiny Krylov budget and max_iter=3: every direction is inexact and
    recorded as such; the solve ends in a failed line search or the iteration limit (an
    exhausted cap alone is not fatal, see test_krylov_cap_is_recorded_not_fatal), and the
    attached result carries converged=False with the recorded Krylov statuses."""
    g = growth20
    prob = StackedProblem(g["eq"], g["y_init"], g["y_ss"], g["exo"])
    with pytest.raises(StackedNewtonError) as excinfo:
        solve_stacked_newton_krylov(prob, preconditioner=None, tol=1e-9, krylov_maxiter=2, inner_m=2,
                                    outer_k=0, max_iter=3)
    err = excinfo.value
    assert isinstance(err.result, StackedNewtonResult) and err.result.converged is False
    assert err.result.preconditioner == "none"
    history = err.result.linear_history
    assert history and all(h["krylov_status"] > 0 for h in history)
    assert all(h["true_linear_residual"] > h["requested_tolerance"] for h in history)
    assert np.isfinite(err.result.to_numpy()).all()
    assert err.result.metadata["converged_by"] in ("iteration_limit", "line_search_failure")


def test_krylov_cap_is_recorded_not_fatal(ramsey):
    """An exhausted Krylov cap is recorded (krylov_status > 0, true residual above the requested
    tolerance) and the inexact direction is line-searched, as in the IO engine; acceptance rests
    on the recomputed residual, and the path matches solve_perfect_foresight to 1e-8 (observed 6e-11)."""
    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]
    y_init = np.array([ramsey["c_ss"], 0.5 * ramsey["k_ss"]])
    T = 60
    prob = StackedProblem(eq, y_init, y_ss, np.ones(T))
    res = solve_stacked_newton_krylov(prob, preconditioner="time_block_jacobi", krylov_maxiter=2, tol=1e-8)
    capped = [h for h in res.linear_history if h["krylov_status"] > 0]
    assert res.converged and capped
    assert all(h["krylov_outer_iterations"] >= 2 for h in capped)
    assert any(h["true_linear_residual"] > h["requested_tolerance"] for h in capped)
    assert res.residual_norm <= 1e-8 * res.metadata["tolerance_scale"]
    ref = solve_perfect_foresight(eq, y_init=y_init, y_ss=y_ss, exogenous_path=np.ones(T), n_periods=T, tol=1e-8)
    np.testing.assert_allclose(res.to_numpy(), ref.path.to_numpy(), rtol=0.0, atol=1e-8)


@pytest.mark.parametrize("case", ["structured_nonfinite_right", "structured_nonfinite_left", "jvp_raises",
                                  "jvp_nonfinite", "factory_operator_raises"])
def test_linear_solve_failure_attaches_the_last_state(ramsey, case):
    """Errors and nonfinite values raised inside a Krylov solve end the solve with
    StackedNewtonError carrying the last state (converged=False, converged_by
    'linear_solve_failure') instead of a bare exception or result=None."""
    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]
    y_init = np.array([ramsey["c_ss"], 0.5 * ramsey["k_ss"]])
    T = 60
    kwargs = {"preconditioner": "steady_block_tridiagonal"}
    problem_kwargs = {}
    if case.startswith("structured_nonfinite"):
        calls = [0]

        def diag_solve(t, r):
            calls[0] += 1
            return np.full_like(r, np.inf) if calls[0] > 150 else r

        kwargs = {"preconditioner": StructuredBlockTridiagonalPreconditioner(T, 2, diag_solve=diag_solve, sweep="jacobi"),
                  "side": case.rsplit("_", 1)[1], "krylov_maxiter": 3}
        message = "Nonfinite structured preconditioner action"
    elif case == "jvp_raises":
        def raising(yp, yc, yl, e, vp, vc, vl):
            raise ValueError("domain error in jvp")
        problem_kwargs = {"jvp_fn": raising}
        kwargs = {"preconditioner": None, "jacobian_scale": 1.0}
        message = "domain error in jvp"
    elif case == "jvp_nonfinite":
        # finite at the steady state (so the steady preconditioner builds), NaN along the transition
        problem_kwargs = {"jvp_fn": lambda yp, yc, yl, e, vp, vc, vl:
                          np.full(2, np.nan) if yc[1] < 0.99 * ramsey["k_ss"] else ramsey["jvp_fn"](yp, yc, yl, e, vp, vc, vl)}
        message = "Nonfinite Jacobian-vector product"
    else:
        def factory(x):
            def matvec(v):
                raise np.linalg.LinAlgError("inner factorization failed")
            return spla.LinearOperator((2 * T, 2 * T), matvec=matvec, dtype=float)
        kwargs = {"preconditioner": factory}
        message = "inner factorization failed"
    prob = StackedProblem(eq, y_init, y_ss, np.ones(T), **problem_kwargs)
    with pytest.raises(StackedNewtonError, match="Linear solve failed") as excinfo:
        solve_stacked_newton_krylov(prob, tol=1e-8, **kwargs)
    assert message in str(excinfo.value)
    res = excinfo.value.result
    assert isinstance(res, StackedNewtonResult) and res.converged is False
    assert res.metadata["converged_by"] == "linear_solve_failure"
    assert res.linear_history[-1]["linear_solve_failed"] is True and message in res.linear_history[-1]["error"]
    assert np.isfinite(res.to_numpy()).all() and res.residual_norm == pytest.approx(res.initial_residual_norm)


def test_type_error_from_user_code_propagates(ramsey):
    """A TypeError from user callables signals a programming error and is not wrapped."""
    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]

    def broken(yp, yc, yl, e, vp, vc, vl):
        raise TypeError("bad call")

    prob = StackedProblem(eq, np.array([ramsey["c_ss"], 0.5 * ramsey["k_ss"]]), y_ss, np.ones(20), jvp_fn=broken,
                          blocks_fn=ramsey["blocks_fn"])
    with pytest.raises(TypeError, match="bad call"):
        solve_stacked_newton_krylov(prob, tol=1e-8)


@pytest.mark.parametrize("options", [{}, {"line_search": False, "max_iter": 6}])
def test_step_test_failure_on_a_noisy_residual_floor(options):
    """A residual with a 1e-12 noise floor meets tol * s, but steps on the noise stay near 1e-9:
    the default step test cannot be met, and the solve raises with converged_by
    'step_test_failure' and a message naming the cause (through the failed line search, or
    through the iteration limit when line_search=False) instead of reporting a line-search
    or iteration failure at a residual below the tolerance. step_tol=None accepts."""
    prob = _noisy_floor_problem(20)
    with pytest.raises(StackedNewtonError, match="noise floor") as excinfo:
        solve_stacked_newton_krylov(prob, tol=1e-8, **options)
    res = excinfo.value.result
    assert res.converged is False and res.metadata["converged_by"] == "step_test_failure"
    assert res.residual_norm <= 1e-8 * res.metadata["tolerance_scale"]
    if options:
        assert "iteration limit (6)" in str(excinfo.value)
    else:
        assert "line search along it failed" in str(excinfo.value)
    accepted = solve_stacked_newton_krylov(prob, tol=1e-8, step_tol=None, **options)
    assert accepted.converged and accepted.metadata["converged_by"] == "residual"
    assert accepted.residual_norm <= 1e-8 * accepted.metadata["tolerance_scale"]
    # the accepted path is the noiseless solution to the noise level
    noiseless = StackedProblem(lambda yp, yc, yl, e: 1e-3 * (yc - 0.5 * yl - 0.4 * yp - 0.1),
                               np.array([0.5]), np.array([1.0]), n_periods=20,
                               jvp_fn=lambda yp, yc, yl, e, vp, vc, vl: 1e-3 * (vc - 0.5 * vl - 0.4 * vp))
    exact = solve_stacked_newton_krylov(noiseless, tol=1e-12)
    np.testing.assert_allclose(accepted.to_numpy(), exact.to_numpy(), rtol=0.0, atol=1e-6)


def test_line_search_failure_raises_with_state():
    """Every trial point raises inside equations_fn (the IO line search treats it as rejected)."""
    def eq(yp, yc, yl, eps):
        if abs(yc[0] - 0.3) > 1e-12:
            raise ValueError("outside the installation domain")
        return [yc[0] - 1.0]

    prob = StackedProblem(eq, np.array([0.3]), np.array([1.0]), n_periods=3,
                          jvp_fn=lambda yp, yc, yl, eps, vp, vc, vl: vc,
                          blocks_fn=lambda yp, yc, yl, eps: (np.zeros((1, 1)), np.eye(1), np.zeros((1, 1))))
    with pytest.raises(StackedNewtonError, match="line search failed") as excinfo:
        solve_stacked_newton_krylov(prob, np.full((3, 1), 0.3), max_backtracks=5)
    res = excinfo.value.result
    assert res.converged is False and res.iterations == 0
    assert res.linear_history[-1]["line_search_failed"] is True
    assert res.linear_history[-1]["backtracks"] == 5
    assert res.metadata["converged_by"] == "line_search_failure"
    np.testing.assert_array_equal(res.to_numpy(), np.full((3, 1), 0.3))
    # default: 24 trial evaluations (23 halvings), the IO native_solver._newton range(24) rule
    trials = []

    def counting(yp, yc, yl, eps):
        trials.append(float(yc[0]))
        return eq(yp, yc, yl, eps)

    prob = StackedProblem(counting, np.array([0.3]), np.array([1.0]), n_periods=3,
                          jvp_fn=lambda yp, yc, yl, eps, vp, vc, vl: vc,
                          blocks_fn=lambda yp, yc, yl, eps: (np.zeros((1, 1)), np.eye(1), np.zeros((1, 1))))
    with pytest.raises(StackedNewtonError, match="line search failed") as excinfo:
        solve_stacked_newton_krylov(prob, np.full((3, 1), 0.3))
    last = excinfo.value.result.linear_history[-1]
    assert last["backtracks"] == 23
    tried = [c for c in trials if abs(c - 0.3) > 1e-12]
    assert len(tried) == 24     # each trial raises at its first date
    # the recorded step is the last step actually tried: d = 0.7 per date, |d|_inf < 0.75 so the
    # first trial step is 1, and the 24th trial uses 2**-23 (not 2**-24)
    assert last["step"] == 0.5 ** 23
    assert tried[-1] - 0.3 == pytest.approx(last["step"] * 0.7, rel=1e-6)


def test_iteration_limit_raises_with_last_state(ramsey):
    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]
    prob = StackedProblem(eq, np.array([ramsey["c_ss"], 0.5 * ramsey["k_ss"]]), y_ss, np.ones(60))
    with pytest.raises(StackedNewtonError, match="iteration limit") as excinfo:
        solve_stacked_newton_krylov(prob, tol=1e-8, max_iter=1)
    res = excinfo.value.result
    assert res.converged is False and res.iterations == 1
    assert res.residual_norm < res.initial_residual_norm
    assert res.metadata["converged_by"] == "iteration_limit"


def test_nonfinite_initial_path_and_option_validation(ramsey):
    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]
    prob = StackedProblem(eq, y_ss, y_ss, np.ones(5))
    kw = dict(tol=1e-8)
    with pytest.raises(TypeError, match="StackedProblem"):
        solve_stacked_newton_krylov(eq)
    with pytest.raises(ValueError, match="Y0 must be finite"):
        solve_stacked_newton_krylov(prob, np.full((5, 2), np.nan))
    with pytest.raises(ValueError, match="entries"):
        solve_stacked_newton_krylov(prob, np.ones((4, 2)))
    with np.errstate(all="ignore"), pytest.raises(ValueError, match="nonfinite residual"):
        solve_stacked_newton_krylov(prob, np.zeros((5, 2)))      # c = 0 -> c**(-sigma) infinite
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="tol must be"):
            solve_stacked_newton_krylov(prob, tol=bad)
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError, match="max_iter"):
            solve_stacked_newton_krylov(prob, max_iter=bad, **kw)
    with pytest.raises(ValueError, match="forcing"):
        solve_stacked_newton_krylov(prob, forcing=(0.5, 0.1), **kw)
    with pytest.raises(ValueError, match="forcing"):
        solve_stacked_newton_krylov(prob, forcing=(1e-5,), **kw)
    with pytest.raises(ValueError, match="side"):
        solve_stacked_newton_krylov(prob, side="both", **kw)
    with pytest.raises(ValueError, match="krylov_method"):
        solve_stacked_newton_krylov(prob, krylov_method="cg", **kw)
    with pytest.raises(ValueError, match="unknown preconditioner"):
        solve_stacked_newton_krylov(prob, preconditioner="galerkin", **kw)
    with pytest.raises(TypeError, match="preconditioner must be"):
        solve_stacked_newton_krylov(prob, preconditioner=3.0, **kw)
    with pytest.raises(ValueError, match="time_block"):
        solve_stacked_newton_krylov(prob, time_block=6, **kw)
    with pytest.raises(ValueError, match="jacobian_scale"):
        solve_stacked_newton_krylov(prob, jacobian_scale=0.0, **kw)
    with pytest.raises(ValueError, match="max_log_move"):
        solve_stacked_newton_krylov(prob, max_log_move=-1.0, **kw)
    with pytest.raises(TypeError, match="progress"):
        solve_stacked_newton_krylov(prob, progress="print", **kw)
    with pytest.raises(TypeError, match="checkpoint"):
        solve_stacked_newton_krylov(prob, checkpoint=1, **kw)


# ---------------------------------------------------------------------------
# Options: MINPACK pre-solve, callbacks, user preconditioners, exports
# ---------------------------------------------------------------------------
def test_direct_below_uses_minpack_and_records_it(ramsey):
    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]
    y_init = np.array([ramsey["c_ss"], 0.5 * ramsey["k_ss"]])
    prob = StackedProblem(eq, y_init, y_ss, np.ones(60), variable_names=ramsey["names"])
    plain = solve_stacked_newton_krylov(prob, tol=1e-8)
    assert plain.linear_history[0].get("method") is None
    res = solve_stacked_newton_krylov(prob, tol=1e-8, direct_below=240)
    assert res.converged
    assert res.linear_history[0]["method"] == "MINPACK" and res.linear_history[0]["evaluations"] > 0
    np.testing.assert_allclose(res.to_numpy(), plain.to_numpy(), rtol=0.0, atol=1e-8)
    assert res.metadata["direct_below"] == 240


def test_progress_and_checkpoint_callbacks(ramsey):
    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]
    y_init = np.array([ramsey["c_ss"], 0.5 * ramsey["k_ss"]])
    prob = StackedProblem(eq, y_init, y_ss, np.ones(40))
    seen, saved = [], []
    res = solve_stacked_newton_krylov(prob, tol=1e-8, progress=seen.append,
                                      checkpoint=lambda x, rec: saved.append((x, rec)))
    assert len(seen) == res.iterations and len(saved) == res.iterations
    assert all(rec["newton_iteration"] == i + 1 for i, rec in enumerate(seen))
    np.testing.assert_array_equal(saved[-1][0], res.to_numpy().ravel())
    # the checkpoint holds a copy: mutating it cannot change the result
    saved[-1][0][:] = 0.0
    assert np.all(res.to_numpy() != 0.0)


def test_user_supplied_preconditioners(growth20, growth20_reference):
    g = growth20
    prob = StackedProblem(g["eq"], g["y_init"], g["y_ss"], g["exo"])
    A, B, C = prob.steady_blocks()
    ref = growth20_reference.path.to_numpy()
    block5 = BlockTridiagonalPreconditioner(A, B, C, g["T"], time_block=5)
    res = solve_stacked_newton_krylov(prob, preconditioner=block5, tol=1e-9, krylov_maxiter=60)
    assert res.converged and res.preconditioner == "block_tridiagonal[thomas, time_block=5]"
    np.testing.assert_allclose(res.to_numpy(), ref, rtol=0.0, atol=1e-8)
    exact = BlockTridiagonalPreconditioner(A, B, C, g["T"])
    res_op = solve_stacked_newton_krylov(prob, preconditioner=exact.as_linear_operator(), tol=1e-9)
    assert res_op.converged and res_op.preconditioner == "LinearOperator"
    np.testing.assert_allclose(res_op.to_numpy(), ref, rtol=0.0, atol=1e-8)
    calls = []

    def factory(x):
        calls.append(x.copy())
        return exact

    res_f = solve_stacked_newton_krylov(prob, preconditioner=factory, tol=1e-9)
    assert res_f.converged and res_f.preconditioner == "factory" and len(calls) == len(res_f.linear_history)
    np.testing.assert_allclose(res_f.to_numpy(), ref, rtol=0.0, atol=1e-8)
    named = solve_stacked_newton_krylov(prob, preconditioner="time_block_jacobi", time_block=10, tol=1e-9,
                                        krylov_maxiter=60)
    assert named.converged and named.preconditioner == "block_tridiagonal[thomas, time_block=10]"
    current = solve_stacked_newton_krylov(prob, preconditioner="current_block_tridiagonal", tol=1e-9)
    assert current.converged and current.preconditioner == "current_block_tridiagonal"
    assert max(current.krylov_outer_iterations()) <= 2
    np.testing.assert_allclose(current.to_numpy(), ref, rtol=0.0, atol=1e-8)


def test_result_exports_summary_and_plot(ramsey):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]
    y_init = np.array([ramsey["c_ss"], 0.5 * ramsey["k_ss"]])
    prob = StackedProblem(eq, y_init, y_ss, np.ones(30), variable_names=ramsey["names"])
    res = solve_stacked_newton_krylov(prob, tol=1e-8)
    md = res.to_markdown(head=3)
    header = md.splitlines()[0]
    assert md.count("\n") == 4 and [cell.strip() for cell in header.strip("|").split("|")] == ["t", "c", "k"]
    assert "\\begin{tabular}" in res.to_latex(head=2)
    assert "#table(" in res.to_typst(head=2)
    assert "CONVERGED" in res.summary() and "block_tridiagonal" in res.summary()
    frame = res.linear_history_frame()
    assert isinstance(frame, pd.DataFrame) and "true_linear_residual" in frame.columns
    assert res["c"].equals(res.path["c"]) and res["converged"] is True
    with pytest.raises(KeyError):
        res["missing"]
    assert res.to_dataframe() is not res.path and res.to_frame().equals(res.path)
    assert res.n_periods == 30 and res.n_vars == 2
    fig, ax = plt.subplots()
    out = res.plot(["k"], ax=ax, title="k path")
    assert out is fig and ax.get_title() == "k path"
    plt.close(fig)
    with pytest.raises(ValueError, match="None of requested"):
        res.plot(["zzz"], ax=ax)
    with pytest.raises(Exception):
        res.converged = False      # frozen dataclass
    for key in ("y_init", "y_end", "exogenous_path"):
        assert not res.metadata[key].flags.writeable
    with pytest.raises(KeyError):
        res[0]                     # non-string keys are KeyError, not TypeError
    # the stored trajectory and history records are read-only; copies are writeable
    with pytest.raises(ValueError, match="read-only"):
        res.path.iloc[0, 0] = 99.0
    with pytest.raises(TypeError, match="read-only"):
        res.linear_history[0]["step"] = -1.0
    frame = res.to_dataframe()
    frame.iloc[0, 0] = 99.0
    assert res.path.iloc[0, 0] != 99.0 and res.to_numpy()[0, 0] != 99.0
    record = dict(res.linear_history[0])
    record["step"] = -1.0
    assert res.linear_history[0]["step"] != -1.0
    # results (and errors carrying them) survive pickling, deep copies and JSON export of the history
    clone = pickle.loads(pickle.dumps(res))
    pd.testing.assert_frame_equal(clone.path, res.path)
    assert clone.linear_history == res.linear_history
    assert copy.deepcopy(res).linear_history[0] == dict(res.linear_history[0])
    assert json.loads(json.dumps(list(res.linear_history)))[0]["newton_iteration"] == 1


# ---------------------------------------------------------------------------
# Horizon comparison
# ---------------------------------------------------------------------------
def test_compare_horizons_rules_and_statistic(ramsey):
    eq, y_ss = ramsey["equations_fn"], ramsey["y_ss"]
    y_init = np.array([ramsey["c_ss"], 0.5 * ramsey["k_ss"]])
    names = ramsey["names"]
    short = solve_stacked_newton_krylov(StackedProblem(eq, y_init, y_ss, np.ones(30), variable_names=names), tol=1e-10)
    long = solve_stacked_newton_krylov(StackedProblem(eq, y_init, y_ss, np.ones(60), variable_names=names), tol=1e-10)
    same = compare_horizons(short.to_numpy(), short.to_numpy()[np.r_[np.arange(30), 29]], periods=20, tolerance=1e-5)
    assert same.passed and same.max_abs_difference == 0.0 and same.short_horizon == 30 and same.long_horizon == 31
    bumped = short.to_numpy()[np.r_[np.arange(30), 29]].copy()
    bumped[2, 1] += 2e-5
    cmp = compare_horizons(short, bumped, periods=20, tolerance=1e-5)
    assert not cmp.passed and cmp.worst_period == 3 and cmp.worst_variable == "k"
    assert cmp.max_abs_difference == pytest.approx(2e-5, abs=1e-12)
    assert cmp.variable_differences[0] == 0.0
    assert isinstance(cmp, HorizonComparison) and "FAILED" in cmp.summary()
    frame = cmp.to_dataframe()
    assert list(frame.index) == names and bool(frame.loc["k", "passed"]) is False
    assert "max_abs_difference" in cmp.to_markdown() and "tabular" in cmp.to_latex() and "#table" in cmp.to_typst()
    # the genuine horizon-doubling statistic (30 vs 60 dates fails at 1e-5, 60 vs 120 passes)
    longer = solve_stacked_newton_krylov(StackedProblem(eq, y_init, y_ss, np.ones(120), variable_names=names), tol=1e-10)
    first = compare_horizons(short, long, periods=20, tolerance=1e-5)
    second = compare_horizons(long, longer, periods=20, tolerance=1e-5)
    assert not first.passed and second.passed and second.max_abs_difference < first.max_abs_difference
    assert compare_stacked_horizons is compare_horizons
    with pytest.raises(ValueError, match="longer horizon"):
        compare_horizons(long, short)
    with pytest.raises(ValueError, match="periods"):
        compare_horizons(short, long, periods=31)
    with pytest.raises(ValueError, match="periods"):
        compare_horizons(short, long, periods=0)
    with pytest.raises(ValueError, match="tolerance"):
        compare_horizons(short, long, tolerance=0.0)
    with pytest.raises(ValueError, match="state dimensions"):
        compare_horizons(short.to_numpy(), long.to_numpy()[:, :1])
    with pytest.raises(ValueError, match="boundaries"):
        other = solve_stacked_newton_krylov(StackedProblem(eq, y_ss, y_ss, np.ones(60), variable_names=names), tol=1e-10)
        compare_horizons(short, other)
    with pytest.raises(ValueError, match="exogenous path"):
        shocked = solve_stacked_newton_krylov(
            StackedProblem(eq, y_init, y_ss, np.r_[np.ones(59), 1.01], variable_names=names), tol=1e-10)
        compare_horizons(short, shocked)
    failed = StackedNewtonResult(path=short.path, converged=False, iterations=0, residual_norm=1.0,
                                 initial_residual_norm=1.0, terminal_error=0.0)
    with pytest.raises(StackedNewtonError, match="unaccepted"):
        compare_horizons(failed, long)
    # DataFrames, PerfectForesightResult objects and arrays are accepted
    pf = solve_perfect_foresight(eq, y_init=y_init, y_ss=y_ss, exogenous_path=np.ones(60), n_periods=60,
                                 tol=1e-10, variable_names=names)
    via_pf = compare_horizons(short.path, pf, periods=20, tolerance=1e-5)
    assert via_pf.max_abs_difference == pytest.approx(first.max_abs_difference, abs=1e-9)


# ---------------------------------------------------------------------------
# Slow: preconditioner ranking regression
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_preconditioner_ranking_regression(growth20, growth20_reference):
    """Time coupling belongs in the preconditioner: exact steady block-tridiagonal inverse
    needs <= 4 Krylov outer iterations per Newton step, time-block Jacobi needs more, and the
    unpreconditioned solve exhausts krylov_maxiter=40 at every step (assessment table)."""
    g = growth20
    prob = StackedProblem(g["eq"], g["y_init"], g["y_ss"], g["exo"])
    steady = solve_stacked_newton_krylov(prob, tol=1e-9, krylov_maxiter=40)
    jacobi = solve_stacked_newton_krylov(prob, preconditioner="time_block_jacobi", tol=1e-9, krylov_maxiter=40)
    assert steady.converged and jacobi.converged
    assert max(steady.krylov_outer_iterations()) <= 4
    assert sum(jacobi.krylov_outer_iterations()) > sum(steady.krylov_outer_iterations())
    with pytest.raises(StackedNewtonError) as excinfo:
        solve_stacked_newton_krylov(prob, preconditioner=None, tol=1e-9, krylov_maxiter=40, max_iter=3)
    history = excinfo.value.result.linear_history
    assert all(h["krylov_outer_iterations"] == 40 and h["krylov_status"] == 40 for h in history)
    assert excinfo.value.result.residual_norm > 1e-3


# ---------------------------------------------------------------------------
# Parity with the IO research engine (skipped when the volume is absent)
# ---------------------------------------------------------------------------
def _analytic_native_data(native_data):
    """Copy of dynamic_model/tests/test_native_dynamics.py::analytic_native_data (no IO test import)."""
    from scipy.sparse import csr_matrix

    z = np.array([[7, 3, 1, 2], [2, 9, 2, 1], [1, 2, 8, 3], [2, 1, 3, 7]], dtype=float)
    consumption = np.array([[22, 3], [20, 4], [3, 22], [4, 20]], dtype=float)
    investment = np.array([[12, 2], [10, 3], [3, 11], [2, 10]], dtype=float)
    tourism = np.array([[0, .5], [0, .3], [.4, 0], [.2, 0]])
    inventory = np.array([[.4, 0], [.2, 0], [0, .3], [0, .2]])
    valuables = np.array([[.1, 0], [0, .1], [0, .1], [.1, 0]])
    output = z.sum(axis=1) + (consumption + investment + tourism + inventory + valuables).sum(axis=1)
    product_tax = np.array([.3, .5, .4, .2])
    production_tax = np.array([.4, .3, .2, .5])
    va = output - z.sum(axis=0) - product_tax
    factor_va = va - production_tax
    alpha = np.array([.35, .48, .30, .52])
    return native_data.NativeData(
        dataset="analytic_test_fixture", countries=("AAA", "BBB"), sectors=("S1", "S2"),
        Z=csr_matrix(z), final_consumption=consumption, final_investment=investment,
        purchases_abroad=tourism, inventory_changes=inventory, valuables=valuables,
        VA=va, TLS=product_tax, production_taxes=production_tax,
        labor_compensation=(1 - alpha) * factor_va, operating_surplus=alpha * factor_va,
        output=output, TFD=np.array([[1.3, .8, .1, .02, .03], [1.1, .7, .08, .01, .02]]),
        merchandise_mask=np.array([True, False]),
        metadata={"source_kind": "analytic software fixture; not an empirical database"})


def _io_model(phi=2.0):
    native_data = _io_module(".", "dynamic_model.native_data")
    native_economy = _io_module(".", "dynamic_model.native_economy")
    calibration = native_economy.calibrate_native(_analytic_native_data(native_data))
    return native_economy, native_economy.NativeEconomy(calibration, adjustment_cost=phi)


def _io_problem(model, policies, *, jvp=True):
    """StackedProblem over NativeEconomy.equations(..., policy_t, policy_next).

    Each exogenous row packs (id of policy_t, id of policy_{t+1}), the last row
    repeating the terminal policy (the IO PathProblem convention). An id is the
    first 48 bits of the policy fingerprint (exact in float64), so the same policy
    gets the same id in every problem: two horizons of one announced schedule have
    equal rows on the shared dates and the held terminal row afterwards, which is
    what compare_horizons' held-exogenous rule checks.
    """
    T = len(policies)
    table = {int(policy.fingerprint[:12], 16): policy for policy in policies}
    ids = [int(policy.fingerprint[:12], 16) for policy in policies]
    exo = np.array([[ids[t], ids[min(t + 1, T - 1)]] for t in range(T)], dtype=float)

    def eq(yp, yc, yl, eps):
        return model.equations(yp, yc, yl, table[int(eps[0])], table[int(eps[1])])

    def jvp_fn(yp, yc, yl, eps, vp, vc, vl):
        return model.equations_jvp(yp, yc, yl, vp, vc, vl, table[int(eps[0])], table[int(eps[1])])

    return StackedProblem(eq, model.baseline, model.baseline, exo, jvp_fn=jvp_fn if jvp else None)


@pytest.mark.parametrize("phi", [0.0, 2.0])
@pytest.mark.parametrize("horizon", [1, 2, 5, 10, 13])
def test_io_parity_stacked_problem_and_exact_block_inverse(phi, horizon):
    """StackedProblem reproduces PathProblem (residual and analytic JVP identical) and the
    steady block-tridiagonal inverse reproduces FiniteBlockCapitalPreconditioner(block_size=T)
    to 5e-11 (IO tolerance; observed <= 2e-13); T=13 also checks block_size=5 block Jacobi."""
    native_numerics = _io_module(".", "dynamic_model.native_numerics")
    native_block = _io_module(".", "dynamic_model.native_block_preconditioner")
    _, model = _io_model(phi)
    policy = model.zero_policy()
    prob = _io_problem(model, [policy] * horizon)
    path_problem = native_numerics.PathProblem(model, [policy] * horizon, model.baseline, model.baseline)
    rng = np.random.default_rng(horizon + 29)
    Y = 0.002 * rng.normal(size=horizon * model.n_vars)
    v = rng.normal(size=horizon * model.n_vars)
    np.testing.assert_allclose(prob.residual(Y), path_problem.evaluate(Y), rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(prob.jvp(Y)(v), path_problem.linearize(Y)(v), rtol=0.0, atol=1e-12)
    central = _io_problem(model, [policy] * horizon, jvp=False)
    np.testing.assert_allclose(central.jvp(Y)(v), path_problem.linearize(Y)(v), rtol=0.0, atol=1e-6)
    A, B, C = prob.steady_blocks()
    rhs = path_problem.linearize(np.zeros(horizon * model.n_vars))(v)
    ours = BlockTridiagonalPreconditioner(A, B, C, horizon)
    theirs = native_block.FiniteBlockCapitalPreconditioner(model, block_size=horizon).operator(horizon)
    np.testing.assert_allclose(ours.apply(rhs), theirs @ rhs, rtol=5e-11, atol=5e-11)
    np.testing.assert_allclose(ours.apply(rhs), v, rtol=5e-11, atol=5e-11)
    if horizon == 13:
        ours5 = BlockTridiagonalPreconditioner(A, B, C, horizon, time_block=5)
        theirs5 = native_block.FiniteBlockCapitalPreconditioner(model, block_size=5).operator(horizon)
        np.testing.assert_allclose(ours5.apply(rhs), theirs5 @ rhs, rtol=5e-11, atol=5e-11)


def test_io_parity_newton_harness_and_transition():
    """With side='left' and jacobian_scale=1 the harness reproduces IO native_solver._newton to
    1e-13 (observed 0.0 on the development machine) on a 40-date announced-tariff transition (same
    residual, JVP and preconditioner; |z|_inf = 0.0065 there, so the path-scaled step bound equals
    the IO absolute bound), and the path agrees with IO solve_transition (capital preconditioner)
    to 1e-9 (observed 3e-14)."""
    native_solver = _io_module(".", "dynamic_model.native_solver")
    native_economy, model = _io_model(2.0)
    calibration = model.p
    rates = np.zeros((calibration.n_cells, calibration.n_countries))
    rates[calibration.country == 0, 1] = .06
    rates[calibration.country == 1, 0] = .08
    tariff = native_economy.NativePolicy(rates, "analytic tariff", consumption_rates=rates * .7,
                                         investment_rates=rates * 1.3)
    zero = model.zero_policy()
    T = 40
    policies = [tariff if 2 <= t < 7 else zero for t in range(T)]
    prob = _io_problem(model, policies)
    reference = native_solver.solve_transition(model, policies, initial=model.baseline,
                                               terminal=model.baseline, tol=1e-10)
    ours = solve_stacked_newton_krylov(prob, np.zeros((T, model.n_vars)), tol=1e-10, jacobian_scale=1.0, side="left")
    assert ours.converged
    np.testing.assert_allclose(ours.to_numpy(), reference.z, rtol=0.0, atol=1e-9)
    A, B, C = prob.steady_blocks()
    operator = BlockTridiagonalPreconditioner(A, B, C, T).as_linear_operator()
    z, error, iterations, history = native_solver._newton(
        prob.residual, np.zeros(T * model.n_vars), tol=1e-10, max_iter=40,
        jvp_factory=lambda x, f: prob.jvp(x), preconditioner_factory=lambda x: operator)
    np.testing.assert_allclose(ours.to_numpy().ravel(), z, rtol=0.0, atol=1e-13)
    assert ours.iterations == iterations
    io_outer = [h["krylov_outer_iterations"] for h in history if "krylov_outer_iterations" in h]
    assert list(ours.krylov_outer_iterations()[:len(io_outer)]) == io_outer
    assert error <= 1e-10 and ours.residual_norm <= 1e-10
    # step_tol=None is the IO residual-only acceptance: no final convergence-check solve,
    # so the whole Krylov history matches, not only its shared prefix
    io_rule = solve_stacked_newton_krylov(prob, np.zeros((T, model.n_vars)), tol=1e-10, jacobian_scale=1.0,
                                          side="left", step_tol=None)
    np.testing.assert_allclose(io_rule.to_numpy().ravel(), z, rtol=0.0, atol=1e-13)
    assert list(io_rule.krylov_outer_iterations()) == io_outer and io_rule.iterations == iterations
    assert io_rule.metadata["converged_by"] == "residual"


def _noisy_floor_problem(T):
    """Linear transition 1e-3 (y_t - .5 y_{t-1} - .4 y_{t+1} - .1) plus deterministic 1e-12 noise.

    The noise stands in for an inner iterative solve: the residual floors near 1e-12,
    far below tol = 1e-8, while Newton steps on the noise stay of order 1e-9 > 1e-10.
    """
    def noise(y):
        bits = np.frombuffer(np.ascontiguousarray(y, dtype=float).tobytes(), dtype=np.uint64)
        return ((bits * np.uint64(2654435761)) % np.uint64(1000003)).astype(float) / 500001.5 - 1.0

    def eq(yp, yc, yl, eps):
        return 1e-3 * (yc - 0.5 * yl - 0.4 * yp - 0.1) + 1e-12 * noise(yc)

    def jvp(yp, yc, yl, eps, vp, vc, vl):
        return 1e-3 * (vc - 0.5 * vl - 0.4 * vp)

    return StackedProblem(eq, np.array([0.5]), np.array([1.0]), n_periods=T, jvp_fn=jvp)


def test_io_parity_residual_only_acceptance_on_a_noise_floor():
    """On a residual with a 1e-12 noise floor (300 unknowns, above the IO MINPACK threshold),
    IO native_solver._newton accepts; step_tol=None with jacobian_scale=1 and side='left'
    reproduces its path to 1e-13 (observed 0.0) with the same Krylov counts, while the default
    step test raises step_test_failure at a residual that already meets the tolerance."""
    native_solver = _io_module(".", "dynamic_model.native_solver")
    T = 300
    prob = _noisy_floor_problem(T)
    A, B, C = prob.steady_blocks()
    operator = BlockTridiagonalPreconditioner(A, B, C, T).as_linear_operator()
    z, error, iterations, history = native_solver._newton(
        prob.residual, prob.linear_initial_path().ravel(), tol=1e-8, max_iter=30,
        jvp_factory=lambda x, f: prob.jvp(x), preconditioner_factory=lambda x: operator)
    ours = solve_stacked_newton_krylov(prob, tol=1e-8, max_iter=30, jacobian_scale=1.0, side="left", step_tol=None)
    np.testing.assert_allclose(ours.to_numpy().ravel(), z, rtol=0.0, atol=1e-13)
    assert ours.iterations == iterations and error <= 1e-8
    assert list(ours.krylov_outer_iterations()) == [h["krylov_outer_iterations"] for h in history]
    with pytest.raises(StackedNewtonError, match="noise floor") as excinfo:
        solve_stacked_newton_krylov(prob, tol=1e-8, max_iter=30, jacobian_scale=1.0, side="left")
    assert excinfo.value.result.metadata["converged_by"] == "step_test_failure"
    assert excinfo.value.result.residual_norm <= 1e-8


def test_io_parity_compare_horizons():
    """Announced 6%/8% bilateral tariff on dates 2..6: compare_horizons on the port's own
    StackedNewtonResult objects (boundary and held-exogenous rules active) returns the IO
    native_solver.compare_horizons statistic on IO solve_transition solutions to 1e-12
    (observed about 2e-15) with the same verdict, for 20 versus 40 and 40 versus 80 dates."""
    native_solver = _io_module(".", "dynamic_model.native_solver")
    native_economy, model = _io_model(2.0)
    calibration = model.p
    rates = np.zeros((calibration.n_cells, calibration.n_countries))
    rates[calibration.country == 0, 1] = .06
    rates[calibration.country == 1, 0] = .08
    tariff = native_economy.NativePolicy(rates, "analytic tariff", consumption_rates=rates * .7,
                                         investment_rates=rates * 1.3)
    zero = model.zero_policy()
    ours, theirs = {}, {}
    for T in (20, 40, 80):
        policies = [tariff if 2 <= t < 7 else zero for t in range(T)]
        ours[T] = solve_stacked_newton_krylov(_io_problem(model, policies), np.zeros((T, model.n_vars)), tol=1e-10)
        theirs[T] = native_solver.solve_transition(model, policies, initial=model.baseline,
                                                   terminal=model.baseline, tol=1e-10)
    statistics = []
    for short, long in ((20, 40), (40, 80)):
        port = compare_horizons(ours[short], ours[long], periods=20, tolerance=1e-5)
        io = native_solver.compare_horizons(theirs[short], theirs[long], periods=20, tolerance=1e-5)
        assert port.max_abs_difference == pytest.approx(io["max_log_difference"], rel=0.0, abs=1e-12)
        assert port.passed == io["passed"]
        assert (port.short_horizon, port.long_horizon) == (io["short_horizon"], io["long_horizon"])
        statistics.append(port.max_abs_difference)
    # the comparison is nontrivial: the statistic falls from about 1e-3 to about 4e-5 and both fail 1e-5
    assert 1e-4 < statistics[0] < 1e-2 and 1e-5 < statistics[1] < 1e-4
    # the held-exogenous rule is active on these results: a schedule that differs beyond the
    # short horizon is rejected
    changed = [tariff if 2 <= t < 7 or t == 30 else zero for t in range(40)]
    other = solve_stacked_newton_krylov(_io_problem(model, changed), np.zeros((40, model.n_vars)), tol=1e-10)
    with pytest.raises(ValueError, match="exogenous path"):
        compare_horizons(ours[20], other, periods=20)
