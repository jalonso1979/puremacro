r"""Generalized Real Schur / Bartels-Stewart Sylvester solver for DSGE perturbation models.

Solves the higher-order state curvature generalized Sylvester equations:
    Order 1: A_hat X + A_+ X h_x = -K_x_ss
    Order 2: A_hat X + A_+ X (h_x \otimes h_x) = -K_xx
    Order 3: A_hat X + A_+ X (h_x \otimes h_x \otimes h_x) = -K_xxx

using the Real Schur decomposition of the state transition matrix h_x:
    h_x = U_h T_h U_h^T

where U_h is orthogonal and T_h is quasi-upper-triangular with 1x1 and 2x2 diagonal blocks.
Because h_x^{\otimes k} = U_h^{\otimes k} T_h^{\otimes k} (U_h^{\otimes k})^T, the transformed
system is block upper-triangular with small diagonal blocks of size m \in {1, 2, 4} (order 2)
or m \in {1, 2, 4, 8} (order 3) and is solved recursively level-by-level in real arithmetic
via Bartels-Stewart block back-substitution.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class SylvesterWorkCounts:
    """Work and conditioning statistics for a Real Schur Sylvester solve."""

    order: int
    n_vars: int
    n_states: int
    scalar_solves: int
    quasi_block_solves: int
    lu_factorizations: int
    lu_cache_hits: int
    lu_cache_misses: int
    lu_cache_evictions: int
    max_cache: int
    max_block_size: int
    back_substitutions: int
    max_block_condition: float
    min_block_rcond: float


@dataclass
class _SylvesterWorkTracker:
    order: int
    n_vars: int
    n_states: int
    max_cache: int
    scalar_solves: int = 0
    quasi_block_solves: int = 0
    lu_factorizations: int = 0
    lu_cache_hits: int = 0
    lu_cache_misses: int = 0
    lu_cache_evictions: int = 0
    max_block_size: int = 0
    back_substitutions: int = 0
    max_block_condition: float = 1.0
    min_block_rcond: float = 1.0

    def record_condition(self, cond: float) -> None:
        if not np.isfinite(cond) or cond <= 0.0:
            self.max_block_condition = float("inf")
            self.min_block_rcond = 0.0
            return
        if cond > self.max_block_condition:
            self.max_block_condition = float(cond)
        rcond = 1.0 / float(cond)
        if rcond < self.min_block_rcond:
            self.min_block_rcond = rcond

    def freeze(self) -> SylvesterWorkCounts:
        return SylvesterWorkCounts(
            order=self.order,
            n_vars=self.n_vars,
            n_states=self.n_states,
            scalar_solves=self.scalar_solves,
            quasi_block_solves=self.quasi_block_solves,
            lu_factorizations=self.lu_factorizations,
            lu_cache_hits=self.lu_cache_hits,
            lu_cache_misses=self.lu_cache_misses,
            lu_cache_evictions=self.lu_cache_evictions,
            max_cache=self.max_cache,
            max_block_size=self.max_block_size,
            back_substitutions=self.back_substitutions,
            max_block_condition=self.max_block_condition,
            min_block_rcond=self.min_block_rcond,
        )


_LAST_WORK_COUNTS: dict[int, SylvesterWorkCounts] = {}


def get_last_sylvester_work_counts(order: int | None = None) -> SylvesterWorkCounts | dict[int, SylvesterWorkCounts] | None:
    """Return the most recent Sylvester work counts (by order or all orders)."""
    if order is None:
        return dict(_LAST_WORK_COUNTS)
    return _LAST_WORK_COUNTS.get(order)


def _lu_pivot_rcond(lu: np.ndarray) -> float:
    """Fast O(n) reciprocal condition proxy from the diagonal pivots of U in LU."""
    if lu.size == 0:
        return 1.0
    diag = np.abs(np.diag(lu))
    d_max = float(np.max(diag))
    if d_max == 0.0 or not np.isfinite(d_max):
        return 0.0
    d_min = float(np.min(diag))
    return float(d_min / d_max)


def _solve_consistent_linear(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    context: str,
    rtol: float = 1e-7,
    atol: float = 1e-10,
    tracker: _SylvesterWorkTracker | None = None,
) -> np.ndarray:
    """Solve lhs @ x = rhs, failing explicitly if lhs is singular and the system is inconsistent."""
    lhs_arr = np.asarray(lhs, dtype=float)
    rhs_arr = np.asarray(rhs, dtype=float)
    if not np.isfinite(lhs_arr).all() or not np.isfinite(rhs_arr).all():
        raise np.linalg.LinAlgError(f"Non-finite entries in linear system ({context})")

    n = lhs_arr.shape[0]
    if n == 0:
        return np.zeros_like(rhs_arr, dtype=float)

    dgetrf = scipy.linalg.lapack.dgetrf
    dgetrs = scipy.linalg.lapack.dgetrs
    lu, piv, info = dgetrf(lhs_arr)
    rcond_est = _lu_pivot_rcond(lu) if info == 0 else 0.0
    if tracker is not None:
        cond_est = (1.0 / rcond_est) if rcond_est > 0.0 else float("inf")
        tracker.record_condition(cond_est)

    if np.all(rhs_arr == 0.0):
        return np.zeros_like(rhs_arr, dtype=float)

    singular_rcond = max(n, 1) * np.finfo(float).eps
    if info == 0 and rcond_est > singular_rcond:
        sol, solve_info = dgetrs(lu, piv, rhs_arr)
        if solve_info == 0 and np.isfinite(sol).all():
            if rcond_est < 1e-8:
                lhs_norm = float(np.linalg.norm(lhs_arr, ord=np.inf))
                rhs_norm = float(np.linalg.norm(rhs_arr, ord=np.inf))
                res_norm = float(np.linalg.norm(lhs_arr @ sol - rhs_arr, ord=np.inf))
                sol_norm = float(np.linalg.norm(sol, ord=np.inf))
                scale = max(rhs_norm, lhs_norm * sol_norm, 1.0)
                if res_norm <= atol + rtol * scale:
                    return np.asarray(sol, dtype=float)
            else:
                return np.asarray(sol, dtype=float)

    lhs_norm = float(np.linalg.norm(lhs_arr, ord=np.inf))
    rhs_norm = float(np.linalg.norm(rhs_arr, ord=np.inf))
    sol, _, rank, svals = np.linalg.lstsq(lhs_arr, rhs_arr, rcond=None)
    s_max = float(svals[0]) if svals.size > 0 else 0.0
    s_min = float(svals[-1]) if svals.size > 0 else 0.0
    cond = (s_max / s_min) if s_min > 0.0 else float("inf")
    if tracker is not None:
        tracker.record_condition(cond)

    if not np.isfinite(sol).all():
        raise np.linalg.LinAlgError(
            f"Singular linear system produced non-finite least-squares solution in {context} "
            f"(rank={int(rank)}/{n}, cond={cond:.3e})"
        )
    res_norm = float(np.linalg.norm(lhs_arr @ sol - rhs_arr, ord=np.inf))
    sol_norm = float(np.linalg.norm(sol, ord=np.inf))
    scale = max(rhs_norm, lhs_norm * sol_norm, 1.0)
    if res_norm > atol + rtol * scale:
        raise np.linalg.LinAlgError(
            f"Singular inconsistent linear system in {context}: "
            f"rank={int(rank)}/{n}, cond={cond:.3e}, "
            f"residual_inf={res_norm:.3e}, rhs_inf={rhs_norm:.3e}"
        )
    return np.asarray(sol, dtype=float)


def _schur_blocks(T: np.ndarray, tol: float = 1e-14) -> list[tuple[int, int]]:
    """Partition quasi-upper-triangular Real Schur matrix T into 1x1 and 2x2 diagonal blocks."""
    n = T.shape[0]
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if i + 1 < n and abs(float(T[i + 1, i])) > tol:
            blocks.append((i, i + 2))
            i += 2
        else:
            blocks.append((i, i + 1))
            i += 1
    return blocks


def _canonical_scalar_tau(*factors: float) -> float:
    """Compute product of scalar Schur factors in canonical sorted order for exact float cache reuse."""
    if not factors:
        return 1.0
    if len(factors) == 1:
        val = float(factors[0])
        return 0.0 if val == 0.0 else val
    if len(factors) == 2:
        val = float(factors[0]) * float(factors[1])
        return 0.0 if val == 0.0 else val
    ordered = sorted(float(f) for f in factors)
    prod = ordered[0]
    for f in ordered[1:]:
        prod = float(prod * f)
    return 0.0 if prod == 0.0 else prod


def _solve_scalar_block(
    A_hat: np.ndarray,
    A_plus: np.ndarray,
    tau: float,
    rhs: np.ndarray,
    lu_cache: OrderedDict[float, tuple[Any, Any, int, float]],
    max_cache: int,
    dgetrf: Any,
    dgetrs: Any,
    dgecon: Any = None,
    tracker: _SylvesterWorkTracker | None = None,
) -> np.ndarray:
    """Solve (A_hat + tau * A_plus) x = rhs in real arithmetic with exact-key bounded LU caching."""
    key = 0.0 if tau == 0.0 else float(tau)
    if tracker is not None:
        tracker.scalar_solves += 1
        if tracker.max_block_size < 1:
            tracker.max_block_size = 1

    use_cache = max_cache > 0
    Mj: np.ndarray | None = None
    if use_cache and key in lu_cache:
        lu, piv, info, rcond_est = lu_cache[key]
        lu_cache.move_to_end(key)
        if tracker is not None:
            tracker.lu_cache_hits += 1
            cond_est = (1.0 / rcond_est) if rcond_est > 0.0 else float("inf")
            tracker.record_condition(cond_est)
    else:
        Mj = A_hat + key * A_plus
        lu, piv, info = dgetrf(Mj)
        rcond_est = _lu_pivot_rcond(lu) if info == 0 else 0.0
        if tracker is not None:
            tracker.lu_cache_misses += 1
            tracker.lu_factorizations += 1
            cond_est = (1.0 / rcond_est) if rcond_est > 0.0 else float("inf")
            tracker.record_condition(cond_est)
        if use_cache:
            if len(lu_cache) >= max_cache:
                lu_cache.popitem(last=False)
                if tracker is not None:
                    tracker.lu_cache_evictions += 1
            lu_cache[key] = (lu, piv, info, rcond_est)

    singular_rcond = max(A_hat.shape[0], 1) * np.finfo(float).eps
    if info == 0 and rcond_est > singular_rcond:
        sol, solve_info = dgetrs(lu, piv, rhs)
        if solve_info == 0 and np.isfinite(sol).all():
            if rcond_est < 1e-8:
                if Mj is None:
                    Mj = A_hat + key * A_plus
                res_norm = float(np.linalg.norm(Mj @ sol - rhs, ord=np.inf))
                rhs_norm = float(np.linalg.norm(rhs, ord=np.inf))
                mj_norm = float(np.linalg.norm(Mj, ord=np.inf))
                sol_norm = float(np.linalg.norm(sol, ord=np.inf))
                scale = max(rhs_norm, mj_norm * sol_norm, 1.0)
                if res_norm <= 1e-10 + 1e-7 * scale:
                    return sol
            else:
                return sol

    if Mj is None:
        Mj = A_hat + key * A_plus
    return _solve_consistent_linear(
        Mj,
        rhs,
        context=f"Sylvester scalar block (tau={key:.16g})",
        tracker=tracker,
    )


def _solve_quasi_block(
    A_hat: np.ndarray,
    A_plus: np.ndarray,
    S: np.ndarray,
    D_block: np.ndarray,
    tracker: _SylvesterWorkTracker | None = None,
) -> np.ndarray:
    """Solve A_hat X_B + A_plus X_B S = D_block for a small quasi-triangular block S of size m <= 8."""
    N, m = D_block.shape
    if tracker is not None:
        if m == 1:
            tracker.scalar_solves += 1
        else:
            tracker.quasi_block_solves += 1
        if m > tracker.max_block_size:
            tracker.max_block_size = m

    if m == 1:
        tau = 0.0 if float(S[0, 0]) == 0.0 else float(S[0, 0])
        Mj = A_hat + tau * A_plus
        sol_col = _solve_consistent_linear(
            Mj,
            D_block[:, 0],
            context=f"Sylvester 1x1 quasi-block (tau={tau:.16g})",
            tracker=tracker,
        )
        return sol_col[:, None]

    # Form small (N*m, N*m) real block system in column-major order:
    # (I_m \otimes A_hat + S^T \otimes A_plus) vec_F(X_B) = vec_F(D_block)
    lhs = np.kron(np.eye(m), A_hat) + np.kron(S.T, A_plus)
    rhs = D_block.reshape(N * m, order="F")
    sol_vec = _solve_consistent_linear(
        lhs,
        rhs,
        context=f"Sylvester quasi-block (m={m})",
        tracker=tracker,
    )
    return sol_vec.reshape((N, m), order="F")


def solve_dense_sylvester_fallback(
    A_hat: np.ndarray,
    A_plus: np.ndarray,
    h_x: np.ndarray,
    K_rhs: np.ndarray,
    *,
    order: int,
    max_dense_dim: int = 4096,
    max_dense_bytes: int = 128 * 1024 * 1024,
    cause: Exception | None = None,
) -> np.ndarray:
    """Solve dense Kronecker Sylvester system with explicit allocation limits and consistency checks."""
    A_hat_arr = np.asarray(A_hat, dtype=float)
    A_plus_arr = np.asarray(A_plus, dtype=float)
    h_x_arr = np.asarray(h_x, dtype=float)
    K_arr = np.asarray(K_rhs, dtype=float)

    N = int(A_hat_arr.shape[0])
    n_x = int(h_x_arr.shape[0])
    m = int(n_x**order)
    dim = N * m
    est_bytes = dim * dim * 8

    if dim > max_dense_dim or est_bytes > max_dense_bytes:
        cause_msg = f" after Schur solver failure ({type(cause).__name__}: {cause})" if cause is not None else ""
        raise np.linalg.LinAlgError(
            f"Dense order-{order} Kronecker Sylvester fallback refused{cause_msg}: "
            f"system dimension {dim}x{dim} (N={N}, n_x={n_x}, n_x^{order}={m}) requires "
            f"{est_bytes / (1024 * 1024):.1f} MiB, exceeding allocation limits "
            f"(max_dense_dim={max_dense_dim}, max_dense_bytes={max_dense_bytes / (1024 * 1024):.1f} MiB)."
        )

    if dim == 0:
        return np.zeros((N, 0), dtype=float)

    hx_kron = np.array([[1.0]], dtype=float)
    for _ in range(order):
        hx_kron = np.kron(hx_kron, h_x_arr)
    sys_mat = np.kron(np.eye(m), A_hat_arr) + np.kron(hx_kron.T, A_plus_arr)
    rhs_vec = -K_arr.reshape(-1, order="F")
    cause_ctx = f", primary cause: {cause}" if cause is not None else ""
    vec_sol = _solve_consistent_linear(
        sys_mat,
        rhs_vec,
        context=f"order-{order} dense Kronecker Sylvester fallback (N={N}, n_x={n_x}{cause_ctx})",
    )
    return np.ascontiguousarray(vec_sol.reshape((N, m), order="F"), dtype=float)


def solve_generalized_sylvester_schur(
    A_hat: np.ndarray,
    A_plus: np.ndarray,
    h_x: np.ndarray,
    K_xx: np.ndarray,
    max_cache: int = 32,
    *,
    return_work_counts: bool = False,
) -> np.ndarray | tuple[np.ndarray, SylvesterWorkCounts]:
    r"""Solve A_hat X + A_+ X (h_x \otimes h_x) = -K_xx via Real Schur Bartels-Stewart recursion.

    Parameters
    ----------
    A_hat : np.ndarray
        Coefficient matrix on X of shape (N, N), typically (A_0 + A_+ g_x P_s).
    A_plus : np.ndarray
        Lead coefficient matrix of shape (N, N).
    h_x : np.ndarray
        First-order state transition matrix of shape (n_x, n_x).
    K_xx : np.ndarray
        Second-order right-hand side curvature matrix of shape (N, n_x^2).
    max_cache : int, default 128
        Maximum number of real LU factorizations to cache.
    return_work_counts : bool, default False
        When True, return ``(g_xx, work_counts)``.

    Returns
    -------
    g_xx : np.ndarray
        Second-order policy curvature matrix of shape (N, n_x^2).
    """
    A_hat = np.asarray(A_hat, dtype=float)
    A_plus = np.asarray(A_plus, dtype=float)
    h_x = np.asarray(h_x, dtype=float)
    K_xx = np.asarray(K_xx, dtype=float)

    N = A_hat.shape[0]
    n_x = h_x.shape[0]
    M = n_x * n_x
    tracker = _SylvesterWorkTracker(order=2, n_vars=N, n_states=n_x, max_cache=max_cache)

    if n_x == 0 or M == 0:
        out = np.zeros((N, 0), dtype=float)
        counts = tracker.freeze()
        _LAST_WORK_COUNTS[2] = counts
        return (out, counts) if return_work_counts else out

    if np.all(K_xx == 0.0):
        out = np.zeros((N, M), dtype=float)
        counts = tracker.freeze()
        _LAST_WORK_COUNTS[2] = counts
        return (out, counts) if return_work_counts else out

    if np.all(h_x == 0.0):
        rhs = -K_xx
        sol = _solve_consistent_linear(
            A_hat,
            rhs,
            context="Order-2 Sylvester with zero transition matrix (h_x = 0)",
            tracker=tracker,
        )
        tracker.scalar_solves += M
        tracker.lu_factorizations += 1
        out = np.ascontiguousarray(np.real(sol), dtype=float)
        counts = tracker.freeze()
        _LAST_WORK_COUNTS[2] = counts
        return (out, counts) if return_work_counts else out

    # 1. Real Schur decomposition: h_x = U_h @ T_h @ U_h^T
    T_h, U_h = scipy.linalg.schur(h_x, output="real")
    blocks = _schur_blocks(T_h)

    # 2. Transform RHS via mode contractions without forming U_h \otimes U_h:
    # \tilde{D} = -K_xx @ (U_h \otimes U_h)
    K_tensor = (-K_xx).reshape(N, n_x, n_x)
    temp = np.tensordot(K_tensor, U_h, axes=(2, 0))
    D_tilde = np.tensordot(temp, U_h, axes=(1, 0)).transpose(0, 2, 1)
    D_tilde = np.ascontiguousarray(D_tilde, dtype=float)
    del temp

    # 3. Block-triangular Bartels-Stewart sweep
    X_tilde = np.zeros((N, n_x, n_x), dtype=float)
    W = np.zeros((N, n_x, n_x), dtype=float)  # caches A_plus @ X_tilde
    dgetrf = scipy.linalg.lapack.dgetrf
    dgetrs = scipy.linalg.lapack.dgetrs
    dgecon = getattr(scipy.linalg.lapack, "dgecon", None)
    lu_cache: OrderedDict[float, tuple[Any, Any, int, float]] = OrderedDict()

    for i1, e1 in blocks:
        m1 = e1 - i1
        T1 = T_h[i1:e1, i1:e1]

        # Subtract Region 1 couplings (k1 < i1, all k2) in one vectorized contraction
        if i1 > 0:
            T1_off = T_h[:i1, i1:e1]
            if np.any(T1_off != 0.0):
                tmp1 = np.tensordot(W[:, :i1, :], T1_off, axes=(1, 0))  # (N, n_x, m1)
                D_tilde[:, i1:e1, :] -= np.tensordot(tmp1, T_h, axes=(1, 0))  # (N, m1, n_x)
                tracker.back_substitutions += 1

        for i2, e2 in blocks:
            m2 = e2 - i2
            T2 = T_h[i2:e2, i2:e2]

            # Subtract Region 2 couplings (k1 in [i1, e1), k2 < i2)
            if i2 > 0:
                T2_off = T_h[:i2, i2:e2]
                if np.any(T2_off != 0.0):
                    if m1 == 1 and m2 == 1:
                        D_tilde[:, i1, i2] -= float(T1[0, 0]) * (W[:, i1, :i2] @ T2_off[:, 0])
                    else:
                        tmp2 = np.tensordot(W[:, i1:e1, :i2], T1, axes=(1, 0))  # (N, i2, m1)
                        D_tilde[:, i1:e1, i2:e2] -= np.tensordot(tmp2, T2_off, axes=(1, 0))  # (N, m1, m2)
                    tracker.back_substitutions += 1

            # Solve diagonal block (b1, b2) of size m = m1 * m2 in {1, 2, 4}
            if m1 == 1 and m2 == 1:
                tau = _canonical_scalar_tau(float(T1[0, 0]), float(T2[0, 0]))
                sol = _solve_scalar_block(
                    A_hat, A_plus, tau, D_tilde[:, i1, i2],
                    lu_cache, max_cache, dgetrf, dgetrs, dgecon, tracker,
                )
                X_tilde[:, i1, i2] = sol
                W[:, i1, i2] = A_plus @ sol
            else:
                S = np.kron(T1, T2)
                D_blk = D_tilde[:, i1:e1, i2:e2].reshape(N, m1 * m2)
                sol_blk = _solve_quasi_block(A_hat, A_plus, S, D_blk, tracker)
                X_tilde[:, i1:e1, i2:e2] = sol_blk.reshape(N, m1, m2)
                W[:, i1:e1, i2:e2] = (A_plus @ sol_blk).reshape(N, m1, m2)

    lu_cache.clear()
    del lu_cache, D_tilde, W

    # 4. Back-transform: X = \tilde{X} @ (U_h \otimes U_h)^T
    U_h_T = U_h.T
    temp_back = np.tensordot(X_tilde, U_h_T, axes=(2, 0))
    g_xx = np.tensordot(temp_back, U_h_T, axes=(1, 0)).transpose(0, 2, 1).reshape(N, M)
    out = np.ascontiguousarray(g_xx, dtype=float)
    counts = tracker.freeze()
    _LAST_WORK_COUNTS[2] = counts
    return (out, counts) if return_work_counts else out


def solve_order3_sylvester_schur(
    A_hat: np.ndarray,
    A_plus: np.ndarray,
    h_x: np.ndarray,
    K_xxx: np.ndarray,
    max_cache: int = 64,
    *,
    return_work_counts: bool = False,
) -> np.ndarray | tuple[np.ndarray, SylvesterWorkCounts]:
    r"""Solve A_hat X + A_+ X (h_x \otimes h_x \otimes h_x) = -K_xxx via Real Schur Bartels-Stewart recursion.

    Parameters
    ----------
    A_hat : np.ndarray
        Coefficient matrix on X of shape (N, N), typically (A_0 + A_+ g_x P_s).
    A_plus : np.ndarray
        Lead coefficient matrix of shape (N, N).
    h_x : np.ndarray
        First-order state transition matrix of shape (n_x, n_x).
    K_xxx : np.ndarray
        Third-order right-hand side curvature matrix of shape (N, n_x^3).
    max_cache : int, default 128
        Maximum number of real LU factorizations to cache (< 2 MB memory).
    return_work_counts : bool, default False
        When True, return ``(g_xxx, work_counts)``.

    Returns
    -------
    g_xxx : np.ndarray
        Third-order policy curvature matrix of shape (N, n_x^3).
    """
    A_hat = np.asarray(A_hat, dtype=float)
    A_plus = np.asarray(A_plus, dtype=float)
    h_x = np.asarray(h_x, dtype=float)
    K_xxx = np.asarray(K_xxx, dtype=float)

    N = A_hat.shape[0]
    n_x = h_x.shape[0]
    M = n_x**3
    tracker = _SylvesterWorkTracker(order=3, n_vars=N, n_states=n_x, max_cache=max_cache)

    if n_x == 0 or M == 0:
        out = np.zeros((N, 0), dtype=float)
        counts = tracker.freeze()
        _LAST_WORK_COUNTS[3] = counts
        return (out, counts) if return_work_counts else out

    if np.all(K_xxx == 0.0):
        out = np.zeros((N, M), dtype=float)
        counts = tracker.freeze()
        _LAST_WORK_COUNTS[3] = counts
        return (out, counts) if return_work_counts else out

    if np.all(h_x == 0.0):
        rhs = -K_xxx
        sol = _solve_consistent_linear(
            A_hat,
            rhs,
            context="Order-3 Sylvester with zero transition matrix (h_x = 0)",
            tracker=tracker,
        )
        tracker.scalar_solves += M
        tracker.lu_factorizations += 1
        out = np.ascontiguousarray(np.real(sol), dtype=float)
        counts = tracker.freeze()
        _LAST_WORK_COUNTS[3] = counts
        return (out, counts) if return_work_counts else out

    # 1. Real Schur decomposition: h_x = U_h @ T_h @ U_h^T
    T_h, U_h = scipy.linalg.schur(h_x, output="real")
    blocks = _schur_blocks(T_h)

    # 2. Transform RHS via 3 successive real tensordot contractions without dense Kronecker
    K_tensor = (-K_xxx).reshape(N, n_x, n_x, n_x)
    temp1 = np.tensordot(K_tensor, U_h, axes=(3, 0))
    temp2 = np.tensordot(temp1, U_h, axes=(2, 0))
    D_tilde = np.tensordot(temp2, U_h, axes=(1, 0)).transpose(0, 3, 2, 1)
    D_tilde = np.ascontiguousarray(D_tilde, dtype=float)
    del temp1, temp2, K_tensor

    # 3. 3-level recursive Bartels-Stewart back-substitution in real arithmetic
    X_tilde = np.zeros((N, n_x, n_x, n_x), dtype=float)
    W = np.zeros((N, n_x, n_x, n_x), dtype=float)  # caches A_plus @ X_tilde
    dgetrf = scipy.linalg.lapack.dgetrf
    dgetrs = scipy.linalg.lapack.dgetrs
    dgecon = getattr(scipy.linalg.lapack, "dgecon", None)
    lu_cache: OrderedDict[float, tuple[Any, Any, int, float]] = OrderedDict()

    for i1, e1 in blocks:
        m1 = e1 - i1
        T1 = T_h[i1:e1, i1:e1]

        # Level 1 coupling: k1 < i1 (contracted across all k2, k3 once per b1)
        if i1 > 0:
            T1_off = T_h[:i1, i1:e1]
            if np.any(T1_off != 0.0):
                u1 = np.tensordot(W[:, :i1, :, :], T1_off, axes=(1, 0))     # (N, n_x, n_x, m1)
                u2 = np.tensordot(u1, T_h, axes=(1, 0))                      # (N, n_x, m1, n_x)
                u3 = np.tensordot(u2, T_h, axes=(1, 0))                      # (N, m1, n_x, n_x)
                D_tilde[:, i1:e1, :, :] -= u3
                tracker.back_substitutions += 1

        for i2, e2 in blocks:
            m2 = e2 - i2
            T2 = T_h[i2:e2, i2:e2]

            # Level 2 coupling: k1 in [i1, e1), k2 < i2 (contracted across all k3 once per (b1, b2))
            if i2 > 0:
                T2_off = T_h[:i2, i2:e2]
                if np.any(T2_off != 0.0):
                    v1 = np.tensordot(W[:, i1:e1, :i2, :], T1, axes=(1, 0))  # (N, i2, n_x, m1)
                    v2 = np.tensordot(v1, T2_off, axes=(1, 0))               # (N, n_x, m1, m2)
                    v3 = np.tensordot(v2, T_h, axes=(1, 0))                  # (N, m1, m2, n_x)
                    D_tilde[:, i1:e1, i2:e2, :] -= v3
                    tracker.back_substitutions += 1

            T12 = np.kron(T1, T2) if (m1 > 1 or m2 > 1) else None
            t12_scalar = float(T1[0, 0] * T2[0, 0]) if (m1 == 1 and m2 == 1) else 0.0

            for i3, e3 in blocks:
                m3 = e3 - i3
                T3 = T_h[i3:e3, i3:e3]

                # Level 3 coupling: k1 in [i1, e1), k2 in [i2, e2), k3 < i3
                if i3 > 0:
                    T3_off = T_h[:i3, i3:e3]
                    if np.any(T3_off != 0.0):
                        if m1 == 1 and m2 == 1 and m3 == 1:
                            D_tilde[:, i1, i2, i3] -= t12_scalar * (W[:, i1, i2, :i3] @ T3_off[:, 0])
                        else:
                            w1 = np.tensordot(W[:, i1:e1, i2:e2, :i3], T1, axes=(1, 0))  # (N, m2, i3, m1)
                            w2 = np.tensordot(w1, T2, axes=(1, 0))                        # (N, i3, m1, m2)
                            w3 = np.tensordot(w2, T3_off, axes=(1, 0))                    # (N, m1, m2, m3)
                            D_tilde[:, i1:e1, i2:e2, i3:e3] -= w3
                        tracker.back_substitutions += 1

                # Solve diagonal block (b1, b2, b3) of size m = m1 * m2 * m3 in {1, 2, 4, 8}
                if m1 == 1 and m2 == 1 and m3 == 1:
                    tau = _canonical_scalar_tau(float(T1[0, 0]), float(T2[0, 0]), float(T3[0, 0]))
                    sol = _solve_scalar_block(
                        A_hat, A_plus, tau, D_tilde[:, i1, i2, i3],
                        lu_cache, max_cache, dgetrf, dgetrs, dgecon, tracker,
                    )
                    X_tilde[:, i1, i2, i3] = sol
                    W[:, i1, i2, i3] = A_plus @ sol
                else:
                    S = (t12_scalar * T3) if T12 is None else np.kron(T12, T3)
                    D_blk = D_tilde[:, i1:e1, i2:e2, i3:e3].reshape(N, m1 * m2 * m3)
                    sol_blk = _solve_quasi_block(A_hat, A_plus, S, D_blk, tracker)
                    X_tilde[:, i1:e1, i2:e2, i3:e3] = sol_blk.reshape(N, m1, m2, m3)
                    W[:, i1:e1, i2:e2, i3:e3] = (A_plus @ sol_blk).reshape(N, m1, m2, m3)

    lu_cache.clear()
    del lu_cache, D_tilde, W

    # 4. Back-transform: X = \tilde{X} @ (U_h \otimes U_h \otimes U_h)^T
    U_h_T = U_h.T
    t1_b = np.tensordot(X_tilde, U_h_T, axes=(3, 0))
    t2_b = np.tensordot(t1_b, U_h_T, axes=(2, 0))
    t3_b = np.tensordot(t2_b, U_h_T, axes=(1, 0)).transpose(0, 3, 2, 1)
    g_xxx = t3_b.reshape(N, M)
    out = np.ascontiguousarray(g_xxx, dtype=float)
    counts = tracker.freeze()
    _LAST_WORK_COUNTS[3] = counts
    return (out, counts) if return_work_counts else out


def solve_generalized_sylvester_kronecker(
    A_hat: np.ndarray,
    A_plus: np.ndarray,
    h_x: np.ndarray,
    K_xx: np.ndarray,
    max_cache: int = 32,
    *,
    return_work_counts: bool = False,
) -> np.ndarray | tuple[np.ndarray, SylvesterWorkCounts]:
    r"""Solve the generalized Sylvester equation A_hat X + A_+ X (h_x \otimes h_x) = -K_xx.

    Routes through :func:`solve_generalized_sylvester_schur` (Real Schur /
    Bartels-Stewart block-triangular solver).
    """
    return solve_generalized_sylvester_schur(
        A_hat, A_plus, h_x, K_xx, max_cache=max_cache, return_work_counts=return_work_counts,
    )


def solve_order3_sylvester_kronecker(
    A_hat: np.ndarray,
    A_plus: np.ndarray,
    h_x: np.ndarray,
    K_xxx: np.ndarray,
    max_cache: int = 128,
    *,
    return_work_counts: bool = False,
) -> np.ndarray | tuple[np.ndarray, SylvesterWorkCounts]:
    r"""Solve generalized Sylvester equation A_hat X + A_+ X (h_x \otimes h_x \otimes h_x) = -K_xxx.

    Routes through :func:`solve_order3_sylvester_schur` (Real Schur /
    Bartels-Stewart block-triangular solver).
    """
    return solve_order3_sylvester_schur(
        A_hat, A_plus, h_x, K_xxx, max_cache=max_cache, return_work_counts=return_work_counts,
    )


def solve_order1_sylvester(
    A_hat: np.ndarray,
    A_plus: np.ndarray,
    h_x: np.ndarray,
    K_x_ss: np.ndarray,
    max_cache: int = 64,
    *,
    return_work_counts: bool = False,
) -> np.ndarray | tuple[np.ndarray, SylvesterWorkCounts]:
    r"""Solve generalized Sylvester equation A_hat X + A_+ X h_x = -K_x_ss.

    Solves for cross-state risk correction tensors such as g_x_sigmasigma
    via Real Schur / Bartels-Stewart block back-substitution.
    """
    A_hat = np.asarray(A_hat, dtype=float)
    A_plus = np.asarray(A_plus, dtype=float)
    h_x = np.asarray(h_x, dtype=float)
    K_x_ss = np.asarray(K_x_ss, dtype=float)

    N = A_hat.shape[0]
    n_x = h_x.shape[0]
    tracker = _SylvesterWorkTracker(order=1, n_vars=N, n_states=n_x, max_cache=max_cache)

    if n_x == 0:
        out = np.zeros((N, 0), dtype=float)
        counts = tracker.freeze()
        _LAST_WORK_COUNTS[1] = counts
        return (out, counts) if return_work_counts else out

    if np.all(K_x_ss == 0.0):
        out = np.zeros((N, n_x), dtype=float)
        counts = tracker.freeze()
        _LAST_WORK_COUNTS[1] = counts
        return (out, counts) if return_work_counts else out

    if np.all(h_x == 0.0):
        rhs = -K_x_ss
        sol = _solve_consistent_linear(
            A_hat,
            rhs,
            context="Order-1 Sylvester with zero transition matrix (h_x = 0)",
            tracker=tracker,
        )
        tracker.scalar_solves += n_x
        tracker.lu_factorizations += 1
        out = np.ascontiguousarray(np.real(sol), dtype=float)
        counts = tracker.freeze()
        _LAST_WORK_COUNTS[1] = counts
        return (out, counts) if return_work_counts else out

    T_h, U_h = scipy.linalg.schur(h_x, output="real")
    blocks = _schur_blocks(T_h)

    D_tilde = np.ascontiguousarray((-K_x_ss) @ U_h, dtype=float)
    X_tilde = np.zeros((N, n_x), dtype=float)

    dgetrf = scipy.linalg.lapack.dgetrf
    dgetrs = scipy.linalg.lapack.dgetrs
    dgecon = getattr(scipy.linalg.lapack, "dgecon", None)
    lu_cache: OrderedDict[float, tuple[Any, Any, int, float]] = OrderedDict()

    for i1, e1 in blocks:
        m1 = e1 - i1
        T1 = T_h[i1:e1, i1:e1]
        if m1 == 1:
            tau = _canonical_scalar_tau(float(T1[0, 0]))
            sol = _solve_scalar_block(
                A_hat, A_plus, tau, D_tilde[:, i1],
                lu_cache, max_cache, dgetrf, dgetrs, dgecon, tracker,
            )
            X_tilde[:, i1] = sol
            if e1 < n_x:
                row = T_h[i1, e1:]
                nz = np.flatnonzero(row)
                if len(nz) > 0:
                    wj = A_plus @ sol
                    D_tilde[:, e1 + nz] -= np.outer(wj, row[nz])
                    tracker.back_substitutions += 1
        else:
            sol_blk = _solve_quasi_block(A_hat, A_plus, T1, D_tilde[:, i1:e1], tracker)
            X_tilde[:, i1:e1] = sol_blk
            if e1 < n_x:
                T_off = T_h[i1:e1, e1:]
                if np.any(T_off != 0.0):
                    D_tilde[:, e1:] -= (A_plus @ sol_blk) @ T_off
                    tracker.back_substitutions += 1

    lu_cache.clear()
    del lu_cache, D_tilde

    g_x_ss = X_tilde @ U_h.T
    out = np.ascontiguousarray(g_x_ss, dtype=float)
    counts = tracker.freeze()
    _LAST_WORK_COUNTS[1] = counts
    return (out, counts) if return_work_counts else out

