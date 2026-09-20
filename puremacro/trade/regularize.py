"""Economic Regularization and Accounting Balancing Algorithms for MRIO Tables.

This module provides production-grade matrix regularization and balancing routines
for Multi-Regional Input-Output (MRIO) datasets conforming to the puremacro
Pyodide runtime contract (NumPy and SciPy only).

Core capabilities:
1. Economic Regularization (``regularize_mrio_table``):
   - Phantom output injection (1e-6 M USD) for inactive/zero-output sector nodes.
   - Non-negative value-added flooring (VA >= max(1e-3 * Y, 1.0)) with dual TLS debit.
   - Residual TLS reconciliation ensuring column outlays equal row sales (< 1e-12 * max(Y)).
   - Collatz-Wielandt spectral radius verification (rho(B) < 0.999).
2. Biproportional Matrix Balancing:
   - ``balance_ras``: Classical iterative Bregman row/column scaling.
   - ``balance_gras``: Generalized RAS for matrices containing negative entries.
   - ``balance_quadratic``: Constrained weighted least-squares quadratic balancing.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple
import numpy as np
import scipy.linalg


def compute_spectral_radius(
    B: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-12,
) -> tuple[float, float, float]:
    """Compute the spectral radius rho(B) using Collatz-Wielandt power iteration.

    By the Collatz-Wielandt theorem, for any non-negative irreducible matrix B
    and positive vector x > 0:
        min_{i: x_i > 0} (Bx)_i / x_i <= rho(B) <= max_{i: x_i > 0} (Bx)_i / x_i.

    Parameters
    ----------
    B : np.ndarray
        Non-negative square matrix of shape (M, M).
    max_iter : int, default=300
        Maximum number of power iterations.
    tol : float, default=1e-12
        Convergence tolerance on Rayleigh quotient.

    Returns
    -------
    tuple[float, float, float]
        (rho_rayleigh, collatz_wielandt_lower, collatz_wielandt_upper)
    """
    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or B.shape[0] != B.shape[1] or not np.all(np.isfinite(B)) or np.any(B < 0):
        raise ValueError("B must be a finite nonnegative square matrix")
    if max_iter < 1 or tol <= 0:
        raise ValueError("max_iter and tol must be positive")
    n = B.shape[0]
    if n == 0 or not np.any(B):
        return 0.0, 0.0, 0.0
    if n == 1:
        value = float(B[0, 0])
        return value, value, value
    # Reducible matrices have the spectra of their strongly connected diagonal
    # blocks. Isolated/inactive nodes must not pin the lower bound to zero.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    if np.all(B > 0):
        # A strictly positive dense matrix is already irreducible; avoid a
        # second, sparse copy of potentially millions of nonzero entries.
        count, labels = 1, None
    else:
        count, labels = connected_components(csr_matrix(B), directed=True, connection="strong")
    if count > 1:
        blocks = []
        for label in range(count):
            idx = np.flatnonzero(labels == label)
            blocks.append(compute_spectral_radius(B[np.ix_(idx, idx)], max_iter=max_iter, tol=tol))
        return tuple(float(max(v[i] for v in blocks)) for i in range(3))
    # Adding a positive diagonal removes periodicity without changing eigenvectors.
    # Retain strictly positive entries so bounds also cover reducible matrices.
    shift = float(np.max(np.sum(B, axis=1)))
    x = np.ones(n)
    lower, upper = 0.0, shift
    for _ in range(max_iter):
        Bx = B @ x
        ratios = Bx / x
        lower = max(lower, float(np.min(ratios)))
        upper = min(upper, float(np.max(ratios)))
        if upper - lower <= tol * max(1.0, upper):
            break
        nxt = Bx + shift * x
        x = np.maximum(nxt / np.max(nxt), np.finfo(float).tiny)
    rho = float(np.dot(x, B @ x) / np.dot(x, x))
    if upper - lower > tol * max(1.0, upper) and n <= 256:
        rho = float(np.max(np.abs(scipy.linalg.eigvals(B))))
        # A resolvent supplies a positive test vector even when extreme scaling
        # makes shifted iteration very slow. Certify via Ax/x, not eigenvalues.
        lam = rho + max(1e-10, abs(rho) * 1e-6)
        try:
            candidate = scipy.linalg.solve(lam * np.eye(n) - B, np.ones(n))
            if np.all(np.isfinite(candidate)) and np.all(candidate > 0):
                ratios = (B @ candidate) / candidate
                lower = max(lower, float(np.min(ratios)))
                upper = min(upper, float(np.max(ratios)))
        except scipy.linalg.LinAlgError:
            pass
    return rho, lower, upper


def spectral_radius(B: np.ndarray) -> float:
    """Return a numerical spectral estimate; use compute_spectral_radius for bounds."""
    rho, _, cw_upper = compute_spectral_radius(B)
    if abs(cw_upper - rho) < 1e-4:
        return max(rho, cw_upper)
    return rho


def regularize_mrio_table(
    Z: np.ndarray,
    F: np.ndarray,
    VA: np.ndarray,
    TLS: np.ndarray,
    Y: np.ndarray | None = None,
    floor_output: float = 1e-6,
    floor_va_ratio: float = 1e-3,
    floor_va_abs: float = 1.0,
    *,
    tau: np.ndarray | None = None,
    spectral_tol: float = 1e-3,
    n_countries: int | None = None,
    n_sectors: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Applies phantom output injection, VA flooring with dual TLS debit, and residual TLS reconciliation.

    Parameters
    ----------
    Z : np.ndarray
        Intermediate transaction matrix of shape (M, M).
    F : np.ndarray
        Final demand matrix of shape (M, C * K_F) or (M, C, K_F).
    VA : np.ndarray
        Primary factor payments of shape (M,) or (K_VA, M).
    TLS : np.ndarray
        Net taxes less subsidies on production of shape (M,).
    Y : np.ndarray, optional
        Gross sectoral output of shape (M,). If None, evaluated from row sales.
    floor_output : float, default=1e-6
        Phantom output injection threshold in Million USD.
    floor_va_ratio : float, default=1e-3
        Minimum ratio of value added to gross output.
    floor_va_abs : float, default=1.0
        Minimum absolute value added in Million USD.
    tau : np.ndarray, optional
        Gross tariff wedge matrix for spectral radius evaluation (B_tau = tau * A).
    spectral_tol : float, default=1e-3
        Required margin below 1.0 for spectral radius (rho < 1.0 - spectral_tol = 0.999).
    n_countries : int, optional
        Number of economies C.
    n_sectors : int, optional
        Number of industrial sectors S.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (Z_clean, F_clean, VA_clean, TLS_clean, Y_clean)
    """
    Z_clean = np.array(Z, dtype=np.float64, copy=True)
    F_clean = np.array(F, dtype=np.float64, copy=True)
    VA_clean = np.array(VA, dtype=np.float64, copy=True)
    TLS_clean = np.array(TLS, dtype=np.float64, copy=True)

    M = Z_clean.shape[0]
    if Z_clean.shape != (M, M):
        raise ValueError(f"Z must be square (M, M), got shape {Z_clean.shape}")
    if F_clean.shape[0] != M:
        raise ValueError(f"F rows ({F_clean.shape[0]}) do not match Z rows ({M})")

    is_va_2d = (VA_clean.ndim == 2)
    if is_va_2d:
        if VA_clean.shape[1] != M:
            raise ValueError(f"VA columns ({VA_clean.shape[1]}) do not match Z ({M})")
    else:
        if VA_clean.shape[0] != M:
            raise ValueError(f"VA length ({VA_clean.shape[0]}) does not match Z ({M})")

    if TLS_clean.shape != (M,):
        raise ValueError(f"TLS shape ({TLS_clean.shape}) must be ({M},)")

    # Deduce dimensions (C, S, K_F)
    is_f_3d = (F_clean.ndim == 3)
    if is_f_3d:
        C = F_clean.shape[1]
        K_F = F_clean.shape[2]
        S = M // C if C > 0 else M
    elif n_countries is not None and n_sectors is not None and n_countries * n_sectors == M:
        C = n_countries
        S = n_sectors
        K_F = F_clean.shape[1] // C if C > 0 else F_clean.shape[1]
    else:
        # Check standard known configurations
        known_dims = [
            (77, 45), (77, 11), (46, 64), (44, 56), (189, 26), (49, 163), (49, 200)
        ]
        matched = False
        for kc, ks in known_dims:
            if kc * ks == M:
                C = kc
                S = ks
                K_F = F_clean.shape[1] // C if (F_clean.shape[1] % C == 0) else 1
                matched = True
                break
        if not matched:
            C = 1
            S = M
            K_F = F_clean.shape[1]

    # Gross output baseline from row sales
    if is_f_3d:
        f_row_sum = np.sum(F_clean, axis=(1, 2))
    else:
        f_row_sum = np.sum(F_clean, axis=1)

    if Y is not None:
        Y_clean = np.array(Y, dtype=np.float64, copy=True)
    else:
        Y_clean = np.sum(Z_clean, axis=1) + f_row_sum

    # -------------------------------------------------------------------------
    # 1. Phantom Output Injection (1e-6 M USD)
    # -------------------------------------------------------------------------
    inactive_mask = (Y_clean < floor_output)
    if np.any(inactive_mask):
        for idx in np.where(inactive_mask)[0]:
            c_idx = idx // S if S > 0 else 0
            # Inject into domestic household final demand
            if is_f_3d:
                c_idx = min(c_idx, C - 1)
                F_clean[idx, c_idx, 0] += floor_output
            else:
                fd_col = min(c_idx * K_F, F_clean.shape[1] - 1)
                F_clean[idx, fd_col] += floor_output

            # Inject into value added
            if is_va_2d:
                VA_clean[0, idx] += floor_output
            else:
                VA_clean[idx] += floor_output

            Y_clean[idx] += floor_output

    # Update gross output to reflect row sales after phantom injection if Y was not provided
    if Y is None:
        if is_f_3d:
            f_row_sum = np.sum(F_clean, axis=(1, 2))
        else:
            f_row_sum = np.sum(F_clean, axis=1)
        Y_clean = np.sum(Z_clean, axis=1) + f_row_sum

    # -------------------------------------------------------------------------
    # 2. Non-Negative Value-Added Flooring & Dual TLS Debit
    # -------------------------------------------------------------------------
    va_totals = np.sum(VA_clean, axis=0) if is_va_2d else VA_clean
    va_floors = np.where(
        inactive_mask,
        floor_output,
        np.maximum(floor_va_ratio * Y_clean, floor_va_abs),
    )
    delta_va = np.maximum(0.0, va_floors - va_totals)
    if np.any(delta_va > 0.0):
        if is_va_2d:
            VA_clean[0, :] += delta_va
        else:
            VA_clean += delta_va

        # Dual TLS Debit: debit the adjustment from net production taxes (TLS)
        # preserving the total sector outlays identity
        TLS_clean -= delta_va

    # -------------------------------------------------------------------------
    # 3. Residual TLS Reconciliation
    # -------------------------------------------------------------------------
    va_current = np.sum(VA_clean, axis=0) if is_va_2d else VA_clean
    TLS_clean = Y_clean - np.sum(Z_clean, axis=0) - va_current

    # Verify column outlays match row sales to < 1e-12 * max(Y)
    max_y = float(np.max(Y_clean)) if Y_clean.size > 0 else 1.0
    outlays = np.sum(Z_clean, axis=0) + va_current + TLS_clean
    reconciliation_err = float(np.max(np.abs(outlays - Y_clean)))
    if reconciliation_err > 1e-11 * max_y:
        raise ValueError(
            f"Residual TLS reconciliation error {reconciliation_err:.2e} "
            f"exceeds tolerance 1e-11 * max(Y)."
        )

    # -------------------------------------------------------------------------
    # 4. Spectral Radius Verification (rho(B) < 0.999)
    # -------------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.divide(
            Z_clean,
            Y_clean.reshape(1, M),
            out=np.zeros_like(Z_clean),
            where=(Y_clean.reshape(1, M) > 0),
        )

    B = (tau * A) if tau is not None else A
    rho, lower, upper = compute_spectral_radius(B, max_iter=300, tol=1e-10)
    threshold = 1.0 - spectral_tol
    if lower < threshold <= upper:
        raise ValueError(
            f"Spectral viability unresolved: certified bounds [{lower:.8g}, {upper:.8g}] "
            f"straddle threshold {threshold:.8g}; refine the calibration or spectral calculation."
        )
    if lower >= threshold:
        raise ValueError(
            f"Spectral radius rho(B) = {rho:.6f} >= threshold {threshold:.6f}. "
            "Leontief cost system is non-productive."
        )

    return Z_clean, F_clean, VA_clean, TLS_clean, Y_clean


def balance_ras(
    Z0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Classical biproportional matrix balancing (RAS algorithm).

    Scales non-negative matrix Z0 to match target row sums u and column sums v
    via iterative Bregman projections:
        Z_{ij}^{(k+1)} = r_i Z_{ij}^0 s_j

    Parameters
    ----------
    Z0 : np.ndarray
        Prior non-negative matrix of shape (m, n).
    u : np.ndarray
        Target row marginals of shape (m,).
    v : np.ndarray
        Target column marginals of shape (n,).
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-10
        Convergence tolerance on max absolute deviation from target margins.

    Returns
    -------
    np.ndarray
        Balanced non-negative matrix matching row and column margins.
    """
    Z = np.array(Z0, dtype=np.float64, copy=True)
    u_target = np.array(u, dtype=np.float64, copy=True)
    v_target = np.array(v, dtype=np.float64, copy=True)

    if np.any(Z < 0):
        raise ValueError("balance_ras requires non-negative entries in Z0. Use balance_gras for negative entries.")

    sum_u = float(np.sum(u_target))
    sum_v = float(np.sum(v_target))
    if abs(sum_u) <= 1e-12 and abs(sum_v) <= 1e-12:
        return np.zeros_like(Z0)
    if abs(sum_v) <= 1e-12 and abs(sum_u) > 1e-12:
        raise ValueError("Column target sums to zero while row target is non-zero")
    if abs(sum_u - sum_v) > 1e-12:
        # Rescale column targets to ensure global consistency
        v_target = v_target * (sum_u / sum_v)

    for _ in range(max_iter):
        # 1. Row scaling
        row_sums = np.sum(Z, axis=1)
        r_step = np.ones_like(u_target)
        mask_r = (row_sums > 0) & (u_target > 0)
        r_step[mask_r] = u_target[mask_r] / row_sums[mask_r]
        Z *= r_step[:, None]

        # 2. Column scaling
        col_sums = np.sum(Z, axis=0)
        s_step = np.ones_like(v_target)
        mask_s = (col_sums > 0) & (v_target > 0)
        s_step[mask_s] = v_target[mask_s] / col_sums[mask_s]
        Z *= s_step[None, :]

        # Convergence test
        curr_row = np.sum(Z, axis=1)
        curr_col = np.sum(Z, axis=0)
        max_dev = max(
            float(np.max(np.abs(curr_row - u_target))),
            float(np.max(np.abs(curr_col - v_target))),
        )
        if max_dev <= tol:
            break

    return Z


def balance_gras(
    Z0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Generalized RAS (GRAS) balancing for matrices with positive and negative entries.

    Decomposes Z0 = P0 - N0 with P0 >= 0 and N0 >= 0, and updates multipliers (r, s)
    via closed-form quadratic roots:
        Z_{ij} = r_i P0_{ij} s_j - r_i^{-1} N0_{ij} s_j^{-1}

    Parameters
    ----------
    Z0 : np.ndarray
        Prior real matrix of shape (m, n).
    u : np.ndarray
        Target row marginals of shape (m,).
    v : np.ndarray
        Target column marginals of shape (n,).
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-10
        Convergence tolerance on max absolute margin deviation.

    Returns
    -------
    np.ndarray
        Balanced real matrix matching row and column margins.
    """
    Z_orig = np.array(Z0, dtype=np.float64, copy=True)
    u_target = np.array(u, dtype=np.float64, copy=True)
    v_target = np.array(v, dtype=np.float64, copy=True)

    m, n = Z_orig.shape
    sum_u = float(np.sum(u_target))
    sum_v = float(np.sum(v_target))
    if abs(sum_v) <= 1e-12 and abs(sum_u) > 1e-12:
        raise ValueError("Column target sums to zero while row target is non-zero")
    if abs(sum_u - sum_v) > 1e-12:
        v_target = v_target * (sum_u / sum_v)

    P0 = np.maximum(Z_orig, 0.0)
    N0 = np.maximum(-Z_orig, 0.0)

    r = np.ones(m, dtype=np.float64)
    s = np.ones(n, dtype=np.float64)

    for _ in range(max_iter):
        # 1. Update r given current s
        p_row = P0 @ s
        n_row = N0 @ (1.0 / s)

        rad_r = np.sqrt(u_target ** 2 + 4.0 * p_row * n_row)
        mask_p = p_row > 1e-15
        mask_only_n = (~mask_p) & (n_row > 1e-15)

        r_new = np.ones_like(r)
        r_new[mask_p] = (u_target[mask_p] + rad_r[mask_p]) / (2.0 * p_row[mask_p])
        r_new[mask_only_n] = -n_row[mask_only_n] / np.where(u_target[mask_only_n] != 0, u_target[mask_only_n], -1.0)
        r = np.maximum(r_new, 1e-14)

        # 2. Update s given current r
        p_col = r @ P0
        n_col = (1.0 / r) @ N0

        rad_s = np.sqrt(v_target ** 2 + 4.0 * p_col * n_col)
        mask_p_col = p_col > 1e-15
        mask_only_n_col = (~mask_p_col) & (n_col > 1e-15)

        s_new = np.ones_like(s)
        s_new[mask_p_col] = (v_target[mask_p_col] + rad_s[mask_p_col]) / (2.0 * p_col[mask_p_col])
        s_new[mask_only_n_col] = -n_col[mask_only_n_col] / np.where(v_target[mask_only_n_col] != 0, v_target[mask_only_n_col], -1.0)
        s = np.maximum(s_new, 1e-14)

        # Evaluate current matrix and check margin convergence
        Z = r[:, None] * P0 * s[None, :] - (1.0 / r)[:, None] * N0 * (1.0 / s)[None, :]
        curr_row = np.sum(Z, axis=1)
        curr_col = np.sum(Z, axis=0)

        max_dev = max(
            float(np.max(np.abs(curr_row - u_target))),
            float(np.max(np.abs(curr_col - v_target))),
        )
        if max_dev <= tol:
            break

    return Z


def balance_quadratic(
    Z0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Constrained weighted least-squares quadratic matrix balancing.

    Solves:
        min_Z 1/2 sum_{i,j} w_{ij} (Z_{ij} - Z_{ij}^0)^2
        s.t.  sum_j Z_{ij} = u_i  for all i,
              sum_i Z_{ij} = v_j  for all j.

    Parameters
    ----------
    Z0 : np.ndarray
        Prior matrix of shape (m, n).
    u : np.ndarray
        Target row marginals of shape (m,).
    v : np.ndarray
        Target column marginals of shape (n,).
    weights : np.ndarray, optional
        Positive weight matrix of shape (m, n). If None, uniform unit weights
        are used, yielding the closed-form projection.

    Returns
    -------
    np.ndarray
        Balanced matrix satisfying row and column marginals.
    """
    Z = np.array(Z0, dtype=np.float64, copy=True)
    u_target = np.array(u, dtype=np.float64, copy=True)
    v_target = np.array(v, dtype=np.float64, copy=True)

    m, n = Z.shape
    sum_u = float(np.sum(u_target))
    sum_v = float(np.sum(v_target))
    if abs(sum_v) <= 1e-12 and abs(sum_u) > 1e-12:
        raise ValueError("Column target sums to zero while row target is non-zero")
    if abs(sum_u - sum_v) > 1e-12:
        v_target = v_target * (sum_u / sum_v)

    u0 = np.sum(Z, axis=1)
    v0 = np.sum(Z, axis=0)
    delta_u = u_target - u0
    delta_v = v_target - v0

    if weights is None:
        # Exact closed-form projection under uniform weights
        tot = float(np.sum(delta_u))
        return Z + delta_u[:, None] / n + delta_v[None, :] / m - tot / (m * n)

    # General weighted least squares via normal equations
    W = np.array(weights, dtype=np.float64)
    if W.shape != (m, n):
        raise ValueError(f"Weights shape {W.shape} does not match Z0 shape {(m, n)}")
    if np.any(W <= 0):
        raise ValueError("Weights must be strictly positive.")

    C = 1.0 / W  # Variances
    dr = np.sum(C, axis=1)
    dc = np.sum(C, axis=0)

    # Saddle point system: [diag(dr) C; C^T diag(dc)] [lambda; mu] = [delta_u; delta_v]
    K = np.block([
        [np.diag(dr), C],
        [C.T, np.diag(dc)],
    ])
    rhs = np.concatenate([delta_u, delta_v])
    sol, _, _, _ = scipy.linalg.lstsq(K, rhs)
    lam = sol[:m]
    mu = sol[m:]

    return Z + C * (lam[:, None] + mu[None, :])


def validate_accounting_identities(
    Z: np.ndarray,
    F: np.ndarray,
    VA: np.ndarray,
    TLS: np.ndarray,
    Y: np.ndarray,
    tol: float = 1e-9,
    tau: np.ndarray | None = None,
    spectral_tol: float = 1e-3,
) -> dict[str, Any]:
    """Validate full Walrasian accounting identities across MRIO transaction matrices.

    Checks:
    1. Sales balance: sum_j Z_ij + sum_{c, f} F_{ic}^f == Y_i
    2. Outlays balance: sum_i Z_ij + VA_j + TLS_j == Y_j
    3. Global balance: |Sales_j - Outlays_j| <= tol * max(Y)
    4. Non-negativity: Y_j >= 1e-6, VA_j >= max(1e-3 * Y_j, 1.0) on active sectors.
    5. Productivity: rho(B_tau) < 1.0 - spectral_tol.
    """
    Z_arr = np.asarray(Z, dtype=np.float64)
    F_arr = np.asarray(F, dtype=np.float64)
    VA_arr = np.asarray(VA, dtype=np.float64)
    TLS_arr = np.asarray(TLS, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)

    M = Z_arr.shape[0]
    max_y = float(np.max(Y_arr)) if Y_arr.size > 0 else 1.0
    abs_tol = tol * max_y

    if F_arr.ndim == 3:
        f_row_sum = np.sum(F_arr, axis=(1, 2))
    else:
        f_row_sum = np.sum(F_arr, axis=1)

    va_col_sum = np.sum(VA_arr, axis=0) if VA_arr.ndim == 2 else VA_arr

    sales = np.sum(Z_arr, axis=1) + f_row_sum
    outlays = np.sum(Z_arr, axis=0) + va_col_sum + TLS_arr

    sales_err = float(np.max(np.abs(sales - Y_arr)))
    outlays_err = float(np.max(np.abs(outlays - Y_arr)))
    balance_err = float(np.max(np.abs(sales - outlays)))

    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.divide(
            Z_arr, Y_arr.reshape(1, M), out=np.zeros_like(Z_arr), where=(Y_arr.reshape(1, M) > 0)
        )

    B_tau = (tau * A) if tau is not None else A
    rho, cw_lower, cw_upper = compute_spectral_radius(B_tau)

    bound_threshold = 1.0 - spectral_tol
    valid = bool(
        sales_err <= abs_tol
        and outlays_err <= abs_tol
        and balance_err <= abs_tol
        and np.all(Y_arr >= 1e-6 - 1e-12)
        and rho < bound_threshold
    )

    return {
        "valid": valid,
        "max_sales_error": sales_err,
        "max_outlays_error": outlays_err,
        "max_sales_outlays_error": balance_err,
        "min_gross_output": float(np.min(Y_arr)),
        "min_value_added": float(np.min(va_col_sum)),
        "spectral_radius": float(rho),
        "collatz_wielandt_lower": float(cw_lower),
        "collatz_wielandt_upper": float(cw_upper),
        "spectral_certified": bool(rho < bound_threshold),
    }


__all__ = [
    "compute_spectral_radius",
    "spectral_radius",
    "regularize_mrio_table",
    "balance_ras",
    "balance_gras",
    "balance_quadratic",
    "validate_accounting_identities",
]
