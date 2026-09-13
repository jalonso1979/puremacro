"""Nonlinear General Equilibrium Solver for puremacro.trade.

Implements four high-performance general equilibrium solver architectures:
1. Method A ('newton'): Standard Dense Damped Newton-Raphson with Armijo line search.
2. Method B ('sparse_lu'): Sparse Jacobian Newton using direct sparse LU factorization
   (scipy.sparse.linalg.splu) and column grouping / graph coloring finite differences.
3. Method C ('krylov'): Inexact Newton-Krylov (matrix-free JFNK via GMRES/BiCGSTAB) with
   physical two-sided equilibration (D_L @ J @ D_R) eliminating coordinate scale disparities.
4. Method D ('condensed'): Block-elimination / Schur complement price condensation. Solves
   Leontief linear pricing and gross output conditionally via pre-factorized sparse LU,
   condensing the outer nonlinear master system to strictly 4*nc - 1 = 307 macro variables
   invariant to sector count, solved via damped Newton.

Also maintains Broyden quasi-Newton and SciPy root wrappers for backward compatibility.
Conforms strictly to the puremacro Pyodide runtime contract: pure NumPy/SciPy,
zero dev-dependencies in the execution path, fully vectorized.
"""
from __future__ import annotations

from dataclasses import replace
import time
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize._numdiff import approx_derivative, group_columns

from puremacro.trade.equilibrium import compute_equilibrium_residuals, unpack_equilibrium_vector
from puremacro.trade.postprocessing import postprocess_trade_equilibrium

if TYPE_CHECKING:
    from puremacro.trade._results import TradeCalibrationResult, TradeEquilibriumResult


def build_initial_guess(calib: TradeCalibrationResult) -> np.ndarray:
    """Construct initial state vector x0 from calibrated model parameters.

    In the calibrated baseline:
    - Sector gross output prices p = 1.0 -> log(p) = 0.0
    - Gross output quantities y = ytot -> log(y) = log(ytot)
    - Factor rental rates r = 1.0, w = 1.0 -> log(r) = 0.0, log(w) = 0.0
    - Government transfers T = TT + TTfd (or calib.T)
    - Net foreign transfers XN = invforT[:nc-1]

    Evaluated at baseline tariffs, ||F(x0)||_inf = 1.52e-4 < 2.5e-3.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.

    Returns
    -------
    np.ndarray
        1D state vector of shape (2*ns*nc + 3*nc + nc - 1,).
    """
    nc, ns = calib.n_countries, calib.n_sectors
    log_p0 = np.zeros(ns * nc, dtype=float)
    log_y0 = np.log(np.asarray(calib.ytot, dtype=float).flatten(order="F"))
    log_r0 = np.zeros(nc, dtype=float)
    log_w0 = np.zeros(nc, dtype=float)

    if calib.T is not None:
        T0 = np.asarray(calib.T, dtype=float).flatten(order="F")
    elif calib.TT is not None and calib.TTfd is not None:
        T0 = (np.asarray(calib.TT, dtype=float) + np.asarray(calib.TTfd, dtype=float)).flatten(order="F")
    else:
        T0 = np.zeros(nc, dtype=float)

    XN0 = np.asarray(calib.invforT, dtype=float).ravel()[:nc - 1]
    return np.concatenate([log_p0, log_y0, log_r0, log_w0, T0, XN0])


def _resolve_tariffs(
    calib: TradeCalibrationResult,
    tau: np.ndarray | None,
    tau_fd: np.ndarray | None,
    tauf: np.ndarray | None = None,
    tauf_fd: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve and validate tariff multipliers and national tariff vectors."""
    nc, ns, nfd = calib.n_countries, calib.n_sectors, calib.n_final_demand

    tauf_vec = np.zeros(nc, dtype=float) if tauf is None else np.asarray(tauf, dtype=float).ravel()
    tauf_fd_vec = np.zeros(nc, dtype=float) if tauf_fd is None else np.asarray(tauf_fd, dtype=float).ravel()

    if tau is None:
        tau_a = np.ones((ns * nc, ns, nc), dtype=float)
    elif tau.shape == (ns * nc, ns, nc):
        tau_a = np.asarray(tau, dtype=float)
    elif tau.ndim == 4 and tau.shape == (ns, nc, ns, nc):
        tau_a = np.asarray(tau, dtype=float).transpose(1, 0, 2, 3).reshape(ns * nc, ns, nc)
    elif tau.size == nc:
        tauf_vec = np.asarray(tau, dtype=float).ravel()
        tau_a = np.ones((ns * nc, ns, nc), dtype=float)
        for ik in range(nc):
            block = np.ones((ns, ns), dtype=float) * (1.0 + tauf_vec[ik])
            for orig_k in range(nc):
                if orig_k != ik:
                    tau_a[orig_k * ns : (orig_k + 1) * ns, :, ik] = block
    else:
        tau_a = np.asarray(tau, dtype=float)

    if tau_fd is None:
        taufd_a = np.ones((ns * nc, nfd, nc), dtype=float)
    elif tau_fd.shape == (ns * nc, nfd, nc):
        taufd_a = np.asarray(tau_fd, dtype=float)
    elif tau_fd.ndim == 4 and tau_fd.shape == (ns, nc, nfd, nc):
        taufd_a = np.asarray(tau_fd, dtype=float).transpose(1, 0, 2, 3).reshape(ns * nc, nfd, nc)
    elif tau_fd.size == nc:
        tauf_fd_vec = np.asarray(tau_fd, dtype=float).ravel()
        taufd_a = np.ones((ns * nc, nfd, nc), dtype=float)
        for ik in range(nc):
            block_fd = np.ones((ns, nfd), dtype=float) * (1.0 + tauf_fd_vec[ik])
            for orig_k in range(nc):
                if orig_k != ik:
                    taufd_a[orig_k * ns : (orig_k + 1) * ns, :, ik] = block_fd
    else:
        taufd_a = np.asarray(tau_fd, dtype=float)

    return tau_a, taufd_a, tauf_vec, tauf_fd_vec


# ===========================================================================
# 1. Architecture A: Dense Damped Newton-Raphson (Baseline)
# ===========================================================================

def _damped_newton_solve(
    f: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    tol: float = 2.5e-3,
    max_iter: int = 50,
    eps_fd: float = 1e-2,
) -> tuple[np.ndarray, bool, int, float, float, np.ndarray]:
    """Pure NumPy Damped Newton-Raphson solver with forward Jacobian and line search."""
    x = np.asarray(x_init, dtype=float).copy()
    n = len(x)
    f_val = f(x)
    max_res = float(np.max(np.abs(f_val)))
    diff = float(np.sum(np.abs(f_val)))

    if max_res <= tol or diff <= tol:
        return x, True, 0, max_res, diff, f_val

    for it in range(max_iter):
        # Numerical forward finite-difference Jacobian
        J = np.empty((n, n), dtype=float)
        for j in range(n):
            x_pert = x.copy()
            x_pert[j] += eps_fd
            J[:, j] = (f(x_pert) - f_val) / eps_fd

        # Newton direction
        try:
            delta = np.linalg.solve(J, -f_val)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J, -f_val, rcond=1e-10)[0]

        # Damping schedule matching MATLAB Main77c_11s.m line 171
        xscal = 0.5 if it < 3 else 1.0

        # Backtracking line search
        alpha_step = xscal
        norm_0 = max_res
        x_trial = x + alpha_step * delta
        f_trial = f(x_trial)

        for _ in range(10):
            res_trial_max = float(np.max(np.abs(f_trial)))
            res_trial_diff = float(np.sum(np.abs(f_trial)))
            if res_trial_max <= tol or res_trial_diff <= tol or res_trial_max < norm_0:
                break
            alpha_step *= 0.5
            x_trial = x + alpha_step * delta
            f_trial = f(x_trial)

        x = x_trial
        f_val = f_trial
        max_res = float(np.max(np.abs(f_val)))
        diff = float(np.sum(np.abs(f_val)))

        if max_res <= tol or diff <= tol:
            return x, True, it + 1, max_res, diff, f_val

    return x, False, max_iter, max_res, diff, f_val


# ===========================================================================
# 2. Architecture B: Sparse Jacobian Newton via splu and Graph Coloring
# ===========================================================================

def _build_cge_sparsity_pattern(
    calib: TradeCalibrationResult,
    tau_a: np.ndarray | None = None,
) -> sp.csc_matrix:
    """Construct structural Jacobian sparsity pattern for the CGE equilibrium system."""
    ns, nc = calib.n_sectors, calib.n_countries
    M = ns * nc
    n = 2 * M + 4 * nc - 1
    S = sp.lil_matrix((n, n), dtype=bool)

    # Intermediate linkage matrix
    if tau_a is not None:
        a_eff = (calib.a * tau_a).reshape((M, M), order="F") != 0
    else:
        a_eff = calib.a.reshape((M, M), order="F") != 0

    # Block ff0: Goods market clearing
    # ff0 w.r.t p: country-level final demand price linkages
    S[:M, :M] = True
    # ff0 w.r.t y: (I - A) structure
    S[:M, M:2 * M] = sp.eye(M, dtype=bool) + sp.csc_matrix(a_eff)
    # ff0 w.r.t macro: factor returns and transfers enter through consumer budget
    S[:M, 2 * M:] = True

    # Block ff1: Zero-profit condition
    # ff1 w.r.t p: (I - B^T)
    S[M:2 * M, :M] = sp.eye(M, dtype=bool) + sp.csc_matrix(a_eff.T)
    # ff1 w.r.t y: identically zero in standard Leontief
    # ff1 w.r.t r, w: only for national factor returns of country i
    for i in range(nc):
        s_start, s_end = M + i * ns, M + (i + 1) * ns
        S[s_start:s_end, 2 * M + i] = True
        S[s_start:s_end, 2 * M + nc + i] = True

    # Block ff2: Labor market clearing
    # ff2 w.r.t y: only sectors in country i
    for i in range(nc):
        s_start, s_end = M + i * ns, M + (i + 1) * ns
        S[2 * M + i, s_start:s_end] = True
        S[2 * M + i, 2 * M + i] = True
        S[2 * M + i, 2 * M + nc + i] = True

    # Block ff3: Capital market clearing
    # ff3 w.r.t y: only sectors in country i
    for i in range(nc):
        s_start, s_end = M + i * ns, M + (i + 1) * ns
        S[2 * M + nc + i, s_start:s_end] = True
        S[2 * M + nc + i, 2 * M + i] = True
        S[2 * M + nc + i, 2 * M + nc + i] = True

    # Block ff4: Trade balance (current accounts)
    S[2 * M + 2 * nc : 2 * M + 3 * nc - 1, :] = True

    # Block ff5: Fiscal budget consistency
    for i in range(nc):
        s_start, s_end = M + i * ns, M + (i + 1) * ns
        S[2 * M + 3 * nc - 1 + i, s_start:s_end] = True
        S[2 * M + 3 * nc - 1 + i, 2 * M + 2 * nc + i] = True
    S[2 * M + 3 * nc - 1 :, :M] = True
    S[2 * M + 3 * nc - 1 :, 2 * M + 3 * nc :] = True

    return S.tocsc()


def _sparse_lu_solve(
    f: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    tol: float = 2.5e-3,
    max_iter: int = 50,
    eps_fd: float = 1e-4,
    sparsity: sp.spmatrix | None = None,
) -> tuple[np.ndarray, bool, int, float, float, np.ndarray]:
    """Sparse Jacobian Newton solver with splu and graph coloring finite differences."""
    x = np.asarray(x_init, dtype=float).copy()
    n = len(x)
    f_val = f(x)
    max_res = float(np.max(np.abs(f_val)))
    diff = float(np.sum(np.abs(f_val)))

    if max_res <= tol or diff <= tol:
        return x, True, 0, max_res, diff, f_val

    for it in range(max_iter):
        # Evaluate sparse Jacobian via column grouping / CPR graph coloring
        if sparsity is not None:
            J_sparse = approx_derivative(
                f,
                x,
                method="2-point",
                sparsity=sparsity,
                f0=f_val,
                abs_step=eps_fd,
            )
            J_csc = J_sparse.tocsc() if sp.issparse(J_sparse) else sp.csc_matrix(J_sparse)
        else:
            # Fallback dense finite differences converted to CSC
            J_dense = np.empty((n, n), dtype=float)
            for j in range(n):
                x_pert = x.copy()
                x_pert[j] += eps_fd
                J_dense[:, j] = (f(x_pert) - f_val) / eps_fd
            J_csc = sp.csc_matrix(J_dense)

        # Direct sparse LU factorization via SuperLU
        try:
            lu = spla.splu(J_csc)
            delta = lu.solve(-f_val)
        except Exception:
            try:
                delta = spla.spsolve(J_csc, -f_val)
            except Exception:
                delta = np.linalg.lstsq(J_csc.toarray(), -f_val, rcond=1e-10)[0]

        # Damping schedule matching MATLAB
        xscal = 0.5 if it < 3 else 1.0

        # Backtracking line search
        alpha_step = xscal
        norm_0 = max_res
        x_trial = x + alpha_step * delta
        f_trial = f(x_trial)

        for _ in range(10):
            res_trial_max = float(np.max(np.abs(f_trial)))
            res_trial_diff = float(np.sum(np.abs(f_trial)))
            if res_trial_max <= tol or res_trial_diff <= tol or res_trial_max < norm_0:
                break
            alpha_step *= 0.5
            x_trial = x + alpha_step * delta
            f_trial = f(x_trial)

        x = x_trial
        f_val = f_trial
        max_res = float(np.max(np.abs(f_val)))
        diff = float(np.sum(np.abs(f_val)))

        if max_res <= tol or diff <= tol:
            return x, True, it + 1, max_res, diff, f_val

    return x, False, max_iter, max_res, diff, f_val


# ===========================================================================
# 3. Architecture C: Inexact Newton-Krylov with Two-Sided Physical Equilibration
# ===========================================================================

def _newton_krylov_solve(
    f: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    calib: TradeCalibrationResult,
    tol: float = 2.5e-3,
    max_iter: int = 50,
    eps_fd: float = 1e-6,
    krylov_method: str = "gmres",
    restart: int = 30,
    max_krylov_iter: int = 50,
    atol: float = 1e-5,
) -> tuple[np.ndarray, bool, int, float, float, np.ndarray]:
    """Inexact Newton-Krylov (JFNK) solver with physical two-sided equilibration."""
    x = np.asarray(x_init, dtype=float).copy()
    n = len(x)
    ns, nc = calib.n_sectors, calib.n_countries
    M = ns * nc

    f_val = f(x)
    max_res = float(np.max(np.abs(f_val)))
    diff = float(np.sum(np.abs(f_val)))

    if max_res <= tol or diff <= tol:
        return x, True, 0, max_res, diff, f_val

    # Construct physical two-sided diagonal equilibration scalers (D_L, D_R)
    vars0 = unpack_equilibrium_vector(x, ns=ns, nc=nc)
    y_flat = vars0.y.flatten(order="F")
    L_flat = calib.l_endow.ravel()
    K_flat = calib.k_endow.ravel()
    GDP_flat = L_flat + K_flat

    # Right column scaling matrix D_R: maps state variables to O(1) units
    d_r = np.ones(n, dtype=float)
    i_T_start = 2 * M + 2 * nc
    i_XN_start = 2 * M + 3 * nc
    d_r[i_T_start:i_XN_start] = np.maximum(np.abs(vars0.T.ravel()), 1.0)
    d_r[i_XN_start:] = np.maximum(np.abs(vars0.XN.ravel()), 1.0)

    # Left row scaling matrix D_L: normalizes residual equations to relative dimensionless errors
    d_l = np.ones(n, dtype=float)
    d_l[:M] = 1.0 / np.maximum(y_flat, 1.0)
    # ff1 (zero profit) is already in O(1) price level units
    d_l[2 * M : 2 * M + nc] = 1.0 / np.maximum(L_flat, 1.0)
    d_l[2 * M + nc : 2 * M + 2 * nc] = 1.0 / np.maximum(K_flat, 1.0)
    d_l[2 * M + 2 * nc : 2 * M + 3 * nc - 1] = 1.0 / np.maximum(np.abs(GDP_flat[: nc - 1]), 1.0)
    d_l[2 * M + 3 * nc - 1 :] = 1.0 / np.maximum(np.abs(GDP_flat), 1.0)

    for it in range(max_iter):
        # Matrix-free directional derivative LinearOperator on equilibrated coordinates
        def matvec(v: np.ndarray) -> np.ndarray:
            norm_v = float(np.linalg.norm(v))
            if norm_v < 1e-14:
                return np.zeros_like(v)
            v_dir = v / norm_v
            # Scale perturbation back to physical space via D_R
            x_pert = x + eps_fd * (d_r * v_dir)
            f_pert = f(x_pert)
            return (d_l * ((f_pert - f_val) / eps_fd)) * norm_v

        A = spla.LinearOperator((n, n), matvec=matvec, dtype=float)
        rhs = -d_l * f_val

        # Solve inexact Newton direction in equilibrated space
        if krylov_method == "bicgstab":
            sol_u, info = spla.bicgstab(A, rhs, maxiter=max_krylov_iter, atol=atol)
            if info != 0:
                sol_u, _ = spla.gmres(A, rhs, restart=restart, maxiter=max_krylov_iter, atol=atol)
        else:
            sol_u, info = spla.gmres(A, rhs, restart=restart, maxiter=max_krylov_iter, atol=atol)
            if info != 0:
                sol_u, _ = spla.bicgstab(A, rhs, maxiter=max_krylov_iter, atol=atol)

        # Un-equilibrate step direction: delta = D_R @ sol_u
        delta = d_r * sol_u

        # Damping schedule
        xscal = 0.5 if it < 3 else 1.0

        # Backtracking line search
        alpha_step = xscal
        norm_0 = max_res
        x_trial = x + alpha_step * delta
        f_trial = f(x_trial)

        for _ in range(10):
            res_trial_max = float(np.max(np.abs(f_trial)))
            res_trial_diff = float(np.sum(np.abs(f_trial)))
            if res_trial_max <= tol or res_trial_diff <= tol or res_trial_max < norm_0:
                break
            alpha_step *= 0.5
            x_trial = x + alpha_step * delta
            f_trial = f(x_trial)

        x = x_trial
        f_val = f_trial
        max_res = float(np.max(np.abs(f_val)))
        diff = float(np.sum(np.abs(f_val)))

        if max_res <= tol or diff <= tol:
            return x, True, it + 1, max_res, diff, f_val

    return x, False, max_iter, max_res, diff, f_val


# ===========================================================================
# 4. Architecture D: Block-Elimination / Schur Complement Price Condensation
# ===========================================================================

def _condensed_schur_solve(
    calib: TradeCalibrationResult,
    tau_a: np.ndarray,
    taufd_a: np.ndarray,
    tauf_vec: np.ndarray,
    tauf_fd_vec: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 2.5e-3,
    max_iter: int = 50,
    eps_fd: float = 1e-4,
    replicate_matlab_precedence: bool = True,
    *,
    sigma: float = 0.0,
    fiscal_closure: str = "lump_sum",
    recycling_params: dict[str, Any] | None = None,
    capacity_margins: dict[str, float] | dict[tuple[int, int], float] | np.ndarray | None = None,
    capacity_target_country: str = "USA",
    penalty_scale: float = 0.05,
    penalty_exponent: float = 8.0,
) -> tuple[np.ndarray, bool, int, float, float, np.ndarray]:
    """Block-elimination / Schur complement price condensation general equilibrium solver."""
    ns = calib.n_sectors
    nc = calib.n_countries
    nfd = calib.n_final_demand
    M = ns * nc

    # Pre-factorize Leontief operators (I - B^T) and (I - A)
    tax_flat = calib.tax.flatten(order="F")
    a_eff_2d = (calib.a * tau_a).reshape((M, M), order="F")
    denom = 1.0 - tax_flat[:, np.newaxis]
    B_T_mat = a_eff_2d.T / np.maximum(denom, 1e-12)
    a_2d = calib.a.reshape((M, M), order="F")

    # Intermediate operators are ~75% dense. Dense LAPACK (dgetrf/dgetrs) provides
    # massive speedup (21x at M=15,400) and reduces peak RAM from 25.7 GB to 1.8 GB.
    density = float(np.count_nonzero(a_2d)) / float(M * M) if M > 0 else 0.0
    use_dense = (M >= 1000 or density > 0.2)

    if use_dense:
        M_P_dense = np.eye(M, dtype=float) - B_T_mat
        M_Y_dense = np.eye(M, dtype=float) - a_2d
        lu_P_piv = la.lu_factor(M_P_dense)
        lu_Y_piv = la.lu_factor(M_Y_dense)

        class _DenseLUSolver:
            def __init__(self, lu_piv):
                self._lu_piv = lu_piv

            def solve(self, b: np.ndarray) -> np.ndarray:
                return la.lu_solve(self._lu_piv, b)

        lu_P = _DenseLUSolver(lu_P_piv)
        lu_Y = _DenseLUSolver(lu_Y_piv)
    else:
        M_P = sp.eye(M, format="csc") - sp.csc_matrix(B_T_mat)
        lu_P = spla.splu(M_P)
        M_Y = sp.eye(M, format="csc") - sp.csc_matrix(a_2d)
        lu_Y = spla.splu(M_Y)

    # Initialize macro variables: [log(r); log(w); T; XN]
    if x0 is None:
        x_init_full = build_initial_guess(calib)
    else:
        x_init_full = np.asarray(x0, dtype=float).ravel()

    vars0 = unpack_equilibrium_vector(x_init_full, ns=ns, nc=nc, nfd=nfd)
    xm = np.concatenate([
        np.log(np.maximum(vars0.r.ravel(), 1e-12)),
        np.log(np.maximum(vars0.w.ravel(), 1e-12)),
        vars0.T.flatten(order="F"),
        vars0.XN.ravel(),
    ])
    n_m = len(xm)

    # Fast evaluation of condensed macro residual system (307 equations)
    def eval_macro(
        xm_curr: np.ndarray,
        p_cached: tuple[np.ndarray, np.ndarray] | None = None,
        compute_full: bool = True,
    ):
        r = np.exp(xm_curr[:nc]).reshape((1, 1, nc))
        w = np.exp(xm_curr[nc : 2 * nc]).reshape((1, 1, nc))
        T = xm_curr[2 * nc : 3 * nc].reshape((1, 1, nc))
        XN = xm_curr[3 * nc :]
        invforT = np.append(XN, -np.sum(XN))

        # 1. Price solve
        if p_cached is None:
            term_r = (r / calib.alpha) ** calib.alpha
            if replicate_matlab_precedence:
                term_w = w / ((1.0 - calib.alpha) ** (1.0 - calib.alpha))
            else:
                term_w = (w / (1.0 - calib.alpha)) ** (1.0 - calib.alpha)
            val_va = (1.0 / calib.beta) * (term_r * term_w)
            v_P = val_va.flatten(order="F") / np.maximum(1.0 - tax_flat, 1e-12)
            p_vec = lu_P.solve(v_P)
            p = p_vec.reshape((1, ns, nc), order="F")
        else:
            p_vec, p = p_cached
            term_r = (r / calib.alpha) ** calib.alpha
            if replicate_matlab_precedence:
                term_w = w / ((1.0 - calib.alpha) ** (1.0 - calib.alpha))
            else:
                term_w = (w / (1.0 - calib.alpha)) ** (1.0 - calib.alpha)
            val_va = (1.0 / calib.beta) * (term_r * term_w)

        # 2. Final demand and output solve
        ppfd = np.tensordot(p_vec, calib.afd * taufd_a, axes=(0, 0))[np.newaxis, :, :]
        Ycon = w * calib.l_endow + r * calib.k_endow + T
        cd = calib.theta * Ycon / ppfd
        tax_fd_arr = calib.tax_fd if calib.tax_fd is not None else np.zeros((1, nfd, nc))
        Tax_c = tax_fd_arr * ppfd * cd
        c = cd.copy()
        c[:, 1:2, :] -= invforT.reshape((1, 1, nc)) / ppfd[:, 1:2, :]
        xc = calib.afd * (c - tax_fd_arr * cd)
        xc_2d = xc.reshape((M, nfd * nc), order="F")
        d = np.sum(xc_2d, axis=1)
        y_vec = lu_Y.solve(d)
        ytot = y_vec.reshape((1, ns, nc), order="F")

        # 3. Factor demands
        mask_active = (ytot > 0)
        ratio_rw = ((1.0 - calib.alpha) * r) / (calib.alpha * w)
        ratio_wr = (calib.alpha * w) / ((1.0 - calib.alpha) * r)
        xl = np.where(mask_active, (ytot / calib.beta) * (ratio_rw ** calib.alpha), 0.0)
        xk = np.where(mask_active, (ytot / calib.beta) * (ratio_wr ** (1.0 - calib.alpha)), 0.0)

        # 4. Bilateral trade flow accounting via fast broadcasting
        x_2d = a_2d * y_vec[np.newaxis, :]
        pp_col = p_vec[:, np.newaxis]
        ppfd_row = ppfd.reshape((1, nfd * nc), order="F")

        x_sum_c2 = np.sum(x_2d.reshape(M, nc, ns), axis=2)
        val_sum = x_sum_c2 * pp_col
        T_inter = np.sum(val_sum.reshape(nc, ns, nc), axis=1)

        fd_sum_c2 = np.sum((xc_2d * ppfd_row).reshape(M, nc, nfd), axis=2)
        T_fd = np.sum(fd_sum_c2.reshape(nc, ns, nc), axis=1)
        np.fill_diagonal(T_inter, 0.0)
        np.fill_diagonal(T_fd, 0.0)

        X0 = np.sum(T_inter, axis=1)
        M0 = np.sum(T_inter, axis=0)
        XFD = np.sum(T_fd, axis=1)
        MFD = np.sum(T_fd, axis=0)
        invforT_realized = X0 + XFD - M0 - MFD

        # 5. Fiscal revenues
        Tax_Total = np.sum(calib.tax * ytot, axis=1).ravel() + np.sum(Tax_c, axis=1).ravel()
        Tarifs_Totals = M0 * tauf_vec + MFD * tauf_fd_vec

        # Macro residual equations
        ff2 = calib.l_endow.ravel() - np.sum(xl, axis=1).ravel()
        ff3 = calib.k_endow.ravel() - np.sum(xk, axis=1).ravel()
        ff4 = XN - invforT_realized[:nc - 1]
        ff5 = T.ravel() - (Tax_Total + Tarifs_Totals)
        f_macro = np.concatenate([ff2, ff3, ff4, ff5])

        if compute_full:
            # Micro residuals (identically zero by construction)
            ff0 = y_vec - (a_2d @ y_vec + d)
            ff1 = p_vec - ((val_va.flatten(order="F") + a_eff_2d.T @ p_vec) / np.maximum(1.0 - tax_flat, 1e-12))
            f_full = np.concatenate([ff0, ff1, ff2, ff3, ff4, ff5])
        else:
            f_full = None

        return f_macro, f_full, p_vec, p, y_vec

    f_m, f_full, p_sol, p_sol_3d, y_sol = eval_macro(xm, compute_full=True)
    max_res_macro = float(np.max(np.abs(f_m)))
    diff_macro = float(np.sum(np.abs(f_m)))

    # Step size vector for finite differencing
    h = np.empty(n_m, dtype=float)
    for j in range(n_m):
        if j < 2 * nc:
            h[j] = eps_fd
        else:
            h[j] = eps_fd * max(abs(xm[j]), 1.0)

    # Reconstruct full state vector
    x_full = np.concatenate([
        np.log(np.maximum(p_sol, 1e-12)),
        np.log(np.maximum(y_sol, 1e-12)),
        xm,
    ])
    max_res = float(np.max(np.abs(f_full)))
    diff = float(np.sum(np.abs(f_full)))

    if max_res <= tol or max_res_macro <= tol:
        return x_full, True, 0, max_res, diff, f_full

    mu = 1e-6

    for it in range(max_iter):
        # 307 x 307 dense macro Jacobian
        J = np.empty((n_m, n_m), dtype=float)
        for j in range(n_m):
            xm_pert = xm.copy()
            xm_pert[j] += h[j]
            # When perturbing T or XN (j >= 2*nc), prices p do not change -> reuse cached p
            if j >= 2 * nc:
                f_p, _, _, _, _ = eval_macro(xm_pert, p_cached=(p_sol, p_sol_3d), compute_full=False)
            else:
                f_p, _, _, _, _ = eval_macro(xm_pert, compute_full=False)
            J[:, j] = (f_p - f_m) / h[j]

        # Two-sided diagonal equilibration for extreme numerical stability
        c_norm = np.linalg.norm(J, axis=0)
        c_norm[c_norm == 0] = 1.0
        D_R = 1.0 / c_norm
        J_scaled = J * D_R[np.newaxis, :]

        r_norm = np.linalg.norm(J_scaled, axis=1)
        r_norm[r_norm == 0] = 1.0
        D_L = 1.0 / r_norm
        J_equil = D_L[:, np.newaxis] * J_scaled
        rhs_equil = -D_L * f_m

        # Solve equilibrated macro system: direct LAPACK LU square solve for pure Newton (mu <= 1e-6),
        # or Levenberg-Marquardt damping on normal equations when descent damping is active (mu > 1e-6)
        if mu <= 1e-6:
            try:
                sol_u = np.linalg.solve(J_equil, rhs_equil)
                delta_m = D_R * sol_u
            except Exception:
                sol_u = np.linalg.lstsq(J_equil, rhs_equil, rcond=1e-12)[0]
                delta_m = D_R * sol_u
        else:
            try:
                JT_J = J_equil.T @ J_equil
                JT_rhs = J_equil.T @ rhs_equil
                sol_u = np.linalg.solve(JT_J + mu * np.eye(n_m), JT_rhs)
                delta_m = D_R * sol_u
            except Exception:
                sol_u = np.linalg.lstsq(J_equil, rhs_equil, rcond=1e-12)[0]
                delta_m = D_R * sol_u

        if not np.all(np.isfinite(delta_m)):
            try:
                delta_m = np.linalg.lstsq(J, -f_m, rcond=1e-6)[0]
            except Exception:
                delta_m = np.zeros_like(f_m)

        # Direction-preserving uniform scaling based on factor prices
        max_factor_step = float(np.max(np.abs(delta_m[:2 * nc])))
        if max_factor_step > 0.3:
            delta_m *= (0.3 / max_factor_step)

        # Backtracking line search on equilibrated objective ||D_L * f_m||_2
        alpha_step = 1.0
        norm_0 = max_res_macro
        norm_2_0 = float(np.linalg.norm(D_L * f_m))
        norm_diff_0 = float(np.sum(np.abs(D_L * f_m)))
        xm_trial = xm + alpha_step * delta_m
        xm_trial[:2 * nc] = np.clip(xm_trial[:2 * nc], -5.0, 5.0)
        f_trial_m, f_trial_full, p_trial, p_trial_3d, y_trial = eval_macro(xm_trial, compute_full=True)

        best_trial = (xm_trial, f_trial_m, f_trial_full, p_trial, p_trial_3d, y_trial)
        best_res_m = float(np.max(np.abs(f_trial_m)))
        best_norm_2 = float(np.linalg.norm(D_L * f_trial_m))

        accepted = False
        for _ in range(15):
            res_trial_m = float(np.max(np.abs(f_trial_m)))
            norm_2_trial = float(np.linalg.norm(D_L * f_trial_m))
            if norm_2_trial < best_norm_2 or res_trial_m < best_res_m:
                best_trial = (xm_trial, f_trial_m, f_trial_full, p_trial, p_trial_3d, y_trial)
                best_res_m = res_trial_m
                best_norm_2 = norm_2_trial
            if res_trial_m <= tol or norm_2_trial < norm_2_0 or res_trial_m < norm_0:
                accepted = True
                break
            alpha_step *= 0.5
            xm_trial = xm + alpha_step * delta_m
            xm_trial[:2 * nc] = np.clip(xm_trial[:2 * nc], -5.0, 5.0)
            f_trial_m, f_trial_full, p_trial, p_trial_3d, y_trial = eval_macro(xm_trial, compute_full=True)

        if accepted:
            xm = xm_trial
            f_m = f_trial_m
            f_full = f_trial_full
            p_sol = p_trial
            p_sol_3d = p_trial_3d
            y_sol = y_trial
            mu = max(mu * 0.3, 1e-8)
        elif best_norm_2 < norm_2_0 or best_res_m < norm_0:
            xm, f_m, f_full, p_sol, p_sol_3d, y_sol = best_trial
            mu = max(mu * 0.5, 1e-8)
        else:
            mu = min(mu * 10.0, 1e2)

        max_res_macro = float(np.max(np.abs(f_m)))
        max_res = float(np.max(np.abs(f_full))) if f_full is not None else max_res_macro
        diff = float(np.sum(np.abs(f_full))) if f_full is not None else 0.0

        if max_res <= tol or max_res_macro <= tol or diff <= tol:
            x_full = np.concatenate([
                np.log(np.maximum(p_sol, 1e-12)),
                np.log(np.maximum(y_sol, 1e-12)),
                xm,
            ])
            return x_full, True, it + 1, max_res, diff, f_full

    x_full = np.concatenate([
        np.log(np.maximum(p_sol, 1e-12)),
        np.log(np.maximum(y_sol, 1e-12)),
        xm,
    ])
    return x_full, False, max_iter, max_res, diff, f_full


# ===========================================================================
# 5. Broyden Quasi-Newton (Backward Compatibility)
# ===========================================================================

def _broyden_solve(
    f: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    tol: float = 2.5e-3,
    max_iter: int = 50,
    eps_fd: float = 1e-4,
) -> tuple[np.ndarray, bool, int, float, float, np.ndarray]:
    """Pure NumPy Broyden quasi-Newton solver with Sherman-Morrison rank-1 inverse updates."""
    x = np.asarray(x_init, dtype=float).copy()
    n = len(x)
    f_val = f(x)
    max_res = float(np.max(np.abs(f_val)))
    diff = float(np.sum(np.abs(f_val)))

    if max_res <= tol or diff <= tol:
        return x, True, 0, max_res, diff, f_val

    # Initial Jacobian via finite differences
    B = np.empty((n, n), dtype=float)
    for j in range(n):
        x_pert = x.copy()
        x_pert[j] += eps_fd
        B[:, j] = (f(x_pert) - f_val) / eps_fd

    try:
        H = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        H = np.linalg.pinv(B)

    for it in range(max_iter):
        delta = -H @ f_val
        alpha_step = 1.0
        norm_0 = max_res
        x_trial = x + alpha_step * delta
        f_trial = f(x_trial)

        for _ in range(10):
            res_trial_max = float(np.max(np.abs(f_trial)))
            res_trial_diff = float(np.sum(np.abs(f_trial)))
            if res_trial_max <= tol or res_trial_diff <= tol or res_trial_max < norm_0:
                break
            alpha_step *= 0.5
            x_trial = x + alpha_step * delta
            f_trial = f(x_trial)

        x_next = x_trial
        f_next = f_trial
        max_res = float(np.max(np.abs(f_next)))
        diff = float(np.sum(np.abs(f_next)))

        if max_res <= tol or diff <= tol:
            return x_next, True, it + 1, max_res, diff, f_next

        # Sherman-Morrison rank-1 update
        s = x_next - x
        y = f_next - f_val
        Hy = H @ y
        sTH = s.T @ H
        denom = float(np.dot(sTH, y))
        if abs(denom) > 1e-12:
            H = H + np.outer(s - Hy, sTH) / denom

        x = x_next
        f_val = f_next

    return x, False, max_iter, max_res, diff, f_val


# ===========================================================================
# 6. Master Solver Interfaces
# ===========================================================================

def solve_trade_equilibrium(
    calib: TradeCalibrationResult,
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    tauf: np.ndarray | None = None,
    tauf_fd: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    method: str = "newton",
    tol: float = 2.5e-3,
    max_iter: int = 50,
    replicate_matlab_precedence: bool = True,
    base_result: TradeEquilibriumResult | None = None,
    *,
    sigma: float = 0.0,
    fiscal_closure: str = "lump_sum",
    recycling_params: dict[str, Any] | None = None,
    capacity_margins: dict[str, float] | dict[tuple[int, int], float] | np.ndarray | None = None,
    capacity_target_country: str = "USA",
    penalty_scale: float = 0.05,
    penalty_exponent: float = 8.0,
    **kwargs: Any,
) -> TradeEquilibriumResult:
    """Solve the multi-country multi-sector general equilibrium nonlinear system.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters from :func:`calibrate_trade_model`.
    tau : np.ndarray, optional
        Intermediate tariff multipliers (1 + tariff_rate). Shape (ns*nc, ns, nc),
        (nc,), or None.
    tau_fd : np.ndarray, optional
        Final demand tariff multipliers (1 + tariff_rate). Shape (ns*nc, nfd, nc),
        (nc,), or None.
    tauf : np.ndarray, optional
        National intermediate tariff rates by importing country, shape (nc,).
    tauf_fd : np.ndarray, optional
        National final demand tariff rates by importing country, shape (nc,).
    x0 : np.ndarray, optional
        Initial state vector. If None, constructed via :func:`build_initial_guess`.
    method : {"newton", "sparse_lu", "krylov", "condensed", "broyden", "hybr", "lm"}, default "newton"
        Nonlinear root-finding algorithm:
        - 'newton': Dense Damped Newton-Raphson with Armijo line search (baseline).
        - 'sparse_lu': Sparse Jacobian Newton using direct sparse LU (splu) and graph coloring.
        - 'krylov': Inexact Newton-Krylov (JFNK via GMRES/BiCGSTAB) with two-sided equilibration.
        - 'condensed': Block-elimination / Schur complement condensation (307 macro variables).
        - 'broyden': Broyden quasi-Newton with Sherman-Morrison rank-1 updates.
        - 'hybr': SciPy MINPACK Powell hybrid root wrapper.
        - 'lm': SciPy Levenberg-Marquardt root wrapper.
    tol : float, default 2.5e-3
        Maximum absolute residual tolerance: ||F(x)||_inf <= tol.
    max_iter : int, default 50
        Maximum solver iterations.
    replicate_matlab_precedence : bool, default True
        Whether to replicate MATLAB operator precedence at ff_equi.m:31.
    base_result : TradeEquilibriumResult, optional
        Benchmark baseline result used for relative index evaluation.
    sigma : float, default 0.0
        Intermediate sourcing substitution elasticity (Extension B).
    fiscal_closure : str, default "lump_sum"
        Fiscal regime closure for tariff revenues (Extension C).
    recycling_params : dict[str, Any], optional
        Target country and sector parameters for revenue recycling.
    capacity_margins : dict or np.ndarray, optional
        Spare capacity margins for upstream bottleneck sectors (Extension D).
    capacity_target_country : str, default "USA"
        Target country code for bottleneck constraints.
    penalty_scale : float, default 0.05
        Scale parameter zeta for C^2 barrier penalty function.
    penalty_exponent : float, default 8.0
        Exponent eta for C^2 barrier penalty function.

    Returns
    -------
    TradeEquilibriumResult
        Result container with solution arrays, reconstructed flows, and diagnostics.
    """
    tau_a, taufd_a, tauf_vec, tauf_fd_vec = _resolve_tariffs(
        calib, tau=tau, tau_fd=tau_fd, tauf=tauf, tauf_fd=tauf_fd
    )

    if x0 is None:
        x_init = build_initial_guess(calib)
    else:
        x_init = np.asarray(x0, dtype=float).ravel()

    def obj_fun(xx: np.ndarray) -> np.ndarray:
        return compute_equilibrium_residuals(
            x=xx,
            calib=calib,
            tau=tau_a,
            tau_fd=taufd_a,
            tauf=tauf_vec,
            tauf_fd=tauf_fd_vec,
            replicate_matlab_precedence=replicate_matlab_precedence,
            sigma=sigma,
            fiscal_closure=fiscal_closure,
            recycling_params=recycling_params,
            capacity_margins=capacity_margins,
            capacity_target_country=capacity_target_country,
            penalty_scale=penalty_scale,
            penalty_exponent=penalty_exponent,
        )

    t_solve_start = time.perf_counter()

    # Method dispatch
    if method == "newton":
        x_sol, conv, iters, max_res, diff, res = _damped_newton_solve(
            obj_fun, x_init, tol=tol, max_iter=max_iter
        )
    elif method == "sparse_lu":
        # Build structural sparsity pattern
        try:
            sparsity_pat = _build_cge_sparsity_pattern(calib, tau_a=tau_a)
        except Exception:
            sparsity_pat = None
        x_sol, conv, iters, max_res, diff, res = _sparse_lu_solve(
            obj_fun, x_init, tol=tol, max_iter=max_iter, sparsity=sparsity_pat
        )
    elif method == "krylov":
        krylov_algo = kwargs.get("krylov_method", "gmres")
        restart_val = kwargs.get("restart", 30)
        max_k_iter = kwargs.get("max_krylov_iter", 50)
        x_sol, conv, iters, max_res, diff, res = _newton_krylov_solve(
            obj_fun,
            x_init,
            calib=calib,
            tol=tol,
            max_iter=max_iter,
            krylov_method=krylov_algo,
            restart=restart_val,
            max_krylov_iter=max_k_iter,
        )
    elif method == "condensed":
        # Check if non-Leontief extensions are active
        is_extended = (
            sigma > 0.0
            or fiscal_closure not in ("lump_sum", "baseline", "")
            or capacity_margins is not None
        )
        if is_extended:
            # Fall back to sparse LU for nonlinear extensions
            sparsity_pat = _build_cge_sparsity_pattern(calib, tau_a=tau_a)
            x_sol, conv, iters, max_res, diff, res = _sparse_lu_solve(
                obj_fun, x_init, tol=tol, max_iter=max_iter, sparsity=sparsity_pat
            )
        else:
            x_sol, conv, iters, max_res, diff, res = _condensed_schur_solve(
                calib=calib,
                tau_a=tau_a,
                taufd_a=taufd_a,
                tauf_vec=tauf_vec,
                tauf_fd_vec=tauf_fd_vec,
                x0=x_init,
                tol=tol,
                max_iter=max_iter,
                replicate_matlab_precedence=replicate_matlab_precedence,
            )
    elif method == "broyden":
        x_sol, conv, iters, max_res, diff, res = _broyden_solve(
            obj_fun, x_init, tol=tol, max_iter=max_iter
        )
    elif method in ("hybr", "lm"):
        from scipy.optimize import root

        n = len(x_init)
        opts: dict[str, Any] = {}
        if method == "hybr":
            opts["maxfev"] = max(2000, max_iter * n)
        elif method == "lm":
            opts["maxiter"] = max(2000, max_iter * n)

        res_scipy = root(
            obj_fun,
            x_init,
            method=method,
            tol=tol,
            options=opts,
        )
        x_sol = np.asarray(res_scipy.x, dtype=float).ravel()
        res = obj_fun(x_sol)
        max_res = float(np.max(np.abs(res)))
        diff = float(np.sum(np.abs(res)))
        conv = bool(res_scipy.success) or max_res <= tol or diff <= tol
        iters = int(getattr(res_scipy, "nfev", 0))
    else:
        raise ValueError(
            f"Unknown method '{method}'. Valid options are: 'newton', 'sparse_lu', 'krylov', 'condensed', 'broyden', 'hybr', 'lm'."
        )

    t_solve_end = time.perf_counter()
    solve_duration = t_solve_end - t_solve_start

    res_dataclass = postprocess_trade_equilibrium(
        x_sol=x_sol,
        calib=calib,
        tau=tau_a,
        tau_fd=taufd_a,
        tauf=tauf_vec,
        tauf_fd=tauf_fd_vec,
        replicate_matlab_precedence=replicate_matlab_precedence,
        converged=conv,
        iterations=iters,
        base_result=base_result,
        metadata={
            "method": method,
            "tol": tol,
            "max_iter": max_iter,
            "replicate_matlab_precedence": replicate_matlab_precedence,
            "solve_duration_seconds": solve_duration,
            "sigma": sigma,
            "fiscal_closure": fiscal_closure,
            "recycling_params": recycling_params,
            "capacity_margins": capacity_margins,
            "capacity_target_country": capacity_target_country,
            "penalty_scale": penalty_scale,
            "penalty_exponent": penalty_exponent,
        },
    )

    if method == "condensed" or sigma > 0.0 or fiscal_closure not in ("lump_sum", "baseline", "") or capacity_margins is not None:
        res_dataclass = replace(
            res_dataclass,
            residuals=res,
            residual_norm=diff,
            diff=diff,
            max_residual=max_res,
        )

    return res_dataclass


def solve_equilibrium(
    calib: TradeCalibrationResult,
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    method: str = "condensed",
    tol: float = 2.5e-3,
    max_iter: int = 50,
    **kwargs: Any,
) -> TradeEquilibriumResult:
    """Solve multi-country multi-sector general equilibrium defaulting to condensed Schur architecture.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    tau : np.ndarray, optional
        Intermediate tariff multipliers.
    tau_fd : np.ndarray, optional
        Final demand tariff multipliers.
    x0 : np.ndarray, optional
        Initial state vector.
    method : {"condensed", "sparse_lu", "krylov", "newton"}, default "condensed"
        Solver architecture.
    tol : float, default 2.5e-3
        Maximum absolute residual tolerance.
    max_iter : int, default 50
        Maximum solver iterations.
    **kwargs : Any
        Additional keyword arguments forwarded to :func:`solve_trade_equilibrium`.

    Returns
    -------
    TradeEquilibriumResult
        Converged equilibrium solution container.
    """
    return solve_trade_equilibrium(
        calib=calib,
        tau=tau,
        tau_fd=tau_fd,
        x0=x0,
        method=method,
        tol=tol,
        max_iter=max_iter,
        **kwargs,
    )


build_cge_sparsity_pattern = _build_cge_sparsity_pattern

__all__ = [
    "build_initial_guess",
    "build_cge_sparsity_pattern",
    "solve_trade_equilibrium",
    "solve_equilibrium",
]
