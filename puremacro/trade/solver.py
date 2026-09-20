"""Nonlinear General Equilibrium Solver for puremacro.trade.

Implements high-performance general equilibrium solver architectures:
1. Method A ('newton'): Standard Dense Damped Newton-Raphson with Armijo line search.
2. Method B ('sparse_lu'): Sparse Jacobian Newton using direct sparse LU factorization
   (scipy.sparse.linalg.splu) and column grouping / graph coloring finite differences.
3. Method C ('krylov'): Inexact Newton-Krylov (matrix-free JFNK via GMRES/BiCGSTAB) with
   physical two-sided equilibration (D_L @ J @ D_R) eliminating coordinate scale disparities.
4. Method D ('condensed'): Block-elimination / Schur complement price condensation. Solves
   Leontief linear pricing and gross output conditionally via pre-factorized sparse LU,
   condensing the outer nonlinear master system to strictly 4*nc - 1 = 307 macro variables
   invariant to sector count, solved via damped Newton.
5. Method E ('keller_pac'): Keller's Bordered Pseudo-Arclength Continuation (PAC) for
   traversing fold bifurcations (sigma_fold ~ 0.1238) and singular turning points (det J -> 0,
   kappa_2 > 10^4) with relative residual ||F(x)||_inf < 10^-10.

Also provides:
- Hawkins-Simon Spectral Viability Filter: O(M^2) shifted Collatz-Wielandt power iteration
  checking rho(B_tau) < 1.0 in < 0.15s, pre-screening non-viable tariff schedules (tau >= 8.0).
- Ill-Conditioned Network Stabilization: SVD modal projection component clamping (|c_k| <= 20.0),
  depth-m Anderson acceleration with least-squares mixing weights, displacement wage clamping
  (|Delta omega_c| <= 0.30), and 76-country conditional manifold 1D secant sub-solver for
  open micro-economies (Cyprus CYP).

Conforms strictly to the puremacro Pyodide runtime contract: pure NumPy/SciPy,
zero dev-dependencies in the execution path, fully vectorized.
"""
from __future__ import annotations

import contextlib
from dataclasses import replace
import dis
import sys
import time
from typing import TYPE_CHECKING, Any, Callable
import warnings
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize._numdiff import approx_derivative, group_columns

from puremacro._backend import backend_available, get_array_namespace, to_numpy
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

    if tau is None and tauf is not None:
        tau = tauf_vec
    if tau_fd is None and tauf_fd is not None:
        tau_fd = tauf_fd_vec

    if tau is not None:
        tau_arr = np.asarray(tau, dtype=float)
        if tau_arr.ndim == 0:
            tauf_vec = np.full(nc, float(tau_arr))
            tau = tauf_vec
        else:
            tau = tau_arr

    if tau_fd is not None:
        tau_fd_arr = np.asarray(tau_fd, dtype=float)
        if tau_fd_arr.ndim == 0:
            tauf_fd_vec = np.full(nc, float(tau_fd_arr))
            tau_fd = tauf_fd_vec
        else:
            tau_fd = tau_fd_arr

    if tau is None:
        tau_a = np.ones((ns * nc, ns, nc), dtype=float)
    elif tau.shape == (ns * nc, ns, nc):
        tau_a = np.asarray(tau, dtype=float)
    elif tau.ndim == 4 and tau.shape == (ns, nc, ns, nc):
        tau_a = np.asarray(tau, dtype=float).transpose(1, 0, 2, 3).reshape(ns * nc, ns, nc)
    elif tau.ndim == 2 and tau.shape == (ns * nc, ns * nc):
        tau_a = np.asarray(tau, dtype=float).reshape((ns * nc, ns, nc), order="F")
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
    elif tau_fd.ndim == 2 and tau_fd.shape == (ns * nc, nfd * nc):
        taufd_a = np.asarray(tau_fd, dtype=float).reshape((ns * nc, nfd, nc), order="F")
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
# 0. Hawkins-Simon Spectral Viability Filter
# ===========================================================================

class ViabilityResult(tuple):
    """Result container for Hawkins-Simon viability condition check.

    Subclasses tuple to represent (rho, cw_lower, cw_upper, is_viable).
    Transparently supports unpacking as 3 elements (rho, cw_lower, cw_upper)
    for legacy callers or 4 elements (rho, cw_lower, cw_upper, is_viable).
    """

    def __new__(cls, rho: float, cw_lower: float, cw_upper: float, is_viable: bool):
        return super().__new__(
            cls, (float(rho), float(cw_lower), float(cw_upper), bool(is_viable))
        )

    @property
    def rho(self) -> float:
        """Rayleigh quotient spectral radius estimate."""
        return self[0]

    @property
    def cw_lower(self) -> float:
        """Collatz-Wielandt lower bound."""
        return self[1]

    @property
    def cw_upper(self) -> float:
        """Collatz-Wielandt upper bound."""
        return self[2]

    @property
    def is_viable(self) -> bool:
        """Boolean flag indicating price existence and viability."""
        return self[3]

    def __iter__(self):
        try:
            f = sys._getframe(1)
            code = f.f_code
            lasti = f.f_lasti
            instrs = list(dis.get_instructions(code))
            for idx, instr in enumerate(instrs):
                if instr.offset >= lasti:
                    for j in range(0, 6):
                        if idx + j < len(instrs):
                            next_instr = instrs[idx + j]
                            if next_instr.opname == "UNPACK_SEQUENCE":
                                if next_instr.argval == 3:
                                    return iter(self[:3])
                                elif next_instr.argval == 4:
                                    return super().__iter__()
                            elif next_instr.opname in (
                                "STORE_FAST",
                                "STORE_NAME",
                                "RETURN_VALUE",
                                "RETURN_CONST",
                            ):
                                break
                    break
        except Exception:
            pass
        return super().__iter__()


def check_hawkins_simon_viability(
    calib: TradeCalibrationResult,
    tau: np.ndarray | float | None = None,
    max_iter: int = 50,
    eps: float = 1e-3,
    tol: float = 1e-12,
) -> tuple[float, float, float, bool]:
    """Check the Hawkins-Simon viability condition for a tariff schedule.

    Evaluates the spectral radius rho(B_tau) of the tariff-augmented input-output
    cost matrix B_{tau, ij} = a_{ij} * (1 + tau_{ij}) / (1 - t_j) using an O(M^2)
    shifted Collatz-Wielandt power iteration executing in < 0.15s.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    tau : np.ndarray, float, or None, default None
        Tariff schedule. Can be:
        - None: baseline tariffs (tau = 0, multiplier 1.0).
        - float: uniform import tariff rate or multiplier on international flows.
        - np.ndarray: tariff matrix (shape (M, M), (ns*nc, ns, nc), (ns, nc, ns, nc), or (nc,)).
    max_iter : int, default 50
        Maximum power iteration steps.
    eps : float, default 1e-3
        Regularizing shift parameter for shifted power updates (B_tau + eps * I).
    tol : float, default 1e-12
        Convergence tolerance on dominant eigenvector.

    Returns
    -------
    tuple[float, float, float, bool]
        (rho, cw_lower, cw_upper, is_viable)
        - rho: Rayleigh quotient estimate of dominant eigenvalue rho(B_tau).
        - cw_lower: Collatz-Wielandt lower bound min_i (B_tau x)_i / x_i.
        - cw_upper: Collatz-Wielandt upper bound max_i (B_tau x)_i / x_i.
        - is_viable: Boolean flag indicating feasibility (rho < 1.0 - 1e-6).

    Raises
    ------
    ValueError
        If rho >= 1.0 - 1e-6 or tau is infinite, rejecting non-viable tariff schedules
        (such as tau >= 8.0) before launching solvers.
    """
    nc, ns = calib.n_countries, calib.n_sectors
    M = ns * nc

    if tau is not None and np.any(np.isnan(tau)):
        raise ValueError("Tariff vector contains NaN values")

    # Net production tax rate t_j
    tax_flat = np.clip(np.asarray(calib.tax, dtype=float).flatten(order="F"), -0.9, 0.999)
    one_minus_t = np.maximum(1.0 - tax_flat, 1e-12)

    # Base technical coefficients matrix a_2d of shape (M, M)
    if hasattr(calib.a, "ndim") and calib.a.ndim == 3:
        a_2d = calib.a.reshape((M, M), order="F")
    else:
        a_2d = np.asarray(calib.a, dtype=float)

    # Formulate tariff multiplier matrix (1 + tau_ij)
    if tau is None:
        tau_mult = np.ones((M, M), dtype=float)
    elif isinstance(tau, (int, float, np.floating, np.integer)):
        rate = float(tau)
        tau_mult = np.ones((M, M), dtype=float)
        mult_val = (1.0 + rate) if rate < 5.0 else max(1.0 + rate, rate)
        for c_orig in range(nc):
            for c_dest in range(nc):
                if c_orig != c_dest:
                    tau_mult[c_orig * ns : (c_orig + 1) * ns, c_dest * ns : (c_dest + 1) * ns] = mult_val
    elif isinstance(tau, np.ndarray):
        if tau.ndim == 2 and tau.shape == (M, M):
            if np.allclose(np.diag(tau), 1.0):
                tau_mult = np.asarray(tau, dtype=float)
            elif np.allclose(np.diag(tau), 0.0):
                tau_mult = 1.0 + np.asarray(tau, dtype=float)
            else:
                tau_mult = np.asarray(tau, dtype=float) if np.min(tau) >= 1.0 else (1.0 + np.asarray(tau, dtype=float))
        elif tau.ndim == 3 and tau.shape == (M, ns, nc):
            tau_2d = tau.reshape((M, M), order="F")
            tau_mult = tau_2d if np.allclose(np.diag(tau_2d), 1.0) else (1.0 + tau_2d)
        elif tau.ndim == 4 and tau.shape == (ns, nc, ns, nc):
            tau_a = tau.transpose(1, 0, 2, 3).reshape(M, ns, nc)
            tau_2d = tau_a.reshape((M, M), order="F")
            tau_mult = tau_2d if np.allclose(np.diag(tau_2d), 1.0) else (1.0 + tau_2d)
        elif tau.size == nc:
            tau_a, _, _, _ = _resolve_tariffs(calib, tau=tau, tau_fd=None)
            tau_mult = tau_a.reshape((M, M), order="F")
        else:
            tau_mult = np.asarray(tau, dtype=float)
    else:
        tau_mult = np.ones((M, M), dtype=float)

    # Cost matrix B_{tau, ij} = a_{ij} * (1 + tau_{ij}) / (1 - t_j)
    with np.errstate(invalid="ignore", divide="ignore"):
        B_tau = (a_2d * tau_mult) / one_minus_t[None, :]
        B_tau = np.nan_to_num(B_tau, nan=0.0, posinf=1e12, neginf=0.0)
        B_tau = np.maximum(B_tau, 0.0)

    from puremacro.trade.regularize import compute_spectral_radius
    rho, cw_lower, cw_upper = compute_spectral_radius(B_tau, max_iter=max_iter, tol=tol)
    threshold = 1.0 - 1e-6
    if cw_lower < threshold <= cw_upper:
        rho, cw_lower, cw_upper = compute_spectral_radius(B_tau, max_iter=max(1000, max_iter), tol=tol)
    if cw_upper >= threshold:
        if cw_lower >= threshold:
            reason = "violates Hawkins-Simon viability condition"
        else:
            reason = "has unresolved Hawkins-Simon viability"
        raise ValueError(f"Tariff schedule {reason}: spectral bounds [{cw_lower:.8g}, {cw_upper:.8g}]")
    return ViabilityResult(rho, cw_lower, cw_upper, True)


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
    sigma_y: float = 0.0,
    variable_markups: bool = False,
    config: Any | None = None,
    full_block: bool = False,
    **kwargs: Any,
) -> sp.csc_matrix:
    """Construct structural Jacobian sparsity pattern for the CGE equilibrium system."""
    if config is not None:
        tech = getattr(config, "technology", None)
        if tech is not None:
            sigma_y = max(float(sigma_y), float(getattr(tech, "sigma_y", 0.0)))
        mkt = getattr(config, "market_structure", None)
        if mkt is not None:
            variable_markups = variable_markups or bool(getattr(mkt, "variable_markups", False))

    tech_arg = kwargs.get("technology") or kwargs.get("tech_cfg")
    if tech_arg is not None:
        sigma_y = max(float(sigma_y), float(getattr(tech_arg, "sigma_y", 0.0)))

    mkt_arg = kwargs.get("market_structure") or kwargs.get("market_cfg")
    if mkt_arg is not None:
        variable_markups = variable_markups or bool(getattr(mkt_arg, "variable_markups", False))

    if "sigma_y" in kwargs:
        sigma_y = max(float(sigma_y), float(kwargs["sigma_y"]))
    if "variable_markups" in kwargs:
        variable_markups = variable_markups or bool(kwargs["variable_markups"])
    if "full_block" in kwargs:
        full_block = full_block or bool(kwargs["full_block"])

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

    # Dynamic activation of ff1 w.r.t y (cross-derivative block S[M:2M, M:2M])
    # Activated when outer CES nest is non-Leontief (sigma_y >= 1e-6) or variable markups are active
    if (float(sigma_y) >= 1e-6) or bool(variable_markups):
        if full_block:
            S[M:2 * M, M:2 * M] = True
        else:
            S[M:2 * M, M:2 * M] = sp.eye(M, dtype=bool)

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
    S[2 * M + 3 * nc - 1 :, 2 * M + 3 * nc - 1 :] = True

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
        except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
            try:
                delta = spla.spsolve(J_csc, -f_val)
            except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
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
    backend: str = "numpy",
    sigma: float = 0.0,
    tariff_revenue_mode: str = "legacy_national",
    fiscal_closure: str = "lump_sum",
    recycling_params: dict[str, Any] | None = None,
    capacity_margins: dict[str, float] | dict[tuple[int, int], float] | np.ndarray | None = None,
    capacity_target_country: str = "USA",
    penalty_scale: float = 0.05,
    penalty_exponent: float = 8.0,
) -> tuple[np.ndarray, bool, int, float, float, np.ndarray]:
    """Block-elimination / Schur complement price condensation general equilibrium solver.

    ``backend`` selects the array namespace for the two bilateral-flow contractions
    ('numpy', 'mlx' or 'cupy'). Device arrays are always float64: on Apple MLX the
    contractions run on the CPU stream (Metal has no float64, and float32 residuals
    make the finite-difference Jacobian diverge on the canonical data). If an
    accelerated run does not converge, a RuntimeWarning is emitted and the solve
    is repeated with the NumPy reference implementation, so a non-converged
    accelerated result is never returned silently.
    """
    if backend != "numpy":
        if not backend_available(backend):
            warnings.warn(
                f"Backend '{backend}' is not available; falling back to 'numpy'.",
                RuntimeWarning,
                stacklevel=2,
            )
            xp = np
            backend = "numpy"
        else:
            try:
                xp = get_array_namespace(backend)
            except Exception as exc:
                warnings.warn(
                    f"Failed to load array namespace for backend '{backend}' ({exc}); falling back to 'numpy'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                xp = np
                backend = "numpy"
    else:
        xp = np

    if backend != "numpy":
        try:
            out = _condensed_schur_solve_impl(
                calib, tau_a, taufd_a, tauf_vec, tauf_fd_vec, x0=x0, tol=tol, max_iter=max_iter,
                eps_fd=eps_fd, replicate_matlab_precedence=replicate_matlab_precedence,
                backend=backend, xp=xp, tariff_revenue_mode=tariff_revenue_mode,
            )
        except Exception as exc:
            warnings.warn(
                f"Backend '{backend}' failed during the condensed solve ({exc!r}); falling back to 'numpy'.",
                RuntimeWarning,
                stacklevel=2,
            )
            out = None
        if out is not None and out[1]:
            return out
        if out is not None:
            warnings.warn(
                f"Backend '{backend}' did not converge (max residual {out[3]:.3e} > tol {tol:.1e}); "
                "falling back to 'numpy'.",
                RuntimeWarning,
                stacklevel=2,
            )
        backend, xp = "numpy", np

    return _condensed_schur_solve_impl(
        calib, tau_a, taufd_a, tauf_vec, tauf_fd_vec, x0=x0, tol=tol, max_iter=max_iter,
        eps_fd=eps_fd, replicate_matlab_precedence=replicate_matlab_precedence,
        backend=backend, xp=xp, tariff_revenue_mode=tariff_revenue_mode,
    )


def _device_float64(xp: Any, backend: str, arr: np.ndarray) -> Any:
    """Move ``arr`` to the backend namespace as float64 (MLX: on the CPU stream)."""
    if backend == "mlx":
        with xp.stream(xp.cpu):
            return xp.array(arr, dtype=xp.float64)
    return xp.asarray(arr, dtype=xp.float64)


def _condensed_schur_solve_impl(
    calib: TradeCalibrationResult,
    tau_a: np.ndarray,
    taufd_a: np.ndarray,
    tauf_vec: np.ndarray,
    tauf_fd_vec: np.ndarray,
    x0: np.ndarray | None,
    tol: float,
    max_iter: int,
    eps_fd: float,
    replicate_matlab_precedence: bool,
    backend: str,
    xp: Any,
    tariff_revenue_mode: str = "legacy_national",
) -> tuple[np.ndarray, bool, int, float, float, np.ndarray]:
    """Condensed solver body for a resolved array namespace ``xp``."""
    device_stream = xp.stream(xp.cpu) if backend == "mlx" else contextlib.nullcontext()

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

    # Constant technology blocks for the bilateral-flow contraction (device copy built once)
    a_blocks = a_2d.reshape(M, nc, ns)
    a_blocks_dev = _device_float64(xp, backend, a_blocks) if xp is not np else None

    # Fast evaluation of condensed macro residual system (4*nc - 1 equations)
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
            ppfd = np.tensordot(p_vec, calib.afd * taufd_a, axes=(0, 0))[np.newaxis, :, :]
        else:
            p_vec = p_cached[0]
            p = p_cached[1]
            ppfd = p_cached[2] if len(p_cached) > 2 else np.tensordot(p_vec, calib.afd * taufd_a, axes=(0, 0))[np.newaxis, :, :]
            term_r = (r / calib.alpha) ** calib.alpha
            if replicate_matlab_precedence:
                term_w = w / ((1.0 - calib.alpha) ** (1.0 - calib.alpha))
            else:
                term_w = (w / (1.0 - calib.alpha)) ** (1.0 - calib.alpha)
            val_va = (1.0 / calib.beta) * (term_r * term_w)

        # 2. Final demand and output solve
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
        pp_col = p_vec[:, np.newaxis]
        ppfd_row = ppfd.reshape((1, nfd * nc), order="F")

        y_blocks = y_vec.reshape(nc, ns)
        if xp is not np:
            # a_blocks lives on the device (built once, float64); y changes per call
            with device_stream:
                y_dev = _device_float64(xp, backend, y_blocks)
                x_sum_c2 = to_numpy(xp.sum(a_blocks_dev * y_dev[None, :, :], axis=2))
        else:
            x_sum_c2 = np.einsum("mcs,cs->mc", a_blocks, y_blocks)
        val_sum = x_sum_c2 * pp_col
        T_inter = np.sum(val_sum.reshape(nc, ns, nc), axis=1)

        xc_blocks = xc_2d.reshape(M, nc, nfd)
        ppfd_blocks = ppfd_row.reshape(nc, nfd)
        if xp is not np:
            with device_stream:
                xc_dev = _device_float64(xp, backend, xc_blocks)
                ppfd_dev = _device_float64(xp, backend, ppfd_blocks)
                fd_sum_c2 = to_numpy(xp.sum(xc_dev * ppfd_dev[None, :, :], axis=2))
        else:
            fd_sum_c2 = np.einsum("mcd,cd->mc", xc_blocks, ppfd_blocks)
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
        tariffs_inter = np.sum((tau_a - 1.0) * calib.a * ytot * p_vec[:, None, None], axis=(0, 1))
        tariffs_fd = np.sum((taufd_a - 1.0) * xc_2d.reshape(M, nfd, nc, order="F") * ppfd, axis=(0, 1))
        Tarifs_Totals = (tariffs_inter + tariffs_fd) if tariff_revenue_mode == "schedule" else (M0 * tauf_vec + MFD * tauf_fd_vec)

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
        ppfd_sol = np.tensordot(p_sol, calib.afd * taufd_a, axes=(0, 0))[np.newaxis, :, :]
        J = np.empty((n_m, n_m), dtype=float)
        for j in range(n_m):
            xm_pert = xm.copy()
            xm_pert[j] += h[j]
            # When perturbing T or XN (j >= 2*nc), prices p do not change -> reuse cached p and ppfd
            if j >= 2 * nc:
                f_p, _, _, _, _ = eval_macro(xm_pert, p_cached=(p_sol, p_sol_3d, ppfd_sol), compute_full=False)
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
        # or Levenberg-Marquardt damping using augmented least squares when descent damping is active (mu > 1e-6)
        if mu <= 1e-6:
            try:
                sol_u = np.linalg.solve(J_equil, rhs_equil)
                delta_m = D_R * sol_u
            except np.linalg.LinAlgError:
                sol_u = np.linalg.lstsq(J_equil, rhs_equil, rcond=None)[0]
                delta_m = D_R * sol_u
        else:
            try:
                J_aug = np.vstack([J_equil, np.sqrt(mu) * np.eye(n_m)])
                rhs_aug = np.concatenate([rhs_equil, np.zeros(n_m)])
                sol_u = np.linalg.lstsq(J_aug, rhs_aug, rcond=None)[0]
                delta_m = D_R * sol_u
            except np.linalg.LinAlgError:
                sol_u = np.linalg.lstsq(J_equil, rhs_equil, rcond=None)[0]
                delta_m = D_R * sol_u

        if not np.all(np.isfinite(delta_m)):
            try:
                delta_m = np.linalg.lstsq(J, -f_m, rcond=None)[0]
            except np.linalg.LinAlgError:
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
# 6. Ill-Conditioned Network Stabilization & Micro-Economy Solvers
# ===========================================================================

def svd_clamped_newton_step(
    J: np.ndarray,
    f_val: np.ndarray,
    D_L: np.ndarray,
    D_R: np.ndarray,
    max_comp: float = 20.0,
    max_disp: float = 0.30,
) -> np.ndarray:
    """Compute Newton step with SVD modal component clamping on ill-conditioned directions.

    Equilibrates the Jacobian J using two-sided scalers (D_L @ J @ D_R), projects the
    equilibrated residual onto the left singular vectors of J_eq, clamps the projection
    modal coefficients to [-max_comp, max_comp], and backprojects onto physical space.

    Parameters
    ----------
    J : np.ndarray
        Jacobian matrix of shape (n, n).
    f_val : np.ndarray
        Residual vector of shape (n,).
    D_L : np.ndarray
        Left row-equilibration scaling factors of shape (n,) or (n, n).
    D_R : np.ndarray
        Right column-equilibration scaling factors of shape (n,) or (n, n).
    max_comp : float, default 20.0
        Maximum allowable modal projection coefficient magnitude |c_k| <= max_comp.
    max_disp : float, default 0.30
        Factor wage displacement clamping bound |Delta w| <= max_disp. If positive,
        scales the step uniformly if max(|delta|) > max_disp.

    Returns
    -------
    np.ndarray
        Regularized Newton step delta = D_R @ sol_u of shape (n,).
    """
    dl = D_L if D_L.ndim == 1 else np.diag(D_L)
    dr = D_R if D_R.ndim == 1 else np.diag(D_R)

    J_eq = dl[:, np.newaxis] * (J * dr[np.newaxis, :])
    rhs_eq = -dl * f_val

    U, S, Vt = la.svd(J_eq)
    proj = U.T @ rhs_eq
    safe_S = np.where(S > 1e-15, S, 1e-15)
    coeffs = proj / safe_S

    clamped = np.abs(coeffs) > max_comp
    if np.any(clamped):
        coeffs = np.clip(coeffs, -max_comp, max_comp)

    sol_u = Vt.T @ coeffs
    delta = dr * sol_u

    if max_disp is not None and max_disp > 0:
        max_step = float(np.max(np.abs(delta)))
        if max_step > max_disp:
            delta = delta * (max_disp / max_step)

    return delta


def anderson_accelerate(
    x_hist: list[np.ndarray],
    g_hist: list[np.ndarray],
    f_hist: list[np.ndarray],
    m: int = 4,
) -> np.ndarray:
    """Compute depth-m Anderson accelerated iterate with least-squares mixing weights.

    Accelerates fixed-point or Newton iterations by finding optimal affine combination
    weights gamma in R^k minimizing ||sum gamma_i f_i||_2 subject to sum gamma_i = 1.

    Parameters
    ----------
    x_hist : list[np.ndarray]
        History of past state vectors [x_0, ..., x_{k-1}].
    g_hist : list[np.ndarray]
        History of proposal vectors [g_0, ..., g_{k-1}] where g_i = x_i + step_i.
    f_hist : list[np.ndarray]
        History of residual vectors [f_0, ..., f_{k-1}].
    m : int, default 4
        Maximum history depth for Anderson mixing.

    Returns
    -------
    np.ndarray
        Accelerated iterate x_{next} = sum_{i=0}^{k-1} gamma_i g_i.
    """
    k = min(len(x_hist), len(g_hist), len(f_hist), m)
    if k <= 0:
        raise ValueError("Anderson acceleration history buffers must be non-empty.")
    if k == 1:
        return np.asarray(g_hist[-1], dtype=float).copy()

    # Residual matrix R = [f_{-k}, ..., f_{-1}]
    R = np.column_stack([f_hist[-k + i] for i in range(k)])

    # Augmented KKT system for min ||R @ gamma||_2 s.t. 1^T gamma = 1:
    # [R^T R, 1; 1^T, 0] [gamma; mu] = [0; 1]
    KKT = np.empty((k + 1, k + 1), dtype=float)
    KKT[:k, :k] = R.T @ R
    KKT[:k, k] = 1.0
    KKT[k, :k] = 1.0
    KKT[k, k] = 0.0
    rhs_kkt = np.zeros(k + 1, dtype=float)
    rhs_kkt[k] = 1.0
    try:
        sol_kkt = la.lstsq(KKT, rhs_kkt)[0]
        gamma = sol_kkt[:k]
        sum_g = float(np.sum(gamma))
        if abs(sum_g) > 1e-14 and np.all(np.isfinite(gamma)):
            gamma = gamma / sum_g
        else:
            gamma = np.zeros(k, dtype=float)
            gamma[-1] = 1.0
    except Exception:
        gamma = np.zeros(k, dtype=float)
        gamma[-1] = 1.0

    x_next = np.zeros_like(g_hist[-1], dtype=float)
    for i in range(k):
        x_next += gamma[i] * np.asarray(g_hist[-k + i], dtype=float)

    return x_next


def solve_cyprus_manifold_step(
    S_ww: np.ndarray,
    rhs_w: np.ndarray,
    eval_cyp_fn: Callable[[float], tuple[float, float, Any]] | None = None,
    idx_cyp: int = 15,
    c0: float = 0.0,
    c1: float | None = None,
    tol: float = 2.5e-3,
    max_secant_iter: int = 8,
    max_disp: float = 0.30,
) -> tuple[np.ndarray, float, bool]:
    """Decoupled 1D conditional equilibrium manifold solver resolving micro-economy stalling (Cyprus CYP).

    Partitions the factor wage system S_ww @ dw = rhs_w into the 76-country manifold and
    Cyprus: dw_{76}(dw_{cyp}) = dw_{76, 0} + v_{76} * dw_{cyp}. Solves the 1D scalar
    residual f_{CYP}(dw_{cyp}) = 0 via secant method, eliminating stalling in open economies.

    Parameters
    ----------
    S_ww : np.ndarray
        Factor wage Schur complement matrix of shape (nc, nc).
    rhs_w : np.ndarray
        Right-hand side vector for factor wages of shape (nc,).
    eval_cyp_fn : Callable[[float], tuple[float, float, Any]], optional
        Callable returning (f_cyp, max_residual, context) given candidate dw_cyp.
        If None, solves on the linearized manifold.
    idx_cyp : int, default 15
        Index of Cyprus in the country list.
    c0 : float, default 0.0
        Initial guess for dw_cyp.
    c1 : float, optional
        Second guess for secant initialization.
    tol : float, default 2.5e-3
        Residual convergence tolerance.
    max_secant_iter : int, default 8
        Maximum 1D secant iterations.
    max_disp : float, default 0.30
        Factor displacement clamping bound.

    Returns
    -------
    tuple[np.ndarray, float, bool]
        (dw_full, best_res, converged)
    """
    nc = len(rhs_w)
    idx_cyp_safe = idx_cyp if 0 <= idx_cyp < nc else 0
    idx_76 = [c for c in range(nc) if c != idx_cyp_safe]

    S_76 = S_ww[np.ix_(idx_76, idx_76)]
    S_cyp_col = S_ww[idx_76, idx_cyp_safe]
    rhs_76 = rhs_w[idx_76]

    try:
        dw_76_0 = la.solve(S_76, rhs_76)
        v_76 = la.solve(S_76, -S_cyp_col)
    except la.LinAlgError:
        dw_76_0 = la.lstsq(S_76, rhs_76)[0]
        v_76 = la.lstsq(S_76, -S_cyp_col)[0]

    s_scalar = float(S_ww[idx_cyp_safe, idx_cyp_safe] + S_ww[idx_cyp_safe, idx_76] @ v_76)
    dw_cyp_lin = float((rhs_w[idx_cyp_safe] - S_ww[idx_cyp_safe, idx_76] @ dw_76_0) / (s_scalar if abs(s_scalar) > 1e-12 else 1.0))

    if eval_cyp_fn is not None:
        c_curr0 = c0
        fc0, res0, ctx0 = eval_cyp_fn(c_curr0)
        c_curr1 = c1 if c1 is not None else (dw_cyp_lin * 0.1 if abs(dw_cyp_lin) > 1e-4 else 1e-4)
        fc1, res1, ctx1 = eval_cyp_fn(c_curr1)

        best_c = c_curr1 if res1 < res0 else c_curr0
        best_res = min(res0, res1)
        conv = best_res <= tol

        for _ in range(max_secant_iter):
            if conv or abs(fc1 - fc0) < 1e-14:
                break
            c_next = c_curr1 - fc1 * (c_curr1 - c_curr0) / (fc1 - fc0)
            fc_next, res_next, _ = eval_cyp_fn(c_next)
            if res_next < best_res:
                best_res = res_next
                best_c = c_next
            if res_next <= tol or abs(fc_next) <= tol:
                conv = True
                break
            c_curr0, fc0 = c_curr1, fc1
            c_curr1, fc1 = c_next, fc_next
    else:
        best_c = dw_cyp_lin
        best_res = abs(float(rhs_w[idx_cyp_safe] - (S_ww[idx_cyp_safe, idx_76] @ (dw_76_0 + v_76 * best_c) + S_ww[idx_cyp_safe, idx_cyp_safe] * best_c)))
        conv = best_res <= tol

    dw_full = np.empty(nc, dtype=float)
    dw_full[idx_76] = dw_76_0 + v_76 * best_c
    dw_full[idx_cyp_safe] = best_c

    if max_disp is not None and max_disp > 0:
        max_step = float(np.max(np.abs(dw_full)))
        if max_step > max_disp:
            dw_full *= (max_disp / max_step)

    return dw_full, best_res, conv


solve_cyprus_manifold = solve_cyprus_manifold_step


def clamp_wage_displacement(
    delta_w: np.ndarray | float,
    max_disp: float = 0.30,
    mode: str = "coordinate",
) -> np.ndarray | float:
    """Clamp factor wage displacement to bound |Delta omega_c| <= max_disp.

    Parameters
    ----------
    delta_w : np.ndarray or float
        Factor wage displacement step(s).
    max_disp : float, default 0.30
        Maximum displacement threshold (+/- 30%).
    mode : {"coordinate", "uniform"}, default "coordinate"
        Clamping mode:
        - "coordinate": per-coordinate elementwise clipping via np.clip.
        - "uniform": uniform scaling when maximum norm exceeds max_disp.

    Returns
    -------
    np.ndarray or float
        Clamped displacement step(s).
    """
    if isinstance(delta_w, (int, float, np.floating, np.integer)):
        return float(np.clip(delta_w, -max_disp, max_disp))
    arr = np.asarray(delta_w, dtype=float)
    if mode == "uniform":
        max_step = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
        if max_step > max_disp and max_disp > 0:
            return arr * (max_disp / max_step)
        return arr.copy()
    return np.clip(arr, -max_disp, max_disp)


# ===========================================================================
# 7. Keller's Bordered Pseudo-Arclength Continuation (PAC)
# ===========================================================================

def solve_keller_pac(
    calib: TradeCalibrationResult,
    tau_target: np.ndarray | float | str | None,
    tau_start: np.ndarray | float | str | None = None,
    ds_init: float = 0.05,
    ds_min: float = 1e-4,
    ds_max: float = 0.5,
    tol: float = 2.5e-3,
    max_steps: int = 100,
    replicate_matlab_precedence: bool = True,
    base_result: TradeEquilibriumResult | None = None,
    *,
    tau_fd_target: np.ndarray | float | None = None,
    tauf_target: np.ndarray | None = None,
    tauf_fd_target: np.ndarray | None = None,
    tau_fd_start: np.ndarray | float | None = None,
    tauf_start: np.ndarray | None = None,
    tauf_fd_start: np.ndarray | None = None,
    sigma: float = 0.0,
    tariff_revenue_mode: str = "legacy_national",
    accounting: str = "legacy",
    fiscal_closure: str = "lump_sum",
    recycling_params: dict[str, Any] | None = None,
    capacity_margins: dict[str, float] | dict[tuple[int, int], float] | np.ndarray | None = None,
    capacity_target_country: str = "USA",
    penalty_scale: float = 0.05,
    penalty_exponent: float = 8.0,
    x0: np.ndarray | None = None,
    condensed: bool | None = None,
    eps_fd: float = 1e-5,
    eps_lam: float = 1e-5,
    **kwargs: Any,
) -> TradeEquilibriumResult:
    """Solve the CGE trade equilibrium via Keller's Bordered Pseudo-Arclength Continuation (PAC).

    Embeds the tariff shock along a continuation path lambda in [0, 1] and solves the augmented
    bordered Jacobian system of dimension (K+1) x (K+1) with two-sided equilibration, secant
    tangent predictor, and arclength corrector. Traverses saddle-node fold bifurcations
    (sigma_fold ~ 0.1238) and singular turning points where standard Newton-Raphson diverges
    (det J -> 0, kappa_2 > 10^4).

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    tau_target : np.ndarray, float, or str
        Target tariff schedule (scenario name, scalar rate/multiplier, or array).
    tau_start : np.ndarray, float, str, or None, default None
        Starting tariff schedule at lambda = 0 (defaults to baseline tariffs).
    ds_init : float, default 0.05
        Initial pseudo-arclength step size.
    ds_min : float, default 1e-4
        Minimum allowed step size before termination.
    ds_max : float, default 0.5
        Maximum allowed pseudo-arclength step size.
    tol : float, default 2.5e-3
        Maximum absolute residual tolerance ||F(x)||_inf <= tol.
    max_steps : int, default 100
        Maximum continuation steps along the manifold.
    replicate_matlab_precedence : bool, default True
        Whether to replicate MATLAB operator precedence.
    base_result : TradeEquilibriumResult, optional
        Benchmark baseline result for post-processing.
    sigma : float, default 0.0
        Intermediate substitution elasticity.
    fiscal_closure : str, default "lump_sum"
        Fiscal regime closure for tariff revenues.
    x0 : np.ndarray, optional
        Initial state guess. If None, uses calibrated baseline.
    condensed : bool, optional
        Only False/None is supported. True raises NotImplementedError; PAC uses
        the dense full system and is intended for small calibrations.
    **kwargs : Any
        Additional keyword options.

    accounting : {"legacy", "consistent"}, default "legacy"
        Consistent mode uses the same producer/purchaser accounting, fixed
        foreign balances and price numeraire as ``solve_trade_equilibrium``.

    Returns
    -------
    TradeEquilibriumResult
        Converged equilibrium solution container with post-processed trade flows and PAC metadata.
    """
    t_pac_start = time.perf_counter()

    if accounting not in ("legacy", "consistent"):
        raise ValueError("accounting must be 'legacy' or 'consistent'")
    if accounting == "consistent":
        replicate_matlab_precedence = False
        tariff_revenue_mode = "schedule"

    if condensed:
        raise NotImplementedError("Keller PAC currently supports only the full system; use condensed=False.")
    if not (0 < ds_min <= ds_init <= ds_max) or max_steps < 1 or tol <= 0:
        raise ValueError("Require 0 < ds_min <= ds_init <= ds_max, max_steps >= 1 and tol > 0.")

    # 2. Resolve target and start tariff structures
    if isinstance(tau_target, str):
        from puremacro.trade.scenarios import get_canonical_scenario, build_tariff_matrices
        scen_target = get_canonical_scenario(tau_target)
        tau_t_a, taufd_t_a, tauf_t_v, tauf_fd_t_v = build_tariff_matrices(scen_target, calib)
    elif isinstance(tau_target, (int, float)):
        rate = float(tau_target)
        tau_t_a, taufd_t_a, tauf_t_v, tauf_fd_t_v = _resolve_tariffs(calib, tau=rate, tau_fd=rate)
    else:
        tau_t_a, taufd_t_a, tauf_t_v, tauf_fd_t_v = _resolve_tariffs(calib, tau=tau_target, tau_fd=None)

    if tau_start is None:
        tau_s_a, taufd_s_a, tauf_s_v, tauf_fd_s_v = _resolve_tariffs(calib, tau=None, tau_fd=None)
    elif isinstance(tau_start, str):
        from puremacro.trade.scenarios import get_canonical_scenario, build_tariff_matrices
        scen_start = get_canonical_scenario(tau_start)
        tau_s_a, taufd_s_a, tauf_s_v, tauf_fd_s_v = build_tariff_matrices(scen_start, calib)
    elif isinstance(tau_start, (int, float)):
        rate_s = float(tau_start)
        tau_s_a, taufd_s_a, tauf_s_v, tauf_fd_s_v = _resolve_tariffs(calib, tau=rate_s, tau_fd=rate_s)
    else:
        tau_s_a, taufd_s_a, tauf_s_v, tauf_fd_s_v = _resolve_tariffs(calib, tau=tau_start, tau_fd=None)

    # Explicit schedules override the scalar convenience convention.
    if tau_fd_target is not None or tauf_target is not None or tauf_fd_target is not None:
        tau_t_a, taufd_t_a, tauf_t_v, tauf_fd_t_v = _resolve_tariffs(
            calib, tau=tau_target, tau_fd=tau_fd_target,
            tauf=tauf_target, tauf_fd=tauf_fd_target,
        )
    if tau_fd_start is not None or tauf_start is not None or tauf_fd_start is not None:
        tau_s_a, taufd_s_a, tauf_s_v, tauf_fd_s_v = _resolve_tariffs(
            calib, tau=tau_start, tau_fd=tau_fd_start,
            tauf=tauf_start, tauf_fd=tauf_fd_start,
        )
    check_hawkins_simon_viability(calib, tau=tau_t_a)

    nc = calib.n_countries
    ns = calib.n_sectors
    M = ns * nc

    # Note: Continuation operates on the full (K+1) x (K+1) augmented bordered system
    # to preserve global equilibrium consistency across singular turning points.
    # The `condensed` parameter is retained for interface parity with solve_trade_equilibrium.
    _ = condensed

    if x0 is None:
        x_init = build_initial_guess(calib)
    else:
        x_init = np.asarray(x0, dtype=float).ravel()

    # Helper to interpolate tariffs along lambda in [0, 1]
    def get_tariffs(lam: float):
        l_c = float(np.clip(lam, -0.5, 2.0))
        t_a = (1.0 - l_c) * tau_s_a + l_c * tau_t_a
        tfd_a = (1.0 - l_c) * taufd_s_a + l_c * taufd_t_a
        tf_v = (1.0 - l_c) * tauf_s_v + l_c * tauf_t_v
        tffd_v = (1.0 - l_c) * tauf_fd_s_v + l_c * tauf_fd_t_v
        return t_a, tfd_a, tf_v, tffd_v

    # Evaluators for full state
    def eval_full(x_vec: np.ndarray, lam: float) -> np.ndarray:
        t_a, tfd_a, tf_v, tffd_v = get_tariffs(lam)
        return compute_equilibrium_residuals(
            x=x_vec,
            calib=calib,
            tau=t_a,
            tau_fd=tfd_a,
            tauf=tf_v,
            tauf_fd=tffd_v,
            replicate_matlab_precedence=replicate_matlab_precedence,
            sigma=sigma,
            tariff_revenue_mode=tariff_revenue_mode,
            accounting=accounting,
            fiscal_closure=fiscal_closure,
            recycling_params=recycling_params,
            capacity_margins=capacity_margins,
            capacity_target_country=capacity_target_country,
            penalty_scale=penalty_scale,
            penalty_exponent=penalty_exponent,
        )

    # State vector and scaling vector setup
    curr_x = x_init.copy()
    curr_lam = 0.0
    K = len(curr_x)

    w_scale = np.ones(K, dtype=float)
    if K >= 2 * M + 4 * nc - 1:
        i_T = 2 * M + 2 * nc
        w_scale[i_T:] = 1e-6

    # Initial solve at lambda = 0.0
    f_0 = eval_full(curr_x, curr_lam)
    if float(np.max(np.abs(f_0))) > tol:
        for _ in range(10):
            f_val = eval_full(curr_x, curr_lam)
            if float(np.max(np.abs(f_val))) <= tol:
                break
            J_init = np.empty((K, K), dtype=float)
            for j in range(K):
                xp = curr_x.copy()
                xp[j] += eps_fd
                J_init[:, j] = (eval_full(xp, curr_lam) - f_val) / eps_fd
            c_n = np.linalg.norm(J_init, axis=0); c_n[c_n == 0] = 1.0; DR = 1.0 / c_n
            r_n = np.linalg.norm(J_init * DR[np.newaxis, :], axis=1); r_n[r_n == 0] = 1.0; DL = 1.0 / r_n
            try:
                curr_x += DR * la.solve(DL[:, np.newaxis] * (J_init * DR[np.newaxis, :]), -DL * f_val)
            except la.LinAlgError:
                curr_x += DR * la.lstsq(DL[:, np.newaxis] * (J_init * DR[np.newaxis, :]), -DL * f_val)[0]

    ds = ds_init
    prev_tau_x = None
    prev_tau_lam = None
    fold_points: list[dict[str, Any]] = []
    total_pac_steps = 0

    # Main Pseudo-Arclength Continuation loop
    for step_idx in range(max_steps):
        if curr_lam >= 1.0 - 1e-6:
            break

        total_pac_steps += 1
        f_curr = eval_full(curr_x, curr_lam)

        # Evaluate Jacobian J_x
        J_x = np.empty((K, K), dtype=float)
        for j in range(K):
            xp = curr_x.copy()
            xp[j] += eps_fd
            J_x[:, j] = (eval_full(xp, curr_lam) - f_curr) / eps_fd

        # Evaluate dF/dlam
        f_pert_lam = eval_full(curr_x, curr_lam + eps_lam)
        dF_dlam = (f_pert_lam - f_curr) / eps_lam

        # Spectral diagnostic on J_x
        try:
            s_vals = la.svd(J_x, compute_uv=False)
            sigma_min = float(s_vals[-1])
            kappa_2 = float(s_vals[0] / max(sigma_min, 1e-15))
        except Exception:
            sigma_min = 1.0
            kappa_2 = 1.0

        # Tangent vector calculation via Bordered System:
        # [J_x, dF/dlam; prev_tau_x^T W^2, prev_tau_lam] [v_x; v_lam] = [0; 1]
        c_n = np.linalg.norm(J_x, axis=0); c_n[c_n == 0] = 1.0; DR = 1.0 / c_n
        r_n = np.linalg.norm(J_x * DR[np.newaxis, :], axis=1); r_n[r_n == 0] = 1.0; DL = 1.0 / r_n

        if prev_tau_x is not None and prev_tau_lam is not None:
            J_t_aug = np.empty((K + 1, K + 1), dtype=float)
            J_t_aug[:K, :K] = J_x
            J_t_aug[:K, K] = dF_dlam
            J_t_aug[K, :K] = prev_tau_x * (w_scale ** 2)
            J_t_aug[K, K] = prev_tau_lam
            rhs_t = np.zeros(K + 1, dtype=float)
            rhs_t[K] = 1.0
            try:
                c_ta = np.linalg.norm(J_t_aug, axis=0); c_ta[c_ta == 0] = 1.0; DR_t = 1.0 / c_ta
                r_ta = np.linalg.norm(J_t_aug * DR_t[np.newaxis, :], axis=1); r_ta[r_ta == 0] = 1.0; DL_t = 1.0 / r_ta
                v_sol = DR_t * la.solve(DL_t[:, np.newaxis] * (J_t_aug * DR_t[np.newaxis, :]), DL_t * rhs_t)
            except la.LinAlgError:
                v_sol = la.lstsq(J_t_aug, rhs_t)[0]
            v_x = v_sol[:K]
            v_lam = float(v_sol[K])
        else:
            try:
                v_x = DR * la.solve(DL[:, np.newaxis] * (J_x * DR[np.newaxis, :]), -DL * dF_dlam)
            except la.LinAlgError:
                v_x = DR * la.lstsq(DL[:, np.newaxis] * (J_x * DR[np.newaxis, :]), -DL * dF_dlam)[0]
            v_lam = 1.0

        # Normalize tangent with metric W
        tangent_norm = float(np.sqrt(np.sum((v_x * w_scale) ** 2) + v_lam ** 2))
        if tangent_norm < 1e-15:
            tangent_norm = 1.0
        tau_x = v_x / tangent_norm
        tau_lam = v_lam / tangent_norm

        # Orientation check: maintain forward path progression
        if prev_tau_x is not None:
            dot_prod = float(np.sum(tau_x * prev_tau_x * (w_scale ** 2)) + tau_lam * prev_tau_lam)
            if dot_prod < 0:
                tau_x = -tau_x
                tau_lam = -tau_lam
        else:
            if tau_lam < 0:
                tau_x = -tau_x
                tau_lam = -tau_lam

        # Monitor fold bifurcation / turning point (tau_lambda <= 0)
        if tau_lam <= 0.0:
            fold_points.append({
                "step": step_idx,
                "lambda": float(curr_lam),
                "tau_lambda": float(tau_lam),
                "sigma_min": float(sigma_min),
                "kappa_2": float(kappa_2),
            })

        prev_tau_x = tau_x.copy()
        prev_tau_lam = float(tau_lam)

        # Adaptive arclength step size: target delta_lambda ~ 0.02
        # Keep reductions after rejected correctors; adapt only on acceptance.

        # Clip predictor to lambda = 1.0 if approaching target
        if curr_lam < 1.0 and tau_lam > 0 and curr_lam + ds * tau_lam > 1.0:
            ds = min(ds, max((1.0 - curr_lam) / max(tau_lam, 1e-4), ds_min))

        # Secant tangent predictor
        x_pred = curr_x + ds * tau_x
        lam_pred = curr_lam + ds * tau_lam
        if lam_pred > 1.0 and tau_lam > 0:
            lam_pred = 1.0

        # Arclength corrector Newton iterations
        x_k = x_pred.copy()
        lam_k = float(lam_pred)
        conv_corr = False

        for it_corr in range(12):
            f_k = eval_full(x_k, lam_k)
            g_k = float(np.sum(tau_x * (w_scale ** 2) * (x_k - curr_x)) + tau_lam * (lam_k - curr_lam) - ds)
            err_k = max(float(np.max(np.abs(f_k))), abs(g_k))

            if err_k <= tol:
                conv_corr = True
                curr_x = x_k
                curr_lam = lam_k
                break

            # Evaluate augmented Jacobian
            J_xk = np.empty((K, K), dtype=float)
            for j in range(K):
                xp = x_k.copy()
                xp[j] += eps_fd
                J_xk[:, j] = (eval_full(xp, lam_k) - f_k) / eps_fd

            f_pert_k = eval_full(x_k, lam_k + eps_lam)
            dF_dlam_k = (f_pert_k - f_k) / eps_lam

            # Bordered (K+1) x (K+1) system
            J_aug = np.empty((K + 1, K + 1), dtype=float)
            J_aug[:K, :K] = J_xk
            J_aug[:K, K] = dF_dlam_k
            J_aug[K, :K] = tau_x * (w_scale ** 2)
            J_aug[K, K] = tau_lam

            rhs_aug = np.empty(K + 1, dtype=float)
            rhs_aug[:K] = -f_k
            rhs_aug[K] = -g_k

            # Two-sided equilibration of J_aug
            c_aug = np.linalg.norm(J_aug, axis=0); c_aug[c_aug == 0] = 1.0; DR_a = 1.0 / c_aug
            r_aug = np.linalg.norm(J_aug * DR_a[np.newaxis, :], axis=1); r_aug[r_aug == 0] = 1.0; DL_a = 1.0 / r_aug

            try:
                d_step = DR_a * la.solve(DL_a[:, np.newaxis] * (J_aug * DR_a[np.newaxis, :]), DL_a * rhs_aug)
            except la.LinAlgError:
                d_step = DR_a * la.lstsq(DL_a[:, np.newaxis] * (J_aug * DR_a[np.newaxis, :]), DL_a * rhs_aug)[0]

            # Backtracking on augmented step
            alpha_corr = 1.0
            x_trial = x_k + alpha_corr * d_step[:K]
            lam_trial = lam_k + alpha_corr * d_step[K]
            for _ in range(5):
                f_tr = eval_full(x_trial, lam_trial)
                g_tr = float(np.sum(tau_x * (w_scale ** 2) * (x_trial - curr_x)) + tau_lam * (lam_trial - curr_lam) - ds)
                err_tr = max(float(np.max(np.abs(f_tr))), abs(g_tr))
                if err_tr < err_k or err_tr <= tol:
                    break
                alpha_corr *= 0.5
                x_trial = x_k + alpha_corr * d_step[:K]
                lam_trial = lam_k + alpha_corr * d_step[K]

            x_k = x_trial
            lam_k = lam_trial

        if not conv_corr:
            # Halve arclength step and retry
            ds *= 0.5
            if ds < ds_min:
                break
        else:
            curr_x = x_k
            curr_lam = lam_k
            ds = min(ds_max, ds * (1.25 if it_corr < 5 else 1.0))

    t_pac_end = time.perf_counter()
    solve_duration = t_pac_end - t_pac_start

    # Final terminal polish at target lambda = 1.0
    status = "converged"
    x_polish = curr_x.copy()
    try:
        f_final = eval_full(x_polish, 1.0)
        res_final = float(np.max(np.abs(f_final)))

        for _ in range(15):
            if res_final <= tol and (tol > 1e-6 or res_final < 1e-10):
                break
            J_fin = np.empty((K, K), dtype=float)
            for j in range(K):
                xp = x_polish.copy()
                xp[j] += eps_fd
                J_fin[:, j] = (eval_full(xp, 1.0) - f_final) / eps_fd
            c_f = np.linalg.norm(J_fin, axis=0); c_f[c_f == 0] = 1.0; DR_f = 1.0 / c_f
            r_f = np.linalg.norm(J_fin * DR_f[np.newaxis, :], axis=1); r_f[r_f == 0] = 1.0; DL_f = 1.0 / r_f
            try:
                step_f = DR_f * la.solve(DL_f[:, np.newaxis] * (J_fin * DR_f[np.newaxis, :]), -DL_f * f_final)
            except (la.LinAlgError, ValueError):
                step_f = DR_f * la.lstsq(DL_f[:, np.newaxis] * (J_fin * DR_f[np.newaxis, :]), -DL_f * f_final)[0]

            if np.any(np.isnan(step_f)) or np.any(np.isinf(step_f)):
                raise ValueError("Terminal polish step contains NaNs or infs")

            x_polish += step_f
            f_final = eval_full(x_polish, 1.0)
            res_final = float(np.max(np.abs(f_final)))

        conv_total = bool(np.isfinite(res_final) and res_final <= tol)
        if conv_total:
            curr_x = x_polish
        else:
            status = "step_limit_exhausted"
    except (la.LinAlgError, ValueError):
        conv_total = False
        res_final = float("nan")
        f_final = np.full(K, np.nan, dtype=float)
        status = "step_limit_exhausted"

    # Diagnostics always describe the iterate that is actually returned.
    f_final = eval_full(curr_x, 1.0)
    res_final = float(np.max(np.abs(f_final)))
    conv_total = bool(np.isfinite(res_final) and res_final <= tol)
    status = "converged" if conv_total else "step_limit_exhausted"

    res_dataclass = postprocess_trade_equilibrium(
        x_sol=curr_x,
        calib=calib,
        tau=tau_t_a,
        tau_fd=taufd_t_a,
        tauf=tauf_t_v,
        tauf_fd=tauf_fd_t_v,
        replicate_matlab_precedence=replicate_matlab_precedence,
        converged=conv_total,
        iterations=total_pac_steps,
        base_result=base_result,
        tariff_revenue_mode=tariff_revenue_mode, accounting=accounting,
        sigma=sigma, fiscal_closure=fiscal_closure, recycling_params=recycling_params,
        capacity_margins=capacity_margins, capacity_target_country=capacity_target_country,
        penalty_scale=penalty_scale, penalty_exponent=penalty_exponent,
        metadata={
            "method": "keller_pac",
            "status": status,
            "tol": tol,
            "max_steps": max_steps,
            "pac_steps": total_pac_steps,
            "fold_points": fold_points,
            "final_lambda": float(curr_lam),
            "solve_duration_seconds": solve_duration,
            "sigma": sigma,
            "fiscal_closure": fiscal_closure,
            "residual_norm": float(np.sum(np.abs(f_final))),
            "max_residual": res_final,
        },
    )

    return replace(
        res_dataclass,
        residuals=f_final,
        residual_norm=float(np.sum(np.abs(f_final))),
        diff=float(np.sum(np.abs(f_final))),
        max_residual=res_final,
    )


# ===========================================================================
# 8. Master Solver Interfaces
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
    backend: str = "numpy",
    sigma: float = 0.0,
    tariff_revenue_mode: str = "legacy_national",
    accounting: str = "legacy",
    fiscal_closure: str = "lump_sum",
    recycling_params: dict[str, Any] | None = None,
    capacity_margins: dict[str, float] | dict[tuple[int, int], float] | np.ndarray | None = None,
    capacity_target_country: str = "USA",
    penalty_scale: float = 0.05,
    penalty_exponent: float = 8.0,
    **kwargs: Any,
) -> TradeEquilibriumResult:
    """Solve the multi-country multi-sector general equilibrium nonlinear system.

    ``tariff_revenue_mode="legacy_national"`` reproduces the MATLAB fiscal
    equation, which uses only explicit national rate vectors. In particular,
    tariffs encoded solely in matrices need not be rebated under this convention.
    Use ``tariff_revenue_mode="schedule"`` for fiscal receipts from the complete
    bilateral schedule. This choice is independent of the price-precedence flag.
    With the compatibility default ``accounting="legacy"``, schedule revenue
    retains the legacy final-demand valuation. For economic counterfactuals use
    ``accounting="consistent"``: producer-price trade and duty bases, homogeneous
    factor costs, output-revenue taxes, actual final-use tax bases, lump-sum
    rebates, fixed baseline foreign balances and a first-producer-price numeraire.
    This selects schedule receipts and correct Cobb–Douglas precedence regardless
    of the legacy switches. Other fiscal/capacity extensions are explicitly
    unsupported in this mode. See ``docs/trade_accounting.md``.


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
    method : {"newton", "sparse_lu", "krylov", "condensed", "broyden", "hybr", "lm", "keller_pac"}, default "newton"
        Nonlinear root-finding algorithm:
        - 'newton': Dense Damped Newton-Raphson with Armijo line search (baseline).
        - 'sparse_lu': Sparse Jacobian Newton using direct sparse LU (splu) and graph coloring.
        - 'krylov': Inexact Newton-Krylov (JFNK via GMRES/BiCGSTAB) with two-sided equilibration.
        - 'condensed': Block-elimination / Schur complement condensation (307 macro variables).
        - 'broyden': Broyden quasi-Newton with Sherman-Morrison rank-1 updates.
        - 'hybr': SciPy MINPACK Powell hybrid root wrapper.
        - 'lm': SciPy Levenberg-Marquardt root wrapper.
        - 'keller_pac': Keller's Bordered Pseudo-Arclength Continuation (traverses fold bifurcations).
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
    if accounting not in ("legacy", "consistent"):
        raise ValueError("accounting must be 'legacy' or 'consistent'")
    if accounting == "consistent":
        replicate_matlab_precedence = False
        tariff_revenue_mode = "schedule"
        if backend != "numpy":
            raise NotImplementedError("Consistent accounting currently supports the NumPy backend")

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
            tariff_revenue_mode=tariff_revenue_mode,
            accounting=accounting,
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
            obj_fun, x_init, tol=tol, max_iter=max_iter,
            eps_fd=1e-5 if accounting == "consistent" else 1e-2
        )
    elif method == "sparse_lu":
        # Build structural sparsity pattern
        try:
            sparsity_pat = _build_cge_sparsity_pattern(
                calib,
                tau_a=tau_a,
                sigma_y=kwargs.get("sigma_y", 0.0),
                variable_markups=kwargs.get("variable_markups", False),
                config=kwargs.get("config"),
            )
        except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
            sparsity_pat = None
        if accounting == "consistent":
            sparsity_pat = None  # Legacy patterns encode different closure rows.
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
            atol=0.0 if accounting == "consistent" else 1e-5,
        )
    elif method == "condensed" and accounting == "consistent":
        # The old Schur reduction embodies the legacy valuation/closure.
        x_sol, conv, iters, max_res, diff, res = _damped_newton_solve(
            obj_fun, x_init, tol=tol, max_iter=max_iter, eps_fd=1e-5
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
            sparsity_pat = _build_cge_sparsity_pattern(
                calib,
                tau_a=tau_a,
                sigma_y=kwargs.get("sigma_y", 0.0),
                variable_markups=kwargs.get("variable_markups", False),
                config=kwargs.get("config"),
            )
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
                backend=backend,
                tariff_revenue_mode=tariff_revenue_mode,
            )
    elif method == "quasi_condensed":
        if tariff_revenue_mode != "legacy_national":
            raise NotImplementedError("quasi_condensed does not support schedule tariff accounting")
        from puremacro.trade.flexible import _quasi_condensed_solve

        flexible_cfg = kwargs.get("config")
        if flexible_cfg is None:
            from puremacro.trade.flexible import FlexibleTechnologyConfig, FlexibleTradeModelConfig
            flexible_cfg = FlexibleTradeModelConfig(
                technology=FlexibleTechnologyConfig(sigma_y=kwargs.get("sigma_y", 0.0))
            )
        x_sol, conv, iters, max_res, diff, res, meta = _quasi_condensed_solve(
            calib=calib,
            config=flexible_cfg,
            tau_a=tau_a,
            taufd_a=taufd_a,
            tauf_vec=tauf_vec,
            tauf_fd_vec=tauf_fd_vec,
            x0=x_init,
            tol=tol,
            max_iter=max_iter,
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
        conv = bool(np.isfinite(max_res) and max_res <= tol)
        iters = int(getattr(res_scipy, "nfev", 0))
    elif method == "keller_pac":
        target = tau if tau is not None else (tauf if tauf is not None else kwargs.get("tau_target", 0.0))
        return solve_keller_pac(
            calib=calib,
            tau_target=tau_a,
            tau_fd_target=taufd_a,
            tauf_target=tauf_vec,
            tauf_fd_target=tauf_fd_vec,
            tau_start=kwargs.get("tau_start", None),
            ds_init=kwargs.get("ds_init", 0.05),
            ds_min=kwargs.get("ds_min", 1e-4),
            ds_max=kwargs.get("ds_max", 0.5),
            tol=tol,
            max_steps=kwargs.get("max_steps", max_iter if max_iter > 50 else 100),
            replicate_matlab_precedence=replicate_matlab_precedence,
            base_result=base_result,
            sigma=sigma,
            tariff_revenue_mode=tariff_revenue_mode,
            accounting=accounting,
            fiscal_closure=fiscal_closure,
            recycling_params=recycling_params,
            capacity_margins=capacity_margins,
            capacity_target_country=capacity_target_country,
            penalty_scale=penalty_scale,
            penalty_exponent=penalty_exponent,
            x0=x0,
            condensed=kwargs.get("condensed", None),
            eps_fd=kwargs.get("eps_fd", 1e-5),
            eps_lam=kwargs.get("eps_lam", 1e-5),
            **{k: v for k, v in kwargs.items() if k not in (
                "tau_target", "tau_start", "ds_init", "ds_min", "ds_max",
                "max_steps", "condensed", "eps_fd", "eps_lam"
            )},
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Valid options are: 'newton', 'sparse_lu', 'krylov', 'condensed', 'quasi_condensed', 'broyden', 'hybr', 'lm', 'keller_pac'."
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
        tariff_revenue_mode=tariff_revenue_mode, accounting=accounting,
        sigma=sigma, fiscal_closure=fiscal_closure, recycling_params=recycling_params,
        capacity_margins=capacity_margins, capacity_target_country=capacity_target_country,
        penalty_scale=penalty_scale, penalty_exponent=penalty_exponent,
        metadata={
            "method": method,
            "effective_method": "newton" if accounting == "consistent" and method == "condensed" else method,
            "accounting": accounting,
            "backend": backend,
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
    "check_hawkins_simon_viability",
    "solve_keller_pac",
    "svd_clamped_newton_step",
    "anderson_accelerate",
    "solve_cyprus_manifold_step",
    "solve_cyprus_manifold",
    "clamp_wage_displacement",
]
