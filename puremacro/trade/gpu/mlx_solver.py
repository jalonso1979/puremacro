"""Apple Silicon MLX General Equilibrium Solver for puremacro.trade.

Provides native Apple MLX acceleration utilizing Unified Memory Architecture (UMA)
for zero-copy tensor sharing, asynchronous GPU stream execution for batched tensor
contractions, and precision-preserving CPU stream execution for LAPACK linear algebra.
"""
from __future__ import annotations

from dataclasses import replace
import time
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from puremacro.trade.gpu.backend import (
    detect_device,
    get_memory_usage,
    has_mlx,
    reset_peak_memory,
)
from puremacro.trade.gpu.batched_jacobian import BatchedJacobianEvaluator
from puremacro.trade.postprocessing import postprocess_trade_equilibrium

if TYPE_CHECKING:
    from puremacro.trade._results import TradeCalibrationResult, TradeEquilibriumResult

if has_mlx():
    import mlx.core as mx


def solve_trade_equilibrium_mlx(
    calib: TradeCalibrationResult,
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    tauf: np.ndarray | None = None,
    tauf_fd: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    batch_size: int = 230,
    max_iter: int = 50,
    tol: float = 2.5e-3,
    damping: float = 1e-4,
    ad_mode: Literal["finite_diff", "forward", "vjp"] = "finite_diff",
    replicate_matlab_precedence: bool = True,
    base_result: TradeEquilibriumResult | None = None,
    verbose: bool = False,
) -> TradeEquilibriumResult:
    """Solve multi-country multi-sector CGE trade equilibrium using Apple MLX.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    tau : np.ndarray | None, default None
        Effective intermediate tariff matrix. If None, defaults to baseline (all 1s).
    tau_fd : np.ndarray | None, default None
        Effective final demand tariff matrix. If None, defaults to baseline (all 1s).
    tauf : np.ndarray | None, default None
        National intermediate tariff vector. If None, defaults to baseline (all 1s).
    tauf_fd : np.ndarray | None, default None
        National final demand tariff vector. If None, defaults to baseline (all 1s).
    x0 : np.ndarray | None, default None
        Initial guess vector (length 230, 307, or full 7,237).
    batch_size : int, default 230
        Perturbation dimension B: 230 or 307.
    max_iter : int, default 50
        Maximum Levenberg-Marquardt iterations.
    tol : float, default 2.5e-3
        Infinity-norm convergence tolerance: max(|F(x)|) <= tol.
    damping : float, default 1e-4
        Initial Levenberg-Marquardt damping parameter mu.
    ad_mode : str, default 'finite_diff'
        Differentiation mode ('finite_diff', 'forward', 'vjp').
    replicate_matlab_precedence : bool, default True
        Whether to replicate MATLAB operator precedence at ff_equi.m:31.
    base_result : TradeEquilibriumResult | None, default None
        Baseline equilibrium for terms-of-trade and CPI calculations.
    verbose : bool, default False
        Whether to print per-iteration convergence diagnostics.

    Returns
    -------
    TradeEquilibriumResult
        Fully post-processed trade equilibrium dataclass container.
    """
    if not has_mlx():
        raise RuntimeError("Apple MLX is not installed or not supported on this platform.")

    t_start = time.perf_counter()
    reset_peak_memory()

    nc = calib.n_countries
    ns = calib.n_sectors
    nfd = calib.n_final_demand
    M = ns * nc

    # Initialize tariffs
    if tau is None:
        tau_a = np.ones((M, ns, nc), dtype=float)
    else:
        tau_a = np.asarray(tau, dtype=float)
        if tau_a.ndim == 4:
            tau_a = tau_a.transpose(1, 0, 2, 3).reshape(M, ns, nc)

    if tau_fd is None:
        taufd_a = np.ones((M, nfd, nc), dtype=float)
    else:
        taufd_a = np.asarray(tau_fd, dtype=float)
        if taufd_a.ndim == 4:
            taufd_a = taufd_a.transpose(1, 0, 2, 3).reshape(M, nfd, nc)

    tauf_vec = np.ones(nc, dtype=float) if tauf is None else np.asarray(tauf, dtype=float).ravel()
    tauf_fd_vec = np.ones(nc, dtype=float) if tauf_fd is None else np.asarray(tauf_fd, dtype=float).ravel()

    # Construct MLX GPU evaluator
    evaluator_mlx = BatchedJacobianEvaluator(
        calib=calib,
        tau_a=tau_a,
        taufd_a=taufd_a,
        tauf_vec=tauf_vec,
        tauf_fd_vec=tauf_fd_vec,
        device="mlx",
        backend="mlx",
        batch_size=batch_size,
        replicate_matlab_precedence=replicate_matlab_precedence,
    )

    # Initial guess handling
    if x0 is None:
        log_w0 = np.zeros(nc, dtype=float)
        T0 = calib.T.ravel() if calib.T is not None else np.zeros(nc, dtype=float)
        XN0 = calib.invforT.ravel()[: nc - 1]
        if batch_size == 230:
            xm = np.concatenate([log_w0, T0, XN0])
        else:
            log_r0 = np.zeros(nc, dtype=float)
            xm = np.concatenate([log_r0, log_w0, T0, XN0])
    elif len(x0) == 2 * M + 4 * nc - 1:
        i_w_start = 2 * M + nc
        i_w_end = 2 * M + 2 * nc
        i_T_end = 2 * M + 3 * nc
        i_XN_end = 2 * M + 4 * nc - 1
        w_raw = x0[i_w_start:i_w_end]
        T_raw = x0[i_w_end:i_T_end]
        XN_raw = x0[i_T_end:i_XN_end]
        if batch_size == 230:
            xm = np.concatenate([w_raw, T_raw, XN_raw])
        else:
            r_raw = x0[2 * M : i_w_start]
            xm = np.concatenate([r_raw, w_raw, T_raw, XN_raw])
    elif len(x0) in (230, 307):
        xm = np.asarray(x0, dtype=float).copy()
    else:
        raise ValueError(
            f"Invalid initial guess dimension {len(x0)}. Expected 230, 307, or {2 * M + 4 * nc - 1}."
        )

    n_m = len(xm)
    n_factor_vars = nc if batch_size == 230 else 2 * nc

    # Initial residual evaluation
    f_m, f_full, p_sol, p_sol_3d, y_sol = evaluator_mlx.eval_macro_single(xm, compute_full=True)
    res_max = float(np.max(np.abs(f_m)))
    diff = float(np.sum(np.abs(f_m)))

    if res_max <= tol:
        if verbose:
            print(f"[solve_trade_equilibrium_mlx] Initial point already converged: max_res = {res_max:.4e}")
        p_vec = p_sol
        y_vec = y_sol
        conv = True
        iters = 0
    else:
        mu = damping
        conv = False
        iters = 0

        for it in range(max_iter):
            iters = it + 1
            res_max = float(np.max(np.abs(f_m)))

            if res_max <= tol:
                conv = True
                break

            # Evaluate batched parallel Jacobian on native Apple MLX:
            # GPU stream (float32) for coarse phase (res_max > 5e6),
            # CPU stream (float64) for precision polishing (res_max <= 5e6) on UMA
            target_stream = mx.gpu if res_max > 5e6 else mx.cpu
            J = evaluator_mlx.evaluate_batched_jacobian(
                xm, f_base=f_m, ad_mode=ad_mode, stream=target_stream
            )

            # Two-sided diagonal equilibration (Jacobi scaling)
            c_norm = np.linalg.norm(J, axis=0)
            c_norm[c_norm == 0] = 1.0
            D_R = 1.0 / c_norm
            J_scaled = J * D_R[np.newaxis, :]

            r_norm = np.linalg.norm(J_scaled, axis=1)
            r_norm[r_norm == 0] = 1.0
            D_L = 1.0 / r_norm
            J_equil = D_L[:, np.newaxis] * J_scaled
            rhs_equil = -D_L * f_m

            # Levenberg-Marquardt normal equations solve via native Apple MLX CPU stream (LAPACK)
            JT_J = J_equil.T @ J_equil
            JT_rhs = J_equil.T @ rhs_equil
            try:
                with mx.stream(mx.cpu):
                    A_reg = mx.array(JT_J + mu * np.eye(n_m), dtype=mx.float64)
                    rhs_mx = mx.array(JT_rhs[:, None], dtype=mx.float64)
                    sol_u_mx = mx.linalg.solve(A_reg, rhs_mx, stream=mx.cpu)
                    sol_u = np.array(sol_u_mx).ravel()
            except Exception:
                with mx.stream(mx.cpu):
                    A_reg = mx.array(JT_J + (mu + 1e-4) * np.eye(n_m), dtype=mx.float64)
                    rhs_mx = mx.array(JT_rhs[:, None], dtype=mx.float64)
                    sol_u_mx = mx.linalg.solve(A_reg, rhs_mx, stream=mx.cpu)
                    sol_u = np.array(sol_u_mx).ravel()

            delta_m = D_R * sol_u

            # Uniform step size clamping on factor prices
            max_factor_step = float(np.max(np.abs(delta_m[:n_factor_vars])))
            if max_factor_step > 0.3:
                delta_m *= (0.3 / max_factor_step)

            # Backtracking Armijo line search on merit function ||D_L * f_m||_2
            alpha_step = 1.0
            norm_0 = res_max
            norm_2_0 = float(np.linalg.norm(D_L * f_m))

            xm_trial = xm + alpha_step * delta_m
            xm_trial[:n_factor_vars] = np.clip(xm_trial[:n_factor_vars], -5.0, 5.0)
            f_trial_m, f_trial_full, p_trial, p_trial_3d, y_trial = evaluator_mlx.eval_macro_single(
                xm_trial, compute_full=True
            )

            best_trial = (xm_trial, f_trial_m, f_trial_full, p_trial, y_trial)
            best_res_m = float(np.max(np.abs(f_trial_m)))
            best_norm_2 = float(np.linalg.norm(D_L * f_trial_m))

            accepted = False
            for _ in range(12):
                res_trial_m = float(np.max(np.abs(f_trial_m)))
                norm_2_trial = float(np.linalg.norm(D_L * f_trial_m))

                if norm_2_trial < best_norm_2 or res_trial_m < best_res_m:
                    best_trial = (xm_trial, f_trial_m, f_trial_full, p_trial, y_trial)
                    best_res_m = res_trial_m
                    best_norm_2 = norm_2_trial

                if res_trial_m <= tol or norm_2_trial < norm_2_0 or res_trial_m < norm_0:
                    accepted = True
                    break

                alpha_step *= 0.5
                xm_trial = xm + alpha_step * delta_m
                xm_trial[:n_factor_vars] = np.clip(xm_trial[:n_factor_vars], -5.0, 5.0)
                f_trial_m, f_trial_full, p_trial, p_trial_3d, y_trial = evaluator_mlx.eval_macro_single(
                    xm_trial, compute_full=True
                )

            if accepted:
                xm = xm_trial
                f_m = f_trial_m
                f_full = f_trial_full
                p_sol = p_trial
                y_sol = y_trial
                mu = max(mu * 0.3, 1e-8)
            elif best_norm_2 < norm_2_0 or best_res_m < norm_0:
                xm, f_m, f_full, p_sol, y_sol = best_trial
                mu = max(mu * 0.5, 1e-8)
            else:
                mu = min(mu * 5.0, 1e2)

            if verbose:
                print(
                    f"[mlx] Iter {it + 1:02d}: "
                    f"max_res = {np.max(np.abs(f_m)):.4e}, "
                    f"step = {alpha_step:.4f}, mu = {mu:.2e}"
                )

    # Reconstruct full general equilibrium state vector in double precision
    p_vec = p_sol
    y_vec = y_sol

    if batch_size == 230:
        log_w = xm[:nc]
        log_r = log_w
        T_vec = xm[nc : 2 * nc]
        XN_vec = xm[2 * nc :]
        x_full = np.concatenate([
            np.log(np.maximum(p_vec, 1e-12)),
            np.log(np.maximum(y_vec, 1e-12)),
            log_r,
            log_w,
            T_vec,
            XN_vec,
        ])
    else:
        log_r = xm[:nc]
        log_w = xm[nc : 2 * nc]
        T_vec = xm[2 * nc : 3 * nc]
        XN_vec = xm[3 * nc :]
        x_full = np.concatenate([
            np.log(np.maximum(p_vec, 1e-12)),
            np.log(np.maximum(y_vec, 1e-12)),
            log_r,
            log_w,
            T_vec,
            XN_vec,
        ])

    # Final residual check
    _, f_full_final, _, _, _ = evaluator_mlx.eval_macro_single(xm, compute_full=True)
    res_final = f_full_final if f_full_final is not None else f_m
    max_res = float(np.max(np.abs(res_final)))
    diff = float(np.sum(np.abs(res_final)))
    conv = bool(max_res <= tol)

    t_end = time.perf_counter()
    mem_info = get_memory_usage("mlx")

    res_dataclass = postprocess_trade_equilibrium(
        x_sol=x_full,
        calib=calib,
        tau=tau_a,
        tau_fd=taufd_a,
        tauf=tauf_vec,
        tauf_fd=tauf_fd_vec,
        replicate_matlab_precedence=replicate_matlab_precedence,
        converged=conv,
        iterations=iters if 'iters' in locals() else 0,
        base_result=base_result,
        metadata={
            "method": "mlx_accelerated",
            "device": "mlx",
            "backend": "mlx",
            "batch_size": batch_size,
            "tol": tol,
            "max_iter": max_iter,
            "solve_duration_seconds": t_end - t_start,
            "memory_usage": mem_info,
        },
    )

    res_dataclass = replace(
        res_dataclass,
        residuals=res_final,
        residual_norm=diff,
        diff=diff,
        max_residual=max_res,
        converged=conv,
    )

    return res_dataclass
