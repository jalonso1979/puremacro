"""Adaptive Homotopy Parameter Continuation for Large Tariff Shocks.

Provides robust numerical continuation along the path lambda in [0, 1]:

    tau(lambda) = tau_0 + lambda * (tau_1 - tau_0)
    taufd(lambda) = taufd_0 + lambda * (taufd_1 - taufd_0)

with adaptive step sizing, warm-starting, and automatic backtracking on non-convergence.
This enables convergence for extreme asymmetric tariff shocks (such as NAFTA / China
tariffs from 25% to 145% in scenarios t10_25 .. t10_145) where standard un-continued
Newton methods diverge.
"""
from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np

from puremacro.trade.gpu.backend import _resolve_backend_device
from puremacro.trade.gpu.solver_gpu import solve_trade_equilibrium_gpu

if TYPE_CHECKING:
    from puremacro.trade._results import TradeCalibrationResult, TradeEquilibriumResult


def solve_homotopy_continuation(
    calib: TradeCalibrationResult,
    target_tau: np.ndarray,
    target_tau_fd: np.ndarray,
    target_tauf: np.ndarray | None = None,
    target_tauf_fd: np.ndarray | None = None,
    base_tau: np.ndarray | None = None,
    base_tau_fd: np.ndarray | None = None,
    base_tauf: np.ndarray | None = None,
    base_tauf_fd: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    device: str | None = None,
    backend: str | None = None,
    batch_size: int | None = None,
    initial_step: float = 0.1,
    min_step: float = 1e-4,
    max_step: float = 0.5,
    tol: float = 2.5e-3,
    damping: float = 1e-4,
    max_iter_per_step: int = 30,
    replicate_matlab_precedence: bool = True,
    base_result: TradeEquilibriumResult | None = None,
    verbose: bool = False,
    callback: Callable[[float, TradeEquilibriumResult], None] | None = None,
    *,
    factor_equivalence: bool | None = None,
) -> TradeEquilibriumResult:
    """Solve CGE trade equilibrium along an adaptive homotopy path from base to target tariffs.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    target_tau : np.ndarray
        Target intermediate tariff matrix (tau_1).
    target_tau_fd : np.ndarray
        Target final demand tariff matrix (taufd_1).
    target_tauf : np.ndarray | None, default None
        Target national intermediate tariff rates (nc,). Defaults to zeros.
    target_tauf_fd : np.ndarray | None, default None
        Target national final demand tariff rates (nc,). Defaults to zeros.
    base_tau : np.ndarray | None, default None
        Base intermediate tariff multipliers (tau_0). Defaults to baseline (all 1.0s).
    base_tau_fd : np.ndarray | None, default None
        Base final demand tariff multipliers (taufd_0). Defaults to baseline (all 1.0s).
    base_tauf : np.ndarray | None, default None
        Base national intermediate tariff rates. Defaults to zeros (no national tariff).
    base_tauf_fd : np.ndarray | None, default None
        Base national final demand tariff rates. Defaults to zeros (no national tariff).
    x0 : np.ndarray | None, default None
        Initial equilibrium state at base tariffs. If None, baseline state is solved.
    device : str | None, default None
        Target compute device ('auto', 'cuda', 'cuda:N', 'mps', 'mlx', 'gpu', 'cpu');
        see :func:`solve_trade_equilibrium_gpu` for the float64 policy.
    backend : str | None, default None
        Target backend ('torch', 'mlx', 'numpy' or None for automatic).
    batch_size : int | None, default None
        Macro dimension; must equal ``3*nc - 1`` or ``4*nc - 1`` for the
        calibration. Prefer ``factor_equivalence``.
    initial_step : float, default 0.1
        Initial homotopy step size Delta lambda.
    min_step : float, default 1e-4
        Minimum allowable step size before aborting.
    max_step : float, default 0.5
        Maximum allowable step size Delta lambda.
    tol : float, default 2.5e-3
        Convergence tolerance on maximum absolute equation residual.
    damping : float, default 1e-4
        Levenberg-Marquardt regularized damping.
    max_iter_per_step : int, default 30
        Maximum Newton iterations per continuation step.
    replicate_matlab_precedence : bool, default True
        Whether to replicate MATLAB operator precedence at ff_equi.m:31.
    base_result : TradeEquilibriumResult | None, default None
        Reference equilibrium for postprocessing metrics.
    verbose : bool, default False
        Whether to log continuation steps and convergence diagnostics.
    callback : callable | None, default None
        Optional hook called after each successful continuation step as callback(lambda_val, result).
    factor_equivalence : bool | None, keyword-only, default None
        Reduced (r == w) versus full macro layout; see :func:`solve_trade_equilibrium_gpu`.

    Returns
    -------
    TradeEquilibriumResult
        Fully converged equilibrium result at the target tariff schedule (lambda = 1.0).
    """
    t_start = time.perf_counter()
    nc = calib.n_countries
    ns = calib.n_sectors
    nfd = calib.n_final_demand
    M = ns * nc
    # Validate the device request once; the per-step solver receives the
    # original strings so its float64 policy (auto vs explicit device) applies.
    _resolve_backend_device(device, backend)
    target_backend = backend
    target_device = device

    # Format target tariffs
    t_tau = np.asarray(target_tau, dtype=float)
    if t_tau.ndim == 4:
        t_tau = t_tau.transpose(1, 0, 2, 3).reshape(M, ns, nc)

    t_tau_fd = np.asarray(target_tau_fd, dtype=float)
    if t_tau_fd.ndim == 4:
        t_tau_fd = t_tau_fd.transpose(1, 0, 2, 3).reshape(M, nfd, nc)

    t_tauf = np.zeros(nc, dtype=float) if target_tauf is None else np.asarray(target_tauf, dtype=float).ravel()
    t_tauf_fd = np.zeros(nc, dtype=float) if target_tauf_fd is None else np.asarray(target_tauf_fd, dtype=float).ravel()

    # Format base tariffs
    b_tau = np.ones((M, ns, nc), dtype=float) if base_tau is None else np.asarray(base_tau, dtype=float)
    if b_tau.ndim == 4:
        b_tau = b_tau.transpose(1, 0, 2, 3).reshape(M, ns, nc)

    b_tau_fd = np.ones((M, nfd, nc), dtype=float) if base_tau_fd is None else np.asarray(base_tau_fd, dtype=float)
    if b_tau_fd.ndim == 4:
        b_tau_fd = b_tau_fd.transpose(1, 0, 2, 3).reshape(M, nfd, nc)

    b_tauf = np.zeros(nc, dtype=float) if base_tauf is None else np.asarray(base_tauf, dtype=float).ravel()
    b_tauf_fd = np.zeros(nc, dtype=float) if base_tauf_fd is None else np.asarray(base_tauf_fd, dtype=float).ravel()

    # Current continuation state
    curr_lambda = 0.0
    step_size = initial_step
    curr_x = x0

    # Ensure baseline state is converged if not provided
    if curr_x is None:
        if verbose:
            print("[homotopy] Solving baseline equilibrium at lambda = 0.0 ...")
        res_base = solve_trade_equilibrium_gpu(
            calib=calib,
            tau=b_tau,
            tau_fd=b_tau_fd,
            tauf=b_tauf,
            tauf_fd=b_tauf_fd,
            x0=None,
            device=target_device,
            backend=target_backend,
            batch_size=batch_size,
            factor_equivalence=factor_equivalence,
            max_iter=max_iter_per_step,
            tol=tol,
            damping=damping,
            replicate_matlab_precedence=replicate_matlab_precedence,
            verbose=verbose,
        )
        if not res_base.converged:
            raise RuntimeError("Baseline equilibrium at lambda = 0.0 failed to converge.")
        curr_x = res_base.x_sol
        last_good_result = res_base
    else:
        last_good_result = None

    last_good_x = curr_x.copy()
    step_count = 0

    while curr_lambda < 1.0 - 1e-10:
        step_count += 1
        target_step_lambda = min(curr_lambda + step_size, 1.0)
        actual_delta = target_step_lambda - curr_lambda

        # Linear tariff interpolation along homotopy path
        lam = target_step_lambda
        lam_tau = (1.0 - lam) * b_tau + lam * t_tau
        lam_tau_fd = (1.0 - lam) * b_tau_fd + lam * t_tau_fd
        lam_tauf = (1.0 - lam) * b_tauf + lam * t_tauf
        lam_tauf_fd = (1.0 - lam) * b_tauf_fd + lam * t_tauf_fd

        if verbose:
            print(
                f"[homotopy] Step {step_count:02d}: Attempting lambda = {lam:.4f} "
                f"(step size = {actual_delta:.4f}) ..."
            )

        with warnings.catch_warnings():
            if step_count > 1 or x0 is None:
                # the float32-device notice was already issued by the first solve
                warnings.filterwarnings(
                    "ignore", message="Device .* only supports float32", category=RuntimeWarning
                )
            step_res = solve_trade_equilibrium_gpu(
                calib=calib,
                tau=lam_tau,
                tau_fd=lam_tau_fd,
                tauf=lam_tauf,
                tauf_fd=lam_tauf_fd,
                x0=last_good_x,
                device=target_device,
                backend=target_backend,
                batch_size=batch_size,
                factor_equivalence=factor_equivalence,
                max_iter=max_iter_per_step,
                tol=tol,
                damping=damping,
                replicate_matlab_precedence=replicate_matlab_precedence,
                base_result=base_result,
                verbose=False,
            )

        if step_res.converged and step_res.max_residual <= tol:
            curr_lambda = target_step_lambda
            last_good_x = step_res.x_sol.copy()
            last_good_result = step_res

            if verbose:
                print(
                    f"[homotopy] Step {step_count:02d} SUCCESS: lambda = {curr_lambda:.4f}, "
                    f"iters = {step_res.iterations}, max_res = {step_res.max_residual:.4e}"
                )

            if callback is not None:
                callback(curr_lambda, step_res)

            # Adaptive step enlargement if solved rapidly
            if step_res.iterations <= 4:
                step_size = min(step_size * 1.5, max_step)
            elif step_res.iterations >= max_iter_per_step - 5:
                step_size = max(step_size * 0.75, min_step)
        else:
            # Backtrack and halve step size
            step_size *= 0.5
            if verbose:
                print(
                    f"[homotopy] Step {step_count:02d} FAILED at lambda = {target_step_lambda:.4f} "
                    f"(max_res = {step_res.max_residual:.4e}). Backtracking, new step = {step_size:.4e}"
                )

            if step_size < min_step:
                raise RuntimeError(
                    f"Homotopy continuation failed: step size {step_size:.6e} fell below "
                    f"minimum threshold {min_step:.6e} at lambda = {curr_lambda:.4f}. "
                    f"Last residual: {step_res.max_residual:.4e}."
                )

    if last_good_result is None:
        raise RuntimeError("Homotopy continuation terminated without producing a result.")

    t_total = time.perf_counter() - t_start
    if verbose:
        print(
            f"[homotopy] COMPLETED full path to lambda = 1.0000 in {t_total:.2f}s "
            f"across {step_count} adaptive steps."
        )

    return last_good_result
