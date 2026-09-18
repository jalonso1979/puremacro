"""High-Performance GPU-Accelerated CGE Solver for puremacro.trade.

Implements a dual-device PyTorch (NVIDIA CUDA / Apple Silicon MPS), Apple MLX and
NumPy general equilibrium solver using Levenberg-Marquardt regularized normal equations:

    (J^T J + mu * I) Delta x_m = -J^T F_m

combined with two-sided diagonal equilibration (Jacobi scaling) and backtracking
Armijo line search on the merit function Phi(x_m) = 1/2 ||F_m(x_m)||_2^2.

Precision policy
----------------
The macro Jacobian is assembled from one-sided finite differences, which are only
usable in float64. Jacobians are therefore evaluated in float64 wherever the device
supports it (PyTorch on CPU/CUDA, the MLX CPU stream). When ``device='auto'`` selects
a float32-only device (Apple MPS, MLX GPU), the solver transparently uses the float64
host evaluator of the same backend instead. When a float32 device is requested
explicitly (``device='mps'`` / ``'mlx'``), a RuntimeWarning is emitted, the device is
used only for the coarse phase (residual >= 5e6 in the first iterations) and the
Jacobian is polished in float64 on the host; the devices actually used are recorded
in ``result.metadata['jacobian_evaluations']``.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import time
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.linalg as la

from puremacro.trade.gpu.backend import (
    _device_info,
    _load_torch,
    _resolve_backend_device,
    get_memory_usage,
    reset_peak_memory,
)
from puremacro.trade.gpu.batched_jacobian import BatchedJacobianEvaluator
from puremacro.trade.postprocessing import postprocess_trade_equilibrium
from puremacro.trade.solver import _resolve_tariffs

if TYPE_CHECKING:
    from puremacro.trade._results import TradeCalibrationResult, TradeEquilibriumResult

#: Residual level above which an explicitly requested float32 device is used for the
#: coarse phase of the iteration; below it the Jacobian is polished in float64.
_COARSE_RESIDUAL_THRESHOLD = 5e6
_COARSE_MAX_ITER = 5


def _jacobian_devices(
    requested_backend: str,
    requested_device: str,
    explicit_device: bool,
) -> tuple[tuple[str, str], tuple[str, str] | None]:
    """Choose the float64 (polish) evaluator and, if any, the float32 coarse evaluator.

    Returns ``((polish_backend, polish_device), coarse)`` where ``coarse`` is
    ``None`` unless the user explicitly requested a float32-only device.
    """
    info = _device_info(requested_backend, requested_device)
    if info.supports_float64:
        return (requested_backend, requested_device), None

    host = ("torch", "cpu") if requested_backend == "torch" else ("mlx", "cpu")
    if not explicit_device:
        # auto-selected float32 device: finite differences need float64
        return host, None

    warnings.warn(
        f"Device {requested_backend}:{requested_device} only supports float32; finite-difference "
        "Jacobians are unusable in float32, so it is used for the coarse phase only and the "
        f"Jacobian is polished in float64 on {host[0]}:{host[1]} "
        "(see result.metadata['jacobian_evaluations']).",
        RuntimeWarning,
        stacklevel=3,
    )
    return host, (requested_backend, requested_device)


def solve_trade_equilibrium_gpu(
    calib: TradeCalibrationResult,
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    tauf: np.ndarray | None = None,
    tauf_fd: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    device: str | None = None,
    backend: str | None = None,
    batch_size: int | None = None,
    max_iter: int = 50,
    tol: float = 2.5e-3,
    damping: float = 1e-4,
    ad_mode: Literal["finite_diff", "forward", "vjp"] = "finite_diff",
    replicate_matlab_precedence: bool = True,
    base_result: TradeEquilibriumResult | None = None,
    verbose: bool = False,
    *,
    factor_equivalence: bool | None = None,
) -> TradeEquilibriumResult:
    """Solve multi-country multi-sector CGE trade equilibrium using GPU acceleration.

    Tariff arguments follow exactly the conventions of
    :func:`puremacro.trade.solver.solve_trade_equilibrium`: the bilateral
    multipliers ``tau`` / ``tau_fd`` are gross rates (``1 + rate``, default all
    ones) and the national vectors ``tauf`` / ``tauf_fd`` are net rates (default
    all zeros), so the default call solves the same baseline as the NumPy solver.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    tau : np.ndarray | None, default None
        Intermediate tariff multipliers (1 + rate), shape (ns*nc, ns, nc), (ns, nc, ns, nc)
        or (nc,). If None, defaults to the tariff-free baseline (all ones).
    tau_fd : np.ndarray | None, default None
        Final demand tariff multipliers (1 + rate), shape (ns*nc, nfd, nc), (ns, nc, nfd, nc)
        or (nc,). If None, defaults to the tariff-free baseline (all ones).
    tauf : np.ndarray | None, default None
        National intermediate tariff rates (nc,). If None, defaults to zeros.
    tauf_fd : np.ndarray | None, default None
        National final demand tariff rates (nc,). If None, defaults to zeros.
    x0 : np.ndarray | None, default None
        Initial guess: the full state vector (length ``2*ns*nc + 4*nc - 1``) or a
        macro vector of the reduced (``3*nc - 1``) or full (``4*nc - 1``) layout.
        If None, constructed from calibrated baseline endowments and transfers.
    device : str | None, default None
        Compute device ('auto', 'cuda', 'cuda:N', 'mps', 'mlx', 'gpu', 'cpu').
        If None, automatically detected; see the module docstring for the
        float64 policy applied to float32-only devices.
    backend : str | None, default None
        Compute backend ('torch', 'mlx', 'numpy' or None for automatic).
    batch_size : int | None, default None
        Macro dimension; must equal ``3*nc - 1`` (reduced layout, r == w) or
        ``4*nc - 1`` (full layout) for the calibration (230 / 307 for the canonical
        77 countries). Kept for backward compatibility; prefer ``factor_equivalence``.
    max_iter : int, default 50
        Maximum Levenberg-Marquardt iterations.
    tol : float, default 2.5e-3
        Infinity-norm convergence tolerance: max(|F(x)|) <= tol.
    damping : float, default 1e-4
        Initial Levenberg-Marquardt regularized damping parameter mu.
    ad_mode : str, default 'finite_diff'
        Differentiation mode ('finite_diff', 'forward', 'vjp'); see
        :meth:`BatchedJacobianEvaluator.evaluate_batched_jacobian`.
    replicate_matlab_precedence : bool, default True
        Whether to replicate MATLAB operator precedence at ff_equi.m:31.
    base_result : TradeEquilibriumResult | None, default None
        Baseline equilibrium for terms-of-trade and CPI deflation calculations.
    verbose : bool, default False
        Whether to print per-iteration convergence diagnostics.
    factor_equivalence : bool | None, keyword-only, default None
        True imposes r == w (reduced macro layout), which is verified to be exact
        for the calibration (uniform capital shares across a country's sectors)
        and raises ValueError otherwise; False solves the full layout with
        separate capital-market clearing. None selects the reduced layout iff it
        is exact.

    Returns
    -------
    TradeEquilibriumResult
        Fully post-processed trade equilibrium dataclass container. The metadata
        records ``device`` / ``backend`` actually used for the Jacobian, the
        requested pair, and ``jacobian_evaluations`` (counts by backend, device
        and dtype).
    """
    t_start = time.perf_counter()

    requested_backend, requested_device = _resolve_backend_device(device, backend)
    explicit_device = device is not None and str(device).lower().strip() not in ("auto", "")
    (polish_backend, polish_device), coarse = _jacobian_devices(
        requested_backend, requested_device, explicit_device
    )
    reset_peak_memory(polish_device, polish_backend)

    # Tariffs: identical defaults and shape handling to the NumPy solver
    tau_arr = None if tau is None else np.asarray(tau, dtype=float)
    tau_fd_arr = None if tau_fd is None else np.asarray(tau_fd, dtype=float)
    tau_a, taufd_a, tauf_vec, tauf_fd_vec = _resolve_tariffs(
        calib, tau=tau_arr, tau_fd=tau_fd_arr, tauf=tauf, tauf_fd=tauf_fd
    )

    # Float64 evaluator (always used for the residuals and the polished Jacobian)
    evaluator = BatchedJacobianEvaluator(
        calib=calib,
        tau_a=tau_a,
        taufd_a=taufd_a,
        tauf_vec=tauf_vec,
        tauf_fd_vec=tauf_fd_vec,
        device=polish_device,
        backend=polish_backend,
        batch_size=batch_size,
        factor_equivalence=factor_equivalence,
        replicate_matlab_precedence=replicate_matlab_precedence,
    )

    # Optional float32 evaluator for the coarse phase of an explicitly requested device
    evaluator_coarse = None
    coarse_stream = None
    if coarse is not None:
        if coarse[0] == "mlx":
            evaluator_coarse = evaluator  # same arrays, GPU stream
            coarse_stream = "gpu"
        else:
            evaluator_coarse = BatchedJacobianEvaluator(
                calib=calib,
                tau_a=tau_a,
                taufd_a=taufd_a,
                tauf_vec=tauf_vec,
                tauf_fd_vec=tauf_fd_vec,
                device=coarse[1],
                backend=coarse[0],
                batch_size=batch_size,
                factor_equivalence=evaluator.factor_equivalence,
                replicate_matlab_precedence=replicate_matlab_precedence,
            )

    nc = calib.n_countries
    xm = evaluator._macro_from_x0(x0)
    n_m = len(xm)
    n_factor_vars = evaluator.n_factor_vars
    torch = _load_torch() if polish_backend == "torch" else None
    jac_log: Counter[str] = Counter()

    # Initial residual evaluation
    f_m, f_full, p_sol, p_sol_3d, y_sol = evaluator.eval_macro_single(xm, compute_full=True)
    res_max = float(np.max(np.abs(f_m)))
    diff = float(np.sum(np.abs(f_m)))
    iters = 0
    conv = res_max <= tol

    if conv:
        if verbose:
            print(f"[solve_trade_equilibrium_gpu] Initial point already converged: max_res = {res_max:.4e}")
    else:
        mu = damping

        for it in range(max_iter):
            iters = it + 1
            res_max = float(np.max(np.abs(f_m)))

            if res_max <= tol:
                conv = True
                break

            # Evaluate batched parallel Jacobian: float32 device only in the coarse phase
            use_coarse = (
                evaluator_coarse is not None
                and res_max >= _COARSE_RESIDUAL_THRESHOLD
                and it < _COARSE_MAX_ITER
            )
            if use_coarse:
                J = evaluator_coarse.evaluate_batched_jacobian(
                    xm, f_base=f_m, ad_mode=ad_mode, stream=coarse_stream
                )
                jac_log[f"{coarse[0]}:{coarse[1]}:float32"] += 1
            else:
                J = evaluator.evaluate_batched_jacobian(xm, f_base=f_m, ad_mode=ad_mode, stream="cpu" if polish_backend == "mlx" else None)
                jac_log[f"{polish_backend}:{polish_device}:float64"] += 1

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

            # Levenberg-Marquardt normal equations solve
            JT_J = J_equil.T @ J_equil
            JT_rhs = J_equil.T @ rhs_equil
            sol_u = None
            if torch is not None:
                try:
                    torch_dev = polish_device if (polish_device == "cpu" or polish_device.startswith("cuda")) else "cpu"
                    A_t = torch.as_tensor(JT_J + mu * np.eye(n_m), device=torch_dev, dtype=torch.float64)
                    b_t = torch.as_tensor(JT_rhs[:, None], device=torch_dev, dtype=torch.float64)
                    sol_t = torch.linalg.solve(A_t, b_t)
                    sol_u = sol_t.cpu().numpy().ravel()
                except Exception:
                    sol_u = None
            if sol_u is None:
                try:
                    sol_u = la.solve(JT_J + mu * np.eye(n_m), JT_rhs)
                except Exception:
                    sol_u = la.lstsq(J_equil, rhs_equil, rcond=1e-12)[0]

            delta_m = D_R * sol_u

            # Uniform step size clamping on factor prices to prevent exponent explosion
            max_factor_step = float(np.max(np.abs(delta_m[:n_factor_vars])))
            if max_factor_step > 0.3:
                delta_m *= (0.3 / max_factor_step)

            # Backtracking Armijo line search on merit function ||D_L * f_m||_2
            alpha_step = 1.0
            norm_0 = res_max
            norm_2_0 = float(np.linalg.norm(D_L * f_m))

            xm_trial = xm + alpha_step * delta_m
            xm_trial[:n_factor_vars] = np.clip(xm_trial[:n_factor_vars], -5.0, 5.0)
            f_trial_m, f_trial_full, p_trial, p_trial_3d, y_trial = evaluator.eval_macro_single(
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
                f_trial_m, f_trial_full, p_trial, p_trial_3d, y_trial = evaluator.eval_macro_single(
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
                    f"[{polish_backend}:{polish_device}] Iter {it + 1:02d}: "
                    f"max_res = {np.max(np.abs(f_m)):.4e}, "
                    f"step = {alpha_step:.4f}, mu = {mu:.2e}"
                )

    # Reconstruct full general equilibrium state vector x_full in double precision
    x_full = evaluator._full_state(xm, p_sol, y_sol)

    # Final residual check
    _, f_full_final, _, _, _ = evaluator.eval_macro_single(xm, compute_full=True)
    res_final = f_full_final if f_full_final is not None else f_m
    max_res = float(np.max(np.abs(res_final)))
    diff = float(np.sum(np.abs(res_final)))
    conv = bool(max_res <= tol)

    t_end = time.perf_counter()
    mem_info = get_memory_usage(polish_device, polish_backend)

    res_dataclass = postprocess_trade_equilibrium(
        x_sol=x_full,
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
            "method": "gpu_accelerated",
            "device": polish_device,
            "backend": polish_backend,
            "requested_device": requested_device,
            "requested_backend": requested_backend,
            "jacobian_device": f"{polish_backend}:{polish_device}",
            "jacobian_dtype": "float64",
            "jacobian_evaluations": dict(jac_log),
            "batch_size": evaluator.batch_size,
            "factor_equivalence": evaluator.factor_equivalence,
            "ad_mode": ad_mode,
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
