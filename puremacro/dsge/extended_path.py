"""Fair & Taylor (1983) Extended Path stochastic simulation for DSGE models.

Simulates non-linear dynamic macroeconomic models under rational expectations
without local perturbation or dynamic programming curse-of-dimensionality by
replacing future stochastic innovations with their conditional expectation
(u_{t+s}^e = 0 for s >= 1), solving the rolling forward deterministic boundary
problem over anticipation horizon T_H at each date t via the SuperLU sparse
stacked Newton-Raphson engine.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np
import pandas as pd

from ._results import ExtendedPathResult
from .build import _Vec
from .perfect_foresight import solve_perfect_foresight

__all__ = ["extended_path", "ExtendedPathResult"]


def extended_path(
    model_or_equations: Any,
    periods: int = 100,
    horizon: int = 100,
    *,
    shocks: np.ndarray | Sequence[Any] | None = None,
    sigma: float | Mapping[str, float] | np.ndarray | None = None,
    seed: int = 0,
    burn: int = 0,
    y_init: np.ndarray | Mapping[str, float] | None = None,
    y_ss: np.ndarray | Mapping[str, float] | None = None,
    tol: float = 1e-8,
    max_iter: int = 50,
    dampening: float = 1.0,
    order: int = 0,
    variable_names: Sequence[str] | None = None,
    shock_names: Sequence[str] | None = None,
    mcp: bool = False,
    mcp_bounds: Mapping[str | int, tuple[float | None, float | None]] | None = None,
    strict: bool = False,
    **kwargs,
) -> ExtendedPathResult:
    r"""Simulate non-linear DSGE model using Fair & Taylor (1983) extended path.

    At each simulation date :math:`t \in \{0, \dots, T-1\}`:
    1. Draw or supply contemporaneous innovation :math:`u_t`.
    2. Formulate the forward deterministic boundary problem over horizon :math:`T_H`,
       setting expected future innovations to zero: :math:`\mathbb{E}_t[u_{t+s}] = 0`
       for :math:`s \ge 1`, and terminal boundary :math:`y_{t+T_H|t} = y_{ss}`.
    3. Solve the stacked system via the sparse SuperLU stacked Newton engine in
       :func:`solve_perfect_foresight`.
    4. Extract contemporaneous endogenous state :math:`y_t = y_{t|t}`, advance
       historical state :math:`y_{t-1} \leftarrow y_t`, and warm-start date :math:`t+1`.

    Parameters
    ----------
    model_or_equations : LinearModel | ParsedModelDAG | Callable
        Solved DSGE model instance or callable residual function
        ``f(y_plus, y_curr, y_lag, shock) -> array_like``.
    periods : int, default 100
        Number of realized simulation periods to return (after ``burn``).
    horizon : int, default 100
        Forward anticipation horizon :math:`T_H` at each simulation date.
    shocks : np.ndarray, optional
        Pre-specified structural shock innovations of shape ``(periods, n_shocks)``
        or ``(periods + burn, n_shocks)``. If provided, random generation is bypassed.
    sigma : float | Mapping[str, float] | np.ndarray, optional
        Standard deviation(s) of structural shocks. Scalar applies to all shocks;
        mapping sets by shock name. Default: 1.0 or model's declared shock covariance.
    seed : int, default 0
        Seed for ``numpy.random.default_rng``.
    burn : int, default 0
        Number of discarded initial burn-in periods.
    y_init : np.ndarray | Mapping[str, float], optional
        Initial predetermined state vector :math:`y_{-1}` at :math:`t=0`.
        Defaults to steady state :math:`y_{ss}`.
    y_ss : np.ndarray | Mapping[str, float], optional
        Terminal steady state vector :math:`y_{ss}`. If None, inferred from model.
    tol : float, default 1e-8
        Convergence tolerance for stacked Newton-Raphson residual infinity-norm.
    max_iter : int, default 50
        Maximum Newton-Raphson iterations allowed per simulation date.
    dampening : float, default 1.0
        Initial Newton step dampening factor in :math:`(0, 1]`.
    order : int, default 0
        Approximation order of future expectations (0 corresponds to standard
        Fair-Taylor certainty equivalence).
    variable_names : Sequence[str], optional
        Names of endogenous variables. Inferred from model when omitted.
    shock_names : Sequence[str], optional
        Names of structural shocks. Inferred from model when omitted.
    mcp : bool, default False
        Whether to enforce mixed complementarity conditions (e.g. ZLB).
    mcp_bounds : Mapping[str | int, tuple[float | None, float | None]], optional
        Bounds for complementarity constraints.
    strict : bool, default False
        If True, raises ``RuntimeError`` if the Newton solver fails to converge
        at any period. If False, marks ``converged=False`` and returns trajectory.
    **kwargs
        Additional keyword arguments forwarded to :func:`solve_perfect_foresight`.

    Returns
    -------
    ExtendedPathResult
        Container holding simulated trajectories, realized shocks, iteration counts,
        residual norms, and presentation export methods.
    """
    # 1. Parameter Validation
    periods = int(periods)
    horizon = int(horizon)
    burn = int(burn)
    max_iter = int(max_iter)
    tol = float(tol)

    if periods < 1:
        raise ValueError(f"periods must be at least 1, got {periods}")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    if burn < 0:
        raise ValueError(f"burn must be non-negative, got {burn}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be an integer >= 1, got {max_iter}")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be a positive finite number, got {tol}")

    # 2. Model & Names Resolution
    model = None
    if (
        hasattr(model_or_equations, "_dynare_equations")
        or hasattr(model_or_equations, "compile_equations")
        or hasattr(model_or_equations, "variables")
    ):
        model = model_or_equations

    if variable_names is not None:
        v_names = tuple(str(v) for v in variable_names)
    elif model is not None and hasattr(model, "variables") and model.variables:
        v_names = tuple(str(v) for v in model.variables)
    elif isinstance(y_ss, Mapping):
        v_names = tuple(str(k) for k in y_ss.keys())
    elif isinstance(y_init, Mapping):
        v_names = tuple(str(k) for k in y_init.keys())
    elif y_ss is not None:
        v_names = tuple(f"y_{i}" for i in range(len(np.asarray(y_ss).ravel())))
    elif y_init is not None:
        v_names = tuple(f"y_{i}" for i in range(len(np.asarray(y_init).ravel())))
    else:
        v_names = ()

    if shock_names is not None:
        s_names = tuple(str(s) for s in shock_names)
    elif model is not None and hasattr(model, "shocks") and model.shocks:
        s_names = tuple(str(s) for s in model.shocks)
    elif shocks is not None:
        sh_arr_tmp = np.asarray(shocks)
        n_s = sh_arr_tmp.shape[1] if sh_arr_tmp.ndim > 1 else 1
        s_names = tuple(f"e_{i+1}" for i in range(n_s))
    else:
        s_names = ("e_1",)

    n_shocks = len(s_names)

    # 3. Pre-Compile Equations Function (one-time compilation)
    if model is not None and hasattr(model, "compile_equations") and hasattr(model, "variables"):
        compiled_raw = model.compile_equations()
        p_dict = dict(getattr(model, "parameter_values", {}) or getattr(model, "_params", {}) or {})
        p_vec = _Vec(list(p_dict.keys()), list(p_dict.values()), what="parameter")
        v_list = list(v_names)
        s_list = list(s_names)

        def precompiled_eq_ast(yp, yc, yl, eps):
            yp_v = _Vec(v_list, yp, what="variable")
            yc_v = _Vec(v_list, yc, what="variable")
            yl_v = _Vec(v_list, yl, what="variable")
            eps_v = _Vec(s_list, np.atleast_1d(eps), what="shock")
            return compiled_raw(yp_v, yc_v, yl_v, eps_v, p_vec)

        eq_fn = precompiled_eq_ast

    elif model is not None and getattr(model, "_dynare_equations", None) is not None:
        dyn_eqs = model._dynare_equations
        p_dict = dict(getattr(model, "_params", {}) or {})
        p_vec = _Vec(list(p_dict.keys()), list(p_dict.values()), what="parameter")
        v_list = list(v_names)
        s_list = list(s_names)

        def precompiled_eq_dyn(yp, yc, yl, eps):
            yp_v = _Vec(v_list, yp, what="variable")
            yc_v = _Vec(v_list, yc, what="variable")
            yl_v = _Vec(v_list, yl, what="variable")
            eps_v = _Vec(s_list, np.atleast_1d(eps), what="shock")
            return dyn_eqs(yp_v, yc_v, yl_v, eps_v, p_vec)

        eq_fn = precompiled_eq_dyn

    elif model is not None and getattr(model, "_A_plus", None) is not None:
        A_plus = model._A_plus
        A_0 = model._A_0
        A_minus = model._A_minus
        B_u = model._B_u

        def precompiled_eq_linear(yp, yc, yl, eps):
            eps_arr = np.atleast_1d(eps)
            return A_plus @ yp + A_0 @ yc + A_minus @ yl + B_u @ eps_arr

        eq_fn = precompiled_eq_linear

    elif callable(model_or_equations):
        eq_fn = model_or_equations
    else:
        raise ValueError("model_or_equations must be a DSGE model or callable residual function")

    # 4. Steady State & Log-Linear Resolution
    if y_ss is not None:
        if isinstance(y_ss, Mapping):
            y_ss_arr = np.array([float(y_ss.get(v, 0.0)) for v in v_names], dtype=float)
        else:
            y_ss_arr = np.asarray(y_ss, dtype=float).ravel()
    elif model is not None:
        if hasattr(model, "steady_state") and model.steady_state is not None:
            if isinstance(model.steady_state, Mapping) or hasattr(model.steady_state, "get"):
                candidate_ss = np.array([float(model.steady_state.get(v, 0.0)) for v in v_names], dtype=float)
            elif hasattr(model.steady_state, "to_dict"):
                ss_d = model.steady_state.to_dict()
                candidate_ss = np.array([float(ss_d.get(v, 0.0)) for v in v_names], dtype=float)
            else:
                candidate_ss = np.asarray(model.steady_state, dtype=float).ravel()
        else:
            candidate_ss = np.zeros(len(v_names), dtype=float)

        zeros = np.zeros(len(v_names), dtype=float)
        zero_shocks = np.zeros(n_shocks, dtype=float)
        with np.errstate(all="ignore"):
            try:
                res_zero = float(np.max(np.abs(eq_fn(zeros, zeros, zeros, zero_shocks))))
            except Exception:
                res_zero = float("inf")

            if res_zero < 1e-6:
                try:
                    res_cand = float(np.max(np.abs(eq_fn(candidate_ss, candidate_ss, candidate_ss, zero_shocks))))
                except Exception:
                    res_cand = float("inf")
                if res_cand > 1e-6:
                    y_ss_arr = zeros
                else:
                    y_ss_arr = candidate_ss
            else:
                y_ss_arr = candidate_ss
    else:
        y_ss_arr = np.zeros(len(v_names) if v_names else 1, dtype=float)

    n_vars = len(y_ss_arr)
    if not v_names:
        v_names = tuple(f"y_{i}" for i in range(n_vars))

    # 5. Initial State Resolution
    if y_init is not None:
        if isinstance(y_init, Mapping):
            y_init_arr = np.array(
                [float(y_init.get(v, y_ss_arr[i])) for i, v in enumerate(v_names)], dtype=float
            )
        else:
            y_init_arr = np.asarray(y_init, dtype=float).ravel()
        if len(y_init_arr) != n_vars:
            raise ValueError(f"y_init dimension ({len(y_init_arr)}) does not match n_vars ({n_vars})")
    else:
        y_init_arr = y_ss_arr.copy()

    # 6. Exogenous Shocks Resolution
    if shocks is not None:
        shocks_in = np.asarray(shocks, dtype=float)
        if shocks_in.ndim == 1:
            if n_shocks == 1:
                shocks_in = shocks_in.reshape(-1, 1)
            elif len(shocks_in) == n_shocks:
                shocks_in = shocks_in.reshape(1, n_shocks)
            else:
                shocks_in = shocks_in.reshape(-1, 1)

        if shocks_in.shape[1] != n_shocks:
            raise ValueError(
                f"shocks column dimension ({shocks_in.shape[1]}) does not match "
                f"model shock dimension ({n_shocks})"
            )

        if shocks_in.shape[0] == periods + burn:
            total_periods = periods + burn
            actual_burn = burn
        elif shocks_in.shape[0] == periods:
            total_periods = periods
            actual_burn = 0
        elif shocks_in.shape[0] > periods:
            total_periods = shocks_in.shape[0]
            actual_burn = total_periods - periods
        else:
            raise ValueError(
                f"shocks row count ({shocks_in.shape[0]}) is less than periods ({periods})"
            )
        all_shocks = shocks_in
    else:
        total_periods = periods + burn
        actual_burn = burn
        rng = np.random.default_rng(seed)

        if sigma == 0.0 or sigma == 0:
            all_shocks = np.zeros((total_periods, n_shocks), dtype=float)
        elif sigma is None:
            if model is not None and hasattr(model, "_shock_cov") and model._shock_cov is not None:
                cov = model._shock_covariance(None) if hasattr(model, "_shock_covariance") else model._shock_cov
                all_shocks = (
                    rng.multivariate_normal(np.zeros(n_shocks), cov, size=total_periods, method="cholesky")
                    if n_shocks
                    else np.zeros((total_periods, 0))
                )
            elif model is not None and hasattr(model, "_shock_sd"):
                sd = model._shock_sd(None, missing=1.0)
                all_shocks = rng.standard_normal((total_periods, n_shocks)) * sd
            else:
                all_shocks = rng.standard_normal((total_periods, n_shocks))
        elif isinstance(sigma, (int, float, np.floating)):
            all_shocks = rng.standard_normal((total_periods, n_shocks)) * float(sigma)
        elif isinstance(sigma, Mapping):
            sd = np.array([float(sigma.get(s, 0.0)) for s in s_names], dtype=float)
            all_shocks = rng.standard_normal((total_periods, n_shocks)) * sd
        elif isinstance(sigma, np.ndarray):
            if sigma.ndim == 1:
                all_shocks = rng.standard_normal((total_periods, n_shocks)) * sigma
            elif sigma.ndim == 2:
                all_shocks = rng.multivariate_normal(np.zeros(n_shocks), sigma, size=total_periods, method="cholesky")
            else:
                raise ValueError(f"Invalid sigma shape: {sigma.shape}")
        else:
            all_shocks = rng.standard_normal((total_periods, n_shocks))

    # 7. Rolling Stacked Newton Simulation Loop
    realized_path = np.zeros((total_periods, n_vars), dtype=float)
    iterations_list: list[int] = []
    max_res_norm = 0.0
    max_term_err = 0.0
    overall_converged = True

    current_state = y_init_arr.copy()
    warm_guess = np.tile(y_ss_arr, (horizon, 1))

    kwargs_pf = kwargs.copy()
    user_y_end = kwargs_pf.pop("y_end", None)

    companion_m = None
    if (
        user_y_end is None
        and model is not None
        and hasattr(model, "solution")
        and hasattr(model, "_reported_loadings")
        and np.allclose(y_ss_arr, 0.0)
    ):
        try:
            G_ep = model.solution.G
            N_ep = model.solution.N
            Mx_ep, Mu_ep = model._reported_loadings()
            order_vars_ep = list(model.states) + list(model.controls)
            v_idx_ep = [order_vars_ep.index(v) for v in model.variables]
            if y_init is not None:
                x_ep = np.array([y_init_arr[list(model.variables).index(s)] for s in model.states], dtype=float)
            else:
                x_ep = np.zeros(model.n_states, dtype=float)
            companion_m = (G_ep, N_ep, Mx_ep, v_idx_ep)
        except Exception:
            companion_m = None

    for t in range(total_periods):
        u_t = all_shocks[t]
        forward_exo = np.zeros((horizon, n_shocks), dtype=float)
        forward_exo[0] = u_t

        if user_y_end is not None:
            y_end_t = user_y_end
        elif companion_m is not None:
            try:
                G_ep, N_ep, Mx_ep, v_idx_ep = companion_m
                x_ep_next = G_ep @ x_ep + N_ep @ u_t
                if horizon > 1:
                    x_term = np.linalg.matrix_power(G_ep, horizon - 1) @ x_ep_next
                else:
                    x_term = x_ep_next
                y_end_t = (Mx_ep @ x_term)[v_idx_ep]
                x_ep = x_ep_next
            except Exception:
                y_end_t = None
        else:
            y_end_t = None

        res_t = solve_perfect_foresight(
            model_or_equations=None,
            y_init=current_state,
            y_ss=y_ss_arr,
            y_end=y_end_t,
            exogenous_path=forward_exo,
            periods=horizon,
            initial_path=warm_guess,
            tol=tol,
            max_iter=max_iter,
            dampening=dampening,
            mcp=mcp,
            mcp_bounds=mcp_bounds,
            variable_names=v_names,
            equations_fn=eq_fn,
            **kwargs_pf,
        )

        y_t = res_t.path.iloc[0].to_numpy()
        realized_path[t] = y_t
        current_state = y_t.copy()

        # Warm-start next step by shifting solved path forward by 1 period
        if horizon > 1:
            warm_guess = np.empty((horizon, n_vars), dtype=float)
            warm_guess[:-1] = res_t.path.to_numpy()[1:]
            warm_guess[-1] = y_ss_arr
        else:
            warm_guess = np.tile(y_ss_arr, (1, 1))

        iter_t = int(res_t.iterations)
        iterations_list.append(iter_t)
        max_res_norm = max(max_res_norm, float(res_t.residual_norm))

        # Terminal boundary error of the stacked system at horizon T_H
        lead_val = y_end_t if y_end_t is not None else y_ss_arr
        if horizon > 1:
            yl_bnd = res_t.path.iloc[-2].to_numpy()
            yc_bnd = res_t.path.iloc[-1].to_numpy()
            bnd_res = float(np.max(np.abs(eq_fn(lead_val, yc_bnd, yl_bnd, np.zeros(n_shocks)))))
        else:
            bnd_res = float(np.max(np.abs(eq_fn(lead_val, res_t.path.iloc[0].to_numpy(), current_state, u_t))))
        max_term_err = max(max_term_err, bnd_res)

        if not res_t.converged:
            overall_converged = False
            if strict:
                raise RuntimeError(
                    f"Extended path solver failed to converge at period {t+1} "
                    f"(iterations={iter_t}, residual_norm={res_t.residual_norm:.4e})"
                )

    # 8. Post-Burn Slicing & Result Formulation
    path_out = realized_path[actual_burn : actual_burn + periods]
    shocks_out = all_shocks[actual_burn : actual_burn + periods]
    iterations_out = iterations_list[actual_burn : actual_burn + periods]

    path_df = pd.DataFrame(
        path_out,
        index=pd.RangeIndex(1, periods + 1, name="t"),
        columns=list(v_names),
    )
    shocks_df = pd.DataFrame(
        shocks_out,
        index=pd.RangeIndex(1, periods + 1, name="t"),
        columns=list(s_names),
    )

    return ExtendedPathResult(
        path=path_df,
        shocks=shocks_df,
        converged=overall_converged,
        iterations=iterations_out,
        residual_norm=max_res_norm,
        terminal_error=max_term_err,
        variable_names=v_names,
        shock_names=s_names,
        horizon=horizon,
    )
