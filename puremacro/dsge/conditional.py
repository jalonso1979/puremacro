"""Waggoner & Zha (1999) conditional forecasting for linear DSGE models.

Enforces target paths on selected endogenous variables by inverting designated
structural shocks. Uncontrolled structural shocks remain pinned at zero (or vary
stochastically around zero without violating target conditions).
"""
from __future__ import annotations

import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from ._results import ConditionalForecastResult

__all__ = ["conditional_forecast", "ConditionalForecastResult"]


def conditional_forecast(
    model: Any,
    conditions: Mapping[str, Sequence[float | None]] | None = None,
    *,
    target_paths: Mapping[str, Sequence[float | None]] | None = None,
    horizon: int | None = None,
    controlled_shocks: Sequence[str] | None = None,
    x0: np.ndarray | Mapping[str, float] | None = None,
    shock_cov: np.ndarray | None = None,
    method: str = "covariance_weighted",
    exact: bool = True,
    ci: float | Sequence[float] | None = None,
    n_sims: int = 0,
    seed: int = 0,
) -> ConditionalForecastResult:
    """Compute conditional forecast using Waggoner & Zha (1999) shock inversion.

    Parameters
    ----------
    model : LinearModel
        Solved first-order DSGE model.
    conditions : Mapping[str, Sequence[float | None]] or None
        Target paths for endogenous variables. Each value is a sequence of target
        values starting at horizon h=1. Use None or NaN for unconstrained periods.
    target_paths : Mapping[str, Sequence[float | None]] or None
        Alias for conditions.
    horizon : int, optional
        Forecast horizon. Defaults to 12 or maximum condition length.
    controlled_shocks : Sequence[str], optional
        Shocks allowed to adjust to hit target paths. If None, defaults to all
        shocks declared in the model.
    x0 : ndarray or dict, optional
        Initial state deviation at t=0. Defaults to 0 (steady state).
    shock_cov : ndarray, optional
        Shock covariance matrix. If None, uses model._shock_cov or identity.
    method : {"covariance_weighted", "minimum_energy"}
        Inversion method. "covariance_weighted" weights shocks by Sigma_U;
        "minimum_energy" minimizes Euclidean 2-norm.
    exact : bool, default True
        If True, raises ValueError when constraints cannot be met exactly.
        If False, solves least-squares and emits warning.
    ci : float or tuple of float, optional
        Confidence level(s) for analytical bands (e.g. 0.90 or (0.68, 0.95)).
    n_sims : int, default 0
        Number of stochastic simulation replications. If > 0, draws Monte Carlo
        paths consistent with conditions.
    seed : int, default 0
        Random seed for stochastic simulation.

    Returns
    -------
    ConditionalForecastResult
    """
    # 1. Resolve conditions alias
    if conditions is None:
        conditions = target_paths if target_paths is not None else {}
    cond_dict: dict[str, list[float | None]] = {}
    for k, v in conditions.items():
        cond_dict[str(k)] = [None if x is None or np.isnan(x) else float(x) for x in v]

    # Count total active restrictions
    active_restrs = []
    for var_name, path in cond_dict.items():
        for h_idx, target_val in enumerate(path):
            if target_val is not None:
                active_restrs.append((var_name, h_idx + 1, target_val))

    # 2. Determine horizon
    max_cond_h = max([h for _, h, _ in active_restrs], default=1)
    if horizon is None:
        H = max(12, max_cond_h)
    else:
        H = int(horizon)
        if H < max_cond_h:
            raise ValueError(f"horizon={H} is shorter than the maximum target horizon {max_cond_h}")

    # 3. Model variables and shocks
    model_vars = list(getattr(model, "variables", []))
    model_shocks = list(getattr(model, "shocks", []))
    n_v = len(model_vars)
    n_u = len(model_shocks)

    for v, _, _ in active_restrs:
        if v not in model_vars:
            raise KeyError(f"Target variable '{v}' not declared in model.variables: {model_vars}")

    # 4. Resolve controlled shocks
    if controlled_shocks is None:
        ctrl_shocks = list(model_shocks)
    else:
        ctrl_shocks = list(controlled_shocks)

    if len(ctrl_shocks) == 0 and len(active_restrs) > 0:
        raise ValueError("Cannot enforce target conditions with empty controlled_shocks")

    for s in ctrl_shocks:
        if s not in model_shocks:
            raise KeyError(f"Controlled shock '{s}' not in model.shocks: {model_shocks}")

    ctrl_idx = [model_shocks.index(s) for s in ctrl_shocks]
    n_ctrl = len(ctrl_idx)

    # 5. Extract state space representation: G, N, C, D
    if hasattr(model, "_dynare_companion"):
        G, N, C, D = model._dynare_companion()
    else:
        # Fallback for standard LinearModel
        G = np.asarray(model.solution.G, dtype=float)
        N = np.asarray(model.solution.N, dtype=float)
        # Observation matrices
        n_s = len(model.states)
        C = np.zeros((n_v, n_s))
        D = np.zeros((n_v, n_u))
        for i, v in enumerate(model_vars):
            if v in model.states:
                C[i, model.states.index(v)] = 1.0
        # If controls have policy rules
        F = np.asarray(model.solution.F, dtype=float)
        L = np.asarray(model.solution.L, dtype=float)
        for i, v in enumerate(model_vars):
            if v in model.controls:
                c_idx = model.controls.index(v)
                C[i, :] = F[c_idx, :]
                D[i, :] = L[c_idx, :]

    n_s = G.shape[0]

    # Initial state s0
    if x0 is None:
        s0 = np.zeros(n_s)
    elif isinstance(x0, Mapping):
        s0 = np.zeros(n_s)
        for st_name, val in x0.items():
            if st_name in model.states:
                s0[model.states.index(st_name)] = float(val)
    elif isinstance(x0, np.ndarray):
        s0 = np.asarray(x0, dtype=float).flatten()
    else:
        raise TypeError(f"Unsupported type for x0: {type(x0).__name__}")

    # Steady state vector
    ss_series = getattr(model, "steady_state", None)
    if ss_series is not None:
        ss_vec = np.array([float(ss_series[v]) for v in model_vars])
    else:
        ss_vec = np.zeros(n_v)

    # 6. Compute impulse response matrices Psi_k (k = 0, ..., H-1)
    Psi = [D.copy()]
    CG = C.copy()
    for k in range(1, H):
        Psi.append(CG @ N)
        CG = CG @ G

    # 7. Compute baseline trajectory (zero future shocks)
    baseline_arr = np.zeros((H, n_v))
    s_curr = s0.copy()
    for h in range(H):
        baseline_arr[h, :] = ss_vec + C @ s_curr
        s_curr = G @ s_curr

    # 8. Assemble constraint matrix A and deviation vector b
    K = len(active_restrs)
    if K == 0:
        # Unconditional forecast
        forecast_df = pd.DataFrame(
            baseline_arr,
            index=pd.RangeIndex(1, H + 1, name="horizon"),
            columns=model_vars,
        )
        shocks_df = pd.DataFrame(
            np.zeros((H, n_u)),
            index=pd.RangeIndex(1, H + 1, name="horizon"),
            columns=model_shocks,
        )
        return ConditionalForecastResult(
            forecast=forecast_df,
            baseline=forecast_df.copy(),
            shocks=shocks_df,
            conditions=cond_dict,
            controlled_shocks=tuple(ctrl_shocks),
            bands=None,
            variable_names=tuple(model_vars),
            shock_names=tuple(model_shocks),
            horizon=H,
        )

    A = np.zeros((K, H * n_ctrl))
    b = np.zeros(K)

    for row_idx, (v_name, h_val, target_val) in enumerate(active_restrs):
        v_idx = model_vars.index(v_name)
        b[row_idx] = target_val - baseline_arr[h_val - 1, v_idx]
        # Impacts from shocks in periods s = 1 ... h_val
        for s in range(1, h_val + 1):
            lag = h_val - s
            # Loading of shock on variable v_name at lag (h_val - s)
            psi_block = Psi[lag][v_idx, ctrl_idx]
            col_start = (s - 1) * n_ctrl
            col_end = s * n_ctrl
            A[row_idx, col_start:col_end] = psi_block

    # 9. Diagnostics on restriction matrix A
    # Check for zero rows (zero impact of controlled shocks on target)
    for row_idx, (v_name, h_val, _) in enumerate(active_restrs):
        row_norm = np.linalg.norm(A[row_idx, :])
        if row_norm < 1e-12:
            raise ValueError(
                f"Controlled shocks {ctrl_shocks} have zero impact on target variable '{v_name}' at horizon {h_val}."
            )

    # Singular value decomposition
    sv = np.linalg.svdvals(A)
    rank_A = int(np.sum(sv > 1e-11))

    pinv_A = np.linalg.pinv(A, rcond=1e-12)
    proj_err = float(np.max(np.abs(b - A @ (pinv_A @ b))))

    if rank_A < K or K > H * n_ctrl:
        if proj_err > 1e-7:
            if exact:
                raise ValueError(
                    f"Target path cannot be achieved with selected shocks: "
                    f"system has rank {rank_A} < {K} restrictions, residual error {proj_err:.3e}."
                )
            else:
                warnings.warn(
                    f"Target conditions cannot be met exactly (rank {rank_A} < {K}); "
                    f"returning least-squares approximation (error {proj_err:.3e}).",
                    UserWarning,
                )

    # 10. Shock covariance Sigma_U
    if shock_cov is not None:
        sigma_mat = np.asarray(shock_cov, dtype=float)
    elif getattr(model, "_shock_cov", None) is not None:
        sigma_mat = np.asarray(model._shock_cov, dtype=float)
    else:
        sigma_mat = np.eye(n_u)

    sigma_ctrl = sigma_mat[np.ix_(ctrl_idx, ctrl_idx)]

    # 11. Shock inversion
    if method == "covariance_weighted":
        # W = Sigma_U @ A^T
        # Block-wise multiplication avoiding large (H*n_ctrl x H*n_ctrl) dense Kronecker
        W = np.zeros((H * n_ctrl, K))
        for s in range(H):
            row_st = s * n_ctrl
            row_en = (s + 1) * n_ctrl
            W[row_st:row_en, :] = sigma_ctrl @ A[:, row_st:row_en].T

        AW = A @ W  # K x K
        try:
            AW_pinv = np.linalg.pinv(AW, rcond=1e-12)
            lambda_vec = AW_pinv @ b
            u_star = W @ lambda_vec
            res_norm = float(np.max(np.abs(A @ u_star - b)))
            if res_norm > 1e-7 and exact and proj_err <= 1e-7:
                # Fall back to minimum energy pinv
                u_star = pinv_A @ b
        except np.linalg.LinAlgError:
            u_star = pinv_A @ b
    else:
        u_star = pinv_A @ b

    final_res = float(np.max(np.abs(A @ u_star - b)))
    if final_res > 1e-7 and exact:
        raise ValueError(
            f"Target path could not be satisfied to required precision (max residual {final_res:.3e})."
        )

    # 12. Full structural shock matrix (H, n_u)
    U_ctrl_mat = u_star.reshape(H, n_ctrl)
    full_shocks = np.zeros((H, n_u))
    for i, c in enumerate(ctrl_idx):
        full_shocks[:, c] = U_ctrl_mat[:, i]

    # 13. Forward simulate realized forecast
    forecast_arr = np.zeros((H, n_v))
    s_curr = s0.copy()
    for h in range(H):
        u_h = full_shocks[h, :]
        s_curr = G @ s_curr + N @ u_h
        forecast_arr[h, :] = ss_vec + C @ s_curr
        # Adjust for direct contemporaneous shock impact D u_h
        # Notice: under Dynare timing s_h = G s_{h-1} + N u_h, and v_h = ss + C s_{h-1} + D u_h
        # But if G, N, C, D companion form is:
        # s_h = G s_{h-1} + N u_h
        # v_h = ss + C s_h (if state contains contemporaneous values) or v_h = ss + C s_{h-1} + D u_h
    # Let's compute directly from moving average representation:
    # v_h = baseline_h + sum_{s=1}^h Psi_{h-s} u_s
    forecast_arr = baseline_arr.copy()
    for h in range(1, H + 1):
        for s in range(1, h + 1):
            lag = h - s
            forecast_arr[h - 1, :] += Psi[lag] @ full_shocks[s - 1, :]

    forecast_df = pd.DataFrame(
        forecast_arr,
        index=pd.RangeIndex(1, H + 1, name="horizon"),
        columns=model_vars,
    )
    baseline_df = pd.DataFrame(
        baseline_arr,
        index=pd.RangeIndex(1, H + 1, name="horizon"),
        columns=model_vars,
    )
    shocks_df = pd.DataFrame(
        full_shocks,
        index=pd.RangeIndex(1, H + 1, name="horizon"),
        columns=model_shocks,
    )

    # 14. Analytical credible bands & stochastic simulation
    bands_dict: dict[str, pd.DataFrame] | None = None
    if ci is not None or n_sims > 0:
        # Full dynamic loading matrix R_full: (H * n_v, H * n_u)
        R_full = np.zeros((H * n_v, H * n_u))
        for h in range(1, H + 1):
            row_st = (h - 1) * n_v
            row_en = h * n_v
            for s in range(1, h + 1):
                col_st = (s - 1) * n_u
                col_en = s * n_u
                R_full[row_st:row_en, col_st:col_en] = Psi[h - s]

        # Prior covariance Sigma_total: H * n_u x H * n_u
        Sigma_total = np.kron(np.eye(H), sigma_mat)

        # Build full restriction matrix A_full: K x (H * n_u)
        A_full = np.zeros((K, H * n_u))
        for row_idx, (v_name, h_val, _) in enumerate(active_restrs):
            v_idx = model_vars.index(v_name)
            for s in range(1, h_val + 1):
                lag = h_val - s
                A_full[row_idx, (s - 1) * n_u : s * n_u] = Psi[lag][v_idx, :]

        # Waggoner-Zha (1999) Theorem 1:
        # Sigma_{U|cond} = Sigma_total - Sigma_total A_full^T (A_full Sigma_total A_full^T)^{-1} A_full Sigma_total
        W_full = Sigma_total @ A_full.T
        M_full = A_full @ W_full
        M_pinv = np.linalg.pinv(M_full, rcond=1e-12)
        Sigma_U_cond = Sigma_total - W_full @ M_pinv @ W_full.T
        Sigma_U_cond = 0.5 * (Sigma_U_cond + Sigma_U_cond.T)

        # Variance of endogenous variables:
        # Var(V | cond) = R_full @ Sigma_{U|cond} @ R_full^T
        Var_V_cond = R_full @ Sigma_U_cond @ R_full.T
        diag_var = np.maximum(0.0, np.diag(Var_V_cond))
        sd_V_cond = np.sqrt(diag_var).reshape(H, n_v)

        # At constrained points, variance must be identically 0.0
        for v_name, h_val, _ in active_restrs:
            v_idx = model_vars.index(v_name)
            sd_V_cond[h_val - 1, v_idx] = 0.0

        bands_dict = {}
        ci_levels = [ci] if isinstance(ci, (int, float)) else (list(ci) if ci is not None else [0.68, 0.90, 0.95])
        for level in ci_levels:
            lvl = float(level)
            alpha_tail = (1.0 - lvl) / 2.0
            z_crit = float(norm.ppf(1.0 - alpha_tail))
            tag = f"{int(round(lvl * 100))}"
            low_df = pd.DataFrame(
                forecast_arr - z_crit * sd_V_cond,
                index=pd.RangeIndex(1, H + 1, name="horizon"),
                columns=model_vars,
            )
            up_df = pd.DataFrame(
                forecast_arr + z_crit * sd_V_cond,
                index=pd.RangeIndex(1, H + 1, name="horizon"),
                columns=model_vars,
            )
            bands_dict[f"lower_{tag}"] = low_df
            bands_dict[f"upper_{tag}"] = up_df
            if abs(lvl - (ci_levels[0] if isinstance(ci, (int, float)) else 0.90)) < 1e-4 or len(ci_levels) == 1:
                bands_dict["lower"] = low_df
                bands_dict["upper"] = up_df

        sim_paths = None
        if n_sims > 0:
            rng_sim = np.random.default_rng(seed)
            evals, evecs = np.linalg.eigh(Sigma_U_cond)
            evals_pos = np.where(evals > 1e-11, evals, 0.0)
            L = evecs @ np.diag(np.sqrt(evals_pos))
            z = rng_sim.standard_normal((n_sims, H * n_u))
            draws_u = full_shocks.reshape(1, H * n_u) + z @ L.T
            draws_v = baseline_arr.reshape(1, H * n_v) + draws_u @ R_full.T
            sim_paths = draws_v.reshape(n_sims, H, n_v)
            for v_name, h_val, t_val in active_restrs:
                v_idx = model_vars.index(v_name)
                sim_paths[:, h_val - 1, v_idx] = t_val
    else:
        sim_paths = None

    return ConditionalForecastResult(
        forecast=forecast_df,
        baseline=baseline_df,
        shocks=shocks_df,
        conditions=cond_dict,
        controlled_shocks=tuple(ctrl_shocks),
        bands=bands_dict,
        variable_names=tuple(model_vars),
        shock_names=tuple(model_shocks),
        horizon=H,
        simulations=sim_paths,
    )
