"""Optimal Simple Rules (OSR) and Policy Regimes (Discretion & Commitment).

This module implements:
1. ``osr()``: Optimal Simple Rules minimizing a quadratic variance loss
   L(gamma) = sum_i w_i Var(y_i; gamma) subject to Blanchard-Kahn determinacy,
   with a continuous numerical penalty surface when determinacy fails.
2. ``discretionary_policy()``: Markov-perfect time-consistent discretionary
   policy equilibrium using Dennis (2007) policy function iteration.
3. ``lq_commitment()``: Linear-quadratic optimal commitment policy under the
   timeless perspective (lambda_{-1} = 0), augmenting the first-order system
   with forward-looking Lagrange multipliers and solving via Klein QZ.
"""
from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize

from puremacro.dsge.klein import BlanchardKahnError, KleinSolution, klein_solve
from puremacro.dsge.dynare import _solve_lead_lag_system
from puremacro.dsge._moments import first_order_moments
from puremacro.dsge._results import OSRResult, PolicyResult
from puremacro.dsge.build import LinearModel, ModelError

__all__ = ["osr", "discretionary_policy", "lq_commitment"]


# ==============================================================================
# Helper: Re-solving LinearModel with Perturbed Parameters
# ==============================================================================

def _solve_with_params(model: LinearModel, new_params: dict[str, float]) -> LinearModel:
    """Re-solve model with updated parameters, preserving declarations."""
    merged_params = dict(getattr(model, "_params", {}) or {})
    merged_params.update(new_params)
    guess = model.steady_state.to_dict()

    if getattr(model, "_dynare_equations", None) is not None:
        from puremacro.dsge.dynare import build_dynare

        m_new = build_dynare(
            model._dynare_equations,
            variables=model.variables,
            shocks=model.shocks,
            params=merged_params,
            steady_state=guess,
            guess=guess,
            states=model.states,
            order=1,
            shock_cov=getattr(model, "_shock_cov", None),
            method=model.method,
            verify_derivatives=False,
            strict=False,
            check_steady_state=False,
        )
    elif getattr(model, "_equations", None) is not None:
        from puremacro.dsge.build import build as _build

        m_new = _build(
            model._equations,
            variables=model.variables,
            states=model.states,
            shocks=model.shocks,
            params=merged_params,
            steady_state=guess,
            guess=guess,
            method=model.method,
            verify_derivatives=False,
            strict=False,
        )
    else:
        raise ValueError(
            "Model carries neither _dynare_equations nor _equations; cannot re-solve parameters."
        )

    # Preserve metadata annotations if present
    replacements = {}
    for attr in ("_varobs", "_estimated_params", "_mod_options", "_equation_tags"):
        val = getattr(model, attr, None)
        if val is not None:
            replacements[attr] = val
    if replacements:
        m_new = dataclasses.replace(m_new, **replacements)

    return m_new


# ==============================================================================
# 1. Optimal Simple Rules (OSR)
# ==============================================================================

def osr(
    model: LinearModel,
    rule_params: Sequence[str] | str,
    weights: Mapping[str, float],
    *,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    target_vars: Sequence[str] | None = None,
    optimizer: str = "Nelder-Mead",
    maxiter: int = 1000,
    penalty: float = 1e6,
) -> OSRResult:
    """Optimize policy rule parameters against theoretical variance loss.

    Minimizes the quadratic variance loss:
        L(gamma) = sum_{i in targets} w_i * Var(y_i; gamma)
    subject to Blanchard-Kahn saddle-path determinacy. When determinacy fails
    or an evaluation error occurs, returns a continuous quadratic penalty:
        L_penalty(gamma) = 10^8 + 10^4 * ||gamma - gamma_0||^2
    preventing simplex collapse and allowing gradient-free optimizers
    (Nelder-Mead, Powell) to contract cleanly away from unstable regions.

    Parameters
    ----------
    model : LinearModel
        The solved linear DSGE model.
    rule_params : Sequence[str] or str
        Names of policy rule parameters to optimize (e.g. ["phi_pi", "phi_y"]).
    weights : Mapping[str, float]
        Weights on target variable variances (e.g. {"pi": 1.0, "y": 0.5}).
    bounds : Mapping[str, tuple[float, float]], optional
        Lower and upper bounds per rule parameter: {param: (low, high)}.
    target_vars : Sequence[str], optional
        Target variable names. Defaults to keys of weights.
    optimizer : str, default "Nelder-Mead"
        Optimization algorithm supported by scipy.optimize.minimize ("Nelder-Mead", "Powell").
    maxiter : int, default 1000
        Maximum optimizer iterations.
    penalty : float, default 1e6
        Baseline penalty scale factor.

    Returns
    -------
    OSRResult
        Frozen dataclass containing optimal coefficients, initial and optimal loss,
        variance breakdown table, and standard presentation methods.
    """
    if isinstance(rule_params, str):
        rule_params = [rule_params]
    rule_params = list(rule_params)

    weights_dict = {str(k): float(v) for k, v in weights.items()}
    if target_vars is None:
        targets = list(weights_dict.keys())
    else:
        targets = list(target_vars)

    # Initial parameter vector
    model_params = getattr(model, "_params", None)
    if model_params is None:
        raise ValueError("Model has no stored parameters in _params; cannot optimize rule parameters.")

    missing_params = [p for p in rule_params if p not in model_params]
    if missing_params:
        raise ValueError(f"Rule parameters {missing_params} not found in model._params.")

    gamma_0 = np.array([float(model_params[p]) for p in rule_params], dtype=float)

    # Evaluate baseline loss and variances
    try:
        tm_init = model.theoretical_moments()
        cov_init = tm_init.covariance
        var_init = {
            v: float(cov_init.loc[v, v]) if v in cov_init.index else 0.0
            for v in targets
        }
        loss_initial = float(sum(weights_dict.get(v, 0.0) * var_init[v] for v in targets))
    except Exception as exc:
        warnings.warn(f"Failed to evaluate initial baseline moments: {exc}")
        var_init = {v: np.nan for v in targets}
        loss_initial = float(penalty)

    # Define continuous BK failure penalty function
    def _penalty(gamma: np.ndarray, extra_dist: float = 0.0) -> float:
        d2 = float(np.sum((gamma - gamma_0) ** 2)) + extra_dist**2
        return 1e8 + 1e4 * d2

    # Objective function
    eval_counter = [0]

    def _objective(gamma: np.ndarray) -> float:
        eval_counter[0] += 1
        # Check bounds if supplied
        extra_bnd_dist = 0.0
        if bounds is not None:
            for idx, p in enumerate(rule_params):
                if p in bounds:
                    low, high = bounds[p]
                    val = gamma[idx]
                    if low is not None and val < low:
                        extra_bnd_dist += (low - val)
                    if high is not None and val > high:
                        extra_bnd_dist += (val - high)
        if extra_bnd_dist > 0:
            return _penalty(gamma, extra_dist=extra_bnd_dist * 1e3)

        trial_params = {p: float(gamma[idx]) for idx, p in enumerate(rule_params)}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m_trial = _solve_with_params(model, trial_params)
                if not m_trial.is_determinate:
                    return _penalty(gamma)
                tm = m_trial.theoretical_moments()
                cov = tm.covariance
                loss = 0.0
                for v in targets:
                    if v not in cov.index:
                        return _penalty(gamma)
                    v_var = float(cov.loc[v, v])
                    if np.isnan(v_var) or np.isinf(v_var) or v_var < 0:
                        return _penalty(gamma)
                    loss += weights_dict.get(v, 0.0) * v_var
                return float(loss)
        except (BlanchardKahnError, ModelError, ValueError, Exception):
            return _penalty(gamma)

    # Bounds for minimize
    scipy_bounds = None
    if bounds is not None:
        scipy_bounds = [bounds.get(p, (None, None)) for p in rule_params]

    # Run optimizer
    opt_res = scipy.optimize.minimize(
        _objective,
        gamma_0,
        method=optimizer,
        bounds=scipy_bounds if optimizer in ("Nelder-Mead", "Powell", "L-BFGS-B") else None,
        options={"maxiter": maxiter},
    )

    opt_gamma = np.asarray(opt_res.x, dtype=float)
    converged = bool(opt_res.success)
    message = str(opt_res.message)
    n_evals = int(getattr(opt_res, "nfev", eval_counter[0]))

    opt_params = {p: float(opt_gamma[idx]) for idx, p in enumerate(rule_params)}
    initial_params = {p: float(gamma_0[idx]) for idx, p in enumerate(rule_params)}

    # Re-solve at optimum to obtain optimal model and exact variances
    try:
        m_opt = _solve_with_params(model, opt_params)
        tm_opt = m_opt.theoretical_moments()
        cov_opt = tm_opt.covariance
        var_optimal = {
            v: float(cov_opt.loc[v, v]) if v in cov_opt.index else np.nan
            for v in targets
        }
        loss_opt = float(sum(weights_dict.get(v, 0.0) * var_optimal[v] for v in targets))
    except Exception:
        m_opt = None
        var_optimal = {v: np.nan for v in targets}
        loss_opt = float(opt_res.fun)

    # Construct variance table
    rows = []
    for v in targets:
        w = weights_dict.get(v, 0.0)
        vi = var_init.get(v, np.nan)
        vo = var_optimal.get(v, np.nan)
        li = w * vi if not np.isnan(vi) else np.nan
        lo = w * vo if not np.isnan(vo) else np.nan
        red_pct = 100.0 * (vi - vo) / vi if (vi > 0 and not np.isnan(vi) and not np.isnan(vo)) else 0.0
        rows.append({
            "weight": w,
            "var_initial": vi,
            "var_optimal": vo,
            "loss_initial": li,
            "loss_optimal": lo,
            "variance_reduction_pct": red_pct,
        })

    var_table = pd.DataFrame(rows, index=targets)

    return OSRResult(
        optimal_params=opt_params,
        initial_params=initial_params,
        loss_opt=loss_opt,
        loss_initial=loss_initial,
        rule_params=tuple(rule_params),
        target_vars=tuple(targets),
        weights=weights_dict,
        variance_table=var_table,
        converged=converged,
        message=message,
        n_evaluations=n_evals,
        optimal_model=m_opt,
    )


# ==============================================================================
# Helper: Policy Equation Identification
# ==============================================================================

def _identify_policy_equations(
    model: LinearModel,
    instruments: Sequence[str],
    policy_eq: int | str | Sequence[int | str] | None = None,
) -> list[int]:
    """Identify which equation row index/indices correspond to the policy instrument(s)."""
    tags = getattr(model, "_equation_tags", None)
    variables = list(model.variables)
    N = len(variables)

    if policy_eq is not None:
        if isinstance(policy_eq, (int, str)):
            policy_eq = [policy_eq]
        indices = []
        for eq in policy_eq:
            if isinstance(eq, int):
                indices.append(eq)
            elif isinstance(eq, str):
                if tags and eq in tags:
                    indices.append(tags.index(eq))
                elif eq in variables:
                    indices.append(variables.index(eq))
                else:
                    raise ValueError(f"Unknown policy equation identifier: {eq}")
            else:
                raise TypeError(f"Invalid policy equation identifier type: {type(eq)}")
        return indices

    A_0 = getattr(model, "A_0", None)
    if A_0 is None:
        A_0 = getattr(model, "_A_0", None)
    A_plus = getattr(model, "A_plus", None)
    if A_plus is None:
        A_plus = getattr(model, "_A_plus", None)

    chosen: list[int] = []
    for inst in instruments:
        if inst not in variables:
            raise ValueError(f"Instrument '{inst}' not found in model variables: {variables}")
        inst_col = variables.index(inst)
        scores = np.zeros(N)

        for r in range(N):
            # Equation tag score
            if tags and r < len(tags):
                t = str(tags[r]).lower()
                if inst.lower() == t:
                    scores[r] += 25
                elif inst.lower() in t:
                    scores[r] += 12
                if any(k in t for k in ("taylor", "policy", "monetary", "rule", "instrument")):
                    scores[r] += 10

            # Presence in A_0
            if A_0 is not None and abs(A_0[r, inst_col]) > 1e-6:
                scores[r] += 8
                # If instrument is dominant in row r
                if abs(A_0[r, inst_col]) == np.max(np.abs(A_0[r, :])):
                    scores[r] += 5

            # Absence of forward expectations in A_plus
            if A_plus is not None and np.all(np.abs(A_plus[r, :]) < 1e-6):
                scores[r] += 6

        # Avoid choosing already picked rows
        for c in chosen:
            scores[c] = -1e9

        best_r = int(np.argmax(scores))
        chosen.append(best_r)

    return chosen


# ==============================================================================
# 2. Discretionary Policy (Dennis 2007)
# ==============================================================================

def discretionary_policy(
    model: LinearModel,
    target_vars: Sequence[str],
    weights: Mapping[str, float] | Sequence[float],
    instruments: Sequence[str] | str,
    *,
    beta: float = 0.99,
    max_iter: int = 1000,
    tol: float = 1e-8,
    policy_eq: int | str | Sequence[int | str] | None = None,
) -> PolicyResult:
    """Solve optimal discretionary policy using Dennis (2007) policy iteration.

    Computes the Markov-perfect time-consistent discretionary policy equilibrium
    by iterating on central bank and private sector feedback matrices under
    quadratic intertemporal loss:
        min_{i_t} E_t sum_{tau=0}^infty beta^tau [ y_{t+tau}^T Q y_{t+tau} + i_{t+tau}^T R i_{t+tau} ]
    subject to the private sector rational expectations system with the
    endogenous policy rule removed:
        A_0 y_t = A_1 y_{t-1} + B_1 i_{t-1} + A_2 E_t y_{t+1} + B_2 E_t i_{t+1} + B_0 i_t + A_3 u_t

    Parameters
    ----------
    model : LinearModel
        The baseline linear DSGE model with lead-lag companion matrices.
    target_vars : Sequence[str]
        Variables entering the policymaker's loss function.
    weights : Mapping[str, float] or Sequence[float]
        Quadratic loss weights. If a sequence is given, matches target_vars.
    instruments : Sequence[str] or str
        Policy instrument variable name(s).
    beta : float, default 0.99
        Policymaker discount factor.
    max_iter : int, default 1000
        Maximum fixed-point policy iterations.
    tol : float, default 1e-8
        Convergence tolerance on feedback matrices.
    policy_eq : int, str, or sequence, optional
        Explicit index or tag name of the policy rule equation(s) to remove.

    Returns
    -------
    PolicyResult
        Frozen dataclass carrying optimal reaction functions, transition matrices,
        unconditional loss, and solved LinearModel under discretion.
    """
    if isinstance(instruments, str):
        instruments = [instruments]
    instruments = list(instruments)
    target_vars = list(target_vars)

    if isinstance(weights, Mapping):
        weights_dict = {k: float(v) for k, v in weights.items()}
    else:
        weights_dict = {k: float(v) for k, v in zip(target_vars, weights)}

    # Extract first-order companion matrices
    A_plus = getattr(model, "A_plus", None)
    if A_plus is None:
        A_plus = getattr(model, "_A_plus", None)
    A_0 = getattr(model, "A_0", None)
    if A_0 is None:
        A_0 = getattr(model, "_A_0", None)
    A_minus = getattr(model, "A_minus", None)
    if A_minus is None:
        A_minus = getattr(model, "_A_minus", None)
    B_u = getattr(model, "B_u", None)
    if B_u is None:
        B_u = getattr(model, "_B_u", None)

    if A_0 is None or A_plus is None or A_minus is None or B_u is None:
        raise ValueError(
            "discretionary_policy requires model to carry lead-lag companion matrices "
            "(A_plus, A_0, A_minus, B_u), typically from build_dynare() or load_mod()."
        )

    variables = list(model.variables)
    N = len(variables)
    n_i = len(instruments)

    # Identify and remove policy equation(s)
    pol_eq_indices = _identify_policy_equations(model, instruments, policy_eq=policy_eq)
    priv_eq_indices = [r for r in range(N) if r not in pol_eq_indices]

    # Partition variables into non-instruments (y) and instruments (i)
    inst_indices = [variables.index(inst) for inst in instruments]
    y_indices = [j for j in range(N) if j not in inst_indices]
    y_names = [variables[j] for j in y_indices]
    n_y = len(y_indices)

    # Private sector equations: A_+ E_t z_{t+1} + A_0 z_t + A_- z_{t-1} + B_u u_t = 0
    A_plus_priv = A_plus[priv_eq_indices, :]
    A_0_priv = A_0[priv_eq_indices, :]
    A_minus_priv = A_minus[priv_eq_indices, :]
    B_u_priv = B_u[priv_eq_indices, :]

    # Convert to Dennis (2007) canonical form:
    # A_0_dennis * y_t = A_1 * y_{t-1} + B_1 * i_{t-1} + A_2 * E_t y_{t+1} + B_2 * E_t i_{t+1} + B_0 * i_t + A_3 * u_t
    A_0_dennis = - A_0_priv[:, y_indices]
    A_1_dennis = A_minus_priv[:, y_indices]
    B_1_dennis = A_minus_priv[:, inst_indices]
    A_2_dennis = A_plus_priv[:, y_indices]
    B_2_dennis = A_plus_priv[:, inst_indices]
    B_0_dennis = A_0_priv[:, inst_indices]
    A_3_dennis = B_u_priv

    # Build loss matrices Q, U, R
    W_full = np.zeros((N, N))
    for v, w in weights_dict.items():
        if v in variables:
            idx = variables.index(v)
            W_full[idx, idx] = float(w)

    Q = W_full[y_indices, :][:, y_indices]
    U = W_full[y_indices, :][:, inst_indices]
    R = W_full[inst_indices, :][:, inst_indices]

    # State vector in Dennis: x_{t-1} = [y_{t-1}; i_{t-1}] of size n_x = n_y + n_i = N
    n_x = n_y + n_i
    n_u = B_u.shape[1]

    D = np.zeros((n_y, n_x))
    F = np.zeros((n_i, n_x))
    V = np.zeros((n_x, n_x))

    converged = False
    for it in range(max_iter):
        D1 = D[:, :n_y]
        D2 = D[:, n_y:]
        F1 = F[:, :n_y]
        F2 = F[:, n_y:]

        # Lead matrix inverse: H = (A_0 - A_2 D1 - B_2 F1)^{-1}
        M_lead = A_0_dennis - A_2_dennis @ D1 - B_2_dennis @ F1
        try:
            H = np.linalg.inv(M_lead)
        except np.linalg.LinAlgError:
            H = np.linalg.pinv(M_lead)

        A_stacked = np.hstack([A_1_dennis, B_1_dennis])
        A_tilde = H @ A_stacked
        B_tilde = H @ (B_0_dennis + A_2_dennis @ D2 + B_2_dennis @ F2)
        C_tilde = H @ A_3_dennis

        # Continuation value blocks
        V_yy = V[:n_y, :n_y]
        V_yi = V[:n_y, n_y:]
        V_ii = V[n_y:, n_y:]

        Omega_yy = Q + beta * V_yy
        Omega_yi = U + beta * V_yi
        Omega_ii = R + beta * V_ii

        # Reaction function update
        Psi = B_tilde.T @ Omega_yy @ B_tilde + B_tilde.T @ Omega_yi + Omega_yi.T @ B_tilde + Omega_ii
        Theta = B_tilde.T @ Omega_yy @ A_tilde + Omega_yi.T @ A_tilde

        try:
            F_new = - np.linalg.solve(Psi, Theta)
        except np.linalg.LinAlgError:
            F_new = - np.linalg.pinv(Psi) @ Theta

        D_new = A_tilde + B_tilde @ F_new

        # Shock response
        try:
            f_u = - np.linalg.solve(Psi, (B_tilde.T @ Omega_yy + Omega_yi.T) @ C_tilde)
        except np.linalg.LinAlgError:
            f_u = - np.linalg.pinv(Psi) @ ((B_tilde.T @ Omega_yy + Omega_yi.T) @ C_tilde)
        d_u = B_tilde @ f_u + C_tilde

        # Riccati update
        M_x = np.vstack([D_new, F_new])
        W_perm = np.block([[Q, U], [U.T, R]])
        V_new = M_x.T @ W_perm @ M_x + beta * M_x.T @ V @ M_x

        diff = max(
            float(np.max(np.abs(D_new - D))),
            float(np.max(np.abs(F_new - F))),
        )
        D, F, V = D_new, F_new, V_new

        if diff < tol:
            converged = True
            break

    if not converged:
        warnings.warn(f"discretionary_policy(): Dennis (2007) iteration did not converge within {max_iter} steps (final diff={diff:.2e}).")

    # Map state transition back to original model variable ordering
    perm = y_indices + inst_indices
    inv_perm = np.argsort(perm)

    G_perm = np.vstack([D, F])
    N_perm = np.vstack([d_u, f_u])

    G = G_perm[inv_perm, :][:, inv_perm]
    N_mat = N_perm[inv_perm, :]

    # Shock covariance and unconditional loss
    sigma_u = getattr(model, "_shock_cov", None)
    if sigma_u is None:
        sigma_u = np.eye(n_u)
    else:
        sigma_u = np.asarray(sigma_u, dtype=float)

    try:
        sigma_x = scipy.linalg.solve_discrete_lyapunov(G, N_mat @ sigma_u @ N_mat.T)
        sigma_x = 0.5 * (sigma_x + sigma_x.T)
        loss = float(np.trace(W_full @ sigma_x))
    except Exception:
        sigma_x = np.zeros((N, N))
        loss = np.nan

    # Policy reaction function table
    # Columns are variables in perm order [y_names + instruments]
    col_names = [variables[j] for j in perm]
    df_policy_rules = pd.DataFrame(F, index=instruments, columns=col_names)

    # Wrap into LinearModel for seamless .irf(), .fevd(), .simulate()
    # In Dynare timing, states are the variables themselves
    sol_discretion = KleinSolution(
        G=G,
        F=np.zeros((0, N)),
        N=N_mat,
        L=np.zeros((0, n_u)),
        eigenvalues=np.sort(np.abs(scipy.linalg.eigvals(G))),
        eu=(1, 1),
    )

    m_discretion = LinearModel(
        variables=tuple(variables),
        states=tuple(variables),
        controls=(),
        shocks=model.shocks,
        steady_state=model.steady_state.copy(),
        units=dict(model.units),
        solution=sol_discretion,
        A=np.eye(N),
        B=G,
        C=N_mat,
        method=model.method,
        residual_norm=0.0,
        _A_plus=None,
        _A_0=np.eye(N),
        _A_minus=-G,
        _B_u=-N_mat,
        _shock_cov=sigma_u,
        timing="dynare",
    )

    return PolicyResult(
        regime="discretion",
        target_vars=tuple(target_vars),
        weights=weights_dict,
        instruments=tuple(instruments),
        beta=float(beta),
        loss=loss,
        policy_rules=df_policy_rules,
        transition_matrix=G,
        impact_matrix=N_mat,
        multipliers=(),
        linear_model=m_discretion,
    )


# ==============================================================================
# 3. Linear-Quadratic Commitment (Ramsey / Timeless Perspective)
# ==============================================================================

def lq_commitment(
    model: LinearModel,
    target_vars: Sequence[str],
    weights: Mapping[str, float] | Sequence[float],
    instruments: Sequence[str] | str,
    *,
    beta: float = 0.99,
    discount: float | None = None,
    timeless: bool = True,
    policy_eq: int | str | Sequence[int | str] | None = None,
) -> PolicyResult:
    """Solve optimal linear-quadratic commitment policy under the timeless perspective.

    Forms the social planner's Lagrangian over the linear rational-expectations
    constraints and quadratic loss:
        min_{z_t} 1/2 E_0 sum_{t=0}^infty beta^t z_t^T W z_t
    subject to the private sector equilibrium conditions:
        A_+ E_t z_{t+1} + A_0 z_t + A_- z_{t-1} + B_u u_t = 0
    yielding the augmented 2N-dimensional saddle-path system in (z_t, lambda_t):
        cal_A_+ E_t X_{t+1} + cal_A_0 X_t + cal_A_- X_{t-1} + cal_B_u u_t = 0
    solved via klein_solve under stationary timeless-perspective initial conditions
    (lambda_{-1} = 0). Policy multipliers are exposed as model variables.

    Parameters
    ----------
    model : LinearModel
        The baseline linear DSGE model.
    target_vars : Sequence[str]
        Variables entering the policymaker's loss function.
    weights : Mapping[str, float] or Sequence[float]
        Quadratic loss weights.
    instruments : Sequence[str] or str
        Policy instrument variable name(s).
    beta : float, default 0.99
        Policymaker discount factor.
    discount : float, optional
        Alias for beta.
    timeless : bool, default True
        If True, applies the timeless-perspective stationary initial condition (lambda_{-1} = 0).
    policy_eq : int, str, or sequence, optional
        Explicit index or tag name of the policy rule equation(s) to remove.

    Returns
    -------
    PolicyResult
        Frozen dataclass carrying augmented LinearModel where Lagrange multipliers
        (mult_*) are accessible as model variables for IRFs, FEVD, and moments.
    """
    if discount is not None:
        beta = float(discount)
    beta = float(beta)

    if isinstance(instruments, str):
        instruments = [instruments]
    instruments = list(instruments)
    target_vars = list(target_vars)

    if isinstance(weights, Mapping):
        weights_dict = {k: float(v) for k, v in weights.items()}
    else:
        weights_dict = {k: float(v) for k, v in zip(target_vars, weights)}

    # Extract first-order companion matrices
    A_plus = getattr(model, "A_plus", None)
    if A_plus is None:
        A_plus = getattr(model, "_A_plus", None)
    A_0 = getattr(model, "A_0", None)
    if A_0 is None:
        A_0 = getattr(model, "_A_0", None)
    A_minus = getattr(model, "A_minus", None)
    if A_minus is None:
        A_minus = getattr(model, "_A_minus", None)
    B_u = getattr(model, "B_u", None)
    if B_u is None:
        B_u = getattr(model, "_B_u", None)

    if A_0 is None or A_plus is None or A_minus is None or B_u is None:
        raise ValueError(
            "lq_commitment requires model to carry lead-lag companion matrices "
            "(A_plus, A_0, A_minus, B_u), typically from build_dynare() or load_mod()."
        )

    variables = list(model.variables)
    N = len(variables)
    tags = getattr(model, "_equation_tags", None)

    # Identify and remove policy equation(s)
    pol_eq_indices = _identify_policy_equations(model, instruments, policy_eq=policy_eq)
    priv_eq_indices = [r for r in range(N) if r not in pol_eq_indices]
    M = len(priv_eq_indices)  # number of private sector constraints

    A_plus_priv = A_plus[priv_eq_indices, :]
    A_0_priv = A_0[priv_eq_indices, :]
    A_minus_priv = A_minus[priv_eq_indices, :]
    B_u_priv = B_u[priv_eq_indices, :]

    # Quadratic weight matrix W (N x N)
    W = np.zeros((N, N))
    for v, w in weights_dict.items():
        if v in variables:
            idx = variables.index(v)
            W[idx, idx] = float(w)

    # Multiplier names
    mult_names: list[str] = []
    for r in priv_eq_indices:
        if tags and r < len(tags):
            tag_name = str(tags[r]).replace(" ", "_")
            mult_names.append(f"mult_{tag_name}")
        else:
            mult_names.append(f"mult_{variables[r]}")

    aug_variables = variables + mult_names
    total_vars = N + M
    n_u = B_u.shape[1]

    # Form augmented Lagrangian system block matrices:
    # cal_A_+ E_t X_{t+1} + cal_A_0 X_t + cal_A_- X_{t-1} + cal_B_u u_t = 0
    # X_t = [z_t; lambda_t]
    # Block 1 (M constraints): A_+ E_t z_{t+1} + A_0 z_t + A_- z_{t-1} + B_u u_t = 0
    # Block 2 (N planner FOCs): W z_t + A_0^T lambda_t + beta^{-1} A_+^T lambda_{t-1} + beta A_-^T E_t lambda_{t+1} = 0
    cal_A_plus = np.zeros((total_vars, total_vars))
    cal_A_0 = np.zeros((total_vars, total_vars))
    cal_A_minus = np.zeros((total_vars, total_vars))
    cal_B_u = np.zeros((total_vars, n_u))

    # Constraints block (rows 0 to M-1)
    cal_A_plus[:M, :N] = A_plus_priv
    cal_A_0[:M, :N] = A_0_priv
    cal_A_minus[:M, :N] = A_minus_priv
    cal_B_u[:M, :] = B_u_priv

    # Planner FOCs block (rows M to total_vars-1)
    cal_A_plus[M:, N:] = beta * A_minus_priv.T
    cal_A_0[M:, :N] = W
    cal_A_0[M:, N:] = A_0_priv.T
    cal_A_minus[M:, N:] = (1.0 / beta) * A_plus_priv.T

    # Identify state variables (those entering with a lag in cal_A_minus)
    state_idx = [
        j for j in range(total_vars)
        if float(np.linalg.norm(cal_A_minus[:, j])) > 1e-10
    ]

    if len(state_idx) == 0:
        # Fallback: if no lags, pick multipliers that have lead terms
        state_idx = list(range(N, total_vars))

    state_names = [aug_variables[j] for j in state_idx]
    control_names = [aug_variables[j] for j in range(total_vars) if j not in state_idx]

    # Solve lead-lag system via Klein QZ
    A_klein, B_klein, C_klein, sol_full, F_full, L_full = _solve_lead_lag_system(
        cal_A_plus,
        cal_A_0,
        cal_A_minus,
        cal_B_u,
        state_idx,
        strict=False,
    )

    ctrl_idx = [j for j in range(total_vars) if j not in state_idx]
    n_s = len(state_idx)
    G = F_full[state_idx, :]
    N_state = L_full[state_idx, :]
    F_ctrl = F_full[ctrl_idx, :]
    L_ctrl = L_full[ctrl_idx, :]

    # Policy reaction functions: instruments as linear combination of predetermined states
    inst_indices = [variables.index(inst) for inst in instruments]
    df_policy_rules = pd.DataFrame(
        F_full[inst_indices, :],
        index=instruments,
        columns=state_names,
    )

    # Theoretical moments & loss
    sigma_u = getattr(model, "_shock_cov", None)
    if sigma_u is None:
        sigma_u = np.eye(n_u)
    else:
        sigma_u = np.asarray(sigma_u, dtype=float)

    try:
        sigma_s = scipy.linalg.solve_discrete_lyapunov(G, N_state @ sigma_u @ N_state.T)
        sigma_s = 0.5 * (sigma_s + sigma_s.T)
        cov_X = F_full @ sigma_s @ F_full.T + L_full @ sigma_u @ L_full.T
        loss = float(np.trace(W @ cov_X[:N, :N]))
    except Exception:
        cov_X = np.zeros((total_vars, total_vars))
        loss = np.nan

    # Build steady state series (original model steady state for z, 0.0 for multipliers)
    ss_dict = {v: float(model.steady_state[v]) for v in variables if v in model.steady_state.index}
    for m in mult_names:
        ss_dict[m] = 0.0
    ss_series = pd.Series(ss_dict)[aug_variables]

    units_dict = {v: "level" for v in aug_variables}

    solution_comm = KleinSolution(
        G=G,
        F=F_ctrl,
        N=N_state,
        L=L_ctrl,
        eu=tuple(sol_full.eu),
        eigenvalues=sol_full.eigenvalues,
    )

    m_commitment = LinearModel(
        variables=tuple(aug_variables),
        states=tuple(state_names),
        controls=tuple(control_names),
        shocks=model.shocks,
        steady_state=ss_series,
        units=units_dict,
        solution=solution_comm,
        A=A_klein,
        B=B_klein,
        C=C_klein,
        method=model.method,
        residual_norm=0.0,
        _A_plus=cal_A_plus,
        _A_0=cal_A_0,
        _A_minus=cal_A_minus,
        _B_u=cal_B_u,
        _shock_cov=sigma_u,
        timing="dynare",
    )

    return PolicyResult(
        regime="commitment",
        target_vars=tuple(target_vars),
        weights=weights_dict,
        instruments=tuple(instruments),
        beta=float(beta),
        loss=loss,
        policy_rules=df_policy_rules,
        transition_matrix=G,
        impact_matrix=N_state,
        multipliers=tuple(mult_names),
        linear_model=m_commitment,
    )
