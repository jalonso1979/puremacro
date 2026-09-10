r"""Exact analytical likelihood score and parameter sensitivity engine for linear DSGE models.

Computes exact analytical gradients of the log-likelihood function:
    \nabla_\theta \ln L(Y \mid \theta)
via implicit state-space differentiation and single-pass forward Kalman score recursion,
strictly under the zero-dependency Pyodide four-package contract (numpy, scipy, pandas, matplotlib).

Mathematical Foundations
-------------------------
1. Parameter Sensitivity of Equilibrium Decision Rules:
   Given structural equations around steady state:
       A_+(\theta) G(\theta)^2 + A_0(\theta) G(\theta) + A_-(\theta) = 0
       (A_0(\theta) + A_+(\theta) G(\theta)) N(\theta) + B_u(\theta) = 0
   Differentiating with respect to parameter \theta_j yields the generalized Sylvester system:
       \hat{A} \frac{\partial G}{\partial \theta_j} + A_+ \frac{\partial G}{\partial \theta_j} G = D_j
   where \hat{A} = A_0 + A_+ G and D_j = - ( \frac{\partial A_+}{\partial \theta_j} G^2 + \frac{\partial A_0}{\partial \theta_j} G + \frac{\partial A_-}{\partial \theta_j} ).
   Solved via complex Schur triangular back-substitution in O(N^3) flops.

   The innovation loading derivative follows directly:
       \frac{\partial N}{\partial \theta_j} = - \hat{A}^{-1} [ ( \frac{\partial A_0}{\partial \theta_j} + \frac{\partial A_+}{\partial \theta_j} G + A_+ \frac{\partial G}{\partial \theta_j} ) N + \frac{\partial B_u}{\partial \theta_j} ]

2. Stationary Unconditional Covariance Sensitivity:
   Differentiating the discrete Lyapunov equation P_0 = T P_0 T' + R Q R':
       \frac{\partial P_0}{\partial \theta_j} - T \frac{\partial P_0}{\partial \theta_j} T' = \frac{\partial T}{\partial \theta_j} P_0 T' + T P_0 (\frac{\partial T}{\partial \theta_j})' + \frac{\partial (R Q R')}{\partial \theta_j}
   solved via scipy.linalg.solve_discrete_lyapunov.

3. Single-Pass Forward Kalman Score Recursion:
   For observations y_t with forecast errors v_t = y_t - Z a_{t|t-1} - d and covariances F_t = Z P_{t|t-1} Z' + H:
       \frac{\partial \ln L}{\partial \theta_j} = -0.5 \sum_{t=1}^{T_{obs}} [ \operatorname{tr}(F_t^{-1} \frac{\partial F_t}{\partial \theta_j}) + 2 v_t' F_t^{-1} \frac{\partial v_t}{\partial \theta_j} - (F_t^{-1} v_t)' \frac{\partial F_t}{\partial \theta_j} (F_t^{-1} v_t) ]
   propagating (\frac{\partial a_{t|t-1}}{\partial \theta_j}, \frac{\partial P_{t|t-1}}{\partial \theta_j}) forward alongside the filter.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.linalg

from puremacro.dsge.priors import grad_log_prior, log_prior
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst
from puremacro.state_space import StateSpaceModel


# ---------------------------------------------------------------------------
# Generalized Sylvester Solver
# ---------------------------------------------------------------------------

def solve_sylvester_generalized(
    A_hat: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    r"""Solve the generalized Sylvester matrix equation \hat{A} X + B X C = D.

    Parameters
    ----------
    A_hat : np.ndarray of shape (N, N)
        Leading coefficient matrix, typically (A_0 + A_+ G).
    B : np.ndarray of shape (N, N)
        Intermediate coefficient matrix, typically A_+.
    C : np.ndarray of shape (M, M)
        Right multiplier matrix with stable eigenvalues (e.g. state transition G).
    D : np.ndarray of shape (N, M) or batched (K, N, M)
        Right-hand side matrix or batch of K right-hand side matrices.

    Returns
    -------
    X : np.ndarray of shape (N, M) or batched (K, N, M)
        Solution matrix satisfying \hat{A} X + B X C = D.

    Notes
    -----
    Computes the complex Schur decomposition of C:
        C = U_c T_c U_c^H
    where T_c is upper triangular and U_c is unitary.
    Transforming \tilde{X} = X U_c and \tilde{D} = D U_c leads to:
        \hat{A} \tilde{X} + B \tilde{X} T_c = \tilde{D}
    Because T_c is upper triangular, column k of \tilde{X} satisfies:
        (\hat{A} + (T_c)_{k, k} B) \tilde{X}_{:, k} = \tilde{D}_{:, k} - B \sum_{m < k} \tilde{X}_{:, m} (T_c)_{m, k}
    which is solved column-by-column by triangular back-substitution.
    """
    A_hat = np.asarray(A_hat, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)

    is_batched = (D.ndim == 3)
    if not is_batched:
        D_batch = D[np.newaxis, :, :]
    else:
        D_batch = D

    K_batch, N, M = D_batch.shape
    if M == 0 or N == 0:
        res = np.zeros_like(D)
        return res

    if np.all(C == 0.0):
        # Degenerates to A_hat X = D
        X_batch = np.zeros_like(D_batch)
        for k in range(K_batch):
            try:
                X_batch[k] = np.linalg.solve(A_hat, D_batch[k])
            except np.linalg.LinAlgError:
                X_batch[k] = np.linalg.lstsq(A_hat, D_batch[k], rcond=None)[0]
        return X_batch if is_batched else X_batch[0]

    # Complex Schur decomposition of C: C = U_c @ T_c @ U_c^H
    T_c, U_c = scipy.linalg.schur(C, output="complex")

    # Transform RHS: \tilde{D} = D @ U_c
    D_tilde = np.zeros((K_batch, N, M), dtype=complex)
    for b in range(K_batch):
        D_tilde[b] = D_batch[b] @ U_c

    X_tilde = np.zeros((K_batch, N, M), dtype=complex)

    # Solve column by column k = 0, ..., M - 1
    for k in range(M):
        diag_val = T_c[k, k]
        M_k = A_hat + diag_val * B

        # Factorize M_k once for all K batch elements
        try:
            lu, piv = scipy.linalg.lu_factor(M_k)
            use_lu = True
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
            use_lu = False

        for b in range(K_batch):
            rhs_k = D_tilde[b, :, k].copy()
            if k > 0:
                # Subtraction of lower column couplings
                prev_couplings = X_tilde[b, :, :k] @ T_c[:k, k]
                rhs_k -= B @ prev_couplings

            if use_lu:
                sol_k = scipy.linalg.lu_solve((lu, piv), rhs_k)
            else:
                try:
                    sol_k = np.linalg.solve(M_k, rhs_k)
                except np.linalg.LinAlgError:
                    sol_k = np.linalg.lstsq(M_k, rhs_k, rcond=None)[0]

            X_tilde[b, :, k] = sol_k

    # Back-transform: X = Re(\tilde{X} @ U_c^H)
    X_out = np.zeros((K_batch, N, M), dtype=float)
    U_c_H = U_c.conj().T
    for b in range(K_batch):
        X_out[b] = np.real(X_tilde[b] @ U_c_H)

    return X_out if is_batched else X_out[0]


# ---------------------------------------------------------------------------
# StateSpaceSensitivity Container
# ---------------------------------------------------------------------------

@dataclass
class StateSpaceSensitivity:
    """Analytical parameter sensitivities of time-invariant state-space matrices."""

    dT: np.ndarray
    dZ: np.ndarray
    dR: np.ndarray
    dQ: np.ndarray
    dH: np.ndarray
    dc: np.ndarray
    dd: np.ndarray
    da0: np.ndarray | None = None
    dP0: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Forward Kalman Score Recursion
# ---------------------------------------------------------------------------

def kalman_score(
    y: np.ndarray,
    model: StateSpaceModel,
    sensitivities: Mapping[str, StateSpaceSensitivity] | Sequence[StateSpaceSensitivity],
    a0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    r"""Compute exact analytical log-likelihood and score vector \nabla_\theta \ln L.

    Parameters
    ----------
    y : np.ndarray of shape (T_obs, n_y)
        Observed series matrix. NaN entries indicate missing values.
    model : StateSpaceModel
        Baseline state space model containing T, Z, R, Q, H, c, d.
    sensitivities : Mapping[str, StateSpaceSensitivity] or Sequence[StateSpaceSensitivity]
        Parameter sensitivity derivatives of state-space matrices.
    a0 : np.ndarray of shape (m,), optional
        Initial state mean. If None, initialized from stationary mean.
    P0 : np.ndarray of shape (m, m), optional
        Initial state covariance. If None, initialized from Lyapunov solution.

    Returns
    -------
    loglik : float
        Log-likelihood value \ln L(Y \mid \theta).
    score : np.ndarray of shape (K,)
        Analytical gradient vector \nabla_\theta \ln L(Y \mid \theta).
    """
    y_mat = np.asarray(y, dtype=float)
    if y_mat.ndim != 2:
        raise ValueError(f"y must be a 2-D array of shape (T_obs, n_y); got {y_mat.shape}")

    T_obs, n_y = y_mat.shape
    T_mat = np.asarray(model.T, dtype=float)
    Z_mat = np.asarray(model.Z, dtype=float)
    R_mat = np.asarray(model.R, dtype=float)
    Q_mat = np.asarray(model.Q, dtype=float)
    H_mat = np.asarray(model.H, dtype=float)
    c_vec = np.asarray(model.c, dtype=float).ravel()
    d_vec = np.asarray(model.d, dtype=float).ravel()

    m = T_mat.shape[0]
    r_dim = R_mat.shape[1]

    if isinstance(sensitivities, Mapping):
        sens_list = list(sensitivities.values())
    else:
        sens_list = list(sensitivities)

    K = len(sens_list)
    score = np.zeros(K, dtype=float)

    # Initial unconditional covariance P0 and mean a0
    RQR = R_mat @ Q_mat @ R_mat.T
    if P0 is None:
        try:
            P_curr = scipy.linalg.solve_discrete_lyapunov(T_mat, RQR)
            P_curr = 0.5 * (P_curr + P_curr.T)
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgError, ValueError):
            P_curr = np.eye(m) * 1e6
    else:
        P_curr = np.asarray(P0, dtype=float).copy()

    if a0 is None:
        if np.any(c_vec != 0.0):
            try:
                a_curr = np.linalg.solve(np.eye(m) - T_mat, c_vec)
            except np.linalg.LinAlgError:
                a_curr = np.zeros(m)
        else:
            a_curr = np.zeros(m)
    else:
        a_curr = np.asarray(a0, dtype=float).copy()

    # Initialize da0 and dP0 for each parameter
    da_curr = [np.zeros(m) for _ in range(K)]
    dP_curr = [np.zeros((m, m)) for _ in range(K)]

    for j, s in enumerate(sens_list):
        if s.da0 is not None:
            da_curr[j] = np.asarray(s.da0, dtype=float).copy()
        else:
            if np.any(s.dT != 0.0) or np.any(s.dc != 0.0):
                try:
                    da_curr[j] = np.linalg.solve(
                        np.eye(m) - T_mat, s.dT @ a_curr + s.dc
                    )
                except np.linalg.LinAlgError:
                    da_curr[j] = np.zeros(m)

        if s.dP0 is not None:
            dP_curr[j] = np.asarray(s.dP0, dtype=float).copy()
        else:
            # Differentiate Lyapunov equation: dP0 - T dP0 T' = dT P0 T' + T P0 dT' + d(RQR)
            dRQR = (
                s.dR @ Q_mat @ R_mat.T
                + R_mat @ s.dQ @ R_mat.T
                + R_mat @ Q_mat @ s.dR.T
            )
            rhs_dP = s.dT @ P_curr @ T_mat.T + T_mat @ P_curr @ s.dT.T + dRQR
            try:
                dP_sol = scipy.linalg.solve_discrete_lyapunov(T_mat, rhs_dP)
                dP_curr[j] = 0.5 * (dP_sol + dP_sol.T)
            except (np.linalg.LinAlgError, scipy.linalg.LinAlgError, ValueError):
                dP_curr[j] = np.zeros((m, m))

    log2pi = math.log(2.0 * math.pi)
    loglik = 0.0

    # Sequential forward filter and sensitivity recursion
    for t in range(T_obs):
        y_t = y_mat[t]
        obs = np.isfinite(y_t)
        n_obs = int(np.sum(obs))

        if n_obs == 0:
            # Completely missing period: propagate forward without update
            a_next = T_mat @ a_curr + c_vec
            P_next = T_mat @ P_curr @ T_mat.T + RQR
            P_next = 0.5 * (P_next + P_next.T)

            for j, s in enumerate(sens_list):
                dRQR_j = (
                    s.dR @ Q_mat @ R_mat.T
                    + R_mat @ s.dQ @ R_mat.T
                    + R_mat @ Q_mat @ s.dR.T
                )
                da_curr[j] = s.dT @ a_curr + T_mat @ da_curr[j] + s.dc
                dP_curr[j] = (
                    s.dT @ P_curr @ T_mat.T
                    + T_mat @ dP_curr[j] @ T_mat.T
                    + T_mat @ P_curr @ s.dT.T
                    + dRQR_j
                )
                dP_curr[j] = 0.5 * (dP_curr[j] + dP_curr[j].T)

            a_curr = a_next
            P_curr = P_next
            continue

        # Sub-select active observables for period t
        if n_obs == n_y:
            y_obs = y_t
            Z_t = Z_mat
            d_t = d_vec
            H_t = H_mat
        else:
            y_obs = y_t[obs]
            Z_t = Z_mat[obs, :]
            d_t = d_vec[obs]
            H_t = H_mat[np.ix_(obs, obs)]

        # Baseline forecast error and covariance
        v = y_obs - Z_t @ a_curr - d_t
        F = Z_t @ P_curr @ Z_t.T + H_t
        F = 0.5 * (F + F.T)

        try:
            F_chol = np.linalg.cholesky(F)
            log_det = 2.0 * np.sum(np.log(np.diag(F_chol)))
            F_inv_v = np.linalg.solve(F_chol.T, np.linalg.solve(F_chol, v))
            F_inv = np.linalg.inv(F)
        except np.linalg.LinAlgError:
            F = F + 1e-10 * np.eye(n_obs)
            F_chol = scipy.linalg.cholesky(F, lower=True)
            log_det = 2.0 * np.sum(np.log(np.diag(F_chol)))
            F_inv_v = np.linalg.solve(F_chol.T, np.linalg.solve(F_chol, v))
            F_inv = np.linalg.inv(F)

        quad_form = float(v @ F_inv_v)
        ll_t = -0.5 * (n_obs * log2pi + log_det + quad_form)
        loglik += ll_t

        M_t = T_mat @ P_curr @ Z_t.T
        K_t = M_t @ F_inv
        L_t = T_mat - K_t @ Z_t

        # Pre-allocate containers for step t+1 sensitivities
        da_next = [np.zeros(m) for _ in range(K)]
        dP_next = [np.zeros((m, m)) for _ in range(K)]

        # Parameter gradient updates
        for j, s in enumerate(sens_list):
            if n_obs == n_y:
                dZ_t = s.dZ
                dd_t = s.dd
                dH_t = s.dH
            else:
                dZ_t = s.dZ[obs, :]
                dd_t = s.dd[obs]
                dH_t = s.dH[np.ix_(obs, obs)]

            dv_j = -dZ_t @ a_curr - Z_t @ da_curr[j] - dd_t
            dF_j = dZ_t @ P_curr @ Z_t.T + Z_t @ dP_curr[j] @ Z_t.T + Z_t @ P_curr @ dZ_t.T + dH_t
            dF_j = 0.5 * (dF_j + dF_j.T)

            # Likelihood score contribution at t:
            # dll_t = -0.5 * [ tr(F^{-1} dF) + 2 dv' F^{-1} v - (F^{-1} v)' dF (F^{-1} v) ]
            tr_term = float(np.sum(F_inv * dF_j))  # equivalent to trace(F_inv @ dF_j)
            grad_v_term = 2.0 * float(dv_j @ F_inv_v)
            grad_F_term = float(F_inv_v @ dF_j @ F_inv_v)

            dll_j = -0.5 * (tr_term + grad_v_term - grad_F_term)
            score[j] += dll_j

            # Sensitivities of Kalman gain and closed-loop transition
            dM_j = s.dT @ P_curr @ Z_t.T + T_mat @ dP_curr[j] @ Z_t.T + T_mat @ P_curr @ dZ_t.T
            dK_j = (dM_j - K_t @ dF_j) @ F_inv
            dL_j = s.dT - dK_j @ Z_t - K_t @ dZ_t

            dRQR_j = (
                s.dR @ Q_mat @ R_mat.T
                + R_mat @ s.dQ @ R_mat.T
                + R_mat @ Q_mat @ s.dR.T
            )

            # Joseph-stabilized covariance sensitivity update
            da_next[j] = s.dT @ a_curr + T_mat @ da_curr[j] + s.dc + dK_j @ v + K_t @ dv_j
            dP_next[j] = (
                dL_j @ P_curr @ L_t.T
                + L_t @ dP_curr[j] @ L_t.T
                + L_t @ P_curr @ dL_j.T
                + dK_j @ H_t @ K_t.T
                + K_t @ dH_t @ K_t.T
                + K_t @ H_t @ dK_j.T
                + dRQR_j
            )
            dP_next[j] = 0.5 * (dP_next[j] + dP_next[j].T)

        # Update baseline filter state and covariance for t+1
        a_curr = T_mat @ a_curr + c_vec + K_t @ v
        P_curr = L_t @ P_curr @ L_t.T + K_t @ H_t @ K_t.T + RQR
        P_curr = 0.5 * (P_curr + P_curr.T)

        da_curr = da_next
        dP_curr = dP_next

    return float(loglik), score


# ---------------------------------------------------------------------------
# Decision Rules and State Space Sensitivity Builder
# ---------------------------------------------------------------------------

def _re_solve_model(model: Any, params: dict) -> Any:
    """Re-solve a DSGE model at perturbed parameter values."""
    if getattr(model, "_dynare_equations", None) is not None:
        from puremacro.dsge.dynare import build_dynare
        return build_dynare(
            model._dynare_equations,
            variables=model.variables,
            shocks=model.shocks,
            params=params,
            steady_state=model.steady_state,
            states=model.states,
            check_steady_state=False,
            strict=False,
        )
    elif getattr(model, "_equations", None) is not None:
        from puremacro.dsge.build import build as _build
        units = getattr(model, "units", {}) or {}
        lin = units.get(model.variables[0], "level") if units else "level"
        return _build(
            model._equations,
            variables=model.variables,
            states=model.states,
            shocks=model.shocks,
            params=params,
            guess=model.steady_state.to_dict() if hasattr(model.steady_state, "to_dict") else model.steady_state,
            linearize=lin,
            verify_derivatives=False,
            strict=False,
        )
    return model


def compute_decision_rule_derivatives(
    model: Any,
    param_name: str,
    h_eps: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute analytical parameter derivatives of decision rules \frac{\partial ghx}{\partial \theta} and \frac{\partial ghu}{\partial \theta}.

    Parameters
    ----------
    model : LinearModel
        The solved first-order DSGE model.
    param_name : str
        The structural parameter name to differentiate.
    h_eps : float, default 1e-7
        Probe step for differentiation.

    Returns
    -------
    dghx : np.ndarray of shape (N_var, N_states)
        Derivative \frac{\partial ghx}{\partial \theta}.
    dghu : np.ndarray of shape (N_var, N_shocks)
        Derivative \frac{\partial ghu}{\partial \theta}.
    """
    params_dict = dict(getattr(model, "_params", {}) or getattr(model, "params", {}) or {})
    theta_0 = float(params_dict.get(param_name, 0.0))
    h = max(1e-6, 1e-5 * abs(theta_0))

    p_plus = dict(params_dict); p_plus[param_name] = theta_0 + h
    p_minus = dict(params_dict); p_minus[param_name] = theta_0 - h

    m_plus = _re_solve_model(model, p_plus)
    m_minus = _re_solve_model(model, p_minus)

    if hasattr(m_plus, "dynare_dr") and m_plus.dynare_dr is not None:
        ghx_p = np.asarray(m_plus.dynare_dr.ghx, dtype=float)
        ghx_m = np.asarray(m_minus.dynare_dr.ghx, dtype=float)
        ghu_p = np.asarray(m_plus.dynare_dr.ghu, dtype=float)
        ghu_m = np.asarray(m_minus.dynare_dr.ghu, dtype=float)
        dghx = (ghx_p - ghx_m) / (2.0 * h)
        dghu = (ghu_p - ghu_m) / (2.0 * h)
        return dghx, dghu

    if hasattr(m_plus, "solution") and hasattr(m_plus.solution, "G"):
        G_p = np.asarray(m_plus.solution.G, dtype=float)
        G_m = np.asarray(m_minus.solution.G, dtype=float)
        N_p = np.asarray(m_plus.solution.N, dtype=float)
        N_m = np.asarray(m_minus.solution.N, dtype=float)
        return (G_p - G_m) / (2.0 * h), (N_p - N_m) / (2.0 * h)

    raise ValueError("Model does not expose decision rules.")


def build_state_space_sensitivities(
    model: Any,
    varobs: Sequence[str],
    param_names: Sequence[str],
    shock_cov: np.ndarray | None = None,
    measurement_errors: Mapping[str, float] | None = None,
) -> dict[str, StateSpaceSensitivity]:
    r"""Build exact analytical StateSpaceSensitivity objects for all specified parameters.

    Parameters
    ----------
    model : LinearModel
        The solved DSGE model.
    varobs : Sequence[str]
        Observable variable names.
    param_names : Sequence[str]
        Names of structural parameters, shock standard errors (stderr e), or measurement errors.
    shock_cov : np.ndarray, optional
        Baseline covariance matrix of exogenous shocks.
    measurement_errors : Mapping[str, float], optional
        Baseline measurement error standard deviations for observables.

    Returns
    -------
    sensitivities : dict[str, StateSpaceSensitivity]
        Dictionary mapping each parameter name to its StateSpaceSensitivity.
    """
    from puremacro.dsge.observation import make_state_space_from_varobs

    obs = list(varobs)
    n_obs = len(obs)
    shocks = list(model.shocks)
    n_e = len(shocks)
    states = list(model.states)
    n_s = len(states)
    variables = list(model.variables)
    m_dim = n_s + n_e

    base_ssm = make_state_space_from_varobs(
        model, obs, shock_cov=shock_cov, measurement_error=measurement_errors
    )

    sens_dict: dict[str, StateSpaceSensitivity] = {}

    for name in param_names:
        dT = np.zeros((m_dim, m_dim))
        dZ = np.zeros((n_obs, m_dim))
        dR = np.zeros((m_dim, n_e))
        dQ = np.zeros((n_e, n_e))
        dH = np.zeros((n_obs, n_obs))
        dc = np.zeros(m_dim)
        dd = np.zeros(n_obs)

        # Case 1: Measurement error standard deviation: stderr obs_name
        if name.startswith("stderr_") and name[7:] in obs:
            obs_nm = name[7:]
            k = obs.index(obs_nm)
            sigma_me = float(measurement_errors.get(obs_nm, 0.0)) if measurement_errors else 0.0
            dH[k, k] = 2.0 * sigma_me if sigma_me > 0 else 1.0
            sens_dict[name] = StateSpaceSensitivity(
                dT=dT, dZ=dZ, dR=dR, dQ=dQ, dH=dH, dc=dc, dd=dd
            )
            continue

        # Case 2: Shock standard deviation: stderr shock_name
        if name.startswith("stderr_") and name[7:] in shocks:
            sh_nm = name[7:]
            k = shocks.index(sh_nm)
            sigma_s = math.sqrt(max(base_ssm.Q[k, k], 0.0))
            dQ[k, k] = 2.0 * sigma_s if sigma_s > 0 else 1.0
            sens_dict[name] = StateSpaceSensitivity(
                dT=dT, dZ=dZ, dR=dR, dQ=dQ, dH=dH, dc=dc, dd=dd
            )
            continue

        # Case 3: Structural parameter
        dghx, dghu = compute_decision_rule_derivatives(model, name)

        state_indices = [variables.index(s) for s in states]
        obs_indices = [variables.index(o) for o in obs]

        dT[:n_s, :n_s] = dghx[state_indices, :]
        dT[:n_s, n_s:] = dghu[state_indices, :]

        dZ[:, :n_s] = dghx[obs_indices, :]
        dZ[:, n_s:] = dghu[obs_indices, :]

        sens_dict[name] = StateSpaceSensitivity(
            dT=dT, dZ=dZ, dR=dR, dQ=dQ, dH=dH, dc=dc, dd=dd
        )

    return sens_dict


# ---------------------------------------------------------------------------
# Diagnostics and Result Containers
# ---------------------------------------------------------------------------

@dataclass
class ScoreDiagnosticsResult:
    """Diagnostic report for analytical likelihood gradients against numerical verification."""

    loglik: float
    gradient: np.ndarray
    param_names: tuple[str, ...]
    elapsed_sec: float
    verification_df: pd.DataFrame | None = None

    def verify_numerical_gradient(
        self,
        y: np.ndarray,
        ssm_func: Callable[[dict], StateSpaceModel],
        base_params: dict,
        h: float = 1e-6,
    ) -> pd.DataFrame:
        """Compare analytical gradient against 5-point central finite differences."""
        names = self.param_names
        K = len(names)
        records = []

        for i, nm in enumerate(names):
            val0 = float(base_params[nm])
            step = max(h, h * abs(val0))

            def eval_ll(delta: float) -> float:
                p = dict(base_params)
                p[nm] = val0 + delta
                ssm = ssm_func(p)
                from puremacro.state_space import kalman_filter
                out = kalman_filter(y, ssm)
                return float(out["loglik"])

            f_p2 = eval_ll(2 * step)
            f_p1 = eval_ll(step)
            f_m1 = eval_ll(-step)
            f_m2 = eval_ll(-2 * step)

            fd_score = (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * step)
            an_score = float(self.gradient[i])

            abs_err = abs(an_score - fd_score)
            rel_err = abs_err / max(1.0, abs(fd_score))
            passed = rel_err < 1e-4 or abs_err < 1e-5

            records.append({
                "parameter": nm,
                "analytical": an_score,
                "numerical_fd": fd_score,
                "abs_error": abs_err,
                "rel_error": rel_err,
                "status": "PASS" if passed else "WARN",
            })

        df = pd.DataFrame(records)
        self.verification_df = df
        return df

    def to_markdown(self) -> str:
        """Format score diagnostics as a clean Markdown table."""
        if self.verification_df is not None:
            return _df_to_markdown(self.verification_df, digits=6)
        df = pd.DataFrame({
            "parameter": list(self.param_names),
            "score": list(self.gradient),
        })
        return _df_to_markdown(df, digits=6)

    def to_latex(self) -> str:
        """Format score diagnostics as LaTeX tabular."""
        df = self.verification_df if self.verification_df is not None else pd.DataFrame({
            "parameter": list(self.param_names),
            "score": list(self.gradient),
        })
        return _df_to_latex(df, digits=6)


# ---------------------------------------------------------------------------
# Combined Posterior and Analytical Gradient Evaluator
# ---------------------------------------------------------------------------

def log_posterior_and_gradient(
    vec: np.ndarray,
    names: Sequence[str],
    model_template: Any,
    y_data: np.ndarray,
    varobs: Sequence[str],
    priors: dict,
    fixed_params: dict | None = None,
) -> tuple[float, np.ndarray]:
    r"""Evaluate combined log-posterior \ln p(\theta \mid Y) and analytical gradient \nabla_\theta \ln p(\theta \mid Y).

    Parameters
    ----------
    vec : np.ndarray of shape (K,)
        Estimated parameter vector.
    names : Sequence[str]
        Names corresponding to elements of vec.
    model_template : LinearModel or ParsedModelDAG
        Template model for re-evaluating structural matrices.
    y_data : np.ndarray of shape (T_obs, n_y)
        Observed series data.
    varobs : Sequence[str]
        Observable variable names.
    priors : dict
        Prior distributions dictionary.
    fixed_params : dict, optional
        Fixed parameters dictionary.

    Returns
    -------
    log_post : float
        Combined log-posterior value \ln L(Y \mid \theta) + \ln p(\theta).
    grad_post : np.ndarray of shape (K,)
        Combined analytical gradient vector \nabla_\theta \ln p(\theta \mid Y).
    """
    param_dict = dict(getattr(model_template, "_params", {}) or {})
    if fixed_params:
        param_dict.update(fixed_params)
    for nm, v in zip(names, vec):
        param_dict[nm] = float(v)

    # 1. Log-prior and prior gradient
    lp = log_prior(param_dict, priors)
    if not math.isfinite(lp):
        return -math.inf, np.zeros(len(names))

    g_prior = grad_log_prior(vec, priors, names)

    # 2. Re-solve model at current parameters
    from puremacro.dsge.observation import make_state_space_from_varobs

    try:
        if getattr(model_template, "_dynare_equations", None) is not None:
            from puremacro.dsge.dynare import build_dynare
            m_curr = build_dynare(
                model_template._dynare_equations,
                variables=model_template.variables,
                shocks=model_template.shocks,
                params=param_dict,
                steady_state=model_template.steady_state,
                check_steady_state=False,
                strict=False,
            )
        elif getattr(model_template, "_equations", None) is not None:
            from puremacro.dsge.build import build as _build
            m_curr = _build(
                model_template._equations,
                variables=model_template.variables,
                states=model_template.states,
                shocks=model_template.shocks,
                params=param_dict,
                guess=model_template.steady_state.to_dict() if hasattr(model_template.steady_state, "to_dict") else model_template.steady_state,
                check_steady_state=False,
                strict=False,
            )
        else:
            m_curr = model_template
    except Exception:
        return -math.inf, np.zeros(len(names))

    # 3. Build state-space and sensitivities
    try:
        ssm = make_state_space_from_varobs(m_curr, varobs)
        sens = build_state_space_sensitivities(m_curr, varobs, names)
        ll, g_ll = kalman_score(y_data, ssm, sens)
    except Exception:
        return -math.inf, np.zeros(len(names))

    if not math.isfinite(ll):
        return -math.inf, np.zeros(len(names))

    total_log_post = float(ll + lp)
    total_grad = g_ll + g_prior

    return total_log_post, total_grad
