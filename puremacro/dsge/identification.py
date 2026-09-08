"""Iskrev (2010) and Ratto (2011) DSGE parameter identification analysis.

Evaluates:
- Solution Jacobian J1 = d vec(T, R, Q, Z, H) / d theta (reduced-form state-space).
- Theoretical moment Jacobian J2 = d m(theta) / d theta with moments
  m(theta) = [ys_O(theta); vec(Gamma_0(theta)); ...; vec(Gamma_p(theta))].
- Fast differentiation for shock standard errors and measurement errors
  skipping Klein QZ rational expectations re-solve.
- SVD rank deficiency detection, null space linear parameter combinations,
  multi-way collinearity R^2, and Ratto (2011) identification strength.
"""
from __future__ import annotations

import difflib
import math
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.linalg

from puremacro.dsge._estimated_params import EstimatedParams, EstimatedParamSpec
from puremacro.dsge._results import IdentificationResult
from puremacro.dsge.observation import make_state_space_from_varobs

__all__ = ["identification"]


@dataclass(frozen=True)
class _ResolvedParam:
    name: str
    kind: str  # "param", "stderr_shock", "corr_shock", "stderr_obs"
    target: tuple[str, ...]
    base_val: float
    prior: Any | None = None
    lb: float | None = None
    ub: float | None = None


def _resolve_observables(model: Any, varobs: Sequence[str] | None) -> list[str]:
    """Resolve observable names against model variables."""
    if varobs is not None:
        names = list(varobs)
    elif getattr(model, "_varobs", None):
        names = list(model._varobs)
    else:
        names = list(model.variables)

    if not names:
        raise ValueError("identification() requires at least one observable variable.")

    known = list(model.variables)
    unknown = [v for v in names if v not in known]
    if unknown:
        hints = []
        for v in unknown:
            close = difflib.get_close_matches(v, known, n=3, cutoff=0.5)
            hints.append(f"{v!r}" + (f" (did you mean {close}?)" if close else ""))
        raise ValueError(
            f"identification(): {', '.join(hints)} not in model variables: {known}"
        )

    return names


def _resolve_params(
    model: Any,
    params: Sequence[str] | Mapping[str, float] | EstimatedParams | None,
    varobs: list[str],
) -> list[_ResolvedParam]:
    """Parse and resolve parameters into typed _ResolvedParam instances."""
    declared_specs: dict[str, EstimatedParamSpec] = {}
    if getattr(model, "_estimated_params", None) is not None:
        declared_specs = {sp.name: sp for sp in model._estimated_params.specs}

    model_params = dict(getattr(model, "_params", {}) or {})
    shocks = list(model.shocks)

    resolved: list[_ResolvedParam] = []

    if params is None:
        if declared_specs:
            for sp in declared_specs.values():
                val = sp.start if sp.start is not None else (
                    model_params.get(sp.name, 1.0)
                )
                resolved.append(
                    _ResolvedParam(
                        name=sp.name,
                        kind=sp.kind,
                        target=sp.target,
                        base_val=float(val),
                        prior=sp.prior,
                        lb=sp.lb,
                        ub=sp.ub,
                    )
                )
        elif model_params:
            for k, v in model_params.items():
                resolved.append(
                    _ResolvedParam(
                        name=k,
                        kind="param",
                        target=(k,),
                        base_val=float(v),
                    )
                )
        else:
            raise ValueError(
                "identification(): no parameters provided and model has no declared parameters."
            )
        return resolved

    if isinstance(params, EstimatedParams):
        for sp in params.specs:
            val = sp.start if sp.start is not None else model_params.get(sp.name, 1.0)
            resolved.append(
                _ResolvedParam(
                    name=sp.name,
                    kind=sp.kind,
                    target=sp.target,
                    base_val=float(val),
                    prior=sp.prior,
                    lb=sp.lb,
                    ub=sp.ub,
                )
            )
        return resolved

    # Check if params is a sequence of EstimatedParamSpec
    if isinstance(params, Sequence) and not isinstance(params, (str, bytes)):
        first_item = params[0] if len(params) > 0 else None
        if isinstance(first_item, EstimatedParamSpec):
            for sp in params:  # type: ignore[union-attr]
                val = sp.start if sp.start is not None else model_params.get(sp.name, 1.0)
                resolved.append(
                    _ResolvedParam(
                        name=sp.name,
                        kind=sp.kind,
                        target=sp.target,
                        base_val=float(val),
                        prior=sp.prior,
                        lb=sp.lb,
                        ub=sp.ub,
                    )
                )
            return resolved

    param_mapping: Mapping[str, float | None]
    if isinstance(params, Mapping):
        param_mapping = {k: float(v) for k, v in params.items()}
    elif isinstance(params, Sequence) and not isinstance(params, (str, bytes)):
        param_mapping = {str(k): None for k in params}
    else:
        raise TypeError(
            f"identification(): expected params to be Sequence, Mapping, or EstimatedParams, got {type(params).__name__}"
        )

    for name, user_val in param_mapping.items():
        if name in declared_specs:
            sp = declared_specs[name]
            base_v = user_val if user_val is not None else (
                sp.start if sp.start is not None else model_params.get(name, 1.0)
            )
            resolved.append(
                _ResolvedParam(
                    name=name,
                    kind=sp.kind,
                    target=sp.target,
                    base_val=float(base_v),
                    prior=sp.prior,
                    lb=sp.lb,
                    ub=sp.ub,
                )
            )
        elif name in model_params:
            base_v = user_val if user_val is not None else model_params[name]
            resolved.append(
                _ResolvedParam(
                    name=name,
                    kind="param",
                    target=(name,),
                    base_val=float(base_v),
                )
            )
        elif name.startswith("SE_") and name[3:] in shocks:
            sh_name = name[3:]
            base_v = user_val if user_val is not None else 1.0
            resolved.append(
                _ResolvedParam(
                    name=name,
                    kind="stderr_shock",
                    target=(sh_name,),
                    base_val=float(base_v),
                    lb=0.0,
                )
            )
        elif name.startswith("CORR_"):
            parts = name[5:].split("_")
            if len(parts) == 2 and parts[0] in shocks and parts[1] in shocks:
                base_v = user_val if user_val is not None else 0.0
                resolved.append(
                    _ResolvedParam(
                        name=name,
                        kind="corr_shock",
                        target=(parts[0], parts[1]),
                        base_val=float(base_v),
                        lb=-1.0,
                        ub=1.0,
                    )
                )
            else:
                raise ValueError(
                    f"identification(): invalid shock correlation name '{name}'. Shocks: {shocks}"
                )
        elif name.startswith("ME_") and name[3:] in varobs:
            obs_name = name[3:]
            base_v = user_val if user_val is not None else 0.0
            resolved.append(
                _ResolvedParam(
                    name=name,
                    kind="stderr_obs",
                    target=(obs_name,),
                    base_val=float(base_v),
                    lb=0.0,
                )
            )
        else:
            raise ValueError(
                f"identification(): unknown parameter '{name}'. "
                f"Model parameters: {sorted(model_params)}. Shocks: {shocks}. Observables: {varobs}."
            )

    return resolved


def _compute_state_space_psi(ssm: Any) -> np.ndarray:
    """Flatten StateSpaceModel matrices into psi1 = vec(T, R, Q, Z, H)."""
    return np.concatenate([
        np.asarray(ssm.T, dtype=float).ravel(),
        np.asarray(ssm.R, dtype=float).ravel(),
        np.asarray(ssm.Q, dtype=float).ravel(),
        np.asarray(ssm.Z, dtype=float).ravel(),
        np.asarray(ssm.H, dtype=float).ravel(),
    ])


def _compute_moments_psi(
    T: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    Z: np.ndarray,
    H: np.ndarray,
    d: np.ndarray,
    lags: int,
) -> np.ndarray:
    """Compute Iskrev (2010) theoretical moments m = [ys_O; vec(Gamma_0); ...; vec(Gamma_p)]."""
    T = np.asarray(T, dtype=float)
    R = np.asarray(R, dtype=float)
    Q = np.asarray(Q, dtype=float)
    Z = np.asarray(Z, dtype=float)
    H = np.asarray(H, dtype=float)
    d = np.asarray(d, dtype=float)

    # Check stability
    eigs = np.abs(scipy.linalg.eigvals(T))
    if np.any(eigs >= 1.0 - 1e-7):
        raise ValueError(
            "State transition matrix T has eigenvalues >= 1.0; stationary moments do not exist."
        )

    # Discrete Lyapunov solve on state space: Sigma_alpha = T Sigma_alpha T' + R Q R'
    sigma_alpha = scipy.linalg.solve_discrete_lyapunov(T, R @ Q @ R.T)
    sigma_alpha = 0.5 * (sigma_alpha + sigma_alpha.T)

    gamma_0 = Z @ sigma_alpha @ Z.T + H
    gamma_0 = 0.5 * (gamma_0 + gamma_0.T)

    moments = [d.ravel(), gamma_0.ravel()]

    T_k = T.copy()
    for _ in range(lags):
        gamma_k = Z @ T_k @ sigma_alpha @ Z.T
        moments.append(gamma_k.ravel())
        T_k = T_k @ T

    return np.concatenate(moments)


def _format_null_combination(
    vec: np.ndarray,
    param_names: Sequence[str],
    tol: float = 1e-4,
) -> str:
    """Format orthonormal null-space vector as human-readable parameter equation."""
    # Ensure positive leading non-zero coefficient
    nz_indices = np.where(np.abs(vec) > tol)[0]
    if len(nz_indices) == 0:
        return "0 = 0"

    v = vec.copy()
    if v[nz_indices[0]] < 0:
        v = -v

    terms = []
    for idx in nz_indices:
        coef = v[idx]
        name = param_names[idx]
        if len(terms) == 0:
            terms.append(f"{abs(coef):.4f} * {name}")
        else:
            if coef >= 0:
                terms.append(f"+ {coef:.4f} * {name}")
            else:
                terms.append(f"- {abs(coef):.4f} * {name}")

    return " ".join(terms) + " = 0"


def _analyze_jacobian(
    J: np.ndarray,
    param_names: tuple[str, ...],
    tol: float = 1e-8,
) -> tuple[int, np.ndarray, tuple[str, ...], pd.DataFrame]:
    """Perform SVD rank, null-space extraction, and multi-way collinearity on Jacobian."""
    n_rows, n_params = J.shape
    if n_params == 0:
        return (
            0,
            np.zeros((0, 0)),
            (),
            pd.DataFrame(columns=["r2", "pairwise_r2", "worst_partner"]),
        )

    # SVD for numerical rank and null space
    U, s, Vt = scipy.linalg.svd(J, full_matrices=False)
    s_max = float(s[0]) if len(s) > 0 else 0.0

    if s_max <= 1e-15:
        rank = 0
        null_space = np.eye(n_params)
    else:
        rank_thresh = max(tol * s_max, 1e-14)
        rank = int(np.sum(s > rank_thresh))
        null_space = Vt[rank:, :].copy()

    # Standardize sign of null space vectors so leading non-zero element is positive
    for row_idx in range(null_space.shape[0]):
        nz = np.where(np.abs(null_space[row_idx]) > tol)[0]
        if len(nz) > 0 and null_space[row_idx, nz[0]] < 0:
            null_space[row_idx] = -null_space[row_idx]

    # Format human-readable null combinations
    null_combinations: list[str] = []
    for row_idx in range(null_space.shape[0]):
        comb = _format_null_combination(null_space[row_idx], param_names)
        null_combinations.append(comb)

    # Multi-way collinearity R^2 via OLS regression: J_j ~ J_{-j}
    collinearity_records = []
    for j in range(n_params):
        y = J[:, j]
        y_norm_sq = float(np.sum(y**2))

        if y_norm_sq < 1e-24:
            r2 = 0.0
        elif n_params == 1:
            r2 = 0.0
        else:
            other_indices = [i for i in range(n_params) if i != j]
            X = J[:, other_indices]
            beta, resids, _, _ = scipy.linalg.lstsq(X, y)
            res = y - X @ beta
            ss_res = float(np.sum(res**2))
            r2 = 1.0 - (ss_res / y_norm_sq)
            r2 = max(0.0, min(1.0, float(r2)))
            if ss_res < 1e-12 * y_norm_sq:
                r2 = 1.0

        # Pairwise collinearity with all other parameters
        worst_pair_r2 = 0.0
        worst_partner = "None"
        if n_params > 1 and y_norm_sq >= 1e-24:
            for k in range(n_params):
                if k == j:
                    continue
                w = J[:, k]
                w_norm_sq = float(np.sum(w**2))
                if w_norm_sq >= 1e-24:
                    pair_r2 = float((np.dot(y, w)**2) / (y_norm_sq * w_norm_sq))
                    if pair_r2 > worst_pair_r2:
                        worst_pair_r2 = pair_r2
                        worst_partner = param_names[k]

        collinearity_records.append({
            "r2": r2,
            "pairwise_r2": worst_pair_r2,
            "worst_partner": worst_partner,
        })

    collinearity_df = pd.DataFrame(collinearity_records, index=list(param_names))
    return rank, null_space, tuple(null_combinations), collinearity_df


def _compute_ratto_strength(
    J: np.ndarray,
    param_names: tuple[str, ...],
    collinearity_df: pd.DataFrame,
    resolved_params: list[_ResolvedParam],
) -> pd.DataFrame:
    """Calculate Ratto (2011) identification strength and sensitivity."""
    n_params = len(param_names)
    sensitivities = np.zeros(n_params)
    strengths = np.zeros(n_params)
    norm_strengths = np.zeros(n_params)

    for j in range(n_params):
        col = J[:, j]
        sens = float(np.linalg.norm(col))
        r2 = float(collinearity_df.loc[param_names[j], "r2"]) if "r2" in collinearity_df.columns else 0.0
        st = sens * math.sqrt(max(0.0, 1.0 - r2))
        sensitivities[j] = sens
        strengths[j] = st

    max_sens = float(np.max(sensitivities)) if n_params > 0 else 0.0
    if max_sens > 1e-14:
        norm_strengths = strengths / max_sens
    else:
        norm_strengths = np.zeros(n_params)

    # Calculate rank (1 = highest identification strength)
    order = np.argsort(-strengths)
    ranks = np.empty(n_params, dtype=int)
    ranks[order] = np.arange(1, n_params + 1)

    records = []
    for j, name in enumerate(param_names):
        records.append({
            "sensitivity": sensitivities[j],
            "collinearity_r2": float(collinearity_df.loc[name, "r2"]),
            "strength": strengths[j],
            "normalized_strength": norm_strengths[j],
            "rank": int(ranks[j]),
        })

    return pd.DataFrame(records, index=list(param_names))


def identification(
    model: Any,
    *,
    varobs: Sequence[str] | None = None,
    params: Sequence[str] | Mapping[str, float] | EstimatedParams | None = None,
    lags: int = 1,
    tol: float = 1e-8,
    prior_mc: int = 0,
    seed: int = 0,
) -> IdentificationResult:
    """Perform Iskrev (2010) and Ratto (2011) parameter identification analysis.

    Evaluates the reduced-form state-space Jacobian:
        J1 = d vec(T, R, Q, Z, H) / d theta
    and the theoretical autocovariance moment Jacobian:
        J2 = d m(theta) / d theta
    where m(theta) = [ys_O(theta); vec(Gamma_0(theta)); ...; vec(Gamma_p(theta))].

    Parameters affecting only shock standard deviations (Q) or measurement
    error standard deviations (H) are differentiated algebraically without
    re-solving the rational expectations model.

    Parameters
    ----------
    model : LinearModel
        The solved linear DSGE model.
    varobs : Sequence[str], optional
        Observable variable names. Defaults to model._varobs or model.variables.
    params : Sequence[str] | Mapping[str, float] | EstimatedParams, optional
        Parameters to evaluate. Defaults to model._estimated_params or calibrated parameters.
    lags : int, default 1
        Autocovariance lags included in theoretical moments J2.
    tol : float, default 1e-8
        Relative SVD tolerance for numerical rank determination (s_i / s_1 <= tol).
    prior_mc : int, default 0
        If > 0, performs identification analysis over a Monte Carlo sample of prior draws.
    seed : int, default 0
        Random seed for prior Monte Carlo simulation.

    Returns
    -------
    IdentificationResult
        Frozen dataclass with rank, null spaces, collinearity, and presentation methods.
    """
    obs = _resolve_observables(model, varobs)
    resolved = _resolve_params(model, params, obs)
    param_names = tuple(p.name for p in resolved)
    n_params = len(resolved)

    shocks = list(model.shocks)
    n_e = len(shocks)
    base_cov = (
        np.asarray(model._shock_cov, dtype=float).copy()
        if getattr(model, "_shock_cov", None) is not None
        else np.eye(n_e)
    )
    base_me: dict[str, float] = {}

    # Initialize baseline state-space model and moments
    base_ssm = make_state_space_from_varobs(
        model, obs, shock_cov=base_cov, measurement_error=base_me or None
    )
    psi1_base = _compute_state_space_psi(base_ssm)
    psi2_base = _compute_moments_psi(
        base_ssm.T, base_ssm.R, base_ssm.Q, base_ssm.Z, base_ssm.H, base_ssm.d, lags
    )

    J1 = np.zeros((len(psi1_base), n_params))
    J2 = np.zeros((len(psi2_base), n_params))

    # Differentiation loop
    for j, p in enumerate(resolved):
        theta_0 = p.base_val
        h = max(1e-5, 1e-4 * abs(theta_0))

        if p.kind == "stderr_obs":
            # Fast analytical differentiation for measurement errors
            # H_ii = sigma_me^2 -> d H_ii / d sigma_me = 2 * sigma_me
            obs_idx = obs.index(p.target[0])
            d_H = 2.0 * theta_0 if abs(theta_0) > 1e-12 else 2.0 * h

            # Update J1: non-zero only in H block
            # H block is the last len(obs)^2 elements of psi1
            h_offset = base_ssm.T.size + base_ssm.R.size + base_ssm.Q.size + base_ssm.Z.size
            diag_pos = obs_idx * len(obs) + obs_idx
            J1[h_offset + diag_pos, j] = d_H

            # Update J2: non-zero only in Gamma_0 block
            # Gamma_0 block starts after d (size len(obs))
            g0_offset = len(obs)
            J2[g0_offset + diag_pos, j] = d_H
            continue

        elif p.kind == "stderr_shock":
            # Fast differentiation for shock standard errors: Q_ii = sigma_s^2
            sh_idx = shocks.index(p.target[0])
            d_Q = 2.0 * theta_0 if abs(theta_0) > 1e-12 else 2.0 * h

            # In J1: non-zero only in Q block
            q_offset = base_ssm.T.size + base_ssm.R.size
            diag_pos = sh_idx * n_e + sh_idx
            J1[q_offset + diag_pos, j] = d_Q

            # In J2: solve Lyapunov for derivative of state covariance
            # d Sigma_alpha = T d Sigma_alpha T' + R (d Q) R'
            dQ_mat = np.zeros((n_e, n_e))
            dQ_mat[sh_idx, sh_idx] = d_Q
            d_sigma_alpha = scipy.linalg.solve_discrete_lyapunov(
                base_ssm.T, base_ssm.R @ dQ_mat @ base_ssm.R.T
            )
            d_sigma_alpha = 0.5 * (d_sigma_alpha + d_sigma_alpha.T)

            d_gamma_0 = base_ssm.Z @ d_sigma_alpha @ base_ssm.Z.T
            d_gamma_0 = 0.5 * (d_gamma_0 + d_gamma_0.T)

            d_moments = [np.zeros_like(base_ssm.d), d_gamma_0.ravel()]
            T_k = base_ssm.T.copy()
            for _ in range(lags):
                d_gamma_k = base_ssm.Z @ T_k @ d_sigma_alpha @ base_ssm.Z.T
                d_moments.append(d_gamma_k.ravel())
                T_k = T_k @ base_ssm.T

            J2[:, j] = np.concatenate(d_moments)
            continue

        elif p.kind == "corr_shock":
            # Fast differentiation for shock correlations
            s1, s2 = p.target
            i1, i2 = shocks.index(s1), shocks.index(s2)
            std1 = math.sqrt(max(base_cov[i1, i1], 0.0))
            std2 = math.sqrt(max(base_cov[i2, i2], 0.0))
            d_cov = std1 * std2

            # In J1
            q_offset = base_ssm.T.size + base_ssm.R.size
            J1[q_offset + i1 * n_e + i2, j] = d_cov
            J1[q_offset + i2 * n_e + i1, j] = d_cov

            # In J2
            dQ_mat = np.zeros((n_e, n_e))
            dQ_mat[i1, i2] = d_cov
            dQ_mat[i2, i1] = d_cov
            d_sigma_alpha = scipy.linalg.solve_discrete_lyapunov(
                base_ssm.T, base_ssm.R @ dQ_mat @ base_ssm.R.T
            )
            d_sigma_alpha = 0.5 * (d_sigma_alpha + d_sigma_alpha.T)

            d_gamma_0 = base_ssm.Z @ d_sigma_alpha @ base_ssm.Z.T
            d_gamma_0 = 0.5 * (d_gamma_0 + d_gamma_0.T)
            d_moments = [np.zeros_like(base_ssm.d), d_gamma_0.ravel()]
            T_k = base_ssm.T.copy()
            for _ in range(lags):
                d_gamma_k = base_ssm.Z @ T_k @ d_sigma_alpha @ base_ssm.Z.T
                d_moments.append(d_gamma_k.ravel())
                T_k = T_k @ base_ssm.T

            J2[:, j] = np.concatenate(d_moments)
            continue

        else:
            # Structural parameter: requires re-solving the model
            def _solve_perturbed(val: float) -> tuple[np.ndarray, np.ndarray]:
                perturbed_params = dict(getattr(model, "_params", {}) or {})
                perturbed_params[p.name] = val
                guess = model.steady_state.to_dict()

                if getattr(model, "_dynare_equations", None) is not None:
                    from puremacro.dsge.dynare import build_dynare

                    m_pert = build_dynare(
                        model._dynare_equations,
                        variables=model.variables,
                        shocks=model.shocks,
                        params=perturbed_params,
                        guess=guess,
                        states=model.states,
                        order=1,
                        method=model.method,
                        verify_derivatives=False,
                        strict=False,
                    )
                elif getattr(model, "_equations", None) is not None:
                    from puremacro.dsge.build import build as _build

                    m_pert = _build(
                        model._equations,
                        variables=model.variables,
                        states=model.states,
                        shocks=model.shocks,
                        params=perturbed_params,
                        guess=guess,
                        method=model.method,
                        verify_derivatives=False,
                        strict=False,
                    )
                else:
                    raise ValueError(
                        "identification(): model carries neither _dynare_equations nor _equations; "
                        "cannot re-solve structural parameters."
                    )

                ssm_pert = make_state_space_from_varobs(
                    m_pert, obs, shock_cov=base_cov, measurement_error=base_me or None
                )
                psi1 = _compute_state_space_psi(ssm_pert)
                psi2 = _compute_moments_psi(
                    ssm_pert.T, ssm_pert.R, ssm_pert.Q, ssm_pert.Z, ssm_pert.H, ssm_pert.d, lags
                )
                return psi1, psi2

            # Try central difference; if bounds or solve fail, fall back gracefully
            use_central = True
            if p.lb is not None and theta_0 - h < p.lb:
                use_central = False
            if p.ub is not None and theta_0 + h > p.ub:
                use_central = False

            if use_central:
                try:
                    psi1_plus, psi2_plus = _solve_perturbed(theta_0 + h)
                    psi1_minus, psi2_minus = _solve_perturbed(theta_0 - h)
                    J1[:, j] = (psi1_plus - psi1_minus) / (2.0 * h)
                    J2[:, j] = (psi2_plus - psi2_minus) / (2.0 * h)
                except Exception:
                    use_central = False

            if not use_central:
                try:
                    psi1_plus, psi2_plus = _solve_perturbed(theta_0 + h)
                    J1[:, j] = (psi1_plus - psi1_base) / h
                    J2[:, j] = (psi2_plus - psi2_base) / h
                except Exception:
                    psi1_minus, psi2_minus = _solve_perturbed(theta_0 - h)
                    J1[:, j] = (psi1_base - psi1_minus) / h
                    J2[:, j] = (psi2_base - psi2_minus) / h

    # Analyze Jacobians J1 and J2
    j1_rank, j1_null, j1_combs, j1_coll = _analyze_jacobian(J1, param_names, tol=tol)
    j2_rank, j2_null, j2_combs, j2_coll = _analyze_jacobian(J2, param_names, tol=tol)

    # Identification strength on moment Jacobian J2
    strength_df = _compute_ratto_strength(J2, param_names, j2_coll, resolved)

    is_identified = bool((j1_rank == n_params) and (j2_rank == n_params))

    # Optional Prior Monte Carlo
    prior_mc_results = None
    if prior_mc > 0:
        rng = np.random.default_rng(seed)
        j1_mc_ranks = []
        j2_mc_ranks = []
        priors_list = [p.prior for p in resolved]

        for _ in range(prior_mc):
            # Draw sample from priors if available
            draw_params: dict[str, float] = {}
            for p, pr in zip(resolved, priors_list):
                if pr is not None and hasattr(pr, "rvs"):
                    draw_params[p.name] = float(pr.rvs(random_state=rng))
                elif pr is not None and hasattr(pr, "sample"):
                    draw_params[p.name] = float(pr.sample(rng))
                else:
                    # Perturb around base_val
                    draw_params[p.name] = float(p.base_val * (1.0 + rng.normal(0, 0.05)))

            try:
                mc_res = identification(
                    model,
                    varobs=obs,
                    params=draw_params,
                    lags=lags,
                    tol=tol,
                    prior_mc=0,
                )
                j1_mc_ranks.append(mc_res.j1_rank)
                j2_mc_ranks.append(mc_res.j2_rank)
            except Exception:
                continue

        if j1_mc_ranks:
            prior_mc_results = {
                "n_draws": len(j1_mc_ranks),
                "j1_identified_rate": float(np.mean(np.array(j1_mc_ranks) == n_params)),
                "j2_identified_rate": float(np.mean(np.array(j2_mc_ranks) == n_params)),
                "mean_j1_rank": float(np.mean(j1_mc_ranks)),
                "mean_j2_rank": float(np.mean(j2_mc_ranks)),
            }

    return IdentificationResult(
        is_identified=is_identified,
        j1_rank=j1_rank,
        j1_n_params=n_params,
        j1_null_space=j1_null,
        j1_null_combinations=j1_combs,
        j1_collinearity=j1_coll,
        j2_rank=j2_rank,
        j2_n_params=n_params,
        j2_null_space=j2_null,
        j2_null_combinations=j2_combs,
        j2_collinearity=j2_coll,
        strength=strength_df,
        param_names=param_names,
        varobs=tuple(obs),
        lags=lags,
        prior_mc_results=prior_mc_results,
    )
