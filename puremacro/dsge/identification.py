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
            rem = name[5:]
            matched_pair: tuple[str, str] | None = None
            for s1 in sorted(shocks, key=len, reverse=True):
                prefix = s1 + "_"
                if rem.startswith(prefix):
                    s2 = rem[len(prefix):]
                    if s2 in shocks:
                        matched_pair = (s1, s2)
                        break
            if matched_pair is not None:
                s1, s2 = matched_pair
                base_v = user_val if user_val is not None else 0.0
                resolved.append(
                    _ResolvedParam(
                        name=name,
                        kind="corr_shock",
                        target=(s1, s2),
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


def _cholesky_or_sqrt(Q: np.ndarray) -> np.ndarray:
    """Compute lower Cholesky factor or positive semidefinite square root of Q."""
    n_e = Q.shape[0]
    if n_e == 0:
        return np.zeros((0, 0))
    if np.allclose(Q, np.diag(np.diag(Q))):
        return np.diag(np.sqrt(np.maximum(np.diag(Q), 0.0)))
    try:
        return scipy.linalg.cholesky(Q, lower=True)
    except np.linalg.LinAlgError:
        vals, vecs = scipy.linalg.eigh(Q)
        vals = np.maximum(vals, 0.0)
        return vecs @ np.diag(np.sqrt(vals))


def _build_frequency_grid(
    n_freq: int = 16,
    frequencies: Sequence[float] | None = None,
    freq_type: str = "uniform",
) -> np.ndarray:
    """Build grid of evaluation frequencies omega in (0, pi)."""
    if frequencies is not None:
        omega = np.asarray(frequencies, dtype=float).ravel()
        if len(omega) == 0:
            raise ValueError("frequencies grid must contain at least one point.")
        return omega

    n_freq = max(1, int(n_freq))
    if freq_type == "gauss_legendre":
        x_nodes, _ = np.polynomial.legendre.leggauss(n_freq)
        return 0.5 * np.pi * x_nodes + 0.5 * np.pi
    else:  # uniform
        return np.linspace(0.0, np.pi, n_freq + 2)[1:-1]


def _compute_frequency_representations(
    T: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    Z: np.ndarray,
    H_me: np.ndarray,
    omega: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Compute transfer function and spectral density representations across frequency grid."""
    m = T.shape[0]
    n_y = Z.shape[0]
    L_Q = _cholesky_or_sqrt(Q)
    L_me = _cholesky_or_sqrt(H_me)
    I_m = np.eye(m)

    bar_H_list: list[np.ndarray] = []
    H_list: list[np.ndarray] = []
    psi_H_parts: list[np.ndarray] = []
    psi_S_parts: list[np.ndarray] = []

    tril_all = np.tril_indices(n_y)
    tril_strict = np.tril_indices(n_y, -1)

    for w in omega:
        resolvent = np.linalg.solve(np.exp(1j * w) * I_m - T, R)
        bar_H_w = Z @ resolvent
        H_shock = bar_H_w @ L_Q
        H_w = np.hstack([H_shock, L_me])
        S_w = (1.0 / (2.0 * np.pi)) * (H_shock @ H_shock.conj().T + H_me)

        bar_H_list.append(bar_H_w)
        H_list.append(H_w)

        psi_H_parts.append(H_w.real.ravel())
        psi_H_parts.append(H_w.imag.ravel())

        psi_S_parts.append(S_w.real[tril_all])
        psi_S_parts.append(S_w.imag[tril_strict])

    psi_H = np.concatenate(psi_H_parts) if psi_H_parts else np.zeros(0)
    psi_S = np.concatenate(psi_S_parts) if psi_S_parts else np.zeros(0)
    return psi_H, psi_S, bar_H_list, H_list


def _vectorize_H_derivative(dH_list: list[np.ndarray]) -> np.ndarray:
    """Vectorize list of transfer function derivative matrices across frequencies."""
    parts = []
    for dH in dH_list:
        parts.append(dH.real.ravel())
        parts.append(dH.imag.ravel())
    return np.concatenate(parts) if parts else np.zeros(0)


def _vectorize_S_derivative(dS_list: list[np.ndarray], n_y: int) -> np.ndarray:
    """Vectorize list of spectral density derivative matrices via non-redundant Hermitian stacking."""
    tril_all = np.tril_indices(n_y)
    tril_strict = np.tril_indices(n_y, -1)
    parts = []
    for dS in dS_list:
        parts.append(dS.real[tril_all])
        parts.append(dS.imag[tril_strict])
    return np.concatenate(parts) if parts else np.zeros(0)


def _detect_collinear_pairs(
    jacobians: dict[str, np.ndarray],
    param_names: tuple[str, ...],
    threshold: float = 0.95,
) -> tuple[tuple[str, str, float, str], ...]:
    """Identify pairs of parameters with pairwise collinearity R^2 > threshold across criteria."""
    n_params = len(param_names)
    pairs = []
    for crit_name, J in jacobians.items():
        if J.shape[1] != n_params:
            continue
        norms_sq = np.sum(J**2, axis=0)
        for j in range(n_params):
            for k in range(j + 1, n_params):
                if norms_sq[j] > 1e-24 and norms_sq[k] > 1e-24:
                    dot = np.dot(J[:, j], J[:, k])
                    r2 = float((dot**2) / (norms_sq[j] * norms_sq[k]))
                    if r2 > threshold:
                        pairs.append((param_names[j], param_names[k], r2, crit_name))
    pairs.sort(key=lambda x: -x[2])
    return tuple(pairs)


def _generate_warnings(
    ranks: dict[str, int],
    cond_numbers: dict[str, float],
    n_params: int,
    collinear_pairs: tuple[tuple[str, str, float, str], ...],
) -> tuple[str, ...]:
    """Generate structured diagnostic warnings for rank deficiency, ill-conditioning, and collinearity."""
    warn_list = []
    for crit, r in ranks.items():
        if r < n_params:
            warn_list.append(
                f"{crit} rank deficient ({r}/{n_params}, deficiency {n_params - r}): "
                f"unidentifiable parameter subspace detected."
            )
    for crit, cond in cond_numbers.items():
        if np.isinf(cond):
            warn_list.append(f"{crit} has infinite condition number (singular Jacobian).")
        elif cond > 1e4:
            warn_list.append(
                f"Near-singular condition number in {crit} (cond = {cond:.2e} > 1e4): "
                f"identification is ill-conditioned."
            )
    seen_pairs: set[tuple[str, str]] = set()
    for p1, p2, r2, crit in collinear_pairs:
        pair_key = (min(p1, p2), max(p1, p2))
        if pair_key not in seen_pairs:
            warn_list.append(
                f"High parameter collinearity between '{p1}' and '{p2}' (R^2 = {r2:.4f} in {crit})."
            )
            seen_pairs.add(pair_key)

    return tuple(warn_list)


def _analyze_jacobian(
    J: np.ndarray,
    param_names: tuple[str, ...],
    tol: float = 1e-8,
) -> tuple[int, np.ndarray, float, np.ndarray, tuple[str, ...], pd.DataFrame]:
    """Perform SVD rank, singular values, condition number, null-space extraction, and collinearity."""
    n_rows, n_params = J.shape
    if n_params == 0:
        return (
            0,
            np.zeros(0),
            1.0,
            np.zeros((0, 0)),
            (),
            pd.DataFrame(columns=["r2", "pairwise_r2", "worst_partner"]),
        )

    # SVD for numerical rank, singular values, and null space
    U, s, Vt = scipy.linalg.svd(J, full_matrices=False)
    s_full = np.zeros(n_params)
    s_full[: len(s)] = s
    s_max = float(s_full[0]) if len(s_full) > 0 else 0.0

    if s_max <= 1e-15:
        rank = 0
        null_space = np.eye(n_params)
        cond_number = float("inf")
    else:
        rank_thresh = max(tol * s_max, 1e-14)
        rank = int(np.sum(s_full > rank_thresh))
        if rank < n_params:
            if n_rows < n_params:
                _, _, Vt_full = scipy.linalg.svd(J, full_matrices=True)
                null_space = Vt_full[rank:, :].copy()
            else:
                null_space = Vt[rank:, :].copy()
            cond_number = float("inf")
        else:
            null_space = np.zeros((0, n_params))
            cond_number = float(s_full[0] / s_full[-1]) if s_full[-1] > 1e-15 else float("inf")

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
    return rank, s_full, cond_number, null_space, tuple(null_combinations), collinearity_df


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
    params: Sequence[str] | Mapping[str, float] | EstimatedParams | None = None,
    *,
    varobs: Sequence[str] | None = None,
    p_dict: Mapping[str, float] | None = None,
    lags: int = 1,
    n_freq: int = 16,
    frequencies: Sequence[float] | None = None,
    freq_type: str = "uniform",
    tol: float = 1e-8,
    prior_mc: int = 0,
    seed: int = 0,
) -> IdentificationResult:
    """Perform Iskrev (2010) and Komunjer & Ng (2011) parameter identification analysis.

    Evaluates four complementary rank identification criteria:
    1. J1 (Iskrev Solution): Reduced-form state space solution Jacobian
       d vec(T, R, Q, Z, H) / d theta.
    2. J2 (Iskrev Moments): Theoretical autocovariance moment Jacobian
       d [ys_O; vec(Gamma_0); ...; vec(Gamma_p)] / d theta.
    3. JH (Komunjer & Ng Transfer): Frequency-domain transfer function Jacobian
       d vec(H(omega)) / d theta across frequency grid omega in (0, pi).
    4. JS (Komunjer & Ng Spectrum): Frequency-domain cross-spectral density Jacobian
       d vec(S_y(omega)) / d theta across frequency grid omega in (0, pi).

    Parameters affecting shock standard deviations (Q) or measurement error
    variances (H) are differentiated analytically in both time and frequency
    domains without re-solving the rational expectations model.

    Parameters
    ----------
    model : LinearModel
        The solved linear DSGE model.
    params : Sequence[str] | Mapping[str, float] | EstimatedParams, optional
        Parameters to evaluate. Defaults to model._estimated_params or calibrated parameters.
    varobs : Sequence[str], optional
        Observable variable names. Defaults to model._varobs or model.variables.
    p_dict : Mapping[str, float], optional
        Convenience alias for params when passed as a dictionary.
    lags : int, default 1
        Autocovariance lags included in theoretical moments J2.
    n_freq : int, default 16
        Number of frequency grid points in (0, pi) for Komunjer-Ng criteria.
    frequencies : Sequence[float], optional
        Custom grid of evaluation frequencies in (0, pi). If specified, overrides n_freq.
    freq_type : {"uniform", "gauss_legendre"}, default "uniform"
        Type of automatic frequency grid in (0, pi).
    tol : float, default 1e-8
        Relative SVD tolerance for numerical rank determination (s_i / s_1 <= tol).
    prior_mc : int, default 0
        If > 0, performs identification analysis over a Monte Carlo sample of prior draws.
    seed : int, default 0
        Random seed for prior Monte Carlo simulation.

    Returns
    -------
    IdentificationResult
        Frozen dataclass with full singular value spectra, condition numbers,
        null space bases, collinear pairs, warnings, and publication tables.
    """
    if params is None and p_dict is not None:
        params = p_dict

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

    # Initialize baseline state-space model and representations
    base_ssm = make_state_space_from_varobs(
        model, obs, shock_cov=base_cov, measurement_error=base_me or None
    )
    psi1_base = _compute_state_space_psi(base_ssm)
    psi2_base = _compute_moments_psi(
        base_ssm.T, base_ssm.R, base_ssm.Q, base_ssm.Z, base_ssm.H, base_ssm.d, lags
    )

    omega = _build_frequency_grid(n_freq=n_freq, frequencies=frequencies, freq_type=freq_type)
    psiH_base, psiS_base, bar_H_base_list, H_base_list = _compute_frequency_representations(
        base_ssm.T, base_ssm.R, base_ssm.Q, base_ssm.Z, base_ssm.H, omega
    )

    J1 = np.zeros((len(psi1_base), n_params))
    J2 = np.zeros((len(psi2_base), n_params))
    JH = np.zeros((len(psiH_base), n_params))
    JS = np.zeros((len(psiS_base), n_params))

    n_y_sq = len(obs) ** 2
    tril_all = np.tril_indices(len(obs))

    # Differentiation loop
    for j, p in enumerate(resolved):
        theta_0 = p.base_val
        h = max(1e-5, 1e-4 * abs(theta_0))

        if p.kind == "stderr_obs":
            # Fast analytical differentiation for measurement errors
            obs_idx = obs.index(p.target[0])
            d_H = 2.0 * theta_0 if abs(theta_0) > 1e-12 else 2.0 * h

            # Update J1: non-zero only in H block
            h_offset = base_ssm.T.size + base_ssm.R.size + base_ssm.Q.size + base_ssm.Z.size
            diag_pos = obs_idx * len(obs) + obs_idx
            J1[h_offset + diag_pos, j] = d_H

            # Update J2: non-zero only in Gamma_0 block
            g0_offset = len(obs)
            J2[g0_offset + diag_pos, j] = d_H

            # Update JH: measurement noise direct feedthrough into observables
            dH_list = []
            for m_idx in range(len(omega)):
                dH_m = np.zeros_like(H_base_list[m_idx])
                dH_m[obs_idx, n_e + obs_idx] = 1.0
                dH_list.append(dH_m)
            JH[:, j] = _vectorize_H_derivative(dH_list)

            # Update JS: affects diagonal of spectral density matrix at all frequencies
            diag_mask = (tril_all[0] == obs_idx) & (tril_all[1] == obs_idx)
            diag_idx = int(np.where(diag_mask)[0][0])
            for m_idx in range(len(omega)):
                JS[m_idx * n_y_sq + diag_idx, j] = d_H / (2.0 * np.pi)
            continue

        elif p.kind == "stderr_shock":
            # Fast differentiation for shock standard errors: Q_ii = sigma_s^2
            sh_idx = shocks.index(p.target[0])
            d_Q = 2.0 * theta_0 if abs(theta_0) > 1e-12 else 2.0 * h

            # J1: non-zero only in Q block
            q_offset = base_ssm.T.size + base_ssm.R.size
            diag_pos = sh_idx * n_e + sh_idx
            J1[q_offset + diag_pos, j] = d_Q

            # J2: solve discrete Lyapunov for d Sigma_alpha
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

            # JH & JS: analytical differentiation of transfer and spectral density
            dH_list = []
            dS_list = []
            for m_idx in range(len(omega)):
                dH_m = np.zeros_like(H_base_list[m_idx])
                dH_m[:, sh_idx] = bar_H_base_list[m_idx][:, sh_idx]
                H_shock_base = H_base_list[m_idx][:, :n_e]
                dH_shock = dH_m[:, :n_e]
                dS_m = (1.0 / (2.0 * np.pi)) * (
                    dH_shock @ H_shock_base.conj().T + H_shock_base @ dH_shock.conj().T
                )
                dH_list.append(dH_m)
                dS_list.append(dS_m)

            JH[:, j] = _vectorize_H_derivative(dH_list)
            JS[:, j] = _vectorize_S_derivative(dS_list, len(obs))
            continue

        elif p.kind == "corr_shock":
            # Fast differentiation for shock correlations
            s1, s2 = p.target
            i1, i2 = shocks.index(s1), shocks.index(s2)
            std1 = math.sqrt(max(base_cov[i1, i1], 0.0))
            std2 = math.sqrt(max(base_cov[i2, i2], 0.0))
            d_cov = std1 * std2

            # J1
            q_offset = base_ssm.T.size + base_ssm.R.size
            J1[q_offset + i1 * n_e + i2, j] = d_cov
            J1[q_offset + i2 * n_e + i1, j] = d_cov

            # J2
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

            # JH & JS
            Q_p = base_ssm.Q.copy()
            Q_m = base_ssm.Q.copy()
            Q_p[i1, i2] += h * d_cov
            Q_p[i2, i1] += h * d_cov
            Q_m[i1, i2] -= h * d_cov
            Q_m[i2, i1] -= h * d_cov
            L_Q_p = _cholesky_or_sqrt(Q_p)
            L_Q_m = _cholesky_or_sqrt(Q_m)
            L_me = _cholesky_or_sqrt(base_ssm.H)
            dH_list = []
            dS_list = []
            for m_idx in range(len(omega)):
                Hp_shock = bar_H_base_list[m_idx] @ L_Q_p
                Hm_shock = bar_H_base_list[m_idx] @ L_Q_m
                Hp = np.hstack([Hp_shock, L_me])
                Hm = np.hstack([Hm_shock, L_me])
                dH_list.append((Hp - Hm) / (2.0 * h))
                Sp = (1.0 / (2.0 * np.pi)) * (Hp_shock @ Hp_shock.conj().T + base_ssm.H)
                Sm = (1.0 / (2.0 * np.pi)) * (Hm_shock @ Hm_shock.conj().T + base_ssm.H)
                dS_list.append((Sp - Sm) / (2.0 * h))
            JH[:, j] = _vectorize_H_derivative(dH_list)
            JS[:, j] = _vectorize_S_derivative(dS_list, len(obs))
            continue

        else:
            # Structural parameter: requires re-solving the model
            def _solve_perturbed(val: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
                psiH, psiS, _, _ = _compute_frequency_representations(
                    ssm_pert.T, ssm_pert.R, ssm_pert.Q, ssm_pert.Z, ssm_pert.H, omega
                )
                return psi1, psi2, psiH, psiS

            # Try central difference; if bounds or solve fail, fall back gracefully
            use_central = True
            if p.lb is not None and theta_0 - h < p.lb:
                use_central = False
            if p.ub is not None and theta_0 + h > p.ub:
                use_central = False

            if use_central:
                try:
                    psi1_plus, psi2_plus, psiH_plus, psiS_plus = _solve_perturbed(theta_0 + h)
                    psi1_minus, psi2_minus, psiH_minus, psiS_minus = _solve_perturbed(theta_0 - h)
                    J1[:, j] = (psi1_plus - psi1_minus) / (2.0 * h)
                    J2[:, j] = (psi2_plus - psi2_minus) / (2.0 * h)
                    JH[:, j] = (psiH_plus - psiH_minus) / (2.0 * h)
                    JS[:, j] = (psiS_plus - psiS_minus) / (2.0 * h)
                except Exception:
                    use_central = False

            if not use_central:
                try:
                    psi1_plus, psi2_plus, psiH_plus, psiS_plus = _solve_perturbed(theta_0 + h)
                    J1[:, j] = (psi1_plus - psi1_base) / h
                    J2[:, j] = (psi2_plus - psi2_base) / h
                    JH[:, j] = (psiH_plus - psiH_base) / h
                    JS[:, j] = (psiS_plus - psiS_base) / h
                except Exception:
                    psi1_minus, psi2_minus, psiH_minus, psiS_minus = _solve_perturbed(theta_0 - h)
                    J1[:, j] = (psi1_base - psi1_minus) / h
                    J2[:, j] = (psi2_base - psi2_minus) / h
                    JH[:, j] = (psiH_base - psiH_minus) / h
                    JS[:, j] = (psiS_base - psiS_minus) / h

    # SVD analysis across all 4 criteria
    j1_rank, j1_s, j1_cond, j1_null, j1_combs, j1_coll = _analyze_jacobian(J1, param_names, tol=tol)
    j2_rank, j2_s, j2_cond, j2_null, j2_combs, j2_coll = _analyze_jacobian(J2, param_names, tol=tol)
    jh_rank, jh_s, jh_cond, jh_null, jh_combs, jh_coll = _analyze_jacobian(JH, param_names, tol=tol)
    js_rank, js_s, js_cond, js_null, js_combs, js_coll = _analyze_jacobian(JS, param_names, tol=tol)

    # Identification strength on moment Jacobian J2
    strength_df = _compute_ratto_strength(J2, param_names, j2_coll, resolved)

    # Collinear parameter pairs across criteria
    jacobians = {"J1": J1, "J2": J2, "JH": JH, "JS": JS}
    collinear_pairs = _detect_collinear_pairs(jacobians, param_names, threshold=0.95)

    # Structured diagnostic warnings
    ranks = {"J1": j1_rank, "J2": j2_rank, "JH": jh_rank, "JS": js_rank}
    cond_numbers = {"J1": j1_cond, "J2": j2_cond, "JH": jh_cond, "JS": js_cond}
    warnings_tuple = _generate_warnings(ranks, cond_numbers, n_params, collinear_pairs)

    is_ident_sol = bool(j1_rank == n_params)
    is_ident_mom = bool(j2_rank == n_params)
    is_ident_tra = bool(jh_rank == n_params)
    is_ident_spe = bool(js_rank == n_params)
    is_identified = bool(is_ident_sol and is_ident_mom and is_ident_tra and is_ident_spe)

    # Optional Prior Monte Carlo
    prior_mc_results = None
    if prior_mc > 0:
        rng = np.random.default_rng(seed)
        j1_mc_ranks = []
        j2_mc_ranks = []
        jh_mc_ranks = []
        js_mc_ranks = []
        priors_list = [p.prior for p in resolved]

        for _ in range(prior_mc):
            draw_params: dict[str, float] = {}
            for p, pr in zip(resolved, priors_list):
                if pr is not None and hasattr(pr, "rvs"):
                    draw_params[p.name] = float(pr.rvs(random_state=rng))
                elif pr is not None and hasattr(pr, "sample"):
                    draw_params[p.name] = float(pr.sample(rng))
                else:
                    draw_params[p.name] = float(p.base_val * (1.0 + rng.normal(0, 0.05)))

            try:
                mc_res = identification(
                    model,
                    params=draw_params,
                    varobs=obs,
                    lags=lags,
                    n_freq=n_freq,
                    frequencies=omega,
                    tol=tol,
                    prior_mc=0,
                )
                j1_mc_ranks.append(mc_res.j1_rank)
                j2_mc_ranks.append(mc_res.j2_rank)
                jh_mc_ranks.append(mc_res.jh_rank)
                js_mc_ranks.append(mc_res.js_rank)
            except Exception:
                continue

        if j1_mc_ranks:
            prior_mc_results = {
                "n_draws": len(j1_mc_ranks),
                "j1_identified_rate": float(np.mean(np.array(j1_mc_ranks) == n_params)),
                "j2_identified_rate": float(np.mean(np.array(j2_mc_ranks) == n_params)),
                "jh_identified_rate": float(np.mean(np.array(jh_mc_ranks) == n_params)),
                "js_identified_rate": float(np.mean(np.array(js_mc_ranks) == n_params)),
                "mean_j1_rank": float(np.mean(j1_mc_ranks)),
                "mean_j2_rank": float(np.mean(j2_mc_ranks)),
                "mean_jh_rank": float(np.mean(jh_mc_ranks)),
                "mean_js_rank": float(np.mean(js_mc_ranks)),
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
        is_identified_solution=is_ident_sol,
        is_identified_moments=is_ident_mom,
        is_identified_transfer=is_ident_tra,
        is_identified_spectrum=is_ident_spe,
        j1_singular_values=j1_s,
        j1_condition_number=j1_cond,
        j2_singular_values=j2_s,
        j2_condition_number=j2_cond,
        jh_rank=jh_rank,
        jh_n_params=n_params,
        jh_singular_values=jh_s,
        jh_condition_number=jh_cond,
        jh_null_space=jh_null,
        jh_null_combinations=jh_combs,
        jh_collinearity=jh_coll,
        js_rank=js_rank,
        js_n_params=n_params,
        js_singular_values=js_s,
        js_condition_number=js_cond,
        js_null_space=js_null,
        js_null_combinations=js_combs,
        js_collinearity=js_coll,
        collinear_pairs=collinear_pairs,
        warnings=warnings_tuple,
        n_freq=len(omega),
        frequencies=omega,
    )
