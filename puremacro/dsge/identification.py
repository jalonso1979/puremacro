"""Iskrev (2010) and Ratto (2011) DSGE parameter identification analysis.

Evaluates:
- Solution Jacobian J1 = d vec(T, R, Q, Z, H) / d theta (reduced-form state-space).
- Theoretical moment Jacobian J2 = d m(theta) / d theta with moments
  m(theta) = [ys_O(theta); vec(Gamma_0(theta)); ...; vec(Gamma_p(theta))].
- Fast differentiation for shock standard errors and measurement errors
  skipping Klein QZ rational expectations re-solve.
- SVD rank deficiency detection, null space linear parameter combinations,
  multi-way collinearity R^2, and Ratto (2011) identification strength.

Every Jacobian is taken at **one** parameter vector theta_0 (Iskrev 2010:
the rank condition is a property of the Jacobian at a point). theta_0 is the
model's calibration overridden by ``fixed_params`` and then by every value the
caller supplies for the analysed parameters; the model is re-solved there once
and every column -- structural, shock and measurement-error -- is evaluated
around that solution. See :func:`identification` for how theta_0 is chosen
when no values are given.
"""
from __future__ import annotations

import difflib
import math
import warnings
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.linalg

from puremacro.dsge._estimated_params import EstimatedParams, EstimatedParamSpec
from puremacro.dsge._results import IdentificationResult
from puremacro.dsge.klein import BlanchardKahnError, KleinResidualError
from puremacro.dsge.observation import make_state_space_from_varobs

__all__ = ["identification"]

# What a failed solve at a parameter point raises: ModelError, SteadyStateError
# and the "no stationary moments" check are ValueErrors; SingularPencilError is
# a LinAlgError; the Klein solver raises BlanchardKahnError / KleinResidualError
# (RuntimeErrors); a parameter outside an equation's domain gives an
# ArithmeticError. Anything else (TypeError, NameError, ...) is a bug and is
# never swallowed.
_SOLVE_ERRORS: tuple[type[BaseException], ...] = (
    ValueError,
    ArithmeticError,
    np.linalg.LinAlgError,
    BlanchardKahnError,
    KleinResidualError,
)


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
    measurement_error: Mapping[str, float] | None = None,
) -> list[_ResolvedParam]:
    """Parse and resolve parameters into typed _ResolvedParam instances.

    ``base_val`` is the parameter's value at the evaluation point theta_0. A
    name given without a value takes its current value: the calibration for a
    structural parameter, the declared innovation standard deviation (square
    root of the diagonal of the model's shock covariance, 1.0 when the model
    declares none) for ``SE_<shock>``, the declared correlation (0.0 when none)
    for ``CORR_<s1>_<s2>``, ``measurement_error[obs]`` (0.0 when not given) for
    ``ME_<obs>``; a name that the model's ``estimated_params`` block declares
    takes that spec's ``start``.
    """
    declared_specs: dict[str, EstimatedParamSpec] = {}
    if getattr(model, "_estimated_params", None) is not None:
        declared_specs = {sp.name: sp for sp in model._estimated_params.specs}

    model_params = dict(getattr(model, "_params", {}) or {})
    shocks = list(model.shocks)
    model_cov = _model_shock_cov(model)

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
            i_sh = shocks.index(sh_name)
            base_v = user_val if user_val is not None else math.sqrt(
                max(float(model_cov[i_sh, i_sh]), 0.0)
            )
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
                if user_val is not None:
                    base_v = user_val
                else:
                    i1, i2 = shocks.index(s1), shocks.index(s2)
                    v1 = float(model_cov[i1, i1])
                    v2 = float(model_cov[i2, i2])
                    base_v = (
                        float(model_cov[i1, i2]) / math.sqrt(v1 * v2)
                        if v1 > 0.0 and v2 > 0.0 else 0.0
                    )
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
            base_v = user_val if user_val is not None else float(
                (measurement_error or {}).get(obs_name, 0.0)
            )
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


# ---------------------------------------------------------------------------
# The evaluation point theta_0
# ---------------------------------------------------------------------------

def _model_shock_cov(model: Any) -> np.ndarray:
    """The model's declared innovation covariance ``Sigma_u`` (identity if none)."""
    n_e = len(model.shocks)
    cov = getattr(model, "_shock_cov", None)
    if cov is None:
        return np.eye(n_e)
    cov = np.asarray(cov, dtype=float)
    return 0.5 * (cov + cov.T)


def _shock_cov_at(
    model_cov: np.ndarray,
    shocks: Sequence[str],
    resolved: Sequence[_ResolvedParam],
) -> np.ndarray:
    """``Sigma_u`` at the evaluation point.

    The same mapping ``LinearModel.estimate`` applies to a draw
    (``build._make_observation_eq``): start from the declared covariance, put
    every ``SE_`` value on the diagonal as a variance (``Q_ii = sigma_i**2``),
    then write every ``CORR_`` value as ``Q_ij = rho_ij * sigma_i * sigma_j``
    with the standard deviations of the evaluation point. Declared off-diagonal
    covariances that no ``CORR_`` parameter names are held fixed as covariances.
    """
    cov = np.array(model_cov, dtype=float, copy=True)
    for p in resolved:
        if p.kind == "stderr_shock":
            i = list(shocks).index(p.target[0])
            cov[i, i] = float(p.base_val) ** 2
    for p in resolved:
        if p.kind == "corr_shock":
            i, j = list(shocks).index(p.target[0]), list(shocks).index(p.target[1])
            c = float(p.base_val) * math.sqrt(max(cov[i, i], 0.0) * max(cov[j, j], 0.0))
            cov[i, j] = cov[j, i] = c
    return cov


@dataclass(frozen=True)
class _EvaluationPoint:
    """The single parameter vector theta_0 at which every Jacobian column is taken.

    ``params`` is the full structural parameter vector handed to the solver,
    ``shock_cov`` the innovation covariance ``Sigma_u(theta_0)`` and
    ``measurement_error`` the measurement-error **standard deviations** by
    observable (``H = diag(sd**2)``, the convention of
    :func:`~puremacro.dsge.observation.make_state_space_from_varobs`).
    """

    params: dict
    shock_cov: np.ndarray
    measurement_error: dict


def _evaluation_point(
    model: Any,
    resolved: Sequence[_ResolvedParam],
    obs: Sequence[str],
    fixed_params: Mapping[str, float] | None,
    measurement_error: Mapping[str, float] | None,
) -> _EvaluationPoint:
    """theta_0 = calibration, overridden by ``fixed_params``, then by ``resolved``."""
    params = dict(getattr(model, "_params", None) or {})
    if fixed_params:
        unknown = [k for k in fixed_params if k not in params]
        if unknown:
            raise ValueError(
                f"identification(): fixed_params names {sorted(unknown)}, which are "
                f"not model parameters, so there is nothing to hold fixed. Model "
                f"parameters: {sorted(params)}."
            )
        params.update({str(k): float(v) for k, v in fixed_params.items()})
    for p in resolved:
        if p.kind == "param":
            params[p.name] = float(p.base_val)

    shock_cov = _shock_cov_at(_model_shock_cov(model), list(model.shocks), resolved)

    me = {str(k): float(v) for k, v in (measurement_error or {}).items()}
    for p in resolved:
        if p.kind == "stderr_obs":
            if p.target[0] not in obs:
                raise ValueError(
                    f"identification(): {p.name} is a measurement error on "
                    f"{p.target[0]!r}, which is not among the observables {list(obs)}."
                )
            me[p.target[0]] = float(p.base_val)
    return _EvaluationPoint(params=params, shock_cov=shock_cov, measurement_error=me)


def _same_as_calibration(model: Any, params: Mapping[str, float]) -> bool:
    """True when ``params`` is exactly the vector the model was solved at."""
    cal = dict(getattr(model, "_params", None) or {})
    if cal.keys() != params.keys():
        return False
    for k, v in cal.items():
        try:
            if float(v) != float(params[k]):
                return False
        except (TypeError, ValueError):
            if v != params[k]:
                return False
    return True


def _linearize_of(model: Any) -> str:
    """The ``linearize=`` a ``build()`` model was solved with.

    ``build`` records it only through ``units``: ``"log"`` falls back to level
    deviations per variable whose steady state is not positive, so a model with
    any log-unit variable was built with ``"log"``, and one with none behaves as
    ``"level"`` at its own steady state.
    """
    lin = getattr(model, "_linearize", None)
    if lin in ("log", "level"):
        return lin
    units = getattr(model, "units", None) or {}
    return "log" if any(u == "log" for u in units.values()) else "level"


def _solve_at(
    model: Any,
    params: Mapping[str, float],
    guess: Mapping[str, float],
    states: Sequence[str] | None = None,
    *,
    strict: bool = True,
) -> Any:
    """Re-solve ``model`` (first order) at the structural parameter vector ``params``.

    Uses the same builder the model came from, with its differentiation
    method, QZ criterium and (for ``build``) linearisation, and by default
    ``strict=True`` so that a point with no unique stable solution raises
    instead of returning zero decision rules. ``states`` (default
    ``model.states``) is the predetermined set; for a ``build_dynare`` /
    ``load_mod`` model the caller passes the set from :func:`_state_set`, so
    that a lag whose coefficient is zero at the calibration is not dropped at
    a point where it is not.
    """
    qz = getattr(model, "_qz_criterium", None)
    extra = {} if qz is None else {"qz_criterium": qz}
    states = tuple(model.states) if states is None else tuple(states)
    if getattr(model, "_dynare_equations", None) is not None:
        from puremacro.dsge.dynare import build_dynare

        return build_dynare(
            model._dynare_equations,
            variables=model.variables,
            shocks=model.shocks,
            params=dict(params),
            guess=dict(guess),
            states=states,
            order=1,
            method=model.method,
            verify_derivatives=False,
            strict=strict,
            **extra,
        )
    if getattr(model, "_equations", None) is not None:
        from puremacro.dsge.build import build as _build

        return _build(
            model._equations,
            variables=model.variables,
            states=states,
            shocks=model.shocks,
            params=dict(params),
            guess=dict(guess),
            linearize=_linearize_of(model),
            method=model.method,
            verify_derivatives=False,
            strict=strict,
            **extra,
        )
    raise ValueError(
        "identification(): model carries neither _dynare_equations nor _equations; "
        "cannot re-solve structural parameters."
    )


def _step(theta: float) -> float:
    """Finite-difference step for a structural parameter: ``max(1e-5, 1e-4 |theta|)``."""
    return max(1e-5, 1e-4 * abs(float(theta)))


def _lag_jacobian(model: Any, params: Mapping[str, float], ss_arr: np.ndarray) -> np.ndarray:
    """``A_- = d f / d y_{t-1}`` of a ``build_dynare`` model at ``(ss_arr, params)``.

    The block ``build_dynare`` reads its states from (a variable is a state when
    its column is non-zero), evaluated with the model's own derivatives: the
    compiled symbolic Jacobian of a ``.mod``, else complex step / central
    differences of the callable.
    """
    eq = model._dynare_equations
    variables = list(model.variables)
    shocks = list(model.shocks)
    ss_arr = np.asarray(ss_arr, dtype=float)
    compiled = getattr(eq, "_compiled", None)
    if compiled is not None:
        _, _, a_minus, _ = compiled.eval_first_order(
            lead=ss_arr, curr=ss_arr, lag=ss_arr,
            shocks=np.zeros(len(shocks)), params=dict(params),
        )
    else:
        from puremacro.dsge.build import _Vec
        from puremacro.dsge.dynare import _lead_lag_jacobians

        par_vec = _Vec(list(params.keys()), [float(v) for v in params.values()], what="parameter")
        _, _, a_minus, _ = _lead_lag_jacobians(
            eq, variables, shocks, par_vec, ss_arr, model.method, verify=False
        )
    return np.asarray(a_minus, dtype=float)


def _state_set(
    model: Any,
    ss_arr: np.ndarray,
    params: Mapping[str, float],
    resolved: Sequence[_ResolvedParam],
) -> tuple[str, ...]:
    """Predetermined variables for every solve of one identification point.

    ``build_dynare`` detects states numerically, as the variables whose column
    of ``A_-`` is non-zero (``> 1e-10``) *at the parameter vector it is solved
    at*. ``model.states`` was detected at the calibration, so a lag whose
    coefficient is 0 there (SW07's ``crhoms``, ``crhopinf``, ``crhow``,
    ``cmap``, ``cmaw``) is not a state of ``model``; re-solving with that set
    at a point where the coefficient is not 0 silently drops the lag, and the
    derivative with respect to the coefficient itself is lost even at the
    calibration. The set returned is the union of

    - ``model.states``;
    - the variables whose ``A_-`` column is non-zero at theta_0 (``params`` at
      the steady state ``ss_arr``);
    - the variables whose ``A_-`` column moves when one analysed structural
      parameter ``theta_j`` moves by its finite-difference step (one parameter
      at a time, stepping away from an active upper bound):
      ``||A_-(theta_0 + h_j e_j)[:, v] - A_-(theta_0)[:, v]|| / h_j > 1e-10``.
      This is what a first derivative needs, and it catches a coefficient
      that is exactly 0 at theta_0.

    ``model.states`` is returned unchanged (same order) when nothing is added;
    otherwise the set is ordered as ``model.variables``, as ``build_dynare``
    orders auto-detected states. Adding a variable that does not enter lagged
    is harmless (a zero column in the transition, a zero eigenvalue), so the
    observables' moments and spectra are unchanged; only their derivatives
    gain the lag they were missing.
    """
    variables = list(model.variables)
    base = tuple(model.states)
    active = set(base)
    # A probe step outside an equation's domain gives NaN (skipped below), not
    # a warning the user should see.
    with np.errstate(all="ignore"):
        try:
            a0 = _lag_jacobian(model, params, ss_arr)
        except _SOLVE_ERRORS:
            return base
        col0 = np.linalg.norm(a0, axis=0)
        active.update(
            v for j, v in enumerate(variables) if np.isfinite(col0[j]) and col0[j] > 1e-10
        )
        for p in resolved:
            if p.kind != "param" or p.name not in params:
                continue
            theta = float(params[p.name])
            h = _step(theta)
            s = -1.0 if (p.ub is not None and math.isfinite(p.ub) and theta + h > p.ub) else 1.0
            moved = dict(params)
            moved[p.name] = theta + s * h
            try:
                a_j = _lag_jacobian(model, moved, ss_arr)
            except _SOLVE_ERRORS:
                continue
            d_col = np.linalg.norm(a_j - a0, axis=0) / h
            active.update(
                v for j, v in enumerate(variables) if np.isfinite(d_col[j]) and d_col[j] > 1e-10
            )
    if active == set(base):
        return base
    return tuple(v for v in variables if v in active)


def _representation(
    m: Any,
    obs: Sequence[str],
    shock_cov: np.ndarray,
    me: Mapping[str, float],
    lags: int,
    omega: np.ndarray,
) -> tuple[Any, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], list, list]:
    """State space of a solved model and its four stacked vectors (psi1, psi2, psiH, psiS)."""
    ssm = make_state_space_from_varobs(
        m, list(obs), shock_cov=shock_cov, measurement_error=dict(me) or None
    )
    psi1 = _compute_state_space_psi(ssm)
    psi2 = _compute_moments_psi(ssm.T, ssm.R, ssm.Q, ssm.Z, ssm.H, ssm.d, lags)
    psiH, psiS, bar_H_list, H_list = _compute_frequency_representations(
        ssm.T, ssm.R, ssm.Q, ssm.Z, ssm.H, omega
    )
    return ssm, (psi1, psi2, psiH, psiS), bar_H_list, H_list


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------

def _structural_derivative(
    f: Any,
    theta_0: float,
    h: float,
    lb: float | None,
    ub: float | None,
    f0: tuple[np.ndarray, ...],
    name: str = "theta",
) -> tuple[np.ndarray, ...]:
    """d f / d theta at ``theta_0`` for a structural parameter (model re-solved per point).

    Central difference ``(f(theta_0 + h) - f(theta_0 - h)) / (2h)`` when both
    points lie inside ``[lb, ub]`` and both solves succeed. Otherwise a one-sided
    difference in a direction ``s = +1/-1`` whose points stay inside the
    declared bounds -- away from the active bound when one is active; ``+1``
    first, then ``-1``, when neither is -- second order when the point
    ``theta_0 + 2sh`` is admissible,
    ``s * (-3 f(theta_0) + 4 f(theta_0 + sh) - f(theta_0 + 2sh)) / (2h)``, and first
    order ``s * (f(theta_0 + sh) - f(theta_0)) / h`` if not. A direction whose
    first step would cross a declared bound is never tried. ``f0 = f(theta_0)``
    is the representation at the evaluation point itself, so every difference
    is taken around the same theta_0. Only a failed solve (``_SOLVE_ERRORS``)
    triggers a fallback; any other exception propagates.
    """
    has_lb = lb is not None and math.isfinite(lb)
    has_ub = ub is not None and math.isfinite(ub)
    below = has_lb and theta_0 - h < lb
    above = has_ub and theta_0 + h > ub

    if not below and not above:
        try:
            fp = f(theta_0 + h)
            fm = f(theta_0 - h)
            return tuple((a - b) / (2.0 * h) for a, b in zip(fp, fm))
        except _SOLVE_ERRORS:
            pass

    if below and above:
        raise ValueError(
            f"identification(): the declared bounds [{lb}, {ub}] of {name} leave no "
            f"room for a finite-difference step of {h:.3g} on either side of "
            f"{name} = {theta_0!r}."
        )
    if above:
        directions: tuple[float, ...] = (-1.0,)
    elif below:
        directions = (1.0,)
    else:
        directions = (1.0, -1.0)

    last_exc: BaseException | None = None
    for s in directions:
        try:
            f1 = f(theta_0 + s * h)
        except _SOLVE_ERRORS as exc:
            last_exc = exc
            continue
        far = theta_0 + 2.0 * s * h
        far_ok = not ((s > 0 and has_ub and far > ub) or (s < 0 and has_lb and far < lb))
        if far_ok:
            try:
                f2 = f(far)
                return tuple(
                    s * (-3.0 * a + 4.0 * b - c) / (2.0 * h) for a, b, c in zip(f0, f1, f2)
                )
            except _SOLVE_ERRORS:
                pass
        return tuple(s * (b - a) / h for a, b in zip(f0, f1))

    sides = "either side" if len(directions) == 2 else (
        "the side away from the bound" + (f" {lb}" if below else f" {ub}")
    )
    raise ValueError(
        f"identification(): could not solve the model on {sides} of "
        f"{name} = {theta_0!r} (step {h:.3g}) to differentiate it: {last_exc}"
    ) from last_exc


def _shock_cov_derivative(
    p: _ResolvedParam,
    resolved: Sequence[_ResolvedParam],
    shocks: Sequence[str],
    cov: np.ndarray,
    h: float,
) -> np.ndarray:
    """d Sigma_u / d theta for an ``SE_`` or ``CORR_`` parameter, analytically.

    Under the mapping of :func:`_shock_cov_at`: for ``SE_i = sigma_i``,
    ``dQ_ii = 2 sigma_i`` and, for every ``CORR_`` parameter linking shock ``i``
    to shock ``k``, ``dQ_ik = dQ_ki = rho_ik * sigma_k``; for ``CORR_ij = rho``,
    ``dQ_ij = dQ_ji = sigma_i sigma_j``. At ``sigma_i = 0`` the diagonal entry
    uses ``2h`` (the one-sided derivative scale), as it always has, so the
    column does not vanish identically.
    """
    shocks = list(shocks)
    n_e = len(shocks)
    dQ = np.zeros((n_e, n_e))
    if p.kind == "stderr_shock":
        i = shocks.index(p.target[0])
        sigma_i = float(p.base_val)
        dQ[i, i] = 2.0 * sigma_i if abs(sigma_i) > 1e-12 else 2.0 * h
        sign_i = -1.0 if sigma_i < 0.0 else 1.0
        for q in resolved:
            if q.kind != "corr_shock":
                continue
            a, b = shocks.index(q.target[0]), shocks.index(q.target[1])
            if a == b or i not in (a, b):
                continue
            k = b if i == a else a
            d = float(q.base_val) * math.sqrt(max(cov[k, k], 0.0)) * sign_i
            dQ[a, b] += d
            dQ[b, a] += d
    elif p.kind == "corr_shock":
        a, b = shocks.index(p.target[0]), shocks.index(p.target[1])
        d = math.sqrt(max(cov[a, a], 0.0) * max(cov[b, b], 0.0))
        dQ[a, b] = d
        dQ[b, a] = d
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"not a shock-covariance parameter: {p.name}")
    return dQ


def _cholesky_factor_derivative(Q: np.ndarray, dQ: np.ndarray, h: float) -> np.ndarray:
    """Derivative of ``L = _cholesky_or_sqrt(Q)`` in the direction ``dQ``.

    - ``Q`` and ``dQ`` diagonal: ``L = diag(sqrt(Q_ii))`` and
      ``dL_ii = dQ_ii / (2 L_ii)`` (``1`` where ``L_ii = 0``: the derivative of
      ``|sigma|`` from the right, which is what a shock standard deviation is).
    - ``Q`` positive definite: the exact Cholesky derivative
      ``dL = L Phi(L^{-1} dQ L^{-T})``, ``Phi`` = lower triangle with the diagonal
      halved (differentiate ``L L' = Q`` and use that ``L^{-1} dL`` is lower
      triangular).
    - otherwise (singular ``Q``): a central difference of ``_cholesky_or_sqrt``
      along ``dQ``.
    """
    n = Q.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    L = _cholesky_or_sqrt(Q)
    off_dQ = dQ - np.diag(np.diag(dQ))
    if np.allclose(Q, np.diag(np.diag(Q))) and not np.any(off_dQ):
        d = np.sqrt(np.maximum(np.diag(Q), 0.0))
        out = np.zeros((n, n))
        for i in range(n):
            if dQ[i, i] != 0.0:
                out[i, i] = dQ[i, i] / (2.0 * d[i]) if d[i] > 0.0 else 1.0
        return out
    diag_L = np.diag(L)
    if np.allclose(L, np.tril(L)) and np.all(diag_L > 1e-12 * max(1.0, float(np.max(np.abs(diag_L))))):
        X = scipy.linalg.solve_triangular(L, dQ, lower=True)
        X = scipy.linalg.solve_triangular(L, X.T, lower=True).T
        Phi = np.tril(X)
        Phi[np.diag_indices(n)] *= 0.5
        return L @ Phi
    return (_cholesky_or_sqrt(Q + h * dQ) - _cholesky_or_sqrt(Q - h * dQ)) / (2.0 * h)


def _shock_cov_columns(
    dQ: np.ndarray,
    dL: np.ndarray,
    ssm: Any,
    bar_H_list: Sequence[np.ndarray],
    n_obs: int,
    lags: int,
    n_psi1: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """J1, J2, JH, JS columns of a parameter that moves only ``Q``, at the base ``ssm``.

    J1: the ``Q`` block of vec(T, R, Q, Z, H) is ``vec(dQ)``.
    J2: ``dSigma_alpha = T dSigma_alpha T' + R dQ R'``, ``dGamma_0 = Z dSigma_alpha Z'``,
    ``dGamma_k = Z T^k dSigma_alpha Z'``; the mean block does not move.
    JH: ``H(w) = [barH(w) L_Q, L_me]`` so ``dH(w) = [barH(w) dL, 0]``.
    JS: ``S(w) = (barH(w) Q barH(w)^* + H_me) / (2 pi)`` so
    ``dS(w) = barH(w) dQ barH(w)^* / (2 pi)``.
    """
    T, R, Z = ssm.T, ssm.R, ssm.Z
    n_e = dQ.shape[0]

    j1 = np.zeros(n_psi1)
    q_offset = T.size + R.size
    j1[q_offset: q_offset + n_e * n_e] = dQ.ravel()

    d_sigma = scipy.linalg.solve_discrete_lyapunov(T, R @ dQ @ R.T)
    d_sigma = 0.5 * (d_sigma + d_sigma.T)
    d_gamma_0 = Z @ d_sigma @ Z.T
    d_gamma_0 = 0.5 * (d_gamma_0 + d_gamma_0.T)
    parts = [np.zeros(n_obs), d_gamma_0.ravel()]
    T_k = T.copy()
    for _ in range(lags):
        parts.append((Z @ T_k @ d_sigma @ Z.T).ravel())
        T_k = T_k @ T
    j2 = np.concatenate(parts)

    dH_list = [np.hstack([bH @ dL, np.zeros((n_obs, n_obs))]) for bH in bar_H_list]
    dS_list = [(bH @ dQ @ bH.conj().T) / (2.0 * np.pi) for bH in bar_H_list]
    return j1, j2, _vectorize_H_derivative(dH_list), _vectorize_S_derivative(dS_list, n_obs)


def _identification_jacobians(
    model: Any,
    resolved: Sequence[_ResolvedParam],
    obs: Sequence[str],
    *,
    lags: int,
    omega: np.ndarray,
    fixed_params: Mapping[str, float] | None = None,
    measurement_error: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(J1, J2, JH, JS)``, every column taken at the same evaluation point theta_0.

    The model is re-solved once at theta_0 (only when theta_0's structural block
    differs from the calibration, or when the predetermined set of a
    ``build_dynare`` model must grow, see :func:`_state_set`); the base state
    space, the shock covariance, the measurement-error covariance and every
    perturbation are built from that solution, with one state set. Column
    ``j`` is ``d psi / d theta_j`` with every other parameter at its theta_0
    value.
    """
    obs = list(obs)
    shocks = list(model.shocks)
    n_e = len(shocks)
    n_obs = len(obs)
    point = _evaluation_point(model, resolved, obs, fixed_params, measurement_error)

    cal = dict(getattr(model, "_params", None) or {})
    moved = {k: v for k, v in point.params.items() if cal.get(k) != v}

    def _unsolvable(exc: BaseException) -> ValueError:
        return ValueError(
            f"identification(): the model cannot be solved at the requested "
            f"evaluation point (parameters that differ from the calibration: "
            f"{moved}): {exc}"
        )

    dynare_form = getattr(model, "_dynare_equations", None) is not None
    at_calibration = _same_as_calibration(model, point.params)
    states = tuple(model.states)
    if at_calibration:
        m0 = model
    else:
        try:
            # build_dynare: not strict yet -- the calibration's state set may be
            # the wrong one here, and this solve only has to supply the steady
            # state at theta_0 until _state_set has checked it.
            m0 = _solve_at(
                model, point.params, model.steady_state.to_dict(), states,
                strict=not dynare_form,
            )
        except _SOLVE_ERRORS as exc:
            raise _unsolvable(exc) from exc
    if dynare_form:
        states = _state_set(
            model,
            m0.steady_state.reindex(list(model.variables)).to_numpy(dtype=float),
            point.params,
            resolved,
        )
        needs_resolve = states != tuple(model.states) or (
            not at_calibration and tuple(m0.solution.eu) != (1, 1)
        )
        if needs_resolve:
            try:
                m0 = _solve_at(
                    model, point.params, m0.steady_state.to_dict(), states, strict=True
                )
            except _SOLVE_ERRORS as exc:
                raise _unsolvable(exc) from exc

    base_cov = point.shock_cov
    base_me = point.measurement_error
    base_ssm, psi_base, bar_H_base_list, H_base_list = _representation(
        m0, obs, base_cov, base_me, lags, omega
    )
    psi1_base, psi2_base, psiH_base, psiS_base = psi_base

    n_params = len(resolved)
    J1 = np.zeros((len(psi1_base), n_params))
    J2 = np.zeros((len(psi2_base), n_params))
    JH = np.zeros((len(psiH_base), n_params))
    JS = np.zeros((len(psiS_base), n_params))

    n_y_sq = n_obs ** 2
    tril_all = np.tril_indices(n_obs)
    guess0 = m0.steady_state.to_dict()

    for j, p in enumerate(resolved):
        theta_0 = float(p.base_val)
        h = _step(theta_0)

        if p.kind == "stderr_obs":
            # H_ii = sigma_me**2 and L_me = diag(|sigma_me|), analytically.
            obs_idx = obs.index(p.target[0])
            d_H = 2.0 * theta_0 if abs(theta_0) > 1e-12 else 2.0 * h

            h_offset = base_ssm.T.size + base_ssm.R.size + base_ssm.Q.size + base_ssm.Z.size
            diag_pos = obs_idx * n_obs + obs_idx
            J1[h_offset + diag_pos, j] = d_H

            g0_offset = n_obs
            J2[g0_offset + diag_pos, j] = d_H

            dH_list = []
            for m_idx in range(len(omega)):
                dH_m = np.zeros_like(H_base_list[m_idx])
                dH_m[obs_idx, n_e + obs_idx] = 1.0
                dH_list.append(dH_m)
            JH[:, j] = _vectorize_H_derivative(dH_list)

            diag_mask = (tril_all[0] == obs_idx) & (tril_all[1] == obs_idx)
            diag_idx = int(np.where(diag_mask)[0][0])
            for m_idx in range(len(omega)):
                JS[m_idx * n_y_sq + diag_idx, j] = d_H / (2.0 * np.pi)
            continue

        if p.kind in ("stderr_shock", "corr_shock"):
            # Q moves, the solved model does not: no re-solve.
            dQ = _shock_cov_derivative(p, resolved, shocks, base_ssm.Q, h)
            dL = _cholesky_factor_derivative(base_ssm.Q, dQ, h)
            J1[:, j], J2[:, j], JH[:, j], JS[:, j] = _shock_cov_columns(
                dQ, dL, base_ssm, bar_H_base_list, n_obs, lags, len(psi1_base)
            )
            continue

        # Structural parameter: re-solve at theta_0 +/- h, everything else at theta_0.
        def _psi_perturbed(val: float, _name: str = p.name) -> tuple[np.ndarray, ...]:
            theta = dict(point.params)
            theta[_name] = float(val)
            m_pert = _solve_at(model, theta, guess0, states)
            _, psi, _, _ = _representation(m_pert, obs, base_cov, base_me, lags, omega)
            return psi

        J1[:, j], J2[:, j], JH[:, j], JS[:, j] = _structural_derivative(
            _psi_perturbed, theta_0, h, p.lb, p.ub, psi_base, name=p.name
        )

    return J1, J2, JH, JS


# ---------------------------------------------------------------------------
# Prior Monte Carlo
# ---------------------------------------------------------------------------

def _raw_prior_draw(prior: Any, rng: np.random.Generator) -> float:
    """One draw from ``prior`` (a :class:`~puremacro.dsge.priors.Prior`), untruncated.

    Parameterisations follow :mod:`puremacro.dsge.priors` (Dynare's): beta and
    gamma by the mean and std of the (shifted) variable; inverse gamma type 1 by
    ``x**2 ~ InvGamma(nu/2, s/2)`` and type 2 by ``x ~ InvGamma(nu/2, s/2)``, i.e.
    ``1/x**2`` (type 1) or ``1/x`` (type 2) ``~ Gamma(shape=nu/2, scale=2/s)``;
    Weibull by ``(shape, scale)`` plus ``shift``.
    """
    from puremacro.dsge.priors import (
        _beta_ab,
        _invgamma2_s_nu,
        _invgamma_s_nu,
        _weibull_shape_scale,
        ensure_prior,
    )

    pr = ensure_prior(prior)
    dist = pr.dist
    if dist == "uniform":
        return float(rng.uniform(pr.lb, pr.ub))
    if dist == "normal":
        return float(rng.normal(pr.mean, pr.std))
    if dist == "beta":
        shift = float(pr.get("shift") or 0.0)
        scale = float(pr.get("scale") or 1.0)
        a, b = _beta_ab((pr.mean - shift) / scale, pr.std / scale)
        return float(shift + scale * rng.beta(a, b))
    if dist == "gamma":
        shift = float(pr.get("shift") or 0.0)
        k = ((pr.mean - shift) / pr.std) ** 2
        theta = pr.std ** 2 / (pr.mean - shift)
        return float(shift + rng.gamma(k, theta))
    if dist == "invgamma":
        kind = pr.get("kind") or "type1"
        s_val, nu_val = pr.get("s"), pr.get("nu")
        if s_val is None or nu_val is None:
            solve = _invgamma2_s_nu if kind == "type2" else _invgamma_s_nu
            s_val, nu_val = solve(float(pr.mean), float(pr.std))
        g = float(rng.gamma(float(nu_val) / 2.0, 2.0 / float(s_val)))
        return 1.0 / g if kind == "type2" else 1.0 / math.sqrt(g)
    if dist == "weibull":
        shift = float(pr.get("shift") or 0.0)
        shape, scale = pr.get("shape"), pr.get("scale")
        if shape is None or scale is None:
            shape, scale = _weibull_shape_scale(float(pr.mean) - shift, float(pr.std))
        return float(shift + float(scale) * rng.weibull(float(shape)))
    raise ValueError(f"identification(): cannot draw from a {dist!r} prior.")


def _draw_parameter(p: _ResolvedParam, rng: np.random.Generator, max_tries: int = 1000) -> float:
    """One prior-MC draw for ``p``.

    With a prior: a draw from it, truncated by rejection to the intersection of
    the prior's support and the parameter's declared ``[lb, ub]``. Without one:
    ``base_val * (1 + N(0, 0.05**2))``, a 5% jitter around the evaluation point.
    """
    if p.prior is None:
        return float(p.base_val * (1.0 + rng.normal(0.0, 0.05)))
    lo = max(float(getattr(p.prior, "lb", -math.inf)), p.lb if p.lb is not None else -math.inf)
    hi = min(float(getattr(p.prior, "ub", math.inf)), p.ub if p.ub is not None else math.inf)
    for _ in range(max_tries):
        x = _raw_prior_draw(p.prior, rng)
        if lo <= x <= hi:
            return x
    raise ValueError(
        f"identification(): no draw from the prior of {p.name} fell inside "
        f"[{lo}, {hi}] in {max_tries} tries."
    )


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
    fixed_params: Mapping[str, float] | None = None,
    measurement_error: Mapping[str, float] | None = None,
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

    **Evaluation point.** Local identification is a property of the Jacobian
    at one parameter vector theta_0 (Iskrev 2010; the rank condition
    ``rank(J(theta_0)) = n_theta``). Every column of every Jacobian is taken
    at the same theta_0, built as

    1. the model's calibration (``model._params``) and declared shock
       covariance (``shocks`` block / ``shock_cov=``; identity if none),
    2. overridden by ``fixed_params`` (structural values held fixed, not
       differentiated) and ``measurement_error``,
    3. overridden by the value of every analysed parameter (below).

    The model is re-solved once at theta_0 when its structural block differs
    from the calibration; column ``j`` then perturbs ``theta_j`` alone, with
    every other parameter at its theta_0 value. The value of an analysed
    parameter is:

    - ``Mapping`` / ``p_dict``: the value given.
    - ``Sequence[str]`` (names only): the current value -- the calibration for
      a structural parameter; ``sqrt(Sigma_u[i, i])`` for ``SE_<shock>`` (1.0
      when the model declares no covariance); the declared correlation for
      ``CORR_<s1>_<s2>`` (0.0 when none); ``measurement_error[<obs>]`` for
      ``ME_<obs>`` (0.0 when not given). A name the model's
      ``estimated_params`` block declares takes that spec's ``start``.
    - ``EstimatedParams`` or a sequence of ``EstimatedParamSpec``: each spec's
      ``start``, i.e. its ``INITVAL`` (``estimated_params`` or
      ``estimated_params_init``) when declared, else its prior mean, else the
      midpoint of its bounds.
    - ``None``: the model's ``estimated_params`` block if it has one (each
      spec at its ``start``, as above), else every calibrated parameter at its
      calibration. Dynare's ``identification`` defaults to
      ``parameter_set = prior_mean``; the two coincide unless the block
      declares an ``INITVAL``. Pass ``params={name: value}`` to choose the
      point explicitly (e.g. the prior means, or the calibration).

    Units and conventions: ``SE_<shock>`` is an innovation **standard
    deviation** (``Q_ii = sigma**2``); ``CORR_<s1>_<s2>`` a correlation
    (``Q_ij = rho * sigma_i * sigma_j``, with the standard deviations at
    theta_0); ``ME_<obs>`` a measurement-error standard deviation
    (``H_ii = sigma**2``). A declared off-diagonal covariance that no ``CORR_``
    parameter names stays fixed as a covariance. This is the mapping
    ``LinearModel.estimate`` applies to a posterior draw. Dynare's
    ``set_all_parameters`` differs for a *declared* correlation: it keeps the
    correlation (not the covariance) fixed when a standard deviation moves.

    **State set** (``build_dynare`` / ``load_mod`` models). ``build_dynare``
    detects the predetermined variables numerically at the calibration, so a
    lag whose coefficient is 0 there is not a state of the calibrated model.
    Every solve of one analysis uses the union of ``model.states``, the lags
    active at theta_0 and the lags any analysed structural parameter
    activates when it moves (the model is re-solved with it, even at the
    calibration, when it differs from ``model.states``). Moments and spectra
    of the observables do not depend on the extra states; their derivatives
    do. ``build()`` models keep their declared states.

    Derivatives: structural parameters by central differences with step
    ``h = max(1e-5, 1e-4 |theta_j|)``, re-solving at ``theta_0 +/- h e_j``.
    When ``theta_j +/- h`` crosses a declared bound (``lb``/``ub``) or a solve
    fails, a second-order one-sided difference
    ``s (-3 psi(theta_0) + 4 psi(theta_0 + s h) - psi(theta_0 + 2 s h)) / (2h)``
    stepping away from the active bound (first order if the ``2h`` point is not
    available); a step never crosses a declared bound. Parameters affecting
    shock standard deviations or correlations
    (Q) or measurement error variances (H) are differentiated analytically in
    both time and frequency domains without re-solving the rational
    expectations model.

    Parameters
    ----------
    model : LinearModel
        The solved linear DSGE model.
    params : Sequence[str] | Mapping[str, float] | EstimatedParams, optional
        Parameters to analyse and, for a Mapping or specs, their values at the
        evaluation point theta_0 (see above). Defaults to model._estimated_params
        or calibrated parameters.
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
        If > 0, repeats the rank analysis at ``prior_mc`` draws of theta_0: each
        analysed parameter with a prior is drawn from it (truncated to its
        declared bounds), one without a prior is jittered by 5% around its
        evaluation value, and each draw is analysed at that full point. Draws
        at which the model cannot be solved are skipped and counted in
        ``prior_mc_results["n_failed"]`` (``n_draws`` counts the analysed
        ones); when none can be solved the rates are NaN and a
        ``RuntimeWarning`` is emitted.
    seed : int, default 0
        Random seed for prior Monte Carlo simulation.
    fixed_params : Mapping[str, float], optional
        Structural parameter values held fixed at theta_0 but not analysed,
        applied on top of the calibration (as ``LinearModel.estimate``'s
        ``fixed_params``). Every name must be a model parameter.
    measurement_error : Mapping[str, float], optional
        Fixed measurement-error standard deviations by observable, as in
        ``LinearModel.estimate``. An ``ME_<obs>`` parameter given with a value
        overrides its entry; one given by name only is evaluated at it.

    Returns
    -------
    IdentificationResult
        Frozen dataclass with full singular value spectra, condition numbers,
        null space bases, collinear pairs, warnings, and publication tables.

    Raises
    ------
    ValueError
        Unknown parameter or observable names, or a theta_0 at which the model
        has no unique stable solution or no stationary moments.
    """
    if params is None and p_dict is not None:
        params = p_dict

    obs = _resolve_observables(model, varobs)
    resolved = _resolve_params(model, params, obs, measurement_error)
    param_names = tuple(p.name for p in resolved)
    n_params = len(resolved)

    omega = _build_frequency_grid(n_freq=n_freq, frequencies=frequencies, freq_type=freq_type)

    J1, J2, JH, JS = _identification_jacobians(
        model, resolved, obs, lags=lags, omega=omega,
        fixed_params=fixed_params, measurement_error=measurement_error,
    )

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

    # Optional Prior Monte Carlo: the whole analysis at each drawn theta_0.
    prior_mc_results = None
    if prior_mc > 0:
        rng = np.random.default_rng(seed)
        mc_ranks: dict[str, list[int]] = {"j1": [], "j2": [], "jh": [], "js": []}
        n_failed = 0

        for _ in range(prior_mc):
            # A prior that cannot be drawn inside its bounds is a specification
            # error, not an unsolvable draw: it propagates.
            drawn = [replace(p, base_val=_draw_parameter(p, rng)) for p in resolved]
            try:
                mc_J = _identification_jacobians(
                    model, drawn, obs, lags=lags, omega=omega,
                    fixed_params=fixed_params, measurement_error=measurement_error,
                )
                mc_r = [_analyze_jacobian(J, param_names, tol=tol)[0] for J in mc_J]
            except _SOLVE_ERRORS:  # a draw the model cannot solve is skipped
                n_failed += 1
                continue
            for key, r in zip(("j1", "j2", "jh", "js"), mc_r):
                mc_ranks[key].append(r)

        n_ok = len(mc_ranks["j1"])
        if n_ok == 0:
            warnings.warn(
                f"identification(): the model could not be solved at any of the "
                f"{prior_mc} prior draws, so prior_mc_results has no rates (NaN).",
                RuntimeWarning,
                stacklevel=2,
            )

        def _rate(key: str) -> float:
            return float(np.mean(np.array(mc_ranks[key]) == n_params)) if n_ok else float("nan")

        def _mean_rank(key: str) -> float:
            return float(np.mean(mc_ranks[key])) if n_ok else float("nan")

        prior_mc_results = {
            "n_draws": n_ok,
            "n_failed": n_failed,
            "j1_identified_rate": _rate("j1"),
            "j2_identified_rate": _rate("j2"),
            "jh_identified_rate": _rate("jh"),
            "js_identified_rate": _rate("js"),
            "mean_j1_rank": _mean_rank("j1"),
            "mean_j2_rank": _mean_rank("j2"),
            "mean_jh_rank": _mean_rank("jh"),
            "mean_js_rank": _mean_rank("js"),
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
