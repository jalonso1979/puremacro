"""Model diagnostics, eigenvalue spectrum analysis, and steady-state residuals for DSGE models."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.linalg

from puremacro.dsge._qz import SingularPencilError, _check_regular, ordqz_sorted
from puremacro.dsge._results import (
    DiagnosticFinding,
    EigenvalueTable,
    ModelDiagnosticsResult,
)
from puremacro.dsge.build import _Vec

if TYPE_CHECKING:
    from puremacro.dsge.build import LinearModel

__all__ = [
    "check",
    "resid",
    "model_diagnostics",
]


def check(
    model: LinearModel,
    *,
    qz_criterium: float = 1.0 + 1e-6,
) -> EigenvalueTable:
    """Evaluate Blanchard-Kahn determinacy and generalized eigenvalue spectrum.

    Computes the generalized eigenvalues of the companion pencil, classifies
    roots into stable, explosive, and unit-root categories, and verifies the
    Blanchard-Kahn order and rank conditions.

    When determinacy fails, isolates the offending generalized eigenvectors
    and extracts top variable loadings, explicitly identifying which economic
    variables drive the missing or excess explosive roots.

    Parameters
    ----------
    model : LinearModel
        The solved linear DSGE model.
    qz_criterium : float, default 1.0 + 1e-6
        Threshold modulus above which a generalized eigenvalue is classified
        as explosive. Moduli within 1e-5 of 1.0 are flagged as unit roots.

    Returns
    -------
    EigenvalueTable
        Frozen dataclass containing eigenvalue details, Blanchard-Kahn status,
        and offending variable loadings.
    """
    A = np.asarray(getattr(model, "A", np.zeros((0, 0))), dtype=float)
    B = np.asarray(getattr(model, "B", np.zeros((0, 0))), dtype=float)
    n = A.shape[0] if A.ndim == 2 else 0
    n_forward = max(0, n - len(getattr(model, "states", ())))

    if n == 0:
        return EigenvalueTable(
            eigenvalues=np.array([], dtype=complex),
            modulus=np.array([], dtype=float),
            real=np.array([], dtype=float),
            imag=np.array([], dtype=float),
            is_explosive=np.array([], dtype=bool),
            is_unit_root=np.array([], dtype=bool),
            n_explosive=0,
            n_forward=0,
            is_determinate=True,
            bk_status="Empty model: 0 equations, 0 variables.",
            loadings=None,
            variables=(),
            qz_criterium=qz_criterium,
        )

    # Generalised Schur decomposition: A = Q S Z', B = Q T Z'
    S, T, alpha, beta, Q, Z = ordqz_sorted(
        A, B, lambda a, b: abs(b) < qz_criterium * abs(a)
    )

    is_pencil_regular = True
    try:
        _check_regular(S, T, where="check")
    except SingularPencilError:
        is_pencil_regular = False

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_eigs = np.where(alpha != 0, beta / alpha, np.inf)

    # Sort eigenvalues by modulus ascending
    mods = np.abs(raw_eigs)
    sort_idx = np.argsort(mods)
    eigs_sorted = raw_eigs[sort_idx]
    mods_sorted = mods[sort_idx]

    real_part = np.real(eigs_sorted)
    imag_part = np.imag(eigs_sorted)
    is_explosive = mods_sorted >= qz_criterium
    is_unit_root = np.abs(mods_sorted - 1.0) <= 1e-5
    n_explosive = int(np.sum(is_explosive))

    is_determinate = bool(is_pencil_regular and (n_explosive == n_forward))

    loadings: dict[int, dict[str, float]] | None = None

    if is_determinate:
        bk_status = (
            f"Blanchard-Kahn condition satisfied: {n_explosive} explosive roots "
            f"for {n_forward} forward-looking variables (unique stable equilibrium)."
        )
    else:
        loadings = {}
        if not is_pencil_regular:
            bk_status = "Singular matrix pencil: det(A - lambda B) vanishes identically."
        elif n_explosive > n_forward:
            # Too many explosive roots (excess explosive / missing stable)
            k = n_explosive - n_forward
            exp_indices = np.where(is_explosive)[0]
            offending_indices = exp_indices[:k]
            for idx in offending_indices:
                mu = eigs_sorted[idx]
                loadings[int(idx)] = _compute_loadings(model, mu)

            first_idx = offending_indices[0]
            first_lds = loadings[int(first_idx)]
            top_vars = [v for v, w in first_lds.items() if w >= 0.10][:5]
            if not top_vars:
                top_vars = list(first_lds.keys())[:2]

            bk_status = (
                f"{n_explosive} explosive roots for {n_forward} forward-looking variables; "
                f"excess root at |lambda| = {mods_sorted[first_idx]:.4f} loads chiefly on "
                f"{', '.join(top_vars)} (no stable solution)."
            )
        else:
            # Too few explosive roots (missing explosive / excess stable)
            k = n_forward - n_explosive
            stable_indices = np.where(~is_explosive)[0]
            offending_indices = stable_indices[-k:]
            for idx in offending_indices:
                mu = eigs_sorted[idx]
                loadings[int(idx)] = _compute_loadings(model, mu)

            first_idx = offending_indices[0]
            first_lds = loadings[int(first_idx)]
            top_vars = [v for v, w in first_lds.items() if w >= 0.10][:5]
            if not top_vars:
                top_vars = list(first_lds.keys())[:2]

            bk_status = (
                f"{n_explosive} explosive roots for {n_forward} forward-looking variables; "
                f"missing root at |lambda| = {mods_sorted[first_idx]:.4f} loads chiefly on "
                f"{', '.join(top_vars)} (indeterminacy)."
            )

    return EigenvalueTable(
        eigenvalues=eigs_sorted,
        modulus=mods_sorted,
        real=real_part,
        imag=imag_part,
        is_explosive=is_explosive,
        is_unit_root=is_unit_root,
        n_explosive=n_explosive,
        n_forward=n_forward,
        is_determinate=is_determinate,
        bk_status=bk_status,
        loadings=loadings,
        variables=tuple(model.variables),
        qz_criterium=qz_criterium,
    )


def _compute_loadings(model: LinearModel, mu: complex) -> dict[str, float]:
    """Extract normalized variable loadings on a generalized eigenvector for mode mu."""
    if len(model.variables) == 0:
        return {}

    A_plus = getattr(model, "_A_plus", None)
    A_0 = getattr(model, "_A_0", None)
    A_minus = getattr(model, "_A_minus", None)

    is_inf = bool(np.isinf(mu) or not np.isfinite(mu) or (np.abs(mu) > 1e16))

    if A_plus is not None and A_0 is not None and A_minus is not None:
        # Polynomial pencil A_+ mu^2 + A_0 mu + A_- = 0
        # As mu -> inf, dividing by mu^2 yields A_+ as the asymptotic pencil matrix
        if is_inf:
            Q = np.asarray(A_plus, dtype=complex)
        else:
            Q = A_plus * (mu ** 2) + A_0 * mu + A_minus
        _, _, Vh = np.linalg.svd(Q)
        v_vars = Vh[-1].conj() if len(Vh) > 0 else np.zeros(len(model.variables), dtype=complex)
    else:
        # Klein pencil B - mu A = 0
        # As mu -> inf, dividing by mu yields -A (or A) as the asymptotic pencil matrix
        if is_inf:
            M = np.asarray(model.A, dtype=complex)
        else:
            M = np.asarray(model.B, dtype=complex) - mu * np.asarray(model.A, dtype=complex)
        _, _, Vh = np.linalg.svd(M)
        v = Vh[-1].conj() if len(Vh) > 0 else np.zeros(len(model.variables), dtype=complex)

        if getattr(model, "timing", "klein") == "dynare":
            n_s = len(model.states)
            v_vars = v[n_s:] if len(v) >= n_s + len(model.variables) else v[:len(model.variables)]
        else:
            order = [model.variables.index(var) for var in (*model.states, *model.controls)]
            v_vars = np.empty(len(model.variables), dtype=complex)
            for col_idx, var_idx in enumerate(order):
                if col_idx < len(v):
                    v_vars[var_idx] = v[col_idx]
                else:
                    v_vars[var_idx] = 0.0

    w = np.abs(v_vars)
    max_w = float(np.max(w)) if len(w) > 0 else 0.0
    norm_w = w / max_w if max_w > 0 else w
    lds = {var: float(norm_w[i]) for i, var in enumerate(model.variables)}

    return dict(sorted(lds.items(), key=lambda item: item[1], reverse=True))


def resid(model: LinearModel) -> pd.Series:
    """Compute deterministic steady-state equation residuals.

    Evaluates the model's dynamic equilibrium equations at steady state
    f(ss, ss, ss, 0) and returns residuals sorted by absolute magnitude.

    Parameters
    ----------
    model : LinearModel
        The solved linear DSGE model.

    Returns
    -------
    pandas.Series
        Equation residuals indexed by equation tags (if available) or
        labels ('eq_1', 'eq_2', ...), sorted descending by |resid|.
    """
    variables = list(getattr(model, "variables", ()))
    if len(variables) == 0:
        return pd.Series([], index=[], dtype=float, name="residual")

    dynare_eqs = getattr(model, "_dynare_equations", None)
    equations = getattr(model, "_equations", None)
    shocks = list(getattr(model, "shocks", ()))
    ss_arr = np.asarray(model.steady_state.loc[variables], dtype=float)
    par_dict = dict(getattr(model, "_params", None) or {})

    if dynare_eqs is not None:
        y_v = _Vec(variables, ss_arr, what="variable")
        e_0 = _Vec(shocks, np.zeros(len(shocks)), what="shock")
        par_v = _Vec(list(par_dict.keys()), list(par_dict.values()), what="parameter")
        raw_res = np.asarray(dynare_eqs(y_v, y_v, y_v, e_0, par_v), dtype=float)
    elif equations is not None:
        ss_v = _Vec(variables, ss_arr, what="variable")
        e_0 = _Vec(shocks, np.zeros(len(shocks)), what="shock")
        par_v = _Vec(tuple(par_dict.keys()), list(par_dict.values()), "parameter")
        raw_res = np.asarray(equations(ss_v, ss_v, e_0, par_v), dtype=float)
    else:
        # Fallback to linear steady-state residual
        A_plus = getattr(model, "_A_plus", None)
        A_0 = getattr(model, "_A_0", None)
        A_minus = getattr(model, "_A_minus", None)
        if A_plus is not None and A_0 is not None and A_minus is not None:
            raw_res = (A_plus + A_0 + A_minus) @ ss_arr
        else:
            raw_res = np.zeros(len(variables), dtype=float)

    eq_tags = getattr(model, "_equation_tags", None)
    if eq_tags is not None and len(eq_tags) == len(raw_res):
        index = list(eq_tags)
    else:
        index = [f"eq_{i+1}" for i in range(len(raw_res))]

    series = pd.Series(raw_res, index=index, name="residual")
    if len(series) == 0:
        return series
    nan_mask = np.isnan(series.values)
    abs_vals = np.nan_to_num(np.abs(series.values), nan=-np.inf)
    sort_order = np.lexsort((-abs_vals, ~nan_mask))
    return series.iloc[sort_order]


def model_diagnostics(
    model: LinearModel,
    *,
    tol: float = 1e-8,
    n_eval_points: int = 5,
) -> ModelDiagnosticsResult:
    """Perform comprehensive structural and numerical diagnostics on the DSGE model.

    Executes static Jacobian rank analysis, computes union numeric incidence
    across random neighbourhood perturbations, verifies dynamic pencil regularity,
    identifies unit roots, and detects stochastic singularity.

    Parameters
    ----------
    model : LinearModel
        The solved DSGE model.
    tol : float, default 1e-8
        Numerical tolerance for singularity, rank deficiency, and residuals.
    n_eval_points : int, default 5
        Number of evaluation points used to construct the numeric incidence
        matrix (steady state + n_eval_points - 1 random perturbations).

    Returns
    -------
    ModelDiagnosticsResult
        Frozen dataclass containing diagnostic findings and inspection methods.
    """
    findings: list[DiagnosticFinding] = []
    variables = tuple(getattr(model, "variables", ()))
    n_vars = len(variables)

    if n_vars == 0:
        return ModelDiagnosticsResult(
            passed=True,
            findings=(),
            static_rank=0,
            static_n_vars=0,
            collinear_equations=(),
            collinear_variables=(),
            unused_variables=(),
            unused_equations=(),
            is_pencil_regular=True,
            stochastic_singularity=False,
            unit_roots=(),
            incidence_matrix=np.zeros((0, 0), dtype=bool),
            eval_points=n_eval_points,
            variable_names=(),
            equation_names=(),
        )

    ss_arr = np.asarray(model.steady_state.loc[list(variables)], dtype=float)

    # 1. Steady-state residual check
    res_series = resid(model)
    if len(res_series) > 0:
        nan_mask = np.isnan(res_series.values)
        if np.any(nan_mask):
            nan_equations = tuple(res_series.index[nan_mask])
            worst_eq = nan_equations[0]
            findings.append(
                DiagnosticFinding(
                    category="steady_state",
                    severity="error",
                    message=f"The steady state could not be evaluated (residual is NaN in {worst_eq}).",
                    details={
                        "max_residual": float("nan"),
                        "worst_equation": worst_eq,
                        "nan_equations": nan_equations,
                    },
                )
            )
        else:
            max_resid = float(np.max(np.abs(res_series.values)))
            if max_resid > tol:
                worst_eq = res_series.index[0]
                findings.append(
                    DiagnosticFinding(
                        category="steady_state",
                        severity="error",
                        message=(
                            f"The steady state does not solve the dynamic equations: "
                            f"max |f(ss, ss, 0)| = {max_resid:.3e} > tol = {tol:.3e} "
                            f"(worst equation: {worst_eq})."
                        ),
                        details={"max_residual": max_resid, "worst_equation": worst_eq},
                    )
                )

    # 2. Static Jacobian Rank Analysis
    A_plus = getattr(model, "_A_plus", None)
    A_0 = getattr(model, "_A_0", None)
    A_minus = getattr(model, "_A_minus", None)
    dynare_eqs = getattr(model, "_dynare_equations", None)
    equations = getattr(model, "_equations", None)
    par_dict = dict(getattr(model, "_params", None) or {})

    if A_plus is not None and A_0 is not None and A_minus is not None:
        J_static = A_plus + A_0 + A_minus
    elif dynare_eqs is not None or equations is not None:
        # Evaluate static Jacobian via central differences
        h = 1e-6
        shocks = list(model.shocks)
        e_0 = _Vec(shocks, np.zeros(len(shocks)), what="shock")
        par_v = _Vec(list(par_dict.keys()), list(par_dict.values()), what="parameter")

        def _eval_static(y):
            y_v = _Vec(variables, y, what="variable")
            if dynare_eqs is not None:
                return np.asarray(dynare_eqs(y_v, y_v, y_v, e_0, par_v), dtype=float)
            return np.asarray(equations(y_v, y_v, e_0, par_v), dtype=float)

        f0 = _eval_static(ss_arr)
        n_eq = len(f0)
        J_static = np.zeros((n_eq, n_vars))
        for j in range(n_vars):
            yp = ss_arr.copy()
            ym = ss_arr.copy()
            yp[j] += h
            ym[j] -= h
            J_static[:, j] = (_eval_static(yp) - _eval_static(ym)) / (2 * h)
    else:
        J_static = np.asarray(model.A - model.B, dtype=float)

    n_eq = J_static.shape[0]
    eq_tags = getattr(model, "_equation_tags", None)
    if eq_tags is not None and len(eq_tags) == n_eq:
        equation_names = tuple(eq_tags)
    else:
        equation_names = tuple(f"eq_{i+1}" for i in range(n_eq))

    collinear_equations: list[str] = []
    collinear_variables: list[str] = []

    if not np.all(np.isfinite(J_static)):
        static_rank = 0
        findings.append(
            DiagnosticFinding(
                category="static_rank",
                severity="error",
                message="Static Jacobian contains non-finite values (NaN/Inf); rank cannot be evaluated.",
                details={"n_vars": n_vars, "n_eq": n_eq},
            )
        )
    else:
        U, s, Vh = np.linalg.svd(J_static)
        s_max = s[0] if len(s) > 0 and s[0] > 0 else 1.0
        rank_tol = max(tol, tol * s_max)
        static_rank = int(np.sum(s > rank_tol))

        if static_rank < n_vars:
            findings.append(
                DiagnosticFinding(
                    category="static_rank",
                    severity="error",
                    message=(
                        f"Static Jacobian is rank-deficient: rank {static_rank} for "
                        f"{n_vars} variables (deficiency = {n_vars - static_rank})."
                    ),
                    details={"rank": static_rank, "n_vars": n_vars, "singular_values": s},
                )
            )
        elif static_rank < n_eq:
            findings.append(
                DiagnosticFinding(
                    category="static_rank",
                    severity="warning",
                    message=(
                        f"Static Jacobian has linearly dependent equations: rank {static_rank} for "
                        f"{n_eq} equations (redundancy = {n_eq - static_rank})."
                    ),
                    details={"rank": static_rank, "n_eq": n_eq, "singular_values": s},
                )
            )

        # 1. Collinear equations from left singular vectors (columns of U)
        for k in range(static_rank, n_eq):
            u_k = U[:, k]
            max_u = float(np.max(np.abs(u_k)))
            w_eq = np.abs(u_k) / max_u if max_u > 0 else np.abs(u_k)
            eq_subset = [
                f"{equation_names[i]} ({u_k[i]:+.3f})"
                for i in range(n_eq)
                if w_eq[i] >= 0.10
            ]
            collinear_equations.append(
                f"Combination #{k - static_rank + 1}: {', '.join(eq_subset)}"
            )

        # 2. Collinear / unconstrained variables from right singular vectors (rows of Vh)
        for k in range(static_rank, n_vars):
            v_k = Vh[k, :]
            max_v = float(np.max(np.abs(v_k)))
            w_var = np.abs(v_k) / max_v if max_v > 0 else np.abs(v_k)
            var_subset = [
                f"{variables[j]} ({v_k[j]:+.3f})"
                for j in range(n_vars)
                if w_var[j] >= 0.10
            ]
            collinear_variables.append(
                f"Combination #{k - static_rank + 1}: {', '.join(var_subset)}"
            )

    # 3. 5-point random perturbation numeric incidence matrix
    if dynare_eqs is not None or equations is not None:
        shocks = list(model.shocks)
        e_0 = _Vec(shocks, np.zeros(len(shocks)), what="shock")
        par_v = _Vec(list(par_dict.keys()), list(par_dict.values()), what="parameter")

        def _eval_f(y):
            y_v = _Vec(variables, y, what="variable")
            if dynare_eqs is not None:
                return np.asarray(dynare_eqs(y_v, y_v, y_v, e_0, par_v), dtype=float)
            return np.asarray(equations(y_v, y_v, e_0, par_v), dtype=float)

        points = [ss_arr]
        rng = np.random.default_rng(seed=42)
        scale = np.maximum(1.0, np.abs(ss_arr))
        for _ in range(max(1, n_eval_points - 1)):
            delta = rng.uniform(-0.05, 0.05, size=n_vars) * scale
            points.append(ss_arr + delta)

        incidence_union = np.zeros((n_eq, n_vars), dtype=bool)
        h = 1e-6
        for pt in points:
            for j in range(n_vars):
                yp = pt.copy()
                ym = pt.copy()
                yp[j] += h
                ym[j] -= h
                diff = np.abs(_eval_f(yp) - _eval_f(ym)) / (2 * h)
                incidence_union[:, j] |= (diff > 1e-8)
    else:
        # Fallback to non-zero pattern of static Jacobian and Klein matrices
        incidence_union = (np.abs(J_static) > 1e-8)
        for mat in (A_plus, A_0, A_minus):
            if mat is not None:
                m_arr = np.asarray(mat)
                if m_arr.ndim == 2:
                    r = min(n_eq, m_arr.shape[0])
                    c = min(n_vars, m_arr.shape[1])
                    if r > 0 and c > 0:
                        incidence_union[:r, :c] |= (np.abs(m_arr[:r, :c]) > 1e-8)
        if hasattr(model, "A") and model.A is not None:
            A_mat = np.asarray(model.A)
            if A_mat.ndim == 2:
                r_a = min(n_eq, A_mat.shape[0])
                c_a = min(n_vars, A_mat.shape[1])
                if r_a > 0 and c_a > 0:
                    incidence_union[:r_a, :c_a] |= (np.abs(A_mat[:r_a, :c_a]) > 1e-8)
        if hasattr(model, "B") and model.B is not None:
            B_mat = np.asarray(model.B)
            if B_mat.ndim == 2:
                r_b = min(n_eq, B_mat.shape[0])
                c_b = min(n_vars, B_mat.shape[1])
                if r_b > 0 and c_b > 0:
                    incidence_union[:r_b, :c_b] |= (np.abs(B_mat[:r_b, :c_b]) > 1e-8)

    unused_variables = tuple(
        variables[j] for j in range(n_vars) if not np.any(incidence_union[:, j])
    )
    if unused_variables:
        findings.append(
            DiagnosticFinding(
                category="incidence",
                severity="error",
                message=f"Unused variables detected: {', '.join(unused_variables)} enter zero equations.",
                details={"unused_variables": unused_variables},
            )
        )

    unused_equations = tuple(
        equation_names[i] for i in range(n_eq) if not np.any(incidence_union[i, :])
    )
    if unused_equations:
        findings.append(
            DiagnosticFinding(
                category="incidence",
                severity="error",
                message=f"Redundant/empty equations detected: {', '.join(unused_equations)} involve zero variables.",
                details={"unused_equations": unused_equations},
            )
        )

    # 4. Dynamic pencil regularity
    is_pencil_regular = True
    A_mat = getattr(model, "A", None)
    B_mat = getattr(model, "B", None)
    if (
        n_vars > 0
        and A_mat is not None
        and B_mat is not None
        and np.asarray(A_mat).ndim == 2
        and np.asarray(B_mat).ndim == 2
        and np.asarray(A_mat).shape == np.asarray(B_mat).shape
        and np.asarray(A_mat).shape[0] == np.asarray(A_mat).shape[1]
        and np.asarray(A_mat).shape[0] > 0
    ):
        try:
            S, T, alpha, beta, Q, Z = ordqz_sorted(
                A_mat, B_mat, lambda a, b: abs(b) < (1.0 + 1e-6) * abs(a)
            )
            _check_regular(S, T, where="model_diagnostics", tol=tol)
        except SingularPencilError as exc:
            is_pencil_regular = False
            findings.append(
                DiagnosticFinding(
                    category="pencil",
                    severity="error",
                    message=(
                        f"Dynamic matrix pencil (A, B) is singular: {exc.n_singular} of "
                        f"{exc.n_total} eigenvalues vanish simultaneously in both Schur "
                        "factors (det(A - lambda B) == 0)."
                    ),
                    details={"n_singular": exc.n_singular, "n_total": exc.n_total},
                )
            )
        except Exception as exc:
            is_pencil_regular = False
            findings.append(
                DiagnosticFinding(
                    category="pencil",
                    severity="error",
                    message=f"Dynamic matrix pencil regularity check failed: {exc}",
                    details=str(exc),
                )
            )

    # 5. Stochastic singularity
    stochastic_singularity = False
    varobs = getattr(model, "_varobs", None)
    if varobs is not None:
        n_obs = len(varobs)
        n_shocks = len(model.shocks)
        n_me = 0
        est_params = getattr(model, "_estimated_params", None)
        if est_params is not None:
            for p_spec in est_params:
                if getattr(p_spec, "kind", "") == "stderr_obs":
                    n_me += 1
        if n_obs > n_shocks + n_me:
            stochastic_singularity = True
            findings.append(
                DiagnosticFinding(
                    category="stochastic_singularity",
                    severity="warning",
                    message=(
                        f"Stochastic singularity: {n_obs} observable variables declared "
                        f"({', '.join(varobs)}) but only {n_shocks} structural shocks "
                        f"and {n_me} measurement errors. Theoretical observable covariance "
                        "is singular."
                    ),
                    details={"n_obs": n_obs, "n_shocks": n_shocks, "n_me": n_me},
                )
            )

    # 6. Unit roots
    unit_roots: list[int] = []
    sol_eigs = getattr(model.solution, "eigenvalues", None)
    if sol_eigs is not None and len(sol_eigs) > 0:
        for idx, eig in enumerate(sol_eigs):
            if np.isfinite(eig) and abs(abs(eig) - 1.0) <= 1e-5:
                unit_roots.append(idx)
        if unit_roots:
            findings.append(
                DiagnosticFinding(
                    category="unit_root",
                    severity="info",
                    message=(
                        f"{len(unit_roots)} unit root(s) detected (|lambda| ~= 1.0). "
                        "Unconditional stationary moments do not exist without "
                        "differencing or diffuse filter."
                    ),
                    details={"unit_roots": unit_roots},
                )
            )

    passed = not any(f.severity == "error" for f in findings)

    return ModelDiagnosticsResult(
        passed=passed,
        findings=tuple(findings),
        static_rank=static_rank,
        static_n_vars=n_vars,
        collinear_equations=tuple(collinear_equations),
        collinear_variables=tuple(collinear_variables),
        unused_variables=unused_variables,
        unused_equations=unused_equations,
        is_pencil_regular=is_pencil_regular,
        stochastic_singularity=stochastic_singularity,
        unit_roots=tuple(unit_roots),
        incidence_matrix=incidence_union,
        eval_points=n_eval_points,
        variable_names=variables,
        equation_names=equation_names,
    )
