"""Dynare-compatible DSGE modelling interface and pure-Python .mod parser.

Bridges the gap between Dynare's canonical dynamic representation:
    E_t [ f(y_{t+1}, y_t, y_{t-1}, u_t; θ) ] = 0

and puremacro's rational expectations solvers:
1. **Natural Lead-Lag Syntax**: write equilibrium equations as
   ``eqs(lead, curr, lag, shocks, params)`` using standard economic conventions
   (e.g., ``curr.c`` for c_t, ``lead.c`` for c_{t+1}, ``lag.k`` for k_{t-1}).
2. **Automatic Variable Classification**: automatically identifies predetermined
   state variables (variables appearing with lags in f) and forward-looking jump
   variables, eliminating the manual ``states=[...]`` requirement.
3. **Pure-Python .mod File Parser**: reads standard Dynare ``.mod`` files directly
   into solved ``LinearModel`` objects with decision rules (`oo_.dr`), theoretical
   moments, and variance decomposition on any device with zero MATLAB/Octave.

References
----------
Adjemian, S. et al. (2011). Dynare: Reference Manual, Version 4. Dynare Working Papers 1.
Schmitt-Grohé, S. and Uribe, M. (2004). Solving dynamic general equilibrium
    models using a second-order approximation to the policy function. JEDC 28, 755-775.
"""
from __future__ import annotations

import dataclasses
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.optimize

from puremacro.dsge.build import (
    LinearModel,
    ModelError,
    SteadyStateError,
    _Vec,
    _CSTEP,
    _FDSTEP,
    _jacobian,
    _verify_jacobian,
)
from puremacro.dsge._estimated_params import (
    parse_estimated_params,
    parse_estimated_params_bounds,
    parse_estimated_params_init,
)
from puremacro.dsge.klein import KleinSolution, klein_solve
from puremacro.dsge.pruning import PrunedDSGESolution, Order3PrunedSolution
from puremacro.dsge._macro import preprocess_macro, DynareMacroError
from puremacro.dsge._parser import parse_mod_to_dag, DynareParseError, ParsedModelDAG
from puremacro.dsge._symbolic import compile_derivatives, CompiledDerivatives
from puremacro.dsge._sylvester import (
    solve_generalized_sylvester_kronecker,
    solve_order3_sylvester_kronecker,
    solve_order1_sylvester,
)
from puremacro.dsge._utils import model_info, write_latex_dynamic_model, detect_linear_model
from puremacro.dsge.smc import (
    SMCSampler,
    SMCResult,
    bootstrap_particle_filter,
    smc_estimate,
)
from puremacro.dsge.ramsey import (
    ramsey_model,
    RamseyResult,
    detrend_bgp,
    detrend_model,
)


def _remove_comments(text: str) -> str:
    """Strip C-style (//, /* */) and MATLAB-style (%) comments."""
    # Block comments /* ... */
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Line comments // and %
    text = re.sub(r"(//|%).*$", "", text, flags=re.MULTILINE)
    return text


class DynareFeatureError(NotImplementedError):
    """A .mod construct puremacro recognises but does not implement yet.

    Deliberately **not** a ``ValueError``: callers wrap parse failures in
    ``except ValueError`` and this must not be swallowed by them. Every other
    unsupported construct fails loudly on its own (a model-local ``#``
    variable over an endogenous variable, ``STEADY_STATE()`` and ``normcdf``
    all raise ``NameError`` when the compiled equations run); the macro
    processor did not, and silently solving a different model from the one the
    file describes is the failure this class exists to prevent.
    """


# Dynare macro-processor syntax. Directives are line-oriented (``@#define``,
# ``@#for``, ``@#if``, ``@#include``, ...); ``@{expr}`` is inline interpolation.
# Both are matched on comment-stripped text, so ``% @#define N = 2`` stays a
# comment.
_MACRO_DIRECTIVE = re.compile(r"^[^\S\n]*@#\s*\w+", re.M)
_MACRO_INTERPOLATION = re.compile(r"@\{")


def _check_no_macro_directives(clean_text: str) -> None:
    """Refuse a .mod file that has remaining unrecognised macro directives."""
    found = _MACRO_DIRECTIVE.search(clean_text) or _MACRO_INTERPOLATION.search(clean_text)
    if found is None:
        return
    raise DynareFeatureError(
        f"this .mod file uses an unsupported Dynare macro directive ({found.group(0).strip()!r} "
        f"at character {found.start()})."
    )


def _lead_lag_jacobians(
    equations: Callable,
    variables: Sequence[str],
    shocks: Sequence[str],
    par_vec: _Vec,
    ss_arr: np.ndarray,
    method: str,
    *,
    verify: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Jacobians of ``f(lead, curr, lag, shocks)`` at the steady state.

    Returns ``(A_plus, A_0, A_minus, B_u)`` = ``(df/d lead, df/d curr,
    df/d lag, df/d shocks)``. With ``verify=True`` (complex step only) each
    block is cross-checked against a central finite difference in a fixed
    random direction and :class:`ModelError` is raised when they disagree —
    the only way to catch a residual function that is not analytic, since
    complex-step returns a *wrong* derivative through ``abs()``, ``max()``
    or a ``float()`` cast without any error.
    """
    variables = list(variables)
    shocks = list(shocks)
    n_vars, n_shocks = len(variables), len(shocks)
    ss_arr = np.asarray(ss_arr, dtype=float)
    e_zero = np.zeros(n_shocks)

    def f_lead(v):
        d = np.asarray(v).dtype
        return equations(_Vec(variables, v), _Vec(variables, ss_arr.astype(d)),
                         _Vec(variables, ss_arr.astype(d)), _Vec(shocks, e_zero.astype(d)), par_vec)

    def f_curr(v):
        d = np.asarray(v).dtype
        return equations(_Vec(variables, ss_arr.astype(d)), _Vec(variables, v),
                         _Vec(variables, ss_arr.astype(d)), _Vec(shocks, e_zero.astype(d)), par_vec)

    def f_lag(v):
        d = np.asarray(v).dtype
        return equations(_Vec(variables, ss_arr.astype(d)), _Vec(variables, ss_arr.astype(d)),
                         _Vec(variables, v), _Vec(shocks, e_zero.astype(d)), par_vec)

    def f_shock(e):
        d = np.asarray(e).dtype
        return equations(_Vec(variables, ss_arr.astype(d)), _Vec(variables, ss_arr.astype(d)),
                         _Vec(variables, ss_arr.astype(d)), _Vec(shocks, e), par_vec)

    A_plus = _jacobian(f_lead, ss_arr, n_vars, method)
    A_0 = _jacobian(f_curr, ss_arr, n_vars, method)
    A_minus = _jacobian(f_lag, ss_arr, n_vars, method)
    B_u = _jacobian(f_shock, e_zero, n_vars, method) if n_shocks else np.zeros((n_vars, 0))

    if verify and method == "complex":
        for label, f, x0, J in (
            ("lead (t+1)", f_lead, ss_arr, A_plus),
            ("current (t)", f_curr, ss_arr, A_0),
            ("lag (t-1)", f_lag, ss_arr, A_minus),
            ("shock", f_shock, e_zero, B_u),
        ):
            _verify_jacobian(label, f, x0, J)

    return A_plus, A_0, A_minus, B_u


def _solve_lead_lag_system(
    A_plus: np.ndarray,
    A_0: np.ndarray,
    A_minus: np.ndarray,
    B_u: np.ndarray,
    state_idx: Sequence[int],
    *,
    strict: bool,
    qz_criterium: float = 1.0 + 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, KleinSolution, np.ndarray, np.ndarray]:
    r"""Solve the lead-lag system via Klein's generalized Schur solver.

    Transforms the first-order conditions ``A_+ E_t y_{t+1} + A_0 y_t +
    A_- s_{t-1} + B_u u_t = 0`` into the standard Klein state-space system
    ``A E_t z_{t+1} = B z_t + C u_t`` by stacking ``z_t = [s_{t-1}; y_t]``,
    so the system is the ``n`` model equations plus ``n_s`` identity rows
    ``s_t = P_s y_t``; nothing is dropped, and a variable may appear at
    ``t-1``, ``t`` and ``t+1`` at once (TFP in every textbook Euler
    equation, C/I/Pi in medium-scale models). The unique stable solution
    ``y_t = F_full s_{t-1} + L_full u_t`` is Dynare's ``ghx`` / ``ghu``
    directly, and ``s_t = G s_{t-1} + N u_t`` with ``G = P_s F_full``,
    ``N = P_s L_full``.

    Returns ``(A_klein, B_klein, C_klein, solution, F_full, L_full)``.
    """
    n = A_0.shape[0]
    n_s = len(state_idx)
    n_u = B_u.shape[1]
    P_s = np.zeros((n_s, n))
    for j, idx in enumerate(state_idx):
        P_s[j, idx] = 1.0

    A_klein = np.zeros((n + n_s, n + n_s))
    B_klein = np.zeros((n + n_s, n + n_s))
    C_klein = np.zeros((n + n_s, n_u))
    A_klein[:n, n_s:] = A_plus
    B_klein[:n, :n_s] = -A_minus[:, list(state_idx)]
    B_klein[:n, n_s:] = -A_0
    C_klein[:n, :] = -B_u
    A_klein[n:, :n_s] = np.eye(n_s)
    B_klein[n:, n_s:] = P_s

    sol_full = klein_solve(A_klein, B_klein, n_pre=n_s, C=C_klein, strict=strict, div=qz_criterium)
    F_full = np.asarray(sol_full.F, dtype=float)
    L_full = np.asarray(sol_full.L, dtype=float)
    return A_klein, B_klein, C_klein, sol_full, F_full, L_full


def _check_lead_lag_residual(
    A_plus: np.ndarray, A_0: np.ndarray, A_minus_s: np.ndarray, B_u: np.ndarray,
    F_full: np.ndarray, L_full: np.ndarray, G: np.ndarray, N: np.ndarray,
) -> float:
    """Max residual of the equilibrium conditions under the solved rule.

    Substituting ``y_t = F s_{t-1} + L u_t`` and ``s_t = G s_{t-1} + N u_t``
    into ``A_+ E_t y_{t+1} + A_0 y_t + A_- s_{t-1} + B_u u_t = 0`` gives
    ``A_+ F G + A_0 F + A_- = 0`` (state terms) and ``A_+ F N + A_0 L + B_u
    = 0`` (shock terms).
    """
    r_x = A_plus @ F_full @ G + A_0 @ F_full + A_minus_s
    r_u = A_plus @ F_full @ N + A_0 @ L_full + B_u
    return max(float(np.max(np.abs(r_x))) if r_x.size else 0.0,
               float(np.max(np.abs(r_u))) if r_u.size else 0.0)


def build_dynare(
    equations: Callable | str | Path,
    *,
    variables: Sequence[str] | None = None,
    shocks: Sequence[str] | None = None,
    params: Mapping[str, float] | None = None,
    steady_state: Mapping[str, float] | Sequence[float] | None = None,
    guess: Mapping[str, float] | Sequence[float] | None = None,
    solve_algo: str = "block",
    homotopy: Mapping[str, tuple[float, float]] | None = None,
    homotopy_steps: int = 10,
    states: Sequence[str] | None = None,
    order: int = 1,
    shock_cov: np.ndarray | None = None,
    tol: float = 1e-8,
    method: str = "complex",
    verify_derivatives: bool = True,
    check_steady_state: bool = True,
    strict: bool = True,
    qz_criterium: float = 1.0 + 1e-8,
) -> LinearModel | PrunedDSGESolution:
    """Solve a DSGE model written in Dynare canonical lead-lag form.

    The model ``E_t f(y_{t+1}, y_t, y_{t-1}, u_t) = 0`` is linearised in
    levels around the steady state and solved as the Klein system whose
    predetermined vector stacks the lagged states ``s_{t-1}`` and whose
    non-predetermined vector is *every* current variable, with identity
    rows ``s_t = P_s y_t``. Variables that appear at ``t-1`` and ``t+1``
    simultaneously (the TFP process inside an Euler equation, habits,
    investment adjustment costs) are handled exactly; nothing is dropped.
    The decision rules come out in Dynare's timing,
    ``y_t = ys + ghx (s_{t-1} - ss) + ghu u_t``.

    Parameters
    ----------
    equations : Callable | str | Path
        Either a function of five arguments ``eqs(lead, curr, lag, shocks, params)``,
        or a Dynare ``.mod`` file path / source string. When a string or Path is
        passed, model loading delegates directly to :func:`load_mod`.
    variables : Sequence[str], optional
        Names of all endogenous variables in model order (required when equations is Callable).
    shocks : Sequence[str], optional
        Names of structural innovations (required when equations is Callable).
    params : Mapping[str, float], optional
        Parameter values.
    steady_state : Mapping[str, float] | Sequence[float], optional
        Exact steady-state values. Verified against the equations.
    guess : Mapping[str, float] | Sequence[float], optional
        Initial guess for numerical steady-state solver.
    states : Sequence[str], optional
        Predetermined state variables. If None (default), **automatically
        detected** from the Jacobian columns with respect to ``lag``.
    order : {1, 2, 3}, default 1
        Perturbation order: 1 (linear), 2 (pruned quadratic), 3 (pruned cubic).
    shock_cov : np.ndarray, optional
        Innovation covariance Σ_u. Used as the default for theoretical
        moments, IRF sizes (one standard deviation) and simulations, and
        for the second-order risk correction.
    tol : float, default 1e-8
        Numerical tolerance for steady-state residual check.
    method : {'complex', 'central'}, default 'complex'
        Differentiation method for Jacobians (and, at order 2, Hessians).
    verify_derivatives : bool, default True
        Cross-check the complex-step Jacobians against finite differences
        and raise :class:`ModelError` if they disagree — the only way to
        catch a residual function that is not analytic (``abs``, ``max``,
        ``float()`` casts), since complex-step fails silently on those.
    check_steady_state : bool, default True
        Verify that the supplied steady state solves the equations.
    strict : bool, default True
        Raise :class:`~puremacro.dsge.klein.BlanchardKahnError` when the
        model has no unique stable solution. With ``strict=False`` the
        model is returned with ``solution.eu`` flagged and zero matrices;
        its decision-rule methods then refuse to run.

    Returns
    -------
    LinearModel | PrunedDSGESolution
        Solved model equipped with `.decision_rules()`, `.theoretical_moments()`,
        `.fevd()`, `.irf()`, and `.simulate()` (order 1), or the pruned
        second-order solution (order 2).
    """
    if isinstance(equations, (str, Path)):
        return load_mod(
            equations,
            params=params,
            steady_state=steady_state,
            guess=guess,
            solve_algo=solve_algo,
            homotopy=homotopy,
            homotopy_steps=homotopy_steps,
            states=states,
            order=order,
            shock_cov=shock_cov,
            tol=tol,
            method=method,
            verify_derivatives=verify_derivatives,
            strict=strict,
            qz_criterium=qz_criterium,
        )

    if variables is None:
        raise TypeError("build_dynare() missing required keyword-only argument: 'variables'")
    if shocks is None:
        raise TypeError("build_dynare() missing required keyword-only argument: 'shocks'")

    if method not in ("complex", "central"):
        raise ValueError(f"unknown method {method!r}; expected 'complex' or 'central'")

    if order == 2:
        return solve_dynare_2nd_order(
            equations,
            variables=variables,
            shocks=shocks,
            params=params,
            steady_state=steady_state,
            guess=guess,
            solve_algo=solve_algo,
            homotopy=homotopy,
            homotopy_steps=homotopy_steps,
            states=states,
            shock_cov=shock_cov,
            tol=tol,
            method=method,
            verify_derivatives=verify_derivatives,
            check_steady_state=check_steady_state,
            strict=strict,
        )
    elif order == 3:
        return solve_dynare_3rd_order(
            equations,
            variables=variables,
            shocks=shocks,
            params=params,
            steady_state=steady_state,
            guess=guess,
            solve_algo=solve_algo,
            homotopy=homotopy,
            homotopy_steps=homotopy_steps,
            states=states,
            shock_cov=shock_cov,
            tol=tol,
            method=method,
            verify_derivatives=verify_derivatives,
            check_steady_state=check_steady_state,
            strict=strict,
        )
    elif order != 1:
        raise ValueError(f"unsupported perturbation order {order}; must be 1, 2, or 3")

    variables = list(variables)
    shocks = list(shocks)
    if len(set(variables)) != len(variables):
        raise ModelError(f"duplicate variable names in {variables}")
    if len(set(shocks)) != len(shocks):
        raise ModelError(f"duplicate shock names in {shocks}")
    par_dict = dict(params or {})
    par_vec = _Vec(list(par_dict.keys()), list(par_dict.values()), what="parameter")
    n_vars = len(variables)
    n_shocks = len(shocks)

    # 1. Steady-state solve / verification
    def ss_res(y_arr: np.ndarray) -> np.ndarray:
        y_v = _Vec(variables, y_arr, what="variable")
        e_0 = _Vec(shocks, np.zeros(n_shocks), what="shock")
        out = equations(y_v, y_v, y_v, e_0, par_vec)
        return np.asarray(out, dtype=float)

    if steady_state is not None:
        if isinstance(steady_state, Mapping):
            missing = [v for v in variables if v not in steady_state]
            if missing:
                raise ModelError(f"steady_state is missing values for {missing}")
            ss_arr = np.array([float(steady_state[v]) for v in variables])
        else:
            ss_arr = np.asarray(steady_state, dtype=float)
        res_0 = ss_res(ss_arr)
        if res_0.shape != (n_vars,):
            raise ModelError(
                f"equations() returned {res_0.shape[0] if res_0.ndim else 1} "
                f"residual(s) for {n_vars} variables — a square system needs one "
                f"equation per variable"
            )
        max_err = float(np.max(np.abs(res_0)))
        if check_steady_state and max_err > tol:
            worst_eq = int(np.argmax(np.abs(res_0)))
            raise SteadyStateError(
                f"the supplied steady state does not solve the model: "
                f"max|f(ss, ss, 0)| = {max_err:.3e}. Worst equation: index {worst_eq}."
            )
    else:
        if guess is None:
            raise SteadyStateError("must supply either 'steady_state' or 'guess'")
        if isinstance(guess, Mapping):
            g_arr = np.array([float(guess.get(v, 1.0)) for v in variables])
        else:
            g_arr = np.asarray(guess, dtype=float)
        res_g = ss_res(g_arr)
        if res_g.shape != (n_vars,):
            raise ModelError(
                f"equations() returned {res_g.shape[0] if res_g.ndim else 1} "
                f"residual(s) for {n_vars} variables — a square system needs one "
                f"equation per variable"
            )
        from .steady import steady
        ss_arr, _ = steady(
            equations,
            variables=variables,
            guess=guess,
            params=par_dict,
            shocks=shocks,
            solve_algo=solve_algo,
            homotopy=homotopy,
            homotopy_steps=homotopy_steps,
            tol=tol,
        )

    ss_series = pd.Series(ss_arr, index=variables, name="steady_state")

    # 2. Jacobians at steady state:  A_+ = df/d(lead), A_0 = df/d(curr),
    #    A_- = df/d(lag), B_u = df/d(shocks)
    compiled = getattr(equations, "_compiled", None)
    if compiled is not None:
        A_plus, A_0, A_minus, B_u = compiled.eval_first_order(
            lead=ss_arr,
            curr=ss_arr,
            lag=ss_arr,
            shocks=np.zeros(n_shocks),
            params=par_dict,
        )
    else:
        A_plus, A_0, A_minus, B_u = _lead_lag_jacobians(
            equations, variables, shocks, par_vec, ss_arr, method,
            verify=verify_derivatives,
        )

    # 3. Variable role classification: a state is anything that enters
    #    with a lag (or was declared predetermined).
    if states is None:
        states_tuple = tuple(
            v for j, v in enumerate(variables)
            if float(np.linalg.norm(A_minus[:, j])) > 1e-10
        )
    else:
        states_tuple = tuple(states)
        unknown = [s for s in states_tuple if s not in variables]
        if unknown:
            raise ModelError(f"states {unknown} are not in variables {variables}")
        if len(set(states_tuple)) != len(states_tuple):
            raise ModelError(f"duplicate state names in {list(states_tuple)}")

    controls_tuple = tuple(v for v in variables if v not in states_tuple)

    if len(states_tuple) == 0:
        raise ModelError(
            "no predetermined state variables detected: column norm of A_- is zero for all variables. "
            "Pass explicit states=[...] if this is an atypical model."
        )

    # 4. Klein QZ on the stacked system z_t = [s_{t-1}; y_t]
    state_idx = [variables.index(v) for v in states_tuple]
    A_klein, B_klein, C_klein, sol_full, F_full, L_full = _solve_lead_lag_system(
        A_plus, A_0, A_minus, B_u, state_idx, strict=strict, qz_criterium=qz_criterium,
    )

    n_s = len(states_tuple)
    ctrl_idx = [variables.index(v) for v in controls_tuple]
    if tuple(sol_full.eu) == (1, 1):
        G = F_full[state_idx]
        N = L_full[state_idx]
        F = F_full[ctrl_idx]
        L = L_full[ctrl_idx]
        resid = _check_lead_lag_residual(
            A_plus, A_0, A_minus[:, state_idx], B_u, F_full, L_full, G, N,
        )
        scale = max(1.0, float(np.max(np.abs(A_plus))), float(np.max(np.abs(A_0))),
                    float(np.max(np.abs(A_minus))), float(np.max(np.abs(B_u))) if B_u.size else 1.0)
        if resid > 1e-6 * scale:
            raise ModelError(
                f"the QZ solution does not satisfy the model's own equilibrium "
                f"conditions (max residual {resid:.2e}); the pencil is probably "
                "numerically singular. Check the model for redundant or "
                "inconsistent equations."
            )
    else:
        G = np.zeros((n_s, n_s))
        N = np.zeros((n_s, n_shocks))
        F = np.zeros((len(ctrl_idx), n_s))
        L = np.zeros((len(ctrl_idx), n_shocks))

    solution = KleinSolution(
        G=G, F=F, N=N, L=L, eu=tuple(sol_full.eu), eigenvalues=sol_full.eigenvalues,
    )

    # Dynare-form models are approximated in levels: every reported
    # deviation is a level deviation, whatever the sign of the steady state.
    units = {v: "level" for v in variables}

    res_norm = float(np.max(np.abs(ss_res(ss_arr))))

    model = LinearModel(
        variables=tuple(variables),
        states=states_tuple,
        controls=controls_tuple,
        shocks=tuple(shocks),
        steady_state=ss_series,
        units=units,
        solution=solution,
        A=A_klein,
        B=B_klein,
        C=C_klein,
        method=method,
        residual_norm=res_norm,
        _dynare_equations=equations,
        _params=par_dict,
        _A_plus=A_plus,
        _A_0=A_0,
        _A_minus=A_minus,
        _B_u=B_u,
        _shock_cov=None if shock_cov is None else np.asarray(shock_cov, dtype=float),
        timing="dynare",
    )
    dag = getattr(equations, "_dag", None)
    is_linear = getattr(compiled, "is_linear", False) if compiled is not None else False
    if dag is not None and getattr(dag, "is_linear", False):
        is_linear = True
    if dag is not None:
        object.__setattr__(model, "_dag", dag)
    if compiled is not None:
        object.__setattr__(model, "_compiled", compiled)
    object.__setattr__(model, "_is_linear", is_linear)
    object.__setattr__(model, "_qz_criterium", qz_criterium)
    return model


def _first_order_pieces(
    m: "LinearModel", vars_list: list[str], shocks_list: list[str]
) -> tuple:
    """Unpack the first-order rules a second-order solve needs from ``m``.

    Returns ``(states, controls, n_x, n_y, g_x, g_u, P_s, P_c, h_x, h_u)``
    where ``g_x``/``g_u`` are Dynare's ``ghx``/``ghu`` over all variables and
    ``P_s``/``P_c`` select the state and control rows.
    """
    states_list = list(m.states)
    controls_list = list(m.controls)
    n_x, n_y, n_v = len(states_list), len(controls_list), len(vars_list)

    dr = m.decision_rules()
    g_x = dr.ghx.loc[vars_list, states_list].to_numpy()
    g_u = dr.ghu.loc[vars_list, shocks_list].to_numpy()

    P_s = np.zeros((n_x, n_v))
    for j, name in enumerate(states_list):
        P_s[j, vars_list.index(name)] = 1.0

    P_c = np.zeros((n_y, n_v))
    for j, name in enumerate(controls_list):
        P_c[j, vars_list.index(name)] = 1.0

    return (states_list, controls_list, n_x, n_y,
            g_x, g_u, P_s, P_c, P_s @ g_x, P_s @ g_u)


def _lag_curvature_states(
    H_f: np.ndarray, vars_list: list[str], states_list: list[str], n_v: int
) -> list[str]:
    """Variables whose lag enters f only through a second-order term.

    ``build_dynare`` classifies a variable as a state from the column norm of
    ``df/d(lag)``, which is exactly zero for a variable whose lag appears only
    inside a quadratic (``y = v(-1)^2``, ``y = a(-1)*b(-1)``).  Such a variable
    is missing from the state vector, so its whole curvature block of ``ghxx``
    is silently zero.  The lag block of the Hessian sees it: if ``f`` does not
    depend on ``lag_j`` at all, row and column ``2N + j`` of ``H_f`` are
    identically zero (the perturbed coordinate never reaches the residual), so
    a non-negligible entry there is a genuine second-order lag dependence.
    """
    lag_block = np.abs(H_f[:, 2 * n_v:3 * n_v, :]).max(axis=(0, 2))
    scale = max(1.0, float(np.abs(H_f).max()))
    known = set(states_list)
    return [
        v for j, v in enumerate(vars_list)
        if v not in known and float(lag_block[j]) > 1e-8 * scale
    ]


def solve_dynare_2nd_order(
    equations: Callable,
    *,
    variables: Sequence[str],
    shocks: Sequence[str],
    params: Mapping[str, float] | None = None,
    steady_state: Mapping[str, float] | Sequence[float] | None = None,
    guess: Mapping[str, float] | Sequence[float] | None = None,
    solve_algo: str = "block",
    homotopy: Mapping[str, tuple[float, float]] | None = None,
    homotopy_steps: int = 10,
    states: Sequence[str] | None = None,
    shock_cov: np.ndarray | None = None,
    tol: float = 1e-8,
    method: str = "complex",
    verify_derivatives: bool = True,
    check_steady_state: bool = True,
    strict: bool = True,
) -> PrunedDSGESolution:
    """Solve second-order DSGE perturbation with pruning (Schmitt-Grohé & Uribe 2004, Kim et al. 2008).

    Solves for the quadratic policy matrices (``ghxx``, ``ghxu``, ``ghuu``)
    and the volatility risk correction (``ghs2``) of Dynare's second-order
    decision rule

        y_t = ys + 0.5 ghs2 + ghx x + ghu u + 0.5 ghxx (x ⊗ x)
              + ghxu (x ⊗ u) + 0.5 ghuu (u ⊗ u),      x = s_{t-1} - ss,

    from the canonical dynamic representation
    ``E_t [ f(y_{t+1}, y_t, y_{t-1}, u_t; θ) ] = 0``.

    Parameters
    ----------
    equations : Callable
        Function of five arguments: ``eqs(lead, curr, lag, shocks, params)``.
    variables : Sequence[str]
        Names of all endogenous variables in model order.
    shocks : Sequence[str]
        Names of structural innovations.
    params : Mapping[str, float], optional
        Parameter values.
    steady_state : Mapping[str, float] | Sequence[float], optional
        Exact steady state. Verified against equilibrium conditions.
    guess : Mapping[str, float] | Sequence[float], optional
        Initial guess for numerical steady-state solver.
    states : Sequence[str], optional
        Predetermined states. If None, auto-detected from the columns of
        ``df/d(lag)`` and then widened, Dynare-style, with any variable whose
        lag reaches the model only through a second-order term (``y = v(-1)^2``,
        ``y = a(-1)*b(-1)``) — such a variable has an all-zero column in
        ``df/d(lag)`` but a non-zero lag block in the Hessian, and leaving it
        out would zero its whole contribution to ``ghxx``. An explicit list is
        used as given; if it omits such a variable a ``UserWarning`` says so.
    shock_cov : np.ndarray, optional
        Covariance matrix of innovations Σ_u used for the risk correction
        ``ghs2`` and as the default shock covariance of the returned
        solution. Defaults to the identity matrix. (When the model comes from
        a .mod file, :func:`load_mod` passes the ``shocks;`` block instead,
        in which a shock the block does not mention has variance 0.)
    tol : float, default 1e-8
        Tolerance for steady-state residual check.
    method : {'complex', 'central'}, default 'complex'
        Differentiation method. ``'central'`` uses finite differences for
        the Jacobians and the Hessians (about 1e-5 relative accuracy) and
        is the choice for residual functions that are not analytic.
    verify_derivatives : bool, default True
        Cross-check complex-step Jacobians against finite differences.
    check_steady_state : bool, default True
        Verify that a supplied steady state solves the equations.
    strict : bool, default True
        Raise :class:`~puremacro.dsge.klein.BlanchardKahnError` when the
        first-order model has no unique stable solution.

    Returns
    -------
    PrunedDSGESolution
        Second-order pruned DSGE solution equipped with `.simulate()`, `.girf()`,
        and `.stochastic_steady_state()`.
    """
    if method not in ("complex", "central"):
        raise ValueError(f"unknown method {method!r}; expected 'complex' or 'central'")

    # 1. Solve 1st-order model via Klein QZ
    m = build_dynare(
        equations,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
        guess=guess,
        solve_algo=solve_algo,
        homotopy=homotopy,
        homotopy_steps=homotopy_steps,
        states=states,
        order=1,
        tol=tol,
        method=method,
        verify_derivatives=verify_derivatives,
        check_steady_state=check_steady_state,
        strict=True,
    )

    assert isinstance(m, LinearModel), "Order 2 perturbation requires a solved LinearModel"
    assert m.steady_state is not None, "Order 2 perturbation requires steady state series"

    vars_list = list(m.variables)
    shocks_list = list(m.shocks)

    N = len(vars_list)
    n_e = len(shocks_list)

    (states_list, controls_list, n_x, n_y,
     g_x, g_u, P_s, P_c, h_x, h_u) = _first_order_pieces(m, vars_list, shocks_list)

    compiled = getattr(equations, "_compiled", None) or getattr(m, "_compiled", None)
    dag = getattr(equations, "_dag", None) or getattr(m, "_dag", None)
    is_linear = getattr(m, "_is_linear", False) or (compiled is not None and getattr(compiled, "is_linear", False))
    par_dict = dict(params or {})
    par_vec = _Vec(list(par_dict.keys()), list(par_dict.values()), what="parameter")

    if is_linear:
        if shock_cov is None:
            sigma_u = np.eye(n_e)
        else:
            sigma_u = np.asarray(shock_cov, dtype=float)
            if sigma_u.shape != (n_e, n_e):
                raise ValueError(f"shock_cov must be ({n_e}, {n_e}), got {sigma_u.shape}")
        g_xx = np.zeros((N, n_x**2), dtype=float)
        g_xu = np.zeros((N, n_x * n_e), dtype=float)
        g_uu = np.zeros((N, n_e**2), dtype=float)
        g_ss = np.zeros(N, dtype=float)
        H_xx = P_s @ g_xx
        G_xx = P_c @ g_xx
        H_xu = P_s @ g_xu
        G_xu = P_c @ g_xu
        H_uu = P_s @ g_uu
        G_uu = P_c @ g_uu
        H_ss = P_s @ g_ss
        G_ss = P_c @ g_ss
        sol = PrunedDSGESolution(
            G=h_x,
            N=h_u,
            F=P_c @ g_x,
            L=P_c @ g_u,
            H_xx=H_xx,
            H_sigmasigma=H_ss,
            G_xx=G_xx,
            G_sigmasigma=G_ss,
            state_names=tuple(states_list),
            control_names=tuple(controls_list),
            shock_names=tuple(shocks_list),
            H_xu=H_xu,
            H_uu=H_uu,
            G_xu=G_xu,
            G_uu=G_uu,
            steady_state=m.steady_state,
            variable_names=tuple(vars_list),
            ghx=g_x,
            ghu=g_u,
            ghxx=g_xx,
            ghxu=g_xu,
            ghuu=g_uu,
            ghs2=g_ss,
            params=par_dict,
            shock_cov=sigma_u,
            first_order=m,
        )
        if dag is not None:
            object.__setattr__(sol, "_dag", dag)
        if compiled is not None:
            object.__setattr__(sol, "_compiled", compiled)
        object.__setattr__(sol, "_is_linear", True)
        return sol

    # 2. First and second derivatives of f at the steady state, stacked
    #    over (lead, curr, lag, shocks)
    ss_arr = m.steady_state.loc[vars_list].to_numpy()
    e0_arr = np.zeros(n_e)
    u0 = np.concatenate([ss_arr, ss_arr, ss_arr, e0_arr])
    K_vars = 3 * N + n_e

    par_dict = dict(params or {})
    par_vec = _Vec(list(par_dict.keys()), list(par_dict.values()), what="parameter")

    A_plus = np.asarray(m._A_plus, dtype=float)
    A_0 = np.asarray(m._A_0, dtype=float)

    if compiled is not None:
        H_f = compiled.eval_second_order(
            lead=ss_arr,
            curr=ss_arr,
            lag=ss_arr,
            shocks=e0_arr,
            params=par_dict,
        )
    else:
        def eval_f(u_vec):
            lead = _Vec(vars_list, u_vec[0:N])
            curr = _Vec(vars_list, u_vec[N:2 * N])
            lag = _Vec(vars_list, u_vec[2 * N:3 * N])
            shk = _Vec(shocks_list, u_vec[3 * N:3 * N + n_e])
            return np.asarray(equations(lead, curr, lag, shk, par_vec))

        if method == "complex":
            hc = _CSTEP

            def grad_f(u_vec):
                G_mat = np.zeros((N, K_vars))
                base = np.asarray(u_vec, dtype=complex)
                for q in range(K_vars):
                    pert = base.copy()
                    pert[q] += 1j * hc
                    G_mat[:, q] = eval_f(pert).imag / hc
                return G_mat

            hd = 1e-5
        else:
            def grad_f(u_vec):
                G_mat = np.zeros((N, K_vars))
                base = np.asarray(u_vec, dtype=float)
                for q in range(K_vars):
                    step = _FDSTEP * max(1.0, abs(base[q]))
                    up, dn = base.copy(), base.copy()
                    up[q] += step
                    dn[q] -= step
                    G_mat[:, q] = (eval_f(up).astype(float) - eval_f(dn).astype(float)) / (2.0 * step)
                return G_mat

            hd = 1e-4

        H_f = np.zeros((N, K_vars, K_vars))
        for p in range(K_vars):
            scale = max(1.0, abs(u0[p]))
            h_step = hd * scale
            up = u0.copy()
            up[p] += h_step
            um = u0.copy()
            um[p] -= h_step
            gp = grad_f(up)
            gm = grad_f(um)
            H_f[:, p, :] = (gp - gm) / (2.0 * h_step)

    for i in range(N):
        H_f[i] = 0.5 * (H_f[i] + H_f[i].T)

    # 2b. A variable whose lag enters only through a second-order term has an
    #     all-zero column in df/d(lag), so the first-order state detection in
    #     build_dynare misses it and its entire curvature block of ghxx would
    #     come out zero.  The Hessian does not depend on the state split, so we
    #     can screen it here and re-solve the first-order model with the state
    #     set Dynare would have used (every variable that appears lagged).
    missed_states = _lag_curvature_states(H_f, vars_list, states_list, N)
    if missed_states and states is None:
        enlarged = [v for v in vars_list if v in set(states_list) | set(missed_states)]
        try:
            m = build_dynare(
                equations,
                variables=variables,
                shocks=shocks,
                params=params,
                steady_state=steady_state,
                guess=guess,
                states=enlarged,
                order=1,
                tol=tol,
                method=method,
                verify_derivatives=verify_derivatives,
                check_steady_state=check_steady_state,
                strict=True,
            )
            assert isinstance(m, LinearModel)
            (states_list, controls_list, n_x, n_y,
             g_x, g_u, P_s, P_c, h_x, h_u) = _first_order_pieces(m, vars_list, shocks_list)
            A_plus = np.asarray(m._A_plus, dtype=float)
            A_0 = np.asarray(m._A_0, dtype=float)
        except Exception as exc:
            warnings.warn(
                f"solve_dynare_2nd_order: the lag of {missed_states} enters the "
                "model only through a second-order term, so df/d(lag) does not "
                "see it and it is not in the auto-detected state vector "
                f"{tuple(states_list)}. Re-solving with it as a state failed "
                f"({type(exc).__name__}: {exc}), so the second-order rules ghxx "
                "and the risk correction ghs2 DROP that variable's curvature "
                "and are wrong. Pass an explicit states=[...] to control the "
                "state vector.",
                UserWarning,
                stacklevel=2,
            )
    elif missed_states:
        warnings.warn(
            f"solve_dynare_2nd_order: the lag of {missed_states} enters the "
            "model through a second-order term but these names are not in the "
            f"states={list(states)!r} you passed. Their curvature contribution "
            "to ghxx, ghxu and the risk correction ghs2 is dropped, so those "
            "coefficients and the ergodic mean are wrong. Add them to states=.",
            UserWarning,
            stacklevel=2,
        )

    # 3. Second-order systems for g_xx, g_xu, g_uu.  With
    #    y_{t+1} = g(h(x, u), u', σ), y_t = g(x, u), y_{t-1} -> x (states only):
    I_states = np.zeros((N, n_x))
    for j, s in enumerate(states_list):
        I_states[vars_list.index(s), j] = 1.0

    M_x = np.vstack([g_x @ h_x, g_x, I_states, np.zeros((n_e, n_x))])
    M_u = np.vstack([g_x @ h_u, g_u, np.zeros((N, n_e)), np.eye(n_e)])

    K_xx_tensor = np.zeros((N, n_x**2))
    K_xu_tensor = np.zeros((N, n_x * n_e))
    K_uu_tensor = np.zeros((N, n_e**2))
    for i in range(N):
        K_xx_tensor[i] = (M_x.T @ H_f[i] @ M_x).flatten()
        K_xu_tensor[i] = (M_x.T @ H_f[i] @ M_u).flatten()
        K_uu_tensor[i] = (M_u.T @ H_f[i] @ M_u).flatten()

    # (A_0 + A_+ g_x P_s) g_xx + A_+ g_xx (h_x ⊗ h_x) = -K_xx
    A_hat = A_0 + A_plus @ g_x @ P_s
    try:
        g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx_tensor)
    except Exception:
        hx_kron = np.kron(h_x, h_x)
        sys_mat = np.kron(np.eye(n_x**2), A_hat) + np.kron(hx_kron.T, A_plus)
        rhs_xx = -K_xx_tensor.reshape(-1, order="F")
        try:
            vec_gxx = scipy.linalg.solve(sys_mat, rhs_xx)
        except scipy.linalg.LinAlgError:
            vec_gxx = scipy.linalg.lstsq(sys_mat, rhs_xx)[0]
        g_xx = vec_gxx.reshape((N, n_x**2), order="F")

    H_xx = P_s @ g_xx
    G_xx = P_c @ g_xx

    # State-shock cross terms: A_hat g_xu = -[ A_+ g_xx (h_x ⊗ h_u) + K_xu ]
    hx_hu = np.kron(h_x, h_u)
    rhs_xu = -(A_plus @ g_xx @ hx_hu + K_xu_tensor)
    try:
        g_xu = scipy.linalg.solve(A_hat, rhs_xu)
    except scipy.linalg.LinAlgError:
        g_xu = scipy.linalg.lstsq(A_hat, rhs_xu)[0]

    H_xu = P_s @ g_xu
    G_xu = P_c @ g_xu

    # Shock quadratic terms: A_hat g_uu = -[ A_+ g_xx (h_u ⊗ h_u) + K_uu ]
    hu_hu = np.kron(h_u, h_u)
    rhs_uu = -(A_plus @ g_xx @ hu_hu + K_uu_tensor)
    try:
        g_uu = scipy.linalg.solve(A_hat, rhs_uu)
    except scipy.linalg.LinAlgError:
        g_uu = scipy.linalg.lstsq(A_hat, rhs_uu)[0]

    H_uu = P_s @ g_uu
    G_uu = P_c @ g_uu

    # 4. Volatility correction g_σσ. Differentiating twice in σ, with
    #    y_{t+1} = g(h(x,u), σ ε', σ) and g_σ = 0:
    #        (A_0 + A_+ g_x P_s + A_+) g_σσ = -[ A_+ g_uu vec(Σ_u)
    #                                          + Σ_i tr(g_u' f_i,y+y+ g_u Σ_u) ]
    #    The first term is next period's *own* shock curvature g_uu — not
    #    g_xx (h_u ⊗ h_u), which would treat u_{t+1} as if it entered
    #    through the state.
    if shock_cov is None:
        sigma_u = np.eye(n_e)
    else:
        sigma_u = np.asarray(shock_cov, dtype=float)
        if sigma_u.shape != (n_e, n_e):
            raise ValueError(f"shock_cov must be ({n_e}, {n_e}), got {sigma_u.shape}")

    W_vec = np.zeros(N)
    for i in range(N):
        W_vec[i] = np.trace(g_u.T @ H_f[i, 0:N, 0:N] @ g_u @ sigma_u)

    vec_sigma = sigma_u.flatten()
    rhs_sig = -(A_plus @ g_uu @ vec_sigma + W_vec)

    sys_sig = A_hat + A_plus
    try:
        g_ss = scipy.linalg.solve(sys_sig, rhs_sig)
    except scipy.linalg.LinAlgError:
        g_ss = scipy.linalg.lstsq(sys_sig, rhs_sig)[0]

    H_ss = P_s @ g_ss
    G_ss = P_c @ g_ss

    sol = PrunedDSGESolution(
        G=h_x,
        N=h_u,
        F=P_c @ g_x,
        L=P_c @ g_u,
        H_xx=H_xx,
        H_sigmasigma=H_ss,
        G_xx=G_xx,
        G_sigmasigma=G_ss,
        state_names=tuple(states_list),
        control_names=tuple(controls_list),
        shock_names=tuple(shocks_list),
        H_xu=H_xu,
        H_uu=H_uu,
        G_xu=G_xu,
        G_uu=G_uu,
        steady_state=m.steady_state,
        variable_names=tuple(vars_list),
        ghx=g_x,
        ghu=g_u,
        ghxx=g_xx,
        ghxu=g_xu,
        ghuu=g_uu,
        ghs2=g_ss,
        params=par_dict,
        shock_cov=sigma_u,
        first_order=m,
    )
    if dag is not None:
        object.__setattr__(sol, "_dag", dag)
    if compiled is not None:
        object.__setattr__(sol, "_compiled", compiled)
    object.__setattr__(sol, "_is_linear", False)
    return sol


def solve_dynare_3rd_order(
    equations: Callable,
    *,
    variables: Sequence[str],
    shocks: Sequence[str],
    params: Mapping[str, float] | None = None,
    steady_state: Mapping[str, float] | Sequence[float] | None = None,
    guess: Mapping[str, float] | Sequence[float] | None = None,
    solve_algo: str = "block",
    homotopy: Mapping[str, tuple[float, float]] | None = None,
    homotopy_steps: int = 10,
    states: Sequence[str] | None = None,
    shock_cov: np.ndarray | None = None,
    tol: float = 1e-8,
    method: str = "complex",
    verify_derivatives: bool = True,
    check_steady_state: bool = True,
    strict: bool = True,
) -> Order3PrunedSolution:
    """Solve third-order DSGE perturbation with pruning (Andreasen et al. 2018).

    Solves for the cubic policy tensors (ghxxx, ghxxu, ghxuu, ghuuu) and the
    volatility-risk slope corrections (ghxss, ghuss) of Dynare's third-order
    decision rule:

        y_t = ys + 0.5 ghs2 + ghx x + ghu u + 0.5 ghxx (x ⊗ x)
              + ghxu (x ⊗ u) + 0.5 ghuu (u ⊗ u)
              + (1/6) ghxxx (x ⊗ x ⊗ x) + 0.5 ghxxu (x ⊗ x ⊗ u)
              + 0.5 ghxuu (x ⊗ u ⊗ u) + (1/6) ghuuu (u ⊗ u ⊗ u)
              + 0.5 ghxss x σ^2 + 0.5 ghuss u σ^2,    x = s_{t-1} - ss,

    from the canonical dynamic representation
    E_t [ f(y_{t+1}, y_t, y_{t-1}, u_t; θ) ] = 0.

    Parameters
    ----------
    equations : Callable
        Function of five arguments: eqs(lead, curr, lag, shocks, params).
    variables : Sequence[str]
        Names of all endogenous variables in model order.
    shocks : Sequence[str]
        Names of structural innovations.
    params : Mapping[str, float], optional
        Parameter values.
    steady_state : Mapping[str, float] | Sequence[float], optional
        Exact steady state. Verified against equilibrium conditions.
    guess : Mapping[str, float] | Sequence[float], optional
        Initial guess for numerical steady-state solver.
    states : Sequence[str], optional
        Predetermined states. If None, auto-detected from df/d(lag).
    shock_cov : np.ndarray, optional
        Covariance matrix of innovations Σ_u. Defaults to identity matrix.
    tol : float, default 1e-8
        Tolerance for steady-state residual check.
    method : {'complex', 'central'}, default 'complex'
        Differentiation method.
    verify_derivatives : bool, default True
        Cross-check complex-step Jacobians against finite differences.
    check_steady_state : bool, default True
        Verify that a supplied steady state solves the equations.
    strict : bool, default True
        Raise BlanchardKahnError when the model has no unique stable solution.

    Returns
    -------
    Order3PrunedSolution
        Third-order pruned DSGE solution equipped with .simulate(), .girf(),
        and .stochastic_steady_state().
    """
    if method not in ("complex", "central"):
        raise ValueError(f"unknown method {method!r}; expected 'complex' or 'central'")

    # 1. Solve 1st-order model via Klein QZ
    m = build_dynare(
        equations,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
        guess=guess,
        solve_algo=solve_algo,
        homotopy=homotopy,
        homotopy_steps=homotopy_steps,
        states=states,
        order=1,
        tol=tol,
        method=method,
        verify_derivatives=verify_derivatives,
        check_steady_state=check_steady_state,
        strict=True,
    )

    assert isinstance(m, LinearModel), "Order 3 perturbation requires a solved LinearModel"
    assert m.steady_state is not None, "Order 3 perturbation requires steady state series"

    vars_list = list(m.variables)
    shocks_list = list(m.shocks)

    N = len(vars_list)
    n_e = len(shocks_list)

    (states_list, controls_list, n_x, n_y,
     g_x, g_u, P_s, P_c, h_x, h_u) = _first_order_pieces(m, vars_list, shocks_list)

    compiled = getattr(equations, "_compiled", None) or getattr(m, "_compiled", None)
    dag = getattr(equations, "_dag", None) or getattr(m, "_dag", None)
    is_linear = getattr(m, "_is_linear", False) or (compiled is not None and getattr(compiled, "is_linear", False))
    par_dict = dict(params or {})
    par_vec = _Vec(list(par_dict.keys()), list(par_dict.values()), what="parameter")

    if shock_cov is None:
        sigma_u = np.eye(n_e)
    else:
        sigma_u = np.asarray(shock_cov, dtype=float)
        if sigma_u.shape != (n_e, n_e):
            raise ValueError(f"shock_cov must be ({n_e}, {n_e}), got {sigma_u.shape}")

    if is_linear:
        g_xx = np.zeros((N, n_x**2), dtype=float)
        g_xu = np.zeros((N, n_x * n_e), dtype=float)
        g_uu = np.zeros((N, n_e**2), dtype=float)
        g_ss = np.zeros(N, dtype=float)
        g_xxx = np.zeros((N, n_x**3), dtype=float)
        g_xxu = np.zeros((N, (n_x**2) * n_e), dtype=float)
        g_xuu = np.zeros((N, n_x * (n_e**2)), dtype=float)
        g_uuu = np.zeros((N, n_e**3), dtype=float)
        g_x_ss = np.zeros((N, n_x), dtype=float)
        g_u_ss = np.zeros((N, n_e), dtype=float)

        sol = Order3PrunedSolution(
            G=h_x,
            N=h_u,
            F=P_c @ g_x,
            L=P_c @ g_u,
            H_xx=P_s @ g_xx,
            H_sigmasigma=P_s @ g_ss,
            G_xx=P_c @ g_xx,
            G_sigmasigma=P_c @ g_ss,
            H_xxx=P_s @ g_xxx,
            H_xxu=P_s @ g_xxu,
            H_xuu=P_s @ g_xuu,
            H_uuu=P_s @ g_uuu,
            H_x_sigmasigma=P_s @ g_x_ss,
            H_u_sigmasigma=P_s @ g_u_ss,
            G_xxx=P_c @ g_xxx,
            G_xxu=P_c @ g_xxu,
            G_xuu=P_c @ g_xuu,
            G_uuu=P_c @ g_uuu,
            G_x_sigmasigma=P_c @ g_x_ss,
            G_u_sigmasigma=P_c @ g_u_ss,
            state_names=tuple(states_list),
            control_names=tuple(controls_list),
            shock_names=tuple(shocks_list),
            H_xu=P_s @ g_xu,
            H_uu=P_s @ g_uu,
            G_xu=P_c @ g_xu,
            G_uu=P_c @ g_uu,
            steady_state=m.steady_state,
            variable_names=tuple(vars_list),
            ghx=g_x,
            ghu=g_u,
            ghxx=g_xx,
            ghxu=g_xu,
            ghuu=g_uu,
            ghs2=g_ss,
            ghxxx=g_xxx,
            ghxxu=g_xxu,
            ghxuu=g_xuu,
            ghuuu=g_uuu,
            ghxss=g_x_ss,
            ghuss=g_u_ss,
            params=par_dict,
            shock_cov=sigma_u,
            first_order=m,
        )
        if dag is not None:
            object.__setattr__(sol, "_dag", dag)
        if compiled is not None:
            object.__setattr__(sol, "_compiled", compiled)
        object.__setattr__(sol, "_is_linear", True)
        return sol

    # 2. Coordinates and derivative evaluation
    ss_arr = m.steady_state.loc[vars_list].to_numpy()
    e0_arr = np.zeros(n_e)
    u0 = np.concatenate([ss_arr, ss_arr, ss_arr, e0_arr])
    K_vars = 3 * N + n_e

    A_plus = np.asarray(m._A_plus, dtype=float)
    A_0 = np.asarray(m._A_0, dtype=float)

    if compiled is not None:
        H_f = compiled.eval_second_order(
            lead=ss_arr,
            curr=ss_arr,
            lag=ss_arr,
            shocks=e0_arr,
            params=par_dict,
        )
    else:
        def eval_f(u_vec):
            lead = _Vec(vars_list, u_vec[0:N])
            curr = _Vec(vars_list, u_vec[N:2 * N])
            lag = _Vec(vars_list, u_vec[2 * N:3 * N])
            shk = _Vec(shocks_list, u_vec[3 * N:3 * N + n_e])
            return np.asarray(equations(lead, curr, lag, shk, par_vec))

        if method == "complex":
            hc = _CSTEP
            def grad_f(u_vec):
                G_mat = np.zeros((N, K_vars))
                base = np.asarray(u_vec, dtype=complex)
                for q in range(K_vars):
                    pert = base.copy()
                    pert[q] += 1j * hc
                    G_mat[:, q] = eval_f(pert).imag / hc
                return G_mat
            hd = 1e-5
        else:
            def grad_f(u_vec):
                G_mat = np.zeros((N, K_vars))
                base = np.asarray(u_vec, dtype=float)
                for q in range(K_vars):
                    step = _FDSTEP * max(1.0, abs(base[q]))
                    up, dn = base.copy(), base.copy()
                    up[q] += step
                    dn[q] -= step
                    G_mat[:, q] = (eval_f(up).astype(float) - eval_f(dn).astype(float)) / (2.0 * step)
                return G_mat
            hd = 1e-4

        H_f = np.zeros((N, K_vars, K_vars))
        for p in range(K_vars):
            scale = max(1.0, abs(u0[p]))
            h_step = hd * scale
            up = u0.copy()
            up[p] += h_step
            um = u0.copy()
            um[p] -= h_step
            gp = grad_f(up)
            gm = grad_f(um)
            H_f[:, p, :] = (gp - gm) / (2.0 * h_step)

    for i in range(N):
        H_f[i] = 0.5 * (H_f[i] + H_f[i].T)

    missed_states = _lag_curvature_states(H_f, vars_list, states_list, N)
    if missed_states and states is None:
        enlarged = [v for v in vars_list if v in set(states_list) | set(missed_states)]
        try:
            m = build_dynare(
                equations,
                variables=variables,
                shocks=shocks,
                params=params,
                steady_state=steady_state,
                guess=guess,
                states=enlarged,
                order=1,
                tol=tol,
                method=method,
                verify_derivatives=verify_derivatives,
                check_steady_state=check_steady_state,
                strict=True,
            )
            assert isinstance(m, LinearModel)
            (states_list, controls_list, n_x, n_y,
             g_x, g_u, P_s, P_c, h_x, h_u) = _first_order_pieces(m, vars_list, shocks_list)
            A_plus = np.asarray(m._A_plus, dtype=float)
            A_0 = np.asarray(m._A_0, dtype=float)
        except Exception:
            pass

    # 3. Second-order systems
    I_states = np.zeros((N, n_x))
    for j, s in enumerate(states_list):
        I_states[vars_list.index(s), j] = 1.0

    M_x = np.vstack([g_x @ h_x, g_x, I_states, np.zeros((n_e, n_x))])
    M_u = np.vstack([g_x @ h_u, g_u, np.zeros((N, n_e)), np.eye(n_e)])

    K_xx_tensor = np.zeros((N, n_x**2))
    K_xu_tensor = np.zeros((N, n_x * n_e))
    K_uu_tensor = np.zeros((N, n_e**2))
    for i in range(N):
        K_xx_tensor[i] = (M_x.T @ H_f[i] @ M_x).flatten()
        K_xu_tensor[i] = (M_x.T @ H_f[i] @ M_u).flatten()
        K_uu_tensor[i] = (M_u.T @ H_f[i] @ M_u).flatten()

    A_hat = A_0 + A_plus @ g_x @ P_s
    try:
        g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx_tensor)
    except Exception:
        hx_kron = np.kron(h_x, h_x)
        sys_mat = np.kron(np.eye(n_x**2), A_hat) + np.kron(hx_kron.T, A_plus)
        rhs_xx = -K_xx_tensor.reshape(-1, order="F")
        try:
            vec_gxx = scipy.linalg.solve(sys_mat, rhs_xx)
        except scipy.linalg.LinAlgError:
            vec_gxx = scipy.linalg.lstsq(sys_mat, rhs_xx)[0]
        g_xx = vec_gxx.reshape((N, n_x**2), order="F")

    hx_hu = np.kron(h_x, h_u)
    rhs_xu = -(A_plus @ g_xx @ hx_hu + K_xu_tensor)
    try:
        g_xu = scipy.linalg.solve(A_hat, rhs_xu)
    except scipy.linalg.LinAlgError:
        g_xu = scipy.linalg.lstsq(A_hat, rhs_xu)[0]

    hu_hu = np.kron(h_u, h_u)
    rhs_uu = -(A_plus @ g_xx @ hu_hu + K_uu_tensor)
    try:
        g_uu = scipy.linalg.solve(A_hat, rhs_uu)
    except scipy.linalg.LinAlgError:
        g_uu = scipy.linalg.lstsq(A_hat, rhs_uu)[0]

    W_vec = np.zeros(N)
    for i in range(N):
        W_vec[i] = np.trace(g_u.T @ H_f[i, 0:N, 0:N] @ g_u @ sigma_u)

    vec_sigma = sigma_u.flatten()
    rhs_sig = -(A_plus @ g_uu @ vec_sigma + W_vec)
    sys_sig = A_hat + A_plus
    try:
        g_ss = scipy.linalg.solve(sys_sig, rhs_sig)
    except scipy.linalg.LinAlgError:
        g_ss = scipy.linalg.lstsq(sys_sig, rhs_sig)[0]

    h_xx = P_s @ g_xx
    h_xu = P_s @ g_xu
    h_uu = P_s @ g_uu
    h_ss = P_s @ g_ss

    # 4. Third-order systems
    from puremacro.dsge._symbolic import SparseDynamicTensor3D
    if compiled is not None and hasattr(compiled, "eval_third_order"):
        T_f = compiled.eval_third_order(
            lead=ss_arr,
            curr=ss_arr,
            lag=ss_arr,
            shocks=e0_arr,
            params=par_dict,
        )
    else:
        # Fallback numerical approximation
        T_entries: dict[tuple[int, int, int, int], float] = {}
        hd3 = 1e-4
        for p in range(K_vars):
            step = hd3 * max(1.0, abs(u0[p]))
            up = u0.copy()
            up[p] += step
            um = u0.copy()
            um[p] -= step
            gp_p = grad_f(up)
            gp_m = grad_f(um)
            H_p = (gp_p - gp_m) / (2.0 * step)
            for i in range(N):
                for q in range(p, K_vars):
                    for r in range(q, K_vars):
                        val = float(H_p[i, q]) if H_p.ndim == 2 else 0.0
                        if abs(val) > 1e-9:
                            T_entries[(i, p, q, r)] = val
        T_f = SparseDynamicTensor3D(shape=(N, K_vars, K_vars, K_vars), entries=T_entries)

    # 4a. Solve g_xxx
    T_xxx = T_f.contract(M_x, M_x, M_x)  # (N, n_x**3)
    hx_hx = np.kron(h_x, h_x)
    N_xx = np.vstack([g_xx @ hx_hx + g_x @ h_xx, g_xx, np.zeros((N, n_x**2)), np.zeros((n_e, n_x**2))])

    S_xxx = np.zeros((N, n_x**3), dtype=float)
    cross_chain = np.zeros((N, n_x**3), dtype=float)
    if n_x > 0:
        for i in range(N):
            V_i = (N_xx.T @ H_f[i] @ M_x).reshape(n_x, n_x, n_x)
            sym_Vi = V_i + V_i.transpose(0, 2, 1) + V_i.transpose(2, 1, 0)
            S_xxx[i] = sym_Vi.reshape(-1)

        gxx_3d = g_xx.reshape(N, n_x, n_x)
        hxx_3d = h_xx.reshape(n_x, n_x, n_x)
        C1 = np.einsum("ijk,jab,kc->iabc", gxx_3d, hxx_3d, h_x)
        sym_C = C1 + C1.transpose(0, 1, 3, 2) + C1.transpose(0, 3, 2, 1)
        cross_chain = A_plus @ sym_C.reshape(N, n_x**3)

    K_xxx = T_xxx + S_xxx + cross_chain
    try:
        g_xxx = solve_order3_sylvester_kronecker(A_hat, A_plus, h_x, K_xxx)
    except Exception:
        hx3 = np.kron(np.kron(h_x, h_x), h_x)
        sys_3 = np.kron(np.eye(n_x**3), A_hat) + np.kron(hx3.T, A_plus)
        rhs_3 = -K_xxx.reshape(-1, order="F")
        try:
            vec_gxxx = scipy.linalg.solve(sys_3, rhs_3)
        except scipy.linalg.LinAlgError:
            vec_gxxx = scipy.linalg.lstsq(sys_3, rhs_3)[0]
        g_xxx = vec_gxxx.reshape((N, n_x**3), order="F")

    # 4b. Solve g_xxu
    T_xxu = T_f.contract(M_x, M_x, M_u)
    N_xu = np.vstack([g_xx @ hx_hu + g_x @ h_xu, g_xu, np.zeros((N, n_x * n_e)), np.zeros((n_e, n_x * n_e))])
    S_xxu = np.zeros((N, (n_x**2) * n_e), dtype=float)
    cross_xxu = np.zeros((N, (n_x**2) * n_e), dtype=float)
    A_g_xxx_h = np.zeros((N, (n_x**2) * n_e), dtype=float)

    if n_x > 0 and n_e > 0:
        for i in range(N):
            V_xx_u = (N_xx.T @ H_f[i] @ M_u).reshape(n_x, n_x, n_e)
            V_xu_x = (N_xu.T @ H_f[i] @ M_x).reshape(n_x, n_e, n_x).transpose(0, 2, 1)
            V_ux_x = (N_xu.T @ H_f[i] @ M_x).reshape(n_x, n_e, n_x).transpose(2, 0, 1)
            S_xxu[i] = (V_xx_u + V_xu_x + V_ux_x).reshape(-1)

        gxxx_3d = g_xxx.reshape(N, n_x, n_x, n_x)
        A_g_xxx_h = A_plus @ np.einsum("iabc,ad,be,cf->idef", gxxx_3d, h_x, h_x, h_u).reshape(N, (n_x**2) * n_e)
        C_xxu = np.einsum("ijk,jab,kc->iabc", gxx_3d, hxx_3d, h_u) + 2.0 * np.einsum("ijk,jac,kb->iabc", gxx_3d, h_xu.reshape(n_x, n_x, n_e), h_x)
        cross_xxu = A_plus @ C_xxu.reshape(N, (n_x**2) * n_e)

    rhs_xxu = -(A_g_xxx_h + T_xxu + S_xxu + cross_xxu)
    try:
        g_xxu = scipy.linalg.solve(A_hat, rhs_xxu)
    except scipy.linalg.LinAlgError:
        g_xxu = scipy.linalg.lstsq(A_hat, rhs_xxu)[0]

    # 4c. Solve g_xuu
    T_xuu = T_f.contract(M_x, M_u, M_u)
    N_uu = np.vstack([g_xx @ hu_hu + g_x @ h_uu, g_uu, np.zeros((N, n_e**2)), np.zeros((n_e, n_e**2))])
    S_xuu = np.zeros((N, n_x * (n_e**2)), dtype=float)
    cross_xuu = np.zeros((N, n_x * (n_e**2)), dtype=float)
    A_g_xxx_hu = np.zeros((N, n_x * (n_e**2)), dtype=float)

    if n_x > 0 and n_e > 0:
        for i in range(N):
            V_uu_x = (N_uu.T @ H_f[i] @ M_x).reshape(n_e, n_e, n_x).transpose(2, 0, 1)
            V_xu_u = (N_xu.T @ H_f[i] @ M_u).reshape(n_x, n_e, n_e)
            S_xuu[i] = (V_uu_x + 2.0 * V_xu_u).reshape(-1)

        A_g_xxx_hu = A_plus @ np.einsum("iabc,ad,be,cf->idef", gxxx_3d, h_x, h_u, h_u).reshape(N, n_x * (n_e**2))
        C_xuu = np.einsum("ijk,ja,kbc->iabc", gxx_3d, h_x, h_uu.reshape(n_x, n_e, n_e)) + 2.0 * np.einsum("ijk,jab,kc->iabc", gxx_3d, h_xu.reshape(n_x, n_x, n_e), h_u)
        cross_xuu = A_plus @ C_xuu.reshape(N, n_x * (n_e**2))

    rhs_xuu = -(A_g_xxx_hu + T_xuu + S_xuu + cross_xuu)
    try:
        g_xuu = scipy.linalg.solve(A_hat, rhs_xuu)
    except scipy.linalg.LinAlgError:
        g_xuu = scipy.linalg.lstsq(A_hat, rhs_xuu)[0]

    # 4d. Solve g_uuu
    T_uuu = T_f.contract(M_u, M_u, M_u)
    S_uuu = np.zeros((N, n_e**3), dtype=float)
    A_g_uuu = np.zeros((N, n_e**3), dtype=float)
    cross_uuu = np.zeros((N, n_e**3), dtype=float)

    if n_e > 0:
        for i in range(N):
            V_uu_u = (N_uu.T @ H_f[i] @ M_u).reshape(n_e, n_e, n_e)
            sym_Vu = V_uu_u + V_uu_u.transpose(0, 2, 1) + V_uu_u.transpose(2, 1, 0)
            S_uuu[i] = sym_Vu.reshape(-1)

        if n_x > 0:
            A_g_uuu = A_plus @ np.einsum("iabc,ad,be,cf->idef", gxxx_3d, h_u, h_u, h_u).reshape(N, n_e**3)
            C_uuu = np.einsum("ijk,jab,kc->iabc", gxx_3d, h_uu.reshape(n_x, n_e, n_e), h_u)
            sym_Cu = C_uuu + C_uuu.transpose(0, 1, 3, 2) + C_uuu.transpose(0, 3, 2, 1)
            cross_uuu = A_plus @ sym_Cu.reshape(N, n_e**3)

    rhs_uuu = -(A_g_uuu + T_uuu + S_uuu + cross_uuu)
    try:
        g_uuu = scipy.linalg.solve(A_hat, rhs_uuu)
    except scipy.linalg.LinAlgError:
        g_uuu = scipy.linalg.lstsq(A_hat, rhs_uuu)[0]

    # 4e. Solve g_x_ss (volatility risk correction slope)
    K_x_ss = np.zeros((N, n_x), dtype=float)
    if n_x > 0:
        term_hss = A_plus @ g_xx @ np.kron(h_ss.reshape(n_x, 1), np.eye(n_x))
        term_xuu = A_plus @ (g_xuu.reshape(N, n_x, n_e * n_e) @ sigma_u.flatten())
        K_x_ss = term_hss + term_xuu
        try:
            g_x_ss = solve_order1_sylvester(A_hat, A_plus, h_x, K_x_ss)
        except Exception:
            sys_1 = np.kron(np.eye(n_x), A_hat) + np.kron(h_x.T, A_plus)
            vec_gxss = scipy.linalg.lstsq(sys_1, -K_x_ss.reshape(-1, order="F"))[0]
            g_x_ss = vec_gxss.reshape((N, n_x), order="F")
    else:
        g_x_ss = np.zeros((N, 0), dtype=float)

    # 4f. Solve g_u_ss (shock volatility risk correction)
    if n_e > 0 and n_x > 0:
        rhs_uss = -(A_plus @ g_x_ss @ h_u + A_plus @ (g_uuu.reshape(N, n_e, n_e * n_e) @ sigma_u.flatten()))
        try:
            g_u_ss = scipy.linalg.solve(A_hat, rhs_uss)
        except scipy.linalg.LinAlgError:
            g_u_ss = scipy.linalg.lstsq(A_hat, rhs_uss)[0]
    else:
        g_u_ss = np.zeros((N, n_e), dtype=float)

    sol = Order3PrunedSolution(
        G=h_x,
        N=h_u,
        F=P_c @ g_x,
        L=P_c @ g_u,
        H_xx=h_xx,
        H_sigmasigma=h_ss,
        G_xx=P_c @ g_xx,
        G_sigmasigma=P_c @ g_ss,
        H_xxx=P_s @ g_xxx,
        H_xxu=P_s @ g_xxu,
        H_xuu=P_s @ g_xuu,
        H_uuu=P_s @ g_uuu,
        H_x_sigmasigma=P_s @ g_x_ss,
        H_u_sigmasigma=P_s @ g_u_ss,
        G_xxx=P_c @ g_xxx,
        G_xxu=P_c @ g_xxu,
        G_xuu=P_c @ g_xuu,
        G_uuu=P_c @ g_uuu,
        G_x_sigmasigma=P_c @ g_x_ss,
        G_u_sigmasigma=P_c @ g_u_ss,
        state_names=tuple(states_list),
        control_names=tuple(controls_list),
        shock_names=tuple(shocks_list),
        H_xu=h_xu,
        H_uu=h_uu,
        G_xu=P_c @ g_xu,
        G_uu=P_c @ g_uu,
        steady_state=m.steady_state,
        variable_names=tuple(vars_list),
        ghx=g_x,
        ghu=g_u,
        ghxx=g_xx,
        ghxu=g_xu,
        ghuu=g_uu,
        ghs2=g_ss,
        ghxxx=g_xxx,
        ghxxu=g_xxu,
        ghxuu=g_xuu,
        ghuuu=g_uuu,
        ghxss=g_x_ss,
        ghuss=g_u_ss,
        params=par_dict,
        shock_cov=sigma_u,
        first_order=m,
    )
    if dag is not None:
        object.__setattr__(sol, "_dag", dag)
    if compiled is not None:
        object.__setattr__(sol, "_compiled", compiled)
    object.__setattr__(sol, "_is_linear", False)
    return sol


def _extract_ids(raw_str: str) -> list[str]:
    """Extract valid identifier names from declaration string, stripping TeX and attributes."""
    s = re.sub(r"\$[^$]*\$", " ", raw_str)
    s = re.sub(r"\([^)]*\)", " ", s)
    tokens = [t.strip() for t in s.replace(",", " ").split()]
    return [t for t in tokens if t and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", t)]


def _expand_multiperiod_leads_lags(
    raw_eqs: list[str],
    variables: list[str],
    steady_state: dict[str, float] | None = None,
    guess: dict[str, float] | None = None,
) -> tuple[list[str], list[str], dict[str, float] | None, dict[str, float] | None]:
    """Expand leads and lags with |offset| >= 2 into first-order auxiliary variables."""
    lag_leads_needed = set()
    for eq in raw_eqs:
        for v in variables:
            matches = re.findall(rf"\b{v}\s*\(\s*([+-]?\d+)\s*\)", eq)
            for m in matches:
                offset = int(m)
                if abs(offset) >= 2:
                    lag_leads_needed.add((v, offset))

    if not lag_leads_needed:
        return raw_eqs, list(variables), steady_state, guess

    aux_vars: list[str] = []
    aux_eqs: list[str] = []
    replacements: dict[str, str] = {}

    for v, offset in sorted(lag_leads_needed, key=lambda x: (x[0], abs(x[1]))):
        if offset <= -2:
            k = abs(offset)
            for step in range(1, k):
                aux_name = f"AUX_LAG_{v}_{step}"
                if aux_name not in aux_vars:
                    aux_vars.append(aux_name)
                    prev = f"{v}(-1)" if step == 1 else f"AUX_LAG_{v}_{step-1}(-1)"
                    aux_eqs.append(f"{aux_name} = {prev}")
            replacements[rf"\b{v}\s*\(\s*{offset}\s*\)"] = f"AUX_LAG_{v}_{k-1}(-1)"
        elif offset >= 2:
            k = offset
            for step in range(1, k):
                aux_name = f"AUX_LEAD_{v}_{step}"
                if aux_name not in aux_vars:
                    aux_vars.append(aux_name)
                    prev = f"{v}(+1)" if step == 1 else f"AUX_LEAD_{v}_{step-1}(+1)"
                    aux_eqs.append(f"{aux_name} = {prev}")
            replacements[rf"\b{v}\s*\(\s*\+?{offset}\s*\)"] = f"AUX_LEAD_{v}_{k-1}(+1)"

    new_eqs = []
    for eq in raw_eqs:
        mod_eq = eq
        for pat, rep in replacements.items():
            mod_eq = re.sub(pat, rep, mod_eq)
        new_eqs.append(mod_eq)

    updated_eqs = new_eqs + aux_eqs
    updated_vars = list(variables) + aux_vars

    updated_ss = dict(steady_state) if steady_state is not None else None
    if updated_ss is not None:
        for aux in aux_vars:
            root_var = aux.split("_")[2]
            updated_ss[aux] = updated_ss.get(root_var, 0.0)

    updated_guess = dict(guess) if guess is not None else None
    if updated_guess is not None:
        for aux in aux_vars:
            root_var = aux.split("_")[2]
            updated_guess[aux] = updated_guess.get(root_var, 1.0)

    return updated_eqs, updated_vars, updated_ss, updated_guess


def parse_mod(mod_text: str, base_dir: Path | str | None = None) -> dict:
    """Parse a Dynare .mod file string into structured declarations and Python equations.

    Parameters
    ----------
    mod_text : str
        The raw text of a Dynare .mod file.
    base_dir : Path or str, optional
        Base directory for macro processor file inclusion resolution.

    Returns
    -------
    dict
        Parsed dictionary with keys:
        - ``variables``: list of endogenous variable names
        - ``shocks``: list of exogenous innovation names
        - ``params``: dict of parameter name -> float value
        - ``predetermined_variables``: list of predetermined variable names (if declared)
        - ``guess``: dict of variable name -> initial steady-state guess
        - ``steady_state``: dict of variable name -> exact steady-state value (if declared)
        - ``equations``: compiled callable ``eqs(lead, curr, lag, shocks, params)``
        - ``shock_cov``: covariance matrix of structural innovations if declared
        - ``options``: dict of parsed stoch_simul options (e.g. order, pruning, irf)
        - ``varobs``: list of observable variables (if declared)
        - ``estimated_params``: :class:`~puremacro.dsge._estimated_params.EstimatedParams`
          parsed from the ``estimated_params;`` block, or ``None`` when the
          file has none
        - ``estimated_params_init``: ``{name: value}`` from
          ``estimated_params_init;``, or ``None``
        - ``estimated_params_bounds``: ``{name: (lb, ub)}`` from
          ``estimated_params_bounds;``, or ``None``

    Raises
    ------
    DynareFeatureError
        The file uses an unsupported Dynare macro directive.
    DynareMacroError
        The file has a macro preprocessor syntax or evaluation error.
    ValueError
        The file has no ``var``/``varexo`` declaration or no ``model;`` block.
    """
    # 0. Dynare macro processor pre-pass
    if _MACRO_DIRECTIVE.search(mod_text) or _MACRO_INTERPOLATION.search(mod_text):
        mod_text = preprocess_macro(mod_text, base_dir=base_dir)

    clean_text = _remove_comments(mod_text)
    _check_no_macro_directives(clean_text)

    # 1. First attempt AST DAG parsing via puremacro.dsge._parser
    try:
        dag = parse_mod_to_dag(clean_text, base_dir=base_dir)
        if not dag.variables:
            raise ValueError("could not find 'var ...;' declaration in .mod content")
        if not dag.shocks:
            raise ValueError("could not find 'varexo ...;' declaration in .mod content")
        if not dag.equations:
            raise ValueError("could not find 'model; ... end;' block in .mod content")
        compiled = compile_derivatives(dag)
        res = dag.to_dynare_dict()
        res["_dag"] = dag
        res["_compiled"] = compiled
        res["is_linear"] = dag.is_linear or compiled.is_linear
        eq_callable = res["equations"]
        try:
            object.__setattr__(eq_callable, "_dag", dag)
            object.__setattr__(eq_callable, "_compiled", compiled)
        except Exception:
            pass
        return res
    except (ValueError, DynareMacroError, DynareFeatureError):
        raise
    except Exception:
        # Fall back to legacy regex parser
        pass

    # Remove known blocks to isolate top-level declarations and parameters
    block_pattern = r"\b(model|initval|steady_state_model|shocks|estimated_params|histval|endval|shock_groups)\b(?:\([^)]*\))?\s*;.*?\bend\s*;"
    non_block_text = re.sub(block_pattern, "", clean_text, flags=re.DOTALL)

    # 1. Parse var
    var_match = re.search(r"\bvar\b\s+([^;]+);", non_block_text)
    if not var_match:
        raise ValueError("could not find 'var ...;' declaration in .mod content")
    variables = _extract_ids(var_match.group(1))

    # 2. Parse varexo
    varexo_match = re.search(r"\bvarexo\b\s+([^;]+);", non_block_text)
    if not varexo_match:
        raise ValueError("could not find 'varexo ...;' declaration in .mod content")
    shocks = _extract_ids(varexo_match.group(1))

    # 2b. Parse varexo_det (deterministic exogenous variables)
    varexo_det_match = re.search(r"\bvarexo_det\b\s+([^;]+);", non_block_text)
    varexo_det = _extract_ids(varexo_det_match.group(1)) if varexo_det_match else []

    # 3. Parse predetermined_variables (if declared)
    predet_match = re.search(r"\bpredetermined_variables\b\s+([^;]+);", non_block_text)
    predetermined_variables = _extract_ids(predet_match.group(1)) if predet_match else []

    # 4. Parse varobs (if declared)
    varobs_match = re.search(r"\bvarobs\b\s+([^;]+);", non_block_text)
    varobs = _extract_ids(varobs_match.group(1)) if varobs_match else []


    # 5. Parse parameters and sequential definitions outside blocks
    params: dict[str, float] = {}
    param_decl_matches = re.findall(r"\bparameters\b\s+([^;]+);", non_block_text)
    declared_params = set()
    for decl in param_decl_matches:
        declared_params.update(_extract_ids(decl))

    eval_scope = {
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "np": np,
        "__builtins__": None,
    }

    for line in non_block_text.split(";"):
        line = line.strip()
        if "=" in line:
            parts = line.split("=", 1)
            pname = parts[0].strip()
            if pname in declared_params:
                rhs_expr = parts[1].strip().replace("^", "**")
                try:
                    val = float(eval(rhs_expr, eval_scope, params))
                    params[pname] = val
                    eval_scope[pname] = val
                except Exception:
                    pass

    # 6. Parse model block and local parameters (#name = expr;)
    model_match = re.search(r"\bmodel\s*(?:\([^)]*\))?\s*;\s*(.*?)\bend\s*;", clean_text, re.DOTALL)
    if not model_match:
        raise ValueError("could not find 'model; ... end;' block in .mod content")

    raw_lines = model_match.group(1).split(";")
    clean_eqs = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        # Strip equation tags e.g. [name='...']
        line = re.sub(r"\[.*?\]", "", line).strip()
        if not line:
            continue
        if line.startswith("#"):
            m_local = re.match(r"#\s*([A-Za-z0-9_]+)\s*=\s*(.*)", line)
            if m_local:
                loc_name = m_local.group(1).strip()
                loc_expr = m_local.group(2).strip().replace("^", "**")
                try:
                    val = float(eval(loc_expr, eval_scope, params))
                    params[loc_name] = val
                    eval_scope[loc_name] = val
                except Exception:
                    pass
        else:
            clean_eqs.append(line)

    def _eval_assignment_block(body: str, block_name: str) -> dict[str, float]:
        """Evaluate ``name = expr;`` lines in order, keeping temporaries in scope.

        Dynare lets ``steady_state_model`` / ``initval`` blocks define helper
        quantities (``rk = 1/beta - 1 + delta;``) that later lines use;
        those must be evaluated, not discarded. An assignment that cannot
        be evaluated is a hard error naming the line, not a silent zero.
        """
        local_scope: dict[str, float] = dict(params)
        for raw in body.split(";"):
            stmt = raw.strip()
            if not stmt or "=" not in stmt:
                continue
            name, expr = stmt.split("=", 1)
            name = name.strip()
            expr = expr.strip().replace("^", "**")
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                continue
            try:
                local_scope[name] = float(eval(expr, eval_scope, local_scope))
            except Exception as exc:
                raise ValueError(
                    f"could not evaluate '{name} = {expr};' in the {block_name} block: "
                    f"{type(exc).__name__}: {exc}"
                ) from None
        return local_scope

    # 7. Parse initval block
    guess_init: dict[str, float] = {}
    initval_match = re.search(r"\binitval\s*;\s*(.*?)\bend\s*;", clean_text, re.DOTALL)
    if initval_match:
        init_scope = _eval_assignment_block(initval_match.group(1), "initval")
        guess_init = {v: init_scope[v] for v in variables if v in init_scope}

    # 8. Parse steady_state_model block (if present)
    steady_state: dict[str, float] | None = None
    ss_match = re.search(r"\bsteady_state_model\s*;\s*(.*?)\bend\s*;", clean_text, re.DOTALL)
    if ss_match:
        ss_scope = _eval_assignment_block(ss_match.group(1), "steady_state_model")
        steady_state = {v: ss_scope.get(v, 0.0) for v in variables}
        for v in variables:
            if v in ss_scope:
                eval_scope[v] = ss_scope[v]

    # 8b. Parse histval block (if present)
    histval: dict[str, float] = {}
    histval_match = re.search(r"\bhistval\s*;\s*(.*?)\bend\s*;", clean_text, re.DOTALL)
    if histval_match:
        hist_scope = _eval_assignment_block(histval_match.group(1), "histval")
        histval = {v: hist_scope[v] for v in hist_scope}

    # 8c. Parse endval block (if present)
    endval: dict[str, float] = {}
    endval_match = re.search(r"\bendval\s*;\s*(.*?)\bend\s*;", clean_text, re.DOTALL)
    if endval_match:
        end_scope = _eval_assignment_block(endval_match.group(1), "endval")
        endval = {v: end_scope[v] for v in end_scope}

    # 9. Parse shocks block.

    #    Dynare initialises M_.Sigma_e to zeros: an exogenous variable that the
    #    shocks; block never mentions has variance 0 and is inert.  Only when
    #    the .mod file has no shocks; block at all do we fall back to the unit
    #    covariance (and then `shock_cov` is not returned anyway).
    shock_cov = np.eye(len(shocks))
    shocks_match = re.search(r"\bshocks\s*;\s*(.*?)\bend\s*;", clean_text, re.DOTALL)
    has_shocks_block = False
    variance_declared: set[str] = set()
    if shocks_match:
        has_shocks_block = True
        shock_cov = np.zeros((len(shocks), len(shocks)))
        current_shock = None
        for stmt in shocks_match.group(1).split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            # Case 1: corr s1, s2 = val;
            corr_m = re.search(r"\bcorr\s+([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*=\s*(.*)", stmt)
            if corr_m:
                s1, s2, val_str = corr_m.group(1), corr_m.group(2), corr_m.group(3).replace("^", "**")
                try:
                    c_val = float(eval(val_str, eval_scope, params))
                    i1, i2 = shocks.index(s1), shocks.index(s2)
                    std1 = np.sqrt(max(0.0, shock_cov[i1, i1]))
                    std2 = np.sqrt(max(0.0, shock_cov[i2, i2]))
                    cov_val = c_val * std1 * std2
                    shock_cov[i1, i2] = cov_val
                    shock_cov[i2, i1] = cov_val
                except Exception:
                    pass
                continue

            # Case 2: var s1, s2 = val; (covariance)
            cov_m = re.search(r"\bvar\s+([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*=\s*(.*)", stmt)
            if cov_m:
                s1, s2, val_str = cov_m.group(1), cov_m.group(2), cov_m.group(3).replace("^", "**")
                try:
                    c_val = float(eval(val_str, eval_scope, params))
                    i1, i2 = shocks.index(s1), shocks.index(s2)
                    shock_cov[i1, i2] = c_val
                    shock_cov[i2, i1] = c_val
                except Exception:
                    pass
                continue

            # Case 3: var s = val;
            var_single = re.search(r"\bvar\s+([A-Za-z0-9_]+)\s*=\s*(.*)", stmt)
            if var_single:
                sname, val_str = var_single.group(1), var_single.group(2).replace("^", "**")
                if sname in shocks:
                    try:
                        s_val = float(eval(val_str, eval_scope, params))
                        s_idx = shocks.index(sname)
                        shock_cov[s_idx, s_idx] = s_val
                        variance_declared.add(sname)
                    except Exception:
                        pass
                continue

            # Case 4: var s, stderr val; or var s; stderr val;
            stderr_inline = re.search(r"\bvar\s+([A-Za-z0-9_]+)\s*(?:;|,)?\s*stderr\s+(.*)", stmt)
            if stderr_inline:
                sname = stderr_inline.group(1).strip()
                val_str = stderr_inline.group(2).strip().replace("^", "**")
                if sname in shocks:
                    try:
                        s_val = float(eval(val_str, eval_scope, params))
                        s_idx = shocks.index(sname)
                        shock_cov[s_idx, s_idx] = s_val**2
                        variance_declared.add(sname)
                    except Exception:
                        pass
                continue

            # Case 5: var s; (sets current shock for subsequent stderr statement)
            var_stmt = re.search(r"\bvar\s+([A-Za-z0-9_]+)$", stmt)
            if var_stmt:
                current_shock = var_stmt.group(1)
                continue

            # Case 6: stderr val; (following previous var s;)
            stderr_stmt = re.search(r"\bstderr\s+(.*)", stmt)
            if stderr_stmt and current_shock:
                val_str = stderr_stmt.group(1).replace("^", "**")
                if current_shock in shocks:
                    try:
                        s_val = float(eval(val_str, eval_scope, params))
                        s_idx = shocks.index(current_shock)
                        shock_cov[s_idx, s_idx] = s_val**2
                        variance_declared.add(current_shock)
                    except Exception:
                        pass
                current_shock = None
                continue

        undeclared = [sh for sh in shocks if sh not in variance_declared]
        if undeclared:
            warnings.warn(
                f"parse_mod: the shocks; block declares no variance for "
                f"{undeclared} — following Dynare (M_.Sigma_e starts at zero), "
                "their innovation variance is 0, so they are inert in the "
                "moments, the IRFs and the second-order risk correction ghs2. "
                "Add 'var <name>; stderr <value>;' to the shocks; block if that "
                "is not what you meant.",
                UserWarning,
                stacklevel=2,
            )

    # 10. Parse stoch_simul options
    options: dict[str, Any] = {}
    # Dynare allows a variable list after the option parentheses:
    # ``stoch_simul(order=2, irf=20) y c k;`` — the options still count.
    stoch_match = re.search(r"\bstoch_simul\s*(?:\(([^)]*)\))?[^;]*;", clean_text)
    if stoch_match:
        opts_str = stoch_match.group(1) or ""
        ord_m = re.search(r"\border\s*=\s*(\d+)", opts_str)
        if ord_m:
            options["order"] = int(ord_m.group(1))
        if "pruning" in opts_str:
            options["pruning"] = True
        irf_m = re.search(r"\birf\s*=\s*(\d+)", opts_str)
        if irf_m:
            options["irf"] = int(irf_m.group(1))
        per_m = re.search(r"\bperiods\s*=\s*(\d+)", opts_str)
        if per_m:
            options["periods"] = int(per_m.group(1))

    # 10b. Parse estimated_params / _init / _bounds.
    #
    # The blocks are stripped from `non_block_text` above so they cannot
    # pollute the top-level parameter scan; they are read here from
    # `clean_text` instead. `\b...\b` keeps `estimated_params` from matching
    # `estimated_params_init`.
    _has = lambda blk: re.search(  # noqa: E731
        rf"\b{blk}\s*(?:\([^)]*\))?\s*;", clean_text) is not None
    estimated_params = parse_estimated_params(
        clean_text, shocks=shocks, varobs=varobs,
        params=sorted(declared_params), variables=variables,
    ) if _has("estimated_params") else None
    estimated_params_init = parse_estimated_params_init(
        clean_text, shocks=shocks, varobs=varobs,
    ) if _has("estimated_params_init") else None
    estimated_params_bounds = parse_estimated_params_bounds(
        clean_text, shocks=shocks, varobs=varobs,
    ) if _has("estimated_params_bounds") else None

    # 11. Multi-period lead and lag expansion
    clean_eqs, variables, steady_state, guess = _expand_multiperiod_leads_lags(
        clean_eqs, variables, steady_state=steady_state, guess=guess_init if guess_init else None
    )

    # 12. Compile equations to Python callable
    def transform_expr(expr: str) -> str:
        s = expr.replace("^", "**")
        s = re.sub(r"\blog\(", "np.log(", s)
        s = re.sub(r"\bexp\(", "np.exp(", s)
        s = re.sub(r"\bsqrt\(", "np.sqrt(", s)
        s = re.sub(r"\bmax\(", "np.maximum(", s)
        s = re.sub(r"\bmin\(", "np.minimum(", s)

        # Sort variables by decreasing length to avoid partial substring collisions
        sorted_vars = sorted(variables, key=len, reverse=True)
        for v in sorted_vars:
            s = re.sub(rf"\b{v}\s*\(\s*\+?\s*1\s*\)", f"lead.{v}", s)
            s = re.sub(rf"\b{v}\s*\(\s*-1\s*\)", f"lag.{v}", s)
        for v in sorted_vars:
            s = re.sub(rf"(?<!lead\.)(?<!lag\.)\b{v}\b", f"curr.{v}", s)

        for sh in shocks:
            s = re.sub(rf"\b{sh}\b", f"shocks.{sh}", s)
        for p in params:
            s = re.sub(rf"\b{p}\b", f"params.{p}", s)
        return s

    py_exprs = []
    for eq in clean_eqs:
        if "=" in eq:
            lhs, rhs = eq.split("=", 1)
            t_lhs = transform_expr(lhs.strip())
            t_rhs = transform_expr(rhs.strip())
            py_exprs.append(f"({t_lhs}) - ({t_rhs})")
        else:
            py_exprs.append(transform_expr(eq.strip()))

    code_body = "def _generated_equations(lead, curr, lag, shocks, params):\n    return [\n"
    for pe in py_exprs:
        code_body += f"        {pe},\n"
    code_body += "    ]\n"

    scope = {"np": np}
    exec(code_body, scope)
    eq_callable = scope["_generated_equations"]

    return {
        "variables": variables,
        "shocks": shocks,
        "params": params,
        "predetermined_variables": predetermined_variables if predetermined_variables else None,
        "guess": guess if guess else None,
        "steady_state": steady_state if steady_state else None,
        "equations": eq_callable,
        "shock_cov": shock_cov if has_shocks_block else None,
        "options": options,
        "varobs": varobs if varobs else None,
        "estimated_params": estimated_params,
        "estimated_params_init": estimated_params_init,
        "estimated_params_bounds": estimated_params_bounds,
        "varexo_det": varexo_det,
        "det_shocks": [],
        "histval": histval,
        "endval": endval,
    }



def load_mod(
    path_or_text: str | Path,
    *,
    params: Mapping[str, float] | None = None,
    steady_state: Mapping[str, float] | Sequence[float] | None = None,
    guess: Mapping[str, float] | Sequence[float] | None = None,
    solve_algo: str = "block",
    homotopy: Mapping[str, tuple[float, float]] | None = None,
    homotopy_steps: int = 10,
    states: Sequence[str] | None = None,
    order: int | None = None,
    shock_cov: np.ndarray | None = None,
    tol: float = 1e-8,
    method: str = "complex",
    verify_derivatives: bool = True,
    strict: bool = True,
    qz_criterium: float = 1.0 + 1e-8,
) -> LinearModel | PrunedDSGESolution:
    """Load and solve a Dynare .mod file directly in puremacro.

    Parameters
    ----------
    path_or_text : str or Path
        Either a file path to a .mod file, or a raw string containing the
        .mod contents. A :class:`~pathlib.Path`, or a string without any
        ``;`` / newline (which cannot be .mod source), is treated as a path
        and must exist.
    params : Mapping[str, float], optional
        Optional parameter overrides.
    steady_state : Mapping[str, float] | Sequence[float], optional
        Optional exact steady state override.
    guess : Mapping[str, float] | Sequence[float], optional
        Optional initial guess override.
    states : Sequence[str], optional
        Optional explicit states override. If None, automatically detected
        or taken from ``predetermined_variables`` if declared.
    order : {1, 2}, optional
        Approximation order:
        - 1: First-order linear approximation (returns LinearModel).
        - 2: Second-order SGU (2004) perturbation with Kim et al. (2008) pruning
             (returns PrunedDSGESolution).
        If None, uses order specified in stoch_simul block if present, else 1.
    shock_cov : np.ndarray, optional
        Covariance matrix of innovations. Defaults to the ``shocks;`` block
        when the file has one, and to the identity matrix when it has none.
        Inside a ``shocks;`` block the Dynare convention applies: a ``varexo``
        the block never gives a ``stderr``/``var`` gets variance **0**, not 1,
        and :func:`parse_mod` warns naming it.
    tol : float, default 1e-8
        Steady-state solver and verification tolerance.
    method : {'complex', 'central'}, default 'complex'
        Differentiation method for Jacobians (and Hessians at order 2).
    verify_derivatives : bool, default True
        Cross-check complex-step Jacobians against finite differences and
        raise :class:`ModelError` on disagreement (non-analytic equations).
    strict : bool, default True
        Raise :class:`~puremacro.dsge.klein.BlanchardKahnError` when the
        Blanchard-Kahn condition fails instead of returning zero decision
        rules.

    Returns
    -------
    LinearModel | PrunedDSGESolution
        Solved model equipped with `.decision_rules()`, `.theoretical_moments()`,
        and `.irf()` (if order=1), or `.simulate()`, `.girf()`,
        `.stochastic_steady_state()`, and `.decision_rules()` (if order=2).

    Raises
    ------
    FileNotFoundError
        ``path_or_text`` looks like a path but no such file exists.
    BlanchardKahnError
        No unique stable solution (when ``strict``).
    """
    text_str = str(path_or_text)
    looks_like_path = isinstance(path_or_text, Path) or (
        "\n" not in text_str and ";" not in text_str
    )
    if looks_like_path:
        p = Path(path_or_text)
        if not p.is_file():
            raise FileNotFoundError(
                f"no .mod file at {p!s} (resolved from {Path.cwd()!s}); pass a path "
                "to an existing file or the .mod source text itself"
            )
        base_dir = p.parent
        text = p.read_text(encoding="utf-8")
    else:
        base_dir = Path.cwd()
        text = text_str

    parsed = parse_mod(text, base_dir=base_dir)

    # Merge parameters
    merged_params = dict(parsed["params"])
    if params:
        merged_params.update(params)

    # Merge steady state
    final_ss = steady_state if steady_state is not None else parsed.get("steady_state")

    # Merge guess
    final_guess = guess or parsed.get("guess")
    if final_ss is None and final_guess is None:
        final_guess = {v: 0.0 for v in parsed["variables"]}

    # Predetermined variables fallback
    final_states = states if states is not None else parsed.get("predetermined_variables")

    # Determine effective order
    if order is None:
        eff_order = parsed.get("options", {}).get("order", 1)
    else:
        eff_order = int(order)

    # Determine effective shock_cov
    eff_shock_cov = shock_cov if shock_cov is not None else parsed.get("shock_cov")

    model = build_dynare(
        parsed["equations"],
        variables=parsed["variables"],
        shocks=parsed["shocks"],
        params=merged_params,
        steady_state=final_ss,
        guess=final_guess,
        solve_algo=solve_algo,
        homotopy=homotopy,
        homotopy_steps=homotopy_steps,
        states=final_states,
        order=eff_order,
        shock_cov=eff_shock_cov,
        tol=tol,
        method=method,
        verify_derivatives=verify_derivatives,
        strict=strict,
        qz_criterium=qz_criterium,
    )

    # Carry the file's own declarations onto the solved model so
    # ``model.estimate(data)`` needs no second copy of them. LinearModel is a
    # frozen dataclass, so this is ``replace``, never assignment.
    dag = parsed.get("_dag", getattr(parsed.get("equations"), "_dag", None))
    compiled = parsed.get("_compiled", getattr(parsed.get("equations"), "_compiled", None))
    is_linear = parsed.get("is_linear", False)

    if isinstance(model, LinearModel):
        model = dataclasses.replace(
            model,
            _varobs=tuple(parsed["varobs"]) if parsed["varobs"] else None,
            _estimated_params=parsed["estimated_params"],
            _mod_options=dict(parsed["options"]) if parsed["options"] else None,
        )
        if dag is not None:
            object.__setattr__(model, "_dag", dag)
        if compiled is not None:
            object.__setattr__(model, "_compiled", compiled)
        object.__setattr__(model, "_is_linear", is_linear)
        if parsed.get("histval"):
            object.__setattr__(model, "_histval", parsed["histval"])
        if parsed.get("endval"):
            object.__setattr__(model, "_endval", parsed["endval"])
        if parsed.get("varexo_det"):
            object.__setattr__(model, "_varexo_det", parsed["varexo_det"])
        if parsed.get("det_shocks"):
            object.__setattr__(model, "_det_shocks", parsed["det_shocks"])
        if parsed.get("shock_groups"):
            object.__setattr__(model, "_shock_groups", parsed["shock_groups"])
        if parsed.get("named_shock_groups"):
            object.__setattr__(model, "_named_shock_groups", parsed["named_shock_groups"])
    elif isinstance(model, (PrunedDSGESolution, Order3PrunedSolution)):
        if dag is not None:
            object.__setattr__(model, "_dag", dag)
        if compiled is not None:
            object.__setattr__(model, "_compiled", compiled)
        object.__setattr__(model, "_is_linear", is_linear)
        if parsed.get("histval"):
            object.__setattr__(model, "_histval", parsed["histval"])
        if parsed.get("endval"):
            object.__setattr__(model, "_endval", parsed["endval"])
        if parsed.get("varexo_det"):
            object.__setattr__(model, "_varexo_det", parsed["varexo_det"])
        if parsed.get("det_shocks"):
            object.__setattr__(model, "_det_shocks", parsed["det_shocks"])


    return model


# Backwards compatibility alias
load_dynare_mod = load_mod


# Attach model_info and write_latex_dynamic_model to LinearModel
def _linear_model_model_info(self) -> Any:
    dag = getattr(self, "_dag", self)
    return model_info(dag)


def _linear_model_write_latex(self, filepath: str | Path | None = None, write_equation_tags: bool = True) -> str:
    dag = getattr(self, "_dag", self)
    return write_latex_dynamic_model(dag, filepath=filepath, write_equation_tags=write_equation_tags)


def _linear_model_solve_third_order(self, shock_cov: np.ndarray | None = None) -> Order3PrunedSolution:
    """Solve third-order perturbation approximation with pruning (Andreasen et al. 2018)."""
    if self._dynare_equations is None:
        raise ModelError(
            "solve_third_order requires the model to be built with build_dynare "
            "or load_mod (using canonical lead-lag equations)."
        )
    cov = shock_cov if shock_cov is not None else self._shock_cov
    return solve_dynare_3rd_order(
        self._dynare_equations,
        variables=self.variables,
        shocks=self.shocks,
        params=self._params,
        steady_state=self.steady_state.to_dict(),
        states=self.states,
        shock_cov=cov,
        method=self.method,
    )


LinearModel.model_info = _linear_model_model_info  # type: ignore[attr-defined]
LinearModel.write_latex_dynamic_model = _linear_model_write_latex  # type: ignore[attr-defined]
LinearModel.solve_third_order = _linear_model_solve_third_order  # type: ignore[attr-defined]

_orig_solve = LinearModel.solve


def _linear_model_solve(
    self,
    order: int = 1,
    *,
    shock_cov: np.ndarray | None = None,
    qz_criterium: float | None = None,
):
    if order == 3:
        return self.solve_third_order(shock_cov=shock_cov)
    return _orig_solve(self, order=order, shock_cov=shock_cov, qz_criterium=qz_criterium)


LinearModel.solve = _linear_model_solve  # type: ignore[assignment]
LinearModel.solve_perturbation = _linear_model_solve  # type: ignore[assignment]

_orig_stoch_simul = LinearModel.stoch_simul


def _linear_model_stoch_simul(self, order: int = 1, **kwargs):
    if order == 3:
        sol3 = self.solve_third_order(shock_cov=self._shock_covariance(kwargs.get("sigma", None)))
        irf = kwargs.get("irf", 40)
        periods = kwargs.get("periods", 0)
        seed = kwargs.get("seed", 0)
        burn = kwargs.get("burn", 100)
        lags = kwargs.get("lags", 5)
        return sol3.stoch_simul(order=3, irf=irf, periods=periods, sigma=1.0, seed=seed, burn=burn, lags=lags)
    return _orig_stoch_simul(self, order=order, **kwargs)


LinearModel.stoch_simul = _linear_model_stoch_simul  # type: ignore[assignment]


def _linear_model_perfect_foresight(
    self,
    periods: int = 100,
    shocks: Any | None = None,
    y_init: np.ndarray | None = None,
    y_end: np.ndarray | None = None,
    mcp: bool = False,
    mcp_bounds: Mapping[str | int, tuple[float | None, float | None]] | None = None,
    **kwargs,
):
    from puremacro.dsge.perfect_foresight import solve_perfect_foresight

    return solve_perfect_foresight(
        self,
        periods=periods,
        shocks=shocks,
        y_init=y_init,
        y_end=y_end,
        mcp=mcp,
        mcp_bounds=mcp_bounds,
        **kwargs,
    )


def _linear_model_simulate_surprise_shocks(
    self,
    surprise_shocks: np.ndarray | Mapping[str, Sequence[float]],
    horizon: int = 100,
    **kwargs,
):
    from puremacro.dsge.perfect_foresight import simulate_surprise_shocks

    return simulate_surprise_shocks(
        self,
        surprise_shocks=surprise_shocks,
        horizon=horizon,
        **kwargs,
    )


LinearModel.perfect_foresight = _linear_model_perfect_foresight  # type: ignore[attr-defined]
LinearModel.solve_perfect_foresight = _linear_model_perfect_foresight  # type: ignore[attr-defined]
LinearModel.simulate_surprise_shocks = _linear_model_simulate_surprise_shocks  # type: ignore[attr-defined]

from puremacro.dsge.perfect_foresight import (
    MCPResult,
    find_steady_state,
    simulate_surprise_shocks,
    solve_perfect_foresight,
)
from puremacro.dsge.pruning import PrunedSimulationResult

PrunedSimulationResult.__len__ = lambda self: len(self.states)  # type: ignore[attr-defined]
PrunedSimulationResult.to_numpy = lambda self, *args, **kwargs: self.to_frame().to_numpy(*args, **kwargs)  # type: ignore[attr-defined]
PrunedSimulationResult.columns = property(lambda self: self.to_frame().columns)  # type: ignore[assignment]
PrunedSimulationResult.__getitem__ = lambda self, key: self.to_frame()[key]  # type: ignore[attr-defined]
PrunedSimulationResult.__getattr__ = lambda self, name: getattr(self.to_frame(), name)  # type: ignore[attr-defined]


__all__ = [
    "build_dynare",
    "parse_mod",
    "load_mod",
    "load_dynare_mod",
    "solve_dynare_2nd_order",
    "solve_dynare_3rd_order",
    "DynareFeatureError",
    "DynareMacroError",
    "DynareParseError",
    "model_info",
    "write_latex_dynamic_model",
    "detect_linear_model",
    "find_steady_state",
    "MCPResult",
    "simulate_surprise_shocks",
    "solve_perfect_foresight",
    "SMCSampler",
    "SMCResult",
    "bootstrap_particle_filter",
    "smc_estimate",
    "ramsey_model",
    "RamseyResult",
    "detrend_bgp",
    "detrend_model",
]

import sys

if "puremacro.dsge" in sys.modules:
    _dsge_mod = sys.modules["puremacro.dsge"]
    setattr(_dsge_mod, "model_info", model_info)
    setattr(_dsge_mod, "write_latex_dynamic_model", write_latex_dynamic_model)
    setattr(_dsge_mod, "detect_linear_model", detect_linear_model)
    setattr(_dsge_mod, "DynareMacroError", DynareMacroError)
    setattr(_dsge_mod, "DynareParseError", DynareParseError)
    setattr(_dsge_mod, "find_steady_state", find_steady_state)
    setattr(_dsge_mod, "MCPResult", MCPResult)
    setattr(_dsge_mod, "simulate_surprise_shocks", simulate_surprise_shocks)
    setattr(_dsge_mod, "solve_perfect_foresight", solve_perfect_foresight)
    setattr(_dsge_mod, "SMCSampler", SMCSampler)
    setattr(_dsge_mod, "SMCResult", SMCResult)
    setattr(_dsge_mod, "bootstrap_particle_filter", bootstrap_particle_filter)
    setattr(_dsge_mod, "smc_estimate", smc_estimate)
    setattr(_dsge_mod, "ramsey_model", ramsey_model)
    setattr(_dsge_mod, "RamseyResult", RamseyResult)
    setattr(_dsge_mod, "detrend_bgp", detrend_bgp)
    setattr(_dsge_mod, "detrend_model", detrend_model)


