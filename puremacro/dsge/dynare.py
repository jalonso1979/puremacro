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

import re
from pathlib import Path
from typing import Callable, Mapping, Sequence

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
)
from puremacro.dsge.klein import klein_solve
from puremacro.dsge.pruning import PrunedDSGESolution


def _remove_comments(text: str) -> str:
    """Strip C-style (//, /* */) and MATLAB-style (%) comments."""
    # Block comments /* ... */
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Line comments // and %
    text = re.sub(r"(//|%).*$", "", text, flags=re.MULTILINE)
    return text


def build_dynare(
    equations: Callable,
    *,
    variables: Sequence[str],
    shocks: Sequence[str],
    params: Mapping[str, float] | None = None,
    steady_state: Mapping[str, float] | Sequence[float] | None = None,
    guess: Mapping[str, float] | Sequence[float] | None = None,
    states: Sequence[str] | None = None,
    order: int = 1,
    shock_cov: np.ndarray | None = None,
    tol: float = 1e-8,
    method: str = "complex",
    verify_derivatives: bool = True,
) -> LinearModel | PrunedDSGESolution:
    """Solve a DSGE model written in Dynare canonical lead-lag form.

    Parameters
    ----------
    equations : Callable
        Function of five arguments: ``eqs(lead, curr, lag, shocks, params)``.
        Each argument supports named attribute access (e.g. ``curr.c``, ``lead.c``,
        ``lag.k``, ``shocks.eps``, ``params.alpha``).
    variables : Sequence[str]
        Names of all endogenous variables in model order.
    shocks : Sequence[str]
        Names of structural innovations.
    params : Mapping[str, float], optional
        Parameter values.
    steady_state : Mapping[str, float] | Sequence[float], optional
        Exact steady-state values. Verified against the equations.
    guess : Mapping[str, float] | Sequence[float], optional
        Initial guess for numerical steady-state solver.
    states : Sequence[str], optional
        Predetermined state variables. If None (default), **automatically
        detected** from the Jacobian columns with respect to ``lag``.
    tol : float, default 1e-8
        Numerical tolerance for steady-state residual check.
    method : {'complex', 'central'}, default 'complex'
        Differentiation method for Jacobians.
    verify_derivatives : bool, default True
        Whether to check complex-step Jacobians against finite differences.

    Returns
    -------
    LinearModel
        Solved model equipped with `.decision_rules()`, `.theoretical_moments()`,
        `.fevd()`, `.irf()`, and `.simulate()`.
    """
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
            states=states,
            shock_cov=shock_cov,
            tol=tol,
        )
    elif order != 1:
        raise ValueError(f"unsupported perturbation order {order}; must be 1 or 2")

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
            ss_arr = np.array([float(steady_state[v]) for v in variables])
        else:
            ss_arr = np.asarray(steady_state, dtype=float)
        res_0 = ss_res(ss_arr)
        max_err = float(np.max(np.abs(res_0)))
        if max_err > tol:
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
        sol = scipy.optimize.root(ss_res, g_arr, method="hybr")
        if not sol.success or float(np.max(np.abs(sol.fun))) > tol:
            raise SteadyStateError(
                f"steady state solver did not converge from guess: {sol.message}"
            )
        ss_arr = sol.x

    ss_series = pd.Series(ss_arr, index=variables, name="steady_state")

    # 2. Complex-step Jacobians at steady state
    # Equations are E_t f(lead, curr, lag, shocks) = 0
    # A_+ = df/d(lead), A_0 = df/d(curr), A_- = df/d(lag), B_u = df/d(shocks)
    A_plus = np.zeros((n_vars, n_vars))
    A_0 = np.zeros((n_vars, n_vars))
    A_minus = np.zeros((n_vars, n_vars))
    B_u = np.zeros((n_vars, n_shocks))

    e_zero = np.zeros(n_shocks)

    if method == "complex":
        step = _CSTEP
        base_ss = np.asarray(ss_arr, dtype=complex)
        base_e = np.asarray(e_zero, dtype=complex)

        for j in range(n_vars):
            pert_p = base_ss.copy()
            pert_p[j] += 1j * step
            out_p = equations(
                _Vec(variables, pert_p),
                _Vec(variables, base_ss),
                _Vec(variables, base_ss),
                _Vec(shocks, base_e),
                par_vec,
            )
            A_plus[:, j] = np.asarray(out_p, dtype=complex).imag / step

            pert_0 = base_ss.copy()
            pert_0[j] += 1j * step
            out_0 = equations(
                _Vec(variables, base_ss),
                _Vec(variables, pert_0),
                _Vec(variables, base_ss),
                _Vec(shocks, base_e),
                par_vec,
            )
            A_0[:, j] = np.asarray(out_0, dtype=complex).imag / step

            pert_m = base_ss.copy()
            pert_m[j] += 1j * step
            out_m = equations(
                _Vec(variables, base_ss),
                _Vec(variables, base_ss),
                _Vec(variables, pert_m),
                _Vec(shocks, base_e),
                par_vec,
            )
            A_minus[:, j] = np.asarray(out_m, dtype=complex).imag / step

        for j in range(n_shocks):
            pert_e = base_e.copy()
            pert_e[j] += 1j * step
            out_e = equations(
                _Vec(variables, base_ss),
                _Vec(variables, base_ss),
                _Vec(variables, base_ss),
                _Vec(shocks, pert_e),
                par_vec,
            )
            B_u[:, j] = np.asarray(out_e, dtype=complex).imag / step
    else:
        # Central difference
        for j in range(n_vars):
            h_j = _FDSTEP * max(1.0, abs(ss_arr[j]))
            up_ss, dn_ss = ss_arr.copy(), ss_arr.copy()
            up_ss[j] += h_j
            dn_ss[j] -= h_j

            out_p_up = equations(
                _Vec(variables, up_ss), _Vec(variables, ss_arr), _Vec(variables, ss_arr), _Vec(shocks, e_zero), par_vec
            )
            out_p_dn = equations(
                _Vec(variables, dn_ss), _Vec(variables, ss_arr), _Vec(variables, ss_arr), _Vec(shocks, e_zero), par_vec
            )
            A_plus[:, j] = (np.asarray(out_p_up, dtype=float) - np.asarray(out_p_dn, dtype=float)) / (2.0 * h_j)

            out_0_up = equations(
                _Vec(variables, ss_arr), _Vec(variables, up_ss), _Vec(variables, ss_arr), _Vec(shocks, e_zero), par_vec
            )
            out_0_dn = equations(
                _Vec(variables, ss_arr), _Vec(variables, dn_ss), _Vec(variables, ss_arr), _Vec(shocks, e_zero), par_vec
            )
            A_0[:, j] = (np.asarray(out_0_up, dtype=float) - np.asarray(out_0_dn, dtype=float)) / (2.0 * h_j)

            out_m_up = equations(
                _Vec(variables, ss_arr), _Vec(variables, ss_arr), _Vec(variables, up_ss), _Vec(shocks, e_zero), par_vec
            )
            out_m_dn = equations(
                _Vec(variables, ss_arr), _Vec(variables, ss_arr), _Vec(variables, dn_ss), _Vec(shocks, e_zero), par_vec
            )
            A_minus[:, j] = (np.asarray(out_m_up, dtype=float) - np.asarray(out_m_dn, dtype=float)) / (2.0 * h_j)

        for j in range(n_shocks):
            h_e = _FDSTEP
            up_e, dn_e = e_zero.copy(), e_zero.copy()
            up_e[j] += h_e
            dn_e[j] -= h_e
            out_e_up = equations(
                _Vec(variables, ss_arr), _Vec(variables, ss_arr), _Vec(variables, ss_arr), _Vec(shocks, up_e), par_vec
            )
            out_e_dn = equations(
                _Vec(variables, ss_arr), _Vec(variables, ss_arr), _Vec(variables, ss_arr), _Vec(shocks, dn_e), par_vec
            )
            B_u[:, j] = (np.asarray(out_e_up, dtype=float) - np.asarray(out_e_dn, dtype=float)) / (2.0 * h_e)

    # 3. Automatic variable role classification
    if states is None:
        detected_states = []
        for j, v in enumerate(variables):
            col_norm = float(np.linalg.norm(A_minus[:, j]))
            if col_norm > 1e-10:
                detected_states.append(v)
        states_tuple = tuple(detected_states)
    else:
        states_tuple = tuple(states)

    controls_tuple = tuple(v for v in variables if v not in states_tuple)

    if len(states_tuple) == 0:
        raise ModelError(
            "no predetermined state variables detected: column norm of A_- is zero for all variables. "
            "Pass explicit states=[...] if this is an atypical model."
        )

    # 4. Partition and solve system via Klein QZ
    # Permute variables so states come first: [states, controls]
    perm_order = list(states_tuple) + list(controls_tuple)
    perm_idx = [variables.index(v) for v in perm_order]
    n_s = len(states_tuple)

    A_p_perm = A_plus[:, perm_idx]
    A_0_perm = A_0[:, perm_idx]
    A_m_perm = A_minus[:, perm_idx]

    # Klein system: A * E_t z_{t+1} = B * z_t + C * u_t
    # In lead-lag timing:
    # (A_0)_{:, :n_s} * s_{t+1} + (A_+)_{:, n_s:} * E_t c_{t+1} = -(A_-)_{:, :n_s} * s_t - (A_0)_{:, n_s:} * c_t - B_u * u_t
    A_klein = np.hstack([A_0_perm[:, :n_s], A_p_perm[:, n_s:]])
    B_klein = np.hstack([-A_m_perm[:, :n_s], -A_0_perm[:, n_s:]])
    C_klein = -B_u

    solution = klein_solve(A_klein, B_klein, n_pre=n_s, C=C_klein)

    # Unit classification
    units = {v: "log" if ss_series[v] > 0 else "level" for v in variables}

    res_norm = float(np.max(np.abs(ss_res(ss_arr))))

    return LinearModel(
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
    )


def solve_dynare_2nd_order(
    equations: Callable,
    *,
    variables: Sequence[str],
    shocks: Sequence[str],
    params: Mapping[str, float] | None = None,
    steady_state: Mapping[str, float] | Sequence[float] | None = None,
    guess: Mapping[str, float] | Sequence[float] | None = None,
    states: Sequence[str] | None = None,
    shock_cov: np.ndarray | None = None,
    tol: float = 1e-8,
) -> PrunedDSGESolution:
    """Solve second-order DSGE perturbation with pruning (Schmitt-Grohé & Uribe 2004, Kim et al. 2008).

    Solves for the quadratic policy matrices (H_xx, G_xx) and volatility risk
    corrections (H_σσ, G_σσ) using the generalized Sylvester system derived from
    Dynare's canonical dynamic representation:
        E_t [ f(y_{t+1}, y_t, y_{t-1}, u_t; θ) ] = 0

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
        Predetermined states. If None, auto-detected from columns of df/d(lag).
    shock_cov : np.ndarray, optional
        Covariance matrix of innovations Σ_u. Defaults to identity matrix I.
    tol : float, default 1e-8
        Tolerance for steady-state residual check.

    Returns
    -------
    PrunedDSGESolution
        Second-order pruned DSGE solution equipped with `.simulate()`, `.girf()`,
        and `.stochastic_steady_state()`.
    """
    # 1. Solve 1st-order model via Klein QZ
    m = build_dynare(
        equations,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
        guess=guess,
        states=states,
        order=1,
        tol=tol,
    )

    vars_list = list(m.variables)
    states_list = list(m.states)
    controls_list = list(m.controls)
    shocks_list = list(m.shocks)

    N = len(vars_list)
    n_x = len(states_list)
    n_y = len(controls_list)
    n_e = len(shocks_list)

    dr = m.decision_rules()
    g_x = dr.ghx.loc[vars_list, states_list].to_numpy()
    g_u = dr.ghu.loc[vars_list, shocks_list].to_numpy()

    # Selectors for states and controls
    P_s = np.zeros((n_x, N))
    for j, s in enumerate(states_list):
        P_s[j, vars_list.index(s)] = 1.0

    P_c = np.zeros((n_y, N))
    for j, c in enumerate(controls_list):
        P_c[j, vars_list.index(c)] = 1.0

    h_x = P_s @ g_x
    h_u = P_s @ g_u

    # 2. Compute second-order Hessian tensor of f at steady state
    ss_arr = m.steady_state.loc[vars_list].to_numpy()
    u0 = np.concatenate([ss_arr, ss_arr, ss_arr])
    K_vars = 3 * N

    par_dict = dict(params or {})
    par_vec = _Vec(list(par_dict.keys()), list(par_dict.values()), what="parameter")
    e0_vec = _Vec(shocks_list, np.zeros(n_e), what="shock")

    def eval_f(u_vec):
        lead = _Vec(vars_list, u_vec[0:N])
        curr = _Vec(vars_list, u_vec[N:2 * N])
        lag = _Vec(vars_list, u_vec[2 * N:3 * N])
        return np.asarray(equations(lead, curr, lag, e0_vec, par_vec))

    hc = _CSTEP
    def grad_f(u_vec):
        G_mat = np.zeros((N, K_vars))
        for q in range(K_vars):
            pert = np.asarray(u_vec, dtype=complex).copy()
            pert[q] += 1j * hc
            G_mat[:, q] = eval_f(pert).imag / hc
        return G_mat

    J0 = grad_f(u0)
    A_plus = J0[:, 0:N]
    A_0 = J0[:, N:2 * N]

    H_f = np.zeros((N, K_vars, K_vars))
    hd = 1e-5
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

    # 3. Form Sylvester system for g_xx
    I_states = np.zeros((N, n_x))
    for j, s in enumerate(states_list):
        I_states[vars_list.index(s), j] = 1.0

    M = np.vstack([g_x @ h_x, g_x, I_states])

    K_tensor = np.zeros((N, n_x**2))
    for i in range(N):
        quad_i = M.T @ H_f[i] @ M
        K_tensor[i] = quad_i.flatten()

    A_hat = A_0 + A_plus @ g_x @ P_s
    hx_kron = np.kron(h_x, h_x)
    sys_mat = np.kron(np.eye(n_x**2), A_hat) + np.kron(hx_kron.T, A_plus)
    rhs = -K_tensor.reshape(-1, order="F")

    try:
        vec_gxx = scipy.linalg.solve(sys_mat, rhs)
    except scipy.linalg.LinAlgError:
        vec_gxx = scipy.linalg.lstsq(sys_mat, rhs)[0]

    g_xx = vec_gxx.reshape((N, n_x**2), order="F")
    H_xx = P_s @ g_xx
    G_xx = P_c @ g_xx

    # 4. Volatility drift terms (H_sigmasigma, G_sigmasigma)
    if shock_cov is None:
        sigma_u = np.eye(n_e)
    else:
        sigma_u = np.asarray(shock_cov, dtype=float)

    W_vec = np.zeros(N)
    for i in range(N):
        W_vec[i] = np.trace(g_u.T @ H_f[i, 0:N, 0:N] @ g_u @ sigma_u)

    hu_cov = h_u @ sigma_u @ h_u.T
    rhs_sig = -(A_plus @ g_xx @ hu_cov.flatten() + W_vec)

    sys_sig = A_hat + A_plus
    try:
        g_ss = scipy.linalg.solve(sys_sig, rhs_sig)
    except scipy.linalg.LinAlgError:
        g_ss = scipy.linalg.lstsq(sys_sig, rhs_sig)[0]

    H_ss = P_s @ g_ss
    G_ss = P_c @ g_ss

    return PrunedDSGESolution(
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
    )


def parse_mod(mod_text: str) -> dict:
    """Parse a Dynare .mod file string into structured declarations and Python equations.

    Parameters
    ----------
    mod_text : str
        The raw text of a Dynare .mod file.

    Returns
    -------
    dict
        Parsed dictionary with keys:
        - ``variables``: list of endogenous variable names
        - ``shocks``: list of exogenous innovation names
        - ``params``: dict of parameter name -> float value
        - ``guess``: dict of variable name -> initial steady-state guess
        - ``steady_state``: dict of variable name -> exact steady-state value (if declared)
        - ``equations``: compiled callable ``eqs(lead, curr, lag, shocks, params)``
    """
    clean_text = _remove_comments(mod_text)

    # 1. Parse var
    var_match = re.search(r"\bvar\s+([^;]+);", clean_text)
    if not var_match:
        raise ValueError("could not find 'var ...;' declaration in .mod content")
    raw_vars = var_match.group(1).replace(",", " ").split()
    variables = [v.strip() for v in raw_vars if v.strip()]

    # 2. Parse varexo
    varexo_match = re.search(r"\bvarexo\s+([^;]+);", clean_text)
    if not varexo_match:
        raise ValueError("could not find 'varexo ...;' declaration in .mod content")
    raw_shocks = varexo_match.group(1).replace(",", " ").split()
    shocks = [s.strip() for s in raw_shocks if s.strip()]

    # 3. Parse parameters and values
    params: dict[str, float] = {}
    param_decl_matches = re.findall(r"\bparameters\s+([^;]+);", clean_text)
    declared_params = set()
    for decl in param_decl_matches:
        for p in decl.replace(",", " ").split():
            if p.strip():
                declared_params.add(p.strip())

    # Parameter assignments outside blocks
    for line in clean_text.split(";"):
        line = line.strip()
        if "=" in line and not any(k in line for k in ["model", "initval", "steady_state_model"]):
            parts = line.split("=")
            pname = parts[0].strip()
            if pname in declared_params:
                try:
                    # Safe arithmetic evaluation of numeric parameter expressions
                    val = float(eval(parts[1].strip(), {"__builtins__": None}, {}))
                    params[pname] = val
                except Exception:
                    pass

    # 4. Parse initval block
    guess: dict[str, float] = {}
    initval_match = re.search(r"\binitval\s*;\s*(.*?)\s*end\s*;", clean_text, re.DOTALL)
    if initval_match:
        for line in initval_match.group(1).split(";"):
            line = line.strip()
            if "=" in line:
                vname, vval = line.split("=")
                vname = vname.strip()
                if vname in variables:
                    try:
                        guess[vname] = float(eval(vval.strip(), {"__builtins__": None}, params))
                    except Exception:
                        pass

    # 5. Parse model block
    model_match = re.search(r"\bmodel\s*(?:\([^)]*\))?\s*;\s*(.*?)\s*end\s*;", clean_text, re.DOTALL)
    if not model_match:
        raise ValueError("could not find 'model; ... end;' block in .mod content")

    raw_eqs = [e.strip() for e in model_match.group(1).split(";") if e.strip()]

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
    for eq in raw_eqs:
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
        "guess": guess if guess else None,
        "equations": eq_callable,
    }


def load_mod(
    path_or_text: str | Path,
    *,
    params: Mapping[str, float] | None = None,
    steady_state: Mapping[str, float] | Sequence[float] | None = None,
    guess: Mapping[str, float] | Sequence[float] | None = None,
    states: Sequence[str] | None = None,
    order: int = 1,
    shock_cov: np.ndarray | None = None,
    tol: float = 1e-8,
    method: str = "complex",
) -> LinearModel | PrunedDSGESolution:
    """Load and solve a Dynare .mod file directly in puremacro.

    Parameters
    ----------
    path_or_text : str or Path
        Either a file path to a .mod file, or a raw string containing the .mod contents.
    params : Mapping[str, float], optional
        Optional parameter overrides.
    steady_state : Mapping[str, float] | Sequence[float], optional
        Optional exact steady state override.
    guess : Mapping[str, float] | Sequence[float], optional
        Optional initial guess override.
    states : Sequence[str], optional
        Optional explicit states override. If None, automatically detected.
    order : {1, 2}, default 1
        Approximation order:
        - 1: First-order linear approximation (returns LinearModel).
        - 2: Second-order SGU (2004) perturbation with Kim et al. (2008) pruning
             (returns PrunedDSGESolution).
    shock_cov : np.ndarray, optional
        Covariance matrix of innovations (default identity I).
    tol : float, default 1e-8
        Steady-state solver and verification tolerance.
    method : {'complex', 'central'}, default 'complex'
        Differentiation method for Jacobians.

    Returns
    -------
    LinearModel | PrunedDSGESolution
        Solved model equipped with `.decision_rules()`, `.theoretical_moments()`,
        and `.irf()` (if order=1), or `.simulate()`, `.girf()`, and
        `.stochastic_steady_state()` (if order=2).
    """
    text_str = str(path_or_text)
    if "\n" in text_str or ";" in text_str:
        text = text_str
    else:
        p = Path(path_or_text)
        if p.is_file():
            text = p.read_text(encoding="utf-8")
        else:
            text = text_str

    parsed = parse_mod(text)

    # Merge parameters
    merged_params = dict(parsed["params"])
    if params:
        merged_params.update(params)

    # Merge guess
    final_guess = guess or parsed.get("guess")

    return build_dynare(
        parsed["equations"],
        variables=parsed["variables"],
        shocks=parsed["shocks"],
        params=merged_params,
        steady_state=steady_state,
        guess=final_guess,
        states=states,
        order=order,
        shock_cov=shock_cov,
        tol=tol,
        method=method,
    )
