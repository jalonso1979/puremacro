"""Optimal Simple Rules (OSR) and Policy Regimes (Discretion & Commitment).

This module implements:
1. ``osr()``: Optimal Simple Rules minimizing a quadratic variance loss
   L(gamma) = sum_i w_i Var(y_i; gamma) subject to Blanchard-Kahn determinacy,
   with a continuous numerical penalty surface when determinacy fails.
2. ``discretionary_policy()``: Markov-perfect time-consistent discretionary
   policy equilibrium using Dennis (2007) policy function iteration.
3. ``lq_commitment()``: Linear-quadratic optimal commitment policy, augmenting
   the first-order system with forward-looking Lagrange multipliers and solving
   via Klein QZ. The solved law of motion is the timeless-perspective rule;
   impulse responses and the conditional loss start from the steady state with
   lambda_{-1} = 0, where the Ramsey plan chosen at t0 and the timeless rule
   coincide, and the unconditional loss averages over the stationary
   distribution of (z_t, lambda_t).

References
----------
Jensen, C., and McCallum, B. T. (2002). The non-optimality of proposed monetary
    policy rules under timeless-perspective commitment. Economics Letters,
    77(2), 163-168 (NBER Working Paper 8882).
Sauer, S. (2010). Discretion rather than rules? When is discretionary
    policymaking better than the timeless perspective? International Journal of
    Central Banking, 6(2), 1-29.
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
from puremacro.dsge._results import OSRResult, PolicyResult, DiscretionaryPolicyResult
from puremacro.dsge.build import LinearModel, ModelError

__all__ = ["osr", "discretionary_policy", "lq_commitment", "optimal_policy"]


# ==============================================================================
# Helper: Re-solving LinearModel with Perturbed Parameters
# ==============================================================================

def _solve_with_params(model: LinearModel, new_params: dict[str, float]) -> LinearModel:
    """Re-solve model with updated parameters, preserving declarations.

    ``build_dynare`` / ``load_mod`` models go through
    :func:`puremacro.dsge.dynare._resolve_dynare_model`: the steady state is
    re-evaluated (``steady_state_model``) or re-solved when the parameters
    move it, the declared shock covariance is kept, and the predetermined set
    includes every lagged variable, so a lag calibrated to 0 is not lost when
    a rule parameter makes it live. ``build`` models go through
    :func:`puremacro.dsge.build._resolve_build_model`, which keeps the
    model's linearisation. Both return ``strict=False`` solutions: an
    indeterminate rule is flagged in ``solution.eu`` for the caller's penalty.
    """
    if getattr(model, "_dynare_equations", None) is not None:
        from puremacro.dsge.dynare import _resolve_dynare_model

        m_new = _resolve_dynare_model(model, new_params, strict=False)
    elif getattr(model, "_equations", None) is not None:
        from puremacro.dsge.build import _resolve_build_model

        m_new = _resolve_build_model(model, new_params, strict=False)
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
        # `replace` keeps only dataclass fields; carry the attributes the
        # builders attach by hand.
        attached = {
            a: getattr(m_new, a)
            for a in ("_dag", "_compiled", "_is_linear", "_qz_criterium",
                      "_steady_state_info", "_linearize", "_resolve_states_cache",
                      "_allow_singular")
            if hasattr(m_new, a)
        }
        m_new = dataclasses.replace(m_new, **replacements)
        for a, v in attached.items():
            object.__setattr__(m_new, a, v)

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
    xatol: float = 1e-8,
    fatol: float | None = None,
    options: Mapping[str, Any] | None = None,
) -> OSRResult:
    """Optimize policy rule parameters against theoretical variance loss.

    Minimizes the quadratic variance loss:
        L(gamma) = sum_{i in targets} w_i * Var(y_i; gamma)
    subject to Blanchard-Kahn saddle-path determinacy. Var is the unconditional
    (stationary) variance of the first-order solution, in the model's units
    (squared deviations from steady state; the shock covariance is the
    model's). When determinacy fails or an evaluation error occurs, returns a
    continuous quadratic penalty:
        L_penalty(gamma) = 10^8 + 10^4 * ||gamma - gamma_0||^2
    preventing simplex collapse and allowing gradient-free optimizers
    (Nelder-Mead, Powell) to contract cleanly away from unstable regions.

    Achievable accuracy. The loss is quadratic at its minimum,
    L(gamma) ~ L* (1 + c/2 * ((gamma - gamma*)/gamma*)^2) with normalised
    curvature c = gamma*^2 L''(gamma*) / L*, so a search that only compares loss
    values locates the minimizing coefficients, and the allocation they imply,
    to a relative error of about sqrt(2 eps / c), where eps is the relative
    precision of the loss (at best machine epsilon, 2.2e-16). For a well-scaled
    problem (c of order one) this is roughly 1e-8, whatever the tolerances; a
    flat or badly scaled loss is located less precisely: for the targeting
    rule x = -phi*pi under the Phillips curve pi = beta*pi(+1) + kappa*x + u
    with AR(1) cost push, where phi* = kappa / (alpha (1 - beta rho)) = 500
    and c = 0.02, osr stops at a relative error of 1e-7 under both the new and
    the old tolerances. The loss itself is then accurate to about 1e-15 relative.
    With the default Nelder-Mead tolerances (``xatol=1e-8``, ``fatol`` scaled
    to 1e-12 of the initial loss), at 25 random calibrations of the Clarida,
    Gali and Gertler (1999) economy with the rule
    ``i = g/phi + phi_pi*pi + phi_x*x`` (bounds phi_pi in [1, 10],
    phi_x in [0, 10]), the implied output-gap response to a cost-push shock
    matches the closed-form optimal simple rule with a relative error of
    9e-9 (median) and 7e-8 (worst), and the loss to within 4e-15. SciPy's own
    Nelder-Mead defaults (xatol = fatol = 1e-4, used before this option
    existed) left errors up to 2.7e-5 when the optimum lies on a bound,
    for about 1.8 times fewer loss evaluations. When the optimum is a ridge
    (a continuum of rules implementing the same allocation), the allocation
    is determined to this accuracy but the coefficients are not: the
    returned point is one point of the ridge.

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
        Unused; kept for backward compatibility. It used to be stored as
        ``loss_initial`` when the baseline moments could not be evaluated,
        which is now NaN with a RuntimeWarning. The penalty surface above does
        not depend on it.
    xatol : float, default 1e-8
        Nelder-Mead only: absolute tolerance on the rule coefficients (the
        largest distance of the simplex vertices from the best vertex), in the
        coefficients' own units. Being absolute, it should scale with the
        coefficients: about 1e-8 times their magnitude. Much smaller values fall
        under the precision floor described above and mostly add evaluations;
        for coefficients much smaller than one, 1e-8 is a loose relative
        tolerance and a smaller value is needed.
    fatol : float, optional
        Nelder-Mead only: absolute tolerance on the spread of the loss across
        the simplex. Default ``1e-12 * max(1, |initial loss|)`` when the
        initial loss is finite, else 1e-12. Nelder-Mead stops when both
        ``xatol`` and ``fatol`` are met.
    options : Mapping[str, Any], optional
        Passed to ``scipy.optimize.minimize(options=...)`` and applied last, so
        it overrides ``maxiter``, ``xatol`` and ``fatol``; use it to set the
        tolerances of other optimizers (e.g. ``{"xtol": ..., "ftol": ...}``
        for Powell).

    Returns
    -------
    OSRResult
        Frozen dataclass containing optimal coefficients, initial and optimal loss,
        variance breakdown table, and standard presentation methods.
        ``converged`` is the optimizer's own success flag.

    Warns
    -----
    RuntimeWarning
        When the baseline loss at the initial coefficients cannot be computed
        (``loss_initial`` is NaN), or when re-solving the model at the returned
        coefficients fails or gives an indeterminate rule (``loss_opt`` is NaN
        and ``optimal_model`` is None). The optimizer's own objective value is
        never reported as the loss, because it may be the penalty. The
        ``variance_reduction_pct`` column is NaN wherever a variance is missing
        or the initial variance is zero.
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

    # Evaluate baseline loss and variances. A baseline that cannot be computed is
    # NaN (with a warning), never a placeholder number.
    try:
        tm_init = model.theoretical_moments()
        cov_init = tm_init.covariance
        var_init = {
            v: float(cov_init.loc[v, v]) if v in cov_init.index else np.nan
            for v in targets
        }
        loss_initial = float(sum(weights_dict.get(v, 0.0) * var_init[v] for v in targets))
    except Exception as exc:
        warnings.warn(
            f"osr(): failed to evaluate initial baseline moments ({type(exc).__name__}: {exc}); "
            "loss_initial is NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        var_init = {v: np.nan for v in targets}
        loss_initial = float("nan")
    else:
        if not np.isfinite(loss_initial):
            missing = [v for v in targets if not np.isfinite(var_init[v])]
            warnings.warn(
                "osr(): the baseline loss at the initial rule is not finite"
                + (f" (no finite variance for {missing})" if missing else "")
                + "; loss_initial is NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
            loss_initial = float("nan")
    initial_loss_ok = bool(np.isfinite(loss_initial))

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

    # Optimizer options: SciPy's Nelder-Mead defaults (xatol = fatol = 1e-4) stop
    # far above the attainable precision, so tighter defaults are set here.
    opt_options: dict[str, Any] = {"maxiter": maxiter}
    if str(optimizer).strip().lower().replace("_", "-") == "nelder-mead":
        if fatol is None:
            scale = abs(loss_initial) if initial_loss_ok else 1.0
            fatol = 1e-12 * max(1.0, scale)
        opt_options["xatol"] = float(xatol)
        opt_options["fatol"] = float(fatol)
    if options:
        opt_options.update(dict(options))

    # Run optimizer
    opt_res = scipy.optimize.minimize(
        _objective,
        gamma_0,
        method=optimizer,
        bounds=scipy_bounds if optimizer in ("Nelder-Mead", "Powell", "L-BFGS-B") else None,
        options=opt_options,
    )

    opt_gamma = np.asarray(opt_res.x, dtype=float)
    converged = bool(opt_res.success)
    message = str(opt_res.message)
    n_evals = int(getattr(opt_res, "nfev", eval_counter[0]))

    opt_params = {p: float(opt_gamma[idx]) for idx, p in enumerate(rule_params)}
    initial_params = {p: float(gamma_0[idx]) for idx, p in enumerate(rule_params)}

    # Re-solve at optimum to obtain optimal model and exact variances. When that
    # fails, or the rule found is not determinate, loss_opt is NaN with a warning:
    # the optimizer's last objective value may be a penalty (>= 1e8), not a loss.
    try:
        m_opt = _solve_with_params(model, opt_params)
        if not m_opt.is_determinate:
            raise RuntimeError(
                f"the rule at the returned coefficients is not determinate (eu = {tuple(m_opt.solution.eu)})"
            )
        tm_opt = m_opt.theoretical_moments()
        cov_opt = tm_opt.covariance
        var_optimal = {
            v: float(cov_opt.loc[v, v]) if v in cov_opt.index else np.nan
            for v in targets
        }
        loss_opt = float(sum(weights_dict.get(v, 0.0) * var_optimal[v] for v in targets))
    except Exception as exc:
        m_opt = None
        var_optimal = {v: np.nan for v in targets}
        loss_opt = float("nan")
        warnings.warn(
            f"osr(): re-solving the model at the returned coefficients failed ({type(exc).__name__}: "
            f"{exc}); loss_opt is NaN and optimal_model is None (the optimizer's last objective "
            f"value was {float(opt_res.fun):.6g}, which is a penalty when >= 1e8).",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        if not np.isfinite(loss_opt):
            warnings.warn(
                "osr(): the loss at the returned coefficients is not finite; loss_opt is NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
            loss_opt = float("nan")

    # Construct variance table (NaN wherever a variance, or the reduction, is undefined)
    rows = []
    for v in targets:
        w = weights_dict.get(v, 0.0)
        vi = var_init.get(v, np.nan)
        vo = var_optimal.get(v, np.nan)
        li = w * vi if np.isfinite(vi) else np.nan
        lo = w * vo if np.isfinite(vo) else np.nan
        red_pct = 100.0 * (vi - vo) / vi if (np.isfinite(vi) and np.isfinite(vo) and vi > 0) else np.nan
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

_LOSS_CRITERIA = ("unconditional", "conditional")

# A closed-loop transition with a root this close to (or beyond) the unit circle
# has no stationary distribution, so its unconditional loss does not exist.
_UNIT_ROOT_TOL = 1e-10


def _spectral_radius(transition: np.ndarray) -> float:
    A = np.asarray(transition, dtype=float)
    if A.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def _nonstationary_warning(caller: str, radius: float) -> None:
    warnings.warn(
        f"{caller}: the closed-loop transition has spectral radius {radius:.10g} >= 1, so the "
        "stationary distribution does not exist and the unconditional loss is NaN.",
        RuntimeWarning,
        stacklevel=3,
    )


def _discounted_second_moment(
    transition: np.ndarray, impact_cov: np.ndarray, beta: float
) -> np.ndarray | None:
    """Return X = sum_j beta^j G^j S G'^j, i.e. the solution of X = beta G X G' + S.

    With S = N Sigma N', X is (1 - beta) E_0 sum_t beta^t x_t x_t' for
    x_t = G x_{t-1} + N u_t started from x_{-1} = 0. Returns None when the
    discounted sum diverges (sqrt(beta) times the spectral radius of G >= 1).
    """
    A = np.sqrt(beta) * np.asarray(transition, dtype=float)
    if A.size and float(np.max(np.abs(np.linalg.eigvals(A)))) >= 1.0:
        return None
    X = scipy.linalg.solve_discrete_lyapunov(A, np.asarray(impact_cov, dtype=float))
    return 0.5 * (X + X.T)


def discretionary_policy(
    model: LinearModel,
    target_vars: Sequence[str],
    weights: Mapping[str, float] | Sequence[float] | pd.Series,
    instruments: Sequence[str] | str,
    *,
    beta: float = 0.99,
    max_iter: int = 2000,
    tol: float = 1e-9,
    policy_eq: int | str | Sequence[int | str] | None = None,
    targets: Mapping[str, float] | None = None,
    y_star: float | None = None,
    target_output: float | None = None,
    compare_commitment: bool = True,
    kappa: float | None = None,
    slope_pc: float | None = None,
    loss_criterion: str = "unconditional",
    **kwargs: Any,
) -> DiscretionaryPolicyResult:
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
    weights : Mapping[str, float], Sequence[float], or pd.Series
        Quadratic loss weights. If a sequence is given, matches target_vars.
    instruments : Sequence[str] or str
        Policy instrument variable name(s).
    beta : float, default 0.99
        Policymaker discount factor.
    max_iter : int, default 2000
        Maximum fixed-point policy iterations.
    tol : float, default 1e-9
        Convergence tolerance on feedback matrices ||F_{k+1} - F_k||_infty.
    policy_eq : int, str, or sequence, optional
        Explicit index or tag name of the policy rule equation(s) to remove.
    targets : Mapping[str, float], optional
        Target values for loss variables (e.g. {"y": 0.05}).
    y_star : float, optional
        Target output gap y* > 0 to quantify inflation bias.
    target_output : float, optional
        Alias for y_star.
    compare_commitment : bool, default True
        Whether to run lq_commitment to quantify stabilization bias. With False,
        ``stabilization_bias`` is NaN (not computed) and ``commitment_result``
        is None.
    kappa : float, optional
        Phillips curve slope parameter for computing theoretical inflation bias.
        If None, resolved dynamically from model parameters, kwargs, or dynamic Jacobian.
    slope_pc : float, optional
        Alias for kappa.
    loss_criterion : {"unconditional", "conditional"}, default "unconditional"
        Loss that ``stabilization_bias`` compares. "unconditional" uses the
        average loss over the stationary distribution (the ``loss`` field of
        each regime), which evaluates the timeless-perspective rule on average.
        Its sign depends on the calibration: for a small enough discount
        factor, discretion beats the timeless rule under this criterion
        whenever the output gap has some weight and prices are not flexible
        (Sauer 2010, Prop. 2). In Sauer's benchmark (Calvo parameter 0.8722,
        i.e. slope 0.02 at beta = 0.99, output weight 0.0625, serially
        uncorrelated shocks) discretion wins for beta < 0.839. Nor is the
        timeless rule the best rule of its own form under this criterion
        (Jensen and McCallum 2002).
        "conditional" uses (1 - beta) E_0 sum_t beta^t loss_t from the steady
        state (the ``conditional_loss`` field), the planner's own criterion,
        under which commitment is the Ramsey optimum and the bias is
        non-negative.
    **kwargs : Any
        Additional options passed to solver or parameter overrides.

    Returns
    -------
    DiscretionaryPolicyResult
        Frozen dataclass carrying optimal reaction functions, Riccati value matrix V,
        policy feedback matrix F, closed-loop transition matrices, unconditional and
        conditional losses, inflation bias, stabilization bias, and solved
        LinearModel under discretion.

    Warns
    -----
    RuntimeWarning
        When the commitment comparison cannot be computed: ``lq_commitment``
        raises (``commitment_result`` is then None), or the discretion or
        commitment loss under ``loss_criterion`` is NaN (for instance when the
        closed-loop transition has a unit root, so the unconditional loss does
        not exist). ``stabilization_bias`` is NaN in both cases.
    """
    loss_criterion = str(loss_criterion).strip().lower()
    if loss_criterion not in _LOSS_CRITERIA:
        raise ValueError(
            f"loss_criterion must be 'unconditional' or 'conditional', got {loss_criterion!r}."
        )
    if isinstance(instruments, str):
        instruments = [instruments]
    instruments = list(instruments)
    target_vars = list(target_vars)

    if isinstance(weights, Mapping):
        weights_dict = {k: float(v) for k, v in weights.items()}
    elif isinstance(weights, pd.Series):
        weights_dict = {str(k): float(v) for k, v in weights.items()}
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
    diff_F = float("inf")
    iterations = max_iter
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

        diff_F = float(np.max(np.abs(F_new - F)))
        diff_D = float(np.max(np.abs(D_new - D)))
        diff = max(diff_D, diff_F)
        D, F, V = D_new, F_new, V_new

        if diff_F < tol:
            converged = True
            iterations = it + 1
            break
    else:
        iterations = max_iter

    if not converged:
        warnings.warn(
            f"discretionary_policy(): Dennis (2007) iteration did not converge within {max_iter} steps (final diff={diff_F:.2e})."
        )

    # Polish and symmetrize Riccati continuation value
    M_x = np.vstack([D, F])
    W_perm = np.block([[Q, U], [U.T, R]])
    V = M_x.T @ W_perm @ M_x + beta * M_x.T @ V @ M_x
    V = 0.5 * (V + V.T)

    # Map state transition back to original model variable ordering
    perm = y_indices + inst_indices
    inv_perm = np.argsort(perm)

    G_perm = np.vstack([D, F])
    N_perm = np.vstack([d_u, f_u])

    G = G_perm[inv_perm, :][:, inv_perm]
    N_mat = N_perm[inv_perm, :]

    V_model = V[inv_perm, :][:, inv_perm]
    V_model = 0.5 * (V_model + V_model.T)
    # Exact discrete Lyapunov solution for Riccati continuation value V in model ordering
    # Bellman equation: V = G.T @ W_full @ G + beta * G.T @ V @ G
    # Discrete Lyapunov form: A_lyap @ V @ A_lyap.T - V + Q_lyap = 0
    # where A_lyap = sqrt(beta) * G.T and Q_lyap = G.T @ W_full @ G
    try:
        A_lyap = np.sqrt(beta) * G.T
        Q_lyap = G.T @ W_full @ G
        V_model = scipy.linalg.solve_discrete_lyapunov(A_lyap, Q_lyap)
        V_model = 0.5 * (V_model + V_model.T)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
        for _ in range(100):
            V_next = G.T @ W_full @ G + beta * G.T @ V_model @ G
            V_next = 0.5 * (V_next + V_next.T)
            if np.max(np.abs(V_next - V_model)) < 1e-12:
                V_model = V_next
                break
            V_model = V_next

    # Ensure eigenvalues of V_model are non-negative (PSD clamp np.maximum(w, 0.0))
    w_v, v_v = np.linalg.eigh(V_model)
    w_v = np.maximum(w_v, 0.0)
    V_model = v_v @ np.diag(w_v) @ v_v.T
    V_model = 0.5 * (V_model + V_model.T)

    # Shock covariance and unconditional loss
    sigma_u = getattr(model, "_shock_cov", None)
    if sigma_u is None:
        sigma_u = np.eye(n_u)
    else:
        sigma_u = np.asarray(sigma_u, dtype=float)

    radius_G = _spectral_radius(G)
    if radius_G >= 1.0 - _UNIT_ROOT_TOL:
        _nonstationary_warning("discretionary_policy()", radius_G)
        sigma_x = np.zeros((N, N))
        loss = np.nan
    else:
        try:
            sigma_x = scipy.linalg.solve_discrete_lyapunov(G, N_mat @ sigma_u @ N_mat.T)
            sigma_x = 0.5 * (sigma_x + sigma_x.T)
            loss = float(np.trace(W_full @ sigma_x))
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            warnings.warn(
                f"discretionary_policy(): the unconditional loss is NaN because the Lyapunov "
                f"solve failed ({type(exc).__name__}: {exc}).",
                RuntimeWarning,
                stacklevel=2,
            )
            sigma_x = np.zeros((N, N))
            loss = np.nan

    # Conditional loss from the steady state: trace(W X), X = beta G X G' + N Sigma N'
    try:
        X_cond = _discounted_second_moment(G, N_mat @ sigma_u @ N_mat.T, beta)
        conditional_loss = np.nan if X_cond is None else float(np.trace(W_full @ X_cond))
    except (ValueError, ArithmeticError, np.linalg.LinAlgError):
        conditional_loss = np.nan

    # Policy reaction function table
    # Columns are variables in perm order [y_names + instruments]
    col_names = [variables[j] for j in perm]
    df_policy_rules = pd.DataFrame(F, index=instruments, columns=col_names)
    df_F = pd.DataFrame(F, index=instruments, columns=col_names)
    df_transition = pd.DataFrame(G, index=variables, columns=variables)
    df_impact = pd.DataFrame(N_mat, index=variables, columns=list(model.shocks))
    weights_series = pd.Series(weights_dict, name="weight")

    # Formal bias quantification
    y_star_val = 0.0
    if y_star is not None:
        y_star_val = float(y_star)
    elif target_output is not None:
        y_star_val = float(target_output)
    elif targets is not None:
        for y_cand in ("y", "y_gap", "output", "x", "gap"):
            if y_cand in targets:
                y_star_val = float(targets[y_cand])
                break
        if y_star_val == 0.0 and len(targets) > 0:
            for k, v in targets.items():
                if k not in ("pi", "pfe", "inflation", "infl"):
                    y_star_val = float(v)
                    break

    # Dynamic parameter resolution for Phillips curve slope kappa:
    # 1. Check explicit kappa, slope_pc arguments and kwargs ("kappa", "slope_pc", "kap", "pc_slope")
    kappa_val = None
    if kappa is not None:
        kappa_val = float(kappa)
    elif slope_pc is not None:
        kappa_val = float(slope_pc)
    else:
        for k_kw in ("kappa", "slope_pc", "kap", "pc_slope"):
            if k_kw in kwargs and kwargs[k_kw] is not None:
                try:
                    kappa_val = float(kwargs[k_kw])
                    break
                except (TypeError, ValueError, Exception):
                    pass

    # 2. Check model._params and model.parameters mappings
    if kappa_val is None:
        param_sources: list[Mapping[str, Any]] = []
        _p = getattr(model, "_params", None)
        if isinstance(_p, Mapping):
            param_sources.append(_p)
        _p2 = getattr(model, "parameters", None)
        if isinstance(_p2, Mapping):
            param_sources.append(_p2)

        for p_map in param_sources:
            for k_name in ("kappa", "slope_pc", "kap", "pc_slope"):
                if k_name in p_map and p_map[k_name] is not None:
                    try:
                        kappa_val = float(p_map[k_name])
                        break
                    except (TypeError, ValueError, Exception):
                        pass
            if kappa_val is not None:
                break

    # 3. Check structural Phillips curve row from dynamic Jacobian if parameters are unassigned
    if kappa_val is None and A_0 is not None:
        pi_idx = None
        for pi_name in ("pi", "inflation", "infl", "pfe"):
            if pi_name in variables:
                pi_idx = variables.index(pi_name)
                break
        y_idx = None
        for y_name in ("y", "y_gap", "output", "x", "gap"):
            if y_name in variables:
                y_idx = variables.index(y_name)
                break
        if pi_idx is not None and y_idx is not None:
            candidates: list[tuple[bool, float]] = []
            for row in priv_eq_indices:
                c_pi = A_0[row, pi_idx]
                c_y = A_0[row, y_idx]
                if abs(c_pi) > 1e-12 and abs(c_y) > 1e-12:
                    has_pi_lead = A_plus is not None and abs(A_plus[row, pi_idx]) > 1e-12
                    slope = abs(c_y / c_pi)
                    candidates.append((has_pi_lead, slope))
            if candidates:
                candidates.sort(key=lambda item: item[0], reverse=True)
                kappa_val = float(candidates[0][1])

    # 4. Fall back to 0.5 only if completely undetermined
    if kappa_val is None:
        kappa_val = 0.5

    w_pi = None
    for pi_name in ("pi", "inflation", "infl", "pfe"):
        if pi_name in weights_dict:
            w_pi = float(weights_dict[pi_name])
            break
    if w_pi is None or w_pi == 0.0:
        w_pi = 1.0

    w_y = None
    for y_name in ("y", "y_gap", "output", "x", "gap"):
        if y_name in weights_dict:
            w_y = float(weights_dict[y_name])
            break
    if w_y is None:
        w_y = 0.25

    lambda_y = float(w_y / w_pi)

    if y_star_val > 0.0:
        denom = lambda_y * (1.0 - beta) + (kappa_val ** 2)
        inflation_bias = float((kappa_val * lambda_y / denom) * y_star_val)
    else:
        inflation_bias = 0.0

    # Stabilization bias: NaN whenever it is not computed (compare_commitment=False)
    # or cannot be computed (the commitment solve fails, or either loss is NaN);
    # the last two cases also warn. It is never reported as a silent 0.0.
    commitment_res = None
    stabilization_bias = float("nan")
    if compare_commitment:
        try:
            commitment_res = lq_commitment(
                model=model,
                target_vars=target_vars,
                weights=weights_dict,
                instruments=instruments,
                beta=beta,
                policy_eq=policy_eq,
            )
        except Exception as exc:  # the comparison must not abort the discretion solve
            commitment_res = None
            warnings.warn(
                f"discretionary_policy(): the commitment solve failed ({type(exc).__name__}: {exc}); "
                "stabilization_bias is NaN and commitment_result is None.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            if loss_criterion == "conditional":
                disc_loss, comm_loss = conditional_loss, commitment_res.conditional_loss
            else:
                disc_loss, comm_loss = loss, commitment_res.loss
            unavailable = [
                f"the {name} {loss_criterion} loss is {'NaN' if np.isnan(value) else 'infinite'}"
                for name, value in (("discretion", disc_loss), ("commitment", comm_loss))
                if not np.isfinite(value)
            ]
            if unavailable:
                warnings.warn(
                    f"discretionary_policy(): {'; '.join(unavailable)}; stabilization_bias is NaN.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                stabilization_bias = float(disc_loss - comm_loss)

    # Wrap into LinearModel for seamless .irf(), .fevd(), .simulate()
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

    return DiscretionaryPolicyResult(
        regime="discretion",
        target_vars=tuple(target_vars),
        weights=weights_series,
        instruments=tuple(instruments),
        beta=float(beta),
        loss=loss,
        conditional_loss=conditional_loss,
        policy_rules=df_policy_rules,
        F=df_F,
        V=V_model,
        transition_matrix=df_transition,
        impact_matrix=df_impact,
        inflation_bias=inflation_bias,
        stabilization_bias=stabilization_bias,
        converged=converged,
        iterations=iterations,
        diff=diff_F,
        multipliers=(),
        linear_model=m_discretion,
        commitment_result=commitment_res,
        loss_criterion=loss_criterion,
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
    timeless: bool | None = None,
    policy_eq: int | str | Sequence[int | str] | None = None,
    **kwargs: Any,
) -> PolicyResult:
    """Solve optimal linear-quadratic commitment policy.

    Forms the social planner's Lagrangian over the linear rational-expectations
    constraints and quadratic loss:
        min_{z_t} 1/2 E_0 sum_{t=0}^infty beta^t z_t^T W z_t,  W = diag(weights),
    subject to the M private sector equilibrium conditions left after removing
    the instrument's policy rule:
        A_+ E_t z_{t+1} + A_0 z_t + A_- z_{t-1} + B_u u_t = 0
    yielding the augmented (N + M)-dimensional saddle-path system in
    X_t = (z_t, lambda_t):
        cal_A_+ E_t X_{t+1} + cal_A_0 X_t + cal_A_- X_{t-1} + cal_B_u u_t = 0
    with planner rows W z_t + A_0' lambda_t + beta^{-1} A_+' lambda_{t-1}
    + beta A_-' E_t lambda_{t+1} = 0, solved by klein_solve. Policy multipliers
    are exposed as model variables (states when they enter with a lag).

    Ramsey versus timeless perspective. The solved law of motion, with lagged
    multipliers as states, is the same under both; they differ only in the
    initial multiplier. The Ramsey plan chosen at t0 sets lambda_{-1} = 0
    whatever the history; the timeless perspective applies the t >= 1
    condition at t0 as well, i.e. uses the multiplier implied by past policy
    (Jensen and McCallum 2002, eqs. 4a-4c and 5). The outputs are evaluated as
    follows:

    * the linear model's impulse responses and ``conditional_loss`` start from
      the steady state with lagged variables and multipliers at zero
      (lambda_{-1} = 0). From the steady state the multiplier implied by past
      policy is also zero, so the Ramsey plan and the timeless rule coincide
      there (JM 2002, fn. 12: rule (5) is optimal if y_0 = 0);
    * ``loss`` is the unconditional expectation over the stationary
      distribution of (z_t, lambda_t): the timeless rule evaluated on average,
      the criterion of the literature reviewed by Jensen and McCallum (2002).
      It is NaN, with a RuntimeWarning, when the closed-loop transition has a
      root on or outside the unit circle.

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
    timeless : bool, optional
        Deprecated; it never changed the result and passing it (either value)
        emits a FutureWarning. See "Ramsey versus timeless perspective" above
        for what the outputs are.
    policy_eq : int, str, or sequence, optional
        Explicit index or tag name of the policy rule equation(s) to remove.

    Returns
    -------
    PolicyResult
        Frozen dataclass carrying augmented LinearModel where Lagrange multipliers
        (mult_*) are accessible as model variables for IRFs, FEVD, and moments.

    References
    ----------
    Jensen, C., and McCallum, B. T. (2002). The non-optimality of proposed
        monetary policy rules under timeless-perspective commitment. Economics
        Letters, 77(2), 163-168 (NBER Working Paper 8882).
    """
    if timeless is not None:
        warnings.warn(
            "lq_commitment(timeless=...) is deprecated and has no effect: the solved law of "
            "motion is the same for the Ramsey plan and the timeless-perspective rule, impulse "
            "responses and conditional_loss start from the steady state (lambda_{-1} = 0), where "
            "the two coincide, and loss averages over the stationary distribution. Omit the "
            "argument.",
            FutureWarning,
            stacklevel=2,
        )
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

    radius_G = _spectral_radius(G)
    if radius_G >= 1.0 - _UNIT_ROOT_TOL:
        _nonstationary_warning("lq_commitment()", radius_G)
        cov_X = np.zeros((total_vars, total_vars))
        loss = np.nan
    else:
        try:
            sigma_s = scipy.linalg.solve_discrete_lyapunov(G, N_state @ sigma_u @ N_state.T)
            sigma_s = 0.5 * (sigma_s + sigma_s.T)
            cov_X = F_full @ sigma_s @ F_full.T + L_full @ sigma_u @ L_full.T
            loss = float(np.trace(W @ cov_X[:N, :N]))
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            warnings.warn(
                f"lq_commitment(): the unconditional loss is NaN because the Lyapunov solve "
                f"failed ({type(exc).__name__}: {exc}).",
                RuntimeWarning,
                stacklevel=2,
            )
            cov_X = np.zeros((total_vars, total_vars))
            loss = np.nan

    # Conditional loss from the steady state (zero lagged multipliers, so the
    # Ramsey plan): z_t = F s_{t-1} + L u_t gives L Sigma L' + beta F X_s F'.
    try:
        X_s = _discounted_second_moment(G, N_state @ sigma_u @ N_state.T, beta)
        if X_s is None:
            conditional_loss = np.nan
        else:
            cov_cond = beta * F_full @ X_s @ F_full.T + L_full @ sigma_u @ L_full.T
            conditional_loss = float(np.trace(W @ cov_cond[:N, :N]))
    except (ValueError, ArithmeticError, np.linalg.LinAlgError):
        conditional_loss = np.nan

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
        conditional_loss=conditional_loss,
    )


# ==============================================================================
# 4. Optimal Policy Top-Level Dispatch
# ==============================================================================

def optimal_policy(
    model: LinearModel,
    loss: Mapping[str, Any] | Sequence[float] | str,
    rule: str = "discretion",
    *,
    instruments: Sequence[str] | str | None = None,
    target_vars: Sequence[str] | None = None,
    beta: float = 0.99,
    targets: Mapping[str, float] | None = None,
    y_star: float | None = None,
    target_output: float | None = None,
    max_iter: int = 2000,
    tol: float = 1e-9,
    policy_eq: int | str | Sequence[int | str] | None = None,
    compare_commitment: bool = True,
    loss_criterion: str = "unconditional",
    **kwargs: Any,
) -> DiscretionaryPolicyResult | PolicyResult:
    """Solve optimal monetary and macroeconomic policy under discretion or commitment.

    Dispatches to Dennis (2007) Markov-perfect discretionary policy or
    timeless-perspective linear-quadratic commitment policy.

    Parameters
    ----------
    model : LinearModel
        The baseline structural linear DSGE model.
    loss : dict, sequence, or str
        Loss specification:
        - dict: e.g. ``{"pi": 1.0, "y": 0.25}``, or with keys
          ``"weights"``, ``"targets"``, ``"instruments"``, ``"target_vars"``.
        - str: quadratic loss string expression like ``"pi^2 + 0.25 * y^2"``
          or ``"pi^2 + 0.25 * (y - 0.05)^2"``.
        - sequence: loss weights matching ``target_vars``.
    rule : {"discretion", "commitment"}, default "discretion"
        Policy regime to solve. "ramsey" and "timeless" are accepted aliases of
        "commitment" and return the same :func:`lq_commitment` result: its law
        of motion is common to both, and its responses and conditional loss
        start from the steady state, where they coincide.
    instruments : Sequence[str] or str, optional
        Policy instrument(s). If omitted, inferred from model variables (e.g. "r", "i").
    target_vars : Sequence[str], optional
        Target variables. If omitted, inferred from loss keys or expression.
    beta : float, default 0.99
        Policymaker discount factor.
    targets : Mapping[str, float], optional
        Target values (e.g. {"y": 0.05} for target output gap).
    y_star : float, optional
        Target output gap y* > 0 to quantify inflation bias.
    target_output : float, optional
        Alias for y_star.
    max_iter : int, default 2000
        Maximum iterations for policy function iteration.
    tol : float, default 1e-9
        Convergence tolerance on policy feedback matrix F.
    policy_eq : int, str, or sequence, optional
        Equation(s) in model corresponding to policy rule to remove.
    compare_commitment : bool, default True
        Under discretion, whether to solve commitment and compute stabilization bias.
    loss_criterion : {"unconditional", "conditional"}, default "unconditional"
        Under discretion, the loss that ``stabilization_bias`` compares; see
        :func:`discretionary_policy`. Both results always carry ``loss`` and
        ``conditional_loss``.
    **kwargs : Any
        Additional keyword arguments forwarded to the solver.

    Returns
    -------
    DiscretionaryPolicyResult or PolicyResult
        Solved policy result container.
    """
    weights: Any = None
    extracted_targets: dict[str, float] = {}

    if isinstance(loss, str):
        import re

        s = loss.replace(" ", "")
        pattern = re.compile(r"([+-]?[0-9.]*)\*?\(?([a-zA-Z_]\w*)(?:-([0-9.]+))?\)?\^2")
        parsed_weights: dict[str, float] = {}
        for match in pattern.finditer(s):
            w_str, var, tgt_str = match.groups()
            if not w_str or w_str == "+":
                w = 1.0
            elif w_str == "-":
                w = -1.0
            else:
                w = float(w_str)
            parsed_weights[var] = w
            if tgt_str:
                extracted_targets[var] = float(tgt_str)
        if not parsed_weights:
            raise ValueError(f"Could not parse quadratic loss from expression: '{loss}'")
        weights = parsed_weights
        if target_vars is None:
            target_vars = list(parsed_weights.keys())

    elif isinstance(loss, Mapping):
        if "weights" in loss:
            weights = loss["weights"]
            if "targets" in loss and targets is None:
                targets = loss["targets"]
            if "target_vars" in loss and target_vars is None:
                target_vars = loss["target_vars"]
            if "instruments" in loss and instruments is None:
                instruments = loss["instruments"]
            if "beta" in loss and beta == 0.99:
                beta = float(loss["beta"])
            if "y_star" in loss and y_star is None:
                y_star = float(loss["y_star"])
            elif "target_output" in loss and target_output is None:
                target_output = float(loss["target_output"])
        else:
            weights = loss
            if target_vars is None:
                target_vars = list(loss.keys())
    else:
        weights = loss

    if targets is None and extracted_targets:
        targets = extracted_targets

    if target_vars is None:
        if isinstance(weights, Mapping):
            target_vars = list(weights.keys())
        elif isinstance(weights, pd.Series):
            target_vars = list(weights.index)
        else:
            raise ValueError("target_vars must be specified when weights is a sequence.")

    if instruments is None:
        for cand in ("r", "i", "R", "interest", "i_t", "r_t"):
            if cand in model.variables:
                instruments = cand
                break
        if instruments is None:
            raise ValueError(
                "Policy instrument(s) must be specified via instruments=... "
                f"Available model variables: {model.variables}"
            )

    rule_lower = rule.strip().lower()
    if rule_lower in ("discretion", "discretionary", "disc", "markov_perfect"):
        return discretionary_policy(
            model=model,
            target_vars=target_vars,
            weights=weights,
            instruments=instruments,
            beta=beta,
            max_iter=max_iter,
            tol=tol,
            policy_eq=policy_eq,
            targets=targets,
            y_star=y_star,
            target_output=target_output,
            compare_commitment=compare_commitment,
            loss_criterion=loss_criterion,
            **kwargs,
        )
    elif rule_lower in ("commitment", "comm", "ramsey", "timeless"):
        return lq_commitment(
            model=model,
            target_vars=target_vars,
            weights=weights,
            instruments=instruments,
            beta=beta,
            policy_eq=policy_eq,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown policy rule '{rule}'. Supported regimes are 'discretion' and 'commitment'."
        )

