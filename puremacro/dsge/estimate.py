"""Generic Bayesian DSGE estimator via Random-Walk Metropolis-Hastings.

Model-agnostic. Mode refinement (scipy.optimize.minimize, L-BFGS-B) →
numerical Hessian → proposal cov c²·H⁻¹ (fallback diag(prior_stds²) if
H not PD) → multi-chain RW-MH via puremacro.mcmc.random_walk_metropolis.

Kalman initialisation
---------------------
A solved linear DSGE is stationary by construction, so the state vector
has a well-defined unconditional distribution and the Kalman recursion
must start from it: ``P0`` solves the discrete Lyapunov equation
``P = T P T' + R Q R'`` and ``a0`` solves ``(I - T) a0 = c``.  This module
computes that per parameter draw and passes it to
:func:`puremacro.state_space.kalman_filter` explicitly.

This matters for the *level* of the likelihood, not just its precision.
``kalman_filter``'s own default is the approximately-diffuse
``P0 = 1e6 * I``, which makes ``loglik`` a function of an arbitrary tuning
constant: on Smets-Wouters (2007) the log-likelihood moves by roughly
-47 points for every factor of 100 in that constant and never converges.
Marginal likelihoods, Bayes factors and any model comparison built on
such a number are meaningless.  ``kalman_filter``'s default is unchanged
(nowcast/forecast callers rely on it); only this driver overrides it.

If a draw's transition matrix has a (near-)unit root the Lyapunov solve
has no bounded solution; the driver then falls back to ``kalman_filter``'s
diffuse default and says so once, by name, in a ``UserWarning``.

This module hosts the helpers that were previously inlined in
``puremacro.dsge.sw07_estimate`` but contained nothing SW07-specific:
``_vec_to_dict``, ``_make_neg_log_posterior``, ``_initial_vec_from_dict``,
``_nearest_pd``, ``_find_finite_start``.
"""
from __future__ import annotations

import math
import warnings
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import scipy.linalg
from puremacro.dsge.mode import find_mode as _find_mode

from puremacro.dsge._results import DSGEPosteriorResult
from puremacro.dsge.priors import (
    log_prior, prior_stds, param_bounds, param_names, _validate_priors,
)
from puremacro.dsge.build import LinearModel, ModelError
from puremacro.dsge.occbin import (
    OccBinConstraint,
    PiecewiseKalmanResult,
    piecewise_kalman_filter,
)
from puremacro.mcmc import random_walk_metropolis
from puremacro.numerics import numerical_hessian
from puremacro.state_space import StateSpaceModel, kalman_filter



# Finite stand-in for +inf handed to the mode optimiser (Dynare's
# ``objective_function_penalty_base`` pattern).  L-BFGS-B cannot
# finite-difference a gradient across an infinite value: when a trial step
# leaves the feasible region the objective is +inf, the difference quotient
# is inf/nan, and scipy stops at iteration 1 reporting ``success=True`` at
# the caller's own starting point.  A large finite value keeps the
# quasi-Newton step arithmetic well defined and, where it cannot rescue the
# step, turns that false "success" into an honest abnormal termination.
#
# Measured on Smets-Wouters (2007): with the stationary P0 in place the
# search runs its full 100 iterations either way, so the penalty is not
# what unblocks that model -- the initialisation is.  The penalty is kept
# because it is strictly more honest, and it is confined to the optimiser:
# the MCMC target keeps +inf/-inf so an infeasible draw is still rejected
# outright, and a penalised point is never accepted as the mode because it
# cannot improve on a finite starting value.
_OPT_PENALTY = 1e10

# Below this spectral radius the Lyapunov solve for P0 is trusted.
_STATIONARITY_TOL = 1e-8


def _stationary_init(model: StateSpaceModel) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Unconditional ``(a0, P0)`` of a stable linear state-space model.

    Solves ``P = T P T' + R Q R'`` and ``(I - T) a0 = c``.  Returns
    ``(None, None)`` — never a silently wrong matrix — when the transition
    matrix is not comfortably inside the unit circle, when the Lyapunov
    solve fails or returns a non-finite/non-residual-checked answer, or
    when ``(I - T)`` is singular.  Callers must treat ``(None, None)`` as
    "no unconditional distribution exists" and say so.
    """
    T = np.asarray(model.T, dtype=float)
    m = T.shape[0]
    if not np.all(np.isfinite(T)):
        return None, None
    try:
        rho = float(np.max(np.abs(np.linalg.eigvals(T))))
    except np.linalg.LinAlgError:
        return None, None
    if not np.isfinite(rho) or rho > 1.0 - _STATIONARITY_TOL:
        return None, None

    RQR = np.asarray(model.R @ model.Q @ model.R.T, dtype=float)
    if not np.all(np.isfinite(RQR)):
        return None, None
    try:
        P0 = scipy.linalg.solve_discrete_lyapunov(T, RQR)
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError, ValueError):
        return None, None
    P0 = np.asarray(P0, dtype=float)
    if not np.all(np.isfinite(P0)):
        return None, None
    P0 = 0.5 * (P0 + P0.T)
    resid = float(np.max(np.abs(P0 - T @ P0 @ T.T - RQR)))
    scale = max(1.0, float(np.max(np.abs(P0))))
    if resid > 1e-6 * scale:
        return None, None

    c = np.asarray(model.c, dtype=float).ravel()
    if np.any(c != 0.0):
        try:
            a0 = np.linalg.solve(np.eye(m) - T, c)
        except np.linalg.LinAlgError:
            return None, None
        if not np.all(np.isfinite(a0)):
            return None, None
    else:
        a0 = np.zeros(m)
    return a0, P0


def _vec_to_dict(
    vec: np.ndarray,
    names: Sequence[str],
    fixed_params: dict | None = None,
) -> dict:
    """Combine an estimated-param vector with fixed params into a single dict."""
    if len(vec) != len(names):
        raise ValueError(f"vec length {len(vec)} doesn't match {len(names)} names")
    out = dict(fixed_params or {})
    for nm, val in zip(names, vec):
        out[nm] = float(val)
    return out


class _LikelihoodFailure:
    """Records the last exception a parameter draw raised, for diagnostics.

    ``_make_neg_log_posterior`` has to turn every failed draw into a
    non-finite objective value, which throws away the filter's own
    diagnosis.  Keeping the last exception lets the driver re-raise it
    (chained) instead of telling the user to "pick a better starting
    point" for a model that is, say, stochastically singular.
    """

    __slots__ = ("exc", "vec")

    def __init__(self) -> None:
        self.exc: BaseException | None = None
        self.vec: np.ndarray | None = None

    def record(self, exc: BaseException, vec: np.ndarray) -> None:
        self.exc = exc
        self.vec = np.array(vec, dtype=float, copy=True)


def _make_neg_log_posterior(
    y: np.ndarray,
    observation_eq: Callable[[dict], StateSpaceModel],
    priors: dict,
    names: Sequence[str],
    fixed_params: dict | None,
    *,
    penalty: float = np.inf,
    failure: "_LikelihoodFailure | None" = None,
    caller: str = "estimate_dsge",
):
    """Closure returning -log_posterior(vec).

    The Kalman recursion is started from the model's **unconditional**
    state distribution (:func:`_stationary_init`), not from
    ``kalman_filter``'s approximately-diffuse ``P0 = 1e6 * I``; see the
    module docstring for why the likelihood level depends on that choice.
    A draw whose transition matrix has a (near-)unit root has no
    unconditional distribution — the diffuse default is used for it and a
    ``UserWarning`` naming ``caller`` is emitted once per closure.

    ``penalty`` is the value returned when a draw is infeasible.  It is
    ``+inf`` for the MCMC target (an infeasible draw must be rejected) and
    a large finite number for the mode optimiser (L-BFGS-B cannot
    finite-difference across ``inf`` and would stop at iteration 1).
    """
    warned = [False]

    def neg_log_post(vec: np.ndarray) -> float:
        try:
            params = _vec_to_dict(vec, names, fixed_params)
        except ValueError:
            return penalty
        lp = log_prior(params, priors)
        if not np.isfinite(lp):
            return penalty
        try:
            ssm = observation_eq(params)
            a0, P0 = _stationary_init(ssm)
            if P0 is None and not warned[0]:
                warned[0] = True
                warnings.warn(
                    f"{caller}: the transition matrix at some parameter "
                    "draws has a (near-)unit root, so the state has no "
                    "unconditional covariance; kalman_filter's diffuse "
                    "P0 = 1e6*I is used for those draws instead. The "
                    "log-likelihood LEVEL of such draws depends on that "
                    "arbitrary constant and is not comparable across "
                    "models -- treat any marginal likelihood built on it "
                    "as invalid.",
                    UserWarning,
                    stacklevel=2,
                )
            out = kalman_filter(y, ssm, a0=a0, P0=P0)
            ll = out["loglik"]
        except (np.linalg.LinAlgError, ValueError, RuntimeError,
                ZeroDivisionError, FloatingPointError) as exc:
            if failure is not None:
                failure.record(exc, vec)
            return penalty
        if not np.isfinite(ll):
            return penalty
        return -(ll + lp)
    return neg_log_post


def _make_piecewise_neg_log_posterior(
    data: pd.DataFrame,
    m_unconstrained: Any,
    m_constrained_dict: Any,
    constraints: Any,
    priors: dict,
    names: Sequence[str],
    fixed_params: dict | None,
    observed_vars: Sequence[str],
    horizon: int = 30,
    penalty: float = np.inf,
    failure: _LikelihoodFailure | None = None,
):
    from typing import Mapping
    from puremacro.dsge.dynare import build_dynare

    base_params = dict(getattr(m_unconstrained, "_params", None) or getattr(m_unconstrained, "params", None) or {})
    if fixed_params:
        base_params.update({k: float(v) for k, v in fixed_params.items()})

    def neg_log_post(vec: np.ndarray) -> float:
        for nm, val in zip(names, vec):
            pr = priors[nm]
            if val < pr.lb or val > pr.ub:
                return penalty
        par_dict = _vec_to_dict(vec, names, fixed_params)
        lp = log_prior(par_dict, priors)
        if not np.isfinite(lp):
            return penalty

        curr_params = dict(base_params)
        curr_params.update(par_dict)

        try:
            if getattr(m_unconstrained, "_dynare_equations", None) is not None:
                ref_m = build_dynare(
                    m_unconstrained._dynare_equations,
                    variables=m_unconstrained.variables,
                    shocks=m_unconstrained.shocks,
                    params=curr_params,
                    steady_state=m_unconstrained.steady_state,
                    check_steady_state=False,
                    strict=False,
                )
            else:
                ref_m = m_unconstrained

            if isinstance(m_constrained_dict, Mapping):
                cons_dict = {}
                for k, m_k in m_constrained_dict.items():
                    if getattr(m_k, "_dynare_equations", None) is not None:
                        k_base = dict(getattr(m_k, "_params", None) or getattr(m_k, "params", None) or base_params)
                        k_params = dict(k_base)
                        k_params.update(par_dict)
                        cons_dict[k] = build_dynare(
                            m_k._dynare_equations,
                            variables=m_k.variables,
                            shocks=m_k.shocks,
                            params=k_params,
                            steady_state=m_k.steady_state,
                            check_steady_state=False,
                            strict=False,
                        )
                    else:
                        cons_dict[k] = m_k
            else:
                cons_dict = m_constrained_dict

            pkf_res = piecewise_kalman_filter(
                ref_m,
                cons_dict,
                data,
                varobs=observed_vars,
                constraints=constraints,
                horizon=horizon,
            )
            ll = pkf_res.log_likelihood
        except Exception as exc:
            if failure is not None:
                failure.record(exc, vec)
            return penalty

        if not np.isfinite(ll):
            return penalty
        return -(ll + lp)

    return neg_log_post


def _interior_point(value: float, spec, *, margin: float = 1e-3) -> float:
    """Clip ``value`` strictly inside ``[lb, ub]``, never returning +-inf.

    The margin shrinks with the width of a narrow support, and an infinite
    bound is simply not clipped against — which is what the old
    ``lb + 1e-3`` rule got wrong: for ``lb = -inf`` it returned ``-inf``,
    which then reached ``rng.uniform`` and raised ``OverflowError``.
    """
    lb = float(spec["lb"])
    ub = float(spec["ub"])
    width = ub - lb
    eps = margin if not np.isfinite(width) else min(margin, 0.01 * width)
    lo = lb + eps if np.isfinite(lb) else -np.inf
    hi = ub - eps if np.isfinite(ub) else np.inf
    if not np.isfinite(value):
        # Fall back to something finite and inside the box.
        if np.isfinite(lo) and np.isfinite(hi):
            value = 0.5 * (lo + hi)
        elif np.isfinite(lo):
            value = lo
        elif np.isfinite(hi):
            value = hi
        else:
            value = 0.0
    return float(np.clip(value, lo, hi))


def _initial_vec_from_dict(
    initial_params: dict,
    priors: dict,
    *,
    caller: str = "estimate_dsge",
) -> np.ndarray:
    """Build the initial parameter vector from a dict.

    A value outside its prior's ``[lb, ub]`` support, or a value the caller
    did not supply at all, is replaced by a point strictly inside the
    support **and a ``UserWarning`` naming the parameter, the value it was
    given and the value used instead** is emitted.  The old behaviour
    snapped silently to ``lb + 1e-3``, so the substituted number was
    reported back to the user as "the posterior mode" with nothing to say
    it had been changed — and for an unbounded prior it substituted
    ``-inf``.
    """
    vec = []
    for name, spec in priors.items():
        lb = float(spec["lb"])
        ub = float(spec["ub"])
        if not (lb < ub):
            raise ValueError(
                f"{caller}: prior {name!r} has lb={lb!r} >= ub={ub!r}; "
                "the support is empty."
            )
        val = initial_params.get(name)
        if val is None:
            fallback = _interior_point(float(spec["mean"]), spec)
            warnings.warn(
                f"{caller}: initial_params has no entry for {name!r}; "
                f"starting the mode search from {fallback!r} (the prior "
                "mean, clipped into the support).",
                UserWarning,
                stacklevel=2,
            )
            val = fallback
        else:
            val = float(val)
            if not (lb <= val <= ub):
                fallback = _interior_point(val, spec)
                warnings.warn(
                    f"{caller}: initial_params[{name!r}] = {val!r} is "
                    f"outside the prior support [{lb!r}, {ub!r}]; using "
                    f"{fallback!r} instead. The value you supplied has "
                    "zero prior density, so it cannot be a mode.",
                    UserWarning,
                    stacklevel=2,
                )
                val = fallback
        vec.append(float(val))
    return np.array(vec, dtype=float)


def _nearest_pd(
    A: np.ndarray,
    *,
    floor_rel: float = 1e-10,
    floor_abs: float | None = None,
) -> np.ndarray:
    """Nearest symmetric positive-definite matrix by an eigenvalue floor.

    Symmetrise, then clip the eigenvalues at
    ``max(floor_rel, floor_rel * lambda_max)``, raised to ``floor_abs``
    when one is given, and rebuild.  ``floor_abs`` is what keeps the
    *inverse* bounded: without it a flat or negative curvature direction
    becomes a proposal variance of order ``1/floor_rel``.  The result is
    verified positive definite before it is returned; a ``RuntimeError`` is
    raised if it is not.

    The previous Higham-style loop returned matrices whose minimum
    eigenvalue was **negative** — ``np.linalg.cholesky`` can succeed on a
    matrix with an ``O(-1e-9)`` eigenvalue, and the loop exited on that
    success — so the advertised "positive definite" contract was violated
    without any signal.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"_nearest_pd expects a square matrix, got {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError(
            "_nearest_pd: matrix contains non-finite entries; it cannot be "
            "repaired into a positive-definite one."
        )
    B = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(B)
    floor = max(floor_rel, floor_rel * float(np.max(w)) if np.max(w) > 0 else floor_rel)
    if floor_abs is not None and np.isfinite(floor_abs) and floor_abs > 0.0:
        floor = max(floor, float(floor_abs))
    out = V @ np.diag(np.maximum(w, floor)) @ V.T
    out = 0.5 * (out + out.T)
    min_eig = float(np.min(np.linalg.eigvalsh(out)))
    if not (min_eig > 0.0):
        raise RuntimeError(
            f"_nearest_pd failed: repaired matrix still has minimum "
            f"eigenvalue {min_eig:.3e} <= 0."
        )
    return out


def _find_finite_start(
    neg_log_post,
    rng: np.random.Generator,
    priors: dict,
    max_tries: int = 200,
    *,
    failure: "_LikelihoodFailure | None" = None,
    caller: str = "estimate_dsge",
) -> np.ndarray:
    """Random points inside the prior box until neg_log_post is finite.

    Infinite prior bounds are replaced by ``mean +- 5*std`` (or the finite
    bound offset by ``10*std``) before sampling; handing ``rng.uniform``
    an infinite range raises ``OverflowError: high - low range exceeds
    valid bounds``, which is exactly what the default ``NormalPrior``
    bounds used to produce.

    Raises ``ValueError`` — chained to the last likelihood exception, when
    one was recorded — if no feasible point is found, instead of returning
    an infeasible fallback for the sampler to choke on downstream.
    """
    sample_box = []
    for spec in priors.values():
        lb = float(spec["lb"])
        ub = float(spec["ub"])
        mean = float(spec["mean"])
        std = float(spec["std"])
        if not np.isfinite(std) or std <= 0.0:
            std = 1.0
        if not np.isfinite(mean):
            mean = 0.0
        lo = lb if np.isfinite(lb) else (ub - 10.0 * std if np.isfinite(ub)
                                         else mean - 5.0 * std)
        hi = ub if np.isfinite(ub) else (lb + 10.0 * std if np.isfinite(lb)
                                         else mean + 5.0 * std)
        if not (lo < hi):
            lo, hi = mean - 5.0 * std, mean + 5.0 * std
        sample_box.append((float(lo), float(hi)))

    for _ in range(max_tries):
        vec = np.array([rng.uniform(lo, hi) for (lo, hi) in sample_box])
        if np.isfinite(neg_log_post(vec)):
            return vec
    # Hard fallback: prior means clipped to the interior of the support.
    vec = np.array([_interior_point(float(spec["mean"]), spec, margin=1e-6)
                    for spec in priors.values()])
    if np.isfinite(neg_log_post(vec)):
        return vec
    msg = (
        f"{caller}: no parameter vector with a finite log-posterior was "
        f"found in {max_tries} draws from the prior box, nor at the prior "
        "means. The log-posterior is -inf everywhere tried, so the model "
        "cannot be estimated as specified."
    )
    if failure is not None and failure.exc is not None:
        raise ValueError(
            msg + " The last failure from the state-space/Kalman step was: "
            f"{type(failure.exc).__name__}: {failure.exc}"
        ) from failure.exc
    raise ValueError(msg)


def _check_stochastic_singularity(
    y: np.ndarray,
    observation_eq: Callable[[dict], StateSpaceModel],
    params: dict,
    *,
    caller: str = "estimate_dsge",
) -> None:
    """Raise a named error when the model cannot generate the observables.

    With ``n`` observables the one-step-ahead innovation covariance is
    ``F = Z P Z' + H``; if the model's shocks plus measurement error span
    fewer than ``n`` dimensions of observable space then ``F`` is singular
    at *every* parameter vector, the likelihood is -inf everywhere, and
    the driver would otherwise report "no finite starting point" — a
    statement about the starting values for what is a structural property
    of the model.  ``rank(Z R Q R' Z' + H) < n`` is exactly that condition.

    Silent on anything it cannot evaluate: this is a diagnostic, never a
    gate.
    """
    n_obs = int(np.asarray(y).reshape(len(y), -1).shape[1])
    try:
        ssm = observation_eq(params)
        Z = np.asarray(ssm.Z, dtype=float)
        RQR = np.asarray(ssm.R @ ssm.Q @ ssm.R.T, dtype=float)
        H = np.asarray(ssm.H, dtype=float)
        F_gen = Z @ RQR @ Z.T + H
        if not np.all(np.isfinite(F_gen)):
            return
        rank = int(np.linalg.matrix_rank(F_gen))
    except Exception:
        return
    if rank < n_obs:
        n_shocks = int(np.asarray(ssm.Q, dtype=float).shape[0])
        raise ValueError(
            f"{caller}: the model is stochastically singular -- "
            f"{n_obs} observed series are driven by a shock/measurement-"
            f"error covariance of rank {rank} (Q is {n_shocks}x{n_shocks}, "
            f"H has rank {int(np.linalg.matrix_rank(H))}). The Kalman "
            "innovation covariance F is singular at every parameter "
            "vector, so the likelihood is -inf everywhere and no starting "
            "point can help. Add measurement error (a positive-definite "
            "H), add structural shocks, or drop observables until their "
            f"number is at most {rank}."
        )


def estimate_dsge(
    data: pd.DataFrame,
    *,
    observation_eq: Callable[[dict], StateSpaceModel] | None = None,
    priors: dict,
    observed_vars: Sequence[str],
    initial_params: dict,
    fixed_params: dict | None = None,
    method: str = "kalman",
    m_unconstrained: Any = None,
    m_constrained_dict: Any = None,
    constraints: Any = None,
    occbin_regimes: Any = None,
    model_template: Any = None,
    model_name: str = "unknown",
    mode_compute: str = "lbfgs",
    n_draws: int = 10_000,
    n_chains: int = 2,
    burn_in: int = 2_000,
    seed: int = 0,
    **kwargs,
) -> DSGEPosteriorResult | NUTSResult:
    """Bayesian DSGE estimation via Random-Walk Metropolis-Hastings or NUTS.

    Supports standard linear Kalman filtering (method="kalman"),
    Piecewise Kalman Filtering (method="piecewise_kalman") for occasionally
    binding constraints (Giovannini, Pfeiffer & Ratto 2021), and No-U-Turn
    Sampler Hamiltonian Monte Carlo (method="nuts") with exact analytic
    likelihood gradients.
    """
    # 1. Validate data.
    missing = set(observed_vars) - set(data.columns)
    if missing:
        raise ValueError(f"data missing columns: {sorted(missing)}")
    if len(data) < 10:
        raise ValueError(f"data has only {len(data)} obs; need >= 10")
    if method != "piecewise_kalman" and data[list(observed_vars)].isna().any().any():
        raise ValueError("data contains NaN in observed_vars")
    y = data[list(observed_vars)].to_numpy()

    _validate_priors(priors, caller="estimate_dsge")

    # 2. Build neg_log_post + initial vec.
    names = param_names(priors)
    fixed = dict(fixed_params or {})
    failure = _LikelihoodFailure()
    init_vec = _initial_vec_from_dict(initial_params, priors,
                                      caller="estimate_dsge")

    if method == "piecewise_kalman":
        cons_dict = m_constrained_dict or occbin_regimes or {}
        neg_log_post = _make_piecewise_neg_log_posterior(
            data, m_unconstrained, cons_dict, constraints,
            priors, names, fixed, observed_vars, failure=failure,
        )
        neg_log_post_opt = _make_piecewise_neg_log_posterior(
            data, m_unconstrained, cons_dict, constraints,
            priors, names, fixed, observed_vars,
            penalty=_OPT_PENALTY, failure=failure,
        )
    else:
        if observation_eq is None:
            raise ValueError("estimate_dsge: observation_eq is required when method='kalman'.")
        neg_log_post = _make_neg_log_posterior(
            y, observation_eq, priors, names, fixed, failure=failure,
        )
        neg_log_post_opt = _make_neg_log_posterior(
            y, observation_eq, priors, names, fixed,
            penalty=_OPT_PENALTY, failure=failure,
        )
        _check_stochastic_singularity(
            y, observation_eq, _vec_to_dict(init_vec, names, fixed),
        )

    # 3. Mode refinement.
    mode_vec = init_vec.copy()
    converged_mle = False

    f_init = float(neg_log_post_opt(init_vec))
    try:
        # ``mode_compute="lbfgs"`` forwards to exactly this scipy call with
        # exactly these options, so the default path is bit-identical to what
        # it was before the menu existed — which is what keeps the frozen SW07
        # parity fixture valid. See puremacro/dsge/mode.py for the menu.
        opt = _find_mode(
            neg_log_post_opt, init_vec,
            method=mode_compute,
            bounds=param_bounds(priors),
            options={"maxiter": 100, "maxfun": 500 * len(init_vec)},
        )
        opt_x = np.asarray(opt.x, dtype=float)
        opt_f = float(opt.fun)
        moved = float(np.max(np.abs(opt_x - init_vec)))
        improved = f_init - opt_f
        message = str(getattr(opt, "message", ""))

        if np.isfinite(opt_f) and moved > 0.0 and improved > 0.0:
            # Keep the best point found, whether or not scipy calls it a
            # convergence. Hitting `maxiter` after climbing thousands of
            # log-posterior points still leaves a far better mode than the
            # starting values, and discarding it was worth ~2330 points on
            # Smets-Wouters (2007).
            mode_vec = opt_x
            converged_mle = True
            if not opt.success:
                warnings.warn(
                    "estimate_dsge: the mode search stopped before "
                    f"converging ({message!r}) after improving the "
                    f"log-posterior by {improved:.2f}. The best point found "
                    "is returned as `mode`, but it is not a verified "
                    "stationary point -- re-run from it if you need the "
                    "actual mode.",
                    UserWarning,
                )
        elif moved == 0.0:
            # scipy reports success at iteration 0/1 without taking a step
            # whenever the objective is flat or non-finite around the
            # start. Reporting the caller's own initial_params back as
            # "the posterior mode" is the single most misleading thing
            # this function can do, so it is called out explicitly.
            warnings.warn(
                "estimate_dsge: the mode search terminated at the starting "
                f"point without moving ({message!r}); initial_params is "
                "being reported as the mode but has NOT been shown to be "
                "one. Supply a better initial_params, or treat `mode` and "
                "`mode_hessian_inv` as descriptive only.",
                UserWarning,
            )
        else:
            warnings.warn(
                "estimate_dsge: mode optimisation did not improve on "
                f"initial_params ({message!r}); using the "
                "bounds-corrected initial_params as the mode.",
                UserWarning,
            )
    except Exception as e:
        warnings.warn(
            f"estimate_dsge: mode optimisation raised {type(e).__name__}; "
            f"using the bounds-corrected initial_params as the mode.",
            UserWarning,
        )

    # 4. Hessian-based proposal cov (only when mode converged).
    use_hessian = False
    inv_H: np.ndarray
    if converged_mle:
        try:
            # numerical_hessian is inside the try: finite-differencing an
            # exploding posterior can raise OverflowError in plain-float
            # arithmetic on some platforms (not just LinAlgError later).
            H = numerical_hessian(neg_log_post, mode_vec, h=1e-4)
            if not np.all(np.isfinite(H)):
                # Almost always a mode sitting on a prior bound, where the
                # central-difference stencil steps outside the support.
                at_bound = [
                    nm for nm, v, (lo, hi) in
                    zip(names, mode_vec, param_bounds(priors))
                    if min(abs(v - lo), abs(hi - v)) <= 1e-4 * max(abs(v), 1.0)
                ]
                raise np.linalg.LinAlgError(
                    "numerical Hessian is not finite at the reported mode"
                    + (f"; parameters on a prior bound: {at_bound}"
                       if at_bound else "")
                )
            eig_H = np.linalg.eigvalsh(0.5 * (H + H.T))
            n_bad = int(np.sum(eig_H <= 0.0))
            if n_bad:
                warnings.warn(
                    f"estimate_dsge: the numerical Hessian at the reported "
                    f"mode has {n_bad} of {len(eig_H)} non-positive "
                    f"eigenvalues (min {eig_H.min():.3e}), so that point is "
                    "NOT a local maximum of the log-posterior. The proposal "
                    "covariance is built from an eigenvalue-floored repair "
                    "of it; mode_hessian_inv must not be read as a Laplace "
                    "posterior covariance.",
                    UserWarning,
                )
            # Floor the curvature at 1/max(prior variance) so that no
            # direction of the proposal is ever more diffuse than the
            # widest prior. Without it a repaired flat/negative direction
            # inverts to a variance of order 1e10 and the chain never
            # accepts anything.
            prior_var_max = float(np.max(
                [max(prior_stds(priors)[n] ** 2, 1e-12) for n in names]
            ))
            H_pd = _nearest_pd(H, floor_abs=1.0 / prior_var_max)
            inv_H = np.linalg.inv(H_pd)
            inv_H = 0.5 * (inv_H + inv_H.T)
            np.linalg.cholesky(inv_H)
            use_hessian = True
        except (np.linalg.LinAlgError, RuntimeError, ValueError,
                OverflowError, FloatingPointError) as e:
            warnings.warn(
                f"estimate_dsge: Hessian-based proposal failed "
                f"({type(e).__name__}: {e}); falling back to "
                "diag(prior_stds**2).",
                UserWarning,
            )
    if not use_hessian:
        stds = np.array([prior_stds(priors)[n] for n in names])
        inv_H = np.diag(stds ** 2)

    if method == "nuts":
        from puremacro.dsge.nuts import nuts_sample

        # Build target evaluator returning log posterior and analytical gradient
        if model_template is not None:
            from puremacro.dsge._gradients import log_posterior_and_gradient

            def target_fn(v):
                return log_posterior_and_gradient(
                    v, names, model_template, y, observed_vars, priors, fixed,
                )
        else:
            def target_fn(v):
                lp = float(-neg_log_post(v))
                if not np.isfinite(lp):
                    return -np.inf, np.zeros(len(v))
                h = 1e-5
                g = np.zeros(len(v))
                for i in range(len(v)):
                    v_p = v.copy()
                    v_p[i] += h
                    v_m = v.copy()
                    v_m[i] -= h
                    lp_p = float(-neg_log_post(v_p))
                    lp_m = float(-neg_log_post(v_m))
                    g[i] = (lp_p - lp_m) / (2.0 * h)
                return lp, g

        mode_dict = _vec_to_dict(mode_vec, names, fixed)
        stds = np.array([prior_stds(priors)[n] for n in names])
        init_M_inv = stds ** 2
        if use_hessian and "inv_H" in locals():
            init_M_inv = np.clip(np.diag(inv_H), 1e-6, 1e6)

        return nuts_sample(
            target_fn,
            init_params=mode_vec,
            n_draws=n_draws,
            n_chains=n_chains,
            warmup=burn_in,
            target_accept=kwargs.get("target_accept", 0.80),
            max_tree_depth=kwargs.get("max_tree_depth", 10),
            step_size=kwargs.get("step_size", None),
            adapt_step_size=kwargs.get("adapt_step_size", True),
            adapt_mass_matrix=kwargs.get("adapt_mass_matrix", True),
            mass_matrix_diag=init_M_inv,
            param_names=names,
            seed=seed,
            model_name=model_name,
            mode=mode_dict,
            mode_hessian_inv=inv_H if use_hessian else None,
            data_n_obs=len(data),
            record_warmup=kwargs.get("record_warmup", False),
        )

    # 5. Proposal scaling.
    n_params = len(names)
    c0 = 2.38 / np.sqrt(n_params) if use_hessian else 0.01
    proposal_cov = c0 ** 2 * inv_H

    # 6. Run chains.
    chains_arr = np.empty((n_chains, n_draws, n_params))
    log_post_arr = np.empty((n_chains, n_draws))
    accept_rates = []

    def log_post_fn(vec):
        return -neg_log_post(vec)

    for chain_idx in range(n_chains):
        rng = np.random.default_rng(seed + chain_idx)
        try:
            perturb = rng.multivariate_normal(np.zeros(n_params), 0.0025 * inv_H)
        except np.linalg.LinAlgError:
            perturb = 0.05 * rng.standard_normal(n_params)
        start = mode_vec + perturb
        if not np.isfinite(neg_log_post(start)):
            start = mode_vec.copy()
        if not np.isfinite(neg_log_post(start)):
            start = _find_finite_start(
                neg_log_post, rng, priors, failure=failure,
                caller="estimate_dsge",
            )

        out = random_walk_metropolis(
            log_post_fn, start, proposal_cov, n_draws=n_draws,
            seed=seed + chain_idx, accept_target=0.25, adapt_burnin=burn_in,
        )
        chains_arr[chain_idx] = out["chain"]
        log_post_arr[chain_idx] = out["log_post"]
        accept_rates.append(out["accept_rate"])

        if not (0.10 <= out["accept_rate"] <= 0.50):
            warnings.warn(
                f"estimate_dsge: chain {chain_idx} accept_rate="
                f"{out['accept_rate']:.3f} outside [0.10, 0.50]; "
                f"mixing may be poor.",
                UserWarning,
            )

    mode_dict = _vec_to_dict(mode_vec, names, fixed)

    # The log posterior at the reported mode, for the Laplace marginal
    # likelihood. Taken from the +inf target, not the penalised one the
    # optimiser saw: a finite penalty is a device for the search, not a value.
    lp_mode = -float(neg_log_post(mode_vec))
    if not np.isfinite(lp_mode):
        lp_mode = None

    return DSGEPosteriorResult(
        draws=chains_arr,
        param_names=names,
        log_posterior_trace=log_post_arr,
        accept_rates=tuple(accept_rates),
        mode=mode_dict,
        mode_hessian_inv=inv_H,
        n_burn_in=burn_in,
        data_n_obs=len(data),
        seed=seed,
        model_name=model_name,
        log_post_mode=lp_mode,
    )


_orig_linear_model_estimate = LinearModel.estimate


def _linear_model_estimate_with_method(
    self,
    data,
    *,
    method: str = "kalman",
    m_constrained_dict=None,
    constraints=None,
    occbin_regimes=None,
    priors=None,
    initial_params=None,
    varobs=None,
    fixed_params=None,
    measurement_error=None,
    prefilter=False,
    observation_trends=None,
    ridge=0.0,
    mode_compute="lbfgs",
    n_draws: int = 10_000,
    n_chains: int = 2,
    burn_in: int = 2_000,
    seed: int = 0,
    model_name=None,
    check_identification: bool | str = False,
    **kwargs,
):
    """Bayesian estimation supporting method='kalman' and method='piecewise_kalman'."""
    if method == "piecewise_kalman":
        from ._estimated_params import EstimatedParams
        from .priors import ensure_prior

        spec_source = self._estimated_params if priors is None else priors
        if spec_source is None:
            raise ModelError("estimate() needs priors: pass priors=... explicitly.")

        if isinstance(spec_source, EstimatedParams):
            prior_dict = spec_source.priors()
            initial = (
                dict(initial_params)
                if initial_params is not None
                else spec_source.initial_params()
            )
        else:
            prior_dict = {k: ensure_prior(v) for k, v in dict(spec_source).items()}
            initial = (
                dict(initial_params)
                if initial_params is not None
                else {
                    k: float(
                        getattr(
                            pr,
                            "start",
                            getattr(
                                pr,
                                "mean",
                                (pr.lb + pr.ub) / 2.0
                                if math.isfinite(getattr(pr, "lb", -math.inf))
                                and math.isfinite(getattr(pr, "ub", math.inf))
                                else 0.0,
                            ),
                        )
                    )
                    for k, pr in prior_dict.items()
                }
            )

        obs = list(varobs) if varobs is not None else (
            list(self._varobs) if getattr(self, "_varobs", None) else list(data.columns)
        )

        return estimate_dsge(
            data,
            method="piecewise_kalman",
            m_unconstrained=self,
            m_constrained_dict=m_constrained_dict or occbin_regimes,
            constraints=constraints,
            priors=prior_dict,
            observed_vars=obs,
            initial_params=initial,
            fixed_params=fixed_params,
            model_name=model_name or "mod",
            mode_compute=mode_compute,
            n_draws=n_draws,
            n_chains=n_chains,
            burn_in=burn_in,
            seed=seed,
            **kwargs,
        )

    return _orig_linear_model_estimate(
        self,
        data,
        priors=priors,
        varobs=varobs,
        fixed_params=fixed_params,
        measurement_error=measurement_error,
        prefilter=prefilter,
        observation_trends=observation_trends,
        ridge=ridge,
        mode_compute=mode_compute,
        n_draws=n_draws,
        n_chains=n_chains,
        burn_in=burn_in,
        seed=seed,
        model_name=model_name,
        check_identification=check_identification,
        method=method,
        **kwargs,
    )


LinearModel.estimate = _linear_model_estimate_with_method


__all__ = [
    "estimate_dsge",
    "piecewise_kalman_filter",
    "PiecewiseKalmanResult",
]

