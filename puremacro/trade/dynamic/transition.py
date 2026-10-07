"""Accepted-residual Newton solves for the dynamic sector-capital economy.

Ported from the IO engine's ``dynamic_model/native_solver.py`` and the
``PathProblem`` / ``steady_linearization`` / ``CapitalPreconditioner`` parts
of ``native_numerics.py``, plus the horizon acceptance rules of
``run_native.py`` and the discounted endpoint diagnostic of
``native_verify.py``.

Dates are genuine perfect-foresight dates: lagged installed capital is fixed
at the initial boundary and next-date prices use the next announced policy.
A terminal steady state is solved under the last policy, never guessed or
copied from the baseline; the last schedule is held beyond the horizon. The
stack ``F(z_1..z_T) = 0`` has ``T (N + 2C)`` unknowns and is solved by
matrix-free inexact Newton: the direction comes from
``scipy.sparse.linalg.lgmres`` (``gmres`` fallback) with forcing
``eta = max(1e-5, min(.1, sqrt(||F||_inf)))``, the step is capped at
``.75 / ||d||_inf`` in log units and halved (up to 24 times) until
``F'F`` falls by ``(1 - 1e-4 step)``. MINPACK ``hybr`` is tried first when a
stack has at most 240 unknowns. Every accepted solution meets its residual
tolerance and passes the 17-entry independent certificate at every date
(floor ``1e-8``); a failed solve raises :class:`DynamicSolveError` and never
returns ``converged=True``.

Horizon acceptance (NATIVE_MODEL.md, README.md): "A finite-horizon transition
is an approximation to an infinite-horizon path. Compare successively longer
solutions on their common early dates, and compare consumption-equivalent
welfare with their stationary tails. Failure of a horizon comparison is
reported as such, not relabeled convergence. Numerical acceptance does not
validate the economic parameter choices." The default ``state_gap`` criterion
also requires the terminal retained state to be within ``1e-4`` of the
stationary state. The explicitly weaker ``discounted_wealth`` mode reports
``verified_window_and_welfare``: "it does not certify full terminal-state
convergence or a rigorous utility-tail error bound."
"""
from __future__ import annotations

import dataclasses
import hashlib
import time
from typing import Any, Callable, Sequence

import numpy as np
from scipy import linalg
from scipy.optimize import root
from scipy.sparse.linalg import LinearOperator, gmres, lgmres

from ._results import (DynamicDeterminacyError, DynamicSolveError, DynamicSteadyStateResult,
                       DynamicTransitionResult, EconomicDomainError, HorizonComparisonResult,
                       HorizonLadderResult, _freeze_state)
from .economy import DynamicEconomy, DynamicTariff
from .stationary import _require_certificate, solve_condensed_steady
from .welfare import consumption_equivalent_welfare

# Only the package-level names are public; PathProblem, CapitalPreconditioner,
# steady_linearization and path_residual are importable internals for tests.
__all__ = ["solve_dynamic_steady_state", "solve_dynamic_transition", "compare_horizons",
           "run_horizon_ladder", "discounted_endpoint_capital_value"]

_TERMINAL_CHECKS = ("state_gap", "discounted_wealth")


def _norm(x: np.ndarray) -> float:
    return float(np.max(np.abs(x), initial=0.0))


def _evaluate(fun: Callable, x: np.ndarray) -> np.ndarray:
    value = np.asarray(fun(x), dtype=float).reshape(-1)
    if value.size != x.size or not np.isfinite(value).all():
        raise DynamicSolveError("Nonfinite or nonsquare dynamic residual.")
    return value


# ---------------------------------------------------------------------------
# Cached path problem, steady linearization, capital preconditioner
# ---------------------------------------------------------------------------
class PathProblem:
    """Cached batched residual and exact Jacobian-vector products of a stacked path (internal).

    Dates ``0..T-1`` are unknown; ``initial`` supplies the lag of date 0 and
    ``terminal`` the lead of date ``T-1`` (held under the last policy).
    """

    def __init__(self, model: DynamicEconomy, policies: Sequence[DynamicTariff],
                 initial: np.ndarray, terminal: np.ndarray):
        self.model, self.policies = model, tuple(policies)
        self.initial, self.terminal = initial, terminal
        self.horizon = len(policies)
        self._x = None
        self.states = None

    def evaluate(self, flat: np.ndarray) -> np.ndarray:
        """Stacked residual ``(F_0, ..., F_{T-1})`` at the flat path; date states are cached by value."""
        path = np.asarray(flat).reshape(self.horizon, self.model.n_vars)
        if self._x is None or not np.array_equal(flat, self._x):
            self.states = self.model.states(np.vstack((path, self.terminal)),
                                            np.vstack((self.initial, path)),
                                            self.policies + (self.policies[-1],))
            self._x = np.array(flat, copy=True)
        return np.asarray([self.model.residual_from_states(self.states[t], self.states[t + 1])
                           for t in range(self.horizon)]).ravel()

    def linearize(self, flat: np.ndarray, _residual: np.ndarray | None = None) -> Callable:
        """Exact matrix-free Jacobian-vector product of :meth:`evaluate` at ``flat`` (batched JVPs)."""
        self.evaluate(flat)
        path = np.asarray(flat).reshape(self.horizon, self.model.n_vars).copy()
        states = self.states
        zero = np.zeros(self.model.n_vars)

        def matvec(flat_direction: np.ndarray) -> np.ndarray:
            direction = np.asarray(flat_direction).reshape(path.shape)
            changes = self.model.states_jvp(np.vstack((path, self.terminal)),
                                            np.vstack((self.initial, path)),
                                            np.vstack((direction, zero)),
                                            np.vstack((zero, direction)),
                                            self.policies + (self.policies[-1],), states)
            return np.asarray([self.model.jvp_from_states(states[t], states[t + 1], changes[t], changes[t + 1])
                               for t in range(self.horizon)]).ravel()
        return matvec


def steady_linearization(model: DynamicEconomy, z: np.ndarray, policy: DynamicTariff) -> Callable:
    """Matrix-vector product of the stationary Jacobian (all three dates move together)."""
    state = model.state(z, z, policy)

    def matvec(direction: np.ndarray) -> np.ndarray:
        change = model.state_jvp(z, z, direction, direction, policy, state)
        return model.jvp_from_states(state, state, change, change)
    return matvec


class CapitalPreconditioner:
    """Baseline national blocks and approximate local capital-time tridiagonals.

    Network effects remain in the outer exact Jacobian. This block-triangular
    approximation accounts for the goods response to investment in national
    labour constraints. It requires ``O(T N + N C + C C)`` working memory. It
    is deliberately approximate: acceptance always uses the original
    nonlinear equations and independent accounts.
    """

    def __init__(self, model: DynamicEconomy):
        self.model = model
        d, nc, n = model.p, model.nc, model.n
        self.nc, self.n = nc, n
        baseline = np.zeros(model.n_vars)
        policy = model.zero_policy()
        state = model.state(baseline, baseline, policy)
        budget_w = np.empty((nc, nc))
        budget_c = np.empty((nc, nc))
        labor_c = np.empty((nc, nc))
        self.euler_current = np.empty((n, 2 * nc))
        self.euler_lead = np.empty((n, 2 * nc))
        directions = np.zeros((2 * nc, model.n_vars))
        directions[np.arange(2 * nc), np.arange(2 * nc)] = 1.
        zeros = np.zeros_like(directions)
        changes = model.states_jvp(zeros, zeros, directions, zeros, (policy,) * (2 * nc), [state] * (2 * nc))
        for k in range(2 * nc):
            ds = changes[k]
            dzero = {key: np.zeros_like(value) for key, value in ds.items()}
            here = model.jvp_from_states(state, state, ds, dzero)
            lead = model.jvp_from_states(state, state, dzero, ds)
            self.euler_current[:, k] = here[2 * nc:]
            self.euler_lead[:, k] = lead[2 * nc:]
            if k < nc:
                budget_w[:, k] = here[nc:2 * nc]
            else:
                budget_c[:, k - nc] = here[nc:2 * nc]
                labor_c[:, k - nc] = here[:nc]
        self.budget_c = budget_c
        self.labor_lu = linalg.lu_factor(labor_c, check_finite=False)
        self.budget_lu = linalg.lu_factor(budget_w, check_finite=False)
        response_i = np.empty((nc, nc))
        for c in range(nc):
            response_i[:, c] = np.bincount(
                d.country, weights=d.L0sector / (1 - d.alpha) / d.y0 * model.YI[:, c], minlength=nc) / d.L0
        self.labor_k = response_i[:, d.country] * d.K0
        self.labor_lag = -(1 - d.delta) * self.labor_k
        self.labor_lag[d.country, np.arange(n)] -= d.L0sector * d.alpha / (1 - d.alpha) / d.L0[d.country]
        self.rental_weight = d.beta * (1 / d.beta - 1 + d.delta) / (1 - d.alpha)

    def operator(self, horizon: int, *, stationary: bool = False) -> LinearOperator:
        """Approximate inverse for a ``horizon``-date stack (or the stationary Jacobian) as a LinearOperator.

        A tridiagonal capital-time solve per cell, then the national labour
        and budget blocks, all taken at the baseline state.
        """
        model, d = self.model, self.model.p
        nc, n, phi = self.nc, self.n, model.adjustment_cost
        # A stationary derivative moves both boundary capitals together.
        diagonal = self.rental_weight.copy() if stationary else phi * (1 + d.beta) + self.rental_weight
        if stationary:
            denominators = diagonal[None, :]
            upper = np.zeros((1, n))
        else:
            denominators = np.empty((horizon, n))
            upper = np.empty((horizon, n))
            denominators[0] = diagonal
            upper[0] = -d.beta * phi / denominators[0]
            for t in range(1, horizon):
                denominators[t] = diagonal + phi * upper[t - 1]
                upper[t] = -d.beta * phi / denominators[t]

        def capital_solve(rhs: np.ndarray) -> np.ndarray:
            result = np.empty_like(rhs)
            result[0] = rhs[0] / denominators[0]
            for t in range(1, horizon):
                result[t] = (rhs[t] + phi * result[t - 1]) / denominators[t]
            for t in range(horizon - 2, -1, -1):
                result[t] -= upper[t] * result[t + 1]
            return result

        def apply(flat: np.ndarray) -> np.ndarray:
            r = np.asarray(flat).reshape(horizon, model.n_vars)
            capital = capital_solve(r[:, 2 * nc:])
            labor_rhs = r[:, :nc] - capital @ self.labor_k.T
            if stationary:
                labor_rhs -= capital @ self.labor_lag.T
            elif horizon > 1:
                labor_rhs[1:] -= capital[:-1] @ self.labor_lag.T
            consumption = linalg.lu_solve(self.labor_lu, labor_rhs.T, check_finite=False).T
            wages = linalg.lu_solve(self.budget_lu,
                                    (r[:, nc:2 * nc] - consumption @ self.budget_c.T).T, check_finite=False).T
            return np.c_[wages, consumption, capital].ravel()
        return LinearOperator((horizon * model.n_vars,) * 2, matvec=apply, dtype=float)


def _capital_preconditioner(model: DynamicEconomy) -> CapitalPreconditioner:
    cached = getattr(model, "_capital_preconditioner", None)
    if cached is None:
        cached = CapitalPreconditioner(model)
        model._capital_preconditioner = cached
    return cached


# ---------------------------------------------------------------------------
# Inexact Newton
# ---------------------------------------------------------------------------
def _newton(fun: Callable, start: np.ndarray, *, tol: float, max_iter: int,
            jvp_factory: Callable | None = None, preconditioner_factory: Callable | None = None,
            progress: Callable | None = None, checkpoint: Callable | None = None,
            diagnose: Callable | None = None) -> tuple[np.ndarray, float, int, list[dict[str, Any]]]:
    if not np.isfinite(tol) or tol <= 0 or max_iter < 1:
        raise ValueError("Positive finite tolerance and iteration limit required.")
    x = np.asarray(start, dtype=float).reshape(-1).copy()
    if not np.isfinite(x).all():
        raise ValueError("Initial guess must be finite.")
    f = _evaluate(fun, x)
    history: list[dict[str, Any]] = []
    # MINPACK is a useful independently implemented direct solve for tiny
    # systems. Large stacks never form a dense Jacobian.
    if x.size <= 240 and _norm(f) > tol:
        try:
            small = root(fun, x, method="hybr", options={"xtol": min(tol, 1e-10), "maxfev": max_iter * (x.size + 1)})
            fs = _evaluate(fun, small.x)
        except (DynamicSolveError, EconomicDomainError, ValueError, FloatingPointError, RuntimeError):
            fs = None
        if fs is not None and _norm(fs) < _norm(f):
            x, f = small.x.copy(), fs
            history.append({"method": "MINPACK", "evaluations": int(small.nfev), "residual": _norm(f)})
    for iteration in range(max_iter + 1):
        error = _norm(f)
        if error <= tol:
            return x, error, iteration, history
        if iteration == max_iter:
            break
        if jvp_factory is not None:
            mv = jvp_factory(x, f)
        else:
            def mv(v, _x=x):
                scale = np.linalg.norm(v)
                if scale == 0:
                    return np.zeros_like(v)
                h = 2e-6 / scale
                return (np.asarray(fun(_x + h * v)).ravel() - np.asarray(fun(_x - h * v)).ravel()) / (2 * h)
        operator = LinearOperator((x.size, x.size), matvec=mv, dtype=float)
        preconditioner = None if preconditioner_factory is None else preconditioner_factory(x)
        eta = max(1e-5, min(0.1, np.sqrt(error)))
        count = [0]
        linear_start = time.perf_counter()
        last_update = [linear_start]

        def callback(_):
            count[0] += 1
            now = time.perf_counter()
            if progress is not None and now - last_update[0] >= 10:
                progress(dict(phase="linear_solve", newton_iteration=iteration + 1,
                              krylov_outer_iteration=count[0], seconds=now - linear_start,
                              nonlinear_residual=error))
                last_update[0] = now
        direction, info = lgmres(operator, -f, M=preconditioner, rtol=eta, atol=0.,
                                 maxiter=35, inner_m=40, outer_k=6, callback=callback)
        linear_method = "lgmres"
        if not np.isfinite(direction).all() or info != 0:
            # Fall back to restarted GMRES and keep whichever direction has the
            # smaller true linear residual.
            try:
                alternative, alt_info = gmres(operator, -f, M=preconditioner, rtol=eta, atol=0.,
                                              restart=60, maxiter=40)
            except Exception:  # pragma: no cover - scipy internal failure
                alternative, alt_info = None, -1
            if alternative is not None and np.isfinite(alternative).all():
                if not np.isfinite(direction).all() or _norm(mv(alternative) + f) < _norm(mv(direction) + f):
                    direction, info, linear_method = alternative, alt_info, "gmres"
        if not np.isfinite(direction).all():
            raise DynamicSolveError(f"Nonfinite Newton direction at residual {error:.3g}.",
                                    residual=error, iterate=x, history=history)
        # Bound log movement, never overwrite or clip the returned trajectory.
        step = min(1., 0.75 / max(0.75, _norm(direction)))
        full_step = step
        merit = float(f @ f)
        fc = None
        rejections = {"domain": 0, "merit": 0, "first_domain_error": None, "smallest_trial_merit_ratio": np.inf}
        for _ in range(24):
            candidate = x + step * direction
            try:
                fc = _evaluate(fun, candidate)
                accepted = float(fc @ fc) <= merit * (1. - 1e-4 * step)
                if not accepted:
                    rejections["merit"] += 1
                    rejections["smallest_trial_merit_ratio"] = min(
                        rejections["smallest_trial_merit_ratio"], float(fc @ fc) / max(merit, 1e-300))
            except (DynamicSolveError, EconomicDomainError, ValueError, FloatingPointError, RuntimeError) as exc:
                accepted = False
                rejections["domain"] += 1
                if rejections["first_domain_error"] is None:
                    rejections["first_domain_error"] = str(exc)
            if accepted:
                break
            step *= 0.5
        else:
            location = None
            if diagnose is not None:
                location = diagnose(x, f, x + full_step * direction, rejections)
            detail = "" if not location else " " + str(location.get("message", ""))
            raise DynamicSolveError(
                f"Newton line search failed at residual {error:.3g}; Krylov status {info}.{detail}",
                residual=error, iterate=x, location=location, history=history)
        item = dict(iteration=iteration + 1, residual_before=error, residual=_norm(fc),
                    step=step, krylov_status=int(info), krylov_outer_iterations=count[0],
                    linear_method=linear_method)
        history.append(item)
        if progress is not None:
            progress(item)
        x, f = candidate, fc
        if checkpoint is not None:
            checkpoint(x.copy(), dict(item))
    raise DynamicSolveError(f"Newton iteration limit: residual {_norm(f):.6g}, tolerance {tol:.3g}.",
                            residual=_norm(f), iterate=x, history=history)


def _equation_location(economy: DynamicEconomy, e: int) -> dict[str, Any]:
    """Block, country or cell and a readable name of retained equation ``e`` of one date."""
    countries, nc = economy.country_codes, economy.nc
    if e < nc:
        return dict(block="labor_clearing", country=countries[e], equation=f"labour-market clearing of {countries[e]}")
    if e < 2 * nc:
        c = e - nc
        if c == economy.omitted_budget_index:
            return dict(block="numeraire", country=countries[c], equation="wage numeraire")
        return dict(block="budget", country=countries[c], equation=f"household budget of {countries[c]}")
    j = e - 2 * nc
    label = economy.cell_labels[j]
    return dict(block="capital_euler", cell=label, country=countries[int(economy.p.country[j])],
                equation=f"capital Euler equation of {label}")


def _diagnose_steady(economy: DynamicEconomy) -> Callable:
    def diagnose(x: np.ndarray, f: np.ndarray, candidate: np.ndarray, rejections: dict) -> dict[str, Any]:
        k = int(np.argmax(np.abs(f)))
        info: dict[str, Any] = {"residual": float(f[k]), "method": "full",
                                "rejected_trials_domain": int(rejections.get("domain", 0)),
                                "rejected_trials_merit": int(rejections.get("merit", 0)),
                                "first_domain_error": rejections.get("first_domain_error")}
        info.update(_equation_location(economy, k))
        message = f"Largest stationary residual: {info['equation']} ({info['residual']:.3g})."
        if info["rejected_trials_domain"]:
            message += (f" {info['rejected_trials_domain']} trial steps left the interior domain (first: "
                        f"{info['first_domain_error']}).")
        info["message"] = message
        return info
    return diagnose


def _economy_fingerprint(economy: DynamicEconomy) -> str:
    """sha256 of the calibration arrays, adjustment cost, risk aversion and wage numeraire.

    The omitted redundant budget is left out on purpose: which budget Walras'
    law drops does not change the equilibrium (the certificate checks all).
    """
    digest = getattr(economy, "_calibration_digest", None)
    if digest is None:
        h = hashlib.sha256()
        cal = economy.p
        for item in dataclasses.fields(cal):
            value = getattr(cal, item.name)
            if isinstance(value, dict):
                continue
            h.update(item.name.encode())
            if hasattr(value, "tocsr"):
                csr = value.tocsr()
                for part in (csr.data, csr.indices, csr.indptr):
                    h.update(np.ascontiguousarray(part).tobytes())
                h.update(repr(csr.shape).encode())
            elif isinstance(value, np.ndarray):
                h.update(repr((value.dtype.str, value.shape)).encode())
                h.update(np.ascontiguousarray(value).tobytes())
            else:
                h.update(repr(value).encode())
        digest = h.hexdigest()
        economy._calibration_digest = digest
    closure = repr((digest, economy.adjustment_cost, economy.risk_aversion, economy.numeraire_index))
    return hashlib.sha256(closure.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Steady state
# ---------------------------------------------------------------------------
def solve_dynamic_steady_state(economy: DynamicEconomy, policy: DynamicTariff | None = None, *,
                               method: str = "auto", start: np.ndarray | None = None,
                               tol: float = 1e-9, max_iter: int = 40,
                               progress: Callable | None = None) -> DynamicSteadyStateResult:
    """Certified stationary equilibrium under ``policy`` (baseline when ``None``).

    ``method="condensed"`` solves the exact ``2C`` condensation
    (:mod:`puremacro.trade.dynamic.stationary`); ``"full"`` solves the
    ``N + 2C`` stationary equations with matrix-free Newton (MINPACK first
    for at most 240 unknowns, the stationary capital preconditioner beyond);
    ``"auto"`` picks ``condensed`` when ``n_vars > 240``. Acceptance: residual
    at or below ``tol`` and every certificate entry at or below
    ``max(1e-8, tol)``; otherwise :class:`DynamicSolveError`.

    Both methods are local Newton iterations without a continuation
    fallback, and ``"auto"`` does not retry with the other method on
    failure; a large duty may need ``start`` from the steady state of a
    smaller duty, and a prohibitive one may have no interior steady state.

    Raises
    ------
    DynamicSolveError
        Iteration limit, line-search stall, singular condensed Jacobian, a
        failed certificate, or a failed iterative price or goods solve
        (``residual``, ``iterate`` and ``history`` attached). A line-search
        stall carries ``location`` naming the equation with the largest
        residual (block, country or cell).
    EconomicDomainError
        A ``ValueError`` subclass raised before any Newton step when
        ``start`` is outside the interior domain or when the policy's
        tariff-adjusted price system has no positive solution (prohibitive
        duties).
    """
    if not isinstance(economy, DynamicEconomy):
        raise TypeError("economy must be a DynamicEconomy")
    if method not in ("auto", "condensed", "full"):
        raise ValueError("method must be 'auto', 'condensed' or 'full'")
    if policy is None:
        policy = economy.zero_policy()
    economy._network(policy)
    if method == "condensed" or (method == "auto" and economy.n_vars > 240):
        return solve_condensed_steady(economy, policy, start=start, tol=tol, max_iter=max_iter, progress=progress)
    tic = time.perf_counter()
    initial = np.zeros(economy.n_vars) if start is None else np.asarray(start, dtype=float)
    if initial.shape != (economy.n_vars,):
        raise ValueError("Steady-state start has the wrong shape.")

    def fun(z):
        return economy.equations(z, z, z, policy, policy)

    def jvp(z, f):
        return steady_linearization(economy, z, policy)
    prec = None
    if economy.n_vars > 240:
        operator = _capital_preconditioner(economy).operator(1, stationary=True)
        prec = lambda z: operator  # noqa: E731
    z, error, iterations, history = _newton(fun, initial, tol=tol, max_iter=max_iter,
                                            jvp_factory=jvp, preconditioner_factory=prec, progress=progress,
                                            diagnose=_diagnose_steady(economy))
    cert = economy.certificate(z, z, z, policy, policy)
    _require_certificate(cert, tol)
    z = np.array(z, dtype=float, copy=True)
    z.flags.writeable = False
    return DynamicSteadyStateResult(
        z=z, converged=True, max_residual=error, iterations=iterations, seconds=time.perf_counter() - tic,
        method="full", history=tuple(history), certificate=cert, state=_freeze_state(economy.state(z, z, policy)),
        policy=policy, country_codes=economy.country_codes, cell_labels=economy.cell_labels,
        benchmark=economy.benchmark(), tol=float(tol),
        metadata={"nonlinear_unknowns": economy.n_vars, "capital_stocks": economy.n})


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------
def path_residual(model: DynamicEconomy, path: np.ndarray, policies: Sequence[DynamicTariff],
                  initial: np.ndarray, terminal: np.ndarray) -> np.ndarray:
    """Original equations at every date, including both boundary dates (``(T, n_vars)``)."""
    path = np.asarray(path).reshape(len(policies), model.n_vars)
    return np.asarray([model.equations(
        path[t + 1] if t + 1 < len(path) else terminal,
        path[t], initial if t == 0 else path[t - 1], policies[t],
        policies[t + 1] if t + 1 < len(path) else policies[-1]) for t in range(len(path))])


def discounted_endpoint_capital_value(result: Any, calibration: Any, risk_aversion: float) -> np.ndarray:
    """Last-date discounted capital shadow value in normalized utility units.

    ``D_c = beta**(T-1) (C_c,T-1/C0_c)**(-sigma) / PC_c,T-1 * sum_j in c q_j,T-1 K_j,T / C0_c``.
    Date ``T-1`` purchases the installed stock ``K_T``, hence the exponent
    ``T-1``. This is neither total household wealth (future labour income is
    omitted) nor a bound on unsolved future utility; it is a reported terminal
    diagnostic (NATIVE_MODEL.md). ``result`` is a :class:`DynamicTransitionResult`
    or a dict with ``z, C, PC, q, Knext`` arrays (dates by rows).
    """
    if isinstance(result, DynamicTransitionResult):
        horizon = result.horizon
        last = result.states[-1]
        C, PC, q, Knext = last["C"], last["PC"], last["q"], last["Knext"]
    else:
        arrays = result
        horizon = len(np.atleast_2d(arrays["z"]))
        C, PC, q, Knext = arrays["C"][-1], arrays["PC"][-1], arrays["q"][-1], arrays["Knext"][-1]
    capital_value = np.bincount(calibration.country, weights=q * Knext, minlength=calibration.n_countries)
    return (calibration.beta ** (horizon - 1) * (C / calibration.C0) ** (-risk_aversion)
            * capital_value / (PC * calibration.C0))


def _boundary_vector(value: Any, model: DynamicEconomy, name: str) -> np.ndarray:
    if isinstance(value, DynamicSteadyStateResult):
        value = value.z
    vector = np.array(value, dtype=float, copy=True)
    if vector.shape != (model.n_vars,):
        raise ValueError(f"{name} state has the wrong shape.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} state must be finite.")
    return vector


def _boundary_log_growth(economy: DynamicEconomy) -> float:
    """``log K'/K`` at zero installation (``x = 0``): the interior domain needs a larger value."""
    delta, phi = economy.p.delta, economy.adjustment_cost
    boundary_growth = 1. - delta - .5 * phi * delta ** 2
    return float(np.log(boundary_growth)) if boundary_growth > 0 else float("-inf")


def _installation_rates(economy: DynamicEconomy, path: np.ndarray, initial: np.ndarray,
                        terminal: np.ndarray) -> np.ndarray:
    """(T+1, N) installation rates ``x = I/K`` along a stacked path.

    Row ``T`` is the terminal boundary date, whose installation runs from the
    last transition stock ``K_T`` to the stationary stock. Raises
    :class:`EconomicDomainError` when capital growth leaves the increasing
    branch of the installation function.
    """
    nc, n, delta, phi = economy.nc, economy.n_vars, economy.p.delta, economy.adjustment_cost
    rows = np.vstack((np.asarray(path).reshape(-1, n), terminal))
    lags = np.vstack((initial, rows[:-1]))
    growth = np.expm1(rows[:, 2 * nc:] - lags[:, 2 * nc:])
    if phi == 0:
        return delta + growth
    discriminant = 1. - 2. * phi * growth
    if np.any(discriminant <= 0):
        raise EconomicDomainError("Capital growth exceeds the increasing installation branch")
    return delta + 2. * growth / (1. + np.sqrt(discriminant))


def _default_start(economy: DynamicEconomy, initial: np.ndarray, terminal: np.ndarray,
                   horizon: int) -> tuple[np.ndarray, str]:
    """IO exponential blend (time constant 12 dates), or a linear blend when that leaves the installation domain.

    The linear blend spreads every cell's capital change evenly over the
    ``T + 1`` installation dates (including the terminal boundary date), so
    it minimises the largest per-date change. If even it leaves the interior
    installation domain, no path of this horizon can stay interior for that
    cell, and :class:`DynamicSolveError` says so with the cell and date.
    """
    dates = np.arange(1, horizon + 1)
    candidates = (("exponential_blend", 1 - np.exp(-dates / 12)), ("linear_blend", dates / (horizon + 1)))
    delta, nc = economy.p.delta, economy.nc
    boundary = _boundary_log_growth(economy)
    for name, fraction in candidates:
        start = initial + fraction[:, None] * (terminal - initial)
        try:
            rates = _installation_rates(economy, start, initial, terminal)
        except EconomicDomainError:
            continue
        if np.all(np.isfinite(rates)) and np.all(rates > 0):
            return start, name
    # The linear blend is infeasible: locate its worst cell. Its per-date log
    # capital step is the same at every date, (log K*/K_initial) / (T + 1).
    phi = economy.adjustment_cost
    step = (terminal[2 * nc:] - initial[2 * nc:]) / (horizon + 1)
    growth = np.expm1(step)
    if phi == 0:
        off_branch = np.zeros(step.shape, dtype=bool)
        rate = delta + growth
    else:
        discriminant = 1. - 2. * phi * growth
        off_branch = discriminant <= 0
        rate = delta + 2. * growth / (1. + np.sqrt(np.maximum(discriminant, 0.)))
    if off_branch.any():
        j = int(np.argmax(np.where(off_branch, step, -np.inf)))
    else:
        j = int(np.argmin(rate))
    per_date = float(step[j])
    cell = economy.cell_labels[j]
    direction = ("decline" if per_date < 0 else "rise")
    raise DynamicSolveError(
        f"The default initial guess cannot keep investment interior over {horizon} dates: reaching the "
        f"terminal capital stock of {cell} needs an average log {direction} of {per_date:.4g} per date "
        f"(including the terminal boundary date), outside the interior installation domain (x = I/K > 0, "
        f"log K'/K > {boundary:.4g} with delta={delta:g}, phi={economy.adjustment_cost:g}, on the "
        f"increasing branch). No path of this horizon keeps that cell interior; lengthen the horizon, "
        f"reduce the shock, or pass start explicitly. No irreversibility complementarity is modelled.",
        iterate=start, location={"default_start": "linear_blend", "cell": cell,
                                 "country": economy.country_codes[int(economy.p.country[j])],
                                 "date": 0, "horizon": horizon, "average_log_capital_step": per_date,
                                 "boundary_log_growth": boundary})


def _diagnose_path(economy: DynamicEconomy, policies: Sequence[DynamicTariff],
                   initial: np.ndarray, terminal: np.ndarray) -> Callable:
    labels = economy.cell_labels
    n, delta = economy.n_vars, economy.p.delta
    boundary_log_growth = _boundary_log_growth(economy)

    def _investment_rates(flat: np.ndarray) -> np.ndarray:
        return _installation_rates(economy, flat, initial, terminal)

    def diagnose(x: np.ndarray, f: np.ndarray, candidate: np.ndarray, rejections: dict) -> dict[str, Any]:
        horizon = len(policies)
        k = int(np.argmax(np.abs(f)))
        t, e = divmod(k, n)
        info: dict[str, Any] = {"date": t, "horizon": horizon, "residual": float(f[k]),
                                "boundary_log_growth": boundary_log_growth,
                                "rejected_trials_domain": int(rejections.get("domain", 0)),
                                "rejected_trials_merit": int(rejections.get("merit", 0)),
                                "first_domain_error": rejections.get("first_domain_error")}
        info.update(_equation_location(economy, e))
        nc = economy.nc
        message = f"Largest residual: {info['equation']} at date {t} of {horizon}."

        def date_label(tt: int) -> str:
            return f"date {tt}" if tt < horizon else f"the terminal boundary date {tt}"
        rates = _investment_rates(x)
        t_min, j_min = np.unravel_index(int(np.argmin(rates)), rates.shape)
        info.update(min_investment_rate=float(rates[t_min, j_min]), min_investment_rate_cell=labels[j_min],
                    min_investment_rate_date=int(t_min))
        if info["block"] == "capital_euler":
            info["investment_rate"] = float(rates[t, e - 2 * nc])
        message += (f" On the last accepted iterate the smallest installation rate x=I/K is "
                    f"{rates[t_min, j_min]:.3g} at {labels[j_min]} on {date_label(int(t_min))} (delta={delta:g}).")
        try:
            trial = _investment_rates(candidate)
            tt, jj = np.unravel_index(int(np.argmin(trial)), trial.shape)
            info.update(full_step_min_investment_rate=float(trial[tt, jj]),
                        full_step_min_investment_rate_cell=labels[jj], full_step_min_investment_rate_date=int(tt))
            if trial[tt, jj] <= 0:
                message += (f" The full Newton step drives x to {trial[tt, jj]:.3g} at {labels[jj]} on "
                            f"{date_label(int(tt))}: the interior installation domain (x > 0, log K'/K > "
                            f"{boundary_log_growth:.4g}) binds and no irreversibility complementarity is modelled.")
            else:
                message += (f" The full Newton step keeps x >= {trial[tt, jj]:.3g} ({labels[jj]}, {date_label(int(tt))}); "
                            f"{info['rejected_trials_domain']} trial steps left the domain and "
                            f"{info['rejected_trials_merit']} failed to reduce the residual.")
        except EconomicDomainError as exc:
            info["full_step_domain_error"] = str(exc)
            message += (f" The full Newton step leaves the interior installation domain ({exc}; "
                        f"log K'/K must exceed {boundary_log_growth:.4g}) and no irreversibility "
                        "complementarity is modelled.")
        info["message"] = message
        return info
    return diagnose


def solve_dynamic_transition(economy: DynamicEconomy, policies: Sequence[DynamicTariff], *,
                             terminal: Any | None = None, initial: Any | None = None,
                             start: np.ndarray | None = None, tol: float = 1e-8, max_iter: int = 40,
                             preconditioner: str = "capital", require_determinacy: bool = True,
                             welfare: bool = True, progress: Callable | None = None,
                             checkpoint: Callable | None = None) -> DynamicTransitionResult:
    """Certified perfect-foresight transition under a date-by-date policy schedule.

    Parameters
    ----------
    economy : DynamicEconomy
    policies : sequence of DynamicTariff, length T
        The policy at every date (build with :func:`tariff_path`). The last
        one is held beyond the horizon and defines the terminal steady state.
    terminal, initial : array, DynamicSteadyStateResult or None
        Terminal stationary state (solved under ``policies[-1]`` when
        ``None``; a supplied one must be stationary and certified at ``tol``)
        and inherited initial state (the benchmark ``z = 0`` when ``None``).
    start : array (T, n_vars), optional
        Initial guess. The default is the IO engine's exponential blend from
        ``initial`` to ``terminal`` (time constant 12 dates); when that blend
        leaves the interior installation domain (large shocks), a linear
        blend over the ``T + 1`` installation dates is used instead, and if
        even that is infeasible no path of this horizon can keep investment
        interior and :class:`DynamicSolveError` is raised (see Raises).
        ``metadata["start"]`` records which guess was used.
    tol, max_iter : float, int
        Residual tolerance and Newton iteration limit.
    preconditioner : {"capital", "none"}
        :class:`CapitalPreconditioner` for stacks above 240 unknowns, or none.
    require_determinacy : bool
        Run :func:`stability_report` at the terminal state under the last
        policy (stationarity tolerance ``max(1e-8, tol)``) and raise
        :class:`DynamicDeterminacyError` when the count of stable roots
        differs from the number of predetermined stocks. The dense
        eigenvalue problem has ``n_vars + N`` rows; for very large tables
        pass ``False`` and run the report explicitly.
    welfare : bool
        Attach :func:`consumption_equivalent_welfare` with the terminal
        stationary consumption as tail, benchmark prices and GDP.
    progress, checkpoint : callables, optional
        ``progress(dict)`` after every Newton step and every 10 s of a linear
        solve; ``checkpoint(path, item)`` after every accepted step (the path
        is an unverified guess).

    Raises
    ------
    DynamicSolveError
        Line-search stall (the message names the equation, date and cell and
        reports the installation-domain diagnostics), iteration limit,
        nonstationary terminal, a failed certificate at any date, a failed
        iterative price or goods solve, or a default initial guess that
        cannot be made interior (``location["default_start"]`` names the
        guess and, when the capital change is infeasible at this horizon,
        the cell and the required average per-date log capital step).
    DynamicDeterminacyError
        When ``require_determinacy`` and the terminal state is indeterminate.
    EconomicDomainError
        A ``ValueError`` subclass raised before any Newton step when a
        supplied ``start``, ``initial`` or ``terminal`` lies outside the
        interior domain (negative investment, installation off the
        increasing branch, nonpositive output or prices) or when a policy's
        tariff-adjusted price system has no positive solution.
    """
    tic = time.perf_counter()
    if not isinstance(economy, DynamicEconomy):
        raise TypeError("economy must be a DynamicEconomy")
    policies = tuple(policies)
    if preconditioner not in {"capital", "none"}:
        raise ValueError("Unknown transition preconditioner.")
    if not policies:
        raise ValueError("At least one policy date is required.")
    for policy in policies:
        economy._network(policy)
    initial = np.zeros(economy.n_vars) if initial is None else _boundary_vector(initial, economy, "Initial")
    terminal_result = None
    if terminal is None:
        terminal_result = solve_dynamic_steady_state(economy, policies[-1], tol=min(tol, 1e-10), progress=progress)
        terminal = terminal_result.z.copy()
    else:
        if isinstance(terminal, DynamicSteadyStateResult):
            terminal_result = terminal
        terminal = _boundary_vector(terminal, economy, "Terminal")
        terminal_error = _norm(economy.equations(terminal, terminal, terminal, policies[-1], policies[-1]))
        if terminal_error > tol:
            raise DynamicSolveError(f"Supplied terminal state is not stationary: {terminal_error:.6g}.",
                                    residual=terminal_error)
        _require_certificate(economy.certificate(terminal, terminal, terminal, policies[-1], policies[-1]), tol)
    stability = None
    if require_determinacy:
        from .stability import stability_report
        stability = economy._stability_cache.get(policies[-1].fingerprint)
        if stability is None:
            stability = stability_report(economy, terminal, policies[-1], stationarity_tol=max(1e-8, tol))
            economy._stability_cache[policies[-1].fingerprint] = stability
        if not stability.determinate:
            raise DynamicDeterminacyError(
                "The linearised dynamics at the terminal steady state are not saddle-path determinate: "
                f"{stability.summary()} Pass require_determinacy=False to solve anyway; finite-horizon "
                "paths then carry a terminal gap that grows with the horizon.", stability)
    horizon = len(policies)
    start_kind = "user"
    if start is None:
        start, start_kind = _default_start(economy, initial, terminal, horizon)
    start = np.asarray(start, dtype=float)
    if start.shape != (horizon, economy.n_vars):
        raise ValueError("Transition start has the wrong shape.")
    problem = PathProblem(economy, policies, initial, terminal)
    if start_kind != "user":
        # A solver-built guess outside the domain is the solver's failure, not
        # a user input error: report it as a located DynamicSolveError.
        try:
            problem.evaluate(start.ravel())
        except EconomicDomainError as exc:
            raise DynamicSolveError(
                f"The default initial guess ({start_kind.replace('_', ' ')} from initial to terminal) lies "
                f"outside the interior domain: {exc}. Pass start explicitly or check initial and terminal.",
                iterate=start, location={"default_start": start_kind, "domain_error": str(exc),
                                         "horizon": horizon}) from exc
    prec = None
    if preconditioner != "none" and horizon * economy.n_vars > 240:
        operator = _capital_preconditioner(economy).operator(horizon)
        prec = lambda z: operator  # noqa: E731
    try:
        z, error, iterations, history = _newton(problem.evaluate, start.ravel(), tol=tol, max_iter=max_iter,
                                                jvp_factory=problem.linearize, preconditioner_factory=prec,
                                                progress=progress, checkpoint=checkpoint,
                                                diagnose=_diagnose_path(economy, policies, initial, terminal))
    except DynamicSolveError as exc:
        if exc.location is not None:
            exc.location.setdefault("start", start_kind)
        raise
    path = z.reshape(horizon, economy.n_vars)
    certificate: dict[str, float] = {}
    checked_states = economy.states(np.vstack((path, terminal)), np.vstack((initial, path)),
                                    policies + (policies[-1],))
    states = []
    for t in range(horizon):
        lag = initial if t == 0 else path[t - 1]
        lead = terminal if t + 1 == horizon else path[t + 1]
        next_policy = policies[min(t + 1, horizon - 1)]
        checks = economy.certificate(lead, path[t], lag, policies[t], next_policy,
                                     _state=checked_states[t], _following=checked_states[t + 1])
        try:
            _require_certificate(checks, tol)
        except DynamicSolveError as exc:
            raise DynamicSolveError(f"Certificate failed at date {t}: {exc}", residual=error, iterate=path,
                                    location={"date": t}, history=history, certificate=checks) from exc
        for key, value in checks.items():
            certificate[key] = max(certificate.get(key, 0.), float(value))
        states.append(_freeze_state(checked_states[t]))
    boundary_state = _freeze_state(checked_states[-1])
    terminal_state = terminal_result.state if terminal_result is not None else economy.state(terminal, terminal, policies[-1])
    welfare_result = None
    if welfare:
        welfare_result = consumption_equivalent_welfare(
            np.asarray([s["C"] for s in states]), baseline_consumption=economy.p.C0, beta=economy.p.beta,
            risk_aversion=economy.risk_aversion, terminal_consumption=terminal_state["C"],
            baseline_price=economy.p.PC0, baseline_gdp=economy.p.Y0, country_codes=economy.country_codes)
    path_out = np.array(path, dtype=float, copy=True)
    path_out.flags.writeable = False
    initial_out = np.array(initial, dtype=float, copy=True)
    initial_out.flags.writeable = False
    terminal_out = np.array(terminal, dtype=float, copy=True)
    terminal_out.flags.writeable = False
    # The terminal diagnostic is computed from the accepted states before the
    # frozen result is constructed (validate, then construct; never mutate).
    wealth = discounted_endpoint_capital_value(
        {"z": path, "C": np.asarray([s["C"] for s in states]), "PC": np.asarray([s["PC"] for s in states]),
         "q": np.asarray([s["q"] for s in states]), "Knext": np.asarray([s["Knext"] for s in states])},
        economy.p, economy.risk_aversion)
    wealth = np.array(wealth, dtype=float, copy=True)
    wealth.flags.writeable = False
    return DynamicTransitionResult(
        path=path_out, policies=policies, initial=initial_out, terminal=terminal_out, states=tuple(states),
        boundary_state=boundary_state, converged=True, max_residual=error, iterations=iterations,
        seconds=time.perf_counter() - tic, history=tuple(history), certificate=certificate,
        welfare=welfare_result, terminal_max_log_gap=float(np.max(np.abs(path[-1] - terminal))),
        discounted_endpoint_capital_value=wealth, stability=stability,
        country_codes=economy.country_codes, cell_labels=economy.cell_labels, benchmark=economy.benchmark(),
        tol=float(tol), metadata={"preconditioner": preconditioner, "unknowns": horizon * economy.n_vars,
                                  "policy_labels": tuple(p.label for p in policies),
                                  "terminal_policy": policies[-1].label,
                                  "determinacy_checked": bool(require_determinacy),
                                  "start": start_kind,
                                  "economy_fingerprint": _economy_fingerprint(economy)})


# ---------------------------------------------------------------------------
# Horizon acceptance
# ---------------------------------------------------------------------------
def compare_horizons(short: DynamicTransitionResult, long: DynamicTransitionResult, *, periods: int = 20,
                     tolerance: float = 1e-5, welfare_tolerance: float = 1e-4,
                     terminal_check: str = "state_gap", terminal_tolerance: float = 1e-4,
                     discounted_wealth_tolerance: float = 1e-6) -> HorizonComparisonResult:
    """Compare two horizons of the same experiment (the IO driver's acceptance rule).

    Checks: (1) ``max |z_short_t - z_long_t|`` over the first ``periods``
    dates at or below ``tolerance``; (2) consumption-equivalent welfare
    differing by at most ``welfare_tolerance`` percentage points (both
    results must carry welfare); (3) the terminal check on the long solution:
    ``state_gap`` (default) requires ``max |z_T - z*| <= terminal_tolerance``;
    ``discounted_wealth`` requires the discounted endpoint capital value at or
    below ``discounted_wealth_tolerance`` and yields the explicitly weaker
    status ``verified_window_and_welfare`` (it does not certify terminal-state
    convergence or a utility-tail error bound). Both solutions must come
    from the same economy (``metadata["economy_fingerprint"]``: calibration
    arrays, adjustment cost, risk aversion and wage numeraire, compared when
    both results carry it) and share identical boundaries and the same
    announced policy held beyond the short horizon; otherwise ``ValueError``.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Horizon tolerance must be finite and positive.")
    for name, value in (("welfare_tolerance", welfare_tolerance), ("terminal_tolerance", terminal_tolerance),
                        ("discounted_wealth_tolerance", discounted_wealth_tolerance)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if terminal_check not in _TERMINAL_CHECKS:
        raise ValueError(f"terminal_check must be one of {_TERMINAL_CHECKS}")
    for name, value in (("short", short), ("long", long)):
        if not isinstance(value, DynamicTransitionResult):
            raise TypeError(f"{name} must be a DynamicTransitionResult")
    if periods < 1 or periods > min(len(short.path), len(long.path)):
        raise ValueError("Comparison periods must lie within both trajectories.")
    if len(long.path) <= len(short.path):
        raise ValueError("Second solution must have a longer horizon.")
    if not short.converged or not long.converged:
        raise DynamicSolveError("Cannot compare unaccepted trajectories.")
    if short.path.ndim != 2 or long.path.ndim != 2 or short.path.shape[1] != long.path.shape[1]:
        raise ValueError("Horizon trajectories must have the same state dimensions.")
    economies = (short.metadata.get("economy_fingerprint"), long.metadata.get("economy_fingerprint"))
    if None not in economies and economies[0] != economies[1]:
        raise ValueError("Horizon comparisons require solutions of the same economy (calibration, adjustment "
                         "cost, risk aversion and numeraire).")
    for name in ("initial", "terminal"):
        first, second = getattr(short, name), getattr(long, name)
        if (first is None or second is None or np.shape(first) != np.shape(second)
                or not np.allclose(first, second, rtol=0., atol=1e-9)):
            raise ValueError(f"Horizon comparisons require identical {name} boundaries.")
    if len(short.policies) != len(short.path) or len(long.policies) != len(long.path):
        raise ValueError("Horizon comparisons require complete policy schedules.")
    for t, policy in enumerate(long.policies):
        held_policy = short.policies[min(t, len(short.policies) - 1)]
        if (getattr(policy, "fingerprint", None) is None
                or policy.fingerprint != getattr(held_policy, "fingerprint", None)):
            raise ValueError("Horizon comparisons require the same announced policy, held beyond the short horizon.")
    if short.welfare is None or long.welfare is None:
        raise ValueError("Both solutions need consumption-equivalent welfare (solve with welfare=True).")
    error = _norm(short.path[:periods] - long.path[:periods])
    window_passed = bool(error <= tolerance)
    welfare_difference = _norm(long.welfare.consumption_equivalent_pct - short.welfare.consumption_equivalent_pct)
    welfare_passed = bool(welfare_difference <= welfare_tolerance)
    terminal_gap = float(long.terminal_max_log_gap)
    max_wealth = float(np.max(long.discounted_endpoint_capital_value))
    terminal_passed = bool(terminal_gap <= terminal_tolerance if terminal_check == "state_gap"
                           else max_wealth <= discounted_wealth_tolerance)
    passed = window_passed and welfare_passed and terminal_passed
    status = (("verified" if terminal_check == "state_gap" else "verified_window_and_welfare")
              if passed else "not_verified")
    scope = ("Early comparison window, consumption-equivalent horizon stability, and discounted "
             "terminal-wealth diagnostic; terminal state convergence is reported separately and no "
             "rigorous infinite-tail error bound is claimed."
             if terminal_check == "discounted_wealth" else
             "Early comparison window, consumption-equivalent horizon stability, and terminal state "
             "convergence at the stated tolerances.")
    return HorizonComparisonResult(
        periods=int(periods), short_horizon=len(short.path), long_horizon=len(long.path),
        max_log_difference=error, tolerance=float(tolerance), window_passed=window_passed,
        welfare_max_difference_pp=welfare_difference, welfare_tolerance=float(welfare_tolerance),
        welfare_passed=welfare_passed, terminal_check=terminal_check, terminal_max_log_gap=terminal_gap,
        terminal_tolerance=float(terminal_tolerance), max_discounted_wealth=max_wealth,
        discounted_wealth_tolerance=float(discounted_wealth_tolerance), terminal_passed=terminal_passed,
        passed=bool(passed), status=status, validation_scope=scope)


def run_horizon_ladder(economy: DynamicEconomy, policy_path_factory: Callable[[int], Sequence[DynamicTariff]],
                       horizons: Sequence[int] = (160, 320, 640), *, initial: Any | None = None,
                       terminal: Any | None = None, periods: int = 20, tolerance: float = 1e-5,
                       welfare_tolerance: float = 1e-4, terminal_check: str = "state_gap",
                       terminal_tolerance: float = 1e-4, discounted_wealth_tolerance: float = 1e-6,
                       tol: float = 1e-8, max_iter: int = 40, preconditioner: str = "capital",
                       require_determinacy: bool = True, progress: Callable | None = None) -> HorizonLadderResult:
    """Solve increasing horizons, warm-starting each from the previous path, until one is accepted.

    ``policy_path_factory(horizon)`` returns the policy schedule for that
    horizon (for example ``lambda T: tariff_path(zero, shock, horizon=T,
    announcement=2)``). The terminal state is solved once under the last
    policy of the first schedule and reused, as in the IO driver; every
    schedule must end with that same policy. Each new horizon starts from the
    previous path extended with the terminal state. The ladder stops at the
    first passing :func:`compare_horizons`; ``status`` is ``verified``,
    ``verified_window_and_welfare`` or ``solved_horizon_not_validated``.
    """
    horizons = tuple(int(h) for h in horizons)
    if len(horizons) < 2 or any(b <= a for a, b in zip(horizons, horizons[1:])):
        raise ValueError("horizons must be at least two strictly increasing integers")
    if terminal_check not in _TERMINAL_CHECKS:
        raise ValueError(f"terminal_check must be one of {_TERMINAL_CHECKS}")
    solutions: list[DynamicTransitionResult] = []
    comparisons: list[HorizonComparisonResult] = []
    previous = None
    terminal_vector = None
    accepted = None
    for horizon in horizons:
        policies = tuple(policy_path_factory(horizon))
        if len(policies) != horizon:
            raise ValueError("policy_path_factory must return exactly one policy per date")
        if terminal_vector is None:
            if terminal is None:
                terminal_vector = solve_dynamic_steady_state(economy, policies[-1], tol=min(tol, 1e-10),
                                                             progress=progress)
            else:
                terminal_vector = terminal
        elif policies[-1].fingerprint != terminal_fingerprint:
            raise ValueError("Every horizon must end with the same terminal policy")
        terminal_fingerprint = policies[-1].fingerprint
        start = None
        if previous is not None:
            z_term = terminal_vector.z if isinstance(terminal_vector, DynamicSteadyStateResult) else np.asarray(terminal_vector)
            start = np.vstack((previous.path, np.tile(z_term, (horizon - len(previous.path), 1))))
        solution = solve_dynamic_transition(economy, policies, terminal=terminal_vector, initial=initial, start=start,
                                            tol=tol, max_iter=max_iter, preconditioner=preconditioner,
                                            require_determinacy=require_determinacy, welfare=True, progress=progress)
        solutions.append(solution)
        if previous is not None:
            comparison = compare_horizons(previous, solution, periods=min(periods, len(previous.path)),
                                          tolerance=tolerance, welfare_tolerance=welfare_tolerance,
                                          terminal_check=terminal_check, terminal_tolerance=terminal_tolerance,
                                          discounted_wealth_tolerance=discounted_wealth_tolerance)
            comparisons.append(comparison)
            if progress is not None:
                progress({"phase": "horizon_comparison", "short": comparison.short_horizon,
                          "long": comparison.long_horizon, "passed": comparison.passed})
            if comparison.passed:
                accepted = horizon
                break
        previous = solution
    if accepted is not None:
        status = "verified" if terminal_check == "state_gap" else "verified_window_and_welfare"
    else:
        status = "solved_horizon_not_validated"
    scope = (comparisons[-1].validation_scope if comparisons else "no horizon comparison was performed")
    if accepted is None:
        scope += " Solved finite-horizon paths; extend the horizon before interpreting lifetime welfare."
    return HorizonLadderResult(horizons=tuple(s.horizon for s in solutions), solutions=tuple(solutions),
                               comparisons=tuple(comparisons), accepted_horizon=accepted, status=status,
                               validation_scope=scope,
                               metadata={"requested_horizons": horizons, "terminal_check": terminal_check})
