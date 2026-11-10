"""Fixed-budget simulated moments with explicit Monte Carlo uncertainty.

Covariances are on the estimator scale. Independent joint simulation replicates
estimate Cov(their average), not the long-run variance of a single observation.
Local normal inference additionally requires smooth, correctly specified,
regularly identified moments and negligible simulation-statistic bias.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.stats import norm

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from .bridge import MomentTargets, _array, _factor, _freeze, _labels, _matrix, fit_structural

__all__ = ["SimulationPlan", "MomentReplication", "SimulatedMoments", "SimulationMomentError",
           "SMMFitResult", "simulate_moments", "moment_error_covariance", "fit_simulated_moments"]


def _snapshot(value: Any) -> np.ndarray:
    """Own numeric storage with a genuinely read-only backing buffer."""
    a = np.asarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _units(value: str | Sequence[str], n: int) -> tuple[str, ...]:
    out = (value,) * n if isinstance(value, str) else tuple(value)
    if len(out) != n or any(not isinstance(v, str) or not v.strip() for v in out):
        raise ValueError("units must provide a nonempty string for every moment")
    return out


@dataclass(frozen=True)
class SimulationPlan:
    """Immutable random streams and simulation budget; seed is always explicit.

    Replication i uses PCG64(SeedSequence([i, stream, role_code, seed])). Role
    codes are fit=0, pilot=1, validation=2. Budget is absent from this key, so
    extending the replication count preserves existing streams. The callback
    must implement ``periods`` and ``burn_in``; the plan cannot inspect its
    internal simulation. Record source/environment separately from ``plan_id``.
    """

    seed: int
    replications: int
    periods: int
    burn_in: int = 0
    stream: int = 0
    role: str = "fit"

    def __post_init__(self) -> None:
        for name, minimum in (("seed", 0), ("replications", 2), ("periods", 1),
                              ("burn_in", 0), ("stream", 0)):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))
        if self.role not in ("fit", "pilot", "validation"):
            raise ValueError("role must be fit, pilot or validation")

    def rng(self, replication: int) -> np.random.Generator:
        """Fresh generator for an index, independent of evaluation order."""
        i = _integer(replication, "replication", 0)
        if i >= self.replications:
            raise ValueError("replication index exceeds the plan budget")
        role = {"fit": 0, "pilot": 1, "validation": 2}[self.role]
        sequence = np.random.SeedSequence([i, self.stream, role, self.seed])
        return np.random.Generator(np.random.PCG64(sequence))

    def to_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "stream": self.stream, "role": self.role,
                "replications": self.replications, "periods": self.periods,
                "burn_in": self.burn_in, "generator": "PCG64",
                "stream_schema": "replication-stream-role-seed-v1", "numpy_version": np.__version__}

    @property
    def plan_id(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class MomentReplication:
    """One joint vector, optionally with effective observation counts/status.

    Nonfinite values, zero counts and statuses other than ``ok`` are retained
    as invalid simulation evidence. Malformed shapes/labels/counts raise errors.
    """

    values: np.ndarray
    counts: np.ndarray | None = None
    status: str = "ok"
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if np.iscomplexobj(self.values):
            raise ValueError("replication values must be real")
        a = np.asarray(self.values, dtype=float)
        if a.ndim != 1 or not len(a):
            raise ValueError("replication values must be a nonempty vector")
        labels = self.labels
        if isinstance(self.values, pd.Series):
            if labels is not None and tuple(self.values.index) != tuple(labels):
                raise ValueError("replication labels must match Series order")
            labels = tuple(self.values.index)
        if labels is not None:
            labels = _labels(labels, len(a))
        if not isinstance(self.status, str) or not self.status.strip():
            raise ValueError("status must be a nonempty string")
        if self.counts is not None:
            counts = _array(self.counts, "counts", 1)
            if (counts.shape != a.shape or np.any(counts < 0) or
                    np.any(counts != np.floor(counts)) or np.asarray(self.counts).dtype.kind == "b"):
                raise ValueError("counts must be nonnegative integers, one per moment")
            object.__setattr__(self, "counts", _snapshot(counts))
        object.__setattr__(self, "values", _snapshot(a))
        object.__setattr__(self, "labels", labels)


@dataclass(frozen=True)
class SimulatedMoments:
    """All joint replicates; any invalid row invalidates the entire aggregate.

    ``covariance`` = sample covariance of replicate vectors / replications.
    There is no further division by periods. Counts absent from callbacks are
    stored as NaN, meaning unknown rather than zero. No failed rows are dropped.
    """

    theta: np.ndarray
    plan: SimulationPlan
    labels: tuple[str, ...]
    units: tuple[str, ...]
    definition_id: str
    replicates: np.ndarray
    counts: np.ndarray
    statuses: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.plan, SimulationPlan):
            raise ValueError("plan must be SimulationPlan")
        theta = _array(self.theta, "theta", 1)
        if not len(theta):
            raise ValueError("theta must be nonempty")
        labels = _labels(self.labels, len(self.labels))
        if not labels:
            raise ValueError("labels must be nonempty")
        shape = (self.plan.replications, len(labels))
        if np.iscomplexobj(self.replicates) or np.iscomplexobj(self.counts):
            raise ValueError("replicates and counts must be real")
        draws, counts = np.asarray(self.replicates, dtype=float), np.asarray(self.counts, dtype=float)
        if draws.shape != shape or counts.shape != shape:
            raise ValueError(f"replicates and counts must have shape {shape}")
        specified = counts[~np.isnan(counts)]
        if np.any(~np.isfinite(specified)) or np.any(specified < 0) or np.any(specified != np.floor(specified)):
            raise ValueError("counts must be nonnegative integers or NaN for unspecified counts")
        if len(self.statuses) != shape[0] or any(not isinstance(s, str) or not s.strip() for s in self.statuses):
            raise ValueError("statuses must contain one nonempty string per replication")
        if not isinstance(self.definition_id, str) or not self.definition_id.strip():
            raise ValueError("definition_id must be a nonempty string describing moment conventions")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        for name, value in (("theta", theta), ("replicates", draws), ("counts", counts)):
            object.__setattr__(self, name, _snapshot(value))
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "units", _units(self.units, len(labels)))
        object.__setattr__(self, "statuses", tuple(self.statuses))
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    @property
    def valid_replicates(self) -> np.ndarray:
        return _snapshot(np.all(np.isfinite(self.replicates), axis=1) &
                         np.all(np.isnan(self.counts) | (self.counts > 0), axis=1) &
                         np.array([s == "ok" for s in self.statuses]))

    def _statistics(self) -> tuple[np.ndarray, np.ndarray]:
        q = len(self.labels)
        if np.all(self.valid_replicates):
            with np.errstate(over="ignore", invalid="ignore"):
                mean = self.replicates.mean(axis=0)
                covariance = np.atleast_2d(np.cov(self.replicates, rowvar=False, ddof=1))
            if np.all(np.isfinite(mean)) and np.all(np.isfinite(covariance)):
                return mean, covariance
        return np.full(q, np.nan), np.full((q, q), np.nan)

    @property
    def valid(self) -> bool:
        return bool(np.all(np.isfinite(self.mean)))

    @property
    def mean(self) -> np.ndarray:
        return _snapshot(self._statistics()[0])

    @property
    def replicate_covariance(self) -> np.ndarray:
        return _snapshot(self._statistics()[1])

    @property
    def covariance(self) -> np.ndarray:
        return _snapshot(self.replicate_covariance / self.plan.replications)

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame({"label": self.labels, "estimate": self.mean,
                              "simulation_mc_se": np.sqrt(np.diag(self.covariance)), "unit": self.units})
        frame.attrs.update({"valid": self.valid, "plan_id": self.plan.plan_id,
                            "definition_id": self.definition_id})
        return frame


class SimulationMomentError(ValueError):
    """Invalid/replay-failing simulation with original evidence attached."""

    def __init__(self, message: str, batch: SimulatedMoments, reference: SimulatedMoments | None = None):
        super().__init__(message)
        self.batch = batch
        self.reference = reference


def simulate_moments(simulator: Callable, theta: Any, plan: SimulationPlan, *,
                     labels: Sequence[str], definition_id: str,
                     units: str | Sequence[str] = "unspecified",
                     metadata: Mapping[str, Any] | None = None) -> SimulatedMoments:
    """Call ``simulator(theta_copy, fresh_rng, plan)`` for every replication.

    Invalid numerical/model results remain in the returned batch. Unexpected
    callback exceptions propagate unchanged; no optimizer penalty is substituted.
    """
    if not callable(simulator) or not isinstance(plan, SimulationPlan):
        raise ValueError("simulator must be callable and plan must be SimulationPlan")
    x = _array(theta, "theta", 1)
    labels = _labels(labels, len(labels))
    units = _units(units, len(labels))
    if not len(x) or not labels:
        raise ValueError("theta and labels must be nonempty")
    if not isinstance(definition_id, str) or not definition_id.strip():
        raise ValueError("definition_id must be a nonempty string describing moment conventions")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    draws, counts, statuses = [], [], []
    for i in range(plan.replications):
        value = simulator(x.copy(), plan.rng(i), plan)
        draw = value if isinstance(value, MomentReplication) else MomentReplication(value)
        if draw.values.shape != (len(labels),):
            raise ValueError("simulator must return one joint value per moment label")
        if draw.labels is not None and draw.labels != labels:
            raise ValueError("simulator labels must exactly match target order")
        draws.append(draw.values)
        counts.append(np.full(len(labels), np.nan) if draw.counts is None else draw.counts)
        statuses.append(draw.status)
    return SimulatedMoments(x, plan, labels, units, definition_id, np.array(draws),
                            np.array(counts), tuple(statuses), metadata if metadata is not None else {})


def _require_valid(batch: SimulatedMoments) -> None:
    if not batch.valid:
        bad = np.flatnonzero(~batch.valid_replicates).tolist()
        raise SimulationMomentError(f"invalid simulated moments: failed replications {bad}; no rows dropped", batch)


def _cross(targets: MomentTargets, assume_independent: bool, cross_covariance: Any) -> np.ndarray:
    if not isinstance(assume_independent, (bool, np.bool_)):
        raise ValueError("assume_independent must be boolean")
    if assume_independent and cross_covariance is not None:
        raise ValueError("declare independence or supply cross_covariance, not both")
    if not assume_independent and cross_covariance is None:
        raise ValueError("declare assume_independent=True or supply data/simulation-mean cross_covariance")
    if cross_covariance is None:
        return np.zeros_like(targets.covariance)
    if isinstance(cross_covariance, pd.DataFrame) and (tuple(cross_covariance.index) != targets.labels or
                                                     tuple(cross_covariance.columns) != targets.labels):
        raise ValueError("cross_covariance labels must exactly match target order")
    cross = _array(cross_covariance, "cross_covariance", 2)
    if cross.shape != targets.covariance.shape:
        raise ValueError("cross_covariance must have one row/column per moment")
    return cross


def _combined(targets: MomentTargets, simulation_covariance: np.ndarray, cross: np.ndarray) -> np.ndarray:
    q = len(targets.labels)
    joint = np.block([[targets.covariance, cross], [cross.T, simulation_covariance]])
    _matrix(joint, tuple(f"joint_{i}" for i in range(2*q)), "joint data/simulation covariance")
    return _matrix(targets.covariance + simulation_covariance - cross - cross.T,
                   targets.labels, "moment-error covariance")


def moment_error_covariance(targets: MomentTargets, simulation: SimulatedMoments, *,
                            assume_independent: bool = False, cross_covariance: Any = None) -> np.ndarray:
    """Return V_data + V_sim - C - C.T, validating the full joint covariance.

    C is Cov(empirical estimator, average simulated estimator), not covariance
    with a single replicate. Independence is an explicit caller assumption.
    """
    if not isinstance(targets, MomentTargets) or not isinstance(simulation, SimulatedMoments):
        raise ValueError("targets and simulation must be MomentTargets and SimulatedMoments")
    if targets.labels != simulation.labels or targets.units != simulation.units:
        raise ValueError("simulation labels and units must exactly match targets")
    _require_valid(simulation)
    cross = _cross(targets, assume_independent, cross_covariance)
    return _snapshot(_combined(targets, simulation.covariance, cross))


@dataclass(frozen=True)
class SMMFitResult:
    """Point fit plus separated sampling, simulation and cross contributions.

    Inference is conditional on local regularity and statistical assumptions;
    optimizer success alone does not make confidence intervals available.
    n_evals counts actual simulation batches, including replays, derivative
    checks, pilot and validation. n_replications_evaluated sums their budgets.
    """

    theta: np.ndarray
    parameter_names: tuple[str, ...]
    covariance: np.ndarray
    standard_errors: np.ndarray
    sampling_covariance: np.ndarray
    simulation_covariance: np.ndarray
    cross_covariance_adjustment: np.ndarray
    moment_covariance: np.ndarray
    weight: np.ndarray
    jacobian: np.ndarray
    influence: np.ndarray
    targets: MomentTargets
    simulation: SimulatedMoments
    pilot: SimulatedMoments | None
    validation: SimulatedMoments | None
    objective: float
    success: bool
    inference_valid: bool
    inference_reasons: tuple[str, ...]
    n_evals: int
    n_replications_evaluated: int
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("theta", "covariance", "standard_errors", "sampling_covariance", "simulation_covariance",
                     "cross_covariance_adjustment", "moment_covariance", "weight", "jacobian", "influence"):
            object.__setattr__(self, name, _snapshot(getattr(self, name)))
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))

    def summary(self, *, ci: float = .95) -> pd.DataFrame:
        if not np.isfinite(ci) or not 0 < ci < 1:
            raise ValueError("ci must lie strictly between zero and one")
        z = norm.ppf((1 + ci) / 2)
        frame = pd.DataFrame({"parameter": self.parameter_names, "estimate": self.theta,
                              "se": self.standard_errors, "ci_lower": self.theta-z*self.standard_errors,
                              "ci_upper": self.theta+z*self.standard_errors})
        frame.attrs.update({"success": self.success, "inference_valid": self.inference_valid,
                            "inference_reasons": self.inference_reasons, "ci": ci, "objective": self.objective})
        return frame

    def to_frame(self, *, ci: float = .95) -> pd.DataFrame:
        return self.summary(ci=ci)

    def to_dataframe(self, *, ci: float = .95) -> pd.DataFrame:
        return self.summary(ci=ci)

    def moment_fit(self, *, validation: bool = False) -> pd.DataFrame:
        batch = self.validation if validation else self.simulation
        if batch is None:
            raise ValueError("no validation plan was supplied")
        frame = self.targets.to_frame().rename(columns={"estimate": "target", "se": "target_se"})
        frame["model"] = batch.mean
        frame["simulation_mc_se"] = np.sqrt(np.diag(batch.covariance))
        frame["residual"] = batch.mean-self.targets.values
        frame["used_in_fit"] = not validation
        frame.attrs.update({"plan_id": batch.plan.plan_id, "definition_id": batch.definition_id,
                            "scope": "descriptive moment comparison"})
        return frame

    def to_markdown(self, *, ci: float = .95, **kwargs: Any) -> str:
        return df_to_markdown(self.summary(ci=ci), index=False, **kwargs)

    def to_latex(self, *, ci: float = .95, **kwargs: Any) -> str:
        return df_to_latex(self.summary(ci=ci), index=False, **kwargs)

    def to_typst(self, *, ci: float = .95, **kwargs: Any) -> str:
        return df_to_typst(self.summary(ci=ci), index=False, **kwargs)


def _check_replay(batch: SimulatedMoments, reference: SimulatedMoments) -> None:
    if (batch.statuses != reference.statuses or
            not np.array_equal(batch.replicates, reference.replicates, equal_nan=True) or
            not np.array_equal(batch.counts, reference.counts, equal_nan=True)):
        raise SimulationMomentError("simulation is not reproducible under the fixed plan", batch, reference)


def _check_derivative(fun: Callable, x: np.ndarray, derivative: np.ndarray,
                      transform: np.ndarray, scale: np.ndarray, bounds: Any,
                      step: float, tolerance: float) -> tuple[list[str], dict[str, Any]]:
    """Compare CRN derivatives at two radii and the left/right slopes."""
    lo = np.full(len(x), -np.inf) if bounds is None else np.array([-np.inf if b[0] is None else b[0] for b in bounds])
    hi = np.full(len(x), np.inf) if bounds is None else np.array([np.inf if b[1] is None else b[1] for b in bounds])
    base = fun(x)
    mismatch, stencil = 0., 0.
    for j in range(len(x)):
        h = min(step*scale[j], (x[j]-lo[j])/2, (hi[j]-x[j])/2)
        if not np.isfinite(h) or h <= 0 or x[j]+h/2 == x[j] or x[j]-h/2 == x[j]:
            return ["unresolved_derivative_stencil"], {"derivative_check": "unrepresentable perturbation"}
        slopes = []
        for radius in (h, h/2):
            plus, minus = x.copy(), x.copy()
            plus[j] += radius
            minus[j] -= radius
            fp, fm = fun(plus), fun(minus)
            slopes.append(((fp-fm)/(plus[j]-minus[j]),
                           (fp-base)/(plus[j]-x[j]), (base-fm)/(x[j]-minus[j])))
        reference = (transform @ slopes[1][0])*scale[j]
        normalizer = max(1., float(np.linalg.norm(reference)))
        error = np.linalg.norm((transform @ (derivative[:, j]-slopes[1][0]))*scale[j])/normalizer
        difference = max(np.linalg.norm((transform @ (slopes[0][0]-slopes[1][0]))*scale[j]),
                         np.linalg.norm((transform @ (slopes[1][1]-slopes[1][2]))*scale[j]))/normalizer
        if not np.isfinite(error) or not np.isfinite(difference):
            return ["unresolved_derivative_stencil"], {"derivative_check": "nonfinite scaled derivative comparison"}
        mismatch, stencil = max(mismatch, float(error)), max(stencil, float(difference))
    reasons = []
    if not np.isfinite(mismatch) or mismatch > tolerance:
        reasons.append("jacobian_mismatch")
    if not np.isfinite(stencil) or stencil > tolerance:
        reasons.append("inconsistent_stencil")
    return reasons, {"derivative_check": "two radii and left/right CRN slopes; diagnostic, not proof of smoothness",
                     "derivative_mismatch": mismatch, "derivative_stencil_difference": stencil,
                     "derivative_step": step, "derivative_tolerance": tolerance}


def fit_simulated_moments(
    simulator: Callable, targets: MomentTargets, theta0: Any, *, plan: SimulationPlan,
    definition_id: str, parameter_names: Sequence[str] | None = None, bounds: Any = None,
    jac: Callable | None = None, weight: Any = None, pilot_plan: SimulationPlan | None = None,
    validation_plan: SimulationPlan | None = None, assume_independent: bool = False,
    cross_covariance: Any = None, assume_smooth: bool = False, max_nfev: int = 2000,
    tol: float = 1e-10, rank_rtol: float = 1e-10, condition_limit: float = 1e8,
    derivative_step: float = 1e-4, derivative_tol: float = 1e-3,
    metadata: Mapping[str, Any] | None = None,
) -> SMMFitResult:
    """Fit locally smooth simulated moments with fixed common random numbers.

    Default weight is empirical precision. An independent pilot estimates a
    weight at theta0 using sample replicate covariance / *fit* replications;
    this weight stays fixed. An explicit SPD weight is also supported. Final
    covariance uses simulation uncertainty at the estimate, not the pilot.

    ``jac`` differentiates the fixed-plan simulated mean, not necessarily the
    population moment. Local normal inference needs ``assume_smooth=True`` and
    passing numerical/stencil checks. They do not establish statistical weak-ID
    robustness or correct specification. Validation streams are descriptive;
    no automatic J p-value or adaptive budget selection is performed.
    """
    if not isinstance(targets, MomentTargets):
        raise ValueError("targets must be MomentTargets")
    for name, value, role in (("plan", plan, "fit"), ("pilot_plan", pilot_plan, "pilot"),
                              ("validation_plan", validation_plan, "validation")):
        if value is None and name != "plan":
            continue
        if not isinstance(value, SimulationPlan) or value.role != role:
            raise ValueError(f"{name} must be a SimulationPlan with role={role}")
    if weight is not None and pilot_plan is not None:
        raise ValueError("supply weight or pilot_plan, not both")
    if pilot_plan is not None and (pilot_plan.periods != plan.periods or pilot_plan.burn_in != plan.burn_in):
        raise ValueError("pilot and fit periods/burn_in must agree")
    if not isinstance(assume_smooth, (bool, np.bool_)):
        raise ValueError("assume_smooth must be boolean")
    for name, numerical_option in (("derivative_step", derivative_step), ("derivative_tol", derivative_tol)):
        if isinstance(numerical_option, (bool, np.bool_)) or not np.isfinite(numerical_option) or numerical_option <= 0:
            raise ValueError(f"{name} must be finite and positive")
    cross = _cross(targets, assume_independent, cross_covariance)
    x0 = _array(theta0, "theta0", 1)
    evaluations = replications = 0
    cache: OrderedDict[bytes, SimulatedMoments] = OrderedDict()

    def run(x: np.ndarray, budget: SimulationPlan) -> SimulatedMoments:
        nonlocal evaluations, replications
        batch = simulate_moments(simulator, x, budget, labels=targets.labels, units=targets.units,
                                  definition_id=definition_id, metadata=metadata)
        evaluations += 1
        replications += budget.replications
        _require_valid(batch)
        return batch

    def checked_run(x: np.ndarray, budget: SimulationPlan) -> SimulatedMoments:
        batch = run(x, budget)
        _check_replay(run(x, budget), batch)
        return batch

    initial = checked_run(x0, plan)
    cache[x0.tobytes()] = initial
    moment_error_covariance(targets, initial, assume_independent=assume_independent, cross_covariance=cross_covariance)
    pilot = None if pilot_plan is None else checked_run(x0, pilot_plan)
    weighting_covariance = targets.covariance if pilot is None else _combined(
        targets, pilot.replicate_covariance/plan.replications, cross)
    optimization_targets = MomentTargets(targets.values, weighting_covariance, targets.labels, targets.units)
    if weight is None:
        scales, factor = _factor(weighting_covariance, "weighting covariance")
        transform = solve_triangular(factor, np.diag(1/scales), lower=True)
        fixed_weight = transform.T @ transform
    else:
        fixed_weight = _matrix(weight, targets.labels, "weight")
        scales, factor = _factor(fixed_weight, "weight")
        transform = factor.T * scales[None, :]

    def mean(x: np.ndarray) -> np.ndarray:
        key = x.tobytes()
        if key not in cache:
            cache[key] = run(x, plan)
            if len(cache) > 8:
                cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key].mean

    fit = fit_structural(mean, optimization_targets, x0, parameter_names=parameter_names,
                         bounds=bounds, jac=jac, weight=weight, max_nfev=max_nfev, tol=tol,
                         rank_rtol=rank_rtol, condition_limit=condition_limit)
    # Bypass the cache so changing global state cannot hide behind a repeated key.
    simulation = checked_run(fit.theta, plan)
    if not np.array_equal(simulation.mean, fit.model_moments):
        reference = cache.get(fit.theta.tobytes())
        raise SimulationMomentError("simulation at the fitted theta is not reproducible", simulation, reference)
    covariance_moments = _combined(targets, simulation.covariance, cross)
    reasons = list(fit.diagnostics["inference_unavailable_reasons"])
    check: dict[str, Any] = {"derivative_check": "not run: smoothness unasserted or numerical regularity failed"}
    if not assume_smooth:
        reasons.append("smoothness_not_asserted")
    elif fit.identification_rank == len(fit.theta) and not np.any(fit.boundary):
        extra, check = _check_derivative(mean, fit.theta, fit.jacobian,
                                         transform/fit.diagnostics["optimizer_residual_scale"],
                                         fit.diagnostics["optimizer_parameter_scale"], bounds,
                                         derivative_step, derivative_tol)
        reasons.extend(extra)
    p, q = len(fit.theta), len(targets.labels)
    influence = np.full((p, q), np.nan)
    covariance, sampling, monte_carlo, adjustment = [np.full((p, p), np.nan) for _ in range(4)]
    se = np.full(p, np.nan)
    if not reasons:
        try:
            scales, factor = _factor(covariance_moments, "moment-error covariance")
        except np.linalg.LinAlgError:
            reasons.append("singular_moment_covariance")
        else:
            u, singular, vt = np.linalg.svd(transform @ fit.jacobian, full_matrices=False)
            influence = ((vt.T/singular) @ u.T) @ transform
            loading = influence @ (scales[:, None]*factor)
            covariance = loading @ loading.T
            sampling = influence @ targets.covariance @ influence.T
            monte_carlo = influence @ simulation.covariance @ influence.T
            adjustment = -influence @ (cross+cross.T) @ influence.T
            if not all(np.all(np.isfinite(a)) for a in (covariance, sampling, monte_carlo, adjustment)):
                reasons.append("nonfinite_parameter_covariance")
                covariance, sampling, monte_carlo, adjustment = [np.full((p, p), np.nan) for _ in range(4)]
                influence[:] = np.nan
            else:
                se = np.sqrt(np.diag(covariance))
    validation = None if validation_plan is None else checked_run(fit.theta, validation_plan)
    diagnostics = dict(fit.diagnostics)
    diagnostics.update(check)
    diagnostics.update({
        "inference_unavailable_reasons": tuple(reasons), "assume_smooth": bool(assume_smooth),
        "assume_independent": bool(assume_independent), "data_simulation_mean_cross_covariance": cross,
        "weighting": "fixed_pilot_precision" if pilot is not None else "fixed_empirical_precision" if weight is None else "fixed_custom",
        "covariance_scale": "empirical estimator plus average of independent simulation replicates",
        "inference_assumptions": "local smooth correct specification; regular interior identification; consistent empirical/simulation/cross covariance and Jacobian; negligible initialization, finite-horizon-statistic and discretization bias",
        "j_test_available": False, "j_test_conditions": "not implemented for simulated moments",
        "validation_scope": "descriptive fresh-stream check at a theta selected using fit streams; no fit-validation p-value",
        "identification_rank": fit.identification_rank, "condition_number": fit.condition_number,
        "singular_values": fit.singular_values, "boundary": fit.boundary, "optimizer_message": fit.message,
        "optimizer_n_evals": fit.n_evals, "plan_id": plan.plan_id, "definition_id": definition_id,
        "replay_scope": "exact joint values/counts/statuses at initial and fitted parameters; pilot/validation also replayed",
        "budget_policy": "fixed throughout optimization; validation is not pooled with fitted moments",
    })
    return SMMFitResult(fit.theta, fit.parameter_names, covariance, se, sampling, monte_carlo,
                        adjustment, covariance_moments, fixed_weight, fit.jacobian, influence,
                        targets, simulation, pilot, validation, fit.objective, fit.success,
                        not reasons, tuple(reasons), evaluations, replications, diagnostics)
