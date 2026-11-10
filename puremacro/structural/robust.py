"""Point-null moment tests and scalar confidence sets without Jacobian inversion.

The statistic uses the full covariance of the moment error at the hypothesized
parameter and q degrees of freedom. It does not subtract a fitted objective or
use q-p. Exact calibration requires a Gaussian error with known covariance;
otherwise an appropriate CLT and consistent covariance are caller assumptions.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.stats import chi2

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from .bridge import MomentTargets, _array, _factor, _freeze, _labels, _matrix
from .simulation import SimulatedMoments, SimulationMomentError, _snapshot, _units, moment_error_covariance

__all__ = ["MomentTestResult", "MomentConfidenceSet", "MomentTestGridResult", "QuadraticMomentCurve",
           "moment_restriction_test", "simulated_moment_test", "invert_quadratic_moments", "evaluate_moment_grid"]


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        out = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} exceeds numerical range") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite real scalar")
    return out


def _settings(alpha: float, calibration: str) -> float:
    alpha = _finite_scalar(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if calibration not in ("asymptotic", "gaussian_known"):
        raise ValueError("calibration must be asymptotic or gaussian_known")
    return float(alpha)


def _whitener(covariance: np.ndarray) -> np.ndarray:
    scales, factor = _factor(covariance, "moment-error covariance")
    return solve_triangular(factor, np.diag(1/scales), lower=True)


class _Reports:
    def to_frame(self) -> pd.DataFrame:
        raise NotImplementedError

    def to_dataframe(self) -> pd.DataFrame:
        return self.to_frame()

    def _report_frame(self) -> pd.DataFrame:
        return self.to_frame()

    def to_markdown(self, **kwargs: Any) -> str:
        return df_to_markdown(self._report_frame(), index=False, **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return df_to_latex(self._report_frame(), index=False, **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return df_to_typst(self._report_frame(), index=False, **kwargs)


@dataclass(frozen=True)
class MomentTestResult(_Reports):
    """Point-null S test; invalid evaluations have accepted=None, never False.

    Numerical validity does not verify the statistical assumptions. The null
    jointly maintains the parameter and the model's moment restrictions.
    """

    statistic: float
    pvalue: float
    critical_value: float
    df: int
    alpha: float
    calibration: str
    status: str
    accepted: bool | None
    error: np.ndarray
    covariance: np.ndarray
    labels: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    simulation: SimulatedMoments | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "error", _snapshot(self.error))
        object.__setattr__(self, "covariance", _snapshot(self.covariance))
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    @property
    def valid(self) -> bool:
        return self.status == "valid"

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"statistic": [self.statistic], "pvalue": [self.pvalue],
                             "critical_value": [self.critical_value], "df": [self.df],
                             "alpha": [self.alpha], "accepted": pd.array([self.accepted], dtype="boolean"),
                             "status": [self.status], "calibration": [self.calibration]})


def moment_restriction_test(targets: MomentTargets, model_moments: Any, *, covariance: Any = None,
                            alpha: float = .05, calibration: str = "asymptotic",
                            metadata: Mapping[str, Any] | None = None) -> MomentTestResult:
    """Test empirical-minus-model moments at one fully specified hypothesis.

    ``covariance`` is the *total moment-error covariance at this hypothesis*,
    not an arbitrary optimization weight. It defaults to targets.covariance
    for deterministic moments. No sample-size multiplier is applied. Singular
    covariance/nonfinite moments produce unresolved results without a ridge.
    Wrong shapes/labels or non-PSD matrices are input errors.
    """
    if not isinstance(targets, MomentTargets):
        raise ValueError("targets must be MomentTargets")
    alpha = _settings(alpha, calibration)
    if isinstance(model_moments, pd.Series) and tuple(model_moments.index) != targets.labels:
        raise ValueError("model labels must exactly match target order")
    if np.iscomplexobj(model_moments):
        raise ValueError("model moments must be real")
    model = np.asarray(model_moments, dtype=float)
    if model.shape != targets.values.shape:
        raise ValueError("model must provide one value per target label")
    cov = targets.covariance if covariance is None else _matrix(covariance, targets.labels, "moment-error covariance")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    meta = dict(metadata or {})
    meta.update({"null": "hypothesized parameter and maintained moment restrictions",
                 "covariance_scale": "full moment-error estimator covariance at the null",
                 "identification": "no Jacobian rank assumption or fitted-objective subtraction",
                 "calibration_assumptions": "exact Gaussian null error with known SPD covariance" if calibration == "gaussian_known" else
                 "null moment CLT and consistent SPD covariance; uniform validity requires corresponding uniform assumptions"})
    q = len(targets.labels)
    critical = float(chi2.isf(alpha, q))
    statistic = pvalue = float("nan")
    accepted = None
    with np.errstate(over="ignore", invalid="ignore"):
        error = targets.values-model
    status = "nonfinite_moments"
    if np.all(np.isfinite(error)):
        try:
            transform = _whitener(cov)
        except np.linalg.LinAlgError as exc:
            status = "unresolved_covariance"
            meta["numerical_failure"] = str(exc)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                whitened = transform @ error
                statistic = float(whitened @ whitened)
            if np.isfinite(statistic):
                pvalue = float(chi2.sf(statistic, q))
                accepted = bool(statistic <= critical)
                status = "valid"
            else:
                statistic = float("nan")
                status = "unresolved_numeric_range"
    return MomentTestResult(statistic, pvalue, critical, q, alpha, calibration, status,
                             accepted, error, cov, targets.labels, meta)


def simulated_moment_test(targets: MomentTargets, simulation: SimulatedMoments, *,
                          assume_independent: bool = False, cross_covariance: Any = None,
                          alpha: float = .05) -> MomentTestResult:
    """Point test using the batch's covariance at its retained theta.

    Finite-replication estimated covariance permits only asymptotic calibration.
    Invalid simulations remain attached to an unresolved result. The caller
    must generate a new batch for each hypothesized parameter; a pilot/fitted
    covariance is not a substitute for covariance at that hypothesis.
    """
    alpha = _settings(alpha, "asymptotic")
    try:
        covariance = moment_error_covariance(targets, simulation, assume_independent=assume_independent,
                                              cross_covariance=cross_covariance)
    except SimulationMomentError as exc:
        q = len(targets.labels)
        return MomentTestResult(float("nan"), float("nan"), float(chi2.isf(alpha, q)),
                                 q, alpha, "asymptotic", "invalid_simulation", None,
                                 np.full(q, np.nan), np.full((q, q), np.nan), targets.labels,
                                 {"reason": str(exc), "plan_id": simulation.plan.plan_id,
                                  "definition_id": simulation.definition_id}, simulation)
    result = moment_restriction_test(targets, simulation.mean, covariance=covariance, alpha=alpha,
                                     metadata={"plan_id": simulation.plan.plan_id,
                                               "definition_id": simulation.definition_id,
                                               "theta": simulation.theta,
                                               "simulation_scope": "estimated simulation-mean covariance; negligible moment bias required"})
    return replace(result, simulation=simulation)


@dataclass(frozen=True)
class QuadraticMomentCurve:
    """Declared scalar curve: offset + loading * (linear*z + quadratic*z**2).

    z=(theta-center)/scale. One common index loads onto q moments. This is not
    an arbitrary vector quadratic map; no other nuisance parameter is estimated.
    """

    offset: np.ndarray
    loading: np.ndarray
    labels: tuple[str, ...]
    definition_id: str
    linear: float = 1.
    quadratic: float = 0.
    center: float = 0.
    scale: float = 1.
    parameter_name: str = "theta"
    units: tuple[str, ...] | str = "unspecified"

    def __post_init__(self) -> None:
        offset, loading = _array(self.offset, "offset", 1), _array(self.loading, "loading", 1)
        if not len(offset) or offset.shape != loading.shape:
            raise ValueError("offset and loading must be nonempty vectors of the same shape")
        labels = _labels(self.labels, len(offset))
        for name in ("offset", "loading"):
            value = getattr(self, name)
            if isinstance(value, pd.Series) and tuple(value.index) != labels:
                raise ValueError(f"{name} labels must match moment order")
        for name in ("linear", "quadratic", "center", "scale"):
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        for name in ("definition_id", "parameter_name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        object.__setattr__(self, "offset", _snapshot(offset))
        object.__setattr__(self, "loading", _snapshot(loading))
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "units", _units(self.units, len(offset)))

    def __call__(self, theta: float) -> np.ndarray:
        theta = _finite_scalar(theta, "theta")
        if np.all(self.loading == 0) or (self.linear == 0 and self.quadratic == 0):
            return self.offset.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            z = (np.float64(theta)-self.center)/self.scale
            value = self.offset+self.loading*((self.quadratic*z+self.linear)*z)
        if not np.all(np.isfinite(value)):
            raise ValueError("curve evaluation exceeds numerical range; rescale the parameter")
        return value


@dataclass(frozen=True)
class MomentConfidenceSet(_Reports):
    """Continuous inverse image for the declared scalar curve and covariance.

    Global components and their intersection with a search domain are separate.
    ``resolved`` is numerical resolution, not proof of the statistical assumptions.
    An unresolved result has no usable membership/emptiness assertion.
    """

    intervals: np.ndarray
    global_intervals: np.ndarray
    domain: tuple[float, float]
    boundary_contact: tuple[bool, bool]
    truncated: bool
    global_geometry: str
    status: str
    alpha: float
    df: int
    critical_value: float
    calibration: str
    curve: QuadraticMomentCurve
    targets: MomentTargets
    covariance: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("intervals", "global_intervals", "covariance"):
            object.__setattr__(self, name, _snapshot(getattr(self, name)))
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"

    def contains(self, theta: Any, *, globally: bool = False) -> Any:
        if not self.resolved:
            raise ValueError("confidence set is unresolved; membership is unavailable")
        if np.iscomplexobj(theta):
            raise ValueError("theta must be real")
        x = np.asarray(theta, dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError("theta must be finite")
        accepted = np.zeros(x.shape, dtype=bool)
        for lower, upper in self.global_intervals if globally else self.intervals:
            accepted |= (x >= lower) & (x <= upper)
        return bool(accepted) if accepted.ndim == 0 else accepted

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.intervals, columns=["lower", "upper"])
        frame.insert(0, "component", np.arange(len(frame)))
        frame["parameter"] = self.curve.parameter_name
        frame["lower_source"] = ["unbounded" if np.isneginf(a) else "search_boundary" if a == self.domain[0] else "test_boundary" for a in self.intervals[:, 0]]
        frame["upper_source"] = ["unbounded" if np.isposinf(b) else "search_boundary" if b == self.domain[1] else "test_boundary" for b in self.intervals[:, 1]]
        frame.attrs.update({"status": self.status, "alpha": self.alpha, "calibration": self.calibration,
                            "truncated": self.truncated, "global_geometry": self.global_geometry,
                            "definition_id": self.curve.definition_id})
        return frame

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({"parameter": [self.curve.parameter_name], "status": [self.status],
                             "components_in_domain": [len(self.intervals) if self.resolved else None],
                             "global_geometry": [self.global_geometry], "truncated": [self.truncated],
                             "alpha": [self.alpha], "calibration": [self.calibration]})

    def _report_frame(self) -> pd.DataFrame:
        frame = self.to_frame()
        if frame.empty:
            frame = pd.DataFrame({"component": [None], "lower": [np.nan], "upper": [np.nan],
                                  "parameter": [self.curve.parameter_name]})
        for name in ("status", "global_geometry", "truncated", "alpha", "calibration"):
            frame[name] = getattr(self, name)
        return frame


class _Unresolved(ArithmeticError):
    pass


def _level_roots(quadratic: float, linear: float, level: float) -> tuple[float, float] | None:
    """All real roots of q*z*z+l*z-level for q>0, with cancellation control."""
    scale = max(abs(quadratic), abs(linear), abs(level))
    a, b, c = quadratic/scale, linear/scale, -level/scale
    if a == 0 or (linear != 0 and b == 0) or (level != 0 and c == 0):
        raise _Unresolved("unresolved_coefficient_range")
    square, cross = b*b, 4*a*c
    if (b != 0 and square == 0) or (a != 0 and c != 0 and cross == 0):
        raise _Unresolved("unresolved_discriminant_range")
    discriminant = math.fsum([square, -cross])
    tolerance = 64*np.finfo(float).eps*(abs(square)+abs(cross))
    if discriminant != 0 and abs(discriminant) <= tolerance:
        raise _Unresolved("unresolved_quadratic_tangency")
    if discriminant < 0:
        return None
    if discriminant == 0:
        root = -b/(2*a)
        if not np.isfinite(root):
            raise _Unresolved("unresolved_root_range")
        return root, root
    numerator = -.5*(b+math.copysign(math.sqrt(discriminant), b))
    roots = sorted((numerator/a, c/numerator))
    if not np.all(np.isfinite(roots)) or roots[0] == roots[1]:
        raise _Unresolved("unresolved_root_range")
    return float(roots[0]), float(roots[1])


def _pullback(curve: QuadraticMomentCurve, low: float, high: float) -> list[tuple[float, float]]:
    linear, quadratic = curve.linear, curve.quadratic
    if quadratic == 0:
        if linear == 0:
            return [(-np.inf, np.inf)] if low <= 0 <= high else []
        roots = sorted((low/linear, high/linear))
        if not np.all(np.isfinite(roots)):
            raise _Unresolved("unresolved_root_range")
        intervals = [(roots[0], roots[1])]
    else:
        if quadratic < 0:
            linear, quadratic, low, high = -linear, -quadratic, -high, -low
        outer = _level_roots(quadratic, linear, high)
        if outer is None:
            return []
        inner = _level_roots(quadratic, linear, low)
        if inner is None or inner[0] == inner[1]:
            intervals = [outer]
        else:
            intervals = [(outer[0], inner[0]), (inner[1], outer[1])]
    mapped = []
    for left, right in intervals:
        with np.errstate(over="ignore", invalid="ignore"):
            lower = float(curve.center+curve.scale*np.float64(left))
            upper = float(curve.center+curve.scale*np.float64(right))
        if (not np.isfinite(lower) or not np.isfinite(upper) or lower > upper or
                (left != right and lower == upper)):
            raise _Unresolved("unresolved_parameter_range")
        mapped.append((lower, upper))
    return mapped


def invert_quadratic_moments(curve: QuadraticMomentCurve, targets: MomentTargets, *,
                             covariance: Any = None, domain: tuple[float, float] = (-np.inf, np.inf),
                             alpha: float = .05, calibration: str = "asymptotic") -> MomentConfidenceSet:
    """Invert the point-null S test for a declared scalar affine/quadratic index.

    Covariance must be constant over theta for this analytic class. Use
    candidate evaluation for varying covariance/general models. A supplied
    covariance is the total error covariance, not a fitted or pilot weight.
    Finite domain endpoints clip a known global set and are marked as such.
    """
    if not isinstance(curve, QuadraticMomentCurve) or not isinstance(targets, MomentTargets):
        raise ValueError("curve and targets must be QuadraticMomentCurve and MomentTargets")
    alpha = _settings(alpha, calibration)
    if curve.labels != targets.labels or curve.units != targets.units:
        raise ValueError("curve labels and units must exactly match targets")
    if np.iscomplexobj(domain):
        raise ValueError("domain must be real")
    limits = np.asarray(domain, dtype=float)
    if limits.shape != (2,) or np.any(np.isnan(limits)) or limits[0] >= limits[1]:
        raise ValueError("domain must be an ordered lower/upper pair without NaNs")
    domain = (float(limits[0]), float(limits[1]))
    cov = targets.covariance if covariance is None else _matrix(covariance, targets.labels, "moment-error covariance")
    q = len(targets.labels)
    critical = float(chi2.isf(alpha, q))
    diagnostic: dict[str, Any] = {
        "scope": "complete inverse image of the declared common scalar quadratic index; subject to floating-point resolution",
        "covariance_scope": "constant full moment-error covariance over the parameter",
        "identification": "no derivative, rank cutoff, estimated parameter, or objective subtraction",
        "statistical_assumptions": "Gaussian null error with known covariance" if calibration == "gaussian_known" else
        "null CLT and consistent covariance; uniform validity requires corresponding uniform assumptions",
        "null": "hypothesized parameter and maintained moment restrictions",
    }
    global_parts: list[tuple[float, float]] = []
    status = "resolved"
    try:
        transform = _whitener(cov)
        with np.errstate(over="ignore", invalid="ignore"):
            a = transform @ (targets.values-curve.offset)
            b = transform @ curve.loading
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise _Unresolved("unresolved_numeric_range")
        bmax = float(np.max(abs(b)))
        if np.all(curve.loading == 0) or (curve.linear == 0 and curve.quadratic == 0):
            s = float(a @ a)
            if not np.isfinite(s):
                raise _Unresolved("unresolved_numeric_range")
            gap = critical-s
            if gap != 0 and abs(gap) <= 64*np.finfo(float).eps*max(1., critical, s):
                raise _Unresolved("unresolved_constant_boundary")
            global_parts = [(-np.inf, np.inf)] if gap >= 0 else []
            diagnostic["constant_statistic"] = s
        else:
            if bmax == 0:
                raise _Unresolved("unresolved_loading_range")
            direction = b/bmax
            length = float(np.linalg.norm(direction))
            direction /= length
            norm_b = bmax*length
            coordinate = float(direction @ a)
            orthogonal = a-direction*coordinate
            minimum = float(orthogonal @ orthogonal)
            if not np.isfinite(minimum) or not np.isfinite(norm_b):
                raise _Unresolved("unresolved_numeric_range")
            gap = critical-minimum
            diagnostic.update({"minimum_over_unrestricted_index": minimum, "whitened_loading_norm": norm_b})
            if gap != 0 and abs(gap) <= 64*np.finfo(float).eps*max(1., critical, minimum):
                raise _Unresolved("unresolved_projection_tangency")
            if gap >= 0:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    midpoint = float(np.float64(coordinate)/norm_b)
                    radius = float(np.sqrt(gap)/norm_b)
                    low, high = midpoint-radius, midpoint+radius
                if not np.all(np.isfinite([low, high])) or (radius > 0 and low == high):
                    raise _Unresolved("unresolved_index_range")
                diagnostic["accepted_index_interval"] = (low, high)
                global_parts = _pullback(curve, low, high)
        # Re-evaluate every finite endpoint in the original moments. A good
        # polynomial root is not enough if parameter transformation lost digits.
        discrepancies = []
        for left, right in global_parts:
            for theta in (left, right):
                if np.isfinite(theta):
                    endpoint = moment_restriction_test(targets, curve(theta), covariance=cov, alpha=alpha,
                                                        calibration=calibration)
                    discrepancy = abs(endpoint.statistic-critical)/max(1., critical)
                    if not endpoint.valid or not np.isfinite(discrepancy) or discrepancy > 1e-7:
                        raise _Unresolved("unresolved_endpoint_residual")
                    discrepancies.append(discrepancy)
        diagnostic["max_endpoint_relative_residual"] = max(discrepancies, default=0.)
    except np.linalg.LinAlgError as exc:
        status = "unresolved_covariance"
        diagnostic["numerical_failure"] = str(exc)
    except _Unresolved as exc:
        status = str(exc)
    if status != "resolved":
        global_parts = []
    global_intervals = np.asarray(global_parts, dtype=float).reshape(-1, 2)
    parts = [(max(a, domain[0]), min(b, domain[1])) for a, b in global_parts
             if max(a, domain[0]) <= min(b, domain[1])]
    intervals = np.asarray(parts, dtype=float).reshape(-1, 2)
    truncated = any(a < domain[0] or b > domain[1] for a, b in global_parts)
    contact = tuple(bool(np.isfinite(bound) and any(a <= bound <= b for a, b in parts)) for bound in domain)
    if status != "resolved":
        geometry = "unresolved"
    elif not global_parts:
        geometry = "empty"
    elif len(global_parts) == 1 and global_parts[0] == (-np.inf, np.inf):
        geometry = "all_real"
    elif np.any(~np.isfinite(global_intervals)):
        geometry = "unbounded"
    else:
        geometry = "bounded"
    return MomentConfidenceSet(intervals, global_intervals, domain, (contact[0], contact[1]),
                                truncated, geometry, status, alpha, q, critical, calibration,
                                curve, targets, cov, diagnostic)


@dataclass(frozen=True)
class MomentTestGridResult(_Reports):
    """Tests at sampled nodes, never a certificate about unsampled interiors."""

    grid: np.ndarray
    tests: tuple[MomentTestResult, ...]
    parameter_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", _snapshot(self.grid))
        object.__setattr__(self, "tests", tuple(self.tests))

    @property
    def continuous_complete(self) -> bool:
        return False

    @property
    def accepted(self) -> np.ndarray:
        return _snapshot(np.array([r.accepted is True for r in self.tests]))

    @property
    def valid(self) -> np.ndarray:
        return _snapshot(np.array([r.valid for r in self.tests]))

    @property
    def boundary_status(self) -> tuple[str, str]:
        def state(r: MomentTestResult) -> str:
            return "unresolved" if not r.valid else "accepted" if r.accepted else "rejected"
        return state(self.tests[0]), state(self.tests[-1])

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame({"parameter": self.parameter_name, "value": self.grid,
                              "statistic": [r.statistic for r in self.tests],
                              "pvalue": [r.pvalue for r in self.tests],
                              "accepted": pd.array([r.accepted for r in self.tests], dtype="boolean"),
                              "status": [r.status for r in self.tests], "scope": "sampled point",
                              "alpha": self.tests[0].alpha, "calibration": self.tests[0].calibration})
        frame.attrs.update({"continuous_complete": False, "scope": "sampled points only",
                            "alpha": self.tests[0].alpha, "calibration": self.tests[0].calibration})
        return frame

    def accepted_runs(self) -> pd.DataFrame:
        """Consecutive accepted-node spans; their interiors are untested."""
        runs, start = [], None
        accepted = self.accepted
        for i, value in enumerate(accepted):
            if value and start is None:
                start = i
            if start is not None and (not value or i == len(accepted)-1):
                last = i if value else i-1
                runs.append((self.grid[start], self.grid[last], last-start+1))
                start = None
        frame = pd.DataFrame(runs, columns=["first_accepted_node", "last_accepted_node", "n_nodes"])
        frame.attrs["scope"] = "sampled spans, not continuous confidence intervals"
        return frame

    def unresolved_cells(self) -> pd.DataFrame:
        return pd.DataFrame({"lower": self.grid[:-1], "upper": self.grid[1:],
                             "reason": ["invalid_endpoint" if not (a.valid and b.valid) else "unsampled_interior"
                                        for a, b in zip(self.tests[:-1], self.tests[1:])]})


def evaluate_moment_grid(test_at: Callable[[float], MomentTestResult], grid: Any, *,
                         parameter_name: str = "theta") -> MomentTestGridResult:
    """Evaluate full point-null tests on a grid without interpolating acceptance.

    Test covariance may vary with the candidate. Every unsampled cell remains
    unresolved, including cells with two rejected endpoints. Invalid test
    results retain their evidence and break accepted-node groups. Unexpected
    callback exceptions propagate; they do not become rejection penalties.
    """
    values = _array(grid, "grid", 1)
    if len(values) < 2 or np.any(np.diff(values) <= 0):
        raise ValueError("grid must contain at least two strictly increasing finite nodes")
    if not callable(test_at):
        raise ValueError("test_at must be callable")
    if not isinstance(parameter_name, str) or not parameter_name.strip():
        raise ValueError("parameter_name must be a nonempty string")
    results: list[MomentTestResult] = []
    for value in values:
        result = test_at(float(value))
        if not isinstance(result, MomentTestResult):
            raise ValueError("test_at must return MomentTestResult")
        if results:
            first = results[0]
            if (result.alpha, result.df, result.labels, result.calibration) != (first.alpha, first.df, first.labels, first.calibration):
                raise ValueError("grid tests must share alpha, labels, df and calibration")
        results.append(result)
    return MomentTestGridResult(values, tuple(results), parameter_name)
