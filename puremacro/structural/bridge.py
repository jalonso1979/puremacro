"""Minimum distance with the covariance of the empirical estimators.

The covariance supplied here is Var(m_hat), not Var(sqrt(n) * m_hat).
No sample-size multiplier or residual-variance rescaling is applied. Local
inference assumes a correctly specified, smooth, regularly identified model;
it is not misspecification-robust or weak-identification-robust inference.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares
from scipy.stats import chi2, norm

from puremacro._linalg import safe_cholesky
from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst


def _array(value: Any, name: str, ndim: int) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    a = np.array(value, dtype=float, copy=True)
    if a.ndim != ndim or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return a


def _readonly(value: Any) -> np.ndarray:
    a = np.array(value, copy=True)
    a.setflags(write=False)
    return a


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, np.ndarray):
        return _readonly(value)
    return deepcopy(value)


def _labels(value: Sequence[str], n: int, name: str = "labels") -> tuple[str, ...]:
    if isinstance(value, str):
        raise ValueError(f"{name} must be a sequence, not a string")
    out = tuple(value)
    if len(out) != n or any(not isinstance(x, str) or not x.strip() for x in out):
        raise ValueError(f"{name} must contain {n} nonempty strings")
    if len(set(out)) != n:
        raise ValueError(f"{name} must be unique")
    return out


def _matrix(value: Any, labels: tuple[str, ...], name: str) -> np.ndarray:
    if isinstance(value, pd.DataFrame):
        if tuple(value.index) != labels or tuple(value.columns) != labels:
            raise ValueError(f"{name} row and column labels must exactly match target order")
    a = _array(value, name, 2)
    n = len(labels)
    if a.shape != (n, n):
        raise ValueError(f"{name} must have shape ({n}, {n})")
    d = np.diag(a)
    if np.any(d < 0):
        raise ValueError(f"{name} must be positive semidefinite (negative diagonal)")
    # Work in correlation units so diagnostics do not depend on moment units.
    scale = np.sqrt(np.where(d > 0, d, 1.0))
    scaled = a / scale[:, None] / scale[None, :]
    if not np.all(np.isfinite(scaled)):
        raise ValueError(f"{name} cannot be represented in standardized units")
    if not np.allclose(scaled, scaled.T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    if np.any(a[d == 0] != 0) or np.any(a[:, d == 0] != 0):
        raise ValueError(f"{name} with zero variance must have zero covariance in that row and column")
    # Averaging only removes accepted floating-point asymmetry; no ridge or
    # eigenvalue clipping is used. Fitting separately requires positive definiteness.
    a = 0.5 * a + 0.5 * a.T
    try:
        eig = np.linalg.eigvalsh((scaled + scaled.T) / 2)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(f"structural {name}: eigenvalue check failed") from exc
    if eig[0] < -1e-12 * max(1.0, float(eig[-1])):
        raise ValueError(f"{name} must be positive semidefinite")
    return a


def _factor(a: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    scales = np.sqrt(np.diag(a))
    if np.any(scales <= 0):
        raise np.linalg.LinAlgError(f"structural {name}: positive definite matrix required; no ridge is applied")
    correlation = a / scales[:, None] / scales[None, :]
    factor = safe_cholesky(correlation, name=f"structural {name} (standardized)")
    return scales, factor


@dataclass(frozen=True)
class MomentTargets:
    """Labeled empirical moments and their full estimator covariance.

    ``covariance`` is Cov(values), already on the estimator's sampling scale.
    It may be positive semidefinite for storage; fitting requires positive
    definiteness. Labeled Series/DataFrames must be in exactly ``labels`` order.
    Missing units default to ``"unspecified"``; no unit conversion is inferred.
    """

    values: np.ndarray
    covariance: np.ndarray
    labels: tuple[str, ...]
    units: tuple[str, ...] | str = "unspecified"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _array(self.values, "values", 1)
        if not len(values):
            raise ValueError("targets must contain at least one moment")
        labels = _labels(self.labels, len(values))
        if isinstance(self.values, pd.Series) and tuple(self.values.index) != labels:
            raise ValueError("values labels must exactly match target order")
        units = (self.units,) * len(values) if isinstance(self.units, str) else tuple(self.units)
        if len(units) != len(values) or any(not isinstance(u, str) or not u.strip() for u in units):
            raise ValueError("units must provide a nonempty string for every moment")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "values", _readonly(values))
        object.__setattr__(self, "covariance", _readonly(_matrix(self.covariance, labels, "covariance")))
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *, covariance: Any,
                   value: str = "estimate", label: str = "label", unit: str = "unit",
                   metadata: Mapping[str, Any] | None = None) -> MomentTargets:
        """Read a moment table, requiring full covariance supplied separately.

        Labels default to the index if ``label`` is absent. Frame ``attrs``
        are preserved and explicit metadata overrides attributes with the same key.
        """
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("frame must be a pandas DataFrame")
        if value not in frame:
            raise ValueError(f"frame has no {value!r} value column")
        labels = tuple(frame[label]) if label in frame else tuple(frame.index)
        units = tuple(frame[unit]) if unit in frame else "unspecified"
        meta = dict(frame.attrs)
        meta.update(metadata or {})
        return cls(frame[value].to_numpy(), covariance, labels, units, meta)

    @classmethod
    def from_draws(cls, draws: Any, *, values: Any, labels: Sequence[str],
                   units: str | Sequence[str] = "unspecified",
                   metadata: Mapping[str, Any] | None = None) -> MomentTargets:
        """Estimate full covariance from joint empirical-estimator replicates.

        Rows must be valid joint bootstrap/sampling replicates and columns
        follow ``labels``. ``values`` is the original empirical point estimate,
        not the average replicate. Uses sample covariance (ddof=1), without
        dividing by the replicate count. Posterior draws or independently
        resampled horizons do not generally estimate the required covariance.
        The caller is responsible for the bootstrap's dependence and validity.
        """
        labels = tuple(labels)
        if isinstance(draws, pd.DataFrame) and tuple(draws.columns) != labels:
            raise ValueError("draw columns must exactly match target labels")
        array = _array(draws, "draws", 2)
        if array.shape[0] < 2 or array.shape[1] != len(labels):
            raise ValueError("draws require at least two joint replicates and one column per label")
        covariance = np.atleast_2d(np.cov(array, rowvar=False, ddof=1))
        meta = dict(metadata or {})
        meta.update({"covariance_source": "joint_estimator_draws", "n_draws": len(array),
                     "covariance_ddof": 1})
        return cls(values, covariance, labels, units, meta)

    @classmethod
    def from_irf(cls, irf: Any, *, covariance: Any, response: str, shock: str,
                 units: str, horizons: Sequence[Any] | None = None,
                 value: str = "beta", metadata: Mapping[str, Any] | None = None) -> MomentTargets:
        """Adapt one selected IRF/LP response to one shock.

        ``irf`` is a 1-D vector or a table containing ``value`` (default
        ``beta``) and optionally ``h``. Select the response/shock of a VAR
        tensor explicitly before calling. Pointwise bands cannot recover a
        joint covariance, so ``covariance`` is always required.
        """
        if not isinstance(response, str) or not response.strip() or not isinstance(shock, str) or not shock.strip():
            raise ValueError("response and shock must be nonempty strings")
        meta: dict[str, Any] = {}
        if isinstance(irf, pd.DataFrame):
            if value not in irf:
                raise ValueError(f"IRF table has no {value!r} column; select one response explicitly")
            values = irf[value].to_numpy()
            source_h = tuple(irf["h"]) if "h" in irf else tuple(irf.index)
            meta.update(irf.attrs)
            # LPResult exposes sample/method information outside attrs too.
            irf_metadata = getattr(irf, "metadata", None)
            if isinstance(irf_metadata, Mapping):
                meta.update(irf_metadata)
            if horizons is not None and tuple(horizons) != source_h:
                raise ValueError("horizons must exactly match the IRF table row order")
            horizons = source_h
        else:
            values = _array(irf, "irf", 1)
        h = tuple(range(len(values))) if horizons is None else tuple(horizons)
        if len(h) != len(values):
            raise ValueError("horizons must match the selected IRF length")
        meta.update(metadata or {})
        meta.update({"response": response, "shock": shock, "horizons": h})
        labels = tuple(f"{response}<-{shock}[h={x}]" for x in h)
        return cls(values, covariance, labels, units, meta)

    def select(self, labels: Sequence[str]) -> MomentTargets:
        """Subset/reorder moments and both axes of their covariance together."""
        selected = _labels(labels, len(labels))
        unknown = set(selected) - set(self.labels)
        if unknown:
            raise ValueError(f"unknown moment labels: {sorted(unknown)}")
        idx = [self.labels.index(label) for label in selected]
        return MomentTargets(self.values[idx], self.covariance[np.ix_(idx, idx)],
                             selected, tuple(self.units[i] for i in idx), self.metadata)

    def stack(self, other: MomentTargets, *, cross_covariance: Any = None,
              assume_independent: bool = False) -> MomentTargets:
        """Combine targets with explicit cross-covariance or independence.

        The cross block has rows in self.labels and columns in other.labels.
        Independence is an assumption, not inferred from different moment names.
        Component metadata (including samples and shock normalizations) is retained.
        """
        if not isinstance(other, MomentTargets):
            raise ValueError("other must be MomentTargets")
        if not isinstance(assume_independent, (bool, np.bool_)):
            raise ValueError("assume_independent must be boolean")
        if cross_covariance is None and not assume_independent:
            raise ValueError("provide cross_covariance or explicitly set assume_independent=True")
        if cross_covariance is not None and assume_independent:
            raise ValueError("choose cross_covariance or independence, not both")
        n, k = len(self.values), len(other.values)
        cross = np.zeros((n, k))
        if cross_covariance is not None:
            if isinstance(cross_covariance, pd.DataFrame):
                if tuple(cross_covariance.index) != self.labels or tuple(cross_covariance.columns) != other.labels:
                    raise ValueError("cross_covariance labels must match the two target orders")
            cross = _array(cross_covariance, "cross_covariance", 2)
            if cross.shape != (n, k):
                raise ValueError(f"cross_covariance must have shape ({n}, {k})")
        covariance = np.block([[self.covariance, cross], [cross.T, other.covariance]])
        metadata = {"components": (self.metadata, other.metadata),
                    "cross_covariance_assumption": "independent" if assume_independent else "supplied"}
        return MomentTargets(np.r_[self.values, other.values], covariance,
                             self.labels + other.labels, self.units + other.units, metadata)

    def rescale(self, factors: Any, *, units: str | Sequence[str]) -> MomentTargets:
        """Change moment units with covariance transformed on both axes."""
        if np.iscomplexobj(factors):
            raise ValueError("factors must be real")
        f = np.asarray(factors, dtype=float)
        if f.ndim == 0:
            f = np.full(len(self.values), f)
        f = _array(f, "factors", 1)
        if f.shape != self.values.shape or np.any(f == 0):
            raise ValueError("factors must be nonzero and match the moments")
        metadata = dict(self.metadata)
        metadata["rescaling"] = {"factors": f, "previous_units": self.units,
                                  "previous_rescaling": self.metadata.get("rescaling")}
        return MomentTargets(self.values * f, self.covariance * f[:, None] * f[None, :],
                             self.labels, units, metadata)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"label": self.labels, "estimate": self.values,
                             "se": np.sqrt(np.diag(self.covariance)), "unit": self.units})


@dataclass(frozen=True)
class StructuralFitResult:
    """Minimum-distance fit with separate optimizer and inference status.

    ``success`` only describes numerical optimization. ``inference_valid``
    records local numerical regularity, conditional on the stated statistical
    assumptions; it is not evidence that the structural model is correct.
    """

    theta: np.ndarray
    parameter_names: tuple[str, ...]
    covariance: np.ndarray
    standard_errors: np.ndarray
    model_moments: np.ndarray
    targets: MomentTargets
    jacobian: np.ndarray
    objective: float
    success: bool
    inference_valid: bool
    identification_rank: int
    condition_number: float
    singular_values: np.ndarray
    boundary: np.ndarray
    n_evals: int
    message: str
    j_statistic: float
    j_df: int
    j_pvalue: float
    diagnostics: Mapping[str, Any]
    held_out_targets: MomentTargets | None = None
    held_out_model: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("theta", "covariance", "standard_errors", "model_moments", "jacobian",
                     "singular_values", "boundary", "held_out_model"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))

    def summary(self, *, ci: float = 0.95) -> pd.DataFrame:
        """Parameter estimates and local normal intervals (NaN when unavailable)."""
        if not np.isfinite(ci) or not 0 < ci < 1:
            raise ValueError("ci must lie strictly between zero and one")
        critical = norm.ppf((1 + ci) / 2)
        frame = pd.DataFrame({"parameter": self.parameter_names, "estimate": self.theta,
                              "se": self.standard_errors,
                              "ci_lower": self.theta - critical * self.standard_errors,
                              "ci_upper": self.theta + critical * self.standard_errors,
                              "boundary": self.boundary})
        frame.attrs.update({"success": self.success, "inference_valid": self.inference_valid,
                            "ci": ci, "objective": self.objective})
        return frame

    def moment_fit(self, *, held_out: bool = False) -> pd.DataFrame:
        """Target/model comparison. Held-out residuals are descriptive only."""
        targets = self.held_out_targets if held_out else self.targets
        model = self.held_out_model if held_out else self.model_moments
        if targets is None or model is None:
            raise ValueError("no held-out moments were supplied")
        frame = targets.to_frame().rename(columns={"estimate": "target", "se": "target_se"})
        frame["model"] = model
        frame["residual"] = model - targets.values
        frame["used_in_fit"] = not held_out
        return frame

    def to_markdown(self, *, ci: float = 0.95, **kwargs: Any) -> str:
        return df_to_markdown(self.summary(ci=ci), index=False, **kwargs)

    def to_latex(self, *, ci: float = 0.95, **kwargs: Any) -> str:
        return df_to_latex(self.summary(ci=ci), index=False, **kwargs)

    def to_typst(self, *, ci: float = 0.95, **kwargs: Any) -> str:
        return df_to_typst(self.summary(ci=ci), index=False, **kwargs)


def _model_values(moments_at: Callable, theta: np.ndarray, targets: MomentTargets) -> np.ndarray:
    values = moments_at(theta.copy())
    if isinstance(values, pd.Series) and tuple(values.index) != targets.labels:
        raise ValueError("model moment labels must exactly match target order")
    out = _array(values, "model moments", 1)
    if out.shape != targets.values.shape:
        raise ValueError(f"model must return {len(targets.values)} moments in target label order")
    return out


def _numerical_jac(fun: Callable, x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Second-order finite differences, using one-sided stencils at bounds."""
    base = fun(x)
    out = np.empty((len(base), len(x)))
    for j in range(len(x)):
        h = np.cbrt(np.finfo(float).eps) * max(1.0, abs(x[j]))
        left, right = x[j] - lo[j], hi[j] - x[j]
        if left >= h and right >= h:
            a, b = x.copy(), x.copy()
            a[j] += h
            b[j] -= h
            out[:, j] = (fun(a) - fun(b)) / (a[j] - b[j])
        else:
            direction, room = (1.0, right) if right >= left else (-1.0, left)
            h = min(h, room / 2)
            a, b = x.copy(), x.copy()
            a[j] += direction * h
            b[j] += direction * 2 * h
            if a[j] == x[j] or b[j] == a[j]:
                raise ValueError("parameter bounds are too narrow for numerical differentiation; supply jac")
            out[:, j] = direction * (-3 * base + 4 * fun(a) - fun(b)) / (2 * h)
    return out


def fit_structural(
    moments_at: Callable[[np.ndarray], Any],
    targets: MomentTargets,
    theta0: Any,
    *,
    parameter_names: Sequence[str] | None = None,
    moment_labels: Sequence[str] | None = None,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    jac: Callable[[np.ndarray], Any] | None = None,
    weight: Any = None,
    held_out: MomentTargets | None = None,
    held_out_moments_at: Callable[[np.ndarray], Any] | None = None,
    max_nfev: int = 2000,
    tol: float = 1e-10,
    rank_rtol: float = 1e-10,
    condition_limit: float = 1e8,
    assume_correct_specification: bool = False,
) -> StructuralFitResult:
    """Fit structural moments to empirical estimates with their joint covariance.

    Minimize (m(theta)-m_hat)' W (m(theta)-m_hat). By default W is the
    inverse empirical-estimator covariance; a supplied fixed SPD ``weight``
    receives sandwich covariance but no chi-square overidentification test.
    ``bounds`` is one (lower, upper) pair per parameter; None is unbounded.
    ``jac`` is the *unweighted* model Jacobian (moments by parameters).
    Arrays from ``moments_at`` must follow target order; ``moment_labels``
    declares/checks that order and Series outputs are checked automatically.

    Local standard errors assume smooth correct specification, consistent
    moment covariance and negligible model/simulation error. They are withheld
    for failed or nonstationary optimization, numerical rank deficiency, excessive Jacobian
    conditioning or a binding/near-binding bound. These numerical diagnostics
    are not a statistical test of weak identification or global identification.
    A J p-value additionally requires ``assume_correct_specification=True``,
    default weighting and positive overidentification degrees of freedom.
    """
    if not isinstance(targets, MomentTargets):
        raise ValueError("targets must be MomentTargets")
    if not callable(moments_at) or (jac is not None and not callable(jac)):
        raise ValueError("moments_at and any supplied jac must be callable")
    x0 = _array(theta0, "theta0", 1)
    p, q = len(x0), len(targets.values)
    if not p:
        raise ValueError("theta0 must contain at least one parameter")
    names = _labels(parameter_names if parameter_names is not None else tuple(f"theta_{i}" for i in range(p)),
                    p, "parameter_names")
    if moment_labels is not None and tuple(moment_labels) != targets.labels:
        raise ValueError("moment_labels must exactly match target order")
    if not isinstance(max_nfev, (int, np.integer)) or isinstance(max_nfev, bool) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    if not np.isfinite(tol) or tol <= np.finfo(float).eps:
        raise ValueError("tol must exceed machine precision")
    if not np.isfinite(rank_rtol) or not 0 < rank_rtol < 1:
        raise ValueError("rank_rtol must lie strictly between zero and one")
    if not np.isfinite(condition_limit) or condition_limit <= 1:
        raise ValueError("condition_limit must be finite and greater than one")
    if not isinstance(assume_correct_specification, (bool, np.bool_)):
        raise ValueError("assume_correct_specification must be boolean")
    if (held_out is None) != (held_out_moments_at is None):
        raise ValueError("held_out and held_out_moments_at must be supplied together")
    if held_out is not None:
        if not isinstance(held_out, MomentTargets) or not callable(held_out_moments_at):
            raise ValueError("held_out must be MomentTargets and held_out_moments_at callable")
        if set(held_out.labels) & set(targets.labels):
            raise ValueError("held-out and fitted moment labels must be disjoint")
    lo, hi = np.full(p, -np.inf), np.full(p, np.inf)
    if bounds is not None:
        if len(bounds) != p or any(len(pair) != 2 for pair in bounds):
            raise ValueError("bounds must contain one (lower, upper) pair per parameter")
        lo = np.array([-np.inf if pair[0] is None else pair[0] for pair in bounds], dtype=float)
        hi = np.array([np.inf if pair[1] is None else pair[1] for pair in bounds], dtype=float)
        if np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo >= hi):
            raise ValueError("bounds require lower < upper without NaNs")
        if np.any(x0 < lo) or np.any(x0 > hi):
            raise ValueError("theta0 lies outside bounds")
    scales, covariance_factor = _factor(targets.covariance, "covariance")
    optimal = weight is None
    if optimal:
        transform = solve_triangular(covariance_factor, np.diag(1 / scales), lower=True)
    else:
        w = _matrix(weight, targets.labels, "weight")
        weight_scale, weight_factor = _factor(w, "weight")
        transform = weight_factor.T * weight_scale[None, :]
    # Multiplying a custom W by a positive constant must not change the fit.
    # Normalize the optimizer's *overall* residual scale to its RMS sampling
    # uncertainty, retaining every relative weight and the original-W objective.
    covariance_root = scales[:, None] * covariance_factor
    sampling_loading = transform @ covariance_root
    loading_max = float(np.max(np.abs(sampling_loading)))
    if not np.isfinite(loading_max) or loading_max <= 0:
        raise ValueError("weight/covariance scales exceed numerical range")
    optimizer_residual_scale = loading_max * float(np.sqrt(np.sum(
        (sampling_loading / loading_max) ** 2) / q))
    if not np.isfinite(optimizer_residual_scale) or optimizer_residual_scale <= 0:
        raise ValueError("weight/covariance scales exceed numerical range")
    optimizer_transform = transform / optimizer_residual_scale
    if not np.all(np.isfinite(optimizer_transform)):
        raise ValueError("normalized weights exceed numerical range; rescale moment units")
    raw_fun = lambda x: _model_values(moments_at, x, targets)

    def raw_jac(x: np.ndarray) -> np.ndarray:
        if jac is None:
            out = _numerical_jac(raw_fun, x, lo, hi)
        else:
            value = jac(x.copy())
            if isinstance(value, pd.DataFrame):
                if tuple(value.index) != targets.labels or tuple(value.columns) != names:
                    raise ValueError("jacobian labels must match moment and parameter order")
            out = _array(value, "jacobian", 2)
        if out.shape != (q, p) or not np.all(np.isfinite(out)):
            raise ValueError(f"jacobian must be a finite array of shape ({q}, {p})")
        return out

    def residual(x: np.ndarray) -> np.ndarray:
        out = transform @ (raw_fun(x) - targets.values)
        if not np.all(np.isfinite(out)):
            raise ValueError("weighted residual is nonfinite; check moment units and covariance")
        return out

    def optimizer_residual(x: np.ndarray) -> np.ndarray:
        out = optimizer_transform @ (raw_fun(x) - targets.values)
        if not np.all(np.isfinite(out)):
            raise ValueError("normalized weighted residual is nonfinite; check moment units and covariance")
        return out

    # x_scale='jac' scales the trust region but not SciPy's absolute gtol.
    # Fit in centered, locally standardized parameter coordinates as well:
    # theta = theta0 + parameter_scale * z. At the start each nonzero weighted
    # Jacobian column then has unit norm. Thus a change of parameter units or
    # a large parameter offset cannot by itself cause initial gtol/xtol success.
    initial_jac = optimizer_transform @ raw_jac(x0)
    if not np.all(np.isfinite(initial_jac)):
        raise ValueError("weighted jacobian is nonfinite; rescale moments or parameters")
    column_max = np.max(np.abs(initial_jac), axis=0)
    parameter_scale = np.ones(p)
    active_columns = column_max > 0
    normalized_columns = initial_jac[:, active_columns] / column_max[active_columns]
    parameter_scale[active_columns] = (1 / column_max[active_columns]) / np.sqrt(
        np.sum(normalized_columns ** 2, axis=0))
    if not np.all(np.isfinite(parameter_scale)) or np.any(parameter_scale <= 0):
        raise ValueError("parameter units exceed numerical range; rescale parameters")
    to_parameter = lambda z: x0 + parameter_scale * z
    fit = least_squares(lambda z: optimizer_residual(to_parameter(z)), np.zeros(p),
                        jac=lambda z: (optimizer_transform @ raw_jac(to_parameter(z))) * parameter_scale,
                        bounds=((lo - x0) / parameter_scale, (hi - x0) / parameter_scale),
                        method="trf", max_nfev=int(max_nfev), x_scale="jac",
                        ftol=tol, xtol=tol, gtol=tol)
    # The optimizer's bounds hold in z; mapping a bound back through
    # theta0 + scale * z can round an ulp past it, which made the estimate
    # unusable as the start of another bounded fit.
    theta = np.clip(to_parameter(fit.x), lo, hi)
    model = raw_fun(theta)
    derivative = raw_jac(theta)
    weighted_jac = transform @ derivative
    try:
        u, singular, vt = np.linalg.svd(weighted_jac, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError("structural identification: Jacobian SVD failed") from exc
    rank = int(np.sum(singular > rank_rtol * singular[0])) if len(singular) and singular[0] > 0 else 0
    condition = float(singular[0] / singular[-1]) if rank == p else float("inf")
    # Relative to the parameter and finite bound magnitudes; avoid treating
    # infinities as tolerance scales. Near bounds ordinary normal inference fails.
    boundary_tol = max(np.sqrt(np.finfo(float).eps), 10 * tol)
    lower_close = np.isfinite(lo) & ((theta - lo) <= boundary_tol * np.maximum(1, np.maximum(np.abs(theta), np.where(np.isfinite(lo), np.abs(lo), 0))))
    upper_close = np.isfinite(hi) & ((hi - theta) <= boundary_tol * np.maximum(1, np.maximum(np.abs(theta), np.where(np.isfinite(hi), np.abs(hi), 0))))
    boundary = lower_close | upper_close | (fit.active_mask != 0)
    weighted_residual = residual(theta)
    standardized_residual = optimizer_residual(theta)
    # The projection onto the weighted Jacobian's column space vanishes at
    # an interior first-order optimum. Unlike G'Wr, its magnitude is invariant
    # to nonsingular parameter rescaling when numerical rank is retained.
    # An overidentified fit can retain a large orthogonal residual at a valid
    # minimum; testing the full residual would incorrectly reject such fits.
    projected_residual = float(np.linalg.norm(u[:, :rank].T @ standardized_residual))
    stationarity_tolerance = max(np.sqrt(tol), 10 * np.sqrt(np.finfo(float).eps)) * max(
        1.0, float(np.linalg.norm(standardized_residual)))
    reasons = []
    if not fit.success:
        reasons.append("optimizer_failed")
    if rank < p:
        reasons.append("rank_deficient")
    elif condition > condition_limit:
        reasons.append("ill_conditioned_jacobian")
    if np.any(boundary):
        reasons.append("parameter_on_boundary")
    elif rank == p and projected_residual > stationarity_tolerance:
        reasons.append("nonstationary_solution")
    covariance = np.full((p, p), np.nan)
    se = np.full(p, np.nan)
    if not reasons:
        # SVD left inverse avoids squaring the Jacobian condition number.
        # A = (G'WG)^-1 G'W. The covariance is A Cov(m_hat) A', with
        # Cov(m_hat)=D L L' D. Computing its square root avoids cancellation.
        left_inverse = (vt.T / singular) @ u.T
        loading = left_inverse @ transform @ covariance_root
        covariance = loading @ loading.T
        if not np.all(np.isfinite(covariance)):
            reasons.append("nonfinite_parameter_covariance")
            covariance[:] = np.nan
        else:
            se = np.sqrt(np.diag(covariance))
    inference_valid = not reasons
    if reasons:
        warnings.warn("Structural local inference unavailable: " + ", ".join(reasons), UserWarning, stacklevel=2)
    objective = float(weighted_residual @ weighted_residual)
    degrees = q - p
    j_available = inference_valid and optimal and assume_correct_specification and degrees > 0
    j_statistic = objective if j_available else float("nan")
    j_pvalue = float(chi2.sf(objective, degrees)) if j_available else float("nan")
    held_model = None if held_out is None else _model_values(held_out_moments_at, theta, held_out)
    diagnostics = {
        "inference_unavailable_reasons": tuple(reasons), "weighting": "optimal" if optimal else "custom",
        "covariance_scale": "empirical_estimator", "rank_rtol": rank_rtol,
        "condition_limit": condition_limit, "jacobian_method": "supplied" if jac is not None else "finite_difference",
        "optimizer_status": int(fit.status), "optimizer_optimality": float(fit.optimality),
        "optimizer_parameter_scaling": "centered coordinates with initial weighted-Jacobian column normalization; adaptive Jacobian trust region",
        "optimizer_parameter_scale": parameter_scale,
        "optimizer_residual_scale": optimizer_residual_scale,
        "projected_residual_norm": projected_residual,
        "stationarity_tolerance": stationarity_tolerance,
        "stationarity_check": "weighted-residual projection onto the local Jacobian column space; interior numerical first-order condition",
        "assume_correct_specification": bool(assume_correct_specification),
        "j_test_available": j_available,
        "j_test_conditions": "correct specification, consistent moment covariance, regular interior identification, optimal weighting, q > p",
        "inference_assumptions": "local smooth correct specification; consistent empirical covariance; negligible model/simulation error",
        "identification_scope": "numerical local rank/conditioning in supplied parameter units; not a weak-identification test or global identification proof",
        "held_out_inference": "descriptive only; no covariance between fitted and held-out estimators is assumed",
    }
    return StructuralFitResult(theta, names, covariance, se, model, targets, derivative,
                               objective, bool(fit.success), inference_valid, rank, condition,
                               singular, boundary, int(fit.nfev), str(fit.message), j_statistic,
                               max(0, degrees), j_pvalue, diagnostics, held_out, held_model)


__all__ = ["MomentTargets", "StructuralFitResult", "fit_structural"]
