"""Finite Gaussian information groups with dated linear observations.

All signal noise is explicitly included in the common zero-mean Gaussian
primitive vector. A signal is an exact linear combination of that vector.
Only the listed observations are processed; market observations are not added
implicitly. Groups share observations internally, not across groups.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from puremacro._numeric_contracts import freeze as _freeze_metadata, thaw as _thaw, readonly_array, real_scalar
from scipy.linalg import solve_triangular

__all__ = ["GaussianSignal", "GaussianInformation", "build_gaussian_information", "audit_gaussian_information"]


def _array(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        out = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if not np.isfinite(out).all() or shape is not None and out.shape != shape:
        raise ValueError(f"{name} must be finite with shape {shape}")
    return readonly_array(out)


def _number(value: Any, name: str) -> float:
    return real_scalar(value, name)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _weights(value: Any, count: int) -> np.ndarray:
    out = _array(np.full(count, 1/count) if value is None else value, "weights", (count,))
    if np.min(out) < 0 or not np.isclose(out.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError("weights must be nonnegative and sum to one")
    return out


def _labels(value: Any, count: int, prefix: str) -> tuple[str, ...]:
    labels = tuple(f"{prefix}{i}" for i in range(count)) if value is None else value
    if (not isinstance(labels, (tuple, list)) or len(labels) != count
            or any(not isinstance(v, str) or not v.strip() for v in labels) or len(set(labels)) != count):
        raise ValueError(f"{prefix} labels must be distinct nonempty strings")
    return tuple(labels)


@dataclass(frozen=True, init=False)
class GaussianSignal:
    """Observation loading @ z, released at an integer date, shared by its group."""
    date: int
    loading: np.ndarray
    label: str

    def __init__(self, date: int, loading: Any, label: str):
        object.__setattr__(self, "date", date)
        object.__setattr__(self, "loading", loading)
        object.__setattr__(self, "label", label)
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(self, "date", _integer(self.date, "signal date"))
        loading = _array(self.loading, "signal loading")
        if loading.ndim != 1 or not len(loading) or np.max(abs(loading)) == 0:
            raise ValueError("signal loading must be a nonzero vector")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("signal label must be nonempty")
        object.__setattr__(self, "loading", loading)


@dataclass(frozen=True)
class GaussianInformation:
    covariance: np.ndarray
    signals: tuple[tuple[GaussianSignal, ...], ...]
    horizon: int
    weights: np.ndarray
    group_names: tuple[str, ...]
    state_names: tuple[str, ...]
    projectors: np.ndarray
    posterior_covariances: np.ndarray
    ranks: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("covariance", "weights", "projectors", "posterior_covariances", "ranks"):
            object.__setattr__(self, name, _array(getattr(self, name), name))
        object.__setattr__(self, "signals", tuple(tuple(group) for group in self.signals))
        object.__setattr__(self, "group_names", tuple(self.group_names))
        object.__setattr__(self, "state_names", tuple(self.state_names))
        object.__setattr__(self, "diagnostics", _freeze_metadata(self.diagnostics))

    @property
    def fingerprint(self) -> str:
        value = dict(covariance=self.covariance.tolist(), horizon=self.horizon,
            weights=self.weights.tolist(), groups=self.group_names, states=self.state_names,
            signals=[[dict(date=s.date, label=s.label, loading=s.loading.tolist()) for s in group]
                     for group in self.signals])
        return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

    @property
    def average_projectors(self) -> np.ndarray:
        return _array(np.einsum("g,tgij->tij", self.weights, self.projectors), "average projectors")

    def beliefs(self, realization: Any) -> np.ndarray:
        z = _array(realization, "realization", (len(self.covariance),))
        return _array(np.einsum("tgij,j->tgi", self.projectors, z), "conditional means")

    def forecast(self, loadings: Any, realization: Any, *, date: int, group: int) -> Any:
        t, g = _integer(date, "date"), _integer(group, "group")
        if t >= self.horizon or g >= len(self.weights):
            raise ValueError("date or group is outside the information panel")
        coefficient = _array(loadings, "loadings")
        if coefficient.ndim not in (1, 2) or coefficient.shape[-1] != len(self.covariance):
            raise ValueError("loadings must be a vector or matrix over the primitive state")
        z = _array(realization, "realization", (len(self.covariance),))
        return coefficient@self.projectors[t, g]@z

    def belief_frame(self, realization: Any) -> pd.DataFrame:
        index = pd.MultiIndex.from_product([range(self.horizon), self.group_names, self.state_names],
                                          names=["date", "group", "state"])
        variance = np.diagonal(self.posterior_covariances, axis1=-2, axis2=-1)
        # Tiny negative diagonal roundoff is zero; the full covariance is audited.
        return pd.DataFrame(dict(conditional_mean=self.beliefs(realization).ravel(),
            posterior_sd=np.sqrt(np.maximum(variance, 0.)).ravel()), index=index)


def _basis(rows: np.ndarray, chol: np.ndarray) -> tuple[np.ndarray, int, list[float]]:
    # Normalize before and after whitening to remove arbitrary observation units.
    scaled = rows/np.max(abs(rows), axis=1)[:, None]
    white = scaled@chol
    white /= np.linalg.norm(white, axis=1)[:, None]
    _, singular, vectors = np.linalg.svd(white, full_matrices=False)
    relative = singular/singular[0]
    if np.any((relative >= 1e-13) & (relative <= 1e-11)):
        raise ValueError("ambiguous information rank near the declared 1e-12 threshold")
    rank = int(np.sum(relative > 1e-12))
    return vectors[:rank].T@vectors[:rank], rank, singular.tolist()


def build_gaussian_information(covariance: Any, signals: Sequence[Sequence[GaussianSignal]], *,
        horizon: int, weights: Any = None, group_names: Any = None,
        state_names: Any = None) -> GaussianInformation:
    """Build nested conditional means for a zero-mean Gaussian primitive vector.

    Noise is represented inside the primitive covariance/loading arrays, not
    added again by this function. Signal dates belong to 0,...,horizon-1.
    Empty signal groups, exact duplicate signals and full revelation are valid.
    The prior must be positive definite, with condition number <=1e12. A
    whitened, row-normalized SVD uses rank threshold 1e-12 and rejects relative
    singular values in [1e-13,1e-11]. There is no jitter or prior regularization.
    Dense panels are limited to two million entries per projector array.
    """
    T = _integer(horizon, "horizon", 1)
    C = _array(covariance, "covariance")
    if C.ndim != 2 or not len(C) or C.shape != (len(C), len(C)):
        raise ValueError("covariance must be a nonempty square matrix")
    if not np.allclose(C, C.T, rtol=1e-12, atol=0):
        raise ValueError("covariance must be symmetric")
    try:
        chol = np.linalg.cholesky(C)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    condition = float(np.linalg.cond(C))
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError("prior covariance is too ill-conditioned; no regularization is applied")
    try:
        groups = tuple(tuple(group) for group in signals)
    except TypeError as exc:
        raise ValueError("signals must contain one sequence per information group") from exc
    G, n = len(groups), len(C)
    if not G or T*G*n*n > 2_000_000 or sum(map(len, groups)) > 4096:
        raise ValueError("empty groups or dense information storage/signal limit exceeded")
    w = _weights(weights, G)
    names, states = _labels(group_names, G, "group"), _labels(state_names, n, "state")
    for group in groups:
        if any(not isinstance(s, GaussianSignal) or s.loading.shape != (n,) or s.date >= T for s in group):
            raise ValueError("signals must match the primitive dimension and lie inside the horizon")
        if len({s.label for s in group}) != len(group):
            raise ValueError("signal labels must be distinct within each group")
    projectors = np.empty((T, G, n, n))
    posterior = np.empty_like(projectors)
    ranks = np.zeros((T, G), dtype=int)
    records = []
    for g, group in enumerate(groups):
        cache = {}
        for t in range(T):
            active = tuple(j for j, signal in enumerate(group) if signal.date <= t)
            if active not in cache:
                if active:
                    Q, rank, singular = _basis(np.stack([group[j].loading for j in active]), chol)
                else:
                    Q, rank, singular = np.zeros((n, n)), 0, []
                P = solve_triangular(chol.T, (chol@Q).T, lower=False).T
                V = chol@(np.eye(n)-Q)@chol.T
                cache[active] = P, (V+V.T)/2, rank
                records.append(dict(date=t, group=g, labels=[group[j].label for j in active],
                                    rank=rank, observations=len(active), singular_values=singular))
            projectors[t, g], posterior[t, g], ranks[t, g] = cache[active]
    report = dict(prior_condition=condition, rank_relative_threshold=1e-12,
                  rank_ambiguity_interval=(1e-13, 1e-11), information_events=records,
                  scope="conditional forecasts under the listed information sets")
    result = GaussianInformation(C, groups, T, w, names, states, projectors, posterior, ranks, report)
    certificate = audit_gaussian_information(result)
    if not certificate["accepted"]:
        raise ValueError(f"Gaussian information audit failed: {certificate}")
    return _replace_diagnostics(result, dict(report, audit_max_error=certificate["max_error"], accepted=True))


def _replace_diagnostics(info: GaussianInformation, diagnostics: Mapping[str, Any]) -> GaussianInformation:
    from dataclasses import replace
    return replace(info, diagnostics=diagnostics)


def audit_gaussian_information(info: GaussianInformation, *, tolerance: float = 1e-9) -> dict[str, Any]:
    """Check retained projections using covariance and observation identities."""
    tol = _number(tolerance, "tolerance")
    if not 0 < tol <= 1e-6 or not isinstance(info, GaussianInformation):
        raise ValueError("require GaussianInformation and tolerance in (0,1e-6]")
    C, T = info.covariance, info.horizon
    if C.ndim != 2 or info.weights.ndim != 1:
        return dict(accepted=False, max_error=None, reason="incompatible retained shapes")
    n, G = len(C), len(info.weights)
    if (not n or not G or isinstance(T, (bool, np.bool_)) or not isinstance(T, (int, np.integer))
            or T < 1 or C.shape != (n, n)
            or info.projectors.shape != (T, G, n, n) or info.posterior_covariances.shape != (T, G, n, n)
            or info.ranks.shape != (T, G)):
        return dict(accepted=False, max_error=None, reason="incompatible retained shapes")
    try:
        _weights(info.weights, G)
        _labels(info.group_names, G, "group")
        _labels(info.state_names, n, "state")
        chol = np.linalg.cholesky(C)
    except (ValueError, np.linalg.LinAlgError):
        return dict(accepted=False, max_error=None, reason="invalid prior, labels or group weights")
    if (not np.allclose(C, C.T, rtol=1e-12, atol=0) or np.linalg.cond(C) > 1e12
            or len(info.signals) != G or any(not isinstance(s, GaussianSignal)
                or s.loading.shape != (n,) or s.date >= T for group in info.signals for s in group)):
        return dict(accepted=False, max_error=None, reason="invalid retained prior or signals")
    errors: dict[str, float] = {}
    def check(name, a, b=0.):
        left, right = np.asarray(a), np.asarray(b)
        error = float(np.max(abs(left-right))/(1+np.max(abs(left))+np.max(abs(right))))
        errors[name] = max(errors.get(name, 0.), error) if np.isfinite(error) else float("inf")
    scale = np.sqrt(np.diag(C))
    for t in range(T):
        for g, group in enumerate(info.signals):
            P, V = info.projectors[t, g], info.posterior_covariances[t, g]
            active = [s.loading for s in group if s.date <= t]
            H = np.stack(active) if active else np.zeros((1, n))
            H = H/np.maximum(np.max(abs(H), axis=1), np.finfo(float).tiny)[:, None]
            white = H@chol
            white /= np.maximum(np.linalg.norm(white, axis=1), np.finfo(float).tiny)[:, None]
            observed_rank = int(np.linalg.matrix_rank(white, tol=1e-12*np.linalg.norm(white, 2)))
            check("rank", np.linalg.matrix_rank(P, tol=1e-8), observed_rank)
            check("recorded_rank", info.ranks[t, g], observed_rank)
            check("idempotence", P@P, P)
            check("observations_reproduced", H@P, H)
            check("conditional_orthogonality", (P@C-C)@H.T)
            check("covariance_self_adjoint", P@C, C@P.T)
            check("posterior_covariance", V/(scale[:, None]*scale), (C-P@C)/(scale[:, None]*scale))
            check("posterior_symmetry", V, V.T)
            check("posterior_psd", min(0., float(np.linalg.eigvalsh(V/(scale[:, None]*scale)).min())))
            if t:
                previous = info.projectors[t-1, g]
                check("individual_tower", previous@P, previous)
                check("nested_information", P@previous, previous)
    maximum = max(errors.values())
    return dict(accepted=bool(np.isfinite(maximum) and maximum <= tol), max_error=maximum,
                checks=errors, tolerance=tol, scope="individual information, not an average tower property")
