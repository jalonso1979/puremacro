"""Model-agnostic prior framework for Bayesian DSGE estimation.

A `priors` dict has the shape:
    {param_name: {"dist": str, "mean": float, "std": float, "lb": float, "ub": float}}

Supported distributions: ``"beta"``, ``"gamma"``, ``"normal"``, ``"invgamma"``
(Dynare ``inv_gamma_pdf(P1=s, P2=nu)`` convention).

This module is the engine-side complement to model-specific prior dicts
like ``puremacro.dsge.sw07_priors.PRIORS``.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def _logpdf_beta(x: float, mean: float, std: float) -> float:
    """log-pdf of Beta(a, b) parameterised by mean/std (Pfeifer convention)."""
    a = mean * (mean * (1 - mean) / std**2 - 1)
    b = a * (1 - mean) / mean
    return float(stats.beta.logpdf(x, a, b))


def _logpdf_gamma(x: float, mean: float, std: float) -> float:
    """log-pdf of Gamma(k, theta) parameterised by mean/std."""
    if x <= 0.0:
        return -math.inf
    k = (mean / std) ** 2
    theta = std ** 2 / mean
    return float(stats.gamma.logpdf(x, a=k, scale=theta))


def _logpdf_normal(x: float, mean: float, std: float) -> float:
    """log-pdf of Normal(mean, std)."""
    return float(stats.norm.logpdf(x, loc=mean, scale=std))


def _logpdf_invgamma(x: float, s: float, nu: float) -> float:
    """log-pdf of Inverse-Gamma under Dynare's inv_gamma_pdf parameterisation.

    Dynare's inv_gamma_pdf(P1=s, P2=nu) corresponds to:
        IG(shape=nu/2, scale=s^2 * nu/2)
    """
    if x <= 0.0:
        return -math.inf
    a = nu / 2.0
    scale = s ** 2 * nu / 2.0
    return float(stats.invgamma.logpdf(x, a=a, scale=scale))


def _logpdf_uniform(x: float, mean: float, std: float) -> float:
    """log-pdf of Uniform distribution parameterized by mean and std."""
    width = math.sqrt(12.0) * std
    return -math.log(max(width, 1e-12))


_DIST_LOGPDF = {
    "beta":     _logpdf_beta,
    "gamma":    _logpdf_gamma,
    "normal":   _logpdf_normal,
    "invgamma": _logpdf_invgamma,
    "uniform":  _logpdf_uniform,
}


def _logpdf_for_spec(spec: dict | Prior, x: float) -> float:
    """Dispatch to the correct log-pdf given a single param spec and a value."""
    lb = spec["lb"]
    ub = spec["ub"]
    if not (lb <= x <= ub):
        return -math.inf
    dist = spec["dist"]
    if dist == "uniform":
        return -math.log(max(ub - lb, 1e-12))
    try:
        fn = _DIST_LOGPDF[dist]
    except KeyError as exc:
        raise ValueError(f"unknown distribution {dist!r} for prior spec") from exc
    return fn(x, spec["mean"], spec["std"])


class Prior:
    """Base class for Bayesian prior distributions in DSGE models."""

    def __init__(
        self,
        dist: str,
        mean: float,
        std: float,
        lb: float = -math.inf,
        ub: float = math.inf,
    ) -> None:
        self.dist = dist
        self.mean = float(mean)
        self.std = float(std)
        self.lb = float(lb)
        self.ub = float(ub)

    def logpdf(self, x: float | np.ndarray) -> float | np.ndarray:
        if isinstance(x, (int, float, np.floating)):
            return _logpdf_for_spec(self, float(x))
        arr = np.asarray(x, dtype=float)
        out = np.empty_like(arr)
        for idx, val in np.ndenumerate(arr):
            out[idx] = _logpdf_for_spec(self, float(val))
        return out

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return np.exp(self.logpdf(x))

    def __getitem__(self, key: str):
        if key == "dist":
            return self.dist
        if key == "mean":
            return self.mean
        if key == "std":
            return self.std
        if key == "lb":
            return self.lb
        if key == "ub":
            return self.ub
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in ("dist", "mean", "std", "lb", "ub")

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict:
        return {
            "dist": self.dist,
            "mean": self.mean,
            "std": self.std,
            "lb": self.lb,
            "ub": self.ub,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, "
            f"lb={self.lb}, ub={self.ub})"
        )


class BetaPrior(Prior):
    """Beta prior parameterized by mean and std (Dynare/Pfeifer convention)."""

    def __init__(
        self,
        mean: float = 0.5,
        std: float = 0.1,
        lb: float = 1e-4,
        ub: float = 0.9999,
    ) -> None:
        super().__init__("beta", mean, std, lb, ub)


class InvGammaPrior(Prior):
    """Inverse-Gamma prior under Dynare inv_gamma_pdf convention (P1=s, P2=nu)."""

    def __init__(
        self,
        mean: float | None = None,
        std: float | None = None,
        lb: float = 1e-4,
        ub: float = math.inf,
        *,
        s: float | None = None,
        nu: float | None = None,
    ) -> None:
        val_s = s if s is not None else (mean if mean is not None else 0.1)
        val_nu = nu if nu is not None else (std if std is not None else 2.0)
        super().__init__("invgamma", val_s, val_nu, lb, ub)

    @property
    def s(self) -> float:
        return self.mean

    @property
    def nu(self) -> float:
        return self.std


class NormalPrior(Prior):
    """Normal prior parameterized by mean and std."""

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        lb: float = -math.inf,
        ub: float = math.inf,
    ) -> None:
        super().__init__("normal", mean, std, lb, ub)


class GammaPrior(Prior):
    """Gamma prior parameterized by mean and std."""

    def __init__(
        self,
        mean: float = 1.0,
        std: float = 0.5,
        lb: float = 1e-4,
        ub: float = math.inf,
    ) -> None:
        super().__init__("gamma", mean, std, lb, ub)


class UniformPrior(Prior):
    """Uniform prior on [lb, ub]."""

    def __init__(self, lb: float = 0.0, ub: float = 1.0) -> None:
        mean = (lb + ub) / 2.0
        std = (ub - lb) / math.sqrt(12.0)
        super().__init__("uniform", mean, std, lb, ub)


def ensure_prior(spec: dict | Prior) -> Prior:
    """Convert dict prior spec or return Prior instance."""
    if isinstance(spec, Prior):
        return spec
    dist = spec["dist"].lower()
    mean = spec.get("mean", 0.0)
    std = spec.get("std", 1.0)
    lb = spec.get("lb", -math.inf)
    ub = spec.get("ub", math.inf)
    if dist == "beta":
        return BetaPrior(mean=mean, std=std, lb=lb, ub=ub)
    elif dist in ("invgamma", "inv_gamma"):
        return InvGammaPrior(mean=mean, std=std, lb=lb, ub=ub)
    elif dist == "normal":
        return NormalPrior(mean=mean, std=std, lb=lb, ub=ub)
    elif dist == "gamma":
        return GammaPrior(mean=mean, std=std, lb=lb, ub=ub)
    elif dist == "uniform":
        return UniformPrior(lb=lb, ub=ub)
    return Prior(dist, mean, std, lb, ub)


def log_prior(params: dict, priors: dict) -> float:
    """Sum of log-prior densities across all parameters in ``priors``.

    Returns -inf if any parameter value is missing, non-finite, or outside
    its declared ``[lb, ub]`` support. Raises ``ValueError`` if any spec
    declares an unsupported ``dist``.
    """
    total = 0.0
    for name, spec in priors.items():
        if name not in params:
            return -math.inf
        x = params[name]
        if not math.isfinite(x):
            return -math.inf
        contrib = _logpdf_for_spec(spec, x)
        if contrib == -math.inf:
            return -math.inf
        total += contrib
    return total


def prior_means(priors: dict) -> dict[str, float]:
    """Return ``{name: mean}`` in priors-dict insertion order."""
    return {name: spec["mean"] for name, spec in priors.items()}


def prior_stds(priors: dict) -> dict[str, float]:
    """Return ``{name: std}`` in priors-dict insertion order."""
    return {name: spec["std"] for name, spec in priors.items()}


def param_bounds(priors: dict) -> list[tuple[float, float]]:
    """Return ``[(lb, ub), ...]`` in priors-dict insertion order.

    Matches the shape scipy.optimize.minimize expects for ``bounds``.
    """
    return [(spec["lb"], spec["ub"]) for spec in priors.values()]


def param_names(priors: dict) -> tuple[str, ...]:
    """Return parameter names in priors-dict insertion order."""
    return tuple(priors.keys())


__all__ = [
    "Prior",
    "BetaPrior",
    "InvGammaPrior",
    "NormalPrior",
    "GammaPrior",
    "UniformPrior",
    "ensure_prior",
    "log_prior",
    "prior_means",
    "prior_stds",
    "param_bounds",
    "param_names",
]
