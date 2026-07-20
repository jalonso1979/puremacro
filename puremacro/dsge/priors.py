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


_DIST_LOGPDF = {
    "beta":     _logpdf_beta,
    "gamma":    _logpdf_gamma,
    "normal":   _logpdf_normal,
    "invgamma": _logpdf_invgamma,
}


def _logpdf_for_spec(spec: dict, x: float) -> float:
    """Dispatch to the correct log-pdf given a single param spec and a value."""
    if not (spec["lb"] <= x <= spec["ub"]):
        return -math.inf
    dist = spec["dist"]
    try:
        fn = _DIST_LOGPDF[dist]
    except KeyError as exc:
        raise ValueError(f"unknown distribution {dist!r} for prior spec") from exc
    return fn(x, spec["mean"], spec["std"])


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
    "log_prior",
    "prior_means",
    "prior_stds",
    "param_bounds",
    "param_names",
]
