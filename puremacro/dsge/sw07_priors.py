"""Smets-Wouters (2007) priors for Bayesian estimation.

Mirrors the priors block in puremacro/dsge/_references/sw07_pfeifer.mod.
The PRIORS dict is the canonical source; log_prior(params) sums the
individual scipy.stats logpdfs across all entries, returning -inf if
any parameter is outside its declared [lb, ub] support.

The Pfeifer .mod file estimated_params block uses the format:
    param_name, INITVAL, LB, UB, PRIOR_SHAPE, PRIOR_P1(mean), PRIOR_P2(std);
For shock standard deviations:
    stderr shock_name, INITVAL, LB, UB, INV_GAMMA_PDF, mean, std;

Inverse-gamma parameterisation follows Dynare convention:
    PRIOR_P1 = desired prior mode-area mean (s), PRIOR_P2 = degrees of freedom (nu).
    For Dynare's inv_gamma_pdf: the distribution is IG(nu/2, s^2*nu/2) in scipy's
    invgamma(a=nu/2, scale=s^2*nu/2), giving mean = s^2*nu/(2*(nu/2-1)) = s*nu/(nu-2)
    when nu > 2. In this file we store the Dynare P1/P2 directly and compute
    scipy shape/scale in _logpdf_invgamma.
"""
from __future__ import annotations

from puremacro.dsge import priors as _priors


# === Declarative priors (Pfeifer mirror) =====================================
# Format: {"dist": ..., "mean": P1, "std": P2, "lb": LB, "ub": UB}
# Ordering matches estimated_params block in sw07_pfeifer.mod (lines 364-400).
# For invgamma entries: "mean" = Dynare PRIOR_P1 (scale parameter s),
#                       "std"  = Dynare PRIOR_P2 (degrees-of-freedom nu).

PRIORS: dict[str, dict] = {
    # --- Shock standard deviations (7 stderr entries, lines 364-370) ---
    "ea":         {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01,  "ub": 3.0},
    "eb":         {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.025, "ub": 5.0},
    "eg":         {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01,  "ub": 3.0},
    "eqs":        {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01,  "ub": 3.0},
    "em":         {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01,  "ub": 3.0},
    "epinf":      {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01,  "ub": 3.0},
    "ew":         {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01,  "ub": 3.0},
    # --- AR(1) persistence parameters (9 beta entries, lines 371-379) ---
    "crhoa":      {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.01,  "ub": 0.9999},
    "crhob":      {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.01,  "ub": 0.9999},
    "crhog":      {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.01,  "ub": 0.9999},
    "crhoqs":     {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.01,  "ub": 0.9999},
    "crhoms":     {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.01,  "ub": 0.9999},
    "crhopinf":   {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.01,  "ub": 0.9999},
    "crhow":      {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.001, "ub": 0.9999},
    "cmap":       {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.01,  "ub": 0.9999},
    "cmaw":       {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.01,  "ub": 0.9999},
    # --- Structural parameters (lines 380-399) ---
    "csadjcost":  {"dist": "normal",   "mean": 4.0,   "std": 1.5,   "lb": 2.0,   "ub": 15.0},
    "csigma":     {"dist": "normal",   "mean": 1.50,  "std": 0.375, "lb": 0.25,  "ub": 3.0},
    "chabb":      {"dist": "beta",     "mean": 0.7,   "std": 0.1,   "lb": 0.001, "ub": 0.99},
    "cprobw":     {"dist": "beta",     "mean": 0.5,   "std": 0.1,   "lb": 0.3,   "ub": 0.95},
    "csigl":      {"dist": "normal",   "mean": 2.0,   "std": 0.75,  "lb": 0.25,  "ub": 10.0},
    "cprobp":     {"dist": "beta",     "mean": 0.5,   "std": 0.10,  "lb": 0.5,   "ub": 0.95},
    "cindw":      {"dist": "beta",     "mean": 0.5,   "std": 0.15,  "lb": 0.01,  "ub": 0.99},
    "cindp":      {"dist": "beta",     "mean": 0.5,   "std": 0.15,  "lb": 0.01,  "ub": 0.99},
    "czcap":      {"dist": "beta",     "mean": 0.5,   "std": 0.15,  "lb": 0.01,  "ub": 1.0},
    "cfc":        {"dist": "normal",   "mean": 1.25,  "std": 0.125, "lb": 1.0,   "ub": 3.0},
    "crpi":       {"dist": "normal",   "mean": 1.5,   "std": 0.25,  "lb": 1.0,   "ub": 3.0},
    "crr":        {"dist": "beta",     "mean": 0.75,  "std": 0.10,  "lb": 0.5,   "ub": 0.975},
    "cry":        {"dist": "normal",   "mean": 0.125, "std": 0.05,  "lb": 0.001, "ub": 0.5},
    "crdy":       {"dist": "normal",   "mean": 0.125, "std": 0.05,  "lb": 0.001, "ub": 0.5},
    "constepinf": {"dist": "gamma",    "mean": 0.625, "std": 0.10,  "lb": 0.1,   "ub": 2.0},
    "constebeta": {"dist": "gamma",    "mean": 0.25,  "std": 0.10,  "lb": 0.01,  "ub": 2.0},
    "constelab":  {"dist": "normal",   "mean": 0.0,   "std": 2.0,   "lb": -10.0, "ub": 10.0},
    "ctrend":     {"dist": "normal",   "mean": 0.4,   "std": 0.10,  "lb": 0.1,   "ub": 0.8},
    "cgy":        {"dist": "normal",   "mean": 0.5,   "std": 0.25,  "lb": 0.01,  "ub": 2.0},
    "calfa":      {"dist": "normal",   "mean": 0.3,   "std": 0.05,  "lb": 0.01,  "ub": 1.0},
}


# === Public API (delegators to puremacro.dsge.priors) =========================

def log_prior(params: dict) -> float:
    """Sum of log-prior densities across the SW07 estimated parameters."""
    return _priors.log_prior(params, PRIORS)


def prior_means() -> dict[str, float]:
    """Return a dict of {param_name: prior_mean} for all SW07 estimated params."""
    return _priors.prior_means(PRIORS)


def prior_stds() -> dict[str, float]:
    """Return a dict of {param_name: prior_std} for all SW07 estimated params."""
    return _priors.prior_stds(PRIORS)


def param_bounds() -> list[tuple[float, float]]:
    """Return list of (lb, ub) tuples in PRIORS dict order."""
    return _priors.param_bounds(PRIORS)


def param_names() -> tuple[str, ...]:
    """Return parameter names in PRIORS dict order."""
    return _priors.param_names(PRIORS)


__all__ = [
    "PRIORS",
    "log_prior",
    "prior_means",
    "prior_stds",
    "param_bounds",
    "param_names",
]
