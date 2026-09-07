"""Reduced-form VAR + identification + IRF/FEVD/bootstrap.

The cross-country counterpart is :func:`gvar`, the Global VAR of Pesaran,
Schuermann and Weiner (2004): country VARX* models linked by trade weights
and solved into one global system, with generalised IRFs, a generalised
FEVD, persistence profiles and a weak-exogeneity test.
"""
from .estimate import estimate_var as fit_var, lag_select, companion, is_stable
from .irf import irf, fevd, gfevd, historical_decomp
from .bootstrap import bootstrap_bands
from .bvar import minnesota_posterior, minnesota_gibbs
from .bvar_sv import bvar_sv, BVAR_SVResult, BVAR_SVForecast
from .peak import peak_summary, peak_distribution
from .favar import favar, FAVARResult
from .gvar import (
    gvar,
    solve_gvar,
    star_variables,
    GVARResult,
    CountryVARX,
    GVARGIRFResult,
    WeakExogeneityResult,
)
from ._results import VarEstimateResult  # noqa: F401

__all__ = [
    "fit_var", "lag_select", "companion", "is_stable",
    "irf", "fevd", "gfevd", "historical_decomp", "bootstrap_bands",
    "minnesota_posterior", "minnesota_gibbs",
    "bvar_sv", "BVAR_SVResult", "BVAR_SVForecast",
    "peak_summary", "peak_distribution",
    "favar", "FAVARResult",
    "gvar", "solve_gvar", "star_variables",
    "GVARResult", "CountryVARX", "GVARGIRFResult", "WeakExogeneityResult",
    "VarEstimateResult",
]
