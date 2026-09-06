"""Reduced-form VAR + identification + IRF/FEVD/bootstrap."""
from .estimate import estimate_var as fit_var, lag_select, companion, is_stable
from .irf import irf, fevd, gfevd, historical_decomp
from .bootstrap import bootstrap_bands
from .bvar import minnesota_posterior, minnesota_gibbs
from .bvar_sv import bvar_sv, BVAR_SVResult, BVAR_SVForecast
from .peak import peak_summary, peak_distribution
from .favar import favar, FAVARResult
from ._results import VarEstimateResult  # noqa: F401

__all__ = [
    "fit_var", "lag_select", "companion", "is_stable",
    "irf", "fevd", "gfevd", "historical_decomp", "bootstrap_bands",
    "minnesota_posterior", "minnesota_gibbs",
    "bvar_sv", "BVAR_SVResult", "BVAR_SVForecast",
    "peak_summary", "peak_distribution",
    "favar", "FAVARResult",
    "VarEstimateResult",
]
