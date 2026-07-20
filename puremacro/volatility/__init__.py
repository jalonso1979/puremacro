"""Volatility / second-moment toolkit.

Public API
----------
Sector covariance as object (port of ``MAV/SigmaObject.m``):
    SigmaObject, project_psd, clean_correlation

Multivariate GARCH:
    bekk_fit, ccc_fit              (DCC lives in :mod:`puremacro.garch`)

Realised-volatility regressions:
    har_rv

Range-based volatility (OHLC → variance):
    parkinson, garman_klass, rogers_satchell

Mis-specification diagnostics:
    arch_lm_test, ljung_box_squared
"""
from .sigma import SigmaObject, project_psd, clean_correlation
from .multivariate import bekk_fit, ccc_fit
from .har import har_rv
from .range import parkinson, garman_klass, rogers_satchell
from .diagnostics import arch_lm_test, ljung_box_squared

__all__ = [
    "SigmaObject", "project_psd", "clean_correlation",
    "bekk_fit", "ccc_fit",
    "har_rv",
    "parkinson", "garman_klass", "rogers_satchell",
    "arch_lm_test", "ljung_box_squared",
]
