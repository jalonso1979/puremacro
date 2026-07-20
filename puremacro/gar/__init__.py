"""Growth-at-Risk and conditional-distribution macro toolkit.

Public API
----------
Quantile regressions:
    qar                       (quantile autoregression, Koenker-Xiao 2006)
    lp_quantile               (re-exported from :mod:`puremacro.lp.quantile`,
                                Adrian-Boyarchenko-Giannone-style quantile LP)

Skew-t conditional density (ABG 2019 vulnerable-growth pipeline):
    fit_skewt_to_quantiles, SkewTFit
    skewt_pdf, skewt_cdf, skewt_ppf

Financial Conditions Index:
    fci, fci_rolling
"""
from .qar import qar
from .skewt import (
    fit_skewt_to_quantiles, SkewTFit,
    skewt_pdf, skewt_cdf, skewt_ppf,
)
from .fci import fci, fci_rolling

# Re-export the panel quantile-LP for one-stop importing.
from ..lp.quantile import lp_quantile

__all__ = [
    "qar", "lp_quantile",
    "fit_skewt_to_quantiles", "SkewTFit",
    "skewt_pdf", "skewt_cdf", "skewt_ppf",
    "fci", "fci_rolling",
]
