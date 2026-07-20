"""High-frequency identification of monetary policy shocks.

Provides:
    - Surprise construction: Gertler-Karadi (2015), Nakamura-Steinsson (2018).
    - Jarociński-Karadi (2020) monetary-vs-information decomposition.

For external-IV SVAR, pipe surprises into
:func:`puremacro.var.identify.proxy.proxy_svar` directly.
"""
from ._results import JKResult
from .jk2020 import jk_median_target, jk_poor_man
from .surprises import aggregate_to_period, gk2015_surprise, ns2018_first_pc

__all__ = [
    "JKResult",
    "aggregate_to_period",
    "gk2015_surprise",
    "jk_median_target",
    "jk_poor_man",
    "ns2018_first_pc",
]
