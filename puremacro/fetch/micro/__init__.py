"""Household and person microdata with their survey designs.

Every provider returns a :class:`MicroFrame`: the records, the
:class:`SurveyDesign` that says how to get standard errors from them,
and the request that produced them (keys stripped). Estimators on the
frame (``mean``, ``total``, ``quantile``, ``share``) return estimate,
standard error and unweighted ``n`` together.

Providers
---------
- :func:`fetch_acs_pums`  — Census ACS PUMS (80 replicate weights)
- :func:`fetch_cps_basic` — Census CPS basic monthly (weights only)
- :func:`fetch_scf`       — Fed Survey of Consumer Finances (5 implicates,
  999 replicate weights)

See :func:`puremacro.fetch.registry.sources` with ``kind="micro"`` for
access terms. Provider modules import the network layer; this package
loads them lazily so the design and estimators stay Pyodide-clean.
"""
from __future__ import annotations

import importlib

from ._design import METHODS, MicroFrame, SurveyDesign, replicate_variance, rubin_combine

_MODULE_OF = {
    "fetch_acs_pums": "census",
    "fetch_cps_basic": "census",
    "fetch_scf": "scf",
}


def __getattr__(name):
    if name in _MODULE_OF:
        return getattr(importlib.import_module(f".{_MODULE_OF[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MicroFrame", "SurveyDesign", "METHODS", "replicate_variance",
           "rubin_combine", *_MODULE_OF]
