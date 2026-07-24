"""Seasonal adjustment helpers.

Three backends are exported:

* :func:`deseasonalize` -- STL (no external binary; works everywhere).
* :func:`deseasonalize_x13` -- the external X-13ARIMA-SEATS binary
  (preferred when ``x13as`` is on PATH or pointed to by ``X13PATH``);
  falls back to STL per-unit when X-13 declines.
* :func:`deseasonalize_x11` -- the NATIVE X-11/ARIMA engine
  (:mod:`puremacro.sa.x11`): pure Python, Pyodide-safe, validated
  against the real X-13 binary via frozen goldens (docs/VALIDATION.md).
  Use this where the binary cannot exist (browser) or a reproducible
  in-package method is wanted.

Default at the package level is X-13 (with STL fallback inside).
"""
from .stl import deseasonalize as deseasonalize_stl
from .stl import residual_seasonality_F
from .x11 import X11Result, deseasonalize_x11, henderson_weights, x11_arima
from .x13 import deseasonalize_x13, x13_available

# Default: prefer X-13 (it falls back to STL when needed).
deseasonalize = deseasonalize_x13

__all__ = [
    "X11Result",
    "deseasonalize",
    "deseasonalize_stl",
    "deseasonalize_x11",
    "deseasonalize_x13",
    "henderson_weights",
    "residual_seasonality_F",
    "x11_arima",
    "x13_available",
]
