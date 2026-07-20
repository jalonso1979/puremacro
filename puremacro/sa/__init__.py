"""Seasonal adjustment helpers.

Two backends are exported:

* :func:`deseasonalize` -- STL (no external binary; works everywhere).
* :func:`deseasonalize_x13` -- X-13ARIMA-SEATS (preferred when the
  ``x13as`` binary is on PATH or pointed to by ``X13PATH``); falls back
  to STL per-unit when X-13 declines.

Default at the package level is X-13 (with STL fallback inside).
"""
from .stl import deseasonalize as deseasonalize_stl
from .stl import residual_seasonality_F
from .x13 import deseasonalize_x13, x13_available

# Default: prefer X-13 (it falls back to STL when needed).
deseasonalize = deseasonalize_x13

__all__ = [
    "deseasonalize",
    "deseasonalize_stl",
    "deseasonalize_x13",
    "residual_seasonality_F",
    "x13_available",
]
