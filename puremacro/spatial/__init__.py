"""Spatial econometrics for regional macro (phase 1).

- :mod:`~puremacro.spatial.weights` — :class:`SpatialWeights` and the builders
  :func:`contiguity_weights`, :func:`knn_weights`, :func:`distance_weights`,
  :func:`economic_weights`; :func:`haversine_km`.
- :mod:`~puremacro.spatial.diagnostics` — :func:`morans_i`, :func:`gearys_c`.
- :mod:`~puremacro.spatial.hac` — :func:`conley_cov`, :func:`conley_se` and the
  space-time :func:`spatial_hac_panel_cov`, also available as
  ``cov_type="conley"`` in :func:`puremacro.lp.panel_lp`.

Shift-share inference with Adão-Kolesár-Morales standard errors lives in
:func:`puremacro.bartik.shift_share_iv`.

Pure numpy / scipy.sparse / pandas: no geometry stack is needed, so the
package runs under Pyodide. See ``docs/spatial.md``.
"""
from .weights import (
    SpatialWeights,
    contiguity_weights,
    knn_weights,
    distance_weights,
    economic_weights,
    haversine_km,
    pairwise_distances,
)
from .diagnostics import morans_i, gearys_c, MoranResult, GearyResult
from .hac import conley_cov, conley_se, spatial_hac_panel_cov, spatial_hac_panel_meat, kernel_matrix

__all__ = [
    "SpatialWeights",
    "contiguity_weights",
    "knn_weights",
    "distance_weights",
    "economic_weights",
    "haversine_km",
    "pairwise_distances",
    "morans_i",
    "gearys_c",
    "MoranResult",
    "GearyResult",
    "conley_cov",
    "conley_se",
    "spatial_hac_panel_cov",
    "spatial_hac_panel_meat",
    "kernel_matrix",
]
