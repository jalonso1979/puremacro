"""Spatial econometrics for regional macro.

**Weights and diagnostics**

- :mod:`~puremacro.spatial.weights` — :class:`SpatialWeights` and the builders
  :func:`contiguity_weights`, :func:`knn_weights`, :func:`distance_weights`,
  :func:`economic_weights`; :func:`haversine_km`.
- :mod:`~puremacro.spatial.diagnostics` — :func:`morans_i`, :func:`gearys_c`.
- :mod:`~puremacro.spatial.hac` — :func:`conley_cov`, :func:`conley_se` and the
  space-time :func:`spatial_hac_panel_cov`, also available as
  ``cov_type="conley"`` in :func:`puremacro.lp.panel_lp`.

**Models**

- :mod:`~puremacro.spatial.models` — cross-section :func:`sar` (spatial lag),
  :func:`sem` (spatial error), :func:`sdm` (spatial Durbin) and :func:`slx`
  by concentrated maximum likelihood or Kelejian-Prucha GMM;
  :func:`spatial_effects` for the LeSage-Pace direct / indirect / total
  decomposition; :func:`ols_spatial` and :func:`lm_spatial_tests` for the
  specification battery that chooses between them.
- :mod:`~puremacro.spatial.panel` — :func:`spatial_panel`: SAR / SEM panels
  with individual, time or two-way fixed effects, estimated by QML with the
  Lee-Yu bias correction.
- :mod:`~puremacro.spatial.lp` — :func:`spatial_lp`: panel local projections
  with a spatially lagged shock, reporting direct, indirect and total
  responses at every horizon.

Related estimators that live elsewhere: shift-share inference with
Adão-Kolesár-Morales standard errors is :func:`puremacro.bartik.shift_share_iv`;
spillover-robust difference-in-differences is
:func:`puremacro.did.spatial_did`; the Global VAR is :func:`puremacro.var.gvar`.
US state and county geography for the weights builders ships with the package
as :func:`puremacro.datasets.load_us_state_centroids` and
:func:`~puremacro.datasets.load_us_county_centroids`.

Pure numpy / scipy / pandas / matplotlib: no geometry stack is needed, so the
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
from .models import (
    sar,
    sem,
    sdm,
    slx,
    ols_spatial,
    lm_spatial_tests,
    spatial_effects,
    SpatialModelResult,
    SpatialOLSResult,
    SpatialLMResult,
    SpatialEffectsResult,
)
from .panel import spatial_panel, SpatialPanelResult
from .lp import spatial_lp, SpatialLPResult, higher_order_weights, spatial_lag_panel

__all__ = [
    # weights
    "SpatialWeights",
    "contiguity_weights",
    "knn_weights",
    "distance_weights",
    "economic_weights",
    "haversine_km",
    "pairwise_distances",
    "higher_order_weights",
    "spatial_lag_panel",
    # diagnostics
    "morans_i",
    "gearys_c",
    "MoranResult",
    "GearyResult",
    # spatial HAC
    "conley_cov",
    "conley_se",
    "spatial_hac_panel_cov",
    "spatial_hac_panel_meat",
    "kernel_matrix",
    # cross-section models
    "sar",
    "sem",
    "sdm",
    "slx",
    "ols_spatial",
    "lm_spatial_tests",
    "spatial_effects",
    "SpatialModelResult",
    "SpatialOLSResult",
    "SpatialLMResult",
    "SpatialEffectsResult",
    # panels
    "spatial_panel",
    "SpatialPanelResult",
    # local projections
    "spatial_lp",
    "SpatialLPResult",
]
