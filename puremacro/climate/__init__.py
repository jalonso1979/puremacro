"""Climate × fertility primitives extracted from My Drive/Fertility/climate_fertility.

The source project remains the canonical full-pipeline implementation
(including xarray-based weather loaders, geopandas zonal aggregation,
and country-specific runners). This subpackage exposes only the
Pyodide-compatible estimator primitives:

- degree-days (CDD / HDD construction from monthly temperatures)
- annual climate-shock LP (paired CDD + HDD, Driscoll-Kraay HAC SE)
- within-year-quintile mediation LP
- monthly distributed-lag estimator (HC1 single-region; cluster panel)
"""
from .degree_days import compute_monthly_cdd_hdd, compute_annual_cdd_hdd
from .annual_lp import climate_annual_lp
from .mediation import climate_mediation_lp
from .monthly_dl import monthly_dl, make_dl_lags

__all__ = [
    "compute_monthly_cdd_hdd",
    "compute_annual_cdd_hdd",
    "climate_annual_lp",
    "climate_mediation_lp",
    "monthly_dl",
    "make_dl_lags",
]
