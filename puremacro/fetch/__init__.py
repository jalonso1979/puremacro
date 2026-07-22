"""puremacro.fetch — public-data fetchers.

Exposes fetchers for major macro/financial data sources, all going
through the unified :mod:`puremacro._http` helpers (UA override,
one-shot SSL fallback, 30s default timeout). Each fetcher returns a
``pandas`` object suitable for direct use in VAR / LP estimation.

Public API
----------
- :func:`fetch_fred`        — FRED public CSV (no API key needed)
- :func:`fetch_fred_alfred` — ALFRED real-time vintages
- :func:`sdmx_get`          — generic SDMX-CSV (OECD, Eurostat, ECB, IMF SDMX Central)
- :func:`oecd_sdmx_instrument` — convenience wrapper that returns
                                 :class:`puremacro.instruments.Instrument` directly

For API-key-requiring FRED via the JSON endpoint, see
:func:`puremacro.instruments.external.load_fred`.
"""
from ._classic import (
    _safe_urlopen,
    fetch_fred,
    fetch_fred_alfred,
)
from .sdmx import (
    sdmx_get,
    oecd_sdmx_instrument,
)
from .labor import (
    fetch_oecd_lfs_panel,
    fetch_labor_panel_union,
    fetch_sectoral_panel_union,
)
from .labor_eurostat import fetch_eurostat_lfs_panel
from .labor_ilostat import (
    fetch_ilostat_lfs_panel,
    fetch_ilostat_sectoral_panel,
)
from .jolts import fetch_jolts
from .vacancies_eurostat import fetch_eurostat_vacancies

__all__ = [
    "fetch_fred",
    "fetch_fred_alfred",
    "sdmx_get",
    "oecd_sdmx_instrument",
    "fetch_oecd_lfs_panel",
    "fetch_eurostat_lfs_panel",
    "fetch_labor_panel_union",
    "fetch_ilostat_lfs_panel",
    "fetch_ilostat_sectoral_panel",
    "fetch_sectoral_panel_union",
    "fetch_jolts",
    "fetch_eurostat_vacancies",
]
