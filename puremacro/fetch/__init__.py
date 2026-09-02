"""puremacro.fetch — public-data fetchers.

Exposes fetchers for major macro/financial data sources, all going
through the unified :mod:`puremacro._http` helpers (UA override,
one-shot SSL fallback, 30s default timeout). Each fetcher returns a
``pandas`` object suitable for direct use in VAR / LP estimation.

Public API
----------
- :func:`fetch_fred`        — FRED public CSV (no API key needed)
- :func:`fetch_fred_alfred` — ALFRED real-time vintages (one series)
- :func:`alfred_vintages`   — every archived edition of one ALFRED series
- :func:`vintage_panel`     — one-call cross-country panel of published
                              editions, across six real-time providers
- :func:`vintage_catalog`   — which series backs which (country, variable)
- :func:`qna_long_panel`    — the OECD spine extended backwards per country by
                              ratio-splicing archived national vintages onto it
                              (Spain to 1970, Japan to 1955)
- :func:`qna_panel`         — one-call cross-country quarterly national accounts
                              (nominal SA levels + implicit deflators, OECD QNA)
- :func:`qna_labor`         — the QNA labour block alone (employment, hours)
- :func:`qna_countries`     — every reference area that flow actually covers
- :func:`qna_rebase`        — put every country on one price reference year
- :func:`qna_identity`      — score all three GDP identities the panel carries
- :func:`qna_contributions` — component contributions to real GDP growth
- :func:`fetch_xrate_monthly` — OECD nominal exchange rates (LCU per USD), monthly
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
from .oecd_qna_panel import (
    qna_panel,
    qna_labor,
    qna_meta,
    qna_countries,
    QNA_AGGREGATES,
    QNA_ACTIVITIES,
    QNA_INCOME,
    QNA_VA_ADDITIVE,
    QNA_VA_MEMO,
    QNA_COMPONENTS,
    QNA_ASSETS,
    QNA_DURABILITY,
    QNA_LABOR,
    QNA_LABOR_UNITS,
    QNA_LABOR_ACTIVITIES,
)
from .oecd_ana_activity import (
    ana_by_activity,
    ana_meta,
    ana_hours_wedge,
    chain_volume,
    ANA_ACTIVITIES,
    ANA_LABOR,
    ANA_LABOR_UNITS,
)
from .oecd_fx import fetch_xrate_monthly
from .oecd_qna_tools import (
    qna_rebase,
    qna_identity,
    qna_contributions,
    IDENTITY_TERMS,
    OUTPUT_TERMS,
    INCOME_TERMS,
    APPROACH_GDP,
)
from .qna_vintages import (
    fetch_qna_vintages,
    QNAVintagePanel,
    get_qna_vintage_catalog,
)
from .longpanel import (
    qna_long_panel,
    long_panel_residual,
    LONG_PANEL_SOURCES,
    KNOWN_GAPS as LONG_PANEL_KNOWN_GAPS,
)
from .realtime import (
    alfred_vintages,
    available_providers,
    vintage_catalog,
    vintage_panel,
    seasonal_signature,
    drop_unadjusted_editions,
    VintagePanel,
    VINTAGE_SEMANTICS,
)

__all__ = [
    "fetch_fred",
    "fetch_fred_alfred",
    "alfred_vintages",
    "vintage_panel",
    "seasonal_signature",
    "drop_unadjusted_editions",
    "qna_long_panel",
    "long_panel_residual",
    "LONG_PANEL_SOURCES",
    "LONG_PANEL_KNOWN_GAPS",
    "VintagePanel",
    "vintage_catalog",
    "available_providers",
    "VINTAGE_SEMANTICS",
    "qna_panel",
    "qna_labor",
    "qna_meta",
    "ana_by_activity",
    "ana_meta",
    "ana_hours_wedge",
    "chain_volume",
    "ANA_ACTIVITIES",
    "ANA_LABOR",
    "ANA_LABOR_UNITS",
    "QNA_LABOR_ACTIVITIES",
    "qna_countries",
    "QNA_AGGREGATES",
    "qna_rebase",
    "qna_identity",
    "qna_contributions",
    "IDENTITY_TERMS",
    "OUTPUT_TERMS",
    "INCOME_TERMS",
    "APPROACH_GDP",
    "QNA_ACTIVITIES",
    "QNA_INCOME",
    "QNA_VA_ADDITIVE",
    "QNA_VA_MEMO",
    "QNA_COMPONENTS",
    "QNA_ASSETS",
    "QNA_DURABILITY",
    "QNA_LABOR",
    "QNA_LABOR_UNITS",
    "fetch_xrate_monthly",
    "fetch_qna_vintages",
    "QNAVintagePanel",
    "get_qna_vintage_catalog",
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
