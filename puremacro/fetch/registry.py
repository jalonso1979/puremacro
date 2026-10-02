"""One table of every data source puremacro can reach.

Before this module the only way to discover a fetcher was the
``fetch/__init__.py`` docstring, and nothing recorded the facts a
researcher needs *before* calling one: does it need a key, may the data
be redistributed, what shape comes back. This is that table.

It follows the pattern of :mod:`puremacro.fetch.realtime.catalog` — a
declarative table, a ``register_*`` hook for additions, a ``pytest -m
network`` audit (``tests/test_registry_live.py``) — but not its type.
``SeriesSpec`` is keyed country -> variable -> series id, which is the
right shape for vintage archives and the wrong one for microdata or text.

THE REGISTRY DESCRIBES, IT DOES NOT WRAP
----------------------------------------
Existing fetchers keep their own signatures; an entry only records where
the callable lives and what it needs. :func:`load` imports it lazily, so
``import puremacro.fetch.registry`` pulls in no network stack and stays
importable under Pyodide. A common call signature is imposed only on
microdata providers (:mod:`puremacro.fetch.micro`), where the returned
object — weights plus a variance design — is the point.

Fields
------
``auth`` is one of ``"none"``, ``"key_optional"``, ``"key"``,
``"registration"`` (free account, terms accepted per collection),
``"approval"`` (project proposal reviewed by the provider), or
``"unaudited"`` (text connectors whose terms have not been reviewed one
by one yet). ``redistributable`` decides whether results may ever be
written into shipped datasets or ``.pmz`` cartridges.

Example
-------
>>> from puremacro.fetch import registry
>>> registry.sources("micro")[["id", "auth", "redistributable"]]  # doctest: +SKIP
>>> acs = registry.load("census.acs_pums")                         # doctest: +SKIP
"""
from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, fields
from typing import Callable

import pandas as pd

KINDS = ("series", "panel", "vintage", "micro", "text")
AUTH_LEVELS = ("none", "key_optional", "key", "registration", "approval",
               "unaudited")


@dataclass(frozen=True)
class SourceInfo:
    """What one source is, what it needs, and where its callable lives.

    ``loader`` is ``"module.path:attribute"``; :func:`load` resolves it on
    demand. ``credential`` names an entry of
    :data:`puremacro.credentials.SERVICES`.
    """
    id: str
    kind: str
    provider: str
    loader: str
    description: str = ""
    credential: str | None = None
    auth: str = "none"
    terms_url: str = ""
    redistributable: bool = True
    freq: str = ""
    note: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"{self.id}: kind {self.kind!r} not in {KINDS}")
        if self.auth not in AUTH_LEVELS:
            raise ValueError(f"{self.id}: auth {self.auth!r} not in {AUTH_LEVELS}")
        if ":" not in self.loader:
            raise ValueError(f"{self.id}: loader must be 'module:attribute'")


_REGISTRY: dict[str, SourceInfo] = {}


def register(info: SourceInfo, *, replace: bool = False) -> SourceInfo:
    """Add ``info`` to the registry.

    Raises on a duplicate id unless ``replace=True``: two connectors
    silently claiming one id is how a user ends up calling the wrong one.
    """
    if info.id in _REGISTRY and not replace:
        raise ValueError(f"source {info.id!r} is already registered")
    _REGISTRY[info.id] = info
    return info


def info(source_id: str) -> SourceInfo:
    """The :class:`SourceInfo` for ``source_id``."""
    _ensure_text()
    try:
        return _REGISTRY[source_id]
    except KeyError:
        raise KeyError(
            f"unknown source {source_id!r}; see puremacro.fetch.registry.sources()"
        ) from None


def load(source_id: str) -> Callable:
    """Import and return the callable behind ``source_id``."""
    module, _, attr = info(source_id).loader.partition(":")
    return getattr(importlib.import_module(module), attr)


def sources(kind: str | None = None, *, provider: str | None = None) -> pd.DataFrame:
    """Every registered source as a frame, optionally filtered."""
    if kind is not None and kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    _ensure_text()
    rows = [asdict(i) for i in _REGISTRY.values()
            if (kind is None or i.kind == kind)
            and (provider is None or i.provider == provider)]
    cols = [f.name for f in fields(SourceInfo)]
    return (pd.DataFrame(rows, columns=cols)
            .sort_values(["kind", "id"], ignore_index=True))


# ---------------------------------------------------------------------------
# Built-in entries
# ---------------------------------------------------------------------------
_F = "puremacro.fetch."
_FRED_TERMS = "https://fred.stlouisfed.org/legal/"


def _s(id, provider, loader, description, kind="series", **kw):
    register(SourceInfo(id=id, kind=kind, provider=provider,
                        loader=_F + loader, description=description, **kw))


# --- FRED / ALFRED ---------------------------------------------------------
_s("fred.csv", "fred", "_classic:fetch_fred",
   "FRED series via the public CSV endpoint", terms_url=_FRED_TERMS)
_s("fred.api", "fred", "fred:fetch_series",
   "FRED series via the official API (fredapi)", credential="fred",
   auth="key", terms_url=_FRED_TERMS)
_s("fred.alfred", "fred", "_classic:fetch_fred_alfred",
   "ALFRED real-time vintages of one series", kind="vintage",
   terms_url=_FRED_TERMS)
_s("fred.states_employment", "fred", "fred_states:fetch_state_employment",
   "US state nonfarm payroll employment", kind="panel", credential="fred",
   auth="key", freq="M")
_s("fred.states_income", "fred", "fred_states:fetch_state_income",
   "US state personal income", kind="panel", credential="fred", auth="key")
_s("frb_phil.coincident", "frb_phil", "frb_phil_coincident:fetch_state_coincident",
   "Philadelphia Fed state coincident indexes (via FRED)", kind="panel",
   credential="fred", auth="key", freq="M")
_s("bls.state_urate", "fred", "bls_state_panel:iter_state_urate_q",
   "State unemployment rates, quarterly (BLS via FRED CSV)", kind="panel", freq="Q")
_s("bls.jolts", "fred", "jolts:fetch_jolts",
   "JOLTS openings, hires, separations (FRED mirror)", freq="M")
_s("census.bfs", "census", "census_bfs:fetch",
   "Business Formation Statistics, state monthly (via FRED)", kind="panel",
   credential="fred", auth="key", freq="M",
   note="Authenticates with the FRED key, not the Census key.")

# --- BLS / BEA / Census / CDC ---------------------------------------------
_s("bls.laus_state", "bls", "laus:fetch_state",
   "LAUS state labour force and unemployment", kind="panel", freq="M")
_s("bls.laus_county", "bls", "laus:fetch_county",
   "LAUS county labour force and unemployment", kind="panel", freq="M")
_s("bls.ces_states", "bls", "ces_states:fetch",
   "CES state employment, hours, earnings by supersector", kind="panel", freq="M")
_s("bls.qcew", "bls", "qcew:fetch",
   "QCEW county x NAICS x quarter employment and wages", kind="panel", freq="Q")
_s("bea.cainc4", "bea", "bea_cainc:fetch_cainc4",
   "BEA CAINC4 local-area wages and salaries", kind="panel", credential="bea",
   auth="key", freq="A")
_s("bea.industry_shares", "bea", "bea_industry_shares:fetch_shares",
   "BEA state-by-industry employment shares", kind="panel", credential="bea",
   auth="key", freq="A")
_s("census.pep_births_county", "census", "census_pep_births:fetch_county_year",
   "Census PEP county births", kind="panel", freq="A")
_s("census.pep_births_state", "census", "census_pep_births:fetch_state_year",
   "Census PEP state births", kind="panel", freq="A")
_s("cdc.births_county", "cdc", "cdc_births_county:fetch",
   "NCHS county-year births (NBER mirror)", kind="panel", freq="A")
_s("cdc.births_state", "cdc", "cdc_births_state:fetch_monthly",
   "State x month US births", kind="panel", freq="M")
_s("cdc.natality_socrata", "cdc", "cdc_socrata_natality:fetch_state_month",
   "NCHS state x month natality via Socrata (2023+)", kind="panel", freq="M")

# --- OECD / SDMX / Eurostat / ILO / IMF / World Bank / BIS -----------------
_s("sdmx.generic", "sdmx", "sdmx:sdmx_get",
   "Generic SDMX-CSV (OECD, Eurostat, ECB, IMF SDMX Central)")
_s("oecd.qna_panel", "oecd", "oecd_qna_panel:qna_panel",
   "Cross-country quarterly national accounts, nominal SA + deflators",
   kind="panel", freq="Q")
_s("oecd.qna_labor", "oecd", "oecd_qna_panel:qna_labor",
   "QNA labour block (employment, hours)", kind="panel", freq="Q")
_s("oecd.qna_long_panel", "oecd", "longpanel.panel:qna_long_panel",
   "OECD QNA spine extended backwards with national vintages", kind="panel",
   freq="Q")
_s("oecd.qna_vintages", "oecd", "qna_vintages:fetch_qna_vintages",
   "Multi-country QNA from historical vintages", kind="vintage", freq="Q")
_s("oecd.ana_activity", "oecd", "oecd_ana_activity:ana_by_activity",
   "Annual national accounts by activity", kind="panel", freq="A")
_s("oecd.fx_monthly", "oecd", "oecd_fx:fetch_xrate_monthly",
   "Nominal exchange rates, LCU per USD", kind="panel", freq="M")
_s("oecd.stes", "oecd", "oecd_mei:fetch",
   "OECD Short-Term Economic Statistics", kind="panel", freq="M")
_s("oecd.energy_cpi", "oecd", "oecd_energy:fetch_energy_cpi",
   "CPI energy sub-component (COICOP CP045)", kind="panel", freq="M")
_s("oecd.lfs_panel", "oecd", "labor:fetch_oecd_lfs_panel",
   "OECD LFS sex x age x measure panel", kind="panel")
_s("eurostat.lfs_panel", "eurostat", "labor_eurostat:fetch_eurostat_lfs_panel",
   "Eurostat LFS unemployment, employment and participation", kind="panel")
_s("eurostat.vacancies", "eurostat", "vacancies_eurostat:fetch_eurostat_vacancies",
   "Eurostat job-vacancy statistics", kind="panel", freq="Q")
_s("ilostat.lfs_panel", "ilostat", "labor_ilostat:fetch_ilostat_lfs_panel",
   "ILOSTAT sex x age labour panel", kind="panel")
_s("ilostat.sectoral_panel", "ilostat", "labor_ilostat:fetch_ilostat_sectoral_panel",
   "ILOSTAT sex x ISIC sectoral panel", kind="panel")
_s("imf.ifs", "imf", "imf_ifs:fetch",
   "IMF International Financial Statistics, monthly", kind="panel", freq="M")
_s("worldbank.emissions", "worldbank", "emissions:fetch_wdi_emissions",
   "WDI greenhouse gas emissions", kind="panel", freq="A")
_s("oecd.ghg", "oecd", "emissions:fetch_oecd_ghg",
   "OECD air emissions inventory by sector", kind="panel", freq="A")
_s("worldbank.energy_transition", "worldbank", "energy_transition:fetch_energy_transition",
   "Energy balances and electricity generation by source", kind="panel", freq="A")
_s("worldbank.pink_sheet", "worldbank", "wb_pink_sheet:fetch_prices",
   "World Bank Pink Sheet commodity prices", freq="M")
_s("commodities.benchmarks", "worldbank", "commodities:fetch_commodity_benchmarks",
   "Standardised commodity benchmark prices and indices", freq="M")
_s("bis.macroprudential", "bis", "financial:fetch_bis_macroprudential",
   "BIS credit-to-GDP gap, total credit, property prices", kind="panel", freq="Q")
_s("financial.sovereign_yields", "fred", "financial:fetch_sovereign_yields",
   "Sovereign bond yields (10Y, 2Y)", kind="panel", freq="M")
_s("financial.policy_rates", "bis", "financial:fetch_policy_rates",
   "Central bank policy rates", kind="panel", freq="M")
_s("financial.conditions", "fred", "financial:fetch_financial_conditions",
   "Financial conditions and credit spreads", freq="M")

# --- Uncertainty and research series --------------------------------------
_s("epu.country", "policyuncertainty", "epu:fetch",
   "Baker-Bloom-Davis country EPU indices", kind="panel", freq="M")
_s("epu.states", "policyuncertainty", "epu_states:fetch",
   "Baker-Bloom-Davis-Levy state EPU", kind="panel", freq="M")
_s("epu.news_historical", "policyuncertainty", "epu_news_historical:fetch",
   "Historical news-based US EPU", freq="M")
_s("gpr.caldara_iacoviello", "iacoviello", "gpr:fetch",
   "Geopolitical Risk index", freq="M")
_s("mpu.husted_rogers_sun", "iacoviello", "hrs_mpu:fetch",
   "Husted-Rogers-Sun monetary policy uncertainty", freq="M")
_s("wui.quarterly", "wui", "wui:fetch_q",
   "World Uncertainty Index", kind="panel", freq="Q")
_s("wui.monthly", "wui", "wui:fetch_m",
   "World Uncertainty Index, monthly", kind="panel", freq="M")
_s("jln.macro_uncertainty", "ludvigson", "jln:fetch",
   "Jurado-Ludvigson-Ng macro uncertainty", freq="M")
_s("lmn.uncertainty", "ludvigson", "lmn:fetch",
   "Ludvigson-Ma-Ng real vs financial uncertainty", freq="M")
_s("fernald.tfp", "frbsf", "fernald:fetch",
   "Fernald utilisation-adjusted TFP", freq="Q")
_s("kenfrench.industry30", "kenfrench", "kenfrench_industry:fetch",
   "Ken French 30-industry value-weighted returns", freq="M")
_s("yahoo.realized_vol", "yahoo", "yahoo:fetch_realized_vol",
   "Realised volatility from daily stock-index returns", kind="panel",
   redistributable=False, note="Yahoo Finance terms restrict redistribution.")

# --- Real-time vintage providers (vintage_panel back ends) -----------------
for _name, _cred, _auth in [
    ("alfred", "fred", "key"), ("banxico", "banxico", "key"),
    ("bcb", None, "none"), ("bcch", "bcch", "key"),
    ("bundesbank", None, "none"), ("ecb_rtd", None, "none"),
    ("inegi", "inegi", "key"), ("oecd_stes", None, "none"),
    ("ons", None, "none"), ("statcan", None, "none"),
]:
    register(SourceInfo(
        id=f"vintage.{_name}", kind="vintage", provider=_name,
        loader="puremacro.fetch.realtime:vintage_panel",
        description=f"Real-time editions from {_name}; call "
                    f"vintage_panel(..., providers=['{_name}'])",
        credential=_cred, auth=_auth))

# --- Microdata -------------------------------------------------------------
_M = "puremacro.fetch.micro."
register(SourceInfo(
    id="census.acs_pums", kind="micro", provider="census",
    loader=_M + "census:fetch_acs_pums",
    description="ACS 1-/5-year PUMS person or household records with 80 "
                "successive-difference replicate weights",
    credential="census", auth="key",
    terms_url="https://www.census.gov/data/developers/about/terms-of-service.html",
    freq="A"))
register(SourceInfo(
    id="census.cps_basic", kind="micro", provider="census",
    loader=_M + "census:fetch_cps_basic",
    description="CPS basic monthly person records (weights only; no "
                "public replicate weights or design variables)",
    credential="census", auth="key",
    terms_url="https://www.census.gov/data/developers/about/terms-of-service.html",
    freq="M"))
register(SourceInfo(
    id="fed.scf", kind="micro", provider="fed",
    loader=_M + "scf:fetch_scf",
    description="Survey of Consumer Finances summary extract: five "
                "implicates plus 999 replicate weights",
    terms_url="https://www.federalreserve.gov/econres/scfindex.htm",
    freq="3A"))


# ---------------------------------------------------------------------------
# Text connectors, read from narrative.sources on first use
# ---------------------------------------------------------------------------
_TEXT_LOADED = False


def _ensure_text() -> None:
    """Register every ``iter_*`` narrative connector as a ``text`` source.

    Read lazily from the connector map in :mod:`puremacro.narrative.sources`
    (which itself imports nothing heavy), so a connector added there shows
    up here without a second edit.
    """
    global _TEXT_LOADED
    if _TEXT_LOADED:
        return
    _TEXT_LOADED = True
    try:
        from ..narrative import sources as _ns
        module_of = dict(getattr(_ns, "_MODULE_OF", {}))
    except ImportError:          # narrative layer unavailable in this build
        return
    for name, module in sorted(module_of.items()):
        if not name.startswith("iter_"):
            continue
        sid = "text." + name[len("iter_"):]
        if sid in _REGISTRY:
            continue
        register(SourceInfo(
            id=sid, kind="text", provider=module,
            loader=f"puremacro.narrative.sources.{module}:{name}",
            description=f"narrative.sources.{name}: yields "
                        "(date, text, url, metadata) records",
            auth="unaudited"))


__all__ = ["SourceInfo", "KINDS", "AUTH_LEVELS", "register", "info", "load",
           "sources"]
