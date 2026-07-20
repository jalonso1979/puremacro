"""Catalog: declaratively registers Phase-1 instrument entries.

Phase 1: 6 narrative replications + 6 narrative connectors + 1 monetary HFI
+ 12 connector stubs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from ._core import Instrument
from ._registry import InstrumentSpec, register

if TYPE_CHECKING:
    from ..narrative import NarrativeInstrument


# --------------------------------------------------------------------------
# Helper: wrap a NarrativeInstrument-returning loader as an Instrument loader
# --------------------------------------------------------------------------
def _wrap_narrative(
    loader_fn: Callable[..., "NarrativeInstrument | dict[str, NarrativeInstrument]"],
    registry_key: str,
    source: str,
) -> Callable[..., Instrument]:
    """Adapter: call loader_fn(**kwargs) → NarrativeInstrument or
    dict[str, NarrativeInstrument]. If a dict, the load-time kwarg
    `select_country=ISO3` selects which country's instrument to return.
    """
    def _load(**kwargs) -> Instrument:
        select_country = kwargs.pop("select_country", None)
        result = loader_fn(**kwargs)
        if isinstance(result, dict):
            if select_country is None:
                raise ValueError(
                    f"{registry_key!r} loader returns a per-country dict; "
                    f"pass select_country=<ISO3> at load time, e.g. "
                    f"load({registry_key!r}, select_country='USA'). "
                    f"Available: {sorted(result.keys())}"
                )
            if select_country not in result:
                raise KeyError(
                    f"select_country {select_country!r} not in "
                    f"{registry_key!r}; available: {sorted(result.keys())}"
                )
            narr = result[select_country]
        else:
            narr = result
        narr.metadata["registry_key"] = registry_key
        narr.metadata["source"] = source
        return narr.as_instrument()
    return _load


# --------------------------------------------------------------------------
# Narrative replications (6)
# --------------------------------------------------------------------------
from ..narrative.replication import (
    load_ramey_2011_defense,
    load_romer_romer_2010,
    load_mertens_ravn_2013,
    load_cloyne_2013_uk,
    load_romer_romer_2017,
    load_dglp_2011,
)


register(InstrumentSpec(
    key="ramey_2011_defense",
    name="Ramey 2011 defense buildup news",
    category="narrative_replication",
    description=(
        "Defense-spending news shocks identified from major military "
        "buildup announcements (Korea, Vietnam, Carter-Reagan, post-9/11)."
    ),
    reference="Ramey, V.A. (2011). Identifying government spending shocks: it's all in the timing. QJE 126(1), 1-50.",
    loader=_wrap_narrative(load_ramey_2011_defense, "ramey_2011_defense",
                           "Ramey 2011 defense buildup events"),
    country="USA",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="romer_romer_2010_fiscal",
    name="Romer-Romer 2010 fiscal shocks",
    category="narrative_replication",
    description=(
        "Exogenous tax-policy changes classified by motivation from "
        "presidential speeches and Congressional records."
    ),
    reference="Romer, C.D. and Romer, D.H. (2010). The macroeconomic effects of tax changes. AER 100(3), 763-801.",
    loader=_wrap_narrative(load_romer_romer_2010, "romer_romer_2010_fiscal",
                           "Romer-Romer 2010 narrative tax shocks"),
    country="USA",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="mertens_ravn_2013_tax",
    name="Mertens-Ravn 2013 tax-rate shocks",
    category="narrative_replication",
    description=(
        "Personal and corporate tax rate change announcements built on "
        "Romer-Romer's narrative coding, used as external instruments."
    ),
    reference="Mertens, K. and Ravn, M.O. (2013). The dynamic effects of personal and corporate income tax changes in the United States. AER 103(4), 1212-1247.",
    loader=_wrap_narrative(load_mertens_ravn_2013, "mertens_ravn_2013_tax",
                           "Mertens-Ravn 2013 narrative tax rate shocks"),
    country="USA",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="cloyne_2013_uk_tax",
    name="Cloyne 2013 UK narrative tax shocks",
    category="narrative_replication",
    description=(
        "UK exogenous tax changes classified by motivation, the British "
        "counterpart to Romer-Romer 2010."
    ),
    reference="Cloyne, J. (2013). Discretionary tax changes and the macroeconomy: new narrative evidence from the United Kingdom. AER 103(4), 1507-1528.",
    loader=_wrap_narrative(load_cloyne_2013_uk, "cloyne_2013_uk_tax",
                           "Cloyne 2013 UK narrative tax shocks"),
    country="GBR",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="romer_romer_2017_fiscal",
    name="Romer-Romer 2017 financial-distress shocks",
    category="narrative_replication",
    description=(
        "Cross-country narrative tax-shock panel for 14 advanced economies "
        "(US, UK, AUS, CAN, DNK, FRA, DEU, IRL, ITA, JPN, NLD, NZL, ESP, "
        "SWE), hand-coded with the Romer-Romer 2010 exogenous/endogenous "
        "classification. Returns a per-country dict; pass "
        "``select_country=<ISO3>`` at load time."
    ),
    reference="Romer, C.D. and Romer, D.H. (2017). Hard times: why the recovery from the Great Recession was so slow. NBER WP / cross-country tax-narrative dataset (authors' replication archive).",
    loader=_wrap_narrative(load_romer_romer_2017, "romer_romer_2017_fiscal",
                           "Romer-Romer 2017 narrative shocks"),
    country=None,
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="dglp_2011_consolidations",
    name="Devries-Guajardo-Leigh-Pescatori 2011 consolidations",
    category="narrative_replication",
    description=(
        "OECD-wide narrative coding of fiscal consolidation episodes (17 "
        "advanced economies, 1978-2009)."
    ),
    reference="Devries, P., Guajardo, J., Leigh, D., Pescatori, A. (2011). A new action-based dataset of fiscal consolidations. IMF Working Paper WP/11/158.",
    loader=_wrap_narrative(load_dglp_2011, "dglp_2011_consolidations",
                           "DGLP 2011 fiscal consolidations"),
    country=None,
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


# --------------------------------------------------------------------------
# Helper: wrap a connector iterator as an Instrument loader
# --------------------------------------------------------------------------
def _wrap_connector(iter_fn, *, registry_key: str, source: str,
                    country: str | None, target: str = "both"):
    """Adapter: pull (date, text, link) records from connector, score with
    keyword backend, aggregate to quarterly, return Instrument.

    The keyword scorer requires a country code for each event. For
    single-country connectors the registered ``country`` is used. For
    cross-country connectors (``country=None`` at registration), the
    user MUST pass ``country=<ISO3>`` at load time to specify the source country
    of the events being scored. There is no silent fallback because
    NarrativeEvent.country is a structured field and writing the wrong
    country into it would corrupt downstream country-filtered analyses.
    """
    def _load(**kwargs) -> Instrument:
        from ..narrative.scoring import score_keyword
        from ..narrative import NarrativeInstrument
        load_country = kwargs.pop("country", None) or country
        if load_country is None:
            raise ValueError(
                f"{registry_key!r} is a cross-country source; pass "
                f"country=<ISO3> at load time, e.g. "
                f"load({registry_key!r}, country='USA')."
            )
        records = list(iter_fn(**kwargs))
        events = list(score_keyword(iter(records), country=load_country))
        narr = NarrativeInstrument.from_events(events, target=target)
        narr.metadata["registry_key"] = registry_key
        narr.metadata["source"] = source
        return narr.as_instrument()
    return _load


# --------------------------------------------------------------------------
# Narrative connectors (6 active)
# --------------------------------------------------------------------------
from ..narrative.sources import (
    us_treasury, us_federal_register, us_dod_contracts,
    oecd_surveys, imf_articleiv, news_api,
)


register(InstrumentSpec(
    key="us_treasury_press",
    name="US Treasury press releases (HTML scrape)",
    category="narrative_connector",
    description="Recent US Treasury press releases scraped from home.treasury.gov/news/press-releases.",
    reference="US Department of the Treasury, public press release listing.",
    loader=_wrap_connector(us_treasury.iter_treasury_press,
                           registry_key="us_treasury_press",
                           source="US Treasury press releases (HTML)",
                           country="USA"),
    country="USA",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="us_federal_register",
    name="US Federal Register documents",
    category="narrative_connector",
    description=(
        "Treasury and DoD presidential documents, rules, and notices from "
        "the Federal Register API."
    ),
    reference="US Government Publishing Office, Federal Register API v1.",
    loader=_wrap_connector(us_federal_register.iter_federal_register,
                           registry_key="us_federal_register",
                           source="US Federal Register API",
                           country="USA"),
    country="USA",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="us_dod_contracts",
    name="US DoD daily contract awards",
    category="narrative_connector",
    description=(
        "Daily defense contract awards (procurement + operating "
        "expenditure) from defense.gov."
    ),
    reference="US Department of Defense, public contracts listing.",
    loader=_wrap_connector(us_dod_contracts.iter_dod_contracts,
                           registry_key="us_dod_contracts",
                           source="US DoD contracts (HTML)",
                           country="USA",
                           target="investment"),
    country="USA",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="oecd_surveys",
    name="OECD Economic Surveys",
    category="narrative_connector",
    description="Country-by-country fiscal-policy assessments from OECD Surveys.",
    reference="OECD Economic Surveys, public PDF/HTML archive.",
    loader=_wrap_connector(oecd_surveys.iter_oecd_surveys,
                           registry_key="oecd_surveys",
                           source="OECD Economic Surveys",
                           country=None),
    country=None,
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="imf_articleiv",
    name="IMF Article IV consultations",
    category="narrative_connector",
    description="Member-country fiscal-policy assessments from IMF Article IV consultations.",
    reference="International Monetary Fund, Article IV staff reports.",
    loader=_wrap_connector(imf_articleiv.iter_imf_articleiv,
                           registry_key="imf_articleiv",
                           source="IMF Article IV consultations",
                           country=None),
    country=None,
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="gdelt_v2_news",
    name="GDELT v2 fiscal-policy news",
    category="narrative_connector",
    description=(
        "Global news event records from GDELT v2 — free public access, "
        "filtered by fiscal-policy keywords. Module name `news_api.py` is "
        "historical; the actual backend is GDELT."
    ),
    reference="GDELT Project, public v2 events API.",
    loader=_wrap_connector(news_api.iter_gdelt_v2,
                           registry_key="gdelt_v2_news",
                           source="GDELT v2 events",
                           country=None),
    country=None,
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


# --------------------------------------------------------------------------
# Monetary HFI (1 active)
# --------------------------------------------------------------------------
def _load_gk2015_ffr_surprise(*, ff_futures_pre, ff_futures_post,
                               days_remaining_in_month, dates,
                               days_in_month=30, freq: str = "M") -> Instrument:
    """Loader for the Gertler-Karadi 2015 month-end-adjusted FFR surprise.

    Parameters
    ----------
    ff_futures_pre, ff_futures_post : array-like
        Federal-funds-futures contracts before and after each FOMC announcement.
    days_remaining_in_month : array-like
        Days remaining in the announcement month at each FOMC date.
    dates : array-like of pandas Timestamps
        FOMC announcement dates aligned with the futures arrays.
    days_in_month : int or array-like, default 30
        Days in the announcement month (scalar or per-announcement).
    freq : str, default "M"
        Aggregation frequency.
    """
    import numpy as np
    from ..hfi import gk2015_surprise, aggregate_to_period
    surprises = gk2015_surprise(
        np.asarray(ff_futures_pre, dtype=float),
        np.asarray(ff_futures_post, dtype=float),
        np.asarray(days_remaining_in_month, dtype=float),
        days_in_month=days_in_month,
    )
    series = aggregate_to_period(surprises, dates, freq=freq)
    return Instrument(
        series=series,
        name="gk2015_ffr_surprise",
        source="Gertler-Karadi 2015 FFR-futures month-end-adjusted surprise",
        category="monetary_hfi",
        frequency=freq,
        metadata={"reference": "Gertler-Karadi 2015"},
    )


register(InstrumentSpec(
    key="gk2015_ffr_surprise",
    name="Gertler-Karadi 2015 FFR surprise",
    category="monetary_hfi",
    description=(
        "Month-end-adjusted FFR-futures monetary policy surprise around "
        "FOMC announcements."
    ),
    reference="Gertler, M. and Karadi, P. (2015). Monetary policy surprises, credit costs, and economic activity. AEJ Macro 7(1), 44-76.",
    loader=_load_gk2015_ffr_surprise,
    country="USA",
    frequency="M",
    requires_network=False,
    requires_fixture=True,
))


# --------------------------------------------------------------------------
# Connector stubs (12) — listed for discoverability; not loadable in Phase 1
# --------------------------------------------------------------------------
def _stub_loader(*args, **kwargs) -> Instrument:
    """Stub loader for catalogued-but-not-yet-wrapped connectors."""
    detail = f" (received kwargs: {sorted(kwargs.keys())})" if kwargs else ""
    raise NotImplementedError(
        f"This connector is catalogued for discoverability only.{detail} "
        "Phase 1 ships loaders for the 6 connectors with offline fixtures; "
        "this one will land in a future patch."
    )


_STUB_CONNECTORS = [
    ("uk_obr",      "UK Office for Budget Responsibility forecasts", "GBR"),
    ("uk_hmt",      "UK HM Treasury press notices",                  "GBR"),
    ("de_bmf",      "German Bundesfinanzministerium press",          "DEU"),
    ("fr_tresor",   "French Trésor (DG) press releases",             "FRA"),
    ("it_mef",      "Italian MEF press releases",                    "ITA"),
    ("jp_mof",      "Japanese Ministry of Finance press",            "JPN"),
    ("ca_dof",      "Canadian Department of Finance press",          "CAN"),
    ("ecb_press",   "ECB press releases",                            None),
    ("eu_ecfin",    "European Commission DG ECFIN press",            None),
    ("imf_news",    "IMF news",                                      None),
    ("google_news", "Google News fiscal queries",                    None),
    ("local_csv",   "User-supplied local CSV of events",             None),
]
for stub_key, stub_name, stub_country in _STUB_CONNECTORS:
    register(InstrumentSpec(
        key=stub_key,
        name=stub_name,
        category="narrative_connector",
        description=f"Stub: {stub_name}. Loader to be implemented in a future patch.",
        reference=f"Connector module puremacro.narrative.sources.{stub_key}",
        loader=_stub_loader,
        country=stub_country,
        frequency="Q",
        requires_network=True,
        requires_fixture=False,
    ))


# --------------------------------------------------------------------------
# Literature shock loaders (4) — see puremacro/instruments/literature/
# --------------------------------------------------------------------------
from .literature import (
    load_bloom_2009,
    load_bbd_epu,
    load_caldara_iacoviello_gpr,
    load_romer_romer_2004,
)


register(InstrumentSpec(
    key="bloom_2009_uncertainty",
    name="Bloom 2009 uncertainty events",
    category="literature",
    description=(
        "17 large-uncertainty announcement-month indicators from Bloom "
        "(2009) Table A.1, monthly Jan-1962 to Dec-2008. Baked-in event "
        "list — no network or fixture needed; fully reproducible."
    ),
    reference="Bloom, N. (2009). The impact of uncertainty shocks. Econometrica 77(3), 623-685.",
    loader=load_bloom_2009,
    country=None,
    frequency="M",
    requires_network=False,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="bbd_epu_us",
    name="Baker-Bloom-Davis EPU (US, monthly)",
    category="literature",
    description=(
        "News-based monthly Economic Policy Uncertainty index for the "
        "US, fetched from policyuncertainty.com. Pass csv_path= to "
        "skip the network call."
    ),
    reference="Baker, S.R., Bloom, N., Davis, S.J. (2016). Measuring economic policy uncertainty. QJE 131(4), 1593-1636.",
    loader=load_bbd_epu,
    country="USA",
    frequency="M",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="caldara_iacoviello_gpr",
    name="Caldara-Iacoviello Geopolitical Risk index (monthly)",
    category="literature",
    description=(
        "News-based monthly Geopolitical Risk index, fetched from "
        "matteoiacoviello.com/gpr.htm. Pass csv_path= to skip the "
        "network call."
    ),
    reference="Caldara, D. and Iacoviello, M. (2022). Measuring geopolitical risk. American Economic Review 112(4), 1194-1225.",
    loader=load_caldara_iacoviello_gpr,
    country=None,
    frequency="M",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="rr_2004_monetary",
    name="Romer-Romer 2004 narrative monetary shocks",
    category="literature",
    description=(
        "Quarterly residual from regressing FOMC intended FFR changes on "
        "Greenbook forecasts (1969Q1-1996Q4 base, extensions to ~2007). "
        "Mirror download attempted from David Romer's UC Berkeley site; "
        "if it fails, pass csv_path= with a local copy."
    ),
    reference="Romer, C.D. and Romer, D.H. (2004). A new measure of monetary shocks: derivation and implications. AER 94(4), 1055-1084.",
    loader=load_romer_romer_2004,
    country="USA",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


# --------------------------------------------------------------------------
# External-CSV providers (7) — see puremacro/instruments/external/
# --------------------------------------------------------------------------
from .external import load_fred, load_bis, load_imf_weo


# --- FRED (4 entries) ---
def _make_fred_loader(series_id: str, frequency: str):
    def _load(**kwargs):
        return load_fred(series_id=series_id, frequency=frequency, **kwargs)
    return _load


register(InstrumentSpec(
    key="fred_nfci",
    name="Chicago Fed NFCI (weekly)",
    category="external_csv",
    description=(
        "Chicago Fed National Financial Conditions Index. Weekly, "
        "1971-present. Standard composite measure of risk, liquidity, "
        "and leverage in US financial markets. Requires FRED API key "
        "(free; set FRED_API_KEY env var)."
    ),
    reference="Federal Reserve Bank of Chicago. NFCI. https://fred.stlouisfed.org/series/NFCI",
    loader=_make_fred_loader("NFCI", "W"),
    country="USA",
    frequency="W",
    requires_network=True,
    requires_fixture=True,
))


register(InstrumentSpec(
    key="fred_vixcls",
    name="CBOE Volatility Index VIX (daily)",
    category="external_csv",
    description=(
        "CBOE VIX, daily close. The canonical proxy for US stock-market "
        "uncertainty. Requires FRED API key (free; set FRED_API_KEY env "
        "var)."
    ),
    reference="Chicago Board Options Exchange. VIX. https://fred.stlouisfed.org/series/VIXCLS",
    loader=_make_fred_loader("VIXCLS", "D"),
    country="USA",
    frequency="D",
    requires_network=True,
    requires_fixture=True,
))


register(InstrumentSpec(
    key="fred_fedfunds",
    name="Effective Federal Funds Rate (monthly)",
    category="external_csv",
    description=(
        "Monthly effective federal funds rate. The standard policy-rate "
        "series in US monetary VARs. Requires FRED API key (free; set "
        "FRED_API_KEY env var)."
    ),
    reference="Board of Governors of the Federal Reserve. FEDFUNDS. https://fred.stlouisfed.org/series/FEDFUNDS",
    loader=_make_fred_loader("FEDFUNDS", "M"),
    country="USA",
    frequency="M",
    requires_network=True,
    requires_fixture=True,
))


register(InstrumentSpec(
    key="fred_stlfsi4",
    name="St. Louis Fed Financial Stress Index v4 (weekly)",
    category="external_csv",
    description=(
        "St. Louis Fed Financial Stress Index, version 4. Weekly. A "
        "complementary financial-conditions measure to the Chicago Fed "
        "NFCI, with different weighting. Requires FRED API key (free; "
        "set FRED_API_KEY env var)."
    ),
    reference="Federal Reserve Bank of St. Louis. STLFSI4. https://fred.stlouisfed.org/series/STLFSI4",
    loader=_make_fred_loader("STLFSI4", "W"),
    country="USA",
    frequency="W",
    requires_network=True,
    requires_fixture=True,
))


# --- BIS (1 entry) ---
def _load_bis_credit_gap_us(**kwargs):
    return load_bis(series_id="credit_to_gdp_gap", country="US", **kwargs)


register(InstrumentSpec(
    key="bis_credit_to_gdp_gap_us",
    name="BIS US credit-to-GDP gap (quarterly)",
    category="external_csv",
    description=(
        "US credit-to-GDP gap from the BIS total-credit statistics. "
        "Quarterly. The BCBS countercyclical-buffer reference variable. "
        "Pass csv_path= to skip the network call."
    ),
    reference="Bank for International Settlements. Credit-to-GDP gaps. https://www.bis.org/statistics/totcredit.htm",
    loader=_load_bis_credit_gap_us,
    country="USA",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


# --- IMF WEO (2 entries) ---
def _make_weo_loader(indicator: str, country: str):
    def _load(**kwargs):
        return load_imf_weo(indicator=indicator, country=country, **kwargs)
    return _load


register(InstrumentSpec(
    key="imf_weo_debt_gdp_usa",
    name="IMF WEO US general government gross debt (% of GDP, annual)",
    category="external_csv",
    description=(
        "US general government gross debt as % of GDP, from the IMF "
        "World Economic Outlook database. Annual. Pass csv_path= to "
        "skip the network call."
    ),
    reference="International Monetary Fund. World Economic Outlook Database — GGXWDG_NGDP indicator. https://www.imf.org/en/Publications/WEO",
    loader=_make_weo_loader("GGXWDG_NGDP", "USA"),
    country="USA",
    frequency="A",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="imf_weo_primary_balance_gdp_usa",
    name="IMF WEO US general government primary balance (% of GDP, annual)",
    category="external_csv",
    description=(
        "US general government primary balance as % of GDP, from the "
        "IMF World Economic Outlook database. Annual. Pass csv_path= "
        "to skip the network call."
    ),
    reference="International Monetary Fund. World Economic Outlook Database — GGXONLB_NGDP indicator. https://www.imf.org/en/Publications/WEO",
    loader=_make_weo_loader("GGXONLB_NGDP", "USA"),
    country="USA",
    frequency="A",
    requires_network=True,
    requires_fixture=False,
))


# --------------------------------------------------------------------------
# Fetch-layer providers (4) — see puremacro/fetch/
# --------------------------------------------------------------------------
import pandas as pd


def _fred_csv_loader(*, series_id: str = "GDPC1", **kwargs) -> Instrument:
    """Wrap fetch_fred (public CSV, no API key) as an Instrument."""
    from ..fetch._classic import fetch_fred  # lazy: avoids circular import
    df = fetch_fred(series_id, **kwargs)
    if series_id in df.columns:
        series = df[series_id].astype(float)
    else:
        cols = [c for c in df.columns if c.upper() != "DATE"]
        series = df[cols[0]].astype(float)
    series.name = f"fetch_fred_{series_id}"
    return Instrument(
        series=series,
        name=f"fetch_fred_{series_id}",
        source=f"FRED public CSV (fetch_fred): {series_id}",
        category="external_csv",
        frequency="Q",
        metadata={
            "reference": (
                "Federal Reserve Bank of St. Louis. FRED public CSV. "
                f"https://fred.stlouisfed.org/series/{series_id}"
            ),
            "series_id": series_id,
        },
    )


register(InstrumentSpec(
    key="fetch_fred_csv",
    name="FRED public CSV (no API key required)",
    category="external_csv",
    description=(
        "Fetch any FRED series via the public CSV endpoint (no API key "
        "required). Pass series_id= at load time. Default series_id is "
        "'GDPC1' (real GDP). Frequency varies by series; default 'Q'."
    ),
    reference="Federal Reserve Bank of St. Louis. FRED public CSV endpoint. https://fred.stlouisfed.org/",
    loader=_fred_csv_loader,
    country="USA",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


def _bis_neer_us_loader(**kwargs) -> Instrument:
    """Wrap fetch_bis_neer for US as an Instrument."""
    from ..bis_neer import fetch_bis_neer  # lazy: avoids circular import
    df = fetch_bis_neer("US", **kwargs)
    if "value" in df.columns:
        series = df["value"].astype(float)
    else:
        numeric_cols = [c for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            series = df[numeric_cols[0]].astype(float)
        else:
            series = df.iloc[:, 0].astype(float)
    series.name = "bis_neer_us"
    return Instrument(
        series=series,
        name="bis_neer_us",
        source="BIS US Nominal Effective Exchange Rate (fetch_bis_neer)",
        category="external_csv",
        frequency="M",
        metadata={
            "reference": "Bank for International Settlements. Effective Exchange Rate indices. https://www.bis.org/statistics/eer.htm",
            "country": "US",
        },
    )


register(InstrumentSpec(
    key="fetch_bis_neer_us",
    name="BIS US Nominal Effective Exchange Rate (monthly)",
    category="external_csv",
    description=(
        "Monthly US nominal effective exchange rate from BIS, fetched "
        "via puremacro.bis_neer.fetch_bis_neer. Pass csv_path= to skip "
        "the network call."
    ),
    reference="Bank for International Settlements. Effective Exchange Rate indices. https://www.bis.org/statistics/eer.htm",
    loader=_bis_neer_us_loader,
    country="USA",
    frequency="M",
    requires_network=True,
    requires_fixture=False,
))


def _make_oecd_stan_loader(country: str, indicator: str):
    def _load(**kwargs) -> Instrument:
        from ..fetch.sdmx import oecd_sdmx_instrument  # lazy: avoids circular import
        return oecd_sdmx_instrument(
            dataset="DSD_STAN", country=country, indicator=indicator,
            **kwargs,
        )
    return _load


register(InstrumentSpec(
    key="oecd_sdmx_stan_usa_valadd",
    name="OECD STAN US Value Added (annual)",
    category="external_csv",
    description=(
        "Annual US value added from OECD-STAN industrial database, "
        "fetched via puremacro.fetch.oecd_sdmx_instrument with "
        "dataset='DSD_STAN', country='USA', indicator='VALADD'. "
        "Pass csv_path= to skip the network call."
    ),
    reference="OECD STAN Industrial Statistics Database. https://www.oecd.org/sti/stan",
    loader=_make_oecd_stan_loader("USA", "VALADD"),
    country="USA",
    frequency="A",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="oecd_sdmx_stan_usa_empn",
    name="OECD STAN US Employment (annual)",
    category="external_csv",
    description=(
        "Annual US total employment from OECD-STAN, fetched via "
        "puremacro.fetch.oecd_sdmx_instrument with dataset='DSD_STAN', "
        "country='USA', indicator='EMPN'. Pass csv_path= to skip the "
        "network call."
    ),
    reference="OECD STAN Industrial Statistics Database. https://www.oecd.org/sti/stan",
    loader=_make_oecd_stan_loader("USA", "EMPN"),
    country="USA",
    frequency="A",
    requires_network=True,
    requires_fixture=False,
))
