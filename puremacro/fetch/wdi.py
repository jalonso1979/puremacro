r"""One-call World Bank WDI panel: 217 economies, annual, from 1960.

``wdi_panel()`` returns the widest and longest annual macro panel a free
provider publishes: the World Development Indicators for every reporting
economy the World Bank lists (217, territories included), indexed by
``(code, date)`` with one column per variable::

    gdp  cons_hh  cons_gov  inv  capform  inventories  stat_disc  exports  imports
    taxes_prod  va_total  va_agri  va_ind  va_mfg  va_services  gni
    gdp_real  gdp_defl  cons_hh_real  cons_hh_defl ...        (real=True)
    pop  pop_0014  pop_1564  pop_65up  lf  urate  epop  lfpr  selfemp_share ...
    gdp_ppp  gdp_pc_ppp  ppp  fx  cpi  credit_priv

Why this module exists
----------------------
The OECD panels (:func:`puremacro.fetch.qna_panel`, ``ana_by_activity``)
stop at some forty economies. A course that wants Argentina next to Austria,
or the Sahel next to the Nordics, needs the World Bank, and it needs it with
the same conventions as the rest of the package: ISO3 codes, period-start
dates, millions of national currency, a registry of English snake_case names
and ``attrs`` that say where every column came from. The two WDI callers the
package had before 4.7 read only the first page of each response (37
economies lost, the United States among them) and let 42 World Bank
aggregates through as countries; both now page through every response and
drop the aggregates through the same explicit list this module uses.

Mechanics
---------
One request per indicator, ``/v2/country/all/indicator/{code}`` with
``per_page=32767`` (the API's ceiling) and a page loop that checks the row
count against ``total``; an all-economy, all-year response is one page of
17 490 rows, 139 KB with the gzip the transport asks for. Bodies travel
through :func:`puremacro._http.safe_get_bytes_cached` (urllib, SQLite cache
outside the repository, one-second spacing per host), so a notebook that
freezes the panel re-reads today's bytes instead of hitting the API again.
A failed indicator never raises: it is listed in ``attrs["missing"]`` with
the reason (``HTTP 400``, a timeout after one retry, or the API's in-band
``message`` body, which arrives with HTTP 200). ``ValueError`` is reserved
for caller errors: an unknown variable name, a code that is not a WDI
economy, ``start`` after ``end``.

The economy universe is the ``/v2/country`` list with ``region.id != "NA"``
(217 ids); the 79 aggregates (regions, income groups, lending groups,
``WLD``) are dropped by that rule and by the explicit
:data:`puremacro._codes.WB_AGGREGATES` list, never by code length. Five
income-group aggregates arrive in data responses with a blank
``countryiso3code`` and are dropped with the rest.

Units are rescaled to the house conventions and recorded per variable in
``attrs["meta"]`` (``unit_mult``): current and constant LCU levels to
*millions* (WDI publishes full units), persons to *thousands*; rates,
indices, per-capita values and conversion factors are left as published.
``real=True`` fetches the constant-LCU twin of every national-accounts item
that has one (``<name>_real``, at each country's own base or reference year,
see :func:`wdi_meta`) and derives ``<name>_defl = 100 * nominal / real``;
on GDP the derived deflator reproduces WDI's ``NY.GDP.DEFL.ZS`` to 1e-15.
A deflator is kept only where both levels are positive (Venezuela publishes
nominal zeros for 1991-2011; capital formation turns negative in a few
small economies); ``attrs["meta"]`` counts the masked cells in
``n_masked``. Sign-changing items (``inventories``, ``stat_disc``,
``taxes_prod``) get no deflator.
Constant-price components are chain-linked for most economies and do not
add up: never sum ``_real`` columns.

WDI has no durable-goods split, no public GFCF (only private,
``inv_priv``), no capital stocks, no hours, no employment level (derive
``lf * (1 - urate / 100)``), no compensation of employees economy-wide and
nothing sub-annual; those live in the OECD, Eurostat and PWT fetchers.

Source: World Bank World Development Indicators, API v2 source 2,
https://api.worldbank.org/v2 (CC BY-4.0).
"""
from __future__ import annotations

import datetime as dt
import json
import time
import warnings
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from puremacro._codes import WB_AGGREGATES

_BASE = "https://api.worldbank.org/v2"
_SOURCE = "World Bank WDI (source 2), api.worldbank.org/v2"
_PER_PAGE = 32767          # the API answers 32 768 with an HTML 400 page
_TIMEOUT = 120.0           # 6 of 118 all-economy pulls stalled 40-83 s in the probe
_PAUSE = 1.0
_RETRY_SLEEP = 10.0
_HEADERS = {"Accept-Encoding": "gzip"}
_FIRST_YEAR = 1960         # nothing exists before 1960; earlier starts are clamped

#: column name -> (WDI indicator code, description, units as returned here).
#: Levels published in full LCU / US$ units are rescaled to millions, persons
#: to thousands (see :data:`_SCALE`); everything else is as published. The
#: national-accounts block is current prices (``.CN``); ``real=True`` adds
#: the constant-price twins listed in :data:`WDI_REAL`.
WDI_INDICATORS: dict[str, tuple[str, str, str]] = {
    # --- national accounts, expenditure side, current LCU -------------------
    "gdp":         ("NY.GDP.MKTP.CN", "Gross domestic product at market prices", "millions of current LCU"),
    "cons_hh":     ("NE.CON.PRVT.CN", "Household final consumption (households + NPISH)", "millions of current LCU"),
    "cons_gov":    ("NE.CON.GOVT.CN", "General government final consumption", "millions of current LCU"),
    "inv":         ("NE.GDI.FTOT.CN", "Gross fixed capital formation", "millions of current LCU"),
    "capform":     ("NE.GDI.TOTL.CN", "Gross capital formation (GFCF + change in inventories)", "millions of current LCU"),
    "inventories": ("NE.GDI.STKB.CN", "Changes in inventories", "millions of current LCU"),
    "stat_disc":   ("NY.GDP.DISC.CN", "Statistical discrepancy of the expenditure identity", "millions of current LCU"),
    "exports":     ("NE.EXP.GNFS.CN", "Exports of goods and services", "millions of current LCU"),
    "imports":     ("NE.IMP.GNFS.CN", "Imports of goods and services", "millions of current LCU"),
    # --- production and income side, current LCU ----------------------------
    "taxes_prod":  ("NY.TAX.NIND.CN", "Taxes less subsidies on products", "millions of current LCU"),
    "va_total":    ("NY.GDP.FCST.CN", "Gross value added, all activities (GDP at factor cost / basic prices)", "millions of current LCU"),
    "va_agri":     ("NV.AGR.TOTL.CN", "Value added, agriculture, forestry and fishing (ISIC A)", "millions of current LCU"),
    "va_ind":      ("NV.IND.TOTL.CN", "Value added, industry including construction (ISIC B-F)", "millions of current LCU"),
    "va_mfg":      ("NV.IND.MANF.CN", "Value added, manufacturing (ISIC C; memo, inside va_ind)", "millions of current LCU"),
    "va_services": ("NV.SRV.TOTL.CN", "Value added, services (ISIC G-U)", "millions of current LCU"),
    "inv_priv":    ("NE.GDI.FPRV.CN", "Private-sector gross fixed capital formation", "millions of current LCU"),
    "gni":         ("NY.GNP.MKTP.CN", "Gross national income", "millions of current LCU"),
    "nfi_abroad":  ("NY.GSR.NFCY.CN", "Net primary income from abroad", "millions of current LCU"),
    "saving_dom":  ("NY.GDS.TOTL.CN", "Gross domestic saving (GDP less final consumption)", "millions of current LCU"),
    "cfc_usd":     ("NY.ADJ.DKAP.CD", "Consumption of fixed capital (1970-2021 only)", "millions of current US$"),
    # --- population -----------------------------------------------------------
    "pop":         ("SP.POP.TOTL", "Total population, mid-year", "thousands of persons"),
    "pop_0014":    ("SP.POP.0014.TO", "Population aged 0-14", "thousands of persons"),
    "pop_1564":    ("SP.POP.1564.TO", "Population aged 15-64", "thousands of persons"),
    "pop_65up":    ("SP.POP.65UP.TO", "Population aged 65 and over", "thousands of persons"),
    # --- labour, ILO modelled estimates (1990/91 on) and national estimates --
    "lf":          ("SL.TLF.TOTL.IN", "Labour force, ILO modelled estimate", "thousands of persons"),
    "urate":       ("SL.UEM.TOTL.ZS", "Unemployment rate, ILO modelled estimate", "percent of labour force"),
    "epop":        ("SL.EMP.TOTL.SP.ZS", "Employment-to-population ratio, ages 15+, ILO modelled", "percent of population 15+"),
    "lfpr":        ("SL.TLF.CACT.ZS", "Labour force participation rate, ages 15+, ILO modelled", "percent of population 15+"),
    "lfpr_1564":   ("SL.TLF.ACTI.ZS", "Labour force participation rate, ages 15-64, ILO modelled", "percent of population 15-64"),
    "urate_nat":   ("SL.UEM.TOTL.NE.ZS", "Unemployment rate, national estimate (survey years only)", "percent of labour force"),
    "epop_nat":    ("SL.EMP.TOTL.SP.NE.ZS", "Employment-to-population ratio, ages 15+, national estimate", "percent of population 15+"),
    "lfpr_nat":    ("SL.TLF.CACT.NE.ZS", "Labour force participation rate, ages 15+, national estimate", "percent of population 15+"),
    "selfemp_share":    ("SL.EMP.SELF.ZS", "Self-employed, ILO modelled", "percent of total employment"),
    "vulnerable_share": ("SL.EMP.VULN.ZS", "Vulnerable employment (own-account + contributing family), ILO modelled", "percent of total employment"),
    "employees_share":  ("SL.EMP.WORK.ZS", "Wage and salaried workers, ILO modelled", "percent of total employment"),
    "employers_share":  ("SL.EMP.MPYR.ZS", "Employers, ILO modelled", "percent of total employment"),
    "family_share":     ("SL.FAM.WORK.ZS", "Contributing family workers, ILO modelled", "percent of total employment"),
    "emp_share_agri":     ("SL.AGR.EMPL.ZS", "Employment in agriculture, ILO modelled", "percent of total employment"),
    "emp_share_ind":      ("SL.IND.EMPL.ZS", "Employment in industry, ILO modelled", "percent of total employment"),
    "emp_share_services": ("SL.SRV.EMPL.ZS", "Employment in services, ILO modelled", "percent of total employment"),
    # --- purchasing power, exchange rates, US dollar measures ----------------
    "gdp_ppp":     ("NY.GDP.MKTP.PP.KD", "GDP at purchasing power parity", "millions of constant 2021 international $"),
    "gdp_pc_ppp":  ("NY.GDP.PCAP.PP.KD", "GDP per capita at purchasing power parity", "constant 2021 international $ per person"),
    "ppp":         ("PA.NUS.PPP", "PPP conversion factor, GDP", "LCU per international $"),
    "ppp_cons":    ("PA.NUS.PRVT.PP", "PPP conversion factor, private consumption", "LCU per international $"),
    "fx":          ("PA.NUS.FCRF", "Official exchange rate, period average", "LCU per US$"),
    "fx_alt":      ("PA.NUS.ATLS", "DEC alternative conversion factor (gdp / fx_alt = GDP in current US$)", "LCU per US$"),
    "gdp_usd_real":    ("NY.GDP.MKTP.KD", "GDP in constant US dollars", "millions of constant 2015 US$"),
    "gdp_pc_usd_real": ("NY.GDP.PCAP.KD", "GDP per capita in constant US dollars", "constant 2015 US$ per person"),
    # --- prices and credit ------------------------------------------------------
    "cpi":         ("FP.CPI.TOTL", "Consumer price index (IMF IFS)", "index, 2010 = 100"),
    "cpi_infl":    ("FP.CPI.TOTL.ZG", "Consumer price inflation, annual", "percent"),
    "credit_priv": ("FS.AST.PRVT.GD.ZS", "Domestic credit to the private sector, all financial corporations", "percent of GDP"),
    "credit_priv_banks": ("FD.AST.PRVT.GD.ZS", "Domestic credit to the private sector by banks (longer back-series)", "percent of GDP"),
}

#: National-accounts names that have a constant-price twin in WDI (the
#: ``.CN`` code with ``.KN``), fetched as ``<name>_real`` when ``real=True``.
#: Constant LCU at the country's own base year (fixed-base economies) or
#: reference year (chain-linked ones): see :func:`wdi_meta`.
WDI_REAL: dict[str, str] = {
    name: code[:-3] + ".KN"
    for name, (code, _, _) in WDI_INDICATORS.items()
    if name in ("gdp", "cons_hh", "cons_gov", "inv", "capform", "inventories",
                "stat_disc", "exports", "imports", "taxes_prod", "va_total",
                "va_agri", "va_ind", "va_mfg", "va_services", "gni")
}

#: Items whose sign changes, so a ratio of nominal to real is not a deflator.
#: Net taxes on products go negative where subsidies exceed taxes (AGO, ARG,
#: EGY, MNG, ...), which made ``taxes_prod_defl`` negative in 111 cells.
_NO_DEFLATOR: frozenset[str] = frozenset({"inventories", "stat_disc", "taxes_prod"})

#: What ``indicators=None`` pulls: the national accounts with their real
#: twins, population, the ILO labour block, PPP, the exchange rate, the CPI
#: and private credit. 35 names, 51 requests with ``real=True``.
WDI_CORE: tuple[str, ...] = (
    "gdp", "cons_hh", "cons_gov", "inv", "capform", "inventories", "stat_disc",
    "exports", "imports", "taxes_prod", "va_total", "va_agri", "va_ind",
    "va_mfg", "va_services", "gni",
    "pop", "pop_0014", "pop_1564", "pop_65up",
    "lf", "urate", "epop", "lfpr", "selfemp_share", "employees_share",
    "emp_share_agri", "emp_share_ind", "emp_share_services",
    "gdp_ppp", "gdp_pc_ppp", "ppp", "fx",
    "cpi", "credit_priv",
)

#: Rescaling applied to what WDI publishes, keyed by the leading word of the
#: units string: WDI levels come in full units, the house unit is millions
#: (currency) and thousands (persons). The value is the house ``unit_mult``.
_SCALE: dict[str, int] = {"millions": 6, "thousands": 3}

_META_COLS = ["variable", "indicator", "label", "description", "units",
              "unit_mult", "sa", "first", "last", "n", "n_codes", "lastupdated"]
_COUNTRY_COLS = ["iso2", "name", "region", "income_group"]
_CMETA_COLS = ["name", "currency", "base_year", "ref_year", "va_valuation", "sna", "notes"]


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def _get(url: str, *, timeout: float = _TIMEOUT, pause: float = _PAUSE,
         refresh: bool = False) -> bytes:
    """The one HTTP seam of this module; tests monkeypatch this name.

    Cached on disk outside the repository, paced ``pause`` seconds per host,
    asks for gzip (undone by the transport). Raises ``urllib.error.HTTPError``
    or ``OSError`` on failure, which the callers turn into ``attrs["missing"]``.
    """
    from puremacro._http import safe_get_bytes_cached
    return safe_get_bytes_cached(url, timeout, headers=_HEADERS,
                                 rate_limit_seconds=pause, refresh=refresh)


def _fetch_json(url: str, *, timeout: float, pause: float,
                refresh: bool) -> tuple[object, str]:
    """One request decoded as JSON: ``(body, "ok")`` or ``(None, reason)``.

    A transport error or a 429 / 5xx answer is retried once after
    ``_RETRY_SLEEP`` seconds: the probe saw one timeout in 118 pulls and it
    succeeded on retry. Other HTTP errors are final.
    """
    import urllib.error

    status = "no attempt"
    for attempt in (1, 2):
        try:
            body = _get(url, timeout=timeout, pause=pause, refresh=refresh)
        except urllib.error.HTTPError as exc:
            status = f"HTTP {exc.code}"
            if attempt == 1 and (exc.code == 429 or exc.code >= 500):
                time.sleep(_RETRY_SLEEP)
                continue
            return None, status
        except OSError as exc:            # URLError, TimeoutError, bad SSL
            status = type(exc).__name__
            if attempt == 1:
                time.sleep(_RETRY_SLEEP)
                continue
            return None, status
        except Exception as exc:          # noqa: BLE001 - e.g. a corrupt cache file
            return None, f"{type(exc).__name__}: {exc}"
        try:
            return json.loads(body.decode("utf-8")), "ok"
        except (ValueError, UnicodeDecodeError):
            # An HTTP-200 body that is not JSON (empty, an HTML error page)
            # is cached for 30 days by the transport; fetch it once afresh so
            # the bad copy is overwritten instead of served on every run.
            if not refresh:
                refresh = True
                status = "bad JSON"
                continue
            return None, "bad JSON"
    return None, status


def _in_band_error(body: object) -> str | None:
    """The API's error message when it answers HTTP 200 with ``[{"message": [...]}]``."""
    if isinstance(body, list) and len(body) == 1 and isinstance(body[0], dict) \
            and "message" in body[0]:
        parts = []
        for m in body[0]["message"] or []:
            if isinstance(m, dict):
                parts.append(": ".join(str(m.get(k, "")) for k in ("id", "key", "value")))
        return "WDI message " + ("; ".join(parts) or "(empty)")
    if isinstance(body, dict) and "message" in body:
        return f"WDI message {body['message']}"
    return None


#: Failure reasons whose body came back HTTP 200 and was therefore cached;
#: :func:`_pages` re-fetches once with ``refresh=True`` before giving up.
#: (``bad JSON`` is already re-fetched inside :func:`_fetch_json`.)
_CACHED_FAILURES: tuple[str, ...] = ("WDI message", "unexpected body", "incomplete pagination")


def _pages(url: str, *, timeout: float, pause: float,
           refresh: bool) -> tuple[dict, list, str]:
    """Every page of a ``[meta, rows]`` endpoint: ``(meta, rows, "ok")`` or the reason.

    ``url`` already carries its query string; ``&page=k`` is appended. The
    row count is checked against ``meta["total"]`` so a page that went
    missing is reported, not silently truncated (the defect the old callers had).
    A failure that arrived as an HTTP-200 body (in-band error, malformed or
    short answer) sits in the on-disk cache, so it is retried once with
    ``refresh=True``; only a second failure is reported.
    """
    meta, rows, status = _pages_once(url, timeout=timeout, pause=pause, refresh=refresh)
    if status != "ok" and not refresh and status.startswith(_CACHED_FAILURES):
        meta, rows, status = _pages_once(url, timeout=timeout, pause=pause, refresh=True)
    return meta, rows, status


def _pages_once(url: str, *, timeout: float, pause: float,
                refresh: bool) -> tuple[dict, list, str]:
    """One pass over every page of ``url``; see :func:`_pages`."""
    rows: list = []
    meta: dict = {}
    page, pages = 1, 1
    while page <= pages:
        body, status = _fetch_json(f"{url}&page={page}", timeout=timeout,
                                   pause=pause, refresh=refresh)
        if body is None:
            return meta, [], status
        err = _in_band_error(body)
        if err is not None:
            return meta, [], err
        if not (isinstance(body, list) and len(body) == 2 and isinstance(body[0], dict)):
            return meta, [], "unexpected body"
        meta = body[0]
        if body[1] is None:              # ``date=2026`` style: pages 0, no table
            break
        if not isinstance(body[1], list):
            return meta, [], "unexpected body"
        rows.extend(body[1])
        try:
            pages = int(meta.get("pages", 1) or 0)
        except (TypeError, ValueError):
            pages = 1
        page += 1
    try:
        total = int(meta.get("total", len(rows)) or 0)
    except (TypeError, ValueError):
        total = len(rows)
    if len(rows) != total:
        return meta, [], f"incomplete pagination ({len(rows)} of {total} rows)"
    return meta, rows, "ok"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Economies and their metadata
# ---------------------------------------------------------------------------

def _empty_countries(reason: str) -> pd.DataFrame:
    out = pd.DataFrame(columns=_COUNTRY_COLS, index=pd.Index([], name="code"))
    out.attrs.update(source=_SOURCE, fetched_at=_now(), missing=({"item": "countries", "reason": reason},),
                     aggregates=())
    return out


def wdi_countries(*, refresh: bool = False) -> pd.DataFrame:
    """The 217 WDI reporting economies, indexed by ISO3 code.

    One request to ``/v2/country``. An economy is an id whose ``region.id``
    is not ``"NA"``; the other 79 ids (regions, income and lending groups,
    the world) are aggregates and are listed in ``attrs["aggregates"]``.
    Territories count as economies (Aruba, Puerto Rico, Hong Kong, Kosovo).

    Returns
    -------
    pandas.DataFrame
        Index ``code`` (ISO3), columns ``iso2``, ``name``, ``region``,
        ``income_group``. ``attrs``: ``source``, ``fetched_at``,
        ``aggregates`` and ``missing`` (empty unless the request failed, in
        which case the frame is empty too and a warning is issued).

    Examples
    --------
    >>> from puremacro.fetch.wdi import wdi_countries
    >>> wdi_countries().loc["MEX"]                          # doctest: +SKIP
    iso2                                 MX
    name                             Mexico
    region          Latin America & Caribbean
    income_group        Upper middle income
    """
    url = f"{_BASE}/country?format=json&per_page=400"
    _, rows, status = _pages(url, timeout=_TIMEOUT, pause=_PAUSE, refresh=refresh)
    if status != "ok":
        warnings.warn(f"wdi_countries: {status}; returning an empty frame", stacklevel=2)
        return _empty_countries(status)
    recs, aggregates = [], []
    for r in rows:
        if not isinstance(r, dict) or not r.get("id"):
            continue
        code = str(r["id"]).upper()
        region = r.get("region") or {}
        if region.get("id") == "NA" or code in WB_AGGREGATES:
            aggregates.append(code)
            continue
        recs.append({
            "code": code,
            "iso2": r.get("iso2Code", ""),
            "name": str(r.get("name", "")).strip(),
            "region": str(region.get("value", "")).strip(),
            "income_group": str((r.get("incomeLevel") or {}).get("value", "")).strip(),
        })
    if not recs:
        warnings.warn("wdi_countries: the country list had no economies", stacklevel=2)
        return _empty_countries("no economies in the country list")
    out = (pd.DataFrame(recs, columns=["code"] + _COUNTRY_COLS)
           .sort_values("code").set_index("code"))
    out.attrs.update(source=_SOURCE, fetched_at=_now(), missing=(),
                     aggregates=tuple(sorted(aggregates)))
    return out


def _empty_meta(reason: str) -> pd.DataFrame:
    out = pd.DataFrame(columns=_CMETA_COLS, index=pd.Index([], name="code"))
    out.attrs.update(source=_SOURCE, fetched_at=_now(),
                     missing=({"item": "metadata", "reason": reason},))
    return out


def _check_codes(codes: Iterable[str] | None, economies: pd.Index) -> list[str] | None:
    """Upper-cased caller codes, or ``ValueError`` naming the ones WDI does not know."""
    if codes is None:
        return None
    if isinstance(codes, str):
        codes = [codes]
    wanted = sorted({str(c).strip().upper() for c in codes})
    unknown = [c for c in wanted if c not in economies]
    if unknown:
        raise ValueError(f"not WDI economies: {unknown}; the {len(economies)} valid "
                         "codes are the index of wdi_countries() (aggregates such as "
                         "EMU, OED or WLD are not economies)")
    return wanted


def wdi_meta(codes: Iterable[str] | None = None, *, refresh: bool = False) -> pd.DataFrame:
    """National-accounts metadata per economy: currency, base year, SNA vintage, notes.

    One request to the source-2 country metadata endpoint, filtered to the
    economies of :func:`wdi_countries`. ``base_year`` is the fixed base year
    of the constant-price series or the sentence ``"Original chained constant
    price data are rescaled."`` for chain-linked economies, whose anchor is
    then ``ref_year``; both are kept as the World Bank writes them (India's
    is ``"2022/23"``). ``notes`` is the ``SpecialNotes`` text, which is where
    fiscal-year reporting and series breaks are declared.

    Parameters
    ----------
    codes : iterable of str, optional
        ISO3 economy codes; ``None`` for all 217. A code that is not a WDI
        economy raises ``ValueError``.
    refresh : bool
        Re-fetch instead of reading the on-disk cache.

    Returns
    -------
    pandas.DataFrame
        Index ``code``; columns ``name``, ``currency``, ``base_year``,
        ``ref_year``, ``va_valuation``, ``sna``, ``notes``. Empty with the
        same shape and ``attrs["missing"]`` set when the provider fails.
    """
    countries = wdi_countries(refresh=refresh)
    if countries.empty:
        return _empty_meta("country list unavailable")
    wanted = _check_codes(codes, countries.index)
    url = f"{_BASE}/sources/2/country/all/metadata?format=json&per_page=20000"
    body, status = _fetch_json(url, timeout=_TIMEOUT, pause=_PAUSE, refresh=refresh)
    err = _in_band_error(body) if body is not None else None
    if err is not None and not refresh:      # a cached in-band error: fetch afresh once
        body, status = _fetch_json(url, timeout=_TIMEOUT, pause=_PAUSE, refresh=True)
        err = _in_band_error(body) if body is not None else None
    if body is None or err is not None:
        reason = err or status
        warnings.warn(f"wdi_meta: {reason}; returning an empty frame", stacklevel=2)
        return _empty_meta(reason)
    variables: list = []
    for src in (body.get("source") or []) if isinstance(body, dict) else []:
        for concept in src.get("concept") or []:
            if concept.get("id") == "Country":
                variables.extend(concept.get("variable") or [])
    keep = set(wanted) if wanted is not None else set(countries.index)
    recs = []
    for v in variables:
        code = str(v.get("id", "")).upper()
        if code not in keep:
            continue
        fields = {m.get("id"): str(m.get("value", "")).strip()
                  for m in v.get("metatype") or [] if isinstance(m, dict)}
        recs.append({
            "code": code,
            "name": countries.at[code, "name"] if code in countries.index
                    else fields.get("ShortName", ""),
            "currency": fields.get("CurrencyUnit", ""),
            "base_year": fields.get("Nationalaccountsbaseyear", ""),
            "ref_year": fields.get("Nationalaccountsreferenceyear", ""),
            "va_valuation": fields.get("SNApricevaluation", ""),
            "sna": fields.get("SystemofNationalAccounts", ""),
            "notes": fields.get("SpecialNotes", ""),
        })
    if not recs:
        warnings.warn("wdi_meta: no economy in the metadata response", stacklevel=2)
        return _empty_meta("no economies in the metadata response")
    out = pd.DataFrame(recs, columns=["code"] + _CMETA_COLS).sort_values("code").set_index("code")
    missing = tuple({"item": c, "reason": "no metadata record"}
                    for c in sorted(keep - set(out.index)))
    out.attrs.update(source=_SOURCE, fetched_at=_now(), missing=missing)
    return out


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

def _columns(plan: Iterable[str]) -> list[str]:
    """The columns a successful call returns for ``plan``, derived deflators included."""
    plan = list(plan)
    out: list[str] = []
    for column in plan:
        out.append(column)
        base = column[:-5] if column.endswith("_real") else None
        if base is not None and base in plan and base in WDI_REAL \
                and base not in _NO_DEFLATOR:
            out.append(base + "_defl")
    return out


def _empty(long: bool, columns: Iterable[str] = ()) -> pd.DataFrame:
    """The frame a provider failure returns: same index names, columns and ``attrs`` keys."""
    columns = _columns(columns)
    if long:
        out = pd.DataFrame({"code": pd.Series(dtype=object),
                            "date": pd.Series(dtype="datetime64[ns]"),
                            "variable": pd.Series(dtype=object),
                            "value": pd.Series(dtype=float)})
    else:
        idx = pd.MultiIndex.from_arrays(
            [pd.Index([], dtype=object), pd.DatetimeIndex([], dtype="datetime64[ns]")], names=["code", "date"])
        out = pd.DataFrame({c: pd.Series(dtype=float) for c in columns}, index=idx)
    out.attrs.update(meta=(), source=_SOURCE, fetched_at=_now(), missing=(), lastupdated=None)
    return out


def _plan(indicators, real: bool) -> dict[str, tuple[str, str, str]]:
    """Column -> (code, description, units) for the request, in column order.

    Accepts ``None`` (:data:`WDI_CORE`), ``"all"``, one name, a sequence of
    registry names (``"gdp_real"`` asks for the twin alone) or a mapping of
    new names to raw WDI codes, which come back unscaled. Unknown names raise
    ``ValueError`` with the registry.
    """
    plan: dict[str, tuple[str, str, str]] = {}
    if isinstance(indicators, Mapping):
        for name, code in indicators.items():
            if not isinstance(code, str) or not code.strip():
                raise ValueError(f"indicator {name!r} needs a WDI code string, got {code!r}")
            plan[str(name)] = (code.strip(), "ad hoc WDI indicator", "as published by WDI")
        return plan
    if indicators is None:
        names: list[str] = list(WDI_CORE)
    elif isinstance(indicators, str):
        names = list(WDI_INDICATORS) if indicators == "all" else [indicators]
    else:
        names = [str(n) for n in indicators]
    unknown = [n for n in names
               if n not in WDI_INDICATORS
               and not (n.endswith("_real") and n[:-5] in WDI_REAL)]
    if unknown:
        raise ValueError(f"unknown WDI variable(s) {unknown}; valid names are "
                         f"{sorted(WDI_INDICATORS)} plus '<name>_real' for "
                         f"{sorted(WDI_REAL)}, 'all', or a mapping name -> WDI code")
    for n in names:
        if n in WDI_INDICATORS:
            code, desc, units = WDI_INDICATORS[n]
            plan[n] = (code, desc, units)
            if real and n in WDI_REAL:
                plan[n + "_real"] = (WDI_REAL[n], desc + " (constant prices)",
                                     "millions of constant LCU, country base year")
        else:                                   # explicit "<name>_real"
            base = n[:-5]
            plan[n] = (WDI_REAL[base], WDI_INDICATORS[base][1] + " (constant prices)",
                       "millions of constant LCU, country base year")
    return plan


def _unit_mult(units: str) -> int:
    return _SCALE.get(units.split(" ", 1)[0].lower(), 0)


def _rows_to_series(rows: list, economies: frozenset[str]) -> tuple[pd.Series, str]:
    """``(code, year) -> value`` for the economies, from one indicator's rows."""
    codes, years, values = [], [], []
    for r in rows:
        if not isinstance(r, dict) or r.get("value") is None:
            continue
        code = r.get("countryiso3code")
        if not code or code not in economies:
            continue
        year = str(r.get("date", ""))
        if len(year) != 4 or not year.isdigit():
            continue                            # a sub-annual code was asked for
        try:
            values.append(float(r["value"]))
        except (TypeError, ValueError):
            continue
        codes.append(code)
        years.append(int(year))
    if not values:
        return pd.Series(dtype=float), "no observations for the requested economies"
    idx = pd.MultiIndex.from_arrays([codes, years], names=["code", "year"])
    s = pd.Series(values, index=idx, dtype=float)
    if s.index.has_duplicates:
        s = s[~s.index.duplicated(keep="first")]
    return s, "ok"


def wdi_panel(indicators: Iterable[str] | Mapping[str, str] | str | None = None,
              codes: Iterable[str] | None = None, *,
              start: int = 1960, end: int | None = None,
              real: bool = True, long: bool = False,
              refresh: bool = False, pause: float = _PAUSE,
              timeout: float = _TIMEOUT) -> pd.DataFrame:
    """Annual World Bank WDI panel for the reporting economies, one column per variable.

    Parameters
    ----------
    indicators : sequence of str, mapping, "all" or None
        Registry names from :data:`WDI_INDICATORS` (``"gdp_real"`` asks for
        a constant-price twin on its own). ``None`` pulls :data:`WDI_CORE`;
        ``"all"`` the whole registry. A mapping ``{name: "NY.GDP.MKTP.CD"}``
        adds raw WDI codes under the caller's names, unscaled.
    codes : iterable of str, optional
        ISO3 economy codes; ``None`` for all 217. Up to fifty codes are
        requested by name, more are taken from the all-economy response.
        A code that is not a WDI economy raises ``ValueError``.
    start, end : int
        First and last year; ``end=None`` means the current year. Nothing
        exists before 1960 and an earlier ``start`` is clamped to it.
    real : bool
        Also fetch the constant-LCU twin of every national-accounts name in
        the request that has one (:data:`WDI_REAL`) as ``<name>_real`` and
        derive ``<name>_defl = 100 * nominal / real`` where both are
        positive (NaN elsewhere; not at all for the sign-changing items
        ``inventories``, ``stat_disc`` and ``taxes_prod``).
    long : bool
        Return ``[code, date, variable, value]`` instead of the wide frame.
    refresh : bool
        Re-fetch every request instead of reading the on-disk cache.
    pause : float
        Seconds between requests to the API (process-wide, per host).
    timeout : float
        Seconds per request; one retry after a transport error.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(code, date)`` with ``date`` the first of January of the
        year (``datetime64``), sorted, rows with no value dropped. Currency
        levels in millions of LCU (or US$), persons in thousands, rates in
        percent. ``attrs``: ``meta`` (a tuple of dicts, one per column:
        variable, indicator, label, description, units, unit_mult, sa, first,
        last, n, n_codes, lastupdated), ``source``, ``fetched_at``,
        ``lastupdated`` (the WDI release date) and ``missing`` (a tuple of
        dicts naming every column asked for but not obtained, with the
        reason). A provider failure never raises: the frame is empty, or
        lacks the failed columns, and a warning says so.

    Examples
    --------
    >>> from puremacro.fetch.wdi import wdi_panel
    >>> p = wdi_panel(["gdp", "pop"], ["MEX", "USA"], start=2015)   # doctest: +SKIP
    >>> p.loc["MEX"].tail(2)                                          # doctest: +SKIP
                         gdp      gdp_real   gdp_defl        pop
    date
    2024-01-01  3.4e+07 ...
    >>> p.attrs["meta"][0]["units"]                                   # doctest: +SKIP
    'millions of current LCU'
    """
    plan = _plan(indicators, real)
    start_y = max(int(start), _FIRST_YEAR)
    end_y = dt.date.today().year if end is None else int(end)
    if end_y < start_y:
        raise ValueError(f"end ({end_y}) is before start ({start_y})")
    if pause < 0 or timeout <= 0:
        raise ValueError("pause must be >= 0 and timeout > 0 seconds")

    countries = wdi_countries(refresh=refresh)
    if countries.empty:
        out = _empty(long, plan)
        out.attrs["missing"] = tuple({"variable": v, "indicator": c, "reason": "country list unavailable"}
                                     for v, (c, _, _) in plan.items())
        warnings.warn("wdi_panel: the WDI country list could not be fetched; empty frame",
                      stacklevel=2)
        return out
    wanted = _check_codes(codes, countries.index)
    economies = frozenset(countries.index) if wanted is None else frozenset(wanted)
    path = "all" if wanted is None or len(wanted) > 50 else ";".join(wanted)

    # One request per distinct code (two names may share one).
    fetched: dict[str, tuple[pd.Series, dict, str, str]] = {}
    for column, (code, _, _) in plan.items():
        if code in fetched:
            continue
        url = (f"{_BASE}/country/{path}/indicator/{code}"
               f"?format=json&date={start_y}:{end_y}&per_page={_PER_PAGE}")
        meta, rows, status = _pages(url, timeout=timeout, pause=pause, refresh=refresh)
        if status == "ok":
            series, status = _rows_to_series(rows, economies)
        else:
            series = pd.Series(dtype=float)
        label = ""
        for r in rows:
            if isinstance(r, dict) and isinstance(r.get("indicator"), dict):
                label = str(r["indicator"].get("value", "")).strip()
                break
        fetched[code] = (series, meta or {}, status, label)

    pieces: dict[str, pd.Series] = {}
    meta_rows: list[dict] = []
    missing: list[dict] = []
    for column, (code, desc, units) in plan.items():
        series, meta, status, label = fetched[code]
        if status != "ok":
            missing.append({"variable": column, "indicator": code, "reason": status})
            continue
        mult = _unit_mult(units)
        if mult:
            series = series / (10.0 ** mult)
        pieces[column] = series
        years = series.index.get_level_values("year")
        meta_rows.append({
            "variable": column, "indicator": code, "label": label,
            "description": desc, "units": units, "unit_mult": mult,
            "sa": "NSA, annual (calendar or fiscal year, see wdi_meta)",
            "first": int(years.min()), "last": int(years.max()),
            "n": int(series.size), "n_codes": int(series.index.get_level_values("code").nunique()),
            "lastupdated": meta.get("lastupdated"),
        })
    if missing:
        warnings.warn("wdi_panel: not obtained: "
                      + "; ".join(f"{m['variable']} ({m['indicator']}: {m['reason']})" for m in missing),
                      stacklevel=2)
    if not pieces:
        out = _empty(long, plan)
        out.attrs["missing"] = tuple(missing)
        return out

    wide = pd.DataFrame(pieces)
    # Deflators, derived rather than downloaded: 100*CN/KN reproduces WDI's
    # own NY.GDP.DEFL.ZS to 1e-15, and the component deflators WDI does not
    # publish come the same way.
    order: list[str] = []
    for column in plan:
        if column not in wide.columns:
            continue
        order.append(column)
        base = column[:-5] if column.endswith("_real") else None
        if base is not None and base in wide.columns and base not in _NO_DEFLATOR \
                and base in WDI_REAL:
            # Only a positive nominal over a positive real is a price index:
            # WDI publishes nominal 0.0 for VEN 1991-2011 (and BRA va_agri)
            # after the redenominations, and capform goes negative in a few
            # small economies. Those cells are set to NaN and counted.
            nom, vol = wide[base], wide[column]
            both = nom.notna() & vol.notna()
            valid = both & (nom > 0) & (vol > 0)
            defl = (100.0 * nom / vol).where(valid)
            name = base + "_defl"
            wide[name] = defl
            order.append(name)
            ok = defl.dropna()
            yrs = ok.index.get_level_values("year")
            meta_rows.append({
                "variable": name, "indicator": f"100*{plan[base][0]}/{plan[column][0]}",
                "label": "Implicit price deflator (derived)",
                "description": plan[base][1] + ", implicit deflator",
                "units": "index, 100 = country base year", "unit_mult": 0,
                "sa": "NSA, annual", "first": int(yrs.min()) if len(ok) else None,
                "last": int(yrs.max()) if len(ok) else None, "n": int(ok.size),
                "n_codes": int(ok.index.get_level_values("code").nunique()) if len(ok) else 0,
                "n_masked": int((both & ~valid).sum()),
                "lastupdated": fetched[plan[base][0]][1].get("lastupdated"),
            })
    wide = wide[order].dropna(how="all").sort_index()
    codes_lvl = wide.index.get_level_values("code")
    dates = pd.to_datetime(wide.index.get_level_values("year").astype(int).astype(str),
                           format="%Y").astype("datetime64[ns]")
    wide.index = pd.MultiIndex.from_arrays([codes_lvl, dates], names=["code", "date"])
    wide = wide.astype(float)

    by_var = {m["variable"]: m for m in meta_rows}
    meta_tuple = tuple(by_var[v] for v in order)
    releases = sorted({m["lastupdated"] for m in meta_tuple if m.get("lastupdated")})
    attrs = dict(meta=meta_tuple, source=_SOURCE, fetched_at=_now(), missing=tuple(missing),
                 lastupdated=releases[-1] if releases else None)
    if long:
        out = (wide.reset_index()
               .melt(id_vars=["code", "date"], var_name="variable", value_name="value")
               .dropna(subset=["value"])
               .sort_values(["code", "variable", "date"]).reset_index(drop=True))
    else:
        out = wide
    out.attrs.update(attrs)
    return out


__all__ = ["wdi_panel", "wdi_countries", "wdi_meta",
           "WDI_INDICATORS", "WDI_CORE", "WDI_REAL"]
