r"""IMF data through the new SDMX 2.1 API: national accounts, prices, rates, WEO.

``imf_nea_panel()`` returns the widest expenditure-side national-accounts
panel the IMF publishes: about 190 economies, annual from 1950 (quarterly for
some 75 economies seasonally adjusted, 114 unadjusted), indexed by
``(code, date)`` with one column per variable::

    gdp  cons  cons_gov  cons_hh  cons_priv  inv  capform  inventories  exports  imports
    gdp_real  cons_real  ...   gdp_defl  cons_defl  ...           (real=True)

and the companion one-call builders do the same for the other IMF flows a
macro course needs: :func:`imf_monthly_panel` (CPI and three COICOP groups,
industrial production, unemployment, seven interest rates, effective and
bilateral exchange rates), :func:`imf_labour_panel`, :func:`imf_pcps`
(primary commodity prices), :func:`imf_weo` (World Economic Outlook, with a
``forecast`` flag) and :func:`imf_icsd` (public and private investment and
capital stocks). :func:`imf_get` is the raw reader underneath all of them and
:func:`imf_dataflows` lists what else the API serves.

Why this module exists
----------------------
Both IMF paths the package had stopped working in 2025-26. The legacy
``dataservices.imf.org`` host behind :mod:`puremacro.fetch.imf_ifs` no longer
resolves (NXDOMAIN), and ``sdmxcentral.imf.org`` is a registry that answers
``501 Data Queries are not implemented``. The IMF now serves data at
``api.imf.org/external/sdmx/2.1``, with ISO3 country codes, agency-qualified
flow ids and the old IFS concepts redistributed over topical flows (``ANEA``,
``QNEA``, ``CPI``, ``MFS_IR``, ``EER``, ``ER``, ``PI``, ``LS``). For the
OECD economies the OECD and Eurostat flows are usually longer; the IMF adds
the other 150 economies under the same conventions.

Mechanics
---------
Every request is ``GET data/{AGENCY},{FLOW}/{KEY}?detail=dataonly`` with the
header ``Accept: application/vnd.sdmx.data+csv;version=1.0.0`` (without it
the API answers XML; without ``detail=dataonly`` every row repeats forty
metadata columns and the body is twenty times larger). Keys list dimension
values joined by ``+``, a blank dimension is a wildcard, and an all-wildcard
key is sent as ``all`` because a path segment ``..`` is refused with a 403 by
the IMF's application gateway. Bodies travel through
:func:`puremacro._http.safe_get_bytes_cached` (urllib, SQLite cache outside
the repository, ``pause`` seconds between requests to the host), so a
notebook that freezes a panel re-reads today's bytes instead of asking
again; ``refresh=True`` re-fetches. The backend is flaky (sporadic ``HTTP
500`` with code 50000 and 60-second stalls on URLs that answered seconds
before), so 5xx answers and timeouts are retried ``retries`` times with a
growing pause, and a 429 waits a minute. A failure never raises: the
function returns what it could build (an empty frame with the same index
names in the worst case), lists what it could not in ``attrs["missing"]``
and warns. ``ValueError`` is kept for caller errors.

Codes and dates. Countries are ISO3; the IMF's ``KOS`` (Kosovo) and ``WBG``
(West Bank and Gaza) are renamed to the World Bank's ``XKX`` and ``PSE`` so
panels join with :mod:`puremacro.fetch.wdi`. IMF group codes (``G001``
world, ``G163`` euro area and the other ``G``/``GX`` codes listed in
:data:`IMF_AGGREGATES`) are dropped. Historical economies (``SUN``, ``YUG``,
``CSK``, ``DDR``, ``ANT``, ``YAR``, ``YMD``, ``CWX``) are kept as published.
Periods ``2024``, ``2024-Q1`` and ``2025-M01`` become period-start
timestamps.

Units. ``OBS_VALUE`` is in plain units whatever the ``SCALE`` attribute
says, except in ``ICSD``, which is in billions. Everything is rescaled to
the house conventions: national-currency levels to *millions*, persons to
*thousands*; rates, indices and prices are left as published. Each
``attrs["meta"]`` row records the stored unit as an SDMX power of ten
(``unit_mult``: 6 for millions, 3 for thousands, 0 otherwise) and the factor
applied to the IMF's ``OBS_VALUE`` to get there (``scale``: ``1e-6`` for NEA
and WEO currency levels, ``1e3`` for ICSD, ``1e-3`` for persons), the same
convention as :mod:`puremacro.fetch.bis` and :mod:`puremacro.fetch.eurostat`.
Volumes (``_real``) are at constant prices of each country's own reference
year, chain-linked for most economies, so they do not add up. Deflators
(``_defl``) are implicit, ``100 * nominal / real``: their base is the
country's constant-price reference year, which is not always inside the
published span (Brazil's, Ecuador's or Australia's ``gdp_defl`` never comes
near 100), so rebase before comparing levels across countries. The IMF's own
``PD`` deflator (2010 = 100) exists for GDP only, which is why it is not used.

Placeholders. Some flows publish exact zeros where there are no data
(Burkina Faso's whole expenditure side 1952-1998 in ``ANEA``, capital stocks
of economies ``ICSD`` does not cover). In the strictly positive levels and
indices (GDP and its components, their volumes, capital stocks, prices,
employment) a zero is set to missing; the count per column is in
``attrs["zeros_dropped"]``, with a warning. Zeros are kept where they can be
true: changes in inventories, the PPP capital stock, unemployment and
interest rates (Denmark's policy rate was 0 for years).

Time budget. Each request is tried ``retries + 1`` times (up to 180 s
each), so a stalled backend can hold one flow for about 12 minutes; a
multi-flow builder (:func:`imf_monthly_panel`) stops asking after two
consecutive flows fail that way and lists the rest as skipped.

What the IMF does not publish here: the income side of the accounts
(compensation of employees, operating surplus), consumption by durability,
GFCF by asset or sector in the NEA flows (sectoral GFCF is in ICSD, to
2019), hours, self-employment, core or energy CPI, money-market tenors and
corporate yields. Those live in the OECD, Eurostat and ILO fetchers.

Source: International Monetary Fund, SDMX 2.1 data API,
https://api.imf.org/external/sdmx/2.1 (flows IMF.STA ANEA, QNEA, CPI, PI,
LS, MFS_IR, EER, ER; IMF.RES PCPS, WEO; IMF.FAD ICSD).
"""
from __future__ import annotations

import datetime as dt
import io
import re
import time
import urllib.error
import warnings
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

_BASE = "https://api.imf.org/external/sdmx/2.1"
_SOURCE = "IMF SDMX 2.1 API, api.imf.org/external/sdmx/2.1"
_ACCEPT_CSV = "application/vnd.sdmx.data+csv;version=1.0.0"
_TIMEOUT = 180.0
_PAUSE = 3.0
_RETRY_SLEEP = 10.0        # first retry after 10 s, then 20 s, 30 s
_SLEEP_429 = 60.0
_MAX_FAILED_FLOWS = 2      # a multi-flow builder stops after this many consecutive failures
_INDEX_NAMES = ["code", "date"]

#: Short flow name -> agency-qualified flow id, for the flows this module
#: reads. :func:`imf_get` accepts either form; any other flow must be passed
#: agency-qualified (``"IMF.STA,PPI"``), as :func:`imf_dataflows` lists them.
IMF_FLOWS: dict[str, str] = {
    "ANEA": "IMF.STA,ANEA",      # national economic accounts, annual
    "QNEA": "IMF.STA,QNEA",      # national economic accounts, quarterly
    "CPI": "IMF.STA,CPI",        # consumer prices (CPI and HICP, COICOP 1999)
    "PI": "IMF.STA,PI",          # production indices
    "LS": "IMF.STA,LS",          # labour statistics
    "MFS_IR": "IMF.STA,MFS_IR",  # monetary and financial statistics: interest rates
    "EER": "IMF.STA,EER",        # effective exchange rates
    "ER": "IMF.STA,ER",          # bilateral exchange rates
    "PCPS": "IMF.RES,PCPS",      # primary commodity prices
    "WEO": "IMF.RES,WEO",        # World Economic Outlook (latest vintage)
    "ICSD": "IMF.FAD,ICSD",      # investment and capital stock dataset
}

#: Dimension order of each flow's key, between COUNTRY (first) and
#: FREQUENCY (last). PCPS has COUNTRY = G001 only.
_FLOW_DIMS: dict[str, tuple[str, ...]] = {
    "IMF.STA,ANEA": ("INDICATOR", "PRICE_TYPE", "TYPE_OF_TRANSFORMATION"),
    "IMF.STA,QNEA": ("INDICATOR", "PRICE_TYPE", "S_ADJUSTMENT", "TYPE_OF_TRANSFORMATION"),
    "IMF.STA,CPI": ("INDEX_TYPE", "COICOP_1999", "TYPE_OF_TRANSFORMATION"),
    "IMF.STA,PI": ("PRODUCTION_INDEX", "TYPE_OF_TRANSFORMATION"),
    "IMF.STA,LS": ("INDICATOR", "TYPE_OF_TRANSFORMATION"),
    "IMF.STA,MFS_IR": ("INDICATOR",),
    "IMF.STA,EER": ("INDICATOR",),
    "IMF.STA,ER": ("INDICATOR", "TYPE_OF_TRANSFORMATION"),
    "IMF.RES,PCPS": ("INDICATOR", "DATA_TRANSFORMATION"),
    "IMF.RES,WEO": ("INDICATOR",),
    "IMF.FAD,ICSD": ("INDICATOR",),
}

#: IMF group codes seen in the data flows (world, euro area, regional and
#: analytical groups). Dropped by this explicit list; any other code with a
#: digit in it is also an IMF group code and is dropped with a warning.
IMF_AGGREGATES: frozenset[str] = frozenset({
    "G001",                                   # world (PCPS, WEO)
    "G110", "G119", "G200", "G205", "G400",   # WEO country groups
    "G505", "G510", "G603", "G903", "G998",
    "G163",                                   # euro area
    "G129", "G309", "G758", "G759",           # ER/EER currency groups
    "GX123",                                  # WEO analytical group
})

#: IMF country codes that differ from the World Bank / ISO3 code the rest of
#: the package uses.
IMF_CODE_RENAMES: dict[str, str] = {"KOS": "XKX", "WBG": "PSE"}
_RENAMES_BACK = {v: k for k, v in IMF_CODE_RENAMES.items()}

#: NEA column -> (IMF NA_STO indicator, description). ``cons_priv`` is not
#: here: it is derived as ``cons - cons_gov`` (households plus NPISH, the
#: concept :func:`puremacro.fetch.qna_panel` calls ``cons_hh``).
IMF_NEA_ITEMS: dict[str, tuple[str, str]] = {
    "gdp":         ("B1GQ",   "Gross domestic product"),
    "cons":        ("P3",     "Final consumption expenditure, all sectors"),
    "cons_gov":    ("P3_S13", "General government final consumption"),
    "cons_hh":     ("P3_S14", "Household final consumption, households only (no NPISH; 67 economies annual, 13 quarterly SA)"),
    "inv":         ("P51G",   "Gross fixed capital formation"),
    "capform":     ("P5",     "Gross capital formation"),
    "inventories": ("P52",    "Changes in inventories"),
    "exports":     ("P6",     "Exports of goods and services"),
    "imports":     ("P7",     "Imports of goods and services"),
}

#: Monthly column -> (flow, dimension values between COUNTRY and FREQUENCY,
#: description, units as returned, multiplier applied, seasonal adjustment).
IMF_MONTHLY: dict[str, tuple[str, tuple[str, ...], str, str, float, str]] = {
    # --- consumer prices (CPI flow, COICOP 1999 groups) ----------------------
    "cpi":           ("IMF.STA,CPI", ("CPI", "_T", "IX"), "Consumer price index, all items", "index, national reference period", 1.0, "NSA"),
    "cpi_food":      ("IMF.STA,CPI", ("CPI", "CP01", "IX"), "CPI, food and non-alcoholic beverages", "index, national reference period", 1.0, "NSA"),
    "cpi_housing":   ("IMF.STA,CPI", ("CPI", "CP04", "IX"), "CPI, housing, water, electricity, gas and other fuels", "index, national reference period", 1.0, "NSA"),
    "cpi_transport": ("IMF.STA,CPI", ("CPI", "CP07", "IX"), "CPI, transport", "index, national reference period", 1.0, "NSA"),
    # --- production indices --------------------------------------------------
    "ip":            ("IMF.STA,PI", ("IND", "IX"), "Industrial production index", "index, 2010 = 100", 1.0, "NSA"),
    "ip_sa":         ("IMF.STA,PI", ("IND", "SA_IX"), "Industrial production index, seasonally adjusted", "index, 2010 = 100", 1.0, "SA"),
    "ip_mfg":        ("IMF.STA,PI", ("C", "IX"), "Manufacturing production index", "index, 2010 = 100", 1.0, "NSA"),
    # --- labour --------------------------------------------------------------
    "urate":         ("IMF.STA,LS", ("U", "PT"), "Unemployment rate", "percent of labour force", 1.0, "NSA"),
    "emp":           ("IMF.STA,LS", ("E", "PE"), "Employed persons", "thousands of persons", 1e-3, "NSA"),
    "lf":            ("IMF.STA,LS", ("LF", "PE"), "Labour force", "thousands of persons", 1e-3, "NSA"),
    # --- interest rates (MFS_IR), percent per annum, period average ----------
    "policy_rate":   ("IMF.STA,MFS_IR", ("MFS166_RT_PT_A_PT",), "Monetary policy-related interest rate", "percent per annum", 1.0, "NSA"),
    "mm_rate":       ("IMF.STA,MFS_IR", ("MMRT_RT_PT_A_PT",), "Money market rate", "percent per annum", 1.0, "NSA"),
    "tbill_rate":    ("IMF.STA,MFS_IR", ("GSTBILY_RT_PT_A_PT",), "Treasury bill yield", "percent per annum", 1.0, "NSA"),
    "bond_yield":    ("IMF.STA,MFS_IR", ("S13BOND_RT_PT_A_PT",), "Government bond yield, long term", "percent per annum", 1.0, "NSA"),
    "discount_rate": ("IMF.STA,MFS_IR", ("DISR_RT_PT_A_PT",), "Central bank discount rate", "percent per annum", 1.0, "NSA"),
    "deposit_rate":  ("IMF.STA,MFS_IR", ("MFS135_RT_PT_A_PT",), "Deposit rate", "percent per annum", 1.0, "NSA"),
    "lending_rate":  ("IMF.STA,MFS_IR", ("MFS162_RT_PT_A_PT",), "Lending rate", "percent per annum", 1.0, "NSA"),
    # --- exchange rates ------------------------------------------------------
    "neer":          ("IMF.STA,EER", ("NEER_IX_RY2010_ACW",), "Nominal effective exchange rate, all-country weights (up = appreciation)", "index, 2010 = 100", 1.0, "NSA"),
    "reer":          ("IMF.STA,EER", ("REER_IX_RY2010_ACW_RCPI",), "Real effective exchange rate, CPI-based, all-country weights", "index, 2010 = 100", 1.0, "NSA"),
    "fx_usd":        ("IMF.STA,ER", ("XDC_USD", "PA_RT"), "National currency per US dollar, period average", "national currency per USD", 1.0, "NSA"),
    "fx_usd_eop":    ("IMF.STA,ER", ("XDC_USD", "EOP_RT"), "National currency per US dollar, end of period", "national currency per USD", 1.0, "NSA"),
}

#: Labour column -> (LS indicator, LS transformation, description, units, multiplier).
IMF_LABOUR: dict[str, tuple[str, str, str, str, float]] = {
    "urate": ("U", "PT", "Unemployment rate", "percent of labour force", 1.0),
    "emp":   ("E", "PE", "Employed persons", "thousands of persons", 1e-3),
    "lf":    ("LF", "PE", "Labour force", "thousands of persons", 1e-3),
    "unemp": ("UP", "PE", "Unemployed persons", "thousands of persons", 1e-3),
}

#: Commodity column -> (PCPS indicator, transformation, description, units).
#: Columns of the headline indices and benchmark prices; every other series
#: in the flow comes back as ``<code>_idx`` / ``<code>_usd`` (lower case).
IMF_PCPS_NAMED: dict[str, tuple[str, str, str, str]] = {
    "pcom_all":      ("PALLFNF", "INDEX", "All commodities", "index, 2016 = 100"),
    "pcom_nonfuel":  ("PNFUEL", "INDEX", "Non-fuel commodities", "index, 2016 = 100"),
    "pcom_energy":   ("PNRG", "INDEX", "Energy", "index, 2016 = 100"),
    "pcom_food":     ("PFOOD", "INDEX", "Food", "index, 2016 = 100"),
    "pcom_bev":      ("PBEVE", "INDEX", "Beverages", "index, 2016 = 100"),
    "pcom_agri":     ("PAGRI", "INDEX", "Agricultural raw materials", "index, 2016 = 100"),
    "pcom_metals":   ("PMETA", "INDEX", "Base metals", "index, 2016 = 100"),
    "pcom_allmetals": ("PALLMETA", "INDEX", "Base and precious metals", "index, 2016 = 100"),
    "pcom_precious": ("PPMETA", "INDEX", "Precious metals", "index, 2016 = 100"),
    "pcom_indus":    ("PINDU", "INDEX", "Industrial inputs", "index, 2016 = 100"),
    "poil":          ("POILAPSP", "USD", "Crude oil, average of Brent, WTI and Dubai", "USD per barrel"),
    "poil_brent":    ("POILBRE", "USD", "Crude oil, Brent", "USD per barrel"),
    "poil_wti":      ("POILWTI", "USD", "Crude oil, West Texas Intermediate", "USD per barrel"),
    "poil_dubai":    ("POILDUB", "USD", "Crude oil, Dubai", "USD per barrel"),
    "pgas_us":       ("PNGASUS", "USD", "Natural gas, US Henry Hub", "USD per million BTU"),
    "pgas_eu":       ("PNGASEU", "USD", "Natural gas, Europe", "USD per million BTU"),
    "pgas_jp":       ("PNGASJP", "USD", "LNG, Japan", "USD per million BTU"),
    "pcoal_au":      ("PCOALAU", "USD", "Coal, Australia", "USD per metric ton"),
    "pcopper":       ("PCOPP", "USD", "Copper", "USD per metric ton"),
    "paluminium":    ("PALUM", "USD", "Aluminium", "USD per metric ton"),
    "pgold":         ("PGOLD", "USD", "Gold", "USD per troy ounce"),
    "pwheat":        ("PWHEAMT", "USD", "Wheat", "USD per metric ton"),
    "pmaize":        ("PMAIZMT", "USD", "Maize", "USD per metric ton"),
    "psoy":          ("PSOYB", "USD", "Soybeans", "USD per metric ton"),
}

#: WEO column -> (WEO indicator, description, units as returned, multiplier).
#: The WEO ``SCALE`` attribute is a display hint only: values arrive in units.
IMF_WEO: dict[str, tuple[str, str, str, float]] = {
    "gdp":             ("NGDP", "GDP, current prices", "millions of national currency", 1e-6),
    "gdp_real":        ("NGDP_R", "GDP, constant prices", "millions of national currency, constant prices", 1e-6),
    "gdp_defl":        ("NGDP_D", "GDP deflator", "index, national base year", 1.0),
    "gdp_growth":      ("NGDP_RPCH", "Real GDP growth", "percent change", 1.0),
    "gdp_usd":         ("NGDPD", "GDP, current prices, US dollars", "millions of USD", 1e-6),
    "gdp_ppp":         ("PPPGDP", "GDP, current prices, PPP international dollars", "millions of international dollars", 1e-6),
    "gdp_pc_ppp":      ("NGDPRPPPPC", "GDP per capita, constant prices, PPP international dollars", "international dollars per person", 1.0),
    "inv_gdp":         ("NID_NGDP", "Total investment", "percent of GDP", 1.0),
    "sav_gdp":         ("NGSD_NGDP", "Gross national saving", "percent of GDP", 1.0),
    "cpi":             ("PCPI", "Consumer prices, period average", "index, national base year", 1.0),
    "inflation":       ("PCPIPCH", "Consumer price inflation, period average", "percent change", 1.0),
    "urate":           ("LUR", "Unemployment rate", "percent of labour force", 1.0),
    "emp":             ("LE", "Employment", "thousands of persons", 1e-3),
    "pop":             ("LP", "Population", "thousands of persons", 1e-3),
    "gov_netlend_gdp": ("GGXCNL_NGDP", "General government net lending/borrowing", "percent of GDP", 1.0),
    "gov_debt_gdp":    ("GGXWDG_NGDP", "General government gross debt", "percent of GDP", 1.0),
    "ca_gdp":          ("BCA_NGDPD", "Current account balance", "percent of GDP", 1.0),
}

#: ICSD column -> (ICSD indicator, description, units as returned, multiplier).
#: ICSD publishes levels in BILLIONS; they are multiplied by 1000 here.
IMF_ICSD: dict[str, tuple[str, str, str, float]] = {
    "gdp":            ("B1GQ_V_XDC", "GDP, current prices", "millions of national currency", 1e3),
    "inv_gov":        ("P51G_S13_V_XDC", "General government GFCF, current prices", "millions of national currency", 1e3),
    "inv_priv":       ("P51G_PS_V_XDC", "Private-sector GFCF, current prices", "millions of national currency", 1e3),
    "k_gov":          ("CAPSTCK_S13_V_XDC", "General government capital stock, current prices", "millions of national currency", 1e3),
    "k_priv":         ("CAPSTCK_PS_V_XDC", "Private capital stock, current prices", "millions of national currency", 1e3),
    "k_pubpriv":      ("CAPSTCK_PUPVT_V_XDC", "Public-private partnership capital stock, current prices", "millions of national currency", 1e3),
    "gdp_ppp":        ("B1GQ_Q_PU_RY2017", "GDP, constant 2017 PPP international dollars", "millions of 2017 international dollars", 1e3),
    "inv_gov_ppp":    ("P51G_S13_Q_PU_RY2017", "General government GFCF, constant 2017 PPP", "millions of 2017 international dollars", 1e3),
    "inv_priv_ppp":   ("P51G_PS_Q_PU_RY2017", "Private GFCF, constant 2017 PPP", "millions of 2017 international dollars", 1e3),
    "k_gov_ppp":      ("CAPSTCK_S13_Q_PU_RY2017", "General government capital stock, constant 2017 PPP", "millions of 2017 international dollars", 1e3),
    "k_priv_ppp":     ("CAPSTCK_PS_Q_PU_RY2017", "Private capital stock, constant 2017 PPP", "millions of 2017 international dollars", 1e3),
    "inv_gov_gdp":    ("P51G_S13_Q_POGDP_PT", "General government GFCF, constant prices", "percent of GDP", 1.0),
    "inv_priv_gdp":   ("P51G_PS_Q_POGDP_PT", "Private GFCF, constant prices", "percent of GDP", 1.0),
    "k_gov_gdp":      ("CAPSTCK_S13_Q_POGDP_PT", "General government capital stock, constant prices", "percent of GDP", 1.0),
    "k_priv_gdp":     ("CAPSTCK_PS_Q_POGDP_PT", "Private capital stock, constant prices", "percent of GDP", 1.0),
}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _get(url: str, *, timeout: float = _TIMEOUT, pause: float = _PAUSE,
         refresh: bool = False, accept: str | None = _ACCEPT_CSV) -> bytes:
    """One cached GET through :func:`puremacro._http.safe_get_bytes_cached`.

    The module-level seam that tests monkeypatch. Raises
    ``urllib.error.HTTPError`` / ``OSError`` on failure; :func:`_fetch`
    turns those into a status string.
    """
    from puremacro._http import safe_get_bytes_cached
    headers = {"Accept": accept} if accept else None
    return safe_get_bytes_cached(url, timeout, headers=headers,
                                 rate_limit_seconds=pause, refresh=refresh)


def _fetch(url: str, *, timeout: float, pause: float, retries: int,
           refresh: bool, accept: str | None = _ACCEPT_CSV) -> tuple[bytes | None, str]:
    """``(body, "ok")`` or ``(None, reason)``; retries 5xx, 429 and timeouts."""
    status = "not requested"
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(_SLEEP_429 if status == "HTTP 429" else _RETRY_SLEEP * attempt)
        try:
            return _get(url, timeout=timeout, pause=pause, refresh=refresh, accept=accept), "ok"
        except urllib.error.HTTPError as exc:
            status = f"HTTP {exc.code}"
            if exc.code != 429 and exc.code < 500:
                # 404 is SDMX "no results"; 400/403 are a bad key: no retry helps.
                return None, status
        except (TimeoutError, OSError) as exc:  # URLError is an OSError
            reason = getattr(exc, "reason", exc)
            status = "timeout" if isinstance(reason, TimeoutError) or "timed out" in str(reason) \
                else f"{type(exc).__name__}: {reason}"
    return None, status if retries == 0 else f"{status} after {retries + 1} attempts"


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------

_PERIOD_RE = re.compile(r"^\d{4}(-(\d{2}|M\d{2}|Q[1-4]))?$")


def _check_period(value: object, name: str) -> str | None:
    if value is None:
        return None
    s = str(value)
    if not _PERIOD_RE.match(s):
        raise ValueError(f"{name}={value!r} is not a period; use YYYY, YYYY-MM, YYYY-Q1 or YYYY-M01")
    return s


def _check_codes(codes: Iterable[str] | str | None) -> list[str] | None:
    """Upper-cased ISO3 codes, or ``ValueError`` for anything that is not one."""
    if codes is None:
        return None
    if isinstance(codes, str):
        codes = [codes]
    out: list[str] = []
    bad: list[object] = []
    for c in codes:
        if isinstance(c, str) and len(c.strip()) == 3 and c.strip().isalpha():
            out.append(c.strip().upper())
        else:
            bad.append(c)
    if bad:
        raise ValueError(f"codes must be ISO3 country codes (three letters); got {bad!r}")
    if not out:
        raise ValueError("codes is empty; pass None for every economy")
    return list(dict.fromkeys(out))


def _check_choice(value: str, name: str, choices: Sequence[str]) -> str:
    v = str(value).upper()
    if v not in choices:
        raise ValueError(f"{name}={value!r} is not valid; choose one of {list(choices)}")
    return v


def _country_part(codes: list[str] | None) -> str:
    if codes is None:
        return ""
    return "+".join(_RENAMES_BACK.get(c, c) for c in codes)


def _key(country: str, dim_values: Sequence[tuple[str, ...]], freq: str) -> str:
    """SDMX key: country part, each dimension's union of values, frequency."""
    parts = [country]
    n = len(dim_values[0]) if dim_values else 0
    for i in range(n):
        parts.append("+".join(dict.fromkeys(t[i] for t in dim_values)))
    parts.append(freq)
    return ".".join(parts)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def _empty_raw(url: str, status: str) -> pd.DataFrame:
    out = pd.DataFrame(columns=["DATAFLOW", "COUNTRY", "TIME_PERIOD", "OBS_VALUE"])
    out.attrs.update(url=url, fetched_at=_now(), status=status)
    return out


def _read_csv(body: bytes, detail: str) -> pd.DataFrame:
    header = body.split(b"\n", 1)[0].decode("utf-8", "replace").strip().lstrip("﻿")
    cols = header.split(",")
    if detail == "dataonly" and "OBS_VALUE" in cols:
        # Attributes after OBS_VALUE are empty under detail=dataonly; skipping
        # them keeps a 300 000-row pull at a tenth of the memory.
        usecols = cols[: cols.index("OBS_VALUE") + 1]
    else:
        usecols = None
    dtypes = {c: str for c in cols if c != "OBS_VALUE"}
    df = pd.read_csv(io.BytesIO(body), usecols=usecols, dtype=dtypes,
                     keep_default_na=False, na_values={"OBS_VALUE": ["", "NaN", "NA"]},
                     low_memory=False)
    if "OBS_VALUE" in df:
        df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    return df


def _period_start(periods: pd.Series) -> pd.Series:
    """``2024`` / ``2024-Q1`` / ``2025-M01`` / ``2025-01`` -> period-start timestamps."""
    s = periods.astype(str).str.strip()
    parts = s.str.extract(r"^(\d{4})(?:-(?:M?(\d{2})|Q([1-4])))?$")
    month = pd.to_numeric(parts[1], errors="coerce")
    month = month.fillna(pd.to_numeric(parts[2], errors="coerce") * 3 - 2).fillna(1)
    year = pd.to_numeric(parts[0], errors="coerce")
    ok = year.notna() & month.between(1, 12)
    stamp = year.where(ok, 1970).astype(int).astype(str) + "-" \
        + month.where(ok, 1).astype(int).map("{:02d}".format) + "-01"
    return pd.to_datetime(stamp, format="%Y-%m-%d").where(ok)


def _clean_codes(codes: pd.Series) -> tuple[pd.Series, set[str], set[str]]:
    """Renamed codes, plus the aggregates dropped and the unexpected group codes."""
    c = codes.astype(str).str.strip().replace(IMF_CODE_RENAMES)
    agg = c.isin(IMF_AGGREGATES)
    odd = ~agg & ~c.str.fullmatch(r"[A-Z]{3}")
    return c.where(~(agg | odd)), set(c[agg]), set(c[odd])


def _tidy(raw: pd.DataFrame, dims: Sequence[str],
          spec: Mapping[tuple[str, ...], str]) -> tuple[pd.DataFrame, set[str]]:
    """Raw SDMX-CSV -> long ``code, date, variable, value`` for the mapped series."""
    if raw.empty or "COUNTRY" not in raw:
        return pd.DataFrame(columns=["code", "date", "variable", "value"]), set()
    df = raw.dropna(subset=["OBS_VALUE"])
    df = df[df["TIME_PERIOD"].astype(str).str.len() > 0]
    if dims:
        keys = list(zip(*(df[d].astype(str) for d in dims)))
        var = pd.Series([spec.get(k) for k in keys], index=df.index, dtype=object)
    else:
        var = pd.Series(next(iter(spec.values())), index=df.index, dtype=object)
    df = df.assign(variable=var).dropna(subset=["variable"])
    code, agg, odd = _clean_codes(df["COUNTRY"])
    if odd:
        warnings.warn(f"imf: dropped unexpected non-ISO3 codes {sorted(odd)} "
                      "(treated as IMF group codes)", stacklevel=3)
    out = pd.DataFrame({"code": code, "date": _period_start(df["TIME_PERIOD"]),
                        "variable": df["variable"], "value": df["OBS_VALUE"].astype(float)})
    return out.dropna(subset=["code", "date"]), agg | odd


def _wide(long: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if long.empty:
        return _empty_panel(columns)
    dup = long.duplicated(["code", "date", "variable"], keep="last")
    if dup.any():
        warnings.warn(f"imf: {int(dup.sum())} duplicate (code, date, variable) cells; "
                      "kept the last", stacklevel=3)
        long = long[~dup]
    wide = long.pivot(index=["code", "date"], columns="variable", values="value")
    wide.columns.name = None
    wide = wide.reindex(columns=list(columns)).astype("float64").sort_index()
    wide.index = wide.index.set_names(_INDEX_NAMES)
    return wide


def _empty_panel(columns: Sequence[str] = ()) -> pd.DataFrame:
    idx = pd.MultiIndex.from_arrays([pd.Index([], dtype=object), pd.DatetimeIndex([])],
                                    names=_INDEX_NAMES)
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns}, index=idx)


def _unit_mult(units: str) -> int:
    """SDMX power of ten of a stored unit: 6 for millions, 3 for thousands, else 0."""
    u = str(units).lower()
    return 6 if u.startswith("millions") else 3 if u.startswith("thousands") else 0


def _meta_row(panel: pd.DataFrame, column: str, **info) -> dict:
    s = panel[column].dropna() if column in panel else pd.Series(dtype=float)
    dates = s.index.get_level_values("date") if len(s) else pd.DatetimeIndex([])
    info.setdefault("unit_mult", _unit_mult(info.get("units", "")))
    row = dict(variable=column, **info)
    row.update(first=str(dates.min().date()) if len(s) else None,
               last=str(dates.max().date()) if len(s) else None,
               n=int(s.index.get_level_values("code").nunique()) if len(s) else 0,
               n_obs=int(len(s)))
    return row


def _drop_zeros(panel: pd.DataFrame, columns: Iterable[str], label: str) -> dict[str, int]:
    """Set exact zeros to NaN in ``columns`` (placeholders in strictly positive
    series); return the count per column and warn once."""
    dropped: dict[str, int] = {}
    for col in columns:
        if col in panel:
            zero = panel[col] == 0
            if zero.any():
                dropped[col] = int(zero.sum())
                panel.loc[zero, col] = np.nan
    if dropped:
        worst = sorted(dropped.items(), key=lambda kv: -kv[1])[:6]
        warnings.warn(f"{label}: {sum(dropped.values())} exact zeros treated as missing "
                      f"(provider placeholders), e.g. {dict(worst)}; see attrs['zeros_dropped']",
                      stacklevel=3)
    return dropped


def _finish(panel: pd.DataFrame, *, meta: list[dict], missing: list[dict],
            codes: list[str] | None, **extra) -> pd.DataFrame:
    """Attach attrs, report requested countries with no data, warn on gaps."""
    if codes is not None:
        got = set(panel.index.get_level_values("code")) if len(panel) else set()
        missing += [{"code": c, "reason": "no observations returned"} for c in codes if c not in got]
    for row in meta:
        if row["n_obs"] == 0 and not any(m.get("variable") == row["variable"] for m in missing):
            missing.append({"variable": row["variable"], "reason": "no observations returned"})
    if missing:
        warnings.warn("imf: not obtained: " + "; ".join(
            f"{m.get('variable') or m.get('code') or m.get('item')} ({m['reason']})" for m in missing[:12])
            + (f"; ... {len(missing) - 12} more" if len(missing) > 12 else ""), stacklevel=3)
    panel.attrs.update(meta=tuple(meta), source=_SOURCE, fetched_at=_now(),
                       missing=tuple(missing), **extra)
    return panel


# ---------------------------------------------------------------------------
# public: raw access
# ---------------------------------------------------------------------------

def imf_get(flow: str, key: str = "all", *, start: str | int | None = None,
            end: str | int | None = None, detail: str = "dataonly",
            refresh: bool = False, timeout: float = _TIMEOUT, retries: int = 3,
            pause: float = _PAUSE) -> pd.DataFrame:
    """Raw SDMX-CSV from ``api.imf.org`` as a DataFrame.

    Parameters
    ----------
    flow : str
        Agency-qualified flow id (``"IMF.STA,CPI"``) or a short name from
        :data:`IMF_FLOWS` (``"CPI"``).
    key : str, default ``"all"``
        SDMX key: dimension values in the flow's order, ``+`` for OR, blank
        for a wildcard (``".CPI._T.IX.M"`` = all-items CPI, every country).
        A key made only of dots is sent as ``all`` (a ``..`` path segment is
        refused by the IMF gateway).
    start, end : str or int, optional
        ``startPeriod`` / ``endPeriod`` (``1990``, ``1990-01``, ``1990-Q1``).
    detail : {"dataonly", "full", "nodata", "serieskeysonly"}
        ``dataonly`` drops the metadata attributes, which repeat on every
        row and make the body twenty times larger; ``full`` keeps them
        (``UPDATE_DATE``, ``SCALE``, ``SERIES_NAME`` ...).
    refresh : bool
        Bypass the on-disk cache and store the fresh response.
    timeout, retries, pause : float, int, float
        Per-request timeout in seconds; retries after a 5xx, 429 or timeout;
        minimum seconds between requests to the host.

    Returns
    -------
    pandas.DataFrame
        The CSV as published: ``DATAFLOW``, the dimension columns (strings),
        ``TIME_PERIOD`` (string) and ``OBS_VALUE`` (float), plus the
        attribute columns unless ``detail="dataonly"``. ``attrs``: ``url``,
        ``fetched_at`` and ``status`` (``"ok"``, ``"HTTP 404"`` for no
        matching series, ``"HTTP 500 after 4 attempts"``, ``"timeout"`` ...).
        On failure the frame is empty and a warning is issued; it never
        raises for a provider failure.

    Examples
    --------
    >>> raw = imf_get("CPI", "USA.CPI._T.IX.M", start=2024)       # doctest: +SKIP
    >>> raw[["COUNTRY", "TIME_PERIOD", "OBS_VALUE"]].head()        # doctest: +SKIP
    """
    if not isinstance(flow, str) or not flow.strip():
        raise ValueError("flow must be a non-empty string such as 'IMF.STA,CPI'")
    flow = IMF_FLOWS.get(flow.strip().upper(), flow.strip())
    if "," not in flow:
        raise ValueError(f"flow {flow!r} is not agency-qualified; use 'AGENCY,FLOW' "
                         f"(see imf_dataflows()) or one of {sorted(IMF_FLOWS)}")
    detail = str(detail).lower()
    if detail not in ("dataonly", "full", "nodata", "serieskeysonly"):
        raise ValueError(f"detail={detail!r}; choose one of "
                         "['dataonly', 'full', 'nodata', 'serieskeysonly']")
    key = (key or "all").strip()
    if set(key) <= {"."}:
        key = "all"
    if "/" in key:
        raise ValueError(f"key {key!r} contains '/'")
    params = []
    if detail != "full":
        params.append(f"detail={detail}")
    s, e = _check_period(start, "start"), _check_period(end, "end")
    if s:
        params.append(f"startPeriod={s}")
    if e:
        params.append(f"endPeriod={e}")
    url = f"{_BASE}/data/{flow}/{key}" + ("?" + "&".join(params) if params else "")
    body, status = _fetch(url, timeout=timeout, pause=pause, retries=retries, refresh=refresh)
    if body is None:
        if status != "HTTP 404":
            warnings.warn(f"imf_get: {flow}/{key} failed ({status})", stacklevel=2)
        return _empty_raw(url, status)
    if not body.lstrip(b"\xef\xbb\xbf").startswith(b"DATAFLOW"):
        warnings.warn(f"imf_get: {flow}/{key} answered something other than SDMX-CSV "
                      f"({body[:60]!r})", stacklevel=2)
        return _empty_raw(url, "unexpected body")
    try:
        df = _read_csv(body, detail)
    except (ValueError, pd.errors.ParserError) as exc:
        warnings.warn(f"imf_get: could not parse {flow}/{key}: {exc}", stacklevel=2)
        return _empty_raw(url, "unparseable CSV")
    df.attrs.update(url=url, fetched_at=_now(), status="ok")
    return df


def imf_dataflows(*, refresh: bool = False) -> pd.DataFrame:
    """Every data flow the IMF SDMX 2.1 API lists (about 220, vintages included).

    Returns
    -------
    pandas.DataFrame
        Columns ``flow`` (the agency-qualified id :func:`imf_get` takes),
        ``agency``, ``id``, ``version``, ``name`` and ``vintage`` (True for
        the frozen ``*_VINTAGE`` snapshots). ``attrs``: ``source``,
        ``fetched_at``, ``missing``. Empty, with a warning, if the request
        fails.

    Examples
    --------
    >>> f = imf_dataflows()                                         # doctest: +SKIP
    >>> f[f["name"].str.contains("Labor")]                          # doctest: +SKIP
    """
    cols = ["flow", "agency", "id", "version", "name", "vintage"]
    url = f"{_BASE}/dataflow"
    body, status = _fetch(url, timeout=_TIMEOUT, pause=_PAUSE, retries=2,
                          refresh=refresh, accept=None)
    rows: list[dict] = []
    if body is not None:
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            body, status = None, f"unparseable XML: {exc}"
        else:
            for el in root.iter():
                if not el.tag.endswith("}Dataflow"):
                    continue
                name = next((n.text for n in el if n.tag.endswith("}Name")
                             and n.get("{http://www.w3.org/XML/1998/namespace}lang", "en") == "en"), None)
                fid, agency = el.get("id", ""), el.get("agencyID", "")
                rows.append(dict(flow=f"{agency},{fid}", agency=agency, id=fid,
                                 version=el.get("version"), name=name,
                                 vintage="VINTAGE" in fid.upper()))
    out = pd.DataFrame(rows, columns=cols)
    missing: tuple = ()
    if body is None:
        warnings.warn(f"imf_dataflows: request failed ({status})", stacklevel=2)
        missing = ({"item": "dataflows", "reason": status},)
    elif not rows:
        # A gateway or HTML error page parses as XML too: it is not "no flows".
        warnings.warn("imf_dataflows: the response holds no Dataflow elements", stacklevel=2)
        missing = ({"item": "dataflows", "reason": "no Dataflow elements in response"},)
    out = out.sort_values(["agency", "id"]).reset_index(drop=True)
    out.attrs.update(source=_SOURCE, fetched_at=_now(), missing=missing)
    return out


# ---------------------------------------------------------------------------
# block reader shared by the panel builders
# ---------------------------------------------------------------------------

def _block(flow: str, country: str, spec: Mapping[tuple[str, ...], str], freq: str, *,
           start: str | None, refresh: bool, key: str | None = None
           ) -> tuple[pd.DataFrame, str, str, set[str]]:
    """One request for ``spec`` -> (long frame, key, status, aggregates dropped)."""
    key = key or _key(country, list(spec), freq)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # failures are reported once, by _finish
        raw = imf_get(flow, key, start=start, refresh=refresh)
    status = raw.attrs.get("status", "ok")
    if status == "HTTP 404":
        status = "no matching series (HTTP 404)"
    long, agg = _tidy(raw, _FLOW_DIMS[flow], spec)
    return long, key, status, agg


# ---------------------------------------------------------------------------
# public: panels
# ---------------------------------------------------------------------------

def imf_nea_panel(codes: Iterable[str] | str | None = None, *, freq: str = "A",
                  sa: str = "SA", real: bool = True, start: str | int | None = None,
                  refresh: bool = False) -> pd.DataFrame:
    """Expenditure-side national accounts from the IMF NEA flows.

    One request (``IMF.STA,ANEA`` for ``freq="A"``, ``IMF.STA,QNEA`` for
    ``"Q"``) returns current prices, constant prices and deflators for
    every economy: 188 with annual GDP (11 from before 1960, 71 from before
    1995; 112 with the full GDP, C, G, I, X, M core), 66 with quarterly SA
    GDP and about 110 unadjusted (counts of 7 Oct 2026, aggregates
    excluded).

    Parameters
    ----------
    codes : iterable of str, optional
        ISO3 codes; ``None`` (default) for every economy the IMF publishes.
    freq : {"A", "Q"}
        Annual (``ANEA``) or quarterly (``QNEA``).
    sa : {"SA", "NSA"}
        Seasonal adjustment of the quarterly flow; ignored for annual data.
    real : bool, default True
        Add ``<name>_real`` (constant prices, IMF price type ``Q``) and the
        implicit deflator ``<name>_defl = 100 * <name> / <name>_real``,
        whose base is the country's constant-price reference year (not
        always inside the published span, so rebase before comparing
        levels across countries).
    start : str or int, optional
        First period (``1990``, ``1990-Q1``); default is the whole history.
    refresh : bool
        Re-fetch instead of reading the on-disk cache.

    Returns
    -------
    pandas.DataFrame
        Index ``(code, date)`` with period-start dates; columns ``gdp, cons,
        cons_gov, cons_hh, cons_priv, inv, capform, inventories, exports,
        imports`` in millions of national currency (quarterly values are
        quarterly flows, not annualised), then ``<name>_real`` for the nine
        published items and ``<name>_defl`` for all but ``inventories``
        (a deflator of a change in stocks is meaningless). ``cons_priv`` is
        ``cons - cons_gov`` (households plus NPISH; the broad measure, 113
        economies annually). ``cons_hh`` is households only and sparse. The
        IMF publishes no income side and no durability or asset split here.
        Exact zeros, which the IMF publishes as placeholders (Burkina Faso
        1952-1998), are set to missing in every column but ``inventories``
        and counted in ``attrs["zeros_dropped"]``. ``attrs``: ``meta`` (one
        dict per column), ``source``, ``fetched_at``, ``missing``, ``freq``,
        ``sa``, ``aggregates_dropped``, ``zeros_dropped``.

    Examples
    --------
    >>> p = imf_nea_panel(["USA", "MEX", "KOR"], freq="Q")          # doctest: +SKIP
    >>> (p["gdp"] / p["gdp_defl"] * 100).unstack("code").tail()    # doctest: +SKIP
    """
    freq = _check_choice(freq, "freq", ("A", "Q"))
    sa = _check_choice(sa, "sa", ("SA", "NSA"))
    codes = _check_codes(codes)
    start = _check_period(start, "start")
    flow = "IMF.STA,ANEA" if freq == "A" else "IMF.STA,QNEA"
    prices = [("V", "XDC", "", 1e-6)]
    if real:
        prices += [("Q", "XDC", "_real", 1e-6)]
    spec: dict[tuple[str, ...], str] = {}
    for name, (ind, _) in IMF_NEA_ITEMS.items():
        for pt, tr, suffix, _mult in prices:
            spec[(ind, pt, tr) if freq == "A" else (ind, pt, sa, tr)] = name + suffix
    columns = list(IMF_NEA_ITEMS)
    columns.insert(columns.index("cons_hh") + 1, "cons_priv")
    if real:
        columns += [n + "_real" for n in IMF_NEA_ITEMS] \
            + [n + "_defl" for n in IMF_NEA_ITEMS if n != "inventories"]
    long, key, status, agg = _block(flow, _country_part(codes), spec, freq,
                                    start=start, refresh=refresh)
    mult = {name + suffix: m for name in IMF_NEA_ITEMS for _, _, suffix, m in prices}
    if not long.empty:
        long = long.assign(value=long["value"] * long["variable"].map(mult))
    panel = _wide(long, columns)
    positive = [c for c in columns if c in panel and not c.startswith("inventories")
                and not c.endswith("_defl") and c != "cons_priv"]
    zeros = _drop_zeros(panel, positive, "imf_nea_panel") if len(panel) else {}
    if len(panel):
        panel["cons_priv"] = panel["cons"] - panel["cons_gov"]
        for col in columns:
            if col.endswith("_defl"):
                base = col.removesuffix("_defl")
                real_ = panel[base + "_real"].where(panel[base + "_real"] != 0)
                panel[col] = 100.0 * panel[base] / real_
    missing: list[dict] = []
    if status != "ok":
        missing.append({"item": f"{flow}/{key}", "reason": status})
    sa_label = sa if freq == "Q" else "NSA (annual)"
    meta = []
    for col in columns:
        base = col.removesuffix("_real").removesuffix("_defl")
        if col == "cons_priv":
            info = dict(source_flow=flow, source_key="P3 - P3_S13 (derived)", price_type="V",
                        description="Private final consumption, households plus NPISH (cons - cons_gov)",
                        units="millions of national currency", scale=1e-6)
        else:
            ind, desc = IMF_NEA_ITEMS[base]
            if col.endswith("_defl"):
                info = dict(source_flow=flow, source_key=f"{ind} V / {ind} Q (derived)",
                            price_type="V/Q", description=f"Implicit deflator: {desc}",
                            units="index, base = the country's constant-price reference year",
                            scale=1.0)
            else:
                pt = "Q" if col.endswith("_real") else "V"
                info = dict(source_flow=flow, source_key=ind, price_type=pt, description=desc,
                            units=("millions of national currency, constant prices" if pt == "Q"
                                   else "millions of national currency"),
                            scale=1e-6)
        meta.append(_meta_row(panel, col, sa=sa_label, **info))
    return _finish(panel, meta=meta, missing=missing, codes=codes, freq=freq, sa=sa_label,
                   aggregates_dropped=tuple(sorted(agg)), zeros_dropped=zeros)


def _multi_block_panel(registry: Mapping[str, tuple], columns: Sequence[str], freq: str,
                       codes: list[str] | None, start: str | None, refresh: bool,
                       *, clip_future: bool) -> tuple[pd.DataFrame, list[dict], dict, set[str], int]:
    """One request per flow in ``registry``.

    Returns the wide panel, the ``missing`` entries, the key sent per flow,
    the aggregate codes dropped and the number of future-dated rows dropped.
    """
    by_flow: dict[str, dict[tuple[str, ...], str]] = {}
    for col in columns:
        flow, dims = registry[col][0], registry[col][1]
        by_flow.setdefault(flow, {})[dims] = col
    longs, missing, keys, aggs = [], [], {}, set()
    country = _country_part(codes)
    failed_in_a_row = 0
    for flow, spec in by_flow.items():
        if failed_in_a_row >= _MAX_FAILED_FLOWS:
            # The API is down or throttling: do not spend another ~12 minutes per flow.
            keys[flow] = _key(country, list(spec), freq)
            missing += [{"variable": c, "flow": flow,
                         "reason": f"skipped after {failed_in_a_row} consecutive flows failed"}
                        for c in spec.values()]
            continue
        long, key, status, agg = _block(flow, country, spec, freq, start=start, refresh=refresh)
        keys[flow] = key
        aggs |= agg
        if status != "ok":
            missing += [{"variable": c, "flow": flow, "reason": status} for c in spec.values()]
        transport = status != "ok" and not status.startswith("no matching series")
        failed_in_a_row = failed_in_a_row + 1 if transport else 0
        longs.append(long)
    long = pd.concat([x for x in longs if len(x)], ignore_index=True) if any(len(x) for x in longs) \
        else pd.DataFrame(columns=["code", "date", "variable", "value"])
    n_future = 0
    if clip_future and len(long):
        today = pd.Timestamp(dt.date.today())
        future = long["date"] > today
        n_future = int(future.sum())
        long = long[~future]
    if len(long):
        mult = {c: registry[c][4] for c in columns}
        long = long.assign(value=long["value"] * long["variable"].map(mult))
    return _wide(long, columns), missing, keys, aggs, n_future


#: Monthly columns that cannot be zero (levels, indices, prices): a zero there
#: is a placeholder. Rates are left alone (policy rates sat at 0 for years).
_MONTHLY_POSITIVE = ("cpi", "cpi_food", "cpi_housing", "cpi_transport", "ip", "ip_sa",
                     "ip_mfg", "emp", "lf", "neer", "reer", "fx_usd", "fx_usd_eop")


def imf_monthly_panel(codes: Iterable[str] | str | None = None, *,
                      variables: Iterable[str] | None = None,
                      start: str | int | None = None,
                      refresh: bool = False) -> pd.DataFrame:
    """Monthly prices, activity, labour, interest and exchange rates from the IMF.

    One request per flow (``CPI``, ``PI``, ``LS``, ``MFS_IR``, ``EER``,
    ``ER``; six for the default variables), joined on ``(code, date)``.
    Coverage on 7 Oct 2026, aggregates excluded: all-items CPI 191
    economies (from 1914 for Canada, 23 from before 1960), the policy rate
    82, money-market rate 89, Treasury-bill yield 86, government bond yield
    67, deposit rate 140, lending rate 135, NEER 99 and REER 93 (from 1979),
    the dollar exchange rate 222 (from 1940), industrial production 66 (48
    SA), unemployment rate 87. An all-economy pull is about 100 MB of CSV
    and took 53 s uncached.

    Parameters
    ----------
    codes : iterable of str, optional
        ISO3 codes; ``None`` (default) for every economy.
    variables : iterable of str, optional
        Subset of :data:`IMF_MONTHLY` (default: all of them).
    start : str or int, optional
        First period (``1990`` or ``1990-01``); default is the whole history.
    refresh : bool
        Re-fetch instead of reading the on-disk cache.

    Returns
    -------
    pandas.DataFrame
        Index ``(code, date)`` with month-start dates; columns ``cpi,
        cpi_food, cpi_housing, cpi_transport`` (indices, national reference
        periods), ``ip, ip_sa, ip_mfg`` (2010 = 100; the IMF publishes no
        seasonally adjusted manufacturing index), ``urate``
        (percent), ``emp, lf`` (thousands), ``policy_rate, mm_rate,
        tbill_rate, bond_yield, discount_rate, deposit_rate, lending_rate``
        (percent per annum), ``neer, reer`` (2010 = 100, up = appreciation),
        ``fx_usd, fx_usd_eop`` (national currency per US dollar). Rows dated
        after today (the ER flow carries some) are dropped and counted in
        ``attrs["future_rows_dropped"]``. Zeros in the levels, indices and
        prices are placeholders and become missing (``attrs["zeros_dropped"]``);
        zero interest rates are kept. If two flows in a row fail on transport
        (5xx, 429, timeout after all retries) the remaining flows are not
        requested and are listed in ``attrs["missing"]`` as skipped, which
        bounds a dead-API call to about 25 minutes. ``attrs``: ``meta``,
        ``source``, ``fetched_at``, ``missing``, ``aggregates_dropped``,
        ``future_rows_dropped``, ``zeros_dropped``, ``keys``.

    Examples
    --------
    >>> m = imf_monthly_panel(["USA", "MEX"], start=2000)           # doctest: +SKIP
    >>> m[["cpi", "policy_rate", "fx_usd"]].loc["MEX"].tail()        # doctest: +SKIP
    """
    codes = _check_codes(codes)
    start = _check_period(start, "start")
    columns = _check_variables(variables, IMF_MONTHLY)
    panel, missing, keys, agg, n_future = _multi_block_panel(IMF_MONTHLY, columns, "M", codes,
                                                             start, refresh, clip_future=True)
    zeros = _drop_zeros(panel, [c for c in columns if c in _MONTHLY_POSITIVE],
                        "imf_monthly_panel") if len(panel) else {}
    meta = []
    for col in columns:
        flow, dims, desc, units, mult, sa = IMF_MONTHLY[col]
        meta.append(_meta_row(panel, col, source_flow=flow, source_key=".".join(dims),
                              description=desc, units=units, scale=mult, sa=sa))
    return _finish(panel, meta=meta, missing=missing, codes=codes,
                   aggregates_dropped=tuple(sorted(agg)), future_rows_dropped=n_future,
                   zeros_dropped=zeros, keys=keys)


def _check_variables(variables: Iterable[str] | str | None, registry: Mapping) -> list[str]:
    if variables is None:
        return list(registry)
    if isinstance(variables, str):
        variables = [variables]
    variables = list(dict.fromkeys(variables))
    bad = [v for v in variables if v not in registry]
    if bad:
        raise ValueError(f"unknown variable(s) {bad}; choose from {list(registry)}")
    if not variables:
        raise ValueError(f"variables is empty; choose from {list(registry)}")
    return variables


def imf_labour_panel(codes: Iterable[str] | str | None = None, *, freq: str = "A",
                     start: str | int | None = None, refresh: bool = False) -> pd.DataFrame:
    """Unemployment rate, employment, labour force and unemployed from IMF ``LS``.

    One request. Annual coverage is the widest anywhere for these levels:
    employment for about 190 economies (54 from before 1980), labour force
    189 (117 from before 1970), unemployment rate 187. Quarterly: about
    120; monthly: 87 for the rate, 69 for employment. Not seasonally
    adjusted; no hours, self-employment or vacancies (see the ILO fetcher).

    Parameters
    ----------
    codes : iterable of str, optional
        ISO3 codes; ``None`` (default) for every economy.
    freq : {"A", "Q", "M"}
    start : str or int, optional
    refresh : bool

    Returns
    -------
    pandas.DataFrame
        Index ``(code, date)`` with period-start dates; columns ``urate``
        (percent of labour force), ``emp``, ``lf``, ``unemp`` (thousands of
        persons). Zeros in ``emp`` and ``lf`` would be placeholders and
        become missing; a zero unemployment rate is kept (Switzerland in the
        1970s). ``attrs``: ``meta``, ``source``, ``fetched_at``,
        ``missing``, ``freq``, ``aggregates_dropped``, ``zeros_dropped``, ``keys``.

    Examples
    --------
    >>> imf_labour_panel(["USA", "MEX"]).loc["MEX"].tail()          # doctest: +SKIP
    """
    freq = _check_choice(freq, "freq", ("A", "Q", "M"))
    codes = _check_codes(codes)
    start = _check_period(start, "start")
    reg = {c: ("IMF.STA,LS", (ind, tr), desc, units, mult)
           for c, (ind, tr, desc, units, mult) in IMF_LABOUR.items()}
    columns = list(IMF_LABOUR)
    panel, missing, keys, agg, _ = _multi_block_panel(reg, columns, freq, codes, start, refresh,
                                                      clip_future=True)
    zeros = _drop_zeros(panel, ["emp", "lf"], "imf_labour_panel") if len(panel) else {}
    meta = [_meta_row(panel, c, source_flow="IMF.STA,LS", source_key=".".join(reg[c][1]),
                      description=reg[c][2], units=reg[c][3], scale=reg[c][4], sa="NSA")
            for c in columns]
    return _finish(panel, meta=meta, missing=missing, codes=codes, freq=freq,
                   aggregates_dropped=tuple(sorted(agg)), zeros_dropped=zeros, keys=keys)


def imf_pcps(*, freq: str = "M", start: str | int | None = None,
             refresh: bool = False) -> pd.DataFrame:
    """IMF primary commodity prices: about 110 indices and benchmark prices.

    One request, ``IMF.RES,PCPS/G001..INDEX+USD.{freq}``. Monthly, quarterly
    and annual series all start in 1992 (earlier history: the World Bank
    Pink Sheet, :mod:`puremacro.fetch.wb_pink_sheet`).

    Parameters
    ----------
    freq : {"M", "Q", "A"}
    start : str or int, optional
    refresh : bool

    Returns
    -------
    pandas.DataFrame
        Index ``(code, date)`` with ``code == "WLD"`` and period-start
        dates. Named columns from :data:`IMF_PCPS_NAMED` first (``pcom_all``,
        ``pcom_energy``, ``poil``, ``pcopper`` ...), then every other series
        as ``<indicator>_idx`` (2016 = 100) or ``<indicator>_usd`` (US dollar
        unit price), lower case. ``attrs``: ``meta`` (indicator,
        transformation, units for every column), ``source``, ``fetched_at``,
        ``missing``, ``freq``.

    Examples
    --------
    >>> c = imf_pcps()                                              # doctest: +SKIP
    >>> c.loc["WLD", ["pcom_all", "poil"]].tail()                   # doctest: +SKIP
    """
    freq = _check_choice(freq, "freq", ("M", "Q", "A"))
    start = _check_period(start, "start")
    flow = "IMF.RES,PCPS"
    key = f"G001..INDEX+USD.{freq}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = imf_get(flow, key, start=start, refresh=refresh)
    status = raw.attrs.get("status", "ok")
    named = {(ind, tr): col for col, (ind, tr, _, _) in IMF_PCPS_NAMED.items()}
    missing: list[dict] = []
    if raw.empty:
        panel = _empty_panel(list(IMF_PCPS_NAMED))
        missing.append({"item": f"{flow}/{key}", "reason": status if status != "ok" else "no observations"})
        meta = [dict(variable=c, source_flow=flow, source_key=f"{i}.{t}", description=d, units=u,
                     unit_mult=0, scale=1.0, sa="NSA", first=None, last=None, n=0, n_obs=0)
                for c, (i, t, d, u) in IMF_PCPS_NAMED.items()]
        return _finish(panel, meta=meta, missing=missing, codes=None, freq=freq)
    raw = raw.dropna(subset=["OBS_VALUE"])
    raw = raw[raw["TIME_PERIOD"].astype(str).str.len() > 0]
    pairs = list(dict.fromkeys(zip(raw["INDICATOR"].astype(str), raw["DATA_TRANSFORMATION"].astype(str))))
    spec = {}
    for ind, tr in pairs:
        spec[(ind, tr)] = named.get((ind, tr)) or f"{ind.lower()}_{'idx' if tr == 'INDEX' else tr.lower()}"
    long, _ = _tidy(raw.assign(COUNTRY="WLD"), _FLOW_DIMS[flow], spec)
    others = sorted(c for c in set(spec.values()) if c not in IMF_PCPS_NAMED)
    columns = [c for c in IMF_PCPS_NAMED] + others
    panel = _wide(long, columns)
    inv = {v: k for k, v in spec.items()}
    meta = []
    for col in columns:
        if col in IMF_PCPS_NAMED:
            ind, tr, desc, units = IMF_PCPS_NAMED[col]
        else:
            ind, tr = inv[col]
            desc = ind
            units = "index, 2016 = 100" if tr == "INDEX" else "US dollars per unit (see IMF PCPS)"
        meta.append(_meta_row(panel, col, source_flow=flow, source_key=f"{ind}.{tr}",
                              description=desc, units=units, scale=1.0, sa="NSA"))
    return _finish(panel, meta=meta, missing=missing, codes=None, freq=freq)


_FY_RE = re.compile(r"FY\s*(\d{4})\s*/\s*(\d{2,4})")
_FY_MAP_RE = re.compile(r"FY\(\s*t\s*-\s*1\s*/\s*t\s*\)\s*=\s*CY\(\s*t\s*\)", re.I)
_FY_END_MONTHS = ("january", "february", "march")


def _latest_actual_year(value: str, notes: str = "", months: str = "") -> float:
    """WEO ``LATEST_ACTUAL_ANNUAL_DATA`` -> the WEO calendar-year label.

    ``"2024"`` -> 2024. A fiscal year ``"FY2024/25"`` is mapped the way the
    series' own ``METHODOLOGY_NOTES`` say: ``FY(t-1/t) = CY(t)`` (Egypt,
    Pakistan, Bangladesh, Haiti: years ending June-September) gives 2025,
    ``FY(t/t+1) = CY(t)`` (India, Iran: years ending in March) gives 2024.
    Without a note, a year that ends in January-March maps to its first
    calendar year and any other to its second. ``"Not applicable"`` and
    blanks give NaN.
    """
    v = str(value or "")
    m = _FY_RE.search(v)
    if m:
        first = int(m.group(1))
        notes = str(notes or "")
        if _FY_MAP_RE.search(notes):
            return float(first + 1)
        if re.search(r"FY\(\s*t\s*/\s*t\s*\+\s*1\s*\)\s*=\s*CY\(\s*t\s*\)", notes, re.I):
            return float(first)
        end = str(months or "").split("/")[-1].strip().lower()
        if end and not end.startswith(_FY_END_MONTHS):
            return float(first + 1)
        return float(first)
    m = re.search(r"(\d{4})", v)
    return float(m.group(1)) if m else float("nan")


#: WEO indicators whose latest-actual year sets a country's row-level
#: ``forecast`` flag, in order of preference (real GDP, then nominal GDP).
_WEO_FLAG_BASIS = ("NGDP_R", "NGDP")


def imf_weo(codes: Iterable[str] | str | None = None, *,
            indicators: Iterable[str] | str | None = None,
            start: str | int | None = None, vintage: str | None = None,
            history_only: bool = False, refresh: bool = False) -> pd.DataFrame:
    """IMF World Economic Outlook, annual from 1980, with a ``forecast`` flag.

    Two requests: the data (``detail=dataonly``) and the series attributes
    without observations (``detail=nodata``), whose
    ``LATEST_ACTUAL_ANNUAL_DATA`` says, per country *and indicator*, where
    the history ends and the staff projections begin. The boundaries differ
    across indicators (Argentina's population is an estimate after 2010, its
    GDP is actual to 2025), so they are applied column by column.

    Parameters
    ----------
    codes : iterable of str, optional
        ISO3 codes; default every economy (about 196).
    indicators : iterable of str, optional
        Column names from :data:`IMF_WEO` (``"gdp_real"``, ``"urate"``) or
        raw WEO indicator codes in upper case (``"GGR_NGDP"``), which come
        back under their own name, unscaled. Default: all of :data:`IMF_WEO`.
    start : str or int, optional
        First year; default 1980, the start of the WEO.
    vintage : str, optional
        A frozen release such as ``"2025_OCT"`` (flow
        ``IMF.RES,WEO_2025_OCT_VINTAGE``); default the latest release.
    history_only : bool, default False
        Blank (NaN) every cell after its own country-indicator latest-actual
        year, then drop rows left with no values: the frame to freeze as
        history. Cells whose indicator has no latest-actual year use the
        country's ``forecast`` boundary.
    refresh : bool

    Returns
    -------
    pandas.DataFrame
        Index ``(code, date)`` with January-1 dates; one float column per
        indicator plus ``forecast`` (nullable boolean), True when the year
        is after the country's latest actual year for real GDP (``NGDP_R``;
        nominal ``NGDP`` if absent; else the most common boundary among the
        returned indicators). It is a row-level guide to the GDP projection
        horizon; other columns can turn into projections earlier, so use
        ``history_only=True`` (or ``attrs["latest_actual"]``,
        ``{code: {column: year}}``) to cut each column at its own boundary.
        Fiscal-year boundaries (``"FY2024/25"``) are mapped to the WEO's
        calendar-year label as each series' methodology note states
        (India 2024, Pakistan 2025). Countries left without any boundary
        get ``forecast = NA`` and are listed in ``attrs["missing"]``.
        ``attrs``: ``meta``, ``source``, ``fetched_at``, ``missing``,
        ``flow``, ``latest_actual``, ``forecast_basis`` (``{code:
        "NGDP_R" | "NGDP" | "mode"}``), ``history_only``,
        ``aggregates_dropped``.

    Examples
    --------
    >>> w = imf_weo(["USA", "ARG"], indicators=["gdp_real", "urate"])      # doctest: +SKIP
    >>> imf_weo(history_only=True).loc["ARG", "gdp_real"].tail()           # doctest: +SKIP
    """
    codes = _check_codes(codes)
    start = _check_period(start, "start")
    if indicators is None:
        cols = list(IMF_WEO)
    else:
        if isinstance(indicators, str):
            indicators = [indicators]
        cols = list(dict.fromkeys(indicators))
        bad = [c for c in cols if c not in IMF_WEO and not re.fullmatch(r"[A-Z][A-Z0-9_]*", str(c))]
        if bad or not cols:
            raise ValueError(f"unknown WEO indicator(s) {bad}; choose from {list(IMF_WEO)} "
                             "or pass raw upper-case WEO codes such as 'GGR_NGDP'")
    flow = "IMF.RES,WEO"
    if vintage is not None:
        v = str(vintage).upper()
        if not re.fullmatch(r"\d{4}_(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)", v):
            raise ValueError(f"vintage={vintage!r}; use 'YYYY_MON' such as '2025_OCT'")
        flow = f"IMF.RES,WEO_{v}_VINTAGE"
    reg = {c: IMF_WEO.get(c, (c, c, "as published", 1.0)) for c in cols}
    spec = {(reg[c][0],): c for c in cols}
    country = _country_part(codes)
    key = _key(country, list(spec), "A")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = imf_get(flow, key, start=start, refresh=refresh)
    status = raw.attrs.get("status", "ok")
    long, agg = _tidy(raw, ("INDICATOR",), spec)
    if len(long):
        mult = {c: reg[c][3] for c in cols}
        long = long.assign(value=long["value"] * long["variable"].map(mult))
    panel = _wide(long, cols)
    missing: list[dict] = []
    if status != "ok":
        missing += [{"variable": c, "flow": flow, "reason": status} for c in cols]

    # --- latest actual year per country and indicator (second request) -------
    latest: dict[str, dict[str, int]] = {}
    basis_years: dict[str, dict[str, int]] = {}
    if len(panel):
        # The GDP indicators are always asked for: they set the row-level flag.
        extra = [(b,) for b in _WEO_FLAG_BASIS if (b,) not in spec]
        attr_key = _key(country, list(spec) + extra, "A")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            attrs_raw = imf_get(flow, attr_key, detail="nodata", refresh=refresh)
        if attrs_raw.attrs.get("status") == "ok" and "LATEST_ACTUAL_ANNUAL_DATA" in attrs_raw:
            a = attrs_raw.reindex(columns=["COUNTRY", "INDICATOR", "LATEST_ACTUAL_ANNUAL_DATA",
                                           "METHODOLOGY_NOTES",
                                           "START_END_MONTHS_OF_REPORTING_YEAR"]).fillna("")
            a = a[a["LATEST_ACTUAL_ANNUAL_DATA"].astype(str).str.strip() != ""]
            years = [_latest_actual_year(v, n, m) for v, n, m in zip(
                a["LATEST_ACTUAL_ANNUAL_DATA"], a["METHODOLOGY_NOTES"],
                a["START_END_MONTHS_OF_REPORTING_YEAR"])]
            a = a.assign(year=years, COUNTRY=a["COUNTRY"].astype(str).replace(IMF_CODE_RENAMES))
            a = a.dropna(subset=["year"])
            back = {reg[c][0]: c for c in cols}
            for row in a.itertuples(index=False):
                col = back.get(row.INDICATOR)
                if col is not None:
                    latest.setdefault(row.COUNTRY, {})[col] = int(row.year)
                if row.INDICATOR in _WEO_FLAG_BASIS:
                    basis_years.setdefault(row.COUNTRY, {})[row.INDICATOR] = int(row.year)
        else:
            missing.append({"item": "forecast flag (LATEST_ACTUAL_ANNUAL_DATA)",
                            "reason": attrs_raw.attrs.get("status", "attribute absent")})

    # --- row-level flag: real GDP's boundary, then nominal GDP's, then the mode
    bound: dict[str, int] = {}
    basis: dict[str, str] = {}
    for code in set(latest) | set(basis_years):
        by = basis_years.get(code, {})
        pick = next((b for b in _WEO_FLAG_BASIS if b in by), None)
        if pick is not None:
            bound[code], basis[code] = by[pick], pick
        elif latest.get(code):
            ys = pd.Series(list(latest[code].values()))
            bound[code], basis[code] = int(ys.mode().min()), "mode"
    flag = pd.Series(pd.NA, index=panel.index, dtype="boolean")
    if len(panel):
        codes_l = panel.index.get_level_values("code")
        yrs = np.asarray(panel.index.get_level_values("date").year, dtype=float)
        b = pd.Series(bound, dtype=float).reindex(codes_l).to_numpy()
        vals = pd.array(np.where(np.isnan(b), None, yrs > b), dtype="boolean")
        flag = pd.Series(vals, index=panel.index, dtype="boolean")
        if latest or basis_years:
            unflagged = sorted(set(codes_l) - set(bound))
            missing += [{"code": c, "item": "forecast flag",
                         "reason": "no LATEST_ACTUAL_ANNUAL_DATA; forecast is NA"}
                        for c in unflagged]
        if history_only:
            for col in cols:
                cut = pd.Series({c: v[col] for c, v in latest.items() if col in v}, dtype=float)
                cut = cut.reindex(codes_l).to_numpy()
                cut = np.where(np.isnan(cut), b, cut)
                # cells past their own boundary, and cells with no boundary at all
                drop = np.isnan(cut) | (yrs > cut)
                panel.loc[drop, col] = np.nan
            keep = panel[cols].notna().any(axis=1).to_numpy()
            panel, flag = panel[keep], flag[keep]
    panel["forecast"] = flag
    meta = []
    for c in cols:
        ind, desc, units, mult = reg[c]
        meta.append(_meta_row(panel, c, source_flow=flow, source_key=ind, description=desc,
                              units=units, scale=mult, sa="NSA"))
    return _finish(panel, meta=meta, missing=missing, codes=codes, flow=flow,
                   latest_actual=latest, forecast_basis=basis, history_only=bool(history_only),
                   aggregates_dropped=tuple(sorted(agg)))


def imf_icsd(codes: Iterable[str] | str | None = None, *, start: str | int | None = None,
             refresh: bool = False) -> pd.DataFrame:
    """Public and private investment and capital stocks, IMF ``ICSD`` (1960-2019).

    The Fiscal Affairs Department's Investment and Capital Stock Dataset:
    general-government and private GFCF and the perpetual-inventory capital
    stocks built from them, for about 175 economies (19 from before 1970,
    139 from before 1980), plus the public-private-partnership capital stock
    (140 economies, from 1985; the matching GFCF series is listed in the
    codelist but carries no data). One request.

    ICSD publishes levels in **billions**; they are multiplied by 1000 here
    so that they are in millions, like every other national-currency level
    in the package (``scale = 1000`` and ``unit_mult = 6`` in ``attrs["meta"]``).

    Parameters
    ----------
    codes : iterable of str, optional
    start : str or int, optional
        First year; default the whole history (1960).
    refresh : bool

    Returns
    -------
    pandas.DataFrame
        Index ``(code, date)`` with January-1 dates; columns ``gdp, inv_gov,
        inv_priv, k_gov, k_priv, k_pubpriv`` (millions of
        national currency, current prices), ``gdp_ppp, inv_gov_ppp,
        inv_priv_ppp, k_gov_ppp, k_priv_ppp`` (millions of constant 2017 PPP
        international dollars) and ``inv_gov_gdp, inv_priv_gdp, k_gov_gdp,
        k_priv_gdp`` (percent of GDP, constant prices). ICSD fills the
        capital stocks of economies it does not estimate (Aruba, Macao,
        Puerto Rico, Somalia ...) with zeros; zeros are set to missing in
        every column but ``k_pubpriv`` (where 0 means no PPP capital) and
        counted in ``attrs["zeros_dropped"]``. ``attrs``: ``meta``,
        ``source``, ``fetched_at``, ``missing``, ``aggregates_dropped``,
        ``zeros_dropped``.

    Examples
    --------
    >>> k = imf_icsd(["USA", "MEX"])                                # doctest: +SKIP
    >>> (k["k_gov"] / k["gdp"]).unstack("code").tail()              # doctest: +SKIP
    """
    codes = _check_codes(codes)
    start = _check_period(start, "start")
    reg = {c: ("IMF.FAD,ICSD", (ind,), desc, units, mult)
           for c, (ind, desc, units, mult) in IMF_ICSD.items()}
    columns = list(IMF_ICSD)
    panel, missing, keys, agg, _ = _multi_block_panel(reg, columns, "A", codes, start, refresh,
                                                      clip_future=False)
    zeros = _drop_zeros(panel, [c for c in columns if c != "k_pubpriv"],
                        "imf_icsd") if len(panel) else {}
    meta = [_meta_row(panel, c, source_flow="IMF.FAD,ICSD", source_key=reg[c][1][0],
                      description=reg[c][2], units=reg[c][3], scale=reg[c][4], sa="NSA")
            for c in columns]
    return _finish(panel, meta=meta, missing=missing, codes=codes,
                   aggregates_dropped=tuple(sorted(agg)), zeros_dropped=zeros)


def imf_meta(panel: pd.DataFrame) -> pd.DataFrame:
    """``attrs["meta"]`` of any panel from this module as a DataFrame indexed by variable."""
    meta = panel.attrs.get("meta", ())
    out = pd.DataFrame(list(meta))
    return out.set_index("variable") if "variable" in out else out


__all__ = [
    "IMF_FLOWS", "IMF_AGGREGATES", "IMF_CODE_RENAMES",
    "IMF_NEA_ITEMS", "IMF_MONTHLY", "IMF_LABOUR", "IMF_PCPS_NAMED", "IMF_WEO", "IMF_ICSD",
    "imf_get", "imf_dataflows",
    "imf_nea_panel", "imf_monthly_panel", "imf_labour_panel",
    "imf_pcps", "imf_weo", "imf_icsd", "imf_meta",
]
