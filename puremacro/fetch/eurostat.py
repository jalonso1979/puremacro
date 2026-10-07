"""Eurostat panels: annual and quarterly national accounts, monthly indicators.

``eurostat_na_panel()`` returns the national accounts of every European
reporter in one frame indexed by ``(code, date)``: the expenditure side in
current prices (millions of national currency) with chain-linked volumes and
deflators, and on request the income side, consumption by durability,
investment by asset and by institutional sector, value added by industry,
employment, hours, population and the net capital stock. ``freq="A"`` reaches
back to 1975 for France and Norway (1949 for French sector accounts) and to
1995 for most of the EU; ``freq="Q"`` to 1978Q1. ``eurostat_monthly_panel()``
does the same for the monthly cycle: industrial production, unemployment,
HICP inflation and its core, money-market rates, the ten-year yield,
effective exchange rates and retail volumes.

Why a dedicated fetcher
-----------------------
The OECD quarterly accounts (:func:`puremacro.fetch.qna_panel`) start in 1995
for most of Europe and carry no capital stock, no sector accounts and no
Balkan or Eastern-partnership reporters; Eurostat is the primary source for
all of them and publishes every series under one API. That API has three
traps that this module absorbs so that a notebook does not have to:

* a large request does not fail, it is *queued*: HTTP 200 with an XML
  envelope holding a job id, which ``pandas.read_csv`` silently parses into a
  one-cell frame. :func:`eurostat_get` polls the job and downloads the result;
* ``compressed=true`` returns a gzip *file* with no ``Content-Encoding``
  header, so the transport layer leaves it zipped; it is unzipped here by its
  magic bytes (10-14 times less traffic than plain CSV);
* one code outside a flow's content constraint fails the whole request with a
  400 that names the code. The panel builders read that message, drop the
  named codes and ask again, recording what was dropped in ``attrs["missing"]``.

Every request leaves the country dimension open and filters afterwards, so a
three-country call and an all-country call hit the same cached response. Geo
codes are Eurostat's ISO-2 (``EL`` Greece, ``UK`` United Kingdom, ``XK``
Kosovo), mapped to ISO-3; euro-area, EU and EEA aggregates are dropped by an
explicit list.

Choices made per country and variable, all recorded in ``attrs["meta"]``:
seasonal adjustment ``SCA > SA > NSA`` (``sa="prefer"``), the volume
reference year (2020 chain-linked volumes, 2015 where 2020 is not published,
as for the United Kingdom), and the deflator (Eurostat's published
``PD20_NAC`` where it matches the volume base, else ``100 * CP / CLV``).

Source: Eurostat dissemination API, SDMX 2.1
(``https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1``), flows
``nama_10_*``, ``namq_10_*``, ``nasa_10_nf_tr``, ``nasq_10_nf_tr``,
``gov_10a_main``, ``gov_10q_ggnfa``, ``sts_inpr_m``, ``une_rt_m``,
``prc_hicp_minr``, ``irt_st_m``, ``irt_lt_mcby_m``, ``ert_eff_ic_m``,
``sts_trtu_m``.
"""
from __future__ import annotations

import datetime as _dt
import gzip
import http.client
import io
import re
import time
import urllib.error
import warnings
import zlib
from typing import Iterable

import numpy as np
import pandas as pd

from .labor_eurostat import _EUROSTAT_GEO_TO_ISO3

_API = "https://ec.europa.eu/eurostat/api/dissemination"
_SDMX = _API + "/sdmx/2.1"
_ASYNC = _API + "/1.0/async"
_USER_AGENT = "Mozilla/5.0 (puremacro/eurostat)"

#: Eurostat geo codes that are aggregates, never countries. Dropped by name.
EUROSTAT_AGGREGATES: frozenset[str] = frozenset({
    "EU", "EU15", "EU25", "EU27", "EU27_2007", "EU27_2020", "EU28",
    "EA", "EA11", "EA12", "EA13", "EA15", "EA16", "EA17", "EA18", "EA19",
    "EA20", "EA21", "EEA", "EEA28", "EEA30_2007", "EFTA", "FX", "DE_TOT",
    "EU_V", "EA_V", "EU27_2020_EFTA", "EU27_2020_IS_K",
})

#: Euro adoption month per member: from this month on its short-term rates
#: are the euro area's (``irt_st_m`` publishes members only as ``EA``).
EURO_ADOPTION: dict[str, str] = {
    "AUT": "1999-01", "BEL": "1999-01", "DEU": "1999-01", "ESP": "1999-01",
    "FIN": "1999-01", "FRA": "1999-01", "IRL": "1999-01", "ITA": "1999-01",
    "LUX": "1999-01", "NLD": "1999-01", "PRT": "1999-01", "GRC": "2001-01",
    "SVN": "2007-01", "CYP": "2008-01", "MLT": "2008-01", "SVK": "2009-01",
    "EST": "2011-01", "LVA": "2014-01", "LTU": "2015-01", "HRV": "2023-01",
    "BGR": "2026-01",
}

# Units: (house description, unit_mult). unit_mult is the power of ten.
_UNITS = {
    "CP_MNAC": ("millions of national currency, current prices", 6),
    "MIO_NAC": ("millions of national currency, current prices", 6),
    "CRC_MNAC": ("millions of national currency, current replacement cost", 6),
    "CLV20_MNAC": ("millions of national currency, chain-linked volumes, 2020 prices", 6),
    "CLV15_MNAC": ("millions of national currency, chain-linked volumes, 2015 prices", 6),
    "PD20_NAC": ("index, 2020 = 100", 0),
    "THS_PER": ("thousands of persons", 3),
    "THS_HW": ("millions of hours worked", 6),
}

# ---------------------------------------------------------------------------
# Block registries. Each block is one Eurostat request per frequency.
#   flow/key per frequency: the key is a template with {units} and {sa}
#   item_dims: the dimensions that identify a column in the response
#   columns: column -> list of alternative code tuples, in preference order
#   volume: True when the block carries CP/CLV/PD units (adds _real, _defl)
#   nominal: the unit code of the nominal series (None: unit is an item dim)
# ---------------------------------------------------------------------------

_VOL_UNITS = "CLV20_MNAC+CLV15_MNAC"

#: Expenditure side (always fetched): column -> Eurostat na_item.
NA_MAIN: dict[str, str] = {
    "gdp": "B1GQ",
    "cons_hh": "P31_S14_S15",
    "cons_gov": "P3_S13",
    "inv": "P51G",
    "capform": "P5G",
    "exports": "P6",
    "imports": "P7",
}

#: Income side (``income=True``), total economy, current prices: column -> na_item.
NA_INCOME: dict[str, str] = {
    "comp_emp": "D1",
    "surplus_mixed": "B2A3G",
    "taxes_prod_imp": "D2",
    "subsidies": "D3",
    "taxes_products": "D21",
    "subsidies_prod": "D31",
    "taxes_prod": "D21X31",
}

#: Income-side detail from the sector accounts (``income=True``, annual only):
#: column -> (na_item, sector). Mixed income of S1 is that of households.
NA_INCOME_SECTOR: dict[str, tuple[str, str]] = {
    "mixed_income": ("B3G", "S1"),
    "surplus_gross": ("B2G", "S1"),
    "cfc": ("P51C", "S1"),
}

#: Household consumption by durability (``durability=True``): column -> na_item.
NA_DURABILITY: dict[str, str] = {
    "cons_dur": "P311_S14",
    "cons_semidur": "P312_S14",
    "cons_nondur": "P313_S14",
    "cons_serv": "P314_S14",
}

#: Gross fixed capital formation by asset (``assets=True``): column -> asset10.
NA_ASSETS: dict[str, str] = {
    "inv_dwell": "N111G",
    "inv_struct": "N112G",
    "inv_constr": "N11KG",
    "inv_equip": "N11MG",
    "inv_ipp": "N117G",
}

#: Gross value added by A*10 industry (``output=True``): column -> nace_r2.
NA_OUTPUT: dict[str, str] = {
    "va_total": "TOTAL",
    "va_agri": "A",
    "va_ind": "B-E",
    "va_mfg": "C",
    "va_constr": "F",
    "va_trade": "G-I",
    "va_info": "J",
    "va_fin": "K",
    "va_realestate": "L",
    "va_business": "M_N",
    "va_public": "O-Q",
    "va_other": "R-U",
}

#: Employment, hours and population (``labor=True``): column -> (unit, na_item).
NA_LABOR: dict[str, tuple[str, str]] = {
    "emp": ("THS_PER", "EMP_DC"),
    "emp_employees": ("THS_PER", "SAL_DC"),
    "emp_selfemp": ("THS_PER", "SELF_DC"),
    "hours": ("THS_HW", "EMP_DC"),
}

#: GFCF by institutional sector (``sectors=True``): column -> sector.
#: General government comes from the government accounts (``inv_gov``),
#: which cover Germany, Spain, Italy, Portugal and Sweden where the sector
#: accounts do not.
NA_SECTORS: dict[str, str] = {
    "inv_corp": "S11",
    "inv_fin": "S12",
    "inv_hh": "S14_S15",
}

#: Fixed assets, total economy (``capital=True``, annual only): column -> asset10.
NA_CAPITAL: dict[str, str] = {
    "k_net": "N11N",
    "k_dwell": "N111N",
    "k_struct": "N112N",
    "k_equip": "N11MN",
    "k_ipp": "N117N",
    "k_gross": "N11G",
}

#: Monthly indicators: column -> (flow, key, item_dims, item code tuple, units, sa).
MONTHLY: dict[str, tuple[str, str, tuple[str, ...], tuple[str, ...], str, str]] = {
    "ip": ("sts_inpr_m", "M.PRD.B-D+C.SCA.I21.", ("nace_r2",), ("B-D",),
           "index, 2021 = 100", "SCA"),
    "ip_mfg": ("sts_inpr_m", "M.PRD.B-D+C.SCA.I21.", ("nace_r2",), ("C",),
               "index, 2021 = 100", "SCA"),
    "urate": ("une_rt_m", "M.SA.TOTAL.PC_ACT.T.", ("unit",), ("PC_ACT",),
              "percent of the labour force, ages 15-74", "SA"),
    "cpi": ("prc_hicp_minr", "M.I15.TOTAL+NRG+FOOD+TOT_X_NRG_FOOD+TOT_X_NRG.",
            ("coicop18",), ("TOTAL",), "HICP index, 2015 = 100", "NSA"),
    "cpi_energy": ("prc_hicp_minr", "M.I15.TOTAL+NRG+FOOD+TOT_X_NRG_FOOD+TOT_X_NRG.",
                   ("coicop18",), ("NRG",), "HICP index, 2015 = 100", "NSA"),
    "cpi_food": ("prc_hicp_minr", "M.I15.TOTAL+NRG+FOOD+TOT_X_NRG_FOOD+TOT_X_NRG.",
                 ("coicop18",), ("FOOD",), "HICP index, 2015 = 100", "NSA"),
    "cpi_core": ("prc_hicp_minr", "M.I15.TOTAL+NRG+FOOD+TOT_X_NRG_FOOD+TOT_X_NRG.",
                 ("coicop18",), ("TOT_X_NRG_FOOD",), "HICP index, 2015 = 100", "NSA"),
    "cpi_xenergy": ("prc_hicp_minr", "M.I15.TOTAL+NRG+FOOD+TOT_X_NRG_FOOD+TOT_X_NRG.",
                    ("coicop18",), ("TOT_X_NRG",), "HICP index, 2015 = 100", "NSA"),
    "rate_on": ("irt_st_m", "M.IRT_DTD+IRT_M1+IRT_M3.", ("int_rt",), ("IRT_DTD",),
                "percent per annum", "NSA"),
    "rate_1m": ("irt_st_m", "M.IRT_DTD+IRT_M1+IRT_M3.", ("int_rt",), ("IRT_M1",),
                "percent per annum", "NSA"),
    "rate_3m": ("irt_st_m", "M.IRT_DTD+IRT_M1+IRT_M3.", ("int_rt",), ("IRT_M3",),
                "percent per annum", "NSA"),
    "yield_10y": ("irt_lt_mcby_m", "M.MCBY.", ("int_rt",), ("MCBY",),
                  "percent per annum (EMU convergence criterion bond yield)", "NSA"),
    "neer": ("ert_eff_ic_m", "M.NEER_IC42+REER_IC42_CPI.I15.", ("exch_rt",),
             ("NEER_IC42",), "index, 2015 = 100 (42 trading partners)", "NSA"),
    "reer": ("ert_eff_ic_m", "M.NEER_IC42+REER_IC42_CPI.I15.", ("exch_rt",),
             ("REER_IC42_CPI",), "index, 2015 = 100 (42 partners, CPI-deflated)", "NSA"),
    "retail_vol": ("sts_trtu_m", "M.VOL_SLS.G47.SCA.I21.", ("nace_r2",), ("G47",),
                   "index, 2021 = 100 (retail trade volume)", "SCA"),
}

#: Pre-euro national short-term rates for euro members: column -> flow.
#: Spliced in before each member's adoption month (see ``EURO_ADOPTION``).
MONTHLY_HISTORICAL: dict[str, str] = {
    "rate_on": "irt_h_ddmr_m",
    "rate_3m": "irt_h_mr3_m",
}

_SA_ORDER = ("SCA", "SA", "NSA")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _get(url: str, *, timeout: float = 120.0, pause: float = 3.0,
         refresh: bool = False) -> bytes:
    """One cached GET (the seam tests patch). Raises HTTPError / OSError."""
    from puremacro._http import safe_get_bytes_cached

    return safe_get_bytes_cached(
        url, timeout, user_agent=_USER_AGENT,
        headers={"Accept-Encoding": "gzip"},
        rate_limit_seconds=pause, refresh=refresh,
    )


def _cache_put(url: str, body: bytes) -> None:
    """Store ``body`` as the cached answer to ``url``.

    After an asynchronous job the cache would otherwise keep the *envelope*
    under the data URL, and the next call would poll a job id that Eurostat
    has long forgotten; storing the finished file there makes a re-run read
    the data straight from disk.
    """
    import os

    if os.environ.get("PUREMACRO_HTTP_NO_CACHE") == "1":
        return
    try:
        from puremacro._http_cache import cache_write, default_cache_dir

        cache_write(default_cache_dir(), url, body)
    except Exception as exc:  # a cache failure never breaks a fetch
        warnings.warn(f"eurostat: could not cache {url}: {exc}", stacklevel=3)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


#: Transport failures that must never escape a fetcher: ``OSError`` covers
#: ``URLError``/``HTTPError``/timeouts; ``HTTPException`` covers a connection
#: dropped mid-body (``IncompleteRead``) or a malformed status line.
_NET_ERRORS = (OSError, http.client.HTTPException)


class _CorruptGzip(Exception):
    """A body that starts with the gzip magic bytes but does not decompress."""


def _gunzip(body: bytes) -> bytes:
    """Undo ``compressed=true`` (gzip without Content-Encoding) by magic bytes.

    Raises :class:`_CorruptGzip` on a truncated or corrupt file.
    """
    if body[:2] == b"\x1f\x8b":
        try:
            return gzip.decompress(body)
        except (OSError, EOFError, zlib.error) as exc:
            raise _CorruptGzip(str(exc)) from None
    return body


def _fault(body: bytes) -> str:
    try:
        body = _gunzip(body)
    except _CorruptGzip:
        pass
    text = body.decode("utf-8", errors="ignore")
    m = re.search(r"<faultstring>(.*?)</faultstring>", text, re.S)
    msg = m.group(1) if m else text[:300]
    return msg.replace("&apos;", "'").replace("&quot;", '"').strip()


def _is_envelope(body: bytes) -> bool:
    head = body[:2000].lstrip()
    return head.startswith(b"<") and (b"queued" in head or b"SUBMITTED" in head)


def _job_id(body: bytes) -> str | None:
    m = re.search(rb"<(?:\w+:)?id>\s*([0-9A-Za-z-]+)\s*</(?:\w+:)?id>", body)
    return m.group(1).decode() if m else None


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _empty_raw(flow: str, key: str, url: str, status: str) -> pd.DataFrame:
    out = pd.DataFrame(columns=["geo", "TIME_PERIOD", "OBS_VALUE"])
    out.attrs.update({"flow": flow, "key": key, "url": url,
                      "last_update": None, "status": status})
    return out


def _await_job(job: str, *, timeout: float, pause: float,
               max_wait: float) -> tuple[bytes | None, str]:
    """Poll an asynchronous job; return (data bytes, status)."""
    waited = 0.0
    step = max(pause, 3.0)
    while waited <= max_wait:
        _sleep(step)
        waited += step
        try:
            st = _gunzip(_get(f"{_ASYNC}/status/{job}", timeout=timeout,
                              pause=0.0, refresh=True))
        except urllib.error.HTTPError as exc:
            return None, f"async status HTTP {exc.code}"
        except _NET_ERRORS as exc:
            return None, f"async status {type(exc).__name__}"
        except _CorruptGzip:
            return None, "async status corrupt gzip"
        text = st.decode("utf-8", errors="ignore")
        if "AVAILABLE" in text:
            try:
                return _get(f"{_ASYNC}/data/{job}", timeout=timeout,
                            pause=0.0, refresh=True), "ok"
            except urllib.error.HTTPError as exc:
                return None, f"async data HTTP {exc.code}"
            except _NET_ERRORS as exc:
                return None, f"async data {type(exc).__name__}"
        if "Fault" in text or "UNKNOWN" in text or "EXPIRED" in text:
            return None, "async job unknown"
    return None, f"async timeout after {max_wait:g} s"


def eurostat_get(flow: str, key: str, *, start: str | int | None = None,
                 end: str | int | None = None, last_n: int | None = None,
                 refresh: bool = False, timeout: float = 120.0,
                 pause: float = 3.0, max_wait: float = 300.0) -> pd.DataFrame:
    """Download one Eurostat SDMX-CSV slice as a raw frame.

    Parameters
    ----------
    flow : Eurostat dataflow id, e.g. ``"nama_10_gdp"`` (case-insensitive).
    key : dot-separated key in the flow's dimension order, ``+`` for OR and
        an empty position for "all", e.g. ``"A.CP_MNAC.B1GQ."``. Print the
        order with :func:`eurostat_codes`.
    start, end : ``startPeriod`` / ``endPeriod`` (``2000``, ``"2000-Q1"``,
        ``"2000-01"``). ``None`` is the whole history.
    last_n : ``lastNObservations``.
    refresh : re-download instead of reading the on-disk cache.
    timeout : seconds per HTTP request.
    pause : minimum seconds between two requests to Eurostat.
    max_wait : seconds to wait for a queued (asynchronous) request.

    Returns
    -------
    DataFrame with Eurostat's SDMX-CSV columns (``DATAFLOW``, ``LAST UPDATE``,
    ``freq``, one column per dimension, ``TIME_PERIOD``, ``OBS_VALUE``,
    ``OBS_FLAG``, ``CONF_STATUS``); codes as strings, ``OBS_VALUE`` float.
    ``attrs``: ``flow``, ``key``, ``url``, ``last_update`` (the release
    stamp) and ``status`` (``"ok"`` or why the frame is empty). A provider
    failure (timeout, 5xx, a job that never finishes) returns an empty frame
    and warns.

    Raises
    ------
    ValueError
        HTTP 400/404: an unknown flow, a wrong number of dimensions, or a code
        outside the flow's content constraint. The message is Eurostat's
        ``faultstring``, which names every offending code.

    Examples
    --------
    >>> eurostat_get("nama_10_gdp", "A.CP_MNAC.B1GQ.DE+FR")  # doctest: +SKIP
    """
    params = ["format=SDMX-CSV", "compressed=true"]
    if start is not None:
        params.append(f"startPeriod={start}")
    if end is not None:
        params.append(f"endPeriod={end}")
    if last_n is not None:
        params.append(f"lastNObservations={int(last_n)}")
    url = f"{_SDMX}/data/{flow}/{key}?" + "&".join(params)

    status = "ok"
    body: bytes | None = None
    for attempt in range(2):
        try:
            body = _get(url, timeout=timeout, pause=pause,
                        refresh=refresh or attempt > 0)
        except urllib.error.HTTPError as exc:
            try:
                err = exc.read() or b""
            except Exception:
                err = b""
            if exc.code in (400, 404):
                raise ValueError(
                    f"eurostat_get({flow!r}, {key!r}): HTTP {exc.code}: "
                    f"{_fault(err)}") from None
            status = f"HTTP {exc.code}"
            body = None
            break
        except _NET_ERRORS as exc:
            status = f"{type(exc).__name__}: {exc}"[:200]
            body = None
            break
        try:
            body = _gunzip(body)
        except _CorruptGzip:
            # A truncated download may sit in the cache: fetch it once more.
            status, body = "corrupt gzip", None
            continue
        if not _is_envelope(body):
            break
        job = _job_id(body)
        if job is None:
            status, body = "async envelope without job id", None
            break
        data, status = _await_job(job, timeout=timeout, pause=pause,
                                  max_wait=max_wait)
        if data is not None:
            try:
                body = _gunzip(data)
            except _CorruptGzip:
                status, body = "async data corrupt gzip", None
                continue
            _cache_put(url, data)
            break
        body = None
        if status != "async job unknown":
            break
        # A stale envelope read back from the cache: ask Eurostat again
        # (``status`` keeps the failure if the second job is unknown too).
        if attempt:
            status = "async job unknown after retry"
    if body is None:
        warnings.warn(f"eurostat_get({flow!r}, {key!r}): {status}; "
                      "returning an empty frame", stacklevel=2)
        return _empty_raw(flow, key, url, status)

    if body.lstrip()[:1] == b"<":
        status = "unexpected XML: " + _fault(body)[:150]
        warnings.warn(f"eurostat_get({flow!r}, {key!r}): {status}", stacklevel=2)
        return _empty_raw(flow, key, url, status)
    try:
        raw = pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False,
                          na_values=[""])
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        warnings.warn(f"eurostat_get({flow!r}, {key!r}): unreadable CSV ({exc})",
                      stacklevel=2)
        return _empty_raw(flow, key, url, "unreadable CSV")
    if "TIME_PERIOD" not in raw.columns or "OBS_VALUE" not in raw.columns:
        warnings.warn(f"eurostat_get({flow!r}, {key!r}): no TIME_PERIOD/OBS_VALUE "
                      "columns", stacklevel=2)
        return _empty_raw(flow, key, url, "unexpected columns")
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")
    last = None
    if "LAST UPDATE" in raw.columns and len(raw):
        last = str(raw["LAST UPDATE"].iloc[0])
    raw.attrs.update({"flow": flow, "key": key, "url": url,
                      "last_update": last, "status": "ok"})
    return raw


def eurostat_codes(flow: str, *, refresh: bool = False,
                   timeout: float = 60.0) -> dict[str, list[str]]:
    """Dimension order and available codes of a Eurostat flow.

    Reads the flow's data structure (for the order of the key) and its content
    constraint (for the codes that actually carry data), two small requests.

    Parameters
    ----------
    flow : dataflow id, e.g. ``"namq_10_gdp"``.
    refresh : re-download instead of reading the cache.
    timeout : seconds per request.

    Returns
    -------
    dict ``{dimension: [codes]}`` in key order, with ``TIME_PERIOD`` (the
    available periods) last. Empty dict, with a warning, when Eurostat does
    not answer.

    Raises
    ------
    ValueError
        Unknown flow (HTTP 404/400).

    Examples
    --------
    >>> list(eurostat_codes("nama_10_gdp"))  # doctest: +SKIP
    ['freq', 'unit', 'na_item', 'geo', 'TIME_PERIOD']
    """
    import xml.etree.ElementTree as ET

    bodies = []
    for url in (f"{_SDMX}/datastructure/ESTAT/{flow}?references=none",
                f"{_SDMX}/contentconstraint/ESTAT/{flow}"):
        try:
            bodies.append(_gunzip(_get(url, timeout=timeout, refresh=refresh)))
        except urllib.error.HTTPError as exc:
            try:
                err = exc.read() or b""
            except Exception:
                err = b""
            if exc.code in (400, 404):
                raise ValueError(f"eurostat_codes({flow!r}): HTTP {exc.code}: "
                                 f"{_fault(err)}") from None
            warnings.warn(f"eurostat_codes({flow!r}): HTTP {exc.code}", stacklevel=2)
            return {}
        except _NET_ERRORS as exc:
            warnings.warn(f"eurostat_codes({flow!r}): {type(exc).__name__}",
                          stacklevel=2)
            return {}
        except _CorruptGzip:
            warnings.warn(f"eurostat_codes({flow!r}): corrupt gzip", stacklevel=2)
            return {}
    try:
        dsd = ET.fromstring(bodies[0])
        cc = ET.fromstring(bodies[1])
    except ET.ParseError:
        warnings.warn(f"eurostat_codes({flow!r}): unreadable XML", stacklevel=2)
        return {}

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    dims: list[tuple[int, str]] = []
    for el in dsd.iter():
        if local(el.tag) in ("Dimension", "TimeDimension") and el.get("id"):
            dims.append((int(el.get("position") or 999), el.get("id")))
    order = [d for _, d in sorted(dims)]
    codes: dict[str, list[str]] = {}
    for el in cc.iter():
        if local(el.tag) == "CubeRegion" and el.get("include", "true") == "true":
            for kv in el:
                if local(kv.tag) == "KeyValue":
                    codes.setdefault(kv.get("id"), []).extend(
                        v.text for v in kv if local(v.tag) == "Value" and v.text)
    out = {d: codes.get(d, []) for d in order if d != "TIME_PERIOD"}
    out["TIME_PERIOD"] = codes.get("TIME_PERIOD", [])
    for d, v in codes.items():  # dimensions the DSD did not list
        out.setdefault(d, v)
    return out


# ---------------------------------------------------------------------------
# Shared reshaping
# ---------------------------------------------------------------------------

def _iso3(geo: pd.Series) -> pd.Series:
    """Map Eurostat geo to ISO-3; aggregates (and unknown codes) become NaN."""
    g = geo.astype(str)
    return g.where(~g.isin(EUROSTAT_AGGREGATES)).map(_EUROSTAT_GEO_TO_ISO3)


def _dates(tp: pd.Series, freq: str) -> pd.Series:
    pf = {"A": "Y", "Q": "Q", "M": "M"}[freq]
    idx = pd.PeriodIndex(tp.astype(str), freq=pf).to_timestamp(how="start")
    return pd.Series(idx, index=tp.index)


_BAD_CODES = re.compile(r"([A-Za-z0-9_]+)=([A-Za-z0-9_\-]+)")


def _pull(flow: str, key: str, *, start, refresh: bool, timeout: float,
          missing: list[str]) -> pd.DataFrame:
    """``eurostat_get`` that never raises and survives retired codes.

    A 400 naming codes outside the content constraint (Eurostat retires codes
    between releases) drops those codes from the key and asks again; what was
    dropped goes to ``missing``.
    """
    for _ in range(3):
        try:
            raw = eurostat_get(flow, key, start=start, refresh=refresh,
                               timeout=timeout)
        except ValueError as exc:
            msg = str(exc)
            bad = [c for _, c in _BAD_CODES.findall(msg.split("not allowed:")[-1])] \
                if "INVALID_QUERY_DIMENSION_VALUE" in msg else []
            parts = key.split(".")
            new = []
            for p in parts:
                toks = [t for t in p.split("+") if t not in bad] if p else []
                if p and not toks:  # every code of a dimension retired
                    missing.append(f"{flow}: {msg}")
                    return _empty_raw(flow, key, "", "invalid key")
                new.append("+".join(toks))
            newkey = ".".join(new)
            if not bad or newkey == key:
                missing.append(f"{flow}: {msg}")
                warnings.warn(msg, stacklevel=3)
                return _empty_raw(flow, key, "", "invalid key")
            missing.append(f"{flow}: codes not published, dropped: {', '.join(bad)}")
            key = newkey
            continue
        if raw.attrs.get("status") != "ok":
            missing.append(f"{flow}/{key}: {raw.attrs.get('status')}")
        return raw
    return _empty_raw(flow, key, "", "invalid key")


def _assemble(raw: pd.DataFrame, *, freq: str, columns: dict[str, list[tuple]],
              item_dims: tuple[str, ...], nominal: str | None, volume: bool,
              sa: str, units: dict[str, str] | None = None,
              scale: dict[str, float] | None = None, flow: str = "",
              key: str = "") -> tuple[pd.DataFrame, list[dict]]:
    """Long SDMX rows -> wide (code, date) frame plus per-country meta.

    Per (country, column): the first alternative code tuple with data wins,
    then the seasonal adjustment by ``sa``, then (``volume``) the 2020 volume
    base with a 2015 fallback, and the deflator ``PD20_NAC`` or CP/CLV.
    """
    units = units or {}
    scale = scale or {}
    if raw is None or raw.empty or any(d not in raw.columns for d in item_dims):
        return pd.DataFrame(), []
    df = raw.copy()
    df["code"] = _iso3(df["geo"])
    df = df[df["code"].notna() & df["OBS_VALUE"].notna()]
    if df.empty:
        return pd.DataFrame(), []
    df["date"] = _dates(df["TIME_PERIOD"], freq).values
    has_sa = "s_adj" in df.columns
    has_unit = "unit" in df.columns
    item = list(zip(*(df[d].astype(str) for d in item_dims)))
    lookup: dict[tuple, tuple[str, int]] = {}
    for col, alts in columns.items():
        for rank, alt in enumerate(alts):
            lookup.setdefault(tuple(alt), (col, rank))
    hit = [lookup.get(t) for t in item]
    df["_col"] = [h[0] if h else None for h in hit]
    df["_rank"] = [h[1] if h else -1 for h in hit]
    df = df[df["_col"].notna()]
    if sa == "prefer":
        sa_order = list(_SA_ORDER)
    else:
        sa_order = [sa]

    series: dict[str, list[pd.Series]] = {}
    meta: list[dict] = []
    for (code, col), g in df.groupby(["code", "_col"], sort=True):
        nom_rows = g if (nominal is None or not has_unit) else g[g["unit"] == nominal]
        if nom_rows.empty:
            continue
        if nominal is None and has_unit and nom_rows["unit"].nunique() > 1:
            # Never mix index bases month by month: keep one unit, by name.
            units_seen = sorted(nom_rows["unit"].astype(str).unique())
            warnings.warn(f"eurostat: {flow} {code} {col} comes in several units "
                          f"{units_seen}; keeping {units_seen[0]}", stacklevel=4)
            nom_rows = nom_rows[nom_rows["unit"] == units_seen[0]]
            g = g[g["unit"] == units_seen[0]]
        rank = int(nom_rows["_rank"].min())
        g = g[g["_rank"] == rank]
        nom_rows = nom_rows[nom_rows["_rank"] == rank]
        chosen_sa = None
        if has_sa:
            avail = set(nom_rows["s_adj"])
            chosen_sa = next((s for s in sa_order if s in avail), None)
            if chosen_sa is None:
                continue
            g = g[g["s_adj"] == chosen_sa]
            nom_rows = nom_rows[nom_rows["s_adj"] == chosen_sa]
        nom = nom_rows.set_index("date")["OBS_VALUE"].sort_index()
        nom = nom[~nom.index.duplicated()] * scale.get(col, 1.0)
        series.setdefault(col, []).append(_tag(nom, code))
        unit_code = nominal or (str(nom_rows["unit"].iloc[0]) if has_unit else "")
        udesc, umult = _UNITS.get(unit_code, (units.get(col, unit_code), 0))
        if col in units:
            udesc = units[col]
        rec = {
            "code": code, "variable": col, "flow": flow, "key": key,
            "item": "/".join(columns[col][rank]), "units": udesc,
            "unit_mult": umult,
            "sa": chosen_sa if has_sa else ("annual" if freq == "A" else None),
            "first": str(nom.index.min().date()), "last": str(nom.index.max().date()),
            "n": int(nom.size),
        }
        if volume and has_unit:
            base = None
            for u in ("CLV20_MNAC", "CLV15_MNAC"):
                r = g[g["unit"] == u]
                if not r.empty:
                    base = u
                    break
            rec["volume_base"] = base
            rec["defl_source"] = None
            if base is not None:
                real = r.set_index("date")["OBS_VALUE"].sort_index()
                real = real[~real.index.duplicated()]
                series.setdefault(col + "_real", []).append(_tag(real, code))
                pdr = g[g["unit"] == "PD20_NAC"] if base == "CLV20_MNAC" else g.iloc[0:0]
                if not pdr.empty:
                    defl = pdr.set_index("date")["OBS_VALUE"].sort_index()
                    defl = defl[~defl.index.duplicated()]
                    rec["defl_source"] = "PD20_NAC"
                else:
                    defl = (100.0 * nom / real.replace(0.0, np.nan)).dropna()
                    rec["defl_source"] = f"100*{unit_code}/{base}"
                series.setdefault(col + "_defl", []).append(_tag(defl, code))
                rec["real_first"] = str(real.index.min().date())
                rec["real_last"] = str(real.index.max().date())
        meta.append(rec)
    if not series:
        return pd.DataFrame(), meta
    wide = pd.concat({c: pd.concat(v) for c, v in series.items()}, axis=1)
    return wide, meta


def _tag(s: pd.Series, code: str) -> pd.Series:
    s = s.astype(float)
    s.index = pd.MultiIndex.from_arrays(
        [np.repeat(code, len(s)), pd.DatetimeIndex(s.index)], names=["code", "date"])
    return s


def _empty_panel(columns: Iterable[str] = ()) -> pd.DataFrame:
    idx = pd.MultiIndex.from_arrays(
        [pd.Index([], dtype=object), pd.DatetimeIndex([])], names=["code", "date"])
    return pd.DataFrame(index=idx, columns=list(columns), dtype=float)


def _check_codes(codes) -> list[str] | None:
    if codes is None:
        return None
    if isinstance(codes, str):
        codes = [codes]
    codes = [str(c).upper() for c in codes]
    valid = sorted(set(_EUROSTAT_GEO_TO_ISO3.values()))
    bad = [c for c in codes if c not in valid]
    if bad:
        raise ValueError(f"unknown country code(s) {bad}; Eurostat reporters are "
                         f"ISO-3 codes among {valid}")
    return codes


def _finish(parts: list[pd.DataFrame], order: list[str], codes: list[str] | None,
            meta: list[dict], missing: list[str], flows: list[str],
            last_update: dict[str, str | None], start) -> pd.DataFrame:
    parts = [p for p in parts if not p.empty]
    if parts:
        out = pd.concat(parts, axis=1)
        out = out.groupby(level=[0, 1]).first() if out.index.has_duplicates else out
    else:
        out = _empty_panel(order)  # stable schema: df["gdp"] still works
    if codes is not None and len(out):
        out = out[out.index.get_level_values("code").isin(codes)]
        meta = [m for m in meta if m["code"] in codes]
    absent_cols = [c for c in order if c not in out.columns]
    out = out.reindex(columns=order).sort_index().astype(float)
    out = out.dropna(how="all")
    out.index = out.index.set_names(["code", "date"])
    if codes is not None:
        have = {(m["code"], m["variable"]) for m in meta}
        base = [c for c in order if not c.endswith(("_real", "_defl"))]
        for c in codes:
            lacking = [v for v in base if (c, v) not in have]
            if len(lacking) == len(base):
                missing.append(f"{c}: no data")
            elif lacking:
                missing.append(f"{c}: " + ", ".join(lacking))
    else:
        absent = [c for c in absent_cols if not c.endswith(("_real", "_defl"))]
        if absent:
            missing.append("not obtained for any country: " + ", ".join(absent))
    out.attrs["meta"] = tuple(dict(m) for m in meta)
    out.attrs["source"] = "Eurostat SDMX 2.1: " + ", ".join(dict.fromkeys(flows))
    out.attrs["fetched_at"] = _now()
    out.attrs["last_update"] = dict(last_update)
    out.attrs["start"] = None if start is None else str(start)
    out.attrs["missing"] = tuple(dict.fromkeys(missing))
    if out.empty:
        warnings.warn("eurostat: nothing obtained; " + "; ".join(missing[:5]),
                      stacklevel=3)
    return out


# ---------------------------------------------------------------------------
# National accounts
# ---------------------------------------------------------------------------

def _na_blocks(freq: str, *, durability, assets, output, income, labor, sectors,
               capital, real) -> list[dict]:
    """Request plan: one dict per Eurostat request."""
    A = freq == "A"
    u = "CP_MNAC+" + _VOL_UNITS + "+PD20_NAC" if real else "CP_MNAC"
    u_nopd = "CP_MNAC+" + _VOL_UNITS if real else "CP_MNAC"
    sa = "SCA+SA+NSA"
    blocks: list[dict] = []

    def one(cols: dict[str, str]) -> dict[str, list[tuple]]:
        return {c: [(v,)] for c, v in cols.items()}

    blocks.append(dict(
        name="main", flow="nama_10_gdp" if A else "namq_10_gdp",
        key=(f"A.{u}." if A else f"Q.{u}.{sa}.") + "+".join(NA_MAIN.values()) + ".",
        item_dims=("na_item",), columns=one(NA_MAIN), nominal="CP_MNAC",
        volume=real))
    if income:
        blocks.append(dict(
            name="income", flow="nama_10_gdp" if A else "namq_10_gdp",
            key=("A.CP_MNAC." if A else f"Q.CP_MNAC.{sa}.")
            + "+".join(NA_INCOME.values()) + ".",
            item_dims=("na_item",), columns=one(NA_INCOME), nominal="CP_MNAC",
            volume=False))
        if A:
            items = "+".join(dict.fromkeys(v[0] for v in NA_INCOME_SECTOR.values()))
            blocks.append(dict(
                name="income_sector", flow="nasa_10_nf_tr",
                # dims freq.unit.direct.na_item.sector.geo; balancing items are
                # published under PAID and RECV with equal values.
                key=f"A.CP_MNAC.PAID+RECV.{items}.S1.",
                item_dims=("direct", "na_item", "sector"),
                columns={c: [("PAID",) + v, ("RECV",) + v]
                         for c, v in NA_INCOME_SECTOR.items()},
                nominal="CP_MNAC", volume=False))
    if durability:
        blocks.append(dict(
            name="durability", flow="nama_10_fcs" if A else "namq_10_fcs",
            # no PD deflators in the fcs flows: deflator = 100 * CP / CLV
            key=(f"A.{u_nopd}." if A else f"Q.{u_nopd}.{sa}.")
            + "+".join(NA_DURABILITY.values()) + ".",
            item_dims=("na_item",), columns=one(NA_DURABILITY), nominal="CP_MNAC",
            volume=real))
    if assets:
        assets_key = "+".join(NA_ASSETS.values())
        blocks.append(dict(
            name="assets", flow="nama_10_an6" if A else "namq_10_an6",
            # annual dims freq.unit.asset10.geo; quarterly asset10 BEFORE s_adj
            key=(f"A.{u}.{assets_key}." if A else f"Q.{u_nopd}.{assets_key}.{sa}."),
            item_dims=("asset10",), columns=one(NA_ASSETS), nominal="CP_MNAC",
            volume=real))
    if output:
        nace = "+".join(NA_OUTPUT.values())
        blocks.append(dict(
            name="output", flow="nama_10_a10" if A else "namq_10_a10",
            key=(f"A.{u}.{nace}.B1G." if A else f"Q.{u_nopd}.{sa}.{nace}.B1G."),
            item_dims=("nace_r2",), columns=one(NA_OUTPUT), nominal="CP_MNAC",
            volume=real))
    if labor:
        blocks.append(dict(
            name="labor", flow="nama_10_a10_e" if A else "namq_10_a10_e",
            key=("A.THS_PER+THS_HW.TOTAL.EMP_DC+SAL_DC+SELF_DC." if A else
                 f"Q.THS_PER+THS_HW.TOTAL.{sa}.EMP_DC+SAL_DC+SELF_DC."),
            item_dims=("unit", "na_item"),
            columns={c: [v] for c, v in NA_LABOR.items()}, nominal=None,
            volume=False, scale={"hours": 1e-3},
            units={"hours": "millions of hours worked"}))
        blocks.append(dict(
            name="pop", flow="nama_10_pe" if A else "namq_10_pe",
            key="A.THS_PER.POP_NC." if A else f"Q.THS_PER.{sa}.POP_NC.",
            item_dims=("na_item",), columns={"pop": [("POP_NC",)]},
            nominal="THS_PER", volume=False))
    if sectors:
        secs = "+".join(NA_SECTORS.values())
        blocks.append(dict(
            name="sectors", flow="nasa_10_nf_tr" if A else "nasq_10_nf_tr",
            key=(f"A.CP_MNAC.PAID.P51G.{secs}." if A else
                 f"Q.CP_MNAC.PAID.{secs}.P51G.SCA+NSA."),
            item_dims=("sector",), columns=one(NA_SECTORS), nominal="CP_MNAC",
            volume=False))
        blocks.append(dict(
            name="inv_gov", flow="gov_10a_main" if A else "gov_10q_ggnfa",
            key="A.MIO_NAC.S13.P51G." if A else f"Q.MIO_NAC.{sa}.S13.P51G.",
            item_dims=("sector",), columns={"inv_gov": [("S13",)]},
            nominal="MIO_NAC", volume=False))
    if capital and A:
        ku = "CRC_MNAC+" + _VOL_UNITS if real else "CRC_MNAC"
        blocks.append(dict(
            name="capital", flow="nama_10_nfa_st",
            # dims freq.unit.nace_r2.asset10.geo
            key=f"A.{ku}.TOTAL." + "+".join(NA_CAPITAL.values()) + ".",
            item_dims=("asset10",), columns=one(NA_CAPITAL), nominal="CRC_MNAC",
            volume=real))
    return blocks


def eurostat_na_panel(codes=None, *, freq: str = "A", start: str | int | None = None,
                      durability: bool = False, assets: bool = False,
                      output: bool = False, income: bool = False,
                      labor: bool = False, sectors: bool = False,
                      capital: bool = False, real: bool = True,
                      sa: str = "prefer", refresh: bool = False,
                      timeout: float = 120.0) -> pd.DataFrame:
    """Eurostat national accounts of every European reporter, one frame.

    Parameters
    ----------
    codes : ISO-3 code or list of codes; ``None`` keeps every country.
        Filtering happens after download, so any subset reuses the cached
        all-country response.
    freq : ``"A"`` (annual, from 1975) or ``"Q"`` (quarterly, from 1978Q1).
    start : first period (``1995``, ``"1995-Q1"``); ``None`` is the whole
        history. It is part of the request, so keep it fixed across calls
        that should share the cache.
    durability : household consumption by durability, ``cons_dur``,
        ``cons_semidur``, ``cons_nondur``, ``cons_serv``.
    assets : GFCF by asset, ``inv_dwell``, ``inv_struct``, ``inv_constr``
        (dwellings + other structures), ``inv_equip``, ``inv_ipp``.
    output : gross value added by A*10 industry, ``va_total``, ``va_agri``,
        ``va_ind`` (B-E), ``va_mfg`` (C), ``va_constr``, ``va_trade`` (G-I),
        ``va_info``, ``va_fin``, ``va_realestate``, ``va_business`` (M-N),
        ``va_public`` (O-Q), ``va_other`` (R-U), plus ``va_services``
        (G-U = total less A, B-E and F; nominal only).
    income : ``comp_emp`` (D1), ``surplus_mixed`` (B2A3G), ``taxes_prod_imp``
        (D2), ``subsidies`` (D3), ``taxes_products`` (D21, taxes on products; VAT D211 is one part of it),
        ``subsidies_prod`` (D31), ``taxes_prod`` (D21X31, net taxes on
        products); annual adds ``mixed_income`` (B3G), ``surplus_gross``
        (B2G) and ``cfc`` (P51C, consumption of fixed capital) from the
        sector accounts.
    labor : ``emp``, ``emp_employees``, ``emp_selfemp`` (thousands of
        persons, domestic concept), ``hours`` (millions of hours worked),
        ``pop`` (thousands).
    sectors : GFCF by sector, ``inv_corp`` (S11), ``inv_fin`` (S12),
        ``inv_hh`` (S14_S15) and ``inv_gov`` (S13, from the government
        accounts).
    capital : net fixed assets at current replacement cost, ``k_net``,
        ``k_dwell``, ``k_struct``, ``k_equip``, ``k_ipp`` and gross
        ``k_gross`` (annual only).
    real : add ``<col>_real`` (chain-linked volumes, 2020 reference year,
        2015 where 2020 is absent) and ``<col>_defl`` (100 in the reference
        year) to every block that publishes volumes.
    sa : quarterly seasonal adjustment, ``"prefer"`` (SCA, else SA, else NSA,
        per country and variable), or exactly ``"SCA"``, ``"SA"``, ``"NSA"``.
    refresh : re-download instead of reading the on-disk cache.
    timeout : seconds per request.

    Returns
    -------
    DataFrame indexed by ``(code, date)`` (ISO-3; period-start timestamps),
    one float column per variable. ``attrs``: ``meta`` (tuple of dicts, one per
    country and variable: flow, key, item, units, unit_mult, sa, volume_base,
    defl_source, first, last, n), ``source``, ``fetched_at``, ``last_update``
    (release stamp per flow) and ``missing`` (blocks or country-variables asked
    for and not obtained). Provider failures leave an empty frame or missing
    columns with a warning; they never raise.

    Raises
    ------
    ValueError
        Unknown ``freq``, ``sa`` or country code.

    Examples
    --------
    >>> df = eurostat_na_panel(["DEU", "FRA"], durability=True)  # doctest: +SKIP
    >>> df.loc["FRA", "gdp_real"].head()  # doctest: +SKIP
    """
    freq = str(freq).upper()
    if freq not in ("A", "Q"):
        raise ValueError(f"freq must be one of ['A', 'Q'], got {freq!r}")
    if sa not in ("prefer",) + _SA_ORDER:
        raise ValueError(f"sa must be one of {['prefer', *_SA_ORDER]}, got {sa!r}")
    codes = _check_codes(codes)
    missing: list[str] = []
    if capital and freq == "Q":
        warnings.warn("eurostat_na_panel: capital=True is annual only; ignored "
                      "for freq='Q'", stacklevel=2)
        missing.append("capital: annual only (nama_10_nfa_st)")
    blocks = _na_blocks(freq, durability=durability, assets=assets, output=output,
                        income=income, labor=labor, sectors=sectors,
                        capital=capital, real=real)
    parts: list[pd.DataFrame] = []
    meta: list[dict] = []
    flows: list[str] = []
    last_update: dict[str, str | None] = {}
    order: list[str] = []
    for b in blocks:
        cols = list(b["columns"])
        order += cols
        if b.get("volume"):
            order += [c + "_real" for c in cols] + [c + "_defl" for c in cols]
        if b["name"] == "output":
            order.insert(order.index("va_total") + 1, "va_services")
        raw = _pull(b["flow"], b["key"], start=start, refresh=refresh,
                    timeout=timeout, missing=missing)
        flows.append(b["flow"])
        last_update[b["flow"]] = raw.attrs.get("last_update")
        wide, m = _assemble(raw, freq=freq, columns=b["columns"],
                            item_dims=b["item_dims"], nominal=b["nominal"],
                            volume=b.get("volume", False), sa=sa,
                            units=b.get("units"), scale=b.get("scale"),
                            flow=b["flow"], key=b["key"])
        if b["name"] == "output" and not wide.empty:
            need = ["va_total", "va_agri", "va_ind", "va_constr"]
            if all(c in wide.columns for c in need):
                wide["va_services"] = (wide["va_total"] - wide["va_agri"]
                                       - wide["va_ind"] - wide["va_constr"])
                tot = {r["code"]: r for r in m if r["variable"] == "va_total"}
                for code, s in wide["va_services"].dropna().groupby(level="code"):
                    dates = s.index.get_level_values("date")
                    rec = {k: v for k, v in tot.get(code, {}).items()
                           if k in ("flow", "key", "units", "unit_mult", "sa")}
                    m.append({**rec, "code": code, "variable": "va_services",
                              "item": "TOTAL-A-(B-E)-F (derived)",
                              "first": str(dates.min().date()),
                              "last": str(dates.max().date()), "n": int(s.size)})
        if wide.empty and raw.attrs.get("status") == "ok":
            missing.append(f"{b['name']} ({b['flow']}): no rows")
        parts.append(wide)
        meta += m
    return _finish(parts, order, codes, meta, missing, flows, last_update, start)


# ---------------------------------------------------------------------------
# Monthly indicators
# ---------------------------------------------------------------------------

def _euro_splice(out: pd.DataFrame, ea: dict[str, pd.Series],
                 hist: dict[str, pd.DataFrame], meta: list[dict],
                 hflows: dict[str, str]) -> pd.DataFrame:
    """Give euro members the euro-area short rates from adoption on.

    Before adoption a member keeps its own ``irt_st_m`` series where Eurostat
    publishes one (Bulgaria, Croatia, the Baltics) and otherwise its historical
    national series (``irt_h_*``, which runs to 2014 for the 3-month rate).
    """
    cols = {c: out[c].dropna() for c in out.columns}
    for col, ea_s in ea.items():
        cur = cols.get(col, pd.Series(dtype=float, index=_empty_panel().index))
        new = []
        for code, adopt in EURO_ADOPTION.items():
            t0 = pd.Timestamp(adopt + "-01")
            own = (cur.xs(code, level="code")
                   if code in cur.index.get_level_values("code")
                   else pd.Series(dtype=float, index=pd.DatetimeIndex([])))
            pre = own[own.index < t0]
            src = "irt_st_m" if not pre.empty else ""
            h = hist.get(col)
            if h is not None and code in h.columns:
                hs = h[code].dropna()
                hs = hs[hs.index < t0]
                if not hs.empty:
                    pre = pre.combine_first(hs) if not pre.empty else hs
                    src = (src + "+" if src else "") + hflows.get(col, "irt_h")
            post = ea_s[ea_s.index >= t0]
            s = pd.concat([pre, post]).sort_index()
            s = s[~s.index.duplicated(keep="first")].dropna()
            if s.empty:
                continue
            new.append(_tag(s, code))
            meta[:] = [m for m in meta
                       if not (m["code"] == code and m["variable"] == col)]
            meta.append({
                "code": code, "variable": col, "flow": "irt_st_m",
                "key": MONTHLY[col][1], "item": MONTHLY[col][3][0],
                "units": MONTHLY[col][4], "unit_mult": 0, "sa": "NSA",
                "splice": (f"{src} before {adopt}; " if src else "")
                + f"EA from {adopt}",
                "first": str(s.index.min().date()),
                "last": str(s.index.max().date()), "n": int(s.size),
            })
        if new:
            members = [c for c in EURO_ADOPTION
                       if c in cur.index.get_level_values("code")]
            rest = cur.drop(index=members, level="code") if members else cur
            cols[col] = pd.concat([rest] + new).sort_index()
    if not cols:
        return out
    return pd.concat(cols, axis=1)


def eurostat_monthly_panel(codes=None, *, start: str | int | None = None,
                           variables: Iterable[str] | None = None,
                           refresh: bool = False,
                           timeout: float = 120.0) -> pd.DataFrame:
    """Monthly Eurostat indicators for every European reporter, one frame.

    Columns (see ``MONTHLY``): ``ip``, ``ip_mfg`` (industrial production and
    manufacturing, 2021 = 100, SCA), ``urate`` (percent, SA), ``cpi``,
    ``cpi_energy``, ``cpi_food``, ``cpi_core`` (all items less energy, food,
    alcohol and tobacco), ``cpi_xenergy`` (HICP, 2015 = 100, NSA),
    ``rate_on``, ``rate_1m``, ``rate_3m`` (money market, percent; euro
    members carry the euro-area rate from their adoption month and their
    national history before it), ``yield_10y`` (percent), ``neer``,
    ``reer`` (2015 = 100, 42 partners) and ``retail_vol`` (2021 = 100, SCA).
    The United States and Japan appear where Eurostat publishes them
    (``urate``, ``rate_*``, ``neer``/``reer`` with ten more partners).

    Parameters
    ----------
    codes : ISO-3 code or list; ``None`` keeps every country.
    start : first month (``"1995-01"`` or ``1995``); ``None`` is the whole
        history.
    variables : subset of the columns above; ``None`` fetches all of them
        (seven requests, plus two for the pre-euro rates).
    refresh : re-download instead of reading the on-disk cache.
    timeout : seconds per request.

    Returns
    -------
    DataFrame indexed by ``(code, date)`` with month-start dates, one float
    column per variable, ``attrs`` as in :func:`eurostat_na_panel`
    (``meta`` per country and variable, ``source``, ``fetched_at``,
    ``last_update``, ``missing``).

    Raises
    ------
    ValueError
        Unknown variable or country code.

    Examples
    --------
    >>> m = eurostat_monthly_panel(["DEU", "ESP"], variables=["cpi", "urate"])  # doctest: +SKIP
    """
    codes = _check_codes(codes)
    if variables is None:
        want = list(MONTHLY)
    else:
        want = [variables] if isinstance(variables, str) else list(variables)
        bad = [v for v in want if v not in MONTHLY]
        if bad:
            raise ValueError(f"unknown variable(s) {bad}; choose among {list(MONTHLY)}")
    if start is not None and re.fullmatch(r"\d{4}", str(start)):
        start = f"{start}-01"
    by_req: dict[tuple[str, str], list[str]] = {}
    for v in want:
        by_req.setdefault(MONTHLY[v][:2], []).append(v)
    missing: list[str] = []
    parts: list[pd.DataFrame] = []
    meta: list[dict] = []
    flows: list[str] = []
    last_update: dict[str, str | None] = {}
    ea: dict[str, pd.Series] = {}
    for (flow, key), cols in by_req.items():
        raw = _pull(flow, key, start=start, refresh=refresh, timeout=timeout,
                    missing=missing)
        flows.append(flow)
        last_update[flow] = raw.attrs.get("last_update")
        spec = {c: [MONTHLY[c][3]] for c in cols}
        dims = MONTHLY[cols[0]][2]
        wide, m = _assemble(raw, freq="M", columns=spec, item_dims=dims,
                            nominal=None, volume=False, sa="prefer",
                            units={c: MONTHLY[c][4] for c in cols},
                            flow=flow, key=key)
        for rec in m:
            rec["sa"] = MONTHLY[rec["variable"]][5]
        parts.append(wide)
        meta += m
        if flow == "irt_st_m" and not raw.empty and dims[0] in raw.columns:
            r = raw[raw["geo"] == "EA"]
            for c in cols:
                s = r[r[dims[0]] == MONTHLY[c][3][0]].dropna(subset=["OBS_VALUE"])
                if not s.empty:
                    ea[c] = pd.Series(s["OBS_VALUE"].values,
                                      index=_dates(s["TIME_PERIOD"], "M").values
                                      ).sort_index()
    out = pd.concat([p for p in parts if not p.empty], axis=1) if any(
        not p.empty for p in parts) else _empty_panel()
    if ea:
        hist: dict[str, pd.DataFrame] = {}
        hflows: dict[str, str] = {}
        for col, hflow in MONTHLY_HISTORICAL.items():
            if col not in ea:
                continue
            raw = _pull(hflow, "M..", start=start, refresh=refresh,
                        timeout=timeout, missing=missing)
            flows.append(hflow)
            last_update[hflow] = raw.attrs.get("last_update")
            if raw.empty:
                continue
            r = raw.copy()
            r["code"] = _iso3(r["geo"])
            r = r[r["code"].notna() & r["OBS_VALUE"].notna()]
            if r.empty:
                continue
            r["date"] = _dates(r["TIME_PERIOD"], "M").values
            hist[col] = r.pivot_table(index="date", columns="code",
                                      values="OBS_VALUE", aggfunc="first")
            hflows[col] = hflow
        out = _euro_splice(out, ea, hist, meta, hflows)
    if "rate_on" in want and "rate_on" in ea and ea["rate_on"].index.max() < (
            pd.Timestamp.now() - pd.DateOffset(months=6)):
        missing.append(
            f"rate_on: euro members end {ea['rate_on'].index.max():%Y-%m} "
            "(EONIA discontinued; irt_st_m carries no euro short-term rate)")
    return _finish([out], want, codes, meta, missing, flows, last_update, start)


def eurostat_meta(panel: pd.DataFrame) -> pd.DataFrame:
    """``panel.attrs["meta"]`` as a table, one row per country and variable.

    Parameters
    ----------
    panel : a frame returned by :func:`eurostat_na_panel` or
        :func:`eurostat_monthly_panel`.

    Returns
    -------
    DataFrame with columns ``code``, ``variable``, ``flow``, ``key``,
    ``item``, ``units``, ``unit_mult``, ``sa``, ``first``, ``last``, ``n`` and,
    where they apply, ``volume_base``, ``defl_source``, ``real_first``,
    ``real_last``, ``splice``. Empty when the panel carries no meta.

    Examples
    --------
    >>> eurostat_meta(eurostat_na_panel(["DEU"]))  # doctest: +SKIP
    """
    rows = list(panel.attrs.get("meta", ()))
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["code", "variable", "flow", "key", "units", "sa", "first",
                 "last", "n"])


__all__ = [
    "EUROSTAT_AGGREGATES",
    "EURO_ADOPTION",
    "MONTHLY",
    "NA_ASSETS",
    "NA_CAPITAL",
    "NA_DURABILITY",
    "NA_INCOME",
    "NA_INCOME_SECTOR",
    "NA_LABOR",
    "NA_MAIN",
    "NA_OUTPUT",
    "NA_SECTORS",
    "eurostat_codes",
    "eurostat_get",
    "eurostat_meta",
    "eurostat_monthly_panel",
    "eurostat_na_panel",
]
