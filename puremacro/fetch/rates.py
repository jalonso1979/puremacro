r"""Interest rates, bond yields, stock indices and commodity prices, monthly, in one call each.

:func:`rates_panel` returns the longest cross-country monthly panel of
money-market rates and bond yields that free providers publish, indexed by
``(code, date)`` with ISO3 codes, month-start dates and one column per
variable, in percent per annum (monthly averages, not seasonally adjusted)::

    rate_on  rate_3m  yield_10y  yield_corp_aaa  yield_corp_baa  yield_corp_de

The names are the house names other fetchers use for the same concepts
(``rate_on`` and ``rate_3m`` as in :mod:`puremacro.fetch.oecd_stes_panel`
and :mod:`puremacro.fetch.eurostat`), so panels from several providers can
be compared and spliced by column name.

:func:`stock_index_monthly` adds each country's headline stock index
(month-end close, local currency) and :func:`commodity_prices_monthly` the
World Bank Pink Sheet (prices and indices, code ``"WLD"``). Two thin
building blocks are public as well: :func:`fetch_fred_many` (several FRED
series, one request each) and :func:`ecb_get` (one ECB Data Portal query as
a raw SDMX-CSV frame).

Why this module exists
----------------------
The OECD Main Economic Indicators carry the classic cross-country interest
rate families (immediate rate, 3-month interbank, 10-year government) and
FRED mirrors them under stable ids, ``IRSTCI01{CC}M156N``,
``IR3TIB01{CC}M156N`` and ``IRLTLT01{CC}M156N``: France and Sweden from
1955, Germany from 1956-60, the United States from 1953-54. FRED only
answers a multi-id request when every id exists, so each series is one
request, drawn from the frozen lists of ids that were verified to exist
(:data:`FRED_COUNTRIES`). Euro members have no national money-market rate
after they adopt the euro, and the FRED mirror of the euro-area rates froze
in January 2026, so from each member's entry date the panel switches to
the ECB's EURIBOR (3 months) and EONIA, then €STR (overnight). The ECB's
convergence-criterion 10-year yields (``IRS``, one request for 28 EU
countries) are the second source for ``yield_10y``: they fill the EU
members FRED lacks (Bulgaria, Croatia, Cyprus, the Baltics, Malta, Romania)
and any month FRED is missing. The ECB's long benchmark histories carry the
United States back to 1900 and Japan back to 1972. Corporate yields exist
free only for the United States (Moody's seasoned Aaa and Baa, 1919) and
Germany (Bundesbank, corporate bonds of non-MFI issuers, 1957).

Mechanics
---------
Every body travels through :func:`puremacro._http.safe_get_bytes_cached`
(urllib, SQLite cache outside the repository, a pause per host), reached via
the module-level :func:`_get` so tests can patch one name. A provider
failure never raises: the series is listed in ``attrs["missing"]`` with the
reason (``HTTP 404``, ``timeout``, ...) and a warning is issued; a call that
gets nothing returns an empty frame with the same index names and ``attrs``
keys. ``attrs["complete"]`` is ``False`` whenever a series that some
provider serves could not be fetched this time (a 429, a timeout), so a
notebook that freezes the panel can refuse a partial pull; the warning
names the first failed series. A cached body that no longer parses (an
HTML error page cached as a 200) is re-downloaded once before it is given
up on. ``ValueError`` is reserved for caller errors (unknown variable, a
malformed code or ``start``). ``attrs["meta"]`` holds one plain dict per
country, variable and source leg actually used (provider, key, units,
first, last, number of observations contributed), so a splice is never
silent. Provider aggregates (``U2``, ``D0``, ``I9``, ``I10``, ``V5`` at the
ECB, ``EZ`` at FRED) are dropped by explicit lists.

What cannot be had free across countries: policy rates (BIS ``WS_CBPOL``,
not wired here), 1-month rates other than EURIBOR, 2-year yields (FRED
removed the OECD family), corporate yields outside the United States and
Germany, and stock indices before 1985 except the three ECB series kept in
``stock_idx_avg``.

Source: FRED (fred.stlouisfed.org, OECD MEI mirror and Moody's), ECB Data
Portal (data-api.ecb.europa.eu, FM and IRS), Deutsche Bundesbank time series
database (BBSIS), Yahoo Finance chart API, World Bank Pink Sheet
(CMO-Historical-Data-Monthly.xlsx).
"""
from __future__ import annotations

import datetime as dt
import io
import json
import re
import urllib.error
import urllib.parse
import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from puremacro._codes import IS_AGGREGATE
from puremacro._http import safe_get_bytes_cached

_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
_ECB_URL = "https://data-api.ecb.europa.eu/service/data/{flow}/{key}"
_BBK_URL = ("https://api.statistiken.bundesbank.de/rest/data/BBSIS/"
            "{key}?format=csv&lang=en")
_YAHOO_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
              "?interval=1mo&period1=0&period2={period2}")
_HEADERS = {"Accept-Encoding": "gzip"}
_TIMEOUT = 120.0
_PCT = "percent per annum"
_SOURCE_RATES = ("FRED (OECD MEI mirror; Moody's); ECB Data Portal (FM, IRS); "
                 "Deutsche Bundesbank (BBSIS)")
_SOURCE_STOCKS = "Yahoo Finance chart API (month-end close); ECB Data Portal FM (monthly averages)"
_SOURCE_PINK = "World Bank Pink Sheet, CMO-Historical-Data-Monthly.xlsx"

#: ISO2 (FRED, ECB) -> ISO3 for every economy any leg of this module covers.
ISO2_TO_ISO3: dict[str, str] = {
    "AT": "AUT", "AU": "AUS", "BE": "BEL", "BG": "BGR", "BR": "BRA",
    "CA": "CAN", "CH": "CHE", "CL": "CHL", "CN": "CHN", "CY": "CYP",
    "CZ": "CZE", "DE": "DEU", "DK": "DNK", "EE": "EST", "ES": "ESP",
    "FI": "FIN", "FR": "FRA", "GB": "GBR", "GR": "GRC", "HR": "HRV",
    "HU": "HUN", "ID": "IDN", "IE": "IRL", "IL": "ISR", "IN": "IND",
    "IS": "ISL", "IT": "ITA", "JP": "JPN", "KR": "KOR", "LT": "LTU",
    "LU": "LUX", "LV": "LVA", "MT": "MLT", "MX": "MEX", "NL": "NLD",
    "NO": "NOR", "NZ": "NZL", "PL": "POL", "PT": "PRT", "RO": "ROU",
    "RU": "RUS", "SE": "SWE", "SI": "SVN", "SK": "SVK", "TR": "TUR",
    "US": "USA", "ZA": "ZAF",
}

#: ECB reference areas that are aggregates, not countries (dropped).
ECB_AGGREGATES: frozenset[str] = frozenset({"U2", "D0", "I8", "I9", "I10", "V5", "4F"})

#: FRED country codes inside the OECD MEI ids that are aggregates (dropped).
FRED_AGGREGATES: frozenset[str] = frozenset({"EZ", "EA", "EU", "OE", "G7"})

#: Variable -> ISO2 countries whose FRED OECD-MEI series exists (checked one
#: id per request on 6 Oct 2026; any other id answers 404). The euro-area
#: ids (``EZ``) exist too but froze in January 2026 and are replaced by ECB.
FRED_COUNTRIES: dict[str, tuple[str, ...]] = {
    "yield_10y": ("AT", "AU", "BE", "CA", "CH", "CL", "CZ", "DE", "DK", "ES",
                  "FI", "FR", "GB", "GR", "HU", "IE", "IL", "IS", "IT", "JP",
                  "KR", "LU", "MX", "NL", "NO", "NZ", "PL", "PT", "RU", "SE",
                  "SI", "SK", "US", "ZA"),
    "rate_3m": ("AT", "AU", "BE", "CA", "CH", "CL", "CN", "CZ", "DE", "DK", "EE",
             "ES", "FI", "FR", "GB", "GR", "HU", "ID", "IE", "IL", "IS", "IT",
             "JP", "KR", "LU", "MX", "NL", "NO", "NZ", "PL", "PT", "RU", "SE",
             "SI", "SK", "TR", "US", "ZA"),
    "rate_on": ("AT", "AU", "BE", "BR", "CA", "CH", "CL", "CN", "CZ", "DE", "DK",
              "EE", "ES", "FI", "FR", "GB", "GR", "HU", "ID", "IE", "IL", "IN",
              "IS", "IT", "JP", "KR", "LU", "MX", "NL", "NO", "NZ", "PL", "PT",
              "RU", "SE", "SI", "SK", "TR", "US", "ZA"),
}

#: FRED id template per variable (``{cc}`` = ISO2).
_FRED_TEMPLATE: dict[str, str] = {
    "yield_10y": "IRLTLT01{cc}M156N",
    "rate_3m": "IR3TIB01{cc}M156N",
    "rate_on": "IRSTCI01{cc}M156N",
}

#: First month each euro member used the euro (ISO3 -> "YYYY-MM").
EURO_ENTRY: dict[str, str] = {
    "AUT": "1999-01", "BEL": "1999-01", "DEU": "1999-01", "ESP": "1999-01",
    "FIN": "1999-01", "FRA": "1999-01", "IRL": "1999-01", "ITA": "1999-01",
    "LUX": "1999-01", "NLD": "1999-01", "PRT": "1999-01", "GRC": "2001-01",
    "SVN": "2007-01", "CYP": "2008-01", "MLT": "2008-01", "SVK": "2009-01",
    "EST": "2011-01", "LVA": "2014-01", "LTU": "2015-01", "HRV": "2023-01",
    "BGR": "2026-01",
}

#: Variable name -> description (all in percent per annum, monthly averages).
RATE_VARIABLES: dict[str, str] = {
    "rate_on": "Overnight / immediate interbank rate (euro members: EONIA, then €STR from 2019-10)",
    "rate_3m": "3-month interbank rate (euro members: 3-month EURIBOR)",
    "yield_10y": "10-year government bond yield",
    "yield_corp_aaa": "Moody's seasoned Aaa corporate bond yield (USA only)",
    "yield_corp_baa": "Moody's seasoned Baa corporate bond yield (USA only)",
    "yield_corp_de": "Yield on corporate bonds outstanding, non-MFI issuers (DEU only, Bundesbank)",
}

#: ECB keys used by :func:`rates_panel` (flow, key); dimension orders in comments.
_ECB_KEYS: dict[str, tuple[str, str]] = {
    # FM: FREQ.REF_AREA.CURRENCY.PROVIDER_FM.INSTRUMENT_FM.PROVIDER_FM_ID.DATA_TYPE_FM
    "money": ("FM", "M.U2.EUR.RT+4F.MM.EURIBOR1MD_+EURIBOR3MD_+EONIA+UONSTR.HSTA"),
    # IRS: FREQ.REF_AREA.IR_TYPE.TR_TYPE.MATURITY_CAT.BS_COUNT_SECTOR.CURRENCY_TRANS.IR_BUS_COV.IR_FV_TYPE
    "irs_10y": ("IRS", "M..L.L40.CI.0000..N.Z"),
    "irs_3m": ("IRS", "M..M.L20.MC.0000..N.Z"),
    "long_10y": ("FM", "M.US+JP..4F.BB.US10YT_RR+JP10YT_RR.YLDA"),
    "stocks": ("FM", "M.US+JP+U2..DS.EI.S_PCOMP+JAPDOWA+DJES50I.HSTA"),
}

#: Overnight euro rate: EONIA until 2019-09, €STR from 2019-10 (when EONIA
#: became €STR plus a fixed 8.5 basis points).
_ESTR_FROM = "2019-10"

#: Bundesbank key of the German corporate bond yield (non-MFI issuers).
_BBK_CORP_DE = "M.I.UMR.RD.EUR.X2000.B.A.A.R.A.A._Z._Z.A"

#: The 28 countries of the ECB convergence-yield dataset (``IRS``, 10 years).
IRS_10Y_COUNTRIES: tuple[str, ...] = (
    "AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST", "FIN",
    "FRA", "GBR", "GRC", "HRV", "HUN", "IRL", "ITA", "LTU", "LUX", "LVA",
    "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "SWE",
)

#: Economies that only the ECB 3-month IRS leg covers or completes.
_IRS_3M = ("CZE", "HUN", "POL", "ROU")

#: Stock-index codes with an ECB FM monthly-average back-history.
_ECB_STOCK_AREA: dict[str, tuple[str, str]] = {
    "USA": ("US", "S_PCOMP"), "JPN": ("JP", "JAPDOWA"), "EMU": ("U2", "DJES50I"),
}

#: Pink Sheet raw price names -> panel column names. Every commodity column
#: carries the ``commodity_`` prefix the course's series registry reserves
#: for ``t01_commodities.csv.gz``; the indices become ``commodity_index_*``.
_PINK_NAMES: dict[str, str] = {
    "oil_avg_m": "commodity_oil_avg", "oil_brent_m": "commodity_brent", "oil_dubai_m": "commodity_dubai",
    "oil_wti_m": "commodity_wti", "coal_au_m": "commodity_coal_au", "coal_za_m": "commodity_coal_za",
    "gas_us_m": "commodity_natgas_us", "gas_eu_m": "commodity_natgas_eu", "gas_jp_lng_m": "commodity_lng_jp",
    "copper_m": "commodity_copper", "aluminum_m": "commodity_aluminum", "iron_ore_m": "commodity_iron_ore",
    "lead_m": "commodity_lead", "tin_m": "commodity_tin", "nickel_m": "commodity_nickel", "zinc_m": "commodity_zinc",
    "gold_m": "commodity_gold", "silver_m": "commodity_silver", "platinum_m": "commodity_platinum",
    "wheat_hrw_m": "commodity_wheat", "wheat_srw_m": "commodity_wheat_srw", "maize_m": "commodity_maize",
    "rice_m": "commodity_rice", "soybeans_m": "commodity_soybeans", "soybean_oil_m": "commodity_soybean_oil",
    "soybean_meal_m": "commodity_soybean_meal", "barley_m": "commodity_barley",
    "phosphate_rock_m": "commodity_phosphate_rock", "dap_m": "commodity_dap", "tsp_m": "commodity_tsp",
    "urea_m": "commodity_urea", "potassium_chloride_m": "commodity_potassium_chloride",
}


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _get(url: str, *, timeout: float = _TIMEOUT, pause: float = 1.0,
         refresh: bool = False) -> bytes:
    """Fetch ``url`` through the house cache; raises on any failure.

    The single network seam of the module: tests patch this name.
    """
    return safe_get_bytes_cached(url, timeout, headers=_HEADERS,
                                 rate_limit_seconds=pause, refresh=refresh)


def _fetch(url: str, *, timeout: float = _TIMEOUT, pause: float = 1.0,
           refresh: bool = False) -> tuple[bytes | None, str]:
    """``(body, "ok")`` or ``(None, reason)``; never raises."""
    try:
        return _get(url, timeout=timeout, pause=pause, refresh=refresh), "ok"
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except TimeoutError:
        return None, "timeout"
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        if isinstance(reason, TimeoutError):
            return None, "timeout"
        return None, f"URLError: {reason}"
    except Exception as e:  # noqa: BLE001 - provider failure must not raise
        return None, f"{type(e).__name__}: {e}"


def _fetch_parsed(url: str, parse, *, timeout: float = _TIMEOUT, pause: float = 1.0,
                  refresh: bool = False, retry_if=None):
    """Fetch ``url`` and ``parse`` the body -> ``(value, reason, body_meta)``; never raises.

    ``parse(body)`` returns ``(value, reason)`` with ``value is None`` on a
    body it rejects. The house cache keeps every 200 response for 30 days,
    including an HTML error page or a JSON error object, so a body read from
    the cache that fails to parse is downloaded once more with
    ``refresh=True`` before the failure is reported. ``retry_if(reason)``
    (default: always) limits which parse failures earn that second request.
    """
    body, status = _fetch(url, timeout=timeout, pause=pause, refresh=refresh)
    if body is None:
        return None, status
    value, reason = parse(body)
    if value is None and not refresh and (retry_if is None or retry_if(reason)):
        body2, status2 = _fetch(url, timeout=timeout, pause=pause, refresh=True)
        if body2 is None:
            return None, status2
        value, reason = parse(body2)
    return value, reason


def _failed_summary(failed: Sequence[dict], limit: int = 10) -> str:
    """``"DEU/yield_10y (HTTP 429), ..."`` for the first ``limit`` failures."""
    items = [f"{m.get('code', m.get('id', '?'))}/{m.get('variable', '')} ({m['reason']})".replace("/ (", " (")
             for m in failed[:limit]]
    more = f" and {len(failed) - limit} more" if len(failed) > limit else ""
    return ", ".join(items) + more


def _month_start(x: object, what: str) -> pd.Timestamp:
    """``"1950"``, ``"1950-03"``, a date or Timestamp -> month start; else ValueError."""
    try:
        ts = pd.Timestamp(str(x)) if isinstance(x, (int, np.integer)) else pd.Timestamp(x)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{what} must be a year or a date such as '1950' or "
                         f"'1950-01', got {x!r}") from e
    if pd.isna(ts):
        raise ValueError(f"{what} must be a year or a date, got {x!r}")
    return ts.to_period("M").to_timestamp()


def _check_codes(codes: Iterable[str] | str | None, *, allow: frozenset[str] = frozenset()) -> list[str] | None:
    """Upper-cased ISO3 codes; ValueError on a malformed code or an aggregate."""
    if codes is None:
        return None
    if isinstance(codes, str):
        codes = [codes]
    out: list[str] = []
    for c in codes:
        cc = str(c).strip().upper()
        if cc in allow:
            out.append(cc)
            continue
        if not re.fullmatch(r"[A-Z]{3}", cc):
            raise ValueError(f"codes must be ISO3 country codes such as 'DEU', got {c!r}")
        if cc in IS_AGGREGATE:
            raise ValueError(f"{cc!r} is an aggregate; this function returns countries")
        if cc not in out:
            out.append(cc)
    if not out:
        raise ValueError("codes is empty; pass None for every covered country")
    return out


def _empty(columns: Sequence[str], *, source: str, missing: Sequence[dict] = (),
           complete: bool = False) -> pd.DataFrame:
    idx = pd.MultiIndex.from_arrays(
        [pd.Index([], dtype=object), pd.DatetimeIndex([], dtype="datetime64[ns]")],
        names=["code", "date"])
    out = pd.DataFrame({c: pd.Series([], dtype=float) for c in columns}, index=idx)
    out.attrs.update(meta=(), source=source, fetched_at=_now(), missing=tuple(missing),
                     complete=complete)
    return out


def _leg(code: str, variable: str, s: pd.Series, *, source: str, key: str,
         role: str, units: str = _PCT, url: str | None = None, **extra) -> dict:
    s = s.dropna()
    d = {
        "code": code, "variable": variable, "source": source, "key": key,
        "role": role, "units": units, "unit_mult": 0, "sa": "NSA",
        "first": s.index.min().strftime("%Y-%m") if len(s) else None,
        "last": s.index.max().strftime("%Y-%m") if len(s) else None,
        "n": int(len(s)),
    }
    if url:
        d["url"] = url
    d.update(extra)
    return d


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------

def _parse_fred_csv(body: bytes) -> pd.Series | None:
    """``observation_date,<ID>`` CSV -> float Series on a DatetimeIndex."""
    text = body.decode("utf-8", errors="ignore").lstrip("\ufeff")
    first = text.split("\n", 1)[0].strip().lower()
    if not (first.startswith("observation_date") or first.startswith("date")):
        return None
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] < 2:
        return None
    idx = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s = pd.Series(vals.to_numpy(dtype=float), index=pd.DatetimeIndex(idx, name="date"))
    s = s[s.index.notna()].dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _fred_series(sid: str, *, refresh: bool, pause: float) -> tuple[pd.Series | None, str, str]:
    url = _FRED_URL.format(sid=urllib.parse.quote(sid))

    def parse(body: bytes):
        s = _parse_fred_csv(body)
        if s is None:
            return None, "unparseable response (not a FRED CSV)"
        if s.empty:
            return None, "no observations"
        return s, "ok"

    s, status = _fetch_parsed(url, parse, pause=pause, refresh=refresh)
    return s, status, url


def fetch_fred_many(ids: Sequence[str] | str, *, start: str | None = None,
                    refresh: bool = False, pause: float = 1.0) -> pd.DataFrame:
    """Several FRED series as one wide frame, one request per id.

    FRED's public ``fredgraph.csv`` accepts a comma-separated list of ids but
    answers 404 for the whole list when a single id does not exist, so each
    id is requested on its own (cached, ``pause`` seconds apart) and a bad id
    costs only itself.

    Parameters
    ----------
    ids : sequence of str or str
        FRED series ids, e.g. ``["AAA", "BAA", "IRLTLT01DEM156N"]``.
    start : str, optional
        Keep observations on or after this date (``"1950"``, ``"1990-01"``).
        The full history is always downloaded so the cache serves any start.
    refresh : bool, default False
        Bypass the on-disk cache and re-download.
    pause : float, default 1.0
        Seconds between requests to FRED.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``date`` (as published: month start for monthly series,
        days for daily ones), one float column per id that answered, in the
        order asked. ``attrs``: ``meta`` (one dict per id: ``id``, ``url``,
        ``first``, ``last``, ``n``), ``missing`` (one dict per id that failed,
        with the reason), ``source`` and ``fetched_at``. Never raises for a
        provider failure; ``ValueError`` for an empty or malformed id list.

    Examples
    --------
    >>> from puremacro.fetch.rates import fetch_fred_many
    >>> df = fetch_fred_many(["AAA", "BAA"], start="1950")  # doctest: +SKIP
    >>> (df["BAA"] - df["AAA"]).mean()                      # doctest: +SKIP
    """
    if isinstance(ids, str):
        ids = [ids]
    ids = [str(i).strip() for i in ids]
    if not ids:
        raise ValueError("ids is empty; pass FRED series ids such as ['AAA', 'BAA']")
    bad = [i for i in ids if not re.fullmatch(r"[A-Za-z0-9_.]+", i)]
    if bad:
        raise ValueError(f"malformed FRED ids: {bad}")
    lo = _month_start(start, "start") if start is not None else None
    cols: dict[str, pd.Series] = {}
    meta: list[dict] = []
    missing: list[dict] = []
    for sid in dict.fromkeys(ids):
        s, status, url = _fred_series(sid, refresh=refresh, pause=pause)
        if s is None:
            missing.append({"id": sid, "reason": status, "url": url})
            continue
        if lo is not None:
            s = s[s.index >= lo]
        cols[sid] = s
        meta.append({"id": sid, "url": url,
                     "first": s.index.min().strftime("%Y-%m-%d") if len(s) else None,
                     "last": s.index.max().strftime("%Y-%m-%d") if len(s) else None,
                     "n": int(len(s))})
    if missing:
        warnings.warn("fetch_fred_many: no data for "
                      + ", ".join(m["id"] + " (" + m["reason"] + ")" for m in missing),
                      UserWarning, stacklevel=2)
    if cols:
        out = pd.concat(cols, axis=1).sort_index()
        out.index.name = "date"
    else:
        out = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    out.attrs.update(meta=tuple(meta), missing=tuple(missing),
                     source="FRED fredgraph.csv (fred.stlouisfed.org)", fetched_at=_now())
    return out


# ---------------------------------------------------------------------------
# ECB
# ---------------------------------------------------------------------------

def ecb_get(flow: str, key: str, *, start: str | None = None, end: str | None = None,
            last_n: int | None = None, refresh: bool = False, timeout: float = 120,
            pause: float = 2.0) -> pd.DataFrame:
    """One ECB Data Portal query as a raw SDMX-CSV frame.

    Calls ``https://data-api.ecb.europa.eu/service/data/{flow}/{key}`` with
    ``format=csvdata&detail=dataonly``: one row per observation, columns
    ``KEY``, the flow's dimensions, ``TIME_PERIOD`` (text, ``"YYYY-MM"`` for
    monthly series) and ``OBS_VALUE`` (float). ``+`` ORs codes inside a
    dimension and an empty position is a wildcard, e.g.
    ``ecb_get("IRS", "M..L.L40.CI.0000..N.Z")`` returns the 10-year
    convergence yields of every EU country in one request.

    Parameters
    ----------
    flow, key : str
        Dataflow id (``"FM"``, ``"IRS"``, ``"MIR"``, ``"BSI"``, ...) and series key.
    start, end : str, optional
        ``startPeriod`` / ``endPeriod`` (``"1999"``, ``"1999-01"``).
    last_n : int, optional
        ``lastNObservations``.
    refresh : bool, default False
        Bypass the on-disk cache.
    timeout : float, default 120
        Seconds before the request is abandoned.
    pause : float, default 2.0
        Seconds between requests to the ECB host.

    Returns
    -------
    pandas.DataFrame
        The rows as published, aggregates included. ``attrs``: ``status``
        (``"ok"``, ``"HTTP 404"`` when the key matches nothing, ``"timeout"``,
        ...), ``url``, ``source``, ``fetched_at``. On failure the frame is
        empty and a warning is issued; nothing is raised except
        ``ValueError`` for a malformed argument.

    Examples
    --------
    >>> from puremacro.fetch.rates import ecb_get
    >>> eur = ecb_get("FM", "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA")  # doctest: +SKIP
    """
    flow = str(flow).strip()
    key = str(key).strip()
    if not re.fullmatch(r"[A-Za-z0-9_]+", flow):
        raise ValueError(f"flow must be an ECB dataflow id such as 'FM' or 'IRS', got {flow!r}")
    if not key or "/" in key or "?" in key:
        raise ValueError(f"key must be an SDMX series key such as 'M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA', got {key!r}")
    if last_n is not None and (not isinstance(last_n, (int, np.integer)) or last_n < 1):
        raise ValueError(f"last_n must be a positive integer, got {last_n!r}")
    q = {"format": "csvdata", "detail": "dataonly"}
    if start is not None:
        q["startPeriod"] = str(start)
    if end is not None:
        q["endPeriod"] = str(end)
    if last_n is not None:
        q["lastNObservations"] = str(int(last_n))
    url = _ECB_URL.format(flow=flow, key=key) + "?" + urllib.parse.urlencode(q)

    def parse(body: bytes):
        text = body.decode("utf-8", errors="ignore").lstrip("\ufeff")
        if not text.strip():
            return None, "no data"
        if not text.startswith("KEY,"):
            return None, "unparseable response (not SDMX-CSV)"
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        df["OBS_VALUE"] = pd.to_numeric(df.get("OBS_VALUE"), errors="coerce")
        return df, "ok"

    out, status = _fetch_parsed(url, parse, timeout=timeout, pause=pause, refresh=refresh)
    if out is None:
        warnings.warn(f"ecb_get: {flow}/{key} returned no data ({status})",
                      UserWarning, stacklevel=2)
        out = pd.DataFrame(columns=["KEY", "TIME_PERIOD", "OBS_VALUE"])
    out.attrs.update(status=status, url=url, fetched_at=_now(),
                     source="ECB Data Portal (data-api.ecb.europa.eu)")
    return out


def _ecb_series(raw: pd.DataFrame) -> dict[str, pd.Series]:
    """KEY -> monthly float Series (month-start index) from an ``ecb_get`` frame."""
    out: dict[str, pd.Series] = {}
    if raw.empty or "TIME_PERIOD" not in raw:
        return out
    for key, g in raw.groupby("KEY", sort=True):
        per = pd.PeriodIndex(g["TIME_PERIOD"].astype(str), freq="M")
        s = pd.Series(g["OBS_VALUE"].to_numpy(dtype=float), index=per.to_timestamp(), name=key)
        s.index.name = "date"
        out[str(key)] = s.dropna().sort_index()
    return out


def _ecb_quiet(flow: str, key: str, *, refresh: bool) -> pd.DataFrame:
    """``ecb_get`` whose warning is folded into the caller's single summary."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return ecb_get(flow, key, refresh=refresh)


def _ecb_by_area(raw: pd.DataFrame) -> dict[str, tuple[pd.Series, str]]:
    """ISO3 -> (Series, series keys without the flow) from an IRS frame; aggregates dropped.

    An area reported in two currencies (Bulgaria BGN then EUR, Romania ROL
    then RON) is merged with the series that runs latest, i.e. the current
    currency, winning wherever both report.
    """
    out: dict[str, tuple[pd.Series, str]] = {}
    if raw.empty or "REF_AREA" not in raw:
        return out
    series = _ecb_series(raw)
    by_area: dict[str, list[str]] = {}
    for key, area in raw[["KEY", "REF_AREA"]].drop_duplicates("KEY").itertuples(index=False):
        area = str(area)
        if area in ECB_AGGREGATES or area not in ISO2_TO_ISO3 or key not in series:
            continue
        by_area.setdefault(area, []).append(str(key))
    for area, keys in by_area.items():
        keys.sort(key=lambda k: (series[k].index.max() if len(series[k]) else pd.Timestamp(0), k),
                  reverse=True)
        s = None
        for k in keys:
            s = series[k] if s is None else s.combine_first(series[k])
        # "IRS.M.BG.L..." -> "M.BG.L...": the flow is named once by the caller
        out[ISO2_TO_ISO3[area]] = (s, " + ".join(k.split(".", 1)[1] for k in keys))
    return out


# ---------------------------------------------------------------------------
# rates_panel
# ---------------------------------------------------------------------------

def _universe(variable: str) -> list[str]:
    """Every ISO3 economy some leg covers for ``variable``."""
    if variable in _FRED_TEMPLATE:
        u = {ISO2_TO_ISO3[c] for c in FRED_COUNTRIES[variable]}
        if variable in ("rate_3m", "rate_on"):
            u |= set(EURO_ENTRY)
        if variable == "rate_3m":
            u |= set(_IRS_3M)
        if variable == "yield_10y":
            u |= set(IRS_10Y_COUNTRIES)
        return sorted(u)
    return {"yield_corp_aaa": ["USA"], "yield_corp_baa": ["USA"], "yield_corp_de": ["DEU"]}[variable]


def _parse_bbk_csv(body: bytes) -> pd.Series | None:
    """Bundesbank wide CSV (metadata rows, then ``YYYY-MM,value,flag``)."""
    text = body.decode("utf-8", errors="ignore").lstrip("\ufeff")
    dates, vals = [], []
    for line in text.splitlines():
        m = re.match(r"^\"?(\d{4}-\d{2})\"?,\"?([^,\"]*)\"?", line)
        if m:
            dates.append(m.group(1))
            vals.append(m.group(2))
    if not dates:
        return None
    s = pd.Series(pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(dtype=float),
                  index=pd.PeriodIndex(dates, freq="M").to_timestamp())
    s.index.name = "date"
    return s.dropna().sort_index()


def rates_panel(codes: Iterable[str] | str | None = None, *, start: str = "1950",
                variables: Sequence[str] = ("rate_on", "rate_3m", "yield_10y", "yield_corp_aaa", "yield_corp_baa"),
                refresh: bool = False) -> pd.DataFrame:
    """Monthly interest rates and bond yields by country, longest free history.

    Parameters
    ----------
    codes : iterable of str, optional
        ISO3 codes. ``None`` (default) means every economy some source covers
        for the requested variables (47 for the defaults). A code with no source for a
        variable is listed in ``attrs["missing"]``, not an error.
    start : str, default ``"1950"``
        First month kept (``"1950"``, ``"1990-01"``). Full histories are
        always downloaded so the cache serves any start.
    variables : sequence of str
        Any of :data:`RATE_VARIABLES`: ``rate_on``, ``rate_3m``, ``yield_10y``,
        ``yield_corp_aaa``, ``yield_corp_baa``, ``yield_corp_de``.
    refresh : bool, default False
        Bypass the on-disk cache and re-download every series.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(code, date)``, ISO3 codes and month-start dates, one
        float column per variable in percent per annum (monthly averages,
        NSA). Rows where every variable is missing are dropped.

        Sources, in order of preference:

        * ``rate_on``, ``rate_3m``: FRED ``IRSTCI01{CC}M156N`` /
          ``IR3TIB01{CC}M156N``; for euro members from their entry month
          (:data:`EURO_ENTRY`) the ECB's EONIA (€STR from 2019-10) and
          3-month EURIBOR; Czechia, Hungary, Poland and Romania completed
          with the ECB ``IRS`` 3-month rate.
        * ``yield_10y``: FRED ``IRLTLT01{CC}M156N``; ECB ``IRS`` 10-year
          convergence yields fill EU countries and months FRED lacks; ECB FM
          benchmark histories extend the United States (1900) and Japan (1972).
        * ``yield_corp_aaa``, ``yield_corp_baa``: FRED ``AAA`` / ``BAA`` (USA, 1919).
        * ``yield_corp_de``: Bundesbank ``BBSIS`` X2000 (DEU, 1957).

        ``attrs["meta"]``: one dict per country, variable and source leg used
        (``source``, ``key``, ``role``, ``units``, ``unit_mult``, ``sa``,
        ``first``, ``last``, ``n`` = observations that leg contributes).
        ``attrs["missing"]``: dicts ``{code, variable, reason}`` for every
        series asked for and not obtained. ``attrs["complete"]``: ``False``
        when some series a provider serves failed to download this time (a
        throttle, a timeout); ``missing`` entries whose reason is ``"no free
        source covers this country"`` do not count. When FRED fails for an
        EU country the ECB leg may still serve it with a shorter history, so
        check ``complete`` before freezing the panel. Also ``source`` and
        ``fetched_at``.

    Raises
    ------
    ValueError
        Unknown variable, malformed code or ``start``.

    Examples
    --------
    >>> from puremacro.fetch.rates import rates_panel
    >>> df = rates_panel(["USA", "DEU", "MEX"], start="1960")      # doctest: +SKIP
    >>> df.loc["DEU", "yield_10y"].dropna().index[0]                # doctest: +SKIP
    Timestamp('1960-01-01 00:00:00')
    """
    if isinstance(variables, str):
        variables = (variables,)
    variables = tuple(dict.fromkeys(variables))
    unknown = [v for v in variables if v not in RATE_VARIABLES]
    if unknown or not variables:
        raise ValueError(f"unknown variables {unknown}; choose from {sorted(RATE_VARIABLES)}")
    lo = _month_start(start, "start")
    asked = _check_codes(codes)

    series: dict[tuple[str, str], pd.Series] = {}
    meta: list[dict] = []
    missing: list[dict] = []
    ecb_cache: dict[str, pd.DataFrame] = {}

    def ecb(name: str) -> pd.DataFrame:
        if name not in ecb_cache:
            ecb_cache[name] = _ecb_quiet(*_ECB_KEYS[name], refresh=refresh)
        return ecb_cache[name]

    def ecb_status(name: str) -> str:
        return str(ecb(name).attrs.get("status", "no data"))

    def add_leg(code: str, v: str, s: pd.Series | None, **leg) -> None:
        """Combine ``s`` under what is there; record what it contributed."""
        if s is None or s.empty:
            return
        s = s[s.index >= lo]
        cur = series.get((code, v))
        contributed = s if cur is None else s[~s.index.isin(cur.dropna().index)]
        if contributed.dropna().empty:
            return
        series[(code, v)] = s if cur is None else cur.combine_first(s)
        meta.append(_leg(code, v, contributed, **leg))

    # --- FRED national legs (rate_on, rate_3m, yield_10y) ----------------------
    for v in ("rate_on", "rate_3m", "yield_10y"):
        if v not in variables:
            continue
        # only ids known to exist: any other answers 404 and costs a request
        for cc in FRED_COUNTRIES[v]:
            code = ISO2_TO_ISO3[cc]
            if cc in FRED_AGGREGATES or (asked is not None and code not in asked):
                continue
            sid = _FRED_TEMPLATE[v].format(cc=cc)
            s, status, url = _fred_series(sid, refresh=refresh, pause=1.0)
            if s is None:
                missing.append({"code": code, "variable": v, "reason": f"FRED {sid}: {status}"})
                continue
            s = s.copy()
            s.index = s.index.to_period("M").to_timestamp()
            entry = EURO_ENTRY.get(code)
            if entry is not None and v in ("rate_on", "rate_3m"):
                s = s[s.index < pd.Timestamp(entry)]
                role = "national rate before euro entry"
            else:
                role = "primary"
            add_leg(code, v, s, source="FRED", key=sid, role=role, url=url)

    # --- euro members from entry: ECB money-market rates ------------------
    if "rate_on" in variables or "rate_3m" in variables:
        members = [c for c in EURO_ENTRY
                   if (asked is None or c in asked)]
        need = [(v, c) for v in ("rate_on", "rate_3m") if v in variables for c in members]
        if need:
            mm = _ecb_series(ecb("money"))
            k_e3 = "FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA"
            k_eo = "FM.M.U2.EUR.4F.MM.EONIA.HSTA"
            k_es = "FM.M.U2.EUR.4F.MM.UONSTR.HSTA"
            on = None
            if k_eo in mm or k_es in mm:
                eo = mm.get(k_eo, pd.Series(dtype=float))
                es = mm.get(k_es, pd.Series(dtype=float))
                cut = pd.Timestamp(_ESTR_FROM)
                on = pd.concat([eo[eo.index < cut], es[es.index >= cut]]).sort_index()
            for v, code in need:
                entry = pd.Timestamp(EURO_ENTRY[code])
                if v == "rate_3m":
                    s, key = mm.get(k_e3), "FM/" + k_e3[3:]
                    label = "3-month EURIBOR from euro entry"
                else:
                    s, key = on, "FM/M.U2.EUR.4F.MM.EONIA.HSTA to 2019-09; FM/M.U2.EUR.4F.MM.UONSTR.HSTA from 2019-10"
                    label = "EONIA / €STR from euro entry"
                if s is None or s.empty:
                    missing.append({"code": code, "variable": v,
                                    "reason": "ECB FM euro money-market rates: " + ecb_status("money")})
                    continue
                add_leg(code, v, s[s.index >= entry], source="ECB", key=key, role=label)

    # --- rate_3m: ECB IRS 3-month for CZ HU PL RO ----------------------------
    if "rate_3m" in variables:
        targets = [c for c in _IRS_3M if asked is None or c in asked]
        if targets:
            by = _ecb_by_area(ecb("irs_3m"))
            for code in targets:
                if code in by:
                    s, key = by[code]
                    add_leg(code, "rate_3m", s, source="ECB", key="IRS/" + key,
                            role="primary" if (code, "rate_3m") not in series else "second source")
                elif (code, "rate_3m") not in series:
                    missing.append({"code": code, "variable": "rate_3m",
                                    "reason": "ECB IRS 3-month: " + ecb_status("irs_3m")})

    # --- yield_10y: ECB IRS for EU countries, FM back-histories -----------
    if "yield_10y" in variables:
        targets = [c for c in IRS_10Y_COUNTRIES if asked is None or c in asked]
        if targets:
            by = _ecb_by_area(ecb("irs_10y"))
            for code in targets:
                if code in by:
                    s, key = by[code]
                    add_leg(code, "yield_10y", s, source="ECB",
                            key="IRS/" + key,
                            role="primary" if (code, "yield_10y") not in series else "second source")
                elif (code, "yield_10y") not in series:
                    missing.append({"code": code, "variable": "yield_10y",
                                    "reason": "ECB IRS 10-year: " + ecb_status("irs_10y")})
        long_codes = [c for c in ("USA", "JPN") if asked is None or c in asked]
        if long_codes:
            ls = _ecb_series(ecb("long_10y"))
            for code, k in (("USA", "FM.M.US.USD.4F.BB.US10YT_RR.YLDA"),
                            ("JPN", "FM.M.JP.JPY.4F.BB.JP10YT_RR.YLDA")):
                if code in long_codes and k in ls:
                    # only before the national series starts: a back-history,
                    # not a competitor for recent months
                    cur = series.get((code, "yield_10y"))
                    back = ls[k] if cur is None else ls[k][ls[k].index < cur.dropna().index.min()]
                    add_leg(code, "yield_10y", back, source="ECB", key="FM/" + k[3:],
                            role="back-history (10-year benchmark)")

    # --- corporate yields -------------------------------------------------
    for v, sid in (("yield_corp_aaa", "AAA"), ("yield_corp_baa", "BAA")):
        if v in variables and (asked is None or "USA" in asked):
            s, status, url = _fred_series(sid, refresh=refresh, pause=1.0)
            if s is None:
                missing.append({"code": "USA", "variable": v, "reason": f"FRED {sid}: {status}"})
            else:
                s = s.copy()
                s.index = s.index.to_period("M").to_timestamp()
                add_leg("USA", v, s, source="FRED", key=sid, role="primary", url=url)
    if "yield_corp_de" in variables and (asked is None or "DEU" in asked):
        url = _BBK_URL.format(key=_BBK_CORP_DE)

        def parse_bbk(body: bytes):
            s = _parse_bbk_csv(body)
            return (s, "ok") if s is not None and not s.empty else (None, "unparseable response")

        s, status = _fetch_parsed(url, parse_bbk, pause=1.0, refresh=refresh)
        if s is None:
            missing.append({"code": "DEU", "variable": "yield_corp_de",
                            "reason": "Bundesbank BBSIS " + _BBK_CORP_DE + ": " + status})
        else:
            add_leg("DEU", "yield_corp_de", s, source="Bundesbank", key="BBSIS/" + _BBK_CORP_DE,
                    role="primary", url=url)

    # --- codes asked for with no source at all -----------------------------
    if asked is not None:
        got = {(c, v) for (c, v) in series}
        noted = {(m["code"], m["variable"]) for m in missing}
        for v in variables:
            for c in asked:
                if (c, v) not in got and (c, v) not in noted:
                    missing.append({"code": c, "variable": v, "reason": "no free source covers this country"})

    failed = [m for m in missing if m["reason"] != "no free source covers this country"]
    if not series:
        if missing:
            warnings.warn(f"rates_panel: nothing obtained; {len(missing)} series missing: "
                          + _failed_summary(missing) + " (see attrs['missing'])",
                          UserWarning, stacklevel=2)
        return _empty(variables, source=_SOURCE_RATES, missing=missing, complete=not failed)

    long = pd.concat({k: v for k, v in series.items()}, names=["code", "variable", "date"])
    out = long.unstack("variable")
    out = out.reindex(columns=list(variables)).astype(float)
    out = out.dropna(how="all").sort_index()
    out.index = out.index.set_names(["code", "date"])
    out.columns.name = None
    if failed:
        warnings.warn(f"rates_panel: {len(failed)} series could not be fetched, so the "
                      "panel is incomplete (attrs['complete'] is False): "
                      + _failed_summary(failed) + "; see attrs['missing']",
                      UserWarning, stacklevel=2)
    meta.sort(key=lambda d: (d["code"], d["variable"], d["first"] or ""))
    out.attrs.update(meta=tuple(meta), source=_SOURCE_RATES, fetched_at=_now(),
                     missing=tuple(missing), complete=not failed)
    return out


# ---------------------------------------------------------------------------
# Stock indices
# ---------------------------------------------------------------------------

def _yahoo_monthly(body: bytes) -> tuple[pd.Series | None, dict, str]:
    """Monthly closes (month-start index, exchange-local month) from chart JSON."""
    try:
        doc = json.loads(body.decode("utf-8", errors="ignore"))
    except ValueError:
        return None, {}, "unparseable response (not JSON)"
    chart = doc.get("chart") or {}
    if chart.get("error"):
        err = chart["error"]
        return None, {}, "Yahoo error: " + str(err.get("description") or err.get("code") or err)
    res = (chart.get("result") or [None])[0]
    if not res or not res.get("timestamp"):
        return None, {}, "no observations"
    m = res.get("meta") or {}
    gran = m.get("dataGranularity")
    if gran not in (None, "1mo"):
        return None, m, f"Yahoo returned {gran} bars, not monthly"
    quote = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    close = pd.to_numeric(pd.Series(quote.get("close") or [], dtype=object), errors="coerce")
    ts = pd.to_datetime(pd.Series(res["timestamp"][: len(close)]), unit="s", utc=True)
    # A bar is stamped at the exchange's local midnight: the Nikkei's January
    # bar is 31 Dec 15:00 UTC, so the month must be read in local time.
    tz = m.get("exchangeTimezoneName")
    try:
        local = ts.dt.tz_convert(tz).dt.tz_localize(None)
    except Exception:  # noqa: BLE001 - missing or unknown zone: use the offset
        local = ts.dt.tz_localize(None) + pd.to_timedelta(int(m.get("gmtoffset") or 0), unit="s")
    months = pd.PeriodIndex(local.dt.to_period("M"))
    s = pd.Series(close.to_numpy(dtype=float), index=months.to_timestamp())
    s = s[~s.index.duplicated(keep="last")].dropna().sort_index()
    this_month = pd.Timestamp(dt.datetime.now(dt.timezone.utc).date()).to_period("M").to_timestamp()
    s = s[s.index < this_month]          # the running month is not a month-end close
    s.index.name = "date"
    return (s if len(s) else None), m, ("ok" if len(s) else "no complete month")


def stock_index_monthly(codes: Iterable[str] | str | None = None, *, start: str | None = None,
                        refresh: bool = False) -> pd.DataFrame:
    """Headline stock index by country, monthly, from Yahoo Finance (plus ECB back-histories).

    Parameters
    ----------
    codes : iterable of str, optional
        ISO3 codes; ``None`` means every country in
        :data:`puremacro.fetch.yahoo.ISO3_TO_INDEX` (34). ``"EMU"`` (euro
        area: Euro Stoxx 50) is accepted when asked for explicitly. Countries
        whose ticker is dead (:data:`puremacro.fetch.yahoo.DEAD_TICKERS`) or
        that have none are listed in ``attrs["missing"]``.
    start : str, optional
        First month kept.
    refresh : bool, default False
        Bypass the on-disk cache.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(code, date)``, month-start dates, columns

        * ``stock_idx``: month-end close of the index in index points and
          local currency (Yahoo ``interval=1mo`` bars, requested with
          explicit ``period1``/``period2``: ``range=max`` silently returns
          *quarterly* bars for long histories). The running month is
          dropped. Yahoo's monthly history starts in 1985 at the earliest.
        * ``stock_idx_avg``: monthly *average* of the same index from the
          ECB (FM flow) for USA (S&P 500, 1964), JPN (Nikkei 225, 1972) and
          EMU (Euro Stoxx 50, 1986); NaN elsewhere. A different statistic
          from the month-end close, so the two columns are not spliced.

        ``attrs["meta"]`` has one dict per country and column (``source``,
        ``key`` = ticker or ECB key, ``currency``, ``units``, ``first``,
        ``last``, ``n``); ``attrs["missing"]``, ``attrs["complete"]``
        (``False`` when a live ticker or the ECB leg failed this time; dead
        tickers and countries without one do not count), ``source``,
        ``fetched_at``.

        ``stock_idx`` is deliberately not called ``share_price``: the OECD
        STES ``share_price`` (:mod:`puremacro.fetch.oecd_stes_panel`) is an
        index with 2015 = 100 built from monthly averages, while
        ``stock_idx`` is the level of one named index at month end. Rebase
        before comparing them.

    Examples
    --------
    >>> from puremacro.fetch.rates import stock_index_monthly
    >>> px = stock_index_monthly(["USA", "MEX", "JPN"])   # doctest: +SKIP
    """
    from puremacro.fetch.yahoo import DEAD_TICKERS, ISO3_TO_INDEX

    tickers = dict(ISO3_TO_INDEX)
    tickers["EMU"] = "^STOXX50E"
    asked = _check_codes(codes, allow=frozenset({"EMU"}))
    lo = _month_start(start, "start") if start is not None else None
    todo = list(ISO3_TO_INDEX) if asked is None else asked
    period2 = int(pd.Timestamp(dt.datetime.now(dt.timezone.utc).date()).to_period("M")
                  .to_timestamp().timestamp()) + 2 * 86400   # stable within a month -> cacheable
    cols: dict[tuple[str, str], pd.Series] = {}
    meta: list[dict] = []
    missing: list[dict] = []
    for code in todo:
        tk = tickers.get(code)
        if tk is None:
            reason = (f"Yahoo ticker {DEAD_TICKERS[code]} no longer serves a history"
                      if code in DEAD_TICKERS else "no Yahoo ticker for this country")
            missing.append({"code": code, "variable": "stock_idx", "reason": reason})
            continue
        url = _YAHOO_URL.format(ticker=urllib.parse.quote(tk, safe=""), period2=period2)
        box: dict = {}

        def parse_yahoo(body: bytes, box=box):
            s_, m_, why_ = _yahoo_monthly(body)
            box["meta"] = m_
            return s_, why_

        s, why = _fetch_parsed(url, parse_yahoo, pause=1.0, refresh=refresh,
                               retry_if=lambda r: r != "no complete month")
        m = box.get("meta", {})
        if s is None:
            missing.append({"code": code, "variable": "stock_idx", "reason": f"Yahoo {tk}: {why}"})
            continue
        if lo is not None:
            s = s[s.index >= lo]
        cols[(code, "stock_idx")] = s
        meta.append(_leg(code, "stock_idx", s, source="Yahoo Finance", key=tk,
                         role="month-end close", units="index points",
                         currency=m.get("currency"), url=url))

    ecb_codes = [c for c in _ECB_STOCK_AREA if (asked is not None and c in asked)
                 or (asked is None and c != "EMU")]
    if ecb_codes:
        raw = _ecb_quiet(*_ECB_KEYS["stocks"], refresh=refresh)
        by = _ecb_series(raw)
        for code in ecb_codes:
            area, inst = _ECB_STOCK_AREA[code]
            k = next((k for k in by if k.startswith(f"FM.M.{area}.") and f".{inst}." in k), None)
            if k is None:
                missing.append({"code": code, "variable": "stock_idx_avg",
                                "reason": "ECB FM: " + str(raw.attrs.get("status"))})
                continue
            s = by[k]
            if lo is not None:
                s = s[s.index >= lo]
            cols[(code, "stock_idx_avg")] = s
            meta.append(_leg(code, "stock_idx_avg", s, source="ECB", key="FM/" + k[3:],
                             role="monthly average", units="index points",
                             currency=k.split(".")[3]))

    columns = ["stock_idx", "stock_idx_avg"]
    failed = [m for m in missing
              if "no Yahoo ticker" not in m["reason"] and "no longer serves" not in m["reason"]]
    if missing and any("no Yahoo ticker" not in m["reason"] for m in missing):
        warnings.warn(f"stock_index_monthly: {len(missing)} series missing: "
                      + _failed_summary(missing) + " (see attrs['missing'])",
                      UserWarning, stacklevel=2)
    if not cols:
        return _empty(columns, source=_SOURCE_STOCKS, missing=missing, complete=False)
    long = pd.concat(cols, names=["code", "variable", "date"])
    out = long.unstack("variable").reindex(columns=columns).astype(float)
    out = out.dropna(how="all").sort_index()
    out.index = out.index.set_names(["code", "date"])
    out.columns.name = None
    out.attrs.update(meta=tuple(meta), source=_SOURCE_STOCKS, fetched_at=_now(),
                     missing=tuple(missing), complete=not failed)
    return out


# ---------------------------------------------------------------------------
# Commodities
# ---------------------------------------------------------------------------

def commodity_prices_monthly(*, start: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """World Bank Pink Sheet monthly prices and indices as a wide ``WLD`` panel.

    Every price the workbook's 'Monthly Prices' sheet carries that the
    package maps (32: crude oil average, Brent, Dubai, WTI, coal, gas, LNG,
    metals, precious metals, grains, oilseeds, fertilisers) and the 16
    'Monthly Indices' (2010 = 100), read with the standard library only
    (:mod:`puremacro.fetch.wb_pink_sheet`, no ``openpyxl``).

    Parameters
    ----------
    start : str, optional
        First month kept; the workbook starts in 1960-01.
    refresh : bool, default False
        Re-read the landing page and re-download the workbook.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(code, date)`` with ``code == "WLD"`` and month-start
        dates; 48 float columns, all prefixed ``commodity_``: 32 prices in
        nominal US dollars per physical unit (``commodity_brent`` in
        ``$/bbl``, ``commodity_copper`` in ``$/mt``, ``commodity_natgas_us``
        in ``$/mmbtu``, ``commodity_gold`` in ``$/troy oz``, ...) and 16
        ``commodity_index_*`` columns (2010 = 100). ``attrs["meta"]`` has one
        dict per column with ``units`` from the workbook's units row. A
        failed download gives an empty frame with the same 48 columns, a
        warning, ``attrs["missing"]`` and ``attrs["complete"] = False``.

    Examples
    --------
    >>> from puremacro.fetch.rates import commodity_prices_monthly
    >>> cm = commodity_prices_monthly(start="1970")     # doctest: +SKIP
    >>> cm.loc["WLD", "commodity_brent"].tail()         # doctest: +SKIP
    """
    from puremacro.fetch import wb_pink_sheet as pk

    lo = _month_start(start, "start") if start is not None else None
    all_columns = list(dict.fromkeys(_PINK_NAMES.values())) + \
        ["commodity_" + v for v in pk._INDEX_COL_MAP.values()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        content = pk._get_workbook_content(refresh=refresh)
    if content is None or len(content) < 100:
        miss = ({"code": "WLD", "variable": "*", "reason": "Pink Sheet workbook could not be downloaded"},)
        warnings.warn("commodity_prices_monthly: the Pink Sheet workbook could not be downloaded",
                      UserWarning, stacklevel=2)
        return _empty(all_columns, source=_SOURCE_PINK, missing=miss, complete=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        prices = pk.fetch_prices(workbook_bytes=content)
        indices = pk.fetch_indices(workbook_bytes=content)
        try:
            units = pk._price_units(content)
        except Exception:  # noqa: BLE001 - units are a courtesy, never a failure
            units = {}
    pk._warn_if_stale(prices)
    long = pd.concat([prices, indices], ignore_index=True)
    if long.empty:
        miss = ({"code": "WLD", "variable": "*", "reason": "Pink Sheet workbook could not be parsed"},)
        warnings.warn("commodity_prices_monthly: the Pink Sheet workbook could not be parsed",
                      UserWarning, stacklevel=2)
        return _empty(all_columns, source=_SOURCE_PINK, missing=miss, complete=False)
    long["name"] = long["variable"].map(lambda v: _PINK_NAMES.get(v, "commodity_" + v))
    wide = long.pivot_table(index="date", columns="name", values="value", aggfunc="first")
    order = [c for c in all_columns if c in wide]
    wide = wide.reindex(columns=order).astype(float).sort_index()
    wide.index = pd.DatetimeIndex(wide.index).to_period("M").to_timestamp()
    if lo is not None:
        wide = wide[wide.index >= lo]
    wide.columns.name = None
    raw_of = {v: k for k, v in _PINK_NAMES.items()}
    meta = []
    for c in wide.columns:
        s = wide[c].dropna()
        is_idx = c.startswith("commodity_index_")
        meta.append({
            "code": "WLD", "variable": c, "source": "World Bank Pink Sheet",
            "key": ("Monthly Indices:" + c[len("commodity_"):] if is_idx
                    else "Monthly Prices:" + raw_of.get(c, c)),
            "units": "index, 2010=100" if is_idx else (units.get(raw_of.get(c, ""), "") or "nominal US$"),
            "unit_mult": 0, "sa": "NSA",
            "first": s.index.min().strftime("%Y-%m") if len(s) else None,
            "last": s.index.max().strftime("%Y-%m") if len(s) else None,
            "n": int(len(s)),
        })
    out = wide.copy()
    out.index = pd.MultiIndex.from_arrays([["WLD"] * len(wide), wide.index], names=["code", "date"])
    out.attrs.update(meta=tuple(meta), source=_SOURCE_PINK, fetched_at=_now(), missing=(),
                     complete=True)
    return out


__all__ = [
    "rates_panel",
    "stock_index_monthly",
    "commodity_prices_monthly",
    "fetch_fred_many",
    "ecb_get",
    "RATE_VARIABLES",
    "FRED_COUNTRIES",
    "EURO_ENTRY",
    "ISO2_TO_ISO3",
    "ECB_AGGREGATES",
    "IRS_10Y_COUNTRIES",
    "FRED_AGGREGATES",
]
