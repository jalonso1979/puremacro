r"""One-call ILOSTAT labour panel: 230 economies, annual, quarterly or monthly.

``ilostat_panel()`` returns the broadest cross-country labour-market panel a
free provider publishes: the ILO's harmonised labour-force-survey series for
every reporting economy, indexed by ``(code, date)`` with one column per
variable::

    emp  une  lf  wap  urate  prate  epop  emp_ees  emp_self  hours
    informal_rate  labour_share                                    (annual)
    emp_model  urate_model  lf_model  wap_model  emp_self_model
    hours_model                                     (``modelled=True``)

Why this module exists
----------------------
The OECD labour tables stop at some forty economies and the World Bank
republishes the ILO's *modelled* estimates only, from 1991. ILOSTAT itself
publishes what the national surveys measured: Mexico from 1988, the United
States from 1947 (quarterly and monthly from 1948), Spain from 1969, and
about 225 economies a year in the annual block. The package's older reader,
:func:`puremacro.fetch.fetch_ilostat_lfs_panel`, has no annual frequency,
reads through a 30-second timeout that an all-country pull exceeds, and
returns a long sex-by-age schema; this module is the one-call panel builder
in the spirit of :func:`puremacro.fetch.qna_panel`, and the older reader now
shares its transport and aggregate list.

Mechanics
---------
One request per dataflow, on the ILO's SDMX REST endpoint
``/rest/data/ILO,{flow},1.0/{areas}.{FREQ}.{MEASURE}.{SEX}.{AGE|STE}`` with
``format=csv``, the frequency and measure pinned in the key and the area
slot left open (every economy) or filled with ``codes``. Variables that share
a flow travel together: ``emp_ees`` and ``emp_self`` are one request with
``STE_AGGREGATE_EES+STE_AGGREGATE_SLF`` in the last slot. ``start`` and
``end`` become year-literal ``startPeriod``/``endPeriod`` parameters (the
provider's firewall answers 403 to ``2025-Q3``) and are applied exactly on
the parsed dates. Bodies travel through
:func:`puremacro._http.safe_get_bytes_cached` (urllib, SQLite cache outside
the repository, ``pause`` seconds between requests to the host); all-country
pulls take 3 to 50 seconds, hence the 300-second default ``timeout``.

Values are as published: persons in thousands, rates in percent, hours per
employed person per week. Every series is the 15-and-over, both-sexes total
(``SEX_T``, ``AGE_YTHADULT_YGE15``); ILOSTAT publishes it not seasonally
adjusted, quarterly values being three-month averages, except the monthly
rates of EU members, which ILOSTAT relays from Eurostat seasonally adjusted
(``SOURCE`` ``LFS-ADJ``, ages 15-74). ``attrs["meta"]`` says which, series by
series, and lists the observation flags: ``B`` marks a break in the series
(a new survey, a census re-weighting), ``U`` an unreliable value, ``I`` an
imputation, ``M`` a model extrapolation. Breaks are kept in the data and
reported, never spliced away.

The ILO's own aggregates (``X01``-``X99``, ``XA1``: world, regions, income
groups) are dropped by the explicit :data:`ILO_AGGREGATES` list, with the
codelist's pattern ``X``-digit-digit as a guard against codes the ILO adds
later (no ISO 3166 code contains a digit). Kosovo (``KOS``), the Channel
Islands (``CHA``) and the former Netherlands Antilles (``ANT``) keep the
ILO's codes. The modelled-estimate flows publish their last years as
projections, with no observation flag to tell them apart; how many differs
by flow (two for most ``*_model`` twins, five for ``wap_model``, one for
``labour_share``, none for ``emp_self_model``: the ``projected`` entry of
:data:`ILOSTAT_VARS`). Unless ``end`` is given those trailing years are cut,
counting back from the last year the flow itself publishes, not from today's
date, so a rebuild in January freezes the same panel. The cut is recorded in
``attrs["meta"]`` as ``trimmed_after``.

A failed request never raises: the variables it carried are listed in
``attrs["missing"]`` with the reason and the rest of the panel is returned.
``ValueError`` is reserved for caller errors: an unknown variable, a variable
the chosen frequency does not have, a bad frequency or code.

Source: International Labour Organization, ILOSTAT SDMX web service,
https://sdmx.ilo.org/rest/ (dataflow catalogue ``/rest/dataflow/ILO``).
"""
from __future__ import annotations

import datetime as dt
import http.client
import io
import re
import urllib.error
import warnings
import zlib
from typing import Iterable

import numpy as np
import pandas as pd


_BASE = "https://sdmx.ilo.org/rest"
_HEADERS = {"Accept-Encoding": "gzip"}
_SOURCE = "ILOSTAT SDMX web service (sdmx.ilo.org)"
_FREQS = ("A", "Q", "M")

# What a failed transfer can raise through the transport: HTTPError and
# URLError are OSErrors; a truncated body surfaces as IncompleteRead (an
# HTTPException) or, once gzip is undone, as EOFError or zlib.error.
_TRANSPORT_ERRORS = (OSError, http.client.HTTPException, EOFError, zlib.error)

_SEX_AGE = "SEX_T.AGE_YTHADULT_YGE15"

#: Registry of ILOSTAT panel variables: name -> dataflow, MEASURE code, the
#: key slots after MEASURE (``SEX.AGE``, ``SEX.STE`` or ``SEX``; empty for the
#: three-dimension labour-share flow), units as published, the frequencies the
#: flow publishes and ``projected``, the number of trailing years a modelled
#: flow publishes as projections, which carry no observation flag (0 for the
#: survey flows). Checked live on 2026-10-07 against the November 2025 edition
#: of the ILO modelled estimates, base year 2025: employment, unemployment,
#: labour force and hours run to 2027 (two projected years), the UN-based
#: working-age population to 2030 (five), labour share to 2026 (one), and
#: self-employment stops at 2025 (none). Key order is ``REF_AREA.FREQ.MEASURE[.SEX[.AGE|.STE]]``.
ILOSTAT_VARS: dict[str, dict] = {
    "emp": dict(flow="DF_EMP_TEMP_SEX_AGE_NB", measure="EMP_TEMP_NB", suffix=_SEX_AGE,
                units="thousands of persons", freqs="AQM", projected=0,
                label="Employment, 15+"),
    "une": dict(flow="DF_UNE_TUNE_SEX_AGE_NB", measure="UNE_TUNE_NB", suffix=_SEX_AGE,
                units="thousands of persons", freqs="AQM", projected=0,
                label="Unemployment, 15+"),
    "lf": dict(flow="DF_EAP_TEAP_SEX_AGE_NB", measure="EAP_TEAP_NB", suffix=_SEX_AGE,
               units="thousands of persons", freqs="AQM", projected=0,
               label="Labour force, 15+"),
    "wap": dict(flow="DF_POP_XWAP_SEX_AGE_NB", measure="POP_XWAP_NB", suffix=_SEX_AGE,
                units="thousands of persons", freqs="AQM", projected=0,
                label="Working-age population, 15+"),
    "urate": dict(flow="DF_UNE_DEAP_SEX_AGE_RT", measure="UNE_DEAP_RT", suffix=_SEX_AGE,
                  units="percent of labour force", freqs="AQM", projected=0,
                  label="Unemployment rate, 15+"),
    "prate": dict(flow="DF_EAP_DWAP_SEX_AGE_RT", measure="EAP_DWAP_RT", suffix=_SEX_AGE,
                  units="percent of working-age population", freqs="AQM", projected=0,
                  label="Labour force participation rate, 15+"),
    "epop": dict(flow="DF_EMP_DWAP_SEX_AGE_RT", measure="EMP_DWAP_RT", suffix=_SEX_AGE,
                 units="percent of working-age population", freqs="AQM", projected=0,
                 label="Employment-to-population ratio, 15+"),
    "emp_ees": dict(flow="DF_EMP_TEMP_SEX_STE_NB", measure="EMP_TEMP_NB",
                    suffix="SEX_T.STE_AGGREGATE_EES", units="thousands of persons",
                    freqs="AQM", projected=0, label="Employees (ICSE-93 1)"),
    "emp_self": dict(flow="DF_EMP_TEMP_SEX_STE_NB", measure="EMP_TEMP_NB",
                     suffix="SEX_T.STE_AGGREGATE_SLF", units="thousands of persons",
                     freqs="AQM", projected=0,
                     label="Self-employed (ICSE-93 2, 3, 4, 5)"),
    "hours": dict(flow="DF_HOW_TEMP_SEX_NB", measure="HOW_TEMP_NB", suffix="SEX_T",
                  units="hours per week per employed person", freqs="AQM", projected=0,
                  label="Mean weekly hours actually worked per employed person"),
    "informal_rate": dict(flow="DF_EMP_NIFL_SEX_RT", measure="EMP_NIFL_RT", suffix="SEX_T",
                          units="percent of employment", freqs="AQM", projected=0,
                          label="Informal employment rate"),
    "labour_share": dict(flow="DF_LAP_2GDP_NOC_RT", measure="LAP_2GDP_RT", suffix="",
                         units="percent of GDP", freqs="A", projected=1,
                         label="Labour income share (ILO modelled estimates)"),
    "emp_model": dict(flow="DF_EMP_2EMP_SEX_AGE_NB", measure="EMP_2EMP_NB", suffix=_SEX_AGE,
                      units="thousands of persons", freqs="A", projected=2,
                      label="Employment, 15+ (ILO modelled estimates)"),
    "urate_model": dict(flow="DF_UNE_2EAP_SEX_AGE_RT", measure="UNE_2EAP_RT", suffix=_SEX_AGE,
                        units="percent of labour force", freqs="A", projected=2,
                        label="Unemployment rate, 15+ (ILO modelled estimates)"),
    "lf_model": dict(flow="DF_EAP_2EAP_SEX_AGE_NB", measure="EAP_2EAP_NB", suffix=_SEX_AGE,
                     units="thousands of persons", freqs="A", projected=2,
                     label="Labour force, 15+ (ILO modelled estimates)"),
    "wap_model": dict(flow="DF_POP_2WAP_SEX_AGE_NB", measure="POP_2WAP_NB", suffix=_SEX_AGE,
                      units="thousands of persons", freqs="A", projected=5,
                      label="Working-age population, 15+ (UN estimates and projections)"),
    "emp_self_model": dict(flow="DF_EMP_2EMP_SEX_STE_NB", measure="EMP_2EMP_NB",
                           suffix="SEX_T.STE_AGGREGATE_SLF", units="thousands of persons",
                           freqs="A", projected=0,
                           label="Self-employed (ILO modelled estimates)"),
    "hours_model": dict(flow="DF_HOW_2EMP_SEX_NB", measure="HOW_2EMP_NB", suffix="SEX_T",
                        units="hours per week per employed person", freqs="A", projected=2,
                        label="Mean weekly hours per employed person (ILO modelled estimates)"),
}

#: The survey-based variables fetched when ``variables`` is None; ``modelled=True``
#: adds the ``*_model`` twins (annual only).
DEFAULT_VARS: tuple[str, ...] = (
    "emp", "une", "lf", "wap", "urate", "prate", "epop",
    "emp_ees", "emp_self", "hours", "informal_rate", "labour_share",
)

#: The ILO's modelled-estimate twins, fetched with ``modelled=True``.
MODEL_VARS: tuple[str, ...] = (
    "emp_model", "urate_model", "lf_model", "wap_model", "emp_self_model", "hours_model",
)

#: ILOSTAT area codes that are aggregates, not economies: world, ILO regions
#: and subregions, income groups and other groupings of the ``CL_AREA``
#: codelist (all-country pulls of October 2026 carried 91 of them).
ILO_AGGREGATES: frozenset[str] = frozenset(
    {f"X{i:02d}" for i in range(1, 100)} | {f"XA{i}" for i in range(10)}
    | {f"XB{i}" for i in range(10)}
)

# Codelist pattern of the aggregate block; a guard for codes added after the
# list above was written. ISO 3166 alpha-3 codes never contain a digit.
_AGGREGATE_RE = re.compile(r"^X[0-9A-Z]\d$")

# OBS_STATUS codes worth naming in the metadata (CL_OBS_STATUS).
_STATUS_LABELS = {
    "B": "break in series", "U": "unreliable", "I": "imputation",
    "M": "model extrapolation", "R": "real value", "A": "adjusted",
}


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def _get(url: str, *, timeout: float = 300.0, pause: float = 3.0,
         refresh: bool = False) -> bytes:
    """The one HTTP seam of this module; tests monkeypatch this name.

    Cached on disk outside the repository, paced ``pause`` seconds per host,
    asks for gzip (undone by the transport). Raises ``urllib.error.HTTPError``
    or ``OSError`` on failure, which the callers turn into ``attrs["missing"]``.
    """
    from puremacro._http import safe_get_bytes_cached
    return safe_get_bytes_cached(url, timeout, headers=_HEADERS,
                                 rate_limit_seconds=pause, refresh=refresh)


def _reason(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        # The NSI web service answers 404 "NoRecordsFound" to a key that
        # matches nothing: not an outage, but nothing to return either.
        return f"HTTP {exc.code}" + (" (no records)" if exc.code == 404 else "")
    if isinstance(exc, TimeoutError) or "timed out" in str(exc):
        return "timeout"
    if isinstance(exc, (http.client.HTTPException, EOFError, zlib.error)):
        return f"truncated or corrupt response ({type(exc).__name__})"
    return f"{type(exc).__name__}: {exc}"[:200]


def _fetch_csv(url: str, *, timeout: float, pause: float,
               refresh: bool) -> tuple[pd.DataFrame | None, str]:
    """One SDMX-CSV request: ``(frame, "ok")`` or ``(None, reason)``.

    A transport failure other than an HTTP 4xx is retried once (the
    host-level pause applies), since all-country pulls near the server's
    own limits occasionally drop. A body that arrives but does not parse as
    SDMX-CSV (an empty body, a maintenance page) has already been cached by
    the transport, so it is fetched once more with ``refresh=True``, which
    overwrites the bad cache entry.
    """
    reason = "no response"
    fresh = refresh
    for attempt in range(3):
        try:
            body = _get(url, timeout=timeout, pause=pause, refresh=fresh)
        except urllib.error.HTTPError as exc:
            reason = _reason(exc)
            if exc.code < 500 or attempt >= 1:
                return None, reason
            fresh = True
            continue
        except _TRANSPORT_ERRORS as exc:
            reason = _reason(exc)
            if attempt >= 1:
                return None, reason
            fresh = True
            continue
        frame, reason = _parse_csv(body)
        if frame is not None or fresh:
            return frame, reason
        fresh = True            # a cached bad body: replace it once
    return None, reason


def _parse_csv(body: bytes) -> tuple[pd.DataFrame | None, str]:
    try:
        frame = pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        return None, f"unreadable response ({type(exc).__name__})"
    needed = {"REF_AREA", "FREQ", "MEASURE", "TIME_PERIOD", "OBS_VALUE"}
    if not needed.issubset(frame.columns):
        return None, "unexpected response (not SDMX-CSV)"
    return frame, "ok"


def _parse_last_update(text: str | None) -> str | None:
    """``"03/10/2026 07:10:53"`` (day first) -> ``"2026-10-03T07:10:53"``."""
    if not text:
        return None
    try:
        return dt.datetime.strptime(text.strip(), "%d/%m/%Y %H:%M:%S").isoformat()
    except ValueError:
        return text.strip()


def ilostat_last_updates(flows: Iterable[str], *, timeout: float = 300.0,
                         pause: float = 3.0, refresh: bool = False) -> dict[str, str | None]:
    """Catalogue ``LAST_UPDATE`` of each ILOSTAT dataflow, as ISO timestamps.

    One request for the whole dataflow catalogue (7 MB, cached), parsed for
    the ``LAST_UPDATE`` annotation of each requested flow. A flow the
    catalogue does not list, or a failed request, maps to ``None``.

    Parameters
    ----------
    flows : iterable of str
        Dataflow ids such as ``"DF_EMP_TEMP_SEX_AGE_NB"``.
    timeout, pause, refresh
        As in :func:`ilostat_panel`.

    Returns
    -------
    dict
        ``{flow: "YYYY-MM-DDTHH:MM:SS" or None}``.
    """
    flows = list(dict.fromkeys(flows))
    out: dict[str, str | None] = {f: None for f in flows}
    try:
        body = _get(f"{_BASE}/dataflow/ILO", timeout=timeout, pause=pause, refresh=refresh)
    except _TRANSPORT_ERRORS as exc:
        warnings.warn(f"ILOSTAT catalogue unavailable ({_reason(exc)}); "
                      "LAST_UPDATE left blank", stacklevel=2)
        return out
    text = body.decode("utf-8", errors="ignore")
    for flow in flows:
        i = text.find(f'Dataflow id="{flow}"')
        if i < 0:
            continue
        j = text.find("</structure:Dataflow>", i)
        block = text[i: j if j > 0 else i + 20000]
        m = re.search(r"<common:AnnotationTitle>([^<]*)</common:AnnotationTitle>\s*"
                      r"<common:AnnotationType>LAST_UPDATE</common:AnnotationType>", block)
        if m:
            out[flow] = _parse_last_update(m.group(1))
    return out


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def is_ilo_aggregate(code: object) -> bool:
    """True when ``code`` is an ILOSTAT aggregate (world, region, income group)."""
    s = str(code)
    return s in ILO_AGGREGATES or bool(_AGGREGATE_RE.match(s))


def _period_start(periods: pd.Series) -> pd.Series:
    """ILOSTAT ``TIME_PERIOD`` (``2025``, ``2025-Q3``, ``2025-M09``) -> period start."""
    parts = periods.astype(str).str.extract(r"^(\d{4})(?:-([QM])(\d{1,2}))?$")
    year = pd.to_numeric(parts[0], errors="coerce")
    sub = pd.to_numeric(parts[2], errors="coerce")
    month = np.where(parts[1] == "Q", (sub - 1) * 3 + 1,
                     np.where(parts[1] == "M", sub, 1))
    ok = year.notna() & pd.notna(month)
    out = pd.Series(pd.NaT, index=periods.index, dtype="datetime64[ns]")
    if ok.any():
        out[ok] = pd.to_datetime(dict(year=year[ok].astype(int),
                                      month=pd.Series(month, index=periods.index)[ok].astype(int),
                                      day=1))
    return out


def _series_suffix(frame: pd.DataFrame) -> pd.Series:
    """The key slots between MEASURE and TIME_PERIOD of each row, dot-joined."""
    cols = list(frame.columns)
    dims = cols[cols.index("MEASURE") + 1: cols.index("TIME_PERIOD")]
    if not dims:
        return pd.Series("", index=frame.index)
    return frame[dims].astype(str).agg(".".join, axis=1)


def _year_of(value) -> int | None:
    if value is None:
        return None
    m = re.match(r"^\s*(\d{4})", str(value))
    if not m:
        raise ValueError(f"start/end must begin with a four-digit year; got {value!r}")
    return int(m.group(1))


def _bound(value, *, end: bool) -> pd.Timestamp | None:
    """``"1995"``, ``1995``, ``"1995-07"``, ``"1995Q3"`` -> a timestamp bound."""
    if value is None:
        return None
    s = str(value).strip()
    m = re.match(r"^(\d{4})(?:[-]?(Q)(\d)|-(M)?(\d{1,2})(?:-\d{1,2})?)?$", s)
    if not m:
        raise ValueError(f"start/end must look like 1995, '1995-07' or '1995Q3'; got {value!r}")
    year = int(m.group(1))
    if m.group(2):
        q = int(m.group(3))
        p = pd.Period(year=year, quarter=q, freq="Q")
    elif m.group(5):
        p = pd.Period(year=year, month=int(m.group(5)), freq="M")
    else:
        p = pd.Period(year=year, freq="Y")
    return p.start_time.normalize() if not end else p.end_time.normalize()


# ---------------------------------------------------------------------------
# Argument checking
# ---------------------------------------------------------------------------

def _check_freq(freq: str) -> str:
    f = str(freq).upper()
    if f not in _FREQS:
        raise ValueError(f"freq must be one of {_FREQS}; got {freq!r}")
    return f


def _check_vars(vars_, freq: str, modelled: bool) -> list[str]:
    if modelled and freq != "A":
        raise ValueError(f"modelled=True: the ILO modelled estimates are annual only; "
                         f"use freq='A' (got freq={freq!r}; valid freqs for the model "
                         f"twins: ('A',))")
    if vars_ is None:
        names = [v for v in DEFAULT_VARS if freq in ILOSTAT_VARS[v]["freqs"]]
        if modelled:
            names += list(MODEL_VARS)
        return names
    if isinstance(vars_, str):
        vars_ = [vars_]
    names = list(dict.fromkeys(vars_))
    unknown = [v for v in names if v not in ILOSTAT_VARS]
    if unknown:
        raise ValueError(f"unknown ILOSTAT variable(s) {unknown}; valid: {sorted(ILOSTAT_VARS)}")
    wrong = [v for v in names if freq not in ILOSTAT_VARS[v]["freqs"]]
    if wrong:
        valid = sorted(v for v, s in ILOSTAT_VARS.items() if freq in s["freqs"])
        raise ValueError(f"variable(s) {wrong} not published at freq={freq!r}; valid: {valid}")
    if modelled:
        names += [v for v in MODEL_VARS if v not in names]
    return names


def _check_codes(codes) -> list[str] | None:
    if codes is None:
        return None
    if isinstance(codes, str):
        codes = [codes]
    out = [str(c).strip().upper() for c in codes]
    bad = [c for c in out if not re.fullmatch(r"[A-Z0-9]{3}", c)]
    if bad:
        raise ValueError(f"codes must be ILOSTAT/ISO3 area codes such as 'MEX'; got {bad}")
    if not out:
        raise ValueError("codes is an empty list; pass None for every economy")
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _empty(names: list[str]) -> pd.DataFrame:
    idx = pd.MultiIndex.from_arrays(
        [pd.Index([], dtype=object), pd.DatetimeIndex([], dtype="datetime64[ns]")],
        names=["code", "date"])
    return pd.DataFrame({v: pd.Series([], dtype=float) for v in names}, index=idx)


def _plan(names: list[str]) -> dict[tuple[str, str, str], list[str]]:
    """Group variables into requests: (flow, measure, leading slots) -> vars."""
    groups: dict[tuple[str, str, str], list[str]] = {}
    for v in names:
        spec = ILOSTAT_VARS[v]
        slots = spec["suffix"].split(".") if spec["suffix"] else []
        lead = ".".join(slots[:-1])
        groups.setdefault((spec["flow"], spec["measure"], lead), []).append(v)
    return groups


def _url(flow: str, freq: str, measure: str, vars_: list[str], areas: str,
         y0: int | None, y1: int | None) -> str:
    slots = [ILOSTAT_VARS[v]["suffix"] for v in vars_]
    if slots[0]:
        parts = [s.split(".") for s in slots]
        last = "+".join(dict.fromkeys(p[-1] for p in parts))
        tail = "." + ".".join(parts[0][:-1] + [last])
    else:
        tail = ""
    url = f"{_BASE}/data/ILO,{flow},1.0/{areas}.{freq}.{measure}{tail}?format=csv"
    if y0 is not None:
        url += f"&startPeriod={y0}"
    if y1 is not None:
        url += f"&endPeriod={y1}"
    return url


def _sa_of(sources: list[str]) -> str:
    adj = ["ADJ" in s.upper() for s in sources if s]
    if not adj:
        return "NSA"
    if all(adj):
        return "SA"
    return "mixed" if any(adj) else "NSA"


def ilostat_panel(
    codes: Iterable[str] | str | None = None,
    *,
    freq: str = "A",
    variables: Iterable[str] | str | None = None,
    start=None,
    end=None,
    modelled: bool = False,
    refresh: bool = False,
    timeout: float = 300.0,
    pause: float = 3.0,
) -> pd.DataFrame:
    """ILOSTAT labour panel, ``(code, date)`` by variable, one call.

    Parameters
    ----------
    codes : str or iterable of str, optional
        ILOSTAT area codes (ISO3, plus ``KOS``, ``CHA``, ``ANT``). Default:
        every economy; ILO aggregates are always dropped.
    freq : {"A", "Q", "M"}
        Annual, quarterly or monthly. The same variable names serve all three;
        ``labour_share`` and the ``*_model`` twins are annual only.
    variables : str or iterable of str, optional
        Names from :data:`ILOSTAT_VARS`. Default: every survey-based variable
        the frequency publishes (:data:`DEFAULT_VARS`).
    start, end : int or str, optional
        Bounds such as ``1995``, ``"1995-07"`` or ``"1995Q3"``, applied
        exactly to period-start dates; sent to the provider as years.
        Default: the earliest and latest periods published. With ``end``
        unset, the projected trailing years of the modelled flows (the
        ``projected`` count in :data:`ILOSTAT_VARS`, counted back from the
        last year the flow publishes) are dropped.
    modelled : bool
        Add the ILO modelled-estimate twins (:data:`MODEL_VARS`): balanced
        panels of 189 economies from 1991 that fill survey gaps with model
        output. Annual only: ``modelled=True`` with ``freq`` other than
        ``"A"`` raises ``ValueError``.
    refresh : bool
        Re-download instead of reading the on-disk cache.
    timeout : float
        Seconds per request; an all-country pull can take 50.
    pause : float
        Seconds between requests to the ILO host.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(code, date)``, dates at period start, one float column
        per variable, values as published (thousands of persons, percent,
        hours per week). ``attrs``:

        - ``meta``: tuple of dicts, one per country and variable: ``variable,
          code, flow, key, measure, units, unit_mult, sa, first, last, n,
          breaks`` (periods flagged ``B``), ``status`` (counts of each
          OBS_STATUS flag), ``surveys`` (the ``SOURCE`` strings, oldest
          first), ``trimmed_after`` (last year kept on a projected flow,
          ``None`` when nothing was cut).
        - ``source``: provider and each flow's catalogue ``LAST_UPDATE``.
        - ``flows``: tuple of dicts ``flow, variables, url, status, rows,
          last_update``.
        - ``fetched_at``: UTC ISO timestamp.
        - ``missing``: tuple of dicts for variables or codes asked for and
          not obtained, with the reason.
        - ``aggregates_dropped``: ILO aggregate codes removed.
        - ``duplicates``: tuple of ``(code, period, variable)`` that the
          provider returned more than once (empty in every live pull so
          far); the first value is kept and a warning is issued.

    Raises
    ------
    ValueError
        Unknown variable, variable not published at ``freq``, bad ``freq``,
        ``modelled=True`` at ``"Q"``/``"M"``, malformed ``codes`` or bounds.
        Never on a provider failure.

    Examples
    --------
    >>> from puremacro.fetch.ilostat import ilostat_panel
    >>> a = ilostat_panel()                                          # doctest: +SKIP
    >>> a.loc["MEX", "urate"].dropna().index[0]                      # doctest: +SKIP
    Timestamp('1988-01-01 00:00:00')
    >>> q = ilostat_panel(["MEX", "USA", "ESP"], freq="Q",
    ...                   variables=["urate", "emp"])                # doctest: +SKIP
    >>> m = ilostat_panel(freq="M", variables=("emp", "une", "urate", "prate"))  # doctest: +SKIP
    """
    f = _check_freq(freq)
    names = _check_vars(variables, f, modelled)
    areas_list = _check_codes(codes)
    t0, t1 = _bound(start, end=False), _bound(end, end=True)
    if t0 is not None and t1 is not None and t0 > t1:
        raise ValueError(f"start {start!r} is after end {end!r}")
    y0, y1 = _year_of(start), _year_of(end)
    areas = "+".join(areas_list) if areas_list else ""
    cut_years: dict[str, int] = {}     # flow -> last year kept

    plan = _plan(names)
    updates = ilostat_last_updates([k[0] for k in plan], timeout=timeout, pause=pause,
                                   refresh=refresh)

    longs: list[pd.DataFrame] = []
    missing: list[dict] = []
    flows_info: list[dict] = []
    for (flow, measure, _lead), group in plan.items():
        url = _url(flow, f, measure, group, areas, y0, y1)
        frame, status = _fetch_csv(url, timeout=timeout, pause=pause, refresh=refresh)
        info = dict(flow=flow, variables=tuple(group), url=url, status=status, rows=0,
                    last_update=updates.get(flow))
        flows_info.append(info)
        if frame is None:
            warnings.warn(f"ILOSTAT {flow} ({', '.join(group)}): {status}", stacklevel=2)
            missing += [dict(variable=v, flow=flow, reason=status) for v in group]
            continue
        frame = frame[(frame["FREQ"] == f) & (frame["MEASURE"] == measure)]
        horizon = max(ILOSTAT_VARS[v]["projected"] for v in group)
        if horizon and end is None and len(frame):
            last = pd.to_numeric(frame["TIME_PERIOD"].str[:4], errors="coerce").max()
            if pd.notna(last):
                cut_years[flow] = int(last) - horizon
        suffix = _series_suffix(frame)
        for v in group:
            spec = ILOSTAT_VARS[v]
            part = frame[suffix == spec["suffix"]]
            if part.empty:
                missing.append(dict(variable=v, flow=flow, reason="no observations"))
                continue
            long = pd.DataFrame({
                "code": part["REF_AREA"].values,
                "date": _period_start(part["TIME_PERIOD"]).values,
                "period": part["TIME_PERIOD"].values,
                "variable": v,
                "value": pd.to_numeric(part["OBS_VALUE"], errors="coerce").values,
                "status": part["OBS_STATUS"].values if "OBS_STATUS" in part else "",
                "survey": part["SOURCE"].values if "SOURCE" in part else "",
                "unit_mult": part["UNIT_MULT"].values if "UNIT_MULT" in part else "",
            })
            info["rows"] += len(long)
            longs.append(long)

    fetched_at = _now()
    source = _SOURCE + "; " + "; ".join(
        f"ILO,{i['flow']},1.0 (LAST_UPDATE {i['last_update'] or 'unknown'})" for i in flows_info)

    if longs:
        df = pd.concat(longs, ignore_index=True)
        df = df[df["date"].notna() & df["value"].notna()]
    else:
        df = pd.DataFrame(columns=["code", "date", "period", "variable", "value",
                                   "status", "survey", "unit_mult"])
    agg_mask = df["code"].map(is_ilo_aggregate).astype(bool) if len(df) else pd.Series([], dtype=bool)
    dropped = tuple(sorted(set(df.loc[agg_mask, "code"]))) if len(df) else ()
    df = df[~agg_mask] if len(df) else df
    if len(df):
        if t0 is not None:
            df = df[df["date"] >= t0]
        if t1 is not None:
            df = df[df["date"] <= t1]
        if cut_years:
            cut = df["variable"].map(lambda v: cut_years.get(ILOSTAT_VARS[v]["flow"], 10**6))
            df = df[df["date"].dt.year <= cut]

    listed = {m.get("variable") for m in missing}
    have = set(df["variable"]) if len(df) else set()
    missing += [dict(variable=v, flow=ILOSTAT_VARS[v]["flow"], reason="no observations in range")
                for v in names if v not in have and v not in listed]

    if areas_list:
        got = set(df["code"]) if len(df) else set()
        for c in areas_list:
            if is_ilo_aggregate(c):
                missing.append(dict(code=c, reason="ILO aggregate, dropped"))
            elif c not in got:
                missing.append(dict(code=c, reason="no observations for any variable"))

    dups: tuple = ()
    if len(df):
        dmask = df.duplicated(["code", "date", "variable"], keep="first")
        if dmask.any():
            dups = tuple(sorted(
                (str(c), str(p), str(v))
                for c, p, v in df.loc[dmask, ["code", "period", "variable"]].itertuples(index=False)))
            warnings.warn(f"ILOSTAT returned {len(dups)} duplicated (code, period, variable) "
                          f"observations, first kept: {list(dups[:5])}", stacklevel=2)
            df = df[~dmask]
        wide = df.pivot_table(index=["code", "date"], columns="variable", values="value",
                              aggfunc="first")
        wide = wide.reindex(columns=names).astype(float)
        wide.columns.name = None
        wide = wide.sort_index()
    else:
        wide = _empty(names)

    meta: list[dict] = []
    if len(df):
        for (v, code), g in df.sort_values("date").groupby(["variable", "code"], sort=True):
            spec = ILOSTAT_VARS[v]
            st = g["status"].astype(str)
            counts = {k: int(n) for k, n in st[st != ""].value_counts().sort_index().items()}
            surveys = [s for s in dict.fromkeys(g["survey"].astype(str)) if s]
            um = [u for u in dict.fromkeys(g["unit_mult"].astype(str)) if u]
            meta.append(dict(
                variable=v, code=code, flow=spec["flow"],
                key=f"{code}.{f}.{spec['measure']}" + (f".{spec['suffix']}" if spec["suffix"] else ""),
                measure=spec["measure"], units=spec["units"],
                unit_mult=int(um[0]) if len(um) == 1 and um[0].isdigit() else (
                    3 if spec["units"].startswith("thousands") else 0),
                sa=_sa_of(surveys), freq=f,
                first=str(g["period"].iloc[0]), last=str(g["period"].iloc[-1]), n=int(len(g)),
                breaks=tuple(g.loc[st == "B", "period"].astype(str)),
                status=counts, surveys=tuple(surveys),
                trimmed_after=cut_years.get(spec["flow"]) if spec["projected"] else None,
                last_update=updates.get(spec["flow"]),
            ))

    wide.attrs = dict(
        meta=tuple(meta), source=source, flows=tuple(flows_info), fetched_at=fetched_at,
        missing=tuple(missing), aggregates_dropped=dropped, duplicates=dups, freq=f,
    )
    return wide


def ilostat_meta(panel: pd.DataFrame) -> pd.DataFrame:
    """``attrs["meta"]`` of an :func:`ilostat_panel` result as a DataFrame.

    One row per country and variable, indexed by ``(variable, code)``; the
    ``breaks`` column holds the periods flagged as series breaks.
    """
    meta = panel.attrs.get("meta", ())
    out = pd.DataFrame(list(meta))
    if out.empty:
        return out
    return out.set_index(["variable", "code"]).sort_index()


__all__ = [
    "DEFAULT_VARS",
    "ILOSTAT_VARS",
    "ILO_AGGREGATES",
    "MODEL_VARS",
    "ilostat_last_updates",
    "ilostat_meta",
    "ilostat_panel",
    "is_ilo_aggregate",
]
