r"""Quarterly blocks that sit beside :func:`qna_panel`: sector GFCF, population, LFS, vacancies.

:func:`puremacro.fetch.qna_panel` covers the three approaches to quarterly
GDP. A cross-country business-cycle panel needs a few more series that the
OECD publishes in other dataflows, each with its own key layout and its own
traps. This module turns each of them into one call that returns the same
shape as ``qna_panel`` -- a frame indexed by ``(code, date)``, ISO3 codes,
quarter- or month-start dates, one column per variable, natural units -- so
the blocks can be joined on the index without any reshaping::

    qna_sector_gfcf(codes)     inv_gov  inv_priv  (+ _real where published)
    qna_population(codes)      pop  emp_nc  emp_employees_nc  emp_selfemp_nc
    oecd_lfs_panel(codes)      urate  urate_1564  emp_lfs  lf  wap  wap_1564
                               prate_1564  epop_1564          (freq "Q" or "M")
    oecd_vacancies(codes)      vacancies  vacancies_new  reg_unemp

Why four functions rather than four more flags on ``qna_panel``: none of
these flows shares the expenditure flow's coverage. Government investment is
published quarterly by a dozen countries in the national-accounts flow and by
about thirty more only in the sector accounts (``DF_QSA``), with a different
key; the labour-force flows key on measure, age and unit rather than on
transaction and sector; registered vacancies come from yet another DSD. Kept
apart, each block reports its own coverage honestly in ``attrs["meta"]`` and
``attrs["missing"]`` instead of widening a panel with columns that are NaN
for most countries.

Mechanics shared by all four:

- Every request goes through :func:`puremacro.fetch._oecd_sdmx.oecd_csv`
  (urllib, ``format=csvfile``, cached on disk outside the repository,
  requests to ``sdmx.oecd.org`` spaced :data:`OECD_PAUSE` seconds apart,
  three attempts on an HTTP 429). It is reached through the module-level
  :func:`_get`, the name the tests patch.
- Countries are sent ten per request. ``codes=None`` asks for every
  reference area: the narrow flows (sector GFCF, population, vacancies) in
  one open request, the labour-force flow by first asking the availability
  endpoint which areas publish the key and then chunking them.
- OECD aggregates (euro area, EU, OECD, G7, World, ...) are dropped by the
  explicit list :data:`puremacro.fetch._oecd_sdmx.OECD_AGGREGATES`.
- A provider failure never raises: the block comes back empty (same index
  names, same columns, same ``attrs`` keys) with a warning, and what was
  asked for but not obtained is listed in ``attrs["missing"]``. Only a caller
  error (bad ``freq`` or ``sa``) raises ``ValueError``.
- Seasonal adjustment: the OECD's own adjusted series wins where it exists.
  ``qna_sector_gfcf(sa="x13")`` adjusts the series published unadjusted only
  with ``qna_panel``'s X-13 path (:func:`puremacro.sa.deseasonalize_x13`,
  binary when present, pure-Python X-11 otherwise; ``statsmodels`` is only
  ever imported lazily there); ``sa="prefer"`` returns them unadjusted. The
  other three blocks prefer the published adjusted series and fall back to
  the unadjusted one. Whatever was used is recorded per country and variable
  in ``attrs["meta"]["sa"]`` (``"oecd"``, ``"puremacro"``, ``"none"``).

Source: OECD SDMX dataflows ``OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_GFCF_SECTOR``,
``OECD.SDD.NAD,DSD_NASEC1@DF_QSA``, ``OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_POP_EMPNC``,
``OECD.SDD.TPS,DSD_LFS@DF_IALFS_INDIC`` and ``OECD.SDD.TPS,DSD_OLAB@DF_OIALAB_INDIC``.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ._oecd_sdmx import OECD_AGGREGATES, oecd_availability, oecd_csv, oecd_key
from .oecd_qna_panel import _adjust_unadjusted, _tidy

#: Seconds between two requests to ``sdmx.oecd.org`` (process-wide, enforced
#: by the transport). The endpoint throttles by request count *and* response
#: size; raise it when several jobs share the address.
OECD_PAUSE: float = 5.0

# --------------------------------------------------------------------------
# Dataflows and their key layouts
# --------------------------------------------------------------------------

_GFCF_SECTOR_FLOW = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_GFCF_SECTOR,"
_QSA_FLOW = "OECD.SDD.NAD,DSD_NASEC1@DF_QSA,"
_POP_FLOW = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_POP_EMPNC,"
_LFS_FLOW = "OECD.SDD.TPS,DSD_LFS@DF_IALFS_INDIC,1.0"
_OLAB_FLOW = "OECD.SDD.TPS,DSD_OLAB@DF_OIALAB_INDIC,"

#: Dimension order of ``DSD_NAMAIN1`` (13 dimensions).
_NAMAIN1_DIMS: tuple[str, ...] = (
    "FREQ", "ADJUSTMENT", "REF_AREA", "SECTOR", "COUNTERPART_SECTOR",
    "TRANSACTION", "INSTR_ASSET", "ACTIVITY", "EXPENDITURE", "UNIT_MEASURE",
    "PRICE_BASE", "TRANSFORMATION", "TABLE_IDENTIFIER")
#: Dimension order of ``DSD_NASEC1`` (14): ACCOUNTING_ENTRY after the
#: counterpart sector and VALUATION after the unit.
_NASEC1_DIMS: tuple[str, ...] = (
    "FREQ", "ADJUSTMENT", "REF_AREA", "SECTOR", "COUNTERPART_SECTOR",
    "ACCOUNTING_ENTRY", "TRANSACTION", "INSTR_ASSET", "EXPENDITURE",
    "UNIT_MEASURE", "VALUATION", "PRICE_BASE", "TRANSFORMATION",
    "TABLE_IDENTIFIER")
#: Dimension order of ``DSD_LFS`` (9). FREQ is the *last* dimension.
_LFS_DIMS: tuple[str, ...] = (
    "REF_AREA", "MEASURE", "UNIT_MEASURE", "TRANSFORMATION", "ADJUSTMENT",
    "SEX", "AGE", "ACTIVITY", "FREQ")
#: Dimension order of ``DSD_OLAB`` (7). FREQ last here too.
_OLAB_DIMS: tuple[str, ...] = (
    "REF_AREA", "MEASURE", "UNIT_MEASURE", "TRANSFORMATION", "ADJUSTMENT",
    "SECTOR", "FREQ")

#: Gross fixed capital formation (P51G) by institutional sector. ``S13`` is
#: general government; ``S1W`` is "all sectors other than general
#: government", i.e. private investment, published as such (no subtraction).
#: Where only the sector accounts publish, private is total (S1) less S13.
SECTOR_GFCF: dict[str, tuple[str, str]] = {
    "inv_gov":  ("S13", "Gross fixed capital formation, general government"),
    "inv_priv": ("S1W", "Gross fixed capital formation, other than general government"),
}

#: Population and employment, national concept (residents, wherever they
#: work), thousands of persons. ``(TRANSACTION, ACTIVITY, description)``.
QNA_POPULATION: dict[str, tuple[str, str, str]] = {
    "pop":              ("POP",  "_Z", "Total population"),
    "emp_nc":           ("EMP",  "_T", "Total employment, national concept"),
    "emp_employees_nc": ("SAL",  "_T", "Employees, national concept"),
    "emp_selfemp_nc":   ("SELF", "_T", "Self-employed, national concept"),
}

#: Labour-force survey indicators, total of both sexes.
#: ``(MEASURE, AGE, UNIT_MEASURE, description)``. Levels in thousands of
#: persons, rates in percent. ``urate`` takes the OECD's harmonised monthly
#: unemployment rate (``UNE_LF_M``, the longer series) where a country
#: publishes it and the plain survey rate (``UNE_LF``) otherwise.
#: Areas that published DF_IALFS_INDIC quarterly on 7 Oct 2026; requested
#: (ten at a time) only when the availability endpoint does not answer.
_LFS_AREAS: tuple[str, ...] = (
    "AUS", "AUT", "BEL", "BGR", "BRA", "CAN", "CHE", "CHL", "COL", "CRI",
    "CZE", "DEU", "DNK", "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HRV",
    "HUN", "IDN", "IRL", "ISL", "ISR", "ITA", "JPN", "KOR", "LTU", "LUX",
    "LVA", "MEX", "NLD", "NOR", "NZL", "POL", "PRT", "ROU", "RUS", "SVK",
    "SVN", "SWE", "TUR", "USA", "ZAF")

LFS_INDICATORS: dict[str, tuple[str, str, str, str]] = {
    "urate":      ("UNE_LF_M", "Y_GE15", "PT_LF_SUB",  "Unemployment rate, 15+, % of labour force"),
    "urate_1564": ("UNE_LF",   "Y15T64", "PT_LF_SUB",  "Unemployment rate, 15-64, % of labour force"),
    "emp_lfs":    ("EMP",      "Y_GE15", "PS",         "Employment (LFS), 15+"),
    "lf":         ("LF",       "Y_GE15", "PS",         "Labour force, 15+"),
    "wap":        ("WAP",      "Y_GE15", "PS",         "Working-age population, 15+"),
    "wap_1564":   ("WAP",      "Y15T64", "PS",         "Working-age population, 15-64"),
    "prate_1564": ("LF_WAP",   "Y15T64", "PT_WAP_SUB", "Participation rate, 15-64, % of population"),
    "epop_1564":  ("EMP_WAP",  "Y15T64", "PT_WAP_SUB", "Employment rate, 15-64, % of population"),
}
#: The fallback series for ``urate`` (BRA, RUS, ZAF publish only this one).
_URATE_FALLBACK = ("UNE_LF", "Y_GE15", "PT_LF_SUB")

#: Registered labour-market series from the public employment services,
#: whole economy (``SECTOR=S1``), thousands of persons. ``VAC_U`` is the
#: stock of unfilled vacancies at the end of the period; ``VAC_N`` is the
#: flow of vacancies notified during it -- different concepts, so they are
#: different columns, and few countries publish the flow.
OECD_VACANCIES: dict[str, tuple[str, str]] = {
    "vacancies":     ("VAC_U",   "Registered unfilled vacancies (stock)"),
    "vacancies_new": ("VAC_N",   "Registered new vacancies (flow during the period)"),
    "reg_unemp":     ("REG_UNE", "Registered unemployed"),
}

_SA_LABEL = {"Y": "oecd", "X": "puremacro", "N": "none"}
_META_KEYS = ("code", "variable", "source", "series", "units", "unit_mult",
              "sa", "first", "last", "n")

_MONEY = "millions of national currency, current prices"
_THOUSANDS = "thousands of persons"


# --------------------------------------------------------------------------
# Transport (the names tests patch)
# --------------------------------------------------------------------------

def _get(agency_flow: str, key: str, *, start_period: str | None = None,
         refresh: bool = False) -> pd.DataFrame:
    """One OECD data request; an empty frame with ``attrs["status"]`` on failure."""
    return oecd_csv(agency_flow, key, start_period=start_period,
                    refresh=refresh, pause=OECD_PAUSE)


def _availability(agency_flow: str, key: str, *, refresh: bool = False) -> dict:
    """The availability endpoint's JSON for ``key``; ``{}`` on failure."""
    return oecd_availability(agency_flow, key, refresh=refresh)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm_codes(codes: Iterable[str] | str | None) -> list[str] | None:
    if codes is None:
        return None
    if isinstance(codes, str):
        codes = [codes]
    out: list[str] = []
    for c in codes:
        c = str(c).strip().upper()
        if c and c not in out:
            out.append(c)
    return out


def _chunks(codes: Sequence[str], size: int = 10) -> list[list[str]]:
    return [list(codes[i:i + size]) for i in range(0, len(codes), size)]


def _ref_areas(payload: dict) -> list[str]:
    """REF_AREA values listed in an availability answer (aggregates removed)."""
    try:
        regions = payload["data"]["contentConstraints"][0]["cubeRegions"][0]
        for kv in regions["keyValues"]:
            if kv["id"] == "REF_AREA":
                return sorted(str(v) for v in kv["values"]
                              if str(v) not in OECD_AGGREGATES)
    except (KeyError, IndexError, TypeError):
        return []
    return []


def _period_start(s: pd.Series, freq: str) -> pd.Series:
    """'2020-Q3' -> 2020-07-01, '2020-07' -> 2020-07-01."""
    return pd.PeriodIndex(s.astype(str), freq=freq).to_timestamp(how="start")


class _Run:
    """Book-keeping shared by one public call: requests made and what is missing."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.failed: list[dict] = []

    def fetch(self, flow: str, key: str, codes: Sequence[str] | None, *,
              start: str | None, refresh: bool) -> pd.DataFrame:
        try:
            raw = _get(flow, key, start_period=start, refresh=refresh)
        except Exception as exc:  # noqa: BLE001 - never raise on a provider failure
            raw = pd.DataFrame()
            raw.attrs["status"] = f"{type(exc).__name__}: {exc}"[:200]
        status = raw.attrs.get("status", "ok" if not raw.empty else "empty")
        self.requests.append({"flow": flow.rstrip(","), "key": key,
                              "status": status, "rows": int(len(raw))})
        # "HTTP 404" is how the OECD says "no data for that key": an answer,
        # not a failure. Anything else that came back empty is a failure the
        # caller must be able to tell apart from "not published".
        if raw.empty and status not in ("ok", "HTTP 404", "empty response"):
            self.failed.append({"flow": flow.rstrip(","), "key": key,
                                "codes": tuple(codes or ("*",)), "status": status})
            warnings.warn(f"OECD request failed ({status}): {flow.rstrip(',')} "
                          f"{key}; those countries are listed in attrs['missing']",
                          UserWarning, stacklevel=3)
        return raw


def _drop_aggregates(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or "REF_AREA" not in raw.columns:
        return raw
    return raw[~raw["REF_AREA"].astype(str).isin(OECD_AGGREGATES)]


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _empty(columns: Sequence[str], source: str) -> pd.DataFrame:
    idx = pd.MultiIndex.from_arrays(
        [pd.Index([], dtype=object), pd.DatetimeIndex([])], names=["code", "date"])
    out = pd.DataFrame(index=idx, columns=list(columns), dtype=float)
    out.attrs.update(meta=(), source=source, fetched_at=_now(), missing=(),
                     requests=())
    return out


def _wide(tidy: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    w = tidy.pivot_table(index=["code", "date"], columns="name", values="value",
                         aggfunc="first")
    w.columns.name = None
    for c in columns:
        if c not in w.columns:
            w[c] = np.nan
    w = w[list(columns)].astype(float).sort_index()
    return w.dropna(how="all")


def _meta_records(tidy: pd.DataFrame, *, source_of, series_of, units_of,
                  ) -> list[dict]:
    """One plain dict per (code, variable) actually present in ``tidy``."""
    rows = []
    for (code, name), g in tidy.groupby(["code", "name"], sort=True):
        mult = (pd.to_numeric(g["UNIT_MULT"], errors="coerce").dropna()
                if "UNIT_MULT" in g.columns else pd.Series(dtype=float))
        kinds = {_SA_LABEL.get(str(k), "none") for k in g["ADJUSTMENT"].unique()}
        rows.append({
            "code": code,
            "variable": name,
            "source": source_of(code, name, g),
            "series": series_of(code, name, g),
            "units": units_of(code, name, g),
            "unit_mult": int(mult.mode().iloc[0]) if len(mult) else None,
            "sa": kinds.pop() if len(kinds) == 1 else "mixed",
            "first": g["date"].min().strftime("%Y-%m-%d"),
            "last": g["date"].max().strftime("%Y-%m-%d"),
            "n": int(g["date"].nunique()),
        })
    return rows


def _missing(requested: Sequence[str] | None, panel: pd.DataFrame,
             columns: Sequence[str], failed: list[dict]) -> tuple[dict, ...]:
    """What was asked for and not obtained, one dict per (code, variable)."""
    out: list[dict] = []
    failed_codes: dict[str, str] = {}
    for f in failed:
        for c in f["codes"]:
            failed_codes.setdefault(c, f"{f['status']} on {f['flow']}")
        if f["codes"] == ("*",):
            out.append({"code": "*", "variable": "*",
                        "reason": f"{f['status']} on {f['flow']} key {f['key']}"})
    universe = (list(requested) if requested is not None
                else sorted({c for f in failed for c in f["codes"] if c != "*"}
                            | set(panel.index.get_level_values("code"))))
    have = panel.notna().groupby(level="code").any() if len(panel) else None
    for code in universe:
        for col in columns:
            ok = (have is not None and code in have.index and bool(have.at[code, col]))
            if not ok:
                out.append({"code": code, "variable": col,
                            "reason": failed_codes.get(code, "not published")})
    return tuple(out)


def _finish(panel: pd.DataFrame, *, meta: list[dict], source: str,
            requested: Sequence[str] | None, columns: Sequence[str],
            run: _Run, **extra) -> pd.DataFrame:
    # pivot_table carries the raw response's attrs (url, status) over; they
    # would contradict ``missing``, so start from a clean slate.
    panel.attrs.clear()
    panel.attrs.update(meta=tuple(meta), source=source, fetched_at=_now(),
                       missing=_missing(requested, panel, columns, run.failed),
                       requests=tuple(run.requests), **extra)
    return panel


def _check_choice(name: str, value: str, valid: Sequence[str]) -> None:
    if value not in valid:
        raise ValueError(f"{name} must be one of {list(valid)}, got {value!r}")


# --------------------------------------------------------------------------
# 1. GFCF by institutional sector
# --------------------------------------------------------------------------

def _sector_gfcf_columns(real: bool = True) -> list[str]:
    cols = list(SECTOR_GFCF)
    return cols + [f"{c}_real" for c in SECTOR_GFCF] if real else cols


def _tidy_sector_flow(raw: pd.DataFrame, sa: str) -> pd.DataFrame:
    """Rows of the QNA sector flow as ``qna_panel``'s tidy frame (V, L, Q)."""
    if raw.empty:
        return pd.DataFrame()
    raw = raw[raw["TIME_PERIOD"].notna()]
    raw = raw[raw.get("TRANSACTION", pd.Series("P51G", index=raw.index)) == "P51G"]
    lookup = {(sector,): name for name, (sector, _) in SECTOR_GFCF.items()}
    tidy = _tidy(raw, lookup, ("SECTOR",), sa=sa)
    if tidy.empty:
        return tidy
    tidy = tidy.copy()
    tidy["_source"] = _GFCF_SECTOR_FLOW.rstrip(",")
    tidy["_series"] = tidy["name"].map({n: f"{s} P51G" for n, (s, _) in SECTOR_GFCF.items()})
    return tidy


def _tidy_qsa(raw: pd.DataFrame, sa: str) -> pd.DataFrame:
    """Government and private GFCF from the sector accounts (current prices).

    Pins TRANSFORMATION N (``LA`` is the same series at an annualised rate,
    four times the level) and ACCOUNTING_ENTRY D, and drops the rows with an
    empty TIME_PERIOD that ``DF_QSA`` returns for series it lists but does
    not fill (Spain, Portugal). Government investment takes the published
    adjusted series where there is one, like every other series here.
    Private investment is total (S1) less government (S13), and the two
    legs are always taken in the same edition: both adjusted where both are
    published adjusted, otherwise both unadjusted (most countries publish
    S1 unadjusted only), in which case ``sa="x13"`` adjusts the difference.
    """
    if raw.empty:
        return pd.DataFrame()
    raw = raw[raw["TIME_PERIOD"].notna() & raw["OBS_VALUE"].notna()]
    for col, val in (("ACCOUNTING_ENTRY", "D"), ("TRANSACTION", "P51G"),
                     ("TRANSFORMATION", "N")):
        if col in raw.columns:
            raw = raw[raw[col] == val]
    if raw.empty:
        return pd.DataFrame()
    gov = _tidy(raw, {("S13",): "inv_gov"}, ("SECTOR",), sa=sa, price_bases=("V",))

    # Both legs, every edition, so the subtraction can pick a matching pair.
    legs = _tidy(raw[raw["ADJUSTMENT"] == "Y"], {("S13",): "g", ("S1",): "t"},
                 ("SECTOR",), sa="prefer", price_bases=("V",))
    legs_n = _tidy(raw[raw["ADJUSTMENT"] == "N"], {("S13",): "g", ("S1",): "t"},
                   ("SECTOR",), sa="prefer", price_bases=("V",))
    legs = _concat([legs, legs_n])
    priv = pd.DataFrame()
    if not legs.empty:
        keys = ["code", "ADJUSTMENT", "date"]
        w = legs.pivot_table(index=keys, columns="name", values="value", aggfunc="first")
        if {"g", "t"} <= set(w.columns):
            pairs = (w["t"] - w["g"]).dropna().rename("value").reset_index()
            # Per country, the adjusted pair if it exists, else the raw pair.
            rank = np.where(pairs["ADJUSTMENT"] == "Y", 0, 1)
            best = pd.Series(rank, index=pairs.index).groupby(pairs["code"]).transform("min")
            pairs = pairs[rank == best]
            template = (legs[legs["name"] == "t"].drop(columns=["value", "name"])
                        .drop_duplicates(subset=keys))
            priv = pairs.merge(template, on=keys, how="left")
            priv["name"] = "inv_priv"
    tidy = _concat([gov, priv])
    if tidy.empty:
        return tidy
    tidy["_source"] = _QSA_FLOW.rstrip(",")
    tidy["_series"] = tidy["name"].map({"inv_gov": "S13 P51G D",
                                        "inv_priv": "S1 P51G D less S13 P51G D"})
    return tidy


def qna_sector_gfcf(codes: Iterable[str] | None = None, *,
                    start: str = "1947", sa: str = "x13",
                    refresh: bool = False) -> pd.DataFrame:
    r"""Quarterly government and private gross fixed capital formation.

    Government investment (``inv_gov``, sector S13) and private investment
    (``inv_priv``) in millions of national currency at current prices, plus
    their volumes (``inv_gov_real``, ``inv_priv_real``) where the source
    publishes them.

    Two sources, chosen per country:

    1. ``DSD_NAMAIN1@DF_QNA_EXPENDITURE_GFCF_SECTOR`` -- the national-accounts
       table, current prices and volumes, S13 and S1W (all sectors other than
       government, i.e. private). About a dozen countries (BEL DEU FIN FRA ISL
       JPN KOR NLD NOR NZL USA ...), the United States from 1947.
    2. ``DSD_NASEC1@DF_QSA`` -- the quarterly sector accounts, current prices
       only, for every requested country the first source does not cover
       (about thirty more: AUS, CAN, GBR, ITA, MEX, ...). Private investment
       there is S1 less S13.

    Parameters
    ----------
    codes
        ISO3 codes. ``None`` asks for every country either source publishes.
    start
        First period requested (``"1947"`` = everything the OECD has).
    sa
        ``"x13"`` (default): the OECD's adjusted series where published,
        otherwise the unadjusted one seasonally adjusted here with X-13 (as
        :func:`qna_panel` does; most ``DF_QSA`` series are unadjusted).
        ``"prefer"``: the adjusted series where published, else the
        unadjusted one as is (flagged ``sa="none"`` in the meta).
    refresh
        Re-download instead of reading the on-disk cache.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(code, date)``, quarter-start dates; columns ``inv_gov``,
        ``inv_priv``, ``inv_gov_real``, ``inv_priv_real``. Volumes are
        chain-linked (``L``) where published, else fixed-base (``Q``), in
        millions of national currency of the reference year; there are none
        for countries served by ``DF_QSA``. ``attrs["meta"]`` holds one dict
        per country and variable (``source`` says which flow served it,
        ``sa`` how it is adjusted); ``attrs["missing"]`` lists the
        (country, variable) pairs asked for and not obtained, and why;
        ``attrs["sa_engines"]`` the X-13 engine used per adjusted series.

    Examples
    --------
    >>> g = qna_sector_gfcf(["USA", "MEX", "GBR"])          # doctest: +SKIP
    >>> (g["inv_gov"] / (g["inv_gov"] + g["inv_priv"])).groupby("code").last()  # doctest: +SKIP
    """
    _check_choice("sa", sa, ("x13", "prefer"))
    columns = _sector_gfcf_columns()
    source = f"{_GFCF_SECTOR_FLOW.rstrip(',')}; {_QSA_FLOW.rstrip(',')}"
    wanted = _norm_codes(codes)
    run = _Run()

    # 1. National-accounts sector table.
    key_of = lambda area: oecd_key(_NAMAIN1_DIMS, FREQ="Q", REF_AREA=area,  # noqa: E731
                                   SECTOR="S13+S1W", TRANSACTION="P51G")
    parts = []
    for chunk in (_chunks(wanted) if wanted is not None else [None]):
        parts.append(run.fetch(_GFCF_SECTOR_FLOW, key_of(chunk), chunk,
                               start=start, refresh=refresh))
    raw1 = _drop_aggregates(_concat(parts))
    tidy1 = _tidy_sector_flow(raw1, sa)
    have1 = (set(tidy1.loc[tidy1["name"] == "inv_gov", "code"])
             if not tidy1.empty else set())

    # 2. Sector accounts for everyone the first source left without inv_gov.
    qsa_key_of = lambda area: oecd_key(  # noqa: E731
        _NASEC1_DIMS, FREQ="Q", REF_AREA=area, SECTOR="S1+S13",
        ACCOUNTING_ENTRY="D", TRANSACTION="P51G", TRANSFORMATION="N")
    parts = []
    if wanted is None:
        parts.append(run.fetch(_QSA_FLOW, qsa_key_of(None), None,
                               start=start, refresh=refresh))
    else:
        rest = [c for c in wanted if c not in have1]
        for chunk in _chunks(rest):
            parts.append(run.fetch(_QSA_FLOW, qsa_key_of(chunk), chunk,
                                   start=start, refresh=refresh))
    raw2 = _drop_aggregates(_concat(parts))
    tidy2 = _tidy_qsa(raw2, sa)
    if not tidy2.empty:
        tidy2 = tidy2[~tidy2["code"].isin(have1)]

    tidy = _concat([tidy1, tidy2])
    if wanted is not None and not tidy.empty:
        tidy = tidy[tidy["code"].isin(wanted)]
    if tidy.empty:
        if run.failed:
            warnings.warn("qna_sector_gfcf: no data obtained; returning an empty frame",
                          UserWarning, stacklevel=2)
        out = _empty(columns, source)
        return _finish(out, meta=[], source=source, requested=wanted,
                       columns=columns, run=run, sa_engines={})

    engines: dict = {}
    if sa == "x13":
        tidy, engines = _adjust_unadjusted(tidy)

    # Volume measure per country: chain-linked when published, else fixed base.
    vol = tidy[tidy["PRICE_BASE"].isin(["L", "Q"])]
    vol_base = {c: ("L" if "L" in set(g["PRICE_BASE"]) else "Q")
                for c, g in vol.groupby("code")}
    nominal = tidy[tidy["PRICE_BASE"] == "V"]
    volume = vol[vol["PRICE_BASE"] == vol["code"].map(vol_base)].copy()
    volume["name"] = volume["name"] + "_real"
    both = pd.concat([nominal, volume], ignore_index=True)
    panel = _wide(both, columns)

    def _units(code, name, g):
        if not name.endswith("_real"):
            return _MONEY
        base = vol_base.get(code, "")
        # Kept as text: one country publishes "2009-2010".
        ref = (g["REF_YEAR_PRICE"].dropna().astype(str).str.strip()
               if "REF_YEAR_PRICE" in g.columns else pd.Series(dtype=str))
        ref = ref[(ref != "") & (ref.str.lower() != "nan")].str.replace(
            r"\.0$", "", regex=True)
        ref_s = f" ({ref.iloc[0]} prices)" if len(ref) else ""
        kind = "chain-linked volume" if base == "L" else "fixed-base volume"
        return f"millions of national currency, {kind}{ref_s}"

    meta = _meta_records(
        both,
        source_of=lambda c, n, g: str(g["_source"].iloc[0]),
        series_of=lambda c, n, g: (str(g["_series"].iloc[0])
                                   + f" {g['PRICE_BASE'].iloc[0]}"),
        units_of=_units)
    return _finish(panel, meta=meta, source=source, requested=wanted,
                   columns=columns, run=run, sa_engines=engines)


# --------------------------------------------------------------------------
# 2. Population and national-concept employment
# --------------------------------------------------------------------------

def qna_population(codes: Iterable[str] | None = None, *, start: str = "1947",
                   refresh: bool = False) -> pd.DataFrame:
    r"""Quarterly population and national-concept employment, thousands of persons.

    From ``DSD_NAMAIN1@DF_QNA_POP_EMPNC``: total population (``pop``, about
    43 countries, the United States from 1947) and employment on the
    national concept -- residents in work, wherever they work -- split into
    employees and self-employed (about 40 countries from 1980; none for the
    United States or Japan). The national concept differs from the domestic
    concept of ``qna_panel``'s ``emp`` by cross-border commuters, which
    matters for Luxembourg and little elsewhere.

    The OECD publishes most of these both adjusted and unadjusted; the
    adjusted series wins where it exists, and the three employment columns
    take one adjustment together so that
    ``emp_employees_nc + emp_selfemp_nc == emp_nc`` holds within each
    country. ``attrs["meta"]["sa"]`` says which was used.

    Parameters
    ----------
    codes
        ISO3 codes; ``None`` asks for every reference area (one request).
    start
        First period requested.
    refresh
        Re-download instead of reading the on-disk cache.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(code, date)``, quarter-start dates, columns ``pop``,
        ``emp_nc``, ``emp_employees_nc``, ``emp_selfemp_nc`` in thousands of
        persons. ``attrs`` as for :func:`qna_sector_gfcf`.

    Examples
    --------
    >>> p = qna_population(["USA", "MEX", "ESP"])            # doctest: +SKIP
    """
    columns = list(QNA_POPULATION)
    source = _POP_FLOW.rstrip(",")
    wanted = _norm_codes(codes)
    run = _Run()
    txns = "+".join(dict.fromkeys(t for t, _, _ in QNA_POPULATION.values()))
    parts = []
    for chunk in (_chunks(wanted) if wanted is not None else [None]):
        key = oecd_key(_NAMAIN1_DIMS, FREQ="Q", REF_AREA=chunk, TRANSACTION=txns)
        parts.append(run.fetch(_POP_FLOW, key, chunk, start=start, refresh=refresh))
    raw = _drop_aggregates(_concat(parts))
    tidy = pd.DataFrame()
    if not raw.empty:
        raw = raw[raw["TIME_PERIOD"].notna()]
        if "SECTOR" in raw.columns and (raw["SECTOR"] == "S1").any():
            raw = raw[raw["SECTOR"] == "S1"]
        lookup = {(t, a): n for n, (t, a, _) in QNA_POPULATION.items()}
        family = {n: ("emp" if n != "pop" else "pop") for n in QNA_POPULATION}
        tidy = _tidy(raw, lookup, ("TRANSACTION", "ACTIVITY"), sa="prefer",
                     units=("PS",), price_bases=("_Z",), mult_target=3,
                     sa_family=family)
    if wanted is not None and not tidy.empty:
        tidy = tidy[tidy["code"].isin(wanted)]
    if tidy.empty:
        if run.failed:
            warnings.warn("qna_population: no data obtained; returning an empty frame",
                          UserWarning, stacklevel=2)
        return _finish(_empty(columns, source), meta=[], source=source,
                       requested=wanted, columns=columns, run=run)
    panel = _wide(tidy, columns)
    meta = _meta_records(
        tidy, source_of=lambda c, n, g: source,
        series_of=lambda c, n, g: f"{QNA_POPULATION[n][0]} {QNA_POPULATION[n][1]} PS",
        units_of=lambda c, n, g: _THOUSANDS)
    return _finish(panel, meta=meta, source=source, requested=wanted,
                   columns=columns, run=run)


# --------------------------------------------------------------------------
# Shared reader for the TPS flows (LFS, OLAB): no price base, FREQ last
# --------------------------------------------------------------------------

def _tidy_tps(raw: pd.DataFrame, lookup: dict[tuple, str], dims: tuple[str, ...],
              *, freq: str, mult_target: dict[str, int]) -> pd.DataFrame:
    """Select one series per (code, name): published-adjusted first, else raw.

    ``mult_target`` maps UNIT_MEASURE to the power of ten the output carries
    (3 = thousands for persons, 0 for rates); the published UNIT_MULT is
    converted to it.
    """
    need = {"REF_AREA", "TIME_PERIOD", "OBS_VALUE", "ADJUSTMENT", "UNIT_MEASURE",
            *dims}
    if raw.empty or not need.issubset(raw.columns):
        return pd.DataFrame()
    df = raw[raw["TIME_PERIOD"].notna()].copy()
    if "TRANSFORMATION" in df.columns:
        # Levels and rates only: growth rates (G1 etc.) share UNIT_MEASURE.
        df = df[df["TRANSFORMATION"].isin(["_Z", "N"]) | df["TRANSFORMATION"].isna()]
    if "FREQ" in df.columns:
        df = df[df["FREQ"] == freq]
    if "SEX" in df.columns:
        df = df[df["SEX"] == "_T"]
    df["name"] = [lookup.get(k) for k in zip(*(df[d] for d in dims))]
    df = df[df["name"].notna()]
    df["value"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["value"])
    if df.empty:
        return pd.DataFrame()
    mult = (pd.to_numeric(df["UNIT_MULT"], errors="coerce")
            if "UNIT_MULT" in df.columns else pd.Series(np.nan, index=df.index))
    target = df["UNIT_MEASURE"].map(mult_target).astype(float)
    target, mult = target.fillna(mult), mult.fillna(target)
    df["value"] = df["value"] * np.power(10.0, (mult - target).fillna(0.0))
    df["date"] = _period_start(df["TIME_PERIOD"], freq)
    df = df.rename(columns={"REF_AREA": "code"})
    df["_rank"] = np.where(df["ADJUSTMENT"] == "Y", 0, 1)
    best = df.groupby(["code", "name"])["_rank"].transform("min")
    df = df[df["_rank"] == best]
    return (df.sort_values(["code", "name", "date"])
              .drop_duplicates(subset=["code", "name", "date"], keep="first"))


def _tps_codes(flow: str, key_open: str, wanted: list[str] | None,
               refresh: bool, *, run: _Run,
               fallback: Sequence[str]) -> list[str]:
    """Countries to request: the caller's, or the availability answer for the key.

    When the availability call fails (or lists nothing), ``fallback`` is used
    and chunked like any other list, rather than sending one open request
    for every area, which is the request most likely to be throttled.
    """
    if wanted is not None:
        return wanted
    try:
        payload = _availability(flow, key_open, refresh=refresh)
    except Exception as exc:  # noqa: BLE001 - never raise on a provider failure
        payload = {}
        status = f"{type(exc).__name__}: {exc}"[:200]
    else:
        status = "ok" if payload else "failed"
    areas = _ref_areas(payload)
    run.requests.append({"flow": flow.rstrip(",") + " (availability)",
                         "key": key_open, "status": status if areas or status != "ok"
                         else "no areas", "rows": len(areas)})
    if areas:
        return areas
    warnings.warn(f"OECD availability call failed for {flow.rstrip(',')}; "
                  f"requesting the {len(fallback)} areas known to publish it",
                  UserWarning, stacklevel=3)
    return list(fallback)


# --------------------------------------------------------------------------
# 3. Labour-force survey indicators
# --------------------------------------------------------------------------

def oecd_lfs_panel(codes: Iterable[str] | None = None, *, freq: str = "Q",
                   start: str = "1950", refresh: bool = False) -> pd.DataFrame:
    r"""Labour-force survey panel: unemployment, participation, employment rates and levels.

    From ``OECD.SDD.TPS,DSD_LFS@DF_IALFS_INDIC`` (the OECD's infra-annual
    labour statistics), total of both sexes:

    ============  ======================================================
    urate         unemployment rate 15+, % (harmonised ``UNE_LF_M`` where
                  published, else the survey rate ``UNE_LF``)
    urate_1564    unemployment rate 15-64, %
    emp_lfs       employment 15+, thousands (survey, not national accounts)
    lf            labour force 15+, thousands
    wap           working-age population 15+, thousands
    wap_1564      working-age population 15-64, thousands
    prate_1564    participation rate 15-64, % of population 15-64
    epop_1564     employment rate 15-64, % of population 15-64
    ============  ======================================================

    About 41-45 countries quarterly, the United States, Canada, Japan and
    Australia from 1955; ``freq="M"`` gives the monthly variant (about 39
    countries for the unemployment rate). Each series is the seasonally
    adjusted one where the OECD publishes it and the unadjusted one
    otherwise (working-age population is often unadjusted only); the choice
    is recorded per country and variable in ``attrs["meta"]["sa"]``.

    Parameters
    ----------
    codes
        ISO3 codes. ``None`` asks the availability endpoint which areas
        publish the key, then requests them ten at a time.
    freq
        ``"Q"`` (quarter-start dates) or ``"M"`` (month-start dates).
    start
        First period requested.
    refresh
        Re-download instead of reading the on-disk cache.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(code, date)``; columns as above. ``attrs`` as for
        :func:`qna_sector_gfcf`.

    Raises
    ------
    ValueError
        If ``freq`` is not ``"Q"`` or ``"M"``.

    Examples
    --------
    >>> lfs = oecd_lfs_panel(["USA", "MEX", "ESP"])          # doctest: +SKIP
    >>> lfs_m = oecd_lfs_panel(["USA"], freq="M")            # doctest: +SKIP
    """
    _check_choice("freq", freq, ("Q", "M"))
    columns = list(LFS_INDICATORS)
    source = _LFS_FLOW
    wanted = _norm_codes(codes)
    run = _Run()
    measures = "+".join(dict.fromkeys(
        [m for m, _, _, _ in LFS_INDICATORS.values()] + [_URATE_FALLBACK[0]]))
    units = "+".join(dict.fromkeys(u for _, _, u, _ in LFS_INDICATORS.values()))
    ages = "+".join(dict.fromkeys(a for _, a, _, _ in LFS_INDICATORS.values()))
    key_of = lambda area: oecd_key(  # noqa: E731
        _LFS_DIMS, REF_AREA=area, MEASURE=measures, UNIT_MEASURE=units,
        TRANSFORMATION="_Z", SEX="_T", AGE=ages, ACTIVITY="_Z", FREQ=freq)
    areas = _tps_codes(_LFS_FLOW, key_of(None), wanted, refresh, run=run,
                       fallback=_LFS_AREAS)
    parts = []
    for chunk in _chunks(areas):
        parts.append(run.fetch(_LFS_FLOW, key_of(chunk), chunk,
                               start=start, refresh=refresh))
    raw = _drop_aggregates(_concat(parts))
    lookup = {(m, a, u): n for n, (m, a, u, _) in LFS_INDICATORS.items()}
    lookup[_URATE_FALLBACK] = "_urate_fallback"
    tidy = _tidy_tps(raw, lookup, ("MEASURE", "AGE", "UNIT_MEASURE"), freq=freq,
                     mult_target={"PS": 3, "PT_LF_SUB": 0, "PT_WAP_SUB": 0})
    if not tidy.empty:
        # urate: the harmonised series where a country has it, else the survey rate.
        has_main = set(tidy.loc[tidy["name"] == "urate", "code"])
        fb = tidy["name"] == "_urate_fallback"
        tidy.loc[fb & ~tidy["code"].isin(has_main), "name"] = "urate"
        tidy = tidy[tidy["name"] != "_urate_fallback"]
    if wanted is not None and not tidy.empty:
        tidy = tidy[tidy["code"].isin(wanted)]
    if tidy.empty:
        if run.failed:
            warnings.warn("oecd_lfs_panel: no data obtained; returning an empty frame",
                          UserWarning, stacklevel=2)
        return _finish(_empty(columns, source), meta=[], source=source,
                       requested=wanted, columns=columns, run=run)
    panel = _wide(tidy, columns)
    meta = _meta_records(
        tidy, source_of=lambda c, n, g: source,
        series_of=lambda c, n, g: (f"{g['MEASURE'].iloc[0]} {g['AGE'].iloc[0]} "
                                   f"{g['UNIT_MEASURE'].iloc[0]} {freq}"),
        units_of=lambda c, n, g: (_THOUSANDS if g["UNIT_MEASURE"].iloc[0] == "PS"
                                  else "percent"))
    return _finish(panel, meta=meta, source=source, requested=wanted,
                   columns=columns, run=run, freq=freq)


# --------------------------------------------------------------------------
# 4. Registered vacancies and unemployment
# --------------------------------------------------------------------------

def oecd_vacancies(codes: Iterable[str] | None = None, *, freq: str = "Q",
                   start: str = "1950", refresh: bool = False) -> pd.DataFrame:
    r"""Registered vacancies and registered unemployment, thousands of persons.

    From ``OECD.SDD.TPS,DSD_OLAB@DF_OIALAB_INDIC``, whole economy
    (``SECTOR=S1``): the stock of unfilled vacancies registered with the
    public employment service (``vacancies``), the flow of newly notified
    vacancies (``vacancies_new``, only a handful of countries: BGR FRA JPN
    TUR), and registered unemployment (``reg_unemp``). Germany from 1955,
    the United Kingdom 1958-2023, the United States from 2001 (JOLTS);
    about 18 countries quarterly and 20 monthly. These are administrative
    counts, not survey estimates: their level depends on how much of the
    market goes through the public agency, so compare them across time
    within a country (a Beveridge curve against ``oecd_lfs_panel``'s
    ``urate``), not across countries.

    Parameters
    ----------
    codes
        ISO3 codes; ``None`` asks for every reference area (one request).
    freq
        ``"Q"`` or ``"M"``.
    start
        First period requested.
    refresh
        Re-download instead of reading the on-disk cache.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(code, date)``; columns ``vacancies``,
        ``vacancies_new``, ``reg_unemp`` in thousands of persons, seasonally
        adjusted where the OECD publishes it (see ``attrs["meta"]["sa"]``).

    Raises
    ------
    ValueError
        If ``freq`` is not ``"Q"`` or ``"M"``.

    Examples
    --------
    >>> v = oecd_vacancies(["DEU", "USA", "GBR"])            # doctest: +SKIP
    """
    _check_choice("freq", freq, ("Q", "M"))
    columns = list(OECD_VACANCIES)
    source = _OLAB_FLOW.rstrip(",")
    wanted = _norm_codes(codes)
    run = _Run()
    measures = "+".join(m for m, _ in OECD_VACANCIES.values())
    parts = []
    for chunk in (_chunks(wanted) if wanted is not None else [None]):
        key = oecd_key(_OLAB_DIMS, REF_AREA=chunk, MEASURE=measures,
                       UNIT_MEASURE="PS", TRANSFORMATION="_Z", SECTOR="S1",
                       FREQ=freq)
        parts.append(run.fetch(_OLAB_FLOW, key, chunk, start=start, refresh=refresh))
    raw = _drop_aggregates(_concat(parts))
    lookup = {(m,): n for n, (m, _) in OECD_VACANCIES.items()}
    if not raw.empty and "SECTOR" in raw.columns:
        raw = raw[raw["SECTOR"] == "S1"]
    tidy = _tidy_tps(raw, lookup, ("MEASURE",), freq=freq, mult_target={"PS": 3})
    if wanted is not None and not tidy.empty:
        tidy = tidy[tidy["code"].isin(wanted)]
    if tidy.empty:
        if run.failed:
            warnings.warn("oecd_vacancies: no data obtained; returning an empty frame",
                          UserWarning, stacklevel=2)
        return _finish(_empty(columns, source), meta=[], source=source,
                       requested=wanted, columns=columns, run=run)
    panel = _wide(tidy, columns)
    meta = _meta_records(
        tidy, source_of=lambda c, n, g: source,
        series_of=lambda c, n, g: f"{OECD_VACANCIES[n][0]} PS S1 {freq}",
        units_of=lambda c, n, g: _THOUSANDS)
    return _finish(panel, meta=meta, source=source, requested=wanted,
                   columns=columns, run=run, freq=freq)


def qna_extras_meta(panel: pd.DataFrame) -> pd.DataFrame:
    """``attrs["meta"]`` of any panel from this module, as a DataFrame.

    The records are stored as a tuple of plain dicts (a DataFrame in
    ``.attrs`` breaks ``pd.concat``); this is the way to read them back.
    """
    return pd.DataFrame(list(panel.attrs.get("meta", ())), columns=list(_META_KEYS))


__all__ = ["LFS_INDICATORS", "OECD_PAUSE", "OECD_VACANCIES", "QNA_POPULATION",
           "SECTOR_GFCF", "oecd_lfs_panel", "oecd_vacancies", "qna_extras_meta",
           "qna_population", "qna_sector_gfcf"]
