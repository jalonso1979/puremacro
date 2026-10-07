r"""One-call cross-country MONTHLY panel from the OECD short-term statistics.

``stes_panel(["USA", "MEX", "DEU"])`` returns a wide monthly panel indexed by
``(code, date)``, the date being the first day of the month, with one column
per concept of :data:`STES_CONCEPTS`::

    ip  ip_mfg  ip_constr  retail_vol            production and sales
    rate_on  rate_3m  rate_10y  share_price      financial markets
    reer_cpi  fx_usd                             exchange rates
    bci  cci  cli                                surveys and the leading indicator
    m1  m3                                       money
    urate  urate_nsa  emp  lf  prate  epop       labour force survey
    vacancies  reg_unemp                         registers
    cpi  cpi_food  cpi_energy  cpi_core          national consumer prices
    hicp  hicp_food  hicp_energy  hicp_core      harmonised consumer prices

All of it comes in **levels**: indices, percentages and head counts exactly
as the OECD publishes them (persons rescaled to thousands). Logs, growth
rates and any seasonal adjustment the OECD does not publish are left to the
caller, because that is where a course wants students to do them.

Why a dedicated builder
-----------------------
The monthly OECD material is scattered across nine dataflows in two
agencies (``OECD.SDD.STES`` for production, markets, surveys and money;
``OECD.SDD.TPS`` for labour and prices), each with its own dimension order
and its own way of producing duplicate rows when a dimension is left open
(growth-rate transformations in the CLI flow, sectors in the registers
flow, the SA and NSA variants everywhere). The registry below pins every
dimension of every concept to the single series the probe of 6 October
2026 verified, so that a ``(code, month)`` pair has exactly one value per
concept; the builder checks that and drops a concept rather than return
an ambiguous column.

Consumer prices need one more step. In January 2026 the EU and EEA members
moved their CPI to COICOP 2018, and the OECD put the new classification in
a separate flow: the COICOP-1999 flow stops at 2025-12 for those countries
(and earlier for JPN, MEX, CHL, CRI, CAN), while the COICOP-2018 flow
carries the full back history but lacks AUS, BRA, CHN, COL, DEU, GBR, IDN,
IND, ISR, KOR and USA. Each price concept is therefore the cell-by-cell
union of the two, COICOP 2018 first: an index month comes from the 1999
flow only when the 2018 flow has nothing for that country and month. Both
are 2015 = 100; on the 75,616 cells they share the median relative gap is
2e-5, but where the 2018 classification redrew an aggregate the gap in
the month the 2018 series starts reaches 13 % (BGR energy), 5 % (NLD core)
and 4 % (BGR core). Months taken from the 1999 flow *before* that month
are therefore ratio-spliced onto the 2018 level, so no reclassification
gap turns into one month's inflation; the factors are in
``attrs["meta"]`` (``splice_factors``).
The HICP columns take the 1999 flow only for
the EU, EEA, Swiss, British and Turkish series: the 1999 flow also carries
"HICP"-methodology series for the USA and others that are not HICPs.

Mechanics: one SDMX request per dataflow (ten in all when every concept is
asked for), all measures of a flow in one key, so the response caches on
disk and a later call for a different subset of concepts reuses it. The
transport is :func:`puremacro.fetch._oecd_sdmx.oecd_csv` (urllib, paced,
retried on HTTP 429, cached under ``~/.cache/puremacro``); a failed request
never raises, it shows up in ``attrs["missing"]`` and ``attrs["requests"]``.
A concept whose *primary* flow failed is dropped even if its fallback flow
answered, so that a throttled COICOP-2018 request can never pass off a
COICOP-1999-only price column (EU series stopping at 2025-12) as complete.

Source: OECD Data Explorer SDMX API (``sdmx.oecd.org``), dataflows
``OECD.SDD.STES`` ``DSD_STES@DF_INDSERV`` 4.3, ``DF_FINMARK`` 4.0,
``DF_CLI`` 4.1, ``DF_MONAGG`` 4.0, ``DSD_KEI@DF_KEI`` 4.0 and
``OECD.SDD.TPS`` ``DSD_LFS@DF_IALFS_UNE_M`` 1.0, ``DSD_LFS@DF_IALFS_INDIC``
1.0, ``DSD_OLAB@DF_OIALAB_INDIC`` 1.1,
``DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL`` 1.0, ``DSD_PRICES@DF_PRICES_ALL`` 1.0.
"""
from __future__ import annotations

import re
import urllib.error
import warnings
from datetime import datetime, timezone
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd

from ._oecd_sdmx import OECD_AGGREGATES, oecd_csv


class StesConcept(NamedTuple):
    """Where one monthly concept lives and what its numbers mean."""

    flow: str
    """Agency and dataflow, ``"OECD.SDD.STES,DSD_STES@DF_INDSERV"``."""
    version: str
    """Dataflow version the key was verified against, ``"4.3"``."""
    key: str
    """Pinned SDMX key with the reference area (first position) left open."""
    unit: str
    """Unit of the returned values, after any rescaling to ``unit_mult``."""
    unit_mult: int
    """Power of ten the returned values are expressed in (3 = thousands)."""
    adjustment: str
    """Seasonal / calendar adjustment as published."""
    sa: bool | None
    """True when seasonally adjusted; None when the notion does not apply."""
    base_period: str | None
    """Reference period of an index (100 = that year), else None."""
    label: str
    """Plain-English description."""
    fallback: tuple[tuple[str, str], ...] = ()
    """Further ``(flow, version)`` pairs with the same key, used cell by cell
    after the primary flow (COICOP 1999 behind COICOP 2018)."""
    fallback_areas: tuple[str, ...] = ()
    """When non-empty, the only reference areas a fallback flow may supply
    (the HICP columns take the 1999 flow for European economies only)."""


_STES = "OECD.SDD.STES,DSD_STES@"
_INDSERV = _STES + "DF_INDSERV"
_FINMARK = _STES + "DF_FINMARK"
_CLI = _STES + "DF_CLI"
_MONAGG = _STES + "DF_MONAGG"
_KEI = "OECD.SDD.STES,DSD_KEI@DF_KEI"
_UNE_M = "OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M"
_LFS = "OECD.SDD.TPS,DSD_LFS@DF_IALFS_INDIC"
_OLAB = "OECD.SDD.TPS,DSD_OLAB@DF_OIALAB_INDIC"
_C2018 = "OECD.SDD.TPS,DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL"
_C1999 = "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL"

#: Dimension order of each dataflow's key, as the DSD declares it. The
#: filter that picks a concept's rows out of a multi-measure response reads
#: the key positions against these names.
_FLOW_DIMS: dict[str, tuple[str, ...]] = {
    _INDSERV: ("REF_AREA", "FREQ", "MEASURE", "UNIT_MEASURE", "ACTIVITY",
               "ADJUSTMENT", "TRANSFORMATION", "TIME_HORIZ", "METHODOLOGY"),
    _FINMARK: ("REF_AREA", "FREQ", "MEASURE", "UNIT_MEASURE", "ACTIVITY",
               "ADJUSTMENT", "TRANSFORMATION", "TIME_HORIZ", "METHODOLOGY"),
    _CLI: ("REF_AREA", "FREQ", "MEASURE", "UNIT_MEASURE", "ACTIVITY",
           "ADJUSTMENT", "TRANSFORMATION", "TIME_HORIZ", "METHODOLOGY"),
    _MONAGG: ("REF_AREA", "FREQ", "MEASURE", "UNIT_MEASURE", "ACTIVITY",
              "ADJUSTMENT", "TRANSFORMATION", "TIME_HORIZ", "METHODOLOGY"),
    _KEI: ("REF_AREA", "FREQ", "MEASURE", "UNIT_MEASURE", "ACTIVITY",
           "ADJUSTMENT", "TRANSFORMATION"),
    _UNE_M: ("REF_AREA", "MEASURE", "UNIT_MEASURE", "TRANSFORMATION",
             "ADJUSTMENT", "SEX", "AGE", "ACTIVITY", "FREQ"),
    _LFS: ("REF_AREA", "MEASURE", "UNIT_MEASURE", "TRANSFORMATION",
           "ADJUSTMENT", "SEX", "AGE", "ACTIVITY", "FREQ"),
    _OLAB: ("REF_AREA", "MEASURE", "UNIT_MEASURE", "TRANSFORMATION",
            "ADJUSTMENT", "SECTOR", "FREQ"),
    _C2018: ("REF_AREA", "FREQ", "METHODOLOGY", "MEASURE", "UNIT_MEASURE",
             "EXPENDITURE", "ADJUSTMENT", "TRANSFORMATION"),
    _C1999: ("REF_AREA", "FREQ", "METHODOLOGY", "MEASURE", "UNIT_MEASURE",
             "EXPENDITURE", "ADJUSTMENT", "TRANSFORMATION"),
}

_IX15 = "index, 2015 = 100"
_SA = "calendar and seasonally adjusted"
_NSA = "not seasonally adjusted"
_NA = "not applicable"
_PRICES_FALLBACK = ((_C1999, "1.0"),)

#: Economies that publish a Harmonised Index of Consumer Prices: the EU 27
#: plus the EEA / candidate members Eurostat covers. The COICOP-1999 flow
#: also carries "HICP"-methodology series for the USA (2001-12 to 2024-12)
#: and others; those are not HICPs and are not used as fallback.
_HICP_AREAS: tuple[str, ...] = (
    "AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST", "FIN",
    "FRA", "GRC", "HRV", "HUN", "IRL", "ITA", "LTU", "LUX", "LVA", "MLT",
    "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "SWE",
    "CHE", "GBR", "ISL", "NOR", "TUR",
)


def _ip(measure: str, activity: str, label: str) -> StesConcept:
    # INDSERV: REF_AREA.FREQ.MEASURE.UNIT_MEASURE.ACTIVITY.ADJUSTMENT.TRANSFORMATION.TIME_HORIZ.METHODOLOGY
    return StesConcept(_INDSERV, "4.3", f".M.{measure}.IX.{activity}.Y._Z._Z.N",
                       _IX15, 0, _SA, True, "2015", label)


def _fin(measure: str, unit_code: str, unit: str, base: str | None,
         label: str) -> StesConcept:
    # FINMARK: same nine dimensions as INDSERV; everything but MEASURE and UNIT is _Z / N.
    return StesConcept(_FINMARK, "4.0", f".M.{measure}.{unit_code}._Z._Z._Z._Z.N",
                       unit, 0, _NA, None, base, label)


def _survey(measure: str, label: str) -> StesConcept:
    # CLI: ADJUSTMENT=AA (amplitude adjusted), TRANSFORMATION=IX, METHODOLOGY=H.
    # Leaving TRANSFORMATION open adds the GY growth rows and duplicates months.
    return StesConcept(_CLI, "4.1", f".M.{measure}.IX._Z.AA.IX._Z.H",
                       "index, long-run average = 100", 0,
                       "amplitude adjusted (seasonally adjusted, smoothed)", True,
                       None, label)


def _money(measure: str, label: str) -> StesConcept:
    # MONAGG: NSA index; the SA variant exists for only 12-15 countries.
    return StesConcept(_MONAGG, "4.0", f".M.{measure}.IX._Z.N._Z._Z.N",
                       _IX15, 0, _NSA, False, "2015", label)


def _lfs(flow: str, measure: str, unit_code: str, adj: str, unit: str,
         unit_mult: int, label: str) -> StesConcept:
    # IALFS: REF_AREA.MEASURE.UNIT_MEASURE.TRANSFORMATION.ADJUSTMENT.SEX.AGE.ACTIVITY.FREQ
    return StesConcept(flow, "1.0", f".{measure}.{unit_code}._Z.{adj}._T.Y_GE15._Z.M",
                       unit, unit_mult, _SA if adj == "Y" else _NSA, adj == "Y",
                       None, label)


def _olab(measure: str, label: str) -> StesConcept:
    # OLAB: REF_AREA.MEASURE.UNIT_MEASURE.TRANSFORMATION.ADJUSTMENT.SECTOR.FREQ.
    # SECTOR pinned to S1 (whole economy); S1D / S1ZS duplicate a few countries.
    return StesConcept(_OLAB, "1.1", f".{measure}.PS._Z.Y.S1.M",
                       "thousands of persons", 3, _SA, True, None, label)


def _price(methodology: str, expenditure: str, label: str) -> StesConcept:
    # PRICES: REF_AREA.FREQ.METHODOLOGY.MEASURE.UNIT_MEASURE.EXPENDITURE.ADJUSTMENT.TRANSFORMATION.
    # ADJUSTMENT=N: the S variant exists for five countries only.
    return StesConcept(_C2018, "1.0", f".M.{methodology}.CPI.IX.{expenditure}.N._Z",
                       _IX15, 0, _NSA, False, "2015", label,
                       fallback=_PRICES_FALLBACK,
                       fallback_areas=_HICP_AREAS if methodology == "HICP" else ())


#: concept -> :class:`StesConcept`. Every key was verified against the live
#: OECD API on 6 October 2026 to return one series per country with no
#: duplicate months. The order is the column order of :func:`stes_panel`.
STES_CONCEPTS: dict[str, StesConcept] = {
    "ip":          _ip("PRVM", "BTE", "Industrial production, industry excl. construction (B-E)"),
    "ip_mfg":      _ip("PRVM", "C", "Industrial production, manufacturing (C)"),
    "ip_constr":   _ip("PRVM", "F", "Production in construction (F)"),
    "retail_vol":  _ip("TOVM", "G47", "Retail trade volume (G47)"),
    "rate_on":     _fin("IRSTCI", "PA", "percent per annum", None,
                        "Overnight / call money interbank rate"),
    "rate_3m":     _fin("IR3TIB", "PA", "percent per annum", None,
                        "Three-month interbank rate"),
    "rate_10y":    _fin("IRLT", "PA", "percent per annum", None,
                        "Long-term (10-year) government bond yield"),
    "share_price": _fin("SHARE", "IX", _IX15, "2015", "Share price index"),
    "reer_cpi":    _fin("CCRE", "IX", _IX15, "2015",
                        "Real effective exchange rate, CPI-based (up = appreciation)"),
    # KEI: REF_AREA.FREQ.MEASURE.UNIT_MEASURE.ACTIVITY.ADJUSTMENT.TRANSFORMATION.
    # FINMARK's CC lacks every euro-area member; KEI splices legacy currency -> EUR.
    "fx_usd":      StesConcept(_KEI, "4.0", ".M.CC.XDC_USD._Z._Z._Z",
                               "national currency per US dollar, monthly average", 0,
                               _NA, None, None,
                               "Nominal exchange rate against the US dollar"),
    "bci":         _survey("BCICP", "Business confidence indicator"),
    "cci":         _survey("CCICP", "Consumer confidence indicator"),
    "cli":         _survey("LI", "Composite leading indicator"),
    "m1":          _money("MANM", "Narrow money (M1)"),
    "m3":          _money("MABM", "Broad money (M3)"),
    "urate":       _lfs(_UNE_M, "UNE_LF_M", "PT_LF_SUB", "Y", "percent of labour force", 0,
                        "Unemployment rate, 15+"),
    "urate_nsa":   _lfs(_UNE_M, "UNE_LF_M", "PT_LF_SUB", "N", "percent of labour force", 0,
                        "Unemployment rate, 15+, not seasonally adjusted"),
    "emp":         _lfs(_LFS, "EMP", "PS", "Y", "thousands of persons", 3, "Employment, 15+"),
    "lf":          _lfs(_LFS, "LF", "PS", "Y", "thousands of persons", 3, "Labour force, 15+"),
    "prate":       _lfs(_LFS, "LF_WAP", "PT_WAP_SUB", "Y",
                        "percent of working-age population", 0,
                        "Labour force participation rate, 15+"),
    "epop":        _lfs(_LFS, "EMP_WAP", "PT_WAP_SUB", "Y",
                        "percent of working-age population", 0,
                        "Employment-to-population ratio, 15+"),
    "vacancies":   _olab("VAC_U", "Unfilled job vacancies (stock)"),
    "reg_unemp":   _olab("REG_UNE", "Registered unemployed"),
    "cpi":         _price("N", "_T", "Consumer prices, all items (national CPI)"),
    "cpi_food":    _price("N", "CP01", "CPI, food and non-alcoholic beverages"),
    "cpi_energy":  _price("N", "CP045_0722", "CPI, energy (household energy + motor fuels)"),
    "cpi_core":    _price("N", "_TXCP01_NRG", "CPI, all items excluding food and energy"),
    "hicp":        _price("HICP", "_T", "HICP, all items"),
    "hicp_food":   _price("HICP", "CP01", "HICP, food and non-alcoholic beverages"),
    "hicp_energy": _price("HICP", "CP045_0722", "HICP, energy"),
    "hicp_core":   _price("HICP", "_TXNRG_01_02",
                          "HICP excluding energy, food, alcohol and tobacco"),
}

_META_COLS = ["concept", "flow", "key", "unit", "unit_mult", "adjustment", "sa",
              "base_period", "label", "n", "n_countries", "first", "last",
              "countries_fallback", "splice_factors", "flows_failed",
              "stale_countries"]

#: A country whose last month is more than this many months before the
#: column's last month is listed in ``meta["stale_countries"]``.
_STALE_MONTHS = 6

_MONTH = re.compile(r"^\d{4}-\d{2}$")
_START = re.compile(r"^\d{4}(-\d{2})?$")


def _get_csv(agency_flow: str, key: str, **kw) -> pd.DataFrame:
    """Module-level seam around :func:`oecd_csv`; tests patch this name."""
    return oecd_csv(agency_flow, key, **kw)


def _fetch(agency_flow: str, key: str, **kw) -> pd.DataFrame:
    """:func:`_get_csv` that never raises on a provider failure.

    ``oecd_csv`` already turns HTTP and transport errors into an empty frame
    with ``attrs["status"]``; this guard keeps that promise even if the
    transport (or a patched seam) lets one through.
    """
    try:
        frame = _get_csv(agency_flow, key, **kw)
    except urllib.error.HTTPError as exc:
        frame = pd.DataFrame()
        frame.attrs["status"] = f"HTTP {exc.code}"
    except (OSError, ValueError) as exc:     # URLError, timeouts, bad payloads
        frame = pd.DataFrame()
        frame.attrs["status"] = type(exc).__name__
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame()
        frame.attrs["status"] = "unexpected response"
    return frame


def _flows_of(spec: StesConcept) -> tuple[tuple[str, str], ...]:
    return ((spec.flow, spec.version),) + tuple(spec.fallback)


def _merged_key(flow: str, codes: list[str] | None) -> str:
    """The one key that asks a flow for every registry concept it serves.

    Built over the whole registry, not just the concepts requested, so that a
    later call for a different subset hits the same cached response.
    """
    keys = [spec.key for spec in STES_CONCEPTS.values()
            if any(f == flow for f, _ in _flows_of(spec))]
    parts = [k.split(".") for k in keys]
    width = len(_FLOW_DIMS[flow])
    merged = []
    for i in range(width):
        seen: list[str] = []
        for p in parts:
            if p[i] and p[i] not in seen:
                seen.append(p[i])
        merged.append("+".join(seen))
    merged[0] = "+".join(codes) if codes else ""
    return ".".join(merged)


def _pick(raw: pd.DataFrame, flow: str, spec: StesConcept) -> pd.DataFrame:
    """Rows of ``raw`` that match every pinned position of ``spec.key``."""
    dims = _FLOW_DIMS[flow]
    mask = pd.Series(True, index=raw.index)
    for dim, value in zip(dims, spec.key.split(".")):
        if not value or dim == "REF_AREA":
            continue
        if dim not in raw.columns:
            return raw.iloc[0:0]
        mask &= raw[dim].astype(str) == value
    return raw.loc[mask]


def _tidy(rows: pd.DataFrame, unit_mult: int, concept: str = "") -> pd.DataFrame:
    """``REF_AREA, TIME_PERIOD, OBS_VALUE`` -> ``code, date, value`` in ``unit_mult``.

    A blank ``UNIT_MULT`` is read as the country's own most common multiplier
    in the response, and failing that as the target ``unit_mult`` (no
    rescaling), never as 0: a blank on a series published in thousands would
    otherwise divide it by a thousand.
    """
    if rows.empty:
        return pd.DataFrame(columns=["code", "date", "value"])
    period = rows["TIME_PERIOD"].astype(str)
    rows = rows.loc[period.str.match(_MONTH)]
    value = pd.to_numeric(rows["OBS_VALUE"], errors="coerce")
    if "UNIT_MULT" in rows.columns:
        mult = pd.to_numeric(rows["UNIT_MULT"], errors="coerce")
        blank = mult.isna() & value.notna()
        if blank.any() and mult.notna().any():
            modal = (mult.groupby(rows["REF_AREA"].astype(str))
                         .agg(lambda m: m.mode().iloc[0] if m.notna().any() else np.nan))
            mult = mult.fillna(rows["REF_AREA"].astype(str).map(modal))
        if blank.any() and unit_mult != 0:
            warnings.warn(f"stes_panel: {concept} has {int(blank.sum())} rows with a blank "
                          "UNIT_MULT; read as the country's usual multiplier, else as "
                          f"10**{unit_mult}", stacklevel=3)
        mult = mult.fillna(unit_mult)
        value = value * np.power(10.0, mult - unit_mult)
    out = pd.DataFrame({
        "code": rows["REF_AREA"].astype(str).to_numpy(),
        "date": pd.to_datetime(rows["TIME_PERIOD"].astype(str) + "-01",
                               format="%Y-%m-%d").to_numpy(),
        "value": value.to_numpy(dtype=float),
    })
    return out.dropna(subset=["value"])


def _ratio_splice(pieces: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], list[str]]:
    """Rescale fallback months that precede a country's primary series.

    Both price flows are 2015 = 100, but where COICOP 2018 reclassified an
    aggregate (energy, core) the two indices differ by up to 13 % in the
    month the primary series starts, and a cell-wise union would put that
    gap into one month's inflation. Fallback cells *before* the primary's
    first month are multiplied by primary / fallback in that month, so the
    level is continuous and the growth rates are the fallback's own. Cells
    with no overlap month to anchor on are left as published.
    """
    primary = pieces[0]
    first = primary.groupby("code")["date"].min()
    level = primary.set_index(["code", "date"])["value"]
    out = [primary]
    notes: list[str] = []
    for fb in pieces[1:]:
        fb = fb.copy()
        fb_level = fb.set_index(["code", "date"])["value"]
        for code, start in first.items():
            anchor = (code, start)
            if anchor not in fb_level.index:
                continue
            before = (fb["code"] == code) & (fb["date"] < start)
            if not before.any() or fb_level[anchor] == 0:
                continue
            factor = float(level[anchor] / fb_level[anchor])
            if abs(factor - 1.0) > 1e-9:
                fb.loc[before, "value"] = fb.loc[before, "value"] * factor
                notes.append(f"{code}:{start:%Y-%m}:{factor:.6g}")
        out.append(fb)
    return out, notes


def _empty(concepts: list[str], missing: list[str], fetched_at: str,
           requests: list[dict]) -> pd.DataFrame:
    idx = pd.MultiIndex.from_arrays([pd.Index([], dtype=object),
                                     pd.DatetimeIndex([])], names=["code", "date"])
    out = pd.DataFrame({c: pd.Series(dtype=float) for c in concepts}, index=idx)
    out.attrs["meta"] = ()
    out.attrs["source"] = _source(concepts)
    out.attrs["fetched_at"] = fetched_at
    out.attrs["missing"] = tuple(missing)
    out.attrs["requests"] = tuple(requests)
    return out


def _source(concepts: Iterable[str]) -> str:
    flows: list[str] = []
    for c in concepts:
        for f, v in _flows_of(STES_CONCEPTS[c]):
            name = f"{f},{v}"
            if name not in flows:
                flows.append(name)
    return "OECD SDMX (sdmx.oecd.org): " + "; ".join(flows)


def _normalise_codes(codes) -> list[str] | None:
    if codes is None:
        return None
    if isinstance(codes, str):
        codes = [codes]
    out = []
    for c in codes:
        if not isinstance(c, str) or not c.strip():
            raise ValueError(f"country codes must be non-empty strings (ISO3), got {c!r}")
        c = c.strip().upper()
        if c not in out:
            out.append(c)
    if not out:
        raise ValueError("codes is empty; pass None for every economy the OECD publishes")
    return out


def _normalise_concepts(concepts) -> list[str]:
    if concepts is None:
        return list(STES_CONCEPTS)
    if isinstance(concepts, str):
        concepts = [concepts]
    concepts = list(dict.fromkeys(concepts))
    unknown = [c for c in concepts if c not in STES_CONCEPTS]
    if unknown:
        raise ValueError(f"unknown concept(s) {unknown}; valid concepts are "
                         f"{list(STES_CONCEPTS)}")
    if not concepts:
        raise ValueError(f"concepts is empty; valid concepts are {list(STES_CONCEPTS)}")
    # Registry order, so the column order does not depend on how they were asked for.
    return [c for c in STES_CONCEPTS if c in concepts]


def stes_meta(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-concept metadata of a :func:`stes_panel` result, as a DataFrame.

    The records live in ``panel.attrs["meta"]`` as a tuple of plain dicts (a
    DataFrame there makes ``pd.concat`` on slices of the panel raise); this
    is the intended way to read them back.
    """
    return pd.DataFrame(list(panel.attrs.get("meta", ())), columns=_META_COLS)


def stes_panel(codes: Iterable[str] | str | None = None, *,
               concepts: Iterable[str] | str | None = None,
               start: str | int = "1950",
               refresh: bool = False,
               pause: float = 5.0,
               retries: int = 3,
               retry_sleep: float = 90.0) -> pd.DataFrame:
    """Monthly cross-country panel of OECD short-term indicators, in levels.

    Parameters
    ----------
    codes : iterable of str, str or None
        ISO3 reference areas (``["USA", "MEX"]``). ``None`` asks for every
        area and drops the OECD aggregates (euro area, EU, OECD, G7, G20,
        ...). An aggregate asked for by name, such as ``"EA20"`` for euro-area
        money, is kept.
    concepts : iterable of str, str or None
        Keys of :data:`STES_CONCEPTS`; ``None`` means all of them.
    start : str or int
        First period, ``"1950"`` or ``"1990-01"``. The default reaches back
        to the start of nearly every series (US industrial production from
        1919 and Canadian CPI from 1914 need ``start="1914"``).
    refresh : bool
        Re-download instead of reading the on-disk cache.
    pause : float
        Minimum seconds between requests to ``sdmx.oecd.org``; raise it when
        other processes are pulling from the OECD at the same time.
    retries, retry_sleep : int, float
        Attempts per request and seconds slept after an HTTP 429, passed to
        :func:`~puremacro.fetch._oecd_sdmx.oecd_csv`. Worst case while the
        OECD throttles: each of the (up to ten) requests spends
        ``(retries - 1) * retry_sleep`` seconds asleep plus the attempts
        themselves, about 30-45 minutes with the defaults for a full call;
        ``retries=1`` fails fast and leaves the gaps in ``attrs["missing"]``.
        Responses already cached on disk cost nothing.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(code, date)`` with ``date`` the first day of the month,
        one float column per requested concept in registry order. ``attrs``:

        - ``meta``: tuple of dicts, one per concept with data (flow, key,
          unit, unit_mult, adjustment, sa, base_period, label, n,
          n_countries, first, last, countries_fallback, splice_factors,
          flows_failed, stale_countries); read it with :func:`stes_meta`.
          ``stale_countries`` lists the countries whose series ends more
          than six months before the column's last month.
        - ``source``: the dataflows used; ``fetched_at``: UTC ISO time.
        - ``missing``: what was asked for but not obtained, ``"cpi_core"``
          for a whole concept (also when its primary flow failed and only a
          fallback answered), ``"cpi_core:NZL"`` for a requested country,
          ``"cpi_core@DF_PRICES_ALL"`` when a fallback flow failed and the
          column was built from the primary flow alone.
        - ``requests``: one dict per SDMX request (flow, key, status, rows).

        On a provider failure the frame is empty, with the same index names,
        columns and ``attrs`` keys, and a warning is issued. An HTTP 404 on
        a request for every area (``codes=None``) is a failure too (an
        unknown flow or version); with explicit codes it means "nothing for
        these countries".

    Raises
    ------
    ValueError
        Unknown concept, malformed ``start`` or malformed country codes.

    Examples
    --------
    >>> from puremacro.fetch.oecd_stes_panel import stes_panel, stes_meta
    >>> df = stes_panel(["USA", "MEX"], concepts=["ip", "cpi", "rate_3m"])  # doctest: +SKIP
    >>> df.loc["MEX", "cpi"].dropna().index[0]                              # doctest: +SKIP
    Timestamp('1969-01-01 00:00:00')
    >>> stes_meta(df)[["concept", "unit", "n_countries", "first", "last"]]  # doctest: +SKIP
    """
    wanted = _normalise_concepts(concepts)
    code_list = _normalise_codes(codes)
    start = str(start).strip()
    if not _START.match(start):
        raise ValueError(f"start must look like 'YYYY' or 'YYYY-MM', got {start!r}")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # One request per flow, in the order the concepts first need them.
    flows: list[tuple[str, str]] = []
    for c in wanted:
        for fv in _flows_of(STES_CONCEPTS[c]):
            if fv not in flows:
                flows.append(fv)
    raw: dict[str, pd.DataFrame] = {}
    failed: set[str] = set()
    requests: list[dict] = []
    for flow, version in flows:
        key = _merged_key(flow, code_list)
        frame = _fetch(f"{flow},{version}", key, start_period=start,
                       refresh=refresh, pause=pause, retries=retries,
                       retry_sleep=retry_sleep)
        status = frame.attrs.get("status", "ok" if not frame.empty else "empty")
        ok = (not frame.empty and {"REF_AREA", "TIME_PERIOD", "OBS_VALUE"}
              <= set(frame.columns))
        if not ok and status == "ok":
            status = "unexpected columns" if not frame.empty else "empty"
        requests.append({"flow": f"{flow},{version}", "key": key, "status": status,
                         "rows": int(len(frame)) if ok else 0})
        if ok:
            frame = frame.loc[~frame["REF_AREA"].astype(str).isin(
                OECD_AGGREGATES - set(code_list or ()))]
            raw[flow] = frame
        elif status == "empty" or (status == "HTTP 404" and code_list):
            # The OECD's "nothing for these countries": not a failure.
            pass
        else:
            # Throttled, timed out, malformed, or a 404 on the whole flow
            # (unknown flow or version): the data exists but did not arrive.
            failed.add(flow)
            warnings.warn(f"stes_panel: {flow},{version} returned no data ({status}); "
                          "the concepts it serves are listed in attrs['missing']",
                          stacklevel=2)

    missing: list[str] = []
    columns: dict[str, pd.Series] = {}
    meta: list[dict] = []
    for c in wanted:
        spec = STES_CONCEPTS[c]
        spec_flows = _flows_of(spec)
        flows_failed = tuple(f"{f},{v}" for f, v in spec_flows if f in failed)
        if spec_flows[0][0] in failed:
            # Never return a fallback-only column as if it were complete.
            missing.append(c)
            continue
        missing.extend(f"{c}@{f.split('@', 1)[1]}" for f, _ in spec_flows[1:]
                       if f in failed)
        pieces = []
        for rank, (flow, version) in enumerate(spec_flows):
            if flow not in raw:
                continue
            tidy = _tidy(_pick(raw[flow], flow, spec), spec.unit_mult, c)
            if rank > 0 and spec.fallback_areas:
                tidy = tidy.loc[tidy["code"].isin(spec.fallback_areas)]
            if tidy.duplicated(["code", "date"]).any():
                n_dup = int(tidy.duplicated(["code", "date"]).sum())
                warnings.warn(f"stes_panel: {c} has {n_dup} duplicate (code, month) "
                              f"rows in {flow},{version} after pinning {spec.key}; "
                              "that flow is skipped for this concept", stacklevel=2)
                continue
            tidy["rank"] = rank
            pieces.append(tidy)
        splices: list[str] = []
        if len(pieces) > 1:
            pieces, splices = _ratio_splice(pieces)
        if pieces:
            both = pd.concat(pieces, ignore_index=True)
            # COICOP 2018 first, COICOP 1999 only where 2018 has nothing.
            both = (both.sort_values(["code", "date", "rank"])
                        .drop_duplicates(["code", "date"], keep="first"))
            both = both.loc[both["date"] >= pd.Timestamp(start if len(start) > 4
                                                          else f"{start}-01")]
        else:
            both = pd.DataFrame(columns=["code", "date", "value", "rank"])
        if both.empty:
            missing.append(c)
            continue
        series = both.set_index(["code", "date"])["value"].astype(float)
        columns[c] = series
        got = set(both["code"])
        if code_list:
            missing.extend(f"{c}:{code}" for code in code_list if code not in got)
        fb = sorted(set(both.loc[both["rank"] > 0, "code"]))
        last_by = both.groupby("code")["date"].max()
        cutoff = last_by.max() - pd.DateOffset(months=_STALE_MONTHS)
        stale = tuple(sorted(last_by.index[last_by < cutoff]))
        meta.append({
            "concept": c,
            "flow": " | ".join(f"{f},{v}" for f, v in _flows_of(spec)),
            "key": spec.key,
            "unit": spec.unit,
            "unit_mult": spec.unit_mult,
            "adjustment": spec.adjustment,
            "sa": spec.sa,
            "base_period": spec.base_period,
            "label": spec.label,
            "n": int(len(series)),
            "n_countries": int(len(got)),
            "first": both["date"].min().strftime("%Y-%m-%d"),
            "last": both["date"].max().strftime("%Y-%m-%d"),
            "countries_fallback": tuple(fb),
            "splice_factors": tuple(splices),
            "flows_failed": flows_failed,
            "stale_countries": stale,
        })

    if not columns:
        return _empty(wanted, missing, fetched_at, requests)

    out = pd.concat(columns, axis=1).reindex(columns=wanted).astype(float)
    out.index = out.index.set_names(["code", "date"])
    out = out.sort_index()
    out.attrs["meta"] = tuple(meta)
    out.attrs["source"] = _source(wanted)
    out.attrs["fetched_at"] = fetched_at
    out.attrs["missing"] = tuple(missing)
    out.attrs["requests"] = tuple(requests)
    return out


__all__ = ["STES_CONCEPTS", "StesConcept", "stes_meta", "stes_panel"]
