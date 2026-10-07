r"""Penn World Table 11.0 and the Maddison Project Database 2023, in one call each.

``fetch_pwt()`` returns any of the five Stata tables of PWT 11.0 and
``fetch_maddison()`` the MPD 2023 file, both indexed by ``(code, date)``
with ISO3 codes and ``date`` at January 1 of each year::

    fetch_pwt("main")      185 economies, 1950-2023, 47 series (rgdpe, rgdpna, pop, emp,
                           hc, cn, rnna, ck, labsh, ctfp, csh_*, pl_*, i_* flags ...)
    fetch_pwt("na")        212 economies, national accounts in national currency
                           (v_c v_i v_g v_x v_m v_gdp v_gfcf, q_* at 2021 prices, pop, xr ...)
    fetch_pwt("capital")   180 economies, investment, stocks and capital consumption by
                           asset (Ic_ Ip_ Nc_ Np_ Dc_ Kc_ Kp_ Ksh_ x Struc/Mach/TraEq/Other)
    fetch_pwt("labor")     the Gollin-style labour-share variants (comp_sh, lab_sh1..4),
                           schooling, employment sources
    fetch_pwt("trade")     export and import price levels and shares by BEC category
    fetch_maddison()       169 economies, real GDP per head (2011 int. $) and population,
                           annual from 1950 by default, back to 1 AD with ``start=None``

Why this module exists
----------------------
PWT is the longest cross-country panel of real output, factor inputs and
price levels, and its national-accounts file is the longest expenditure
panel in national currency that any free source publishes (76 economies
from before 1960). Maddison extends real GDP per head back two centuries
for the economies where it exists. The course notebooks used to download
the PWT workbook with their own HTTP code and read it with ``openpyxl``,
which an iPad does not have; the Stata files on Dataverse carry the same
numbers (the main file equals the workbook's "Data" sheet to 5e-15) and
``pandas.read_stata`` is pure pandas. Both databases are licensed CC BY 4.0:
the data may be redistributed, frozen and shared, with the citations kept
in ``attrs["citation"]``.

Mechanics
---------
Every file is one GET of ``https://dataverse.nl/api/access/datafile/<id>``,
which answers 303 to a signed object-store URL that urllib follows. The
bytes travel through :func:`puremacro._http.safe_get_bytes_cached` (SQLite
cache outside the repository, three-second spacing per host) and are kept
for ten years: a datafile id is an immutable release, so ``refresh=True`` is
only needed to repair a damaged cache entry. dataverse.nl sits behind the
Anubis bot wall, which answers any User-Agent beginning ``Mozilla/`` (the
default of :mod:`puremacro._http`) with an HTTP 200 HTML challenge page;
this module therefore sends its own agent string and checks that the body
begins ``<stata_dta`` before parsing it. A body that does not (a cached
challenge page, say), or one that begins so but does not parse, is fetched
once more bypassing the cache.

Things the files do not tell you
--------------------------------
* **The price base is 2021 everywhere**, although several labels say
  otherwise: ``q_*`` in the national-accounts file say "2017 prices", the
  capital-detail ``Ip_/Np_/Kp_`` indices say "2017=1" and the trade-detail
  price levels say "USA GDPo in 2011=1". ``q_gdp == v_gdp`` in 2021 for all
  212 economies and every ``Ip_*`` equals 1 in 2021. ``attrs["units"]``
  states the true base; ``attrs["labels"]`` keeps the published labels.
* **C + I + G + X - M is not GDP** in more than a third of the
  national-accounts rows (Britain by a median 9 %, Japan 2 %): PWT takes the
  components and the total from different sources. Use ``v_gdp`` as the
  denominator of any share and never back out a component as a residual.
* ``v_gfcf`` is effectively a 1970+ series (three economies earlier);
  ``v_i`` is gross capital formation, inventories included. The four
  capital-detail ``Ic_*`` add up to ``v_gfcf`` to float32 precision for 179
  of 180 economies; Taiwan's sum exceeds it by 6-8 % in every year.
* The labour-detail file publishes ``emp`` in persons and ``year`` as a
  float, and carries three empty World Bank columns (``countryname``,
  ``indicatorname``, ``seriescode``). Here ``emp`` is rescaled to millions,
  as in the other tables, ``year`` becomes the integer it is, and the stray
  columns are dropped.
* Flags (``i_*`` and the labour file's ``source``) are Stata categoricals;
  they are returned as integer codes (nullable ``Int64``) and their value
  labels sit in ``attrs["value_labels"]``.
* The national-accounts file also lists ``CH2``, PWT's alternative series
  for China 1952-2021; it is not an economy and is dropped unless asked for
  by code.
* Former economies (``ANT``, ``CSK``, ``SUN``, ``YUG``) have no GDP in PWT
  but the national-accounts file keeps population and exchange rates for
  them (``CSK``, ``SUN``, ``YUG`` 1970-1990; ``ANT`` 1970-2012, also in the
  labour file with ``emp``). They are kept and listed in
  ``attrs["historical_entities"]``, which is why ``fetch_pwt("na")`` has 216
  codes but only 212 with ``v_gdp``. In Maddison ``CSK``, ``SUN`` and ``YUG``
  carry GDP and population and are listed the same way.
* Kosovo is ``RKS`` in PWT (national-accounts and labour files), not the
  ``XKX`` of the World Bank and the IMF. The code is kept as published and
  flagged in ``attrs["nonstandard_codes"]``; rename it before merging.
* Maddison ``gdppc`` is in 2011 international dollars, not 2021, and ``pop``
  is in thousands. MPD asks that the original sources (the "Sources" sheet
  of its workbook) be cited as well whenever its data are graphed or fewer
  than twelve countries are used (``attrs["citation_policy"]``).

Rows that are entirely empty (PWT publishes a full 1950-2023 grid, NaN
before a country's first year) are dropped, so row counts are coverage. A
provider failure never raises: the frame comes back empty, with the
table's columns, the same ``attrs`` keys and ``attrs["missing"]`` saying
what failed and why; a warning says so too. ``ValueError`` is kept for
caller errors (unknown table or release).

Source: Feenstra, Inklaar and Timmer (2015), Penn World Table 11.0,
Groningen Growth and Development Centre, doi:10.34894/FABVLR; Bolt and van
Zanden (2024), Maddison Project Database 2023, doi:10.34894/INZBF2. Both
served by DataverseNL under CC BY 4.0.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import http.client
import io
import urllib.error
import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from puremacro import __version__
from puremacro._http import safe_get_bytes_cached


_DATAFILE_URL = "https://dataverse.nl/api/access/datafile/{id}"

# Anubis on dataverse.nl challenges any agent that starts with "Mozilla/".
_UA = f"puremacro/{__version__} (urllib)"

# A datafile id is an immutable release: keep the bytes for ten years.
_TTL_SECONDS = 10 * 365 * 24 * 3600
_RATE_LIMIT_SECONDS = 3.0

_STATA_MAGIC = b"<stata_dta"

_LICENSE = "CC BY 4.0"

_PWT_CITATION = (
    "Feenstra, Robert C., Robert Inklaar and Marcel P. Timmer (2015), "
    "\"The Next Generation of the Penn World Table\", American Economic Review "
    "105(10), 3150-3182, doi:10.1257/aer.20130954, available for download at "
    "www.ggdc.net/pwt"
)

_MADDISON_CITATION = (
    "Bolt, Jutta and Jan Luiten van Zanden (2024), \"Maddison style estimates of "
    "the evolution of the world economy: A new 2023 update\", Journal of Economic "
    "Surveys, 1-41, doi:10.1111/joes.12618"
)

_MADDISON_CITATION_POLICY = (
    "Cite Bolt and van Zanden (2024). In addition, cite the original sources "
    "(the 'Sources' sheet of mpd2023_web.xlsx) whenever the data are shown in "
    "any graphical form or a subset of fewer than 12 countries is used."
)

#: PWT releases this module knows: Dataverse DOI, dataset id, release date of
#: the Dataverse version, the reference year of every constant-price and
#: price-level series, and the datafile id and file name of each table.
PWT_RELEASES: dict[str, dict] = {
    "11.0": {
        "doi": "10.34894/FABVLR",
        "dataset_id": 554010,
        "release": "2025-10-10",
        "price_base": 2021,
        "files": {
            "main":    (554030, "pwt110.dta"),
            "na":      (554024, "pwt110_na_data.dta"),
            "capital": (554026, "pwt110_capital_detail.dta"),
            "labor":   (554028, "pwt110_labor_detail.dta"),
            "trade":   (554023, "pwt110_trade_detail.dta"),
        },
    },
}

#: The Maddison Project Database release this module reads.
MADDISON_RELEASE: dict = {
    "name": "Maddison Project Database 2023",
    "doi": "10.34894/INZBF2",
    "dataset_id": 421301,
    "release": "2024-04-26",
    "price_base": 2011,
    "datafile": (421303, "maddison2023_web.dta"),
}

#: Codes in the PWT files that are not economies: ``CH2`` is PWT's
#: alternative series for China (1952-2021). Dropped unless asked for.
PWT_NON_ECONOMIES: frozenset[str] = frozenset({"CH2"})

#: Former economies with rows in the PWT na/labor files but no GDP (kept,
#: listed in ``attrs["historical_entities"]``).
PWT_HISTORICAL: frozenset[str] = frozenset({"ANT", "CSK", "SUN", "YUG"})

#: Codes PWT publishes that are not ISO 3166 alpha-3, with their usual equivalent.
PWT_NONSTANDARD_CODES: dict[str, str] = {"RKS": "Kosovo; World Bank and IMF code: XKX"}

#: Entities in MPD 2023 that no longer exist (kept, flagged in attrs).
MADDISON_HISTORICAL: frozenset[str] = frozenset({"CSK", "SUN", "YUG"})

_M_USD = "millions of 2021 US$ (PPP)"
_M_LCU = "millions of current national currency"
_M_LCU_REAL = "millions of national currency at constant 2021 prices"
_PL = "price level (PPP/XR), USA GDPo in 2021 = 1"
_SHARE = "share (fraction)"
_FLAG = "integer code, see attrs['value_labels']"
_TEXT = "text"

_ASSETS = ("Struc", "Mach", "TraEq", "Other")


def _assets(prefix: str, unit: str) -> dict[str, str]:
    return {f"{prefix}_{a}": unit for a in _ASSETS}


#: Units of every published column, table by table, with the true price
#: base (the labels of several files still say 2017 or 2011). Text columns
#: are kept as columns; ``year`` and ``countrycode`` become the index.
PWT_UNITS: dict[str, dict[str, str]] = {
    "main": {
        "country": _TEXT, "currency_unit": _TEXT,
        "rgdpe": _M_USD, "rgdpo": _M_USD,
        "pop": "millions of persons", "emp": "millions of persons",
        "avh": "hours per person engaged per year", "hc": "index",
        "ccon": _M_USD, "cda": _M_USD, "cgdpe": _M_USD, "cgdpo": _M_USD,
        "cn": _M_USD, "ck": "index, USA = 1", "ctfp": "index, USA = 1",
        "cwtfp": "index, USA = 1",
        "rgdpna": _M_USD, "rconna": _M_USD, "rdana": _M_USD, "rnna": _M_USD,
        "rkna": "index, 2021 = 1", "rtfpna": "index, 2021 = 1",
        "rwtfpna": "index, 2021 = 1",
        "labsh": _SHARE, "irr": "rate (fraction per year)",
        "delta": "rate (fraction per year)",
        "xr": "national currency per US$ (market + estimated)",
        "pl_con": _PL, "pl_da": _PL, "pl_gdpo": _PL,
        "i_cig": _FLAG, "i_xm": _FLAG, "i_xr": _FLAG, "i_outlier": _FLAG,
        "i_irr": _FLAG, "cor_exp": "correlation",
        "csh_c": _SHARE, "csh_i": _SHARE, "csh_g": _SHARE, "csh_x": _SHARE,
        "csh_m": _SHARE, "csh_r": _SHARE,
        "pl_c": _PL, "pl_i": _PL, "pl_g": _PL, "pl_x": _PL, "pl_m": _PL,
        "pl_n": "price level, USA 2021 = 1", "pl_k": "price level, USA = 1",
    },
    "na": {
        "v_c": _M_LCU, "v_i": _M_LCU, "v_g": _M_LCU, "v_x": _M_LCU,
        "v_m": _M_LCU, "v_gdp": _M_LCU,
        "q_c": _M_LCU_REAL, "q_i": _M_LCU_REAL, "q_g": _M_LCU_REAL,
        "q_x": _M_LCU_REAL, "q_m": _M_LCU_REAL, "q_gdp": _M_LCU_REAL,
        "pop": "millions of persons",
        "xr": "national currency per US$ (market)",
        "xr2": "national currency per US$ (market + estimated)",
        "v_gfcf": _M_LCU, "q_gfcf": _M_LCU_REAL,
        "emp": "millions of persons",
        "avh": "hours per person engaged per year",
    },
    "capital": {
        **_assets("Ic", _M_LCU),
        **_assets("Ip", "price index, 2021 = 1"),
        **_assets("Nc", _M_LCU),
        **_assets("Np", "price index, 2021 = 1"),
        **_assets("Dc", _M_LCU),
        **_assets("Kc", _M_LCU),
        **_assets("Kp", "price index, 2021 = 1"),
        **_assets("Ksh", _SHARE),
    },
    "labor": {
        "emp": "millions of persons",
        "avh": "hours per person engaged per year",
        "i_emp": _FLAG,
        "yr_sch": "years",
        "source": _FLAG,
        "labsh": _SHARE, "i_labsh": _FLAG, "i_labsh2": _FLAG,
        "comp_sh": _SHARE, "i_mix": "0/1 flag",
        "lab_sh1": _SHARE, "lab_sh2": _SHARE, "lab_sh3": _SHARE, "lab_sh4": _SHARE,
        "i_comp_sh": _FLAG, "i_lab_sh1": _FLAG, "i_lab_sh2": _FLAG,
        "i_lab_sh3": _FLAG, "i_lab_sh4": _FLAG,
    },
    "trade": {
        **{f"pl_{d}{k}": _PL for d in ("x", "m") for k in range(1, 7)},
        "pl_x": _PL, "pl_m": _PL,
        **{f"csh_{d}{k}": _SHARE for k in range(1, 7) for d in ("x", "m")},
        "csh_x": _SHARE, "csh_m": _SHARE,
    },
}

#: Units of the two MPD 2023 series (the .dta carries no labels; the
#: workbook header reads "GDP pc 2011 prices").
MADDISON_UNITS: dict[str, str] = {
    "country": _TEXT,
    "region": _TEXT,
    "gdppc": "2011 international dollars (PPP) per person",
    "pop": "thousands of persons",
}

#: Powers of ten in which each unit is expressed (``attrs["meta"]["unit_mult"]``).
_UNIT_MULT = {_M_USD: 6, _M_LCU: 6, _M_LCU_REAL: 6, "millions of persons": 6,
              "thousands of persons": 3}

#: Columns the files publish that are dropped: empty World Bank leftovers in
#: the labour-detail file.
_DROP_COLUMNS = frozenset({"countryname", "indicatorname", "seriescode"})

_LABEL_NOTES = {
    "na": "published label says 2017 prices; the reference year is 2021",
    "capital": "published label says 2017=1; the index equals 1 in 2021",
    "trade": "published label says USA GDPo in 2011=1; the base is USA GDPo in 2021",
}

_INDEX_NAMES = ["code", "date"]


def _stale(label: str) -> bool:
    """Whether a published label names a price base the data do not use."""
    # Stata truncates labels at 80 characters: one capital-detail label ends "(201".
    return "2017" in label or "2011=1" in label or label.rstrip().endswith("(201")


class _ProviderError(Exception):
    """A download or parse failure, carried as a reason string."""


def _get(url: str, *, timeout: float, refresh: bool = False) -> bytes:
    """GET ``url`` through the house cache with an agent Anubis lets through.

    The module-level seam tests patch. Raises ``urllib.error.HTTPError`` or
    ``OSError`` (``http.client.HTTPException`` on a truncated body) on
    failure; the callers turn those into ``attrs["missing"]``.
    """
    return safe_get_bytes_cached(url, timeout, user_agent=_UA, ttl_seconds=_TTL_SECONDS,
                                 rate_limit_seconds=_RATE_LIMIT_SECONDS, refresh=refresh)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _reason(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower():
        return "timeout"
    return f"{type(exc).__name__}: {exc}"


def _download(datafile_id: int, *, timeout: float, refresh: bool) -> tuple[bytes, str]:
    """Return ``(stata bytes, url)`` or raise :class:`_ProviderError`."""
    url = _DATAFILE_URL.format(id=datafile_id)
    try:
        raw = _get(url, timeout=timeout, refresh=refresh)
        if raw[:10] != _STATA_MAGIC and not refresh:
            # A bot-check page or a truncated body may sit in the cache.
            raw = _get(url, timeout=timeout, refresh=True)
    except (urllib.error.HTTPError, OSError, http.client.HTTPException) as exc:
        raise _ProviderError(_reason(exc)) from exc
    if raw[:10] != _STATA_MAGIC:
        head = raw[:200].decode("utf-8", errors="ignore").lower()
        what = "bot-check page" if ("not a bot" in head or "<html" in head) else "not a Stata file"
        raise _ProviderError(f"{what} ({len(raw)} bytes) instead of a Stata file")
    return raw, url


def _read_dta(raw: bytes) -> tuple[pd.DataFrame, dict[str, str], dict[str, dict]]:
    """Parse Stata bytes: (frame with integer flags, variable labels, value labels by column)."""
    try:
        with pd.read_stata(io.BytesIO(raw), convert_categoricals=False, iterator=True) as rd:
            labels = dict(rd.variable_labels())
            sets = rd.value_labels()
            df = rd.read()
            # Which label set each column uses (private but stable since pandas 0.x;
            # without it the value labels are still returned keyed by set name).
            lbllist = list(getattr(rd, "_lbllist", []) or [])
    except Exception as exc:  # any parse failure is the provider's, not the caller's
        raise _ProviderError(f"unreadable Stata file: {type(exc).__name__}: {exc}") from exc
    by_col: dict[str, dict] = {}
    if len(lbllist) == len(df.columns):
        for col, name in zip(df.columns, lbllist):
            if name and name in sets:
                by_col[col] = {int(k): str(v) for k, v in sets[name].items()}
    else:
        by_col = {k: {int(a): str(b) for a, b in v.items()} for k, v in sets.items()}
    return df, labels, by_col


def _obtain(datafile_id: int, *, timeout: float, refresh: bool):
    """Download and parse one file: ``(raw, url, frame, labels, value labels)``.

    A cached body that starts like a Stata file but does not parse is fetched
    once more bypassing the cache, so a damaged entry cannot stick for the
    ten-year TTL. Raises :class:`_ProviderError`.
    """
    raw, url = _download(datafile_id, timeout=timeout, refresh=refresh)
    try:
        df, labels, value_labels = _read_dta(raw)
    except _ProviderError:
        if refresh:
            raise
        raw, url = _download(datafile_id, timeout=timeout, refresh=True)
        df, labels, value_labels = _read_dta(raw)
    return raw, url, df, labels, value_labels


def _year_dates(years: pd.Series | np.ndarray) -> pd.DatetimeIndex:
    """January 1 of each year; years before 1678 need second resolution on pandas 2."""
    y = np.asarray(years, dtype=np.int64)
    strings = np.char.add(np.char.zfill(y.astype(str), 4), "-01-01")
    try:
        return pd.DatetimeIndex(pd.to_datetime(strings, format="%Y-%m-%d"))
    except (pd.errors.OutOfBoundsDatetime, ValueError):
        return pd.DatetimeIndex(strings.astype("datetime64[s]"))


def _check_start(start) -> int | None:
    if start is None:
        return None
    try:
        return int(str(start)[:4]) if not isinstance(start, (int, np.integer)) else int(start)
    except ValueError:
        raise ValueError(f"start must be a year (int or 'YYYY'), got {start!r}") from None


def _norm_codes(codes: Iterable[str] | str | None) -> list[str] | None:
    if codes is None:
        return None
    if isinstance(codes, str):
        codes = [codes]
    out = [str(c).strip().upper() for c in codes]
    if not out:
        raise ValueError("codes is empty; pass None for every economy")
    return list(dict.fromkeys(out))


def _empty(units: dict[str, str]) -> pd.DataFrame:
    """An empty ``(code, date)`` frame with the dtypes a successful read has."""
    idx = pd.MultiIndex.from_arrays(
        [pd.Index([], dtype=object), pd.DatetimeIndex([])], names=_INDEX_NAMES)
    flags = set(_flag_columns(units))
    out = pd.DataFrame(index=idx)
    for c, u in units.items():
        dtype = object if u == _TEXT else ("Int64" if c in flags else np.float64)
        out[c] = pd.Series(dtype=dtype, index=idx)
    return out


def _flag_columns(table_units: dict[str, str]) -> list[str]:
    return [c for c, u in table_units.items() if u == _FLAG or u == "0/1 flag"]


def _shape(df: pd.DataFrame, units: dict[str, str], *, codes: list[str] | None,
           start: int | None, drop_codes: frozenset[str]) -> tuple[pd.DataFrame, list[str]]:
    """Index by (code, date), type the columns, filter, drop empty rows.

    Returns the frame and the requested codes that have no data.
    """
    df = df.drop(columns=[c for c in df.columns if c in _DROP_COLUMNS])
    df = df.rename(columns={"countrycode": "code"})
    df["code"] = df["code"].astype(str).str.strip().str.upper()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].round().astype(np.int64)

    text = [c for c in df.columns if units.get(c) == _TEXT]
    flags = [c for c in _flag_columns(units) if c in df.columns]
    values = [c for c in df.columns if c not in ("code", "year", *text, *flags)]
    for c in values:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float64)
    for c in flags:
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")
    for c in text:
        df[c] = df[c].astype(object).where(df[c].notna(), None)
        df[c] = df[c].map(lambda s: s.strip() if isinstance(s, str) else s)

    if codes is None:
        df = df[~df["code"].isin(drop_codes)]
    else:
        df = df[df["code"].isin(codes)]
    if start is not None:
        df = df[df["year"] >= start]
    df = df.dropna(subset=values, how="all") if values else df
    missing = [] if codes is None else [c for c in codes if c not in set(df["code"])]

    df = df.sort_values(["code", "year"])
    df.index = pd.MultiIndex.from_arrays([df["code"].to_numpy(dtype=object),
                                          _year_dates(df["year"])], names=_INDEX_NAMES)
    order = [c for c in units if c in df.columns] + \
            [c for c in df.columns if c not in units and c not in ("code", "year")]
    return df[order], missing


def _meta(df: pd.DataFrame, units: dict[str, str], labels: dict[str, str], *,
          file: str, url: str, source: str, note: str | None) -> tuple[dict, ...]:
    rows = []
    for c in df.columns:
        u = units.get(c, "")
        if u == _TEXT:
            continue
        s = df[c].dropna()
        years = s.index.get_level_values("date").year if len(s) else None
        label = labels.get(c, "")
        row = {
            "variable": c, "source": source, "file": file, "url": url,
            "label": label, "units": u, "unit_mult": _UNIT_MULT.get(u, 0),
            "sa": "annual",
            "first": int(years.min()) if years is not None else None,
            "last": int(years.max()) if years is not None else None,
            "n": int(s.size),
            "n_codes": int(s.index.get_level_values("code").nunique()) if len(s) else 0,
        }
        if note and _stale(label):
            row["note"] = note
        rows.append(row)
    return tuple(rows)


def _check_pwt(table: str, version: str) -> dict:
    if version not in PWT_RELEASES:
        raise ValueError(f"unknown PWT version {version!r}; available: {sorted(PWT_RELEASES)}")
    files = PWT_RELEASES[version]["files"]
    if table not in files:
        raise ValueError(f"unknown PWT table {table!r}; available: {list(files)}")
    return PWT_RELEASES[version]


def fetch_pwt(
    table: str = "main",
    *,
    version: str = "11.0",
    codes: Iterable[str] | str | None = None,
    start: int | str | None = None,
    refresh: bool = False,
    timeout: float = 180.0,
) -> pd.DataFrame:
    """One table of the Penn World Table, indexed by ``(code, date)``.

    Parameters
    ----------
    table : {"main", "na", "capital", "labor", "trade"}
        ``"main"`` is ``pwt110.dta`` (real GDP at PPPs, inputs, TFP, price
        levels); ``"na"`` the national accounts in national currency;
        ``"capital"`` investment and stocks by asset; ``"labor"`` the
        labour-share variants and employment sources; ``"trade"`` price
        levels and shares by BEC category.
    version : str
        PWT release; only ``"11.0"`` is registered (:data:`PWT_RELEASES`).
    codes : iterable of str or str, optional
        ISO3 codes; ``None`` keeps every economy (``CH2``, PWT's alternative
        China series, is dropped unless named). Codes with no data are
        listed in ``attrs["missing"]``.
    start : int or str, optional
        First year kept; ``None`` keeps everything from 1950.
    refresh : bool
        Re-download instead of reading the on-disk cache.
    timeout : float
        Seconds for the download.

    Returns
    -------
    pandas.DataFrame
        Index ``(code, date)`` with ``date`` = January 1 of each year; the
        published column names, values as ``float64``, flags as ``Int64``
        codes, text columns (``country``, ``currency_unit``) as strings.
        Rows with no value at all are dropped. ``attrs``: ``source``,
        ``doi``, ``url``, ``datafile_id``, ``file``, ``sha256``, ``release``,
        ``license``, ``citation``, ``price_base`` (2021), ``units``,
        ``labels`` (as published), ``value_labels``, ``historical_entities``
        (former economies present, see :data:`PWT_HISTORICAL`),
        ``nonstandard_codes`` (``RKS`` for Kosovo when present), ``nbytes``,
        ``meta`` (one dict per column: units, unit_mult, first, last, n,
        n_codes), ``missing`` and ``fetched_at``. On a provider failure the frame is empty, with the
        table's columns, and ``attrs["missing"]`` names the reason.

    Raises
    ------
    ValueError
        Unknown ``table`` or ``version``, or a ``start`` that is not a year.

    Examples
    --------
    >>> from puremacro.fetch.pwt import fetch_pwt
    >>> na = fetch_pwt("na", codes=["MEX", "USA"])                 # doctest: +SKIP
    >>> na.loc["MEX", "v_gdp"].iloc[0]                             # doctest: +SKIP
    55.8...
    >>> fetch_pwt("main").attrs["citation"]                        # doctest: +SKIP
    'Feenstra, Robert C., Robert Inklaar and Marcel P. Timmer (2015), ...'
    """
    rel = _check_pwt(table, version)
    start_y = _check_start(start)
    wanted = _norm_codes(codes)
    datafile_id, file = rel["files"][table]
    units = PWT_UNITS[table]
    url = _DATAFILE_URL.format(id=datafile_id)
    source = f"Penn World Table {version}, {file} (DataverseNL doi:{rel['doi']})"
    attrs = {
        "source": source, "table": table, "version": version, "doi": rel["doi"],
        "url": url, "datafile_id": datafile_id, "file": file, "sha256": None,
        "release": rel["release"], "license": _LICENSE, "citation": _PWT_CITATION,
        "price_base": rel["price_base"], "units": dict(units), "labels": {},
        "value_labels": {}, "historical_entities": (), "nonstandard_codes": {},
        "nbytes": None, "meta": (), "missing": (), "fetched_at": _now(),
    }
    try:
        raw, url, df, labels, value_labels = _obtain(datafile_id, timeout=timeout,
                                                     refresh=refresh)
    except _ProviderError as exc:
        warnings.warn(f"fetch_pwt: {file} not obtained ({exc}); empty frame", stacklevel=2)
        out = _empty(units)
        attrs["missing"] = ({"table": table, "file": file, "datafile_id": datafile_id,
                             "reason": str(exc)},)
        out.attrs.update(attrs)
        return out

    if table == "labor" and "emp" in df.columns:
        # The labour-detail file publishes persons; every other table, millions.
        df["emp"] = pd.to_numeric(df["emp"], errors="coerce").astype(np.float64) / 1e6
        labels["emp"] = labels.get("emp", "") + " [rescaled here from persons to millions]"
    drop = PWT_NON_ECONOMIES
    out, absent = _shape(df, units, codes=wanted, start=start_y, drop_codes=drop)
    missing = tuple({"code": c, "reason": f"no data in PWT {version} {table}"} for c in absent)
    if missing:
        warnings.warn("fetch_pwt: no data for " + ", ".join(m["code"] for m in missing),
                      stacklevel=2)
    present = set(out.index.get_level_values("code"))
    attrs.update(
        url=url, sha256=hashlib.sha256(raw).hexdigest(), nbytes=len(raw),
        historical_entities=tuple(sorted(PWT_HISTORICAL & present)),
        nonstandard_codes={k: v for k, v in PWT_NONSTANDARD_CODES.items() if k in present},
        labels={k: v for k, v in labels.items() if k not in _DROP_COLUMNS},
        value_labels={k: v for k, v in value_labels.items() if k in out.columns},
        meta=_meta(out, units, labels, file=file, url=url, source=source,
                   note=_LABEL_NOTES.get(table)),
        missing=missing,
    )
    out.attrs.update(attrs)
    return out


def pwt_variables(
    table: str = "main",
    *,
    version: str = "11.0",
    refresh: bool = False,
    timeout: float = 180.0,
) -> pd.DataFrame:
    """The variable table of one PWT file: published label, true units, note.

    Parameters
    ----------
    table, version, refresh, timeout
        As in :func:`fetch_pwt`.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``variable`` (in registry order, :data:`PWT_UNITS`) with columns ``label`` (the
        Stata label as published, ``None`` if the file could not be read),
        ``units`` (with the true 2021 price base) and ``note`` (where the
        published label is stale). Reads the same cached bytes as
        :func:`fetch_pwt`; on a provider failure the labels are ``None``
        and a warning is issued.

    Raises
    ------
    ValueError
        Unknown ``table`` or ``version``.
    """
    rel = _check_pwt(table, version)
    datafile_id, file = rel["files"][table]
    units = PWT_UNITS[table]
    labels: dict[str, str] = {}
    try:
        _, _, _, labels, _ = _obtain(datafile_id, timeout=timeout, refresh=refresh)
    except _ProviderError as exc:
        warnings.warn(f"pwt_variables: {file} not obtained ({exc}); labels are None",
                      stacklevel=2)
    note = _LABEL_NOTES.get(table)
    rows = []
    for var, unit in units.items():
        label = labels.get(var) if labels else None
        if var == "emp" and table == "labor" and label is not None:
            label += " [rescaled here from persons to millions]"
        rows.append({
            "variable": var, "label": label, "units": unit,
            "note": note if (note and label and _stale(label)) else None,
        })
    out = pd.DataFrame(rows).set_index("variable")
    out.attrs.update(source=f"Penn World Table {version}, {file}", doi=rel["doi"],
                     url=_DATAFILE_URL.format(id=datafile_id), price_base=rel["price_base"])
    return out


def fetch_maddison(
    codes: Iterable[str] | str | None = None,
    *,
    start: int | str | None = 1950,
    refresh: bool = False,
    timeout: float = 180.0,
) -> pd.DataFrame:
    """The Maddison Project Database 2023, indexed by ``(code, date)``.

    Parameters
    ----------
    codes : iterable of str or str, optional
        ISO3 codes (MPD's own, which include ``CSK``, ``SUN``, ``YUG``);
        ``None`` keeps all 169. Codes with no data go to ``attrs["missing"]``.
    start : int or str or None
        First year kept, 1950 by default (annual for every economy from
        there); ``None`` keeps the whole file back to 1 AD, whose early
        grid is sparse and not annual (1, 730, 1000, ...).
    refresh : bool
        Re-download instead of reading the on-disk cache.
    timeout : float
        Seconds for the download.

    Returns
    -------
    pandas.DataFrame
        Index ``(code, date)``, date = January 1; columns ``country``,
        ``region`` (text), ``gdppc`` (real GDP per head, 2011 international
        dollars) and ``pop`` (thousands). Rows with neither series are
        dropped. ``attrs`` as in :func:`fetch_pwt` plus ``citation_policy``
        and ``historical_entities``; ``price_base`` is 2011.

    Raises
    ------
    ValueError
        A ``start`` that is not a year.

    Examples
    --------
    >>> from puremacro.fetch.pwt import fetch_maddison
    >>> m = fetch_maddison(["USA", "MEX"], start=None)              # doctest: +SKIP
    >>> m.loc["USA", "gdppc"].first_valid_index()                    # doctest: +SKIP
    Timestamp('1800-01-01 00:00:00')
    """
    start_y = _check_start(start)
    wanted = _norm_codes(codes)
    rel = MADDISON_RELEASE
    datafile_id, file = rel["datafile"]
    url = _DATAFILE_URL.format(id=datafile_id)
    source = f"{rel['name']}, {file} (DataverseNL doi:{rel['doi']})"
    attrs = {
        "source": source, "doi": rel["doi"], "url": url, "datafile_id": datafile_id,
        "file": file, "sha256": None, "release": rel["release"], "license": _LICENSE,
        "citation": _MADDISON_CITATION, "citation_policy": _MADDISON_CITATION_POLICY,
        "price_base": rel["price_base"], "units": dict(MADDISON_UNITS),
        "labels": {}, "value_labels": {}, "historical_entities": (),
        "nonstandard_codes": {}, "nbytes": None,
        "meta": (), "missing": (), "fetched_at": _now(),
    }
    try:
        raw, url, df, labels, _ = _obtain(datafile_id, timeout=timeout, refresh=refresh)
    except _ProviderError as exc:
        warnings.warn(f"fetch_maddison: {file} not obtained ({exc}); empty frame",
                      stacklevel=2)
        out = _empty(MADDISON_UNITS)
        attrs["missing"] = ({"file": file, "datafile_id": datafile_id, "reason": str(exc)},)
        out.attrs.update(attrs)
        return out

    out, absent = _shape(df, MADDISON_UNITS, codes=wanted, start=start_y,
                         drop_codes=frozenset())
    missing = tuple({"code": c, "reason": "no data in MPD 2023"} for c in absent)
    if missing:
        warnings.warn("fetch_maddison: no data for " + ", ".join(m["code"] for m in missing),
                      stacklevel=2)
    present = set(out.index.get_level_values("code"))
    labels = {k: v for k, v in labels.items() if v}
    labels.setdefault("gdppc", "Real GDP per capita in 2011$ (workbook header: GDP pc 2011 prices)")
    labels.setdefault("pop", "Population, mid-year (thousands)")
    attrs.update(
        url=url, sha256=hashlib.sha256(raw).hexdigest(), nbytes=len(raw), labels=labels,
        historical_entities=tuple(sorted(MADDISON_HISTORICAL & present)),
        meta=_meta(out, MADDISON_UNITS, labels, file=file, url=url, source=source, note=None),
        missing=missing,
    )
    out.attrs.update(attrs)
    return out


__all__ = [
    "fetch_pwt", "fetch_maddison", "pwt_variables",
    "PWT_RELEASES", "PWT_UNITS", "PWT_NON_ECONOMIES", "PWT_HISTORICAL",
    "PWT_NONSTANDARD_CODES",
    "MADDISON_RELEASE", "MADDISON_UNITS", "MADDISON_HISTORICAL",
]
