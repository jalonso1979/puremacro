"""Deutsche Bundesbank real-time database ("Gerda"), dataflow ``BBKRT``.

Served over the Bundesbank's SDMX REST service, no key required::

    https://api.statistiken.bundesbank.de/rest/data/BBKRT/<key>?format=csv&lang=en

HOW THE VINTAGE IS ENCODED
--------------------------
As **three dimensions of the series key**, positions 9-11:
``BBK_RTD_REL_YEAR``, ``BBK_RTD_REL_MONTH``, ``BBK_RTD_REL_DAY``. Every
vintage is its own SDMX series, so an 8-part key is "all vintages" and
an 11-part key is one specific edition::

    Q.DE.Y.A.AG1.CA010.A.I              -> every vintage
    Q.DE.Y.A.AG1.CA010.A.I.2026.07.30   -> the 30 July 2026 edition

Unlike ALFRED's ingestion stamp, this **is the real publication date**:
the observed dates track the Destatis release calendar (flash estimate
at end of month, detailed accounts around the 22nd-25th), and the Gerda
documentation states that "the exact publication date is thus part of
the key structure". So Bundesbank vintages may legitimately be used as
event dates.

TWO SENTINEL DAYS
-----------------
``CL_BBK_RTD_REL_DAY`` defines ``40`` = "exact date unknown" and ``41``
= "exact date unknown, first release of the month". They appear in real
CSV headers as ``40.02.1997``. Parsed naively with ``%d.%m.%Y`` they
raise; parsed leniently they can become NaT and drop a whole edition.
Here they are mapped to the first of the named month and flagged.

REAL GDP IS SPLIT ACROSS TWO KEYS AT MAY 2005
---------------------------------------------
``...CA010.C.A`` (constant prices, absolute) covers the 1995-2005
vintages; ``...CA010.A.I`` (chain-linked index, 2020=100) covers 2005
onward. They are different units, so they are exposed as separate keys
rather than silently spliced — see :data:`BBK_RTD_KEYS`.
"""
from __future__ import annotations

import csv
import io
import re

import pandas as pd

from ..._http import safe_get_bytes, safe_get_bytes_cached
from ._base import VintagePanel, normalize_vintage_frame, register_provider
from .catalog import SeriesSpec, register_catalog


BBK_RTD_URL = (
    "https://api.statistiken.bundesbank.de/rest/data/BBKRT/{key}"
    "?format=csv&lang=en"
)

_UA = "puremacro (real-time vintage reader)"

#: Canonical variable -> BBKRT series key (vintage dimensions omitted,
#: which asks for every edition).
BBK_RTD_KEYS: dict[str, str] = {
    "gdp_real": "Q.DE.Y.A.AG1.CA010.A.I",
    "gdp_nom": "Q.DE.N.A.AG1.CA010.V.A",
    "gfcf_real": "Q.DE.Y.A.CD1.CA010.A.I",
    "ip": "M.DE.Y.I.IP1.ACM01.C.I",
}

#: Pre-2005 real GDP, in constant-price absolute terms rather than a
#: chain-linked index. Different units: use deliberately, do not splice.
BBK_RTD_LEGACY_KEYS: dict[str, str] = {
    "gdp_real": "Q.DE.Y.A.AG1.CA010.C.A",
}

#: Day codes that mean "we do not know the exact day".
UNKNOWN_DAY_CODES = {"40", "41"}

# Reference-period labels: 1991 | 1991-Q1 | 2005-03. Anything else in
# column 0 is one of the ragged attribute rows (Baseyear, Decimals,
# unit, ...), whose label set varies by series — so rows are classified
# by the shape of column 0, never by row index or a hardcoded label list.
_PERIOD_RE = re.compile(r"^\s*(\d{4})(?:-(Q[1-4])|-(\d{2}))?\s*$")


def _parse_period(label: str) -> pd.Timestamp | None:
    m = _PERIOD_RE.match(str(label))
    if not m:
        return None
    year, quarter, month = m.groups()
    if quarter:
        return pd.Period(f"{year}{quarter}", freq="Q").to_timestamp()
    if month:
        return pd.Timestamp(f"{year}-{month}-01")
    return pd.Timestamp(f"{year}-01-01")


def parse_bbk_vintage_date(token: str) -> tuple[pd.Timestamp | None, bool]:
    """Parse a ``DD.MM.YYYY`` Bundesbank vintage header.

    Returns ``(timestamp, day_is_exact)``. Day codes 40 and 41 mean the
    exact day is unknown; they map to the first of the month with
    ``day_is_exact=False`` rather than being discarded.
    """
    parts = str(token).strip().strip('"').split(".")
    if len(parts) != 3:
        return None, True
    dd, mm, yyyy = (p.strip() for p in parts)
    if not (yyyy.isdigit() and mm.isdigit()):
        return None, True
    if dd in UNKNOWN_DAY_CODES:
        try:
            return pd.Timestamp(int(yyyy), int(mm), 1), False
        except ValueError:
            return None, False
    try:
        return pd.Timestamp(int(yyyy), int(mm), int(dd)), True
    except ValueError:
        return None, True


def parse_bbk_rtd_csv(raw: bytes | str) -> pd.DataFrame:
    """Parse a BBKRT CSV matrix into long ``[date, vintage, value]``.

    Pure function. The payload is a wide real-time matrix: row 0 holds
    the vintage dates, row 1 is an empty spacer, then a variable number
    of attribute rows, then the data. Empty cells mean "this reference
    period did not exist in that vintage" — the ragged edge of the
    triangle — and are dropped, never filled.
    """
    if isinstance(raw, bytes):
        text = raw.decode("utf-8-sig", errors="ignore")
    else:
        text = raw.lstrip("﻿")
    if not text.strip():
        return pd.DataFrame(columns=["date", "vintage", "value"])

    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return pd.DataFrame(columns=["date", "vintage", "value"])

    header = rows[0]
    vintages: list[pd.Timestamp | None] = [None]
    # Both sentinel day codes (40 and 41) collapse to the first of the
    # month, and a real 01.MM.YYYY release can land there too. Left
    # alone, the (date, vintage) de-duplication downstream would then
    # silently discard a whole edition. Inexact dates are nudged forward
    # a day at a time, in column order, until they no longer collide;
    # exact dates always keep the day the Bundesbank published.
    parsed = [parse_bbk_vintage_date(cell) for cell in header[1:]]
    # Two passes, because an exact date must win its own day even when a
    # sentinel in an earlier column already claimed it.
    taken = {ts for ts, exact in parsed if exact and ts is not None}
    for ts, exact in parsed:
        if ts is not None and not exact:
            while ts in taken:
                ts = ts + pd.Timedelta(days=1)
            taken.add(ts)
        vintages.append(ts)

    records = []
    for row in rows[1:]:
        if not row:
            continue
        date = _parse_period(row[0])
        if date is None:
            continue                     # spacer or attribute row
        for i, cell in enumerate(row[1:], start=1):
            if i >= len(vintages) or vintages[i] is None:
                continue
            cell = (cell or "").strip()
            if cell == "":
                continue                 # ragged edge, not missing data
            try:
                value = float(cell.replace(",", "."))
            except ValueError:
                continue
            records.append((date, vintages[i], value))

    if not records:
        return pd.DataFrame(columns=["date", "vintage", "value"])
    out = pd.DataFrame(records, columns=["date", "vintage", "value"])
    return (out.drop_duplicates(subset=["date", "vintage"], keep="last")
               .sort_values(["date", "vintage"]).reset_index(drop=True))


def fetch_bundesbank_vintages(
    variable: str = "gdp_real",
    *,
    key: str | None = None,
    legacy: bool = False,
    timeout: float = 60.0,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Long ``[date, vintage, value]`` for one German variable.

    ``legacy=True`` selects the pre-2005 constant-price key, which is in
    different units from the modern chain-linked index.
    """
    if key is None:
        table = BBK_RTD_LEGACY_KEYS if legacy else BBK_RTD_KEYS
        if variable not in table:
            raise ValueError(
                f"no Bundesbank RTD key mapped for {variable!r}"
                f"{' (legacy)' if legacy else ''}; available: {sorted(table)}"
            )
        key = table[variable]
    getter = safe_get_bytes_cached if use_cache else safe_get_bytes
    raw = getter(BBK_RTD_URL.format(key=key), timeout, user_agent=_UA)
    return parse_bbk_rtd_csv(raw)


def fetch_bundesbank_panel(
    countries, variables, *, timeout: float = 60.0, use_cache: bool = True,
    **_ignored,
) -> VintagePanel:
    """Registry entry point."""
    frames, failed = [], {}
    for country in countries:
        if str(country).upper() != "DEU":
            continue
        for variable in variables:
            if variable not in BBK_RTD_KEYS:
                continue
            try:
                long = fetch_bundesbank_vintages(
                    variable, timeout=timeout, use_cache=use_cache)
            except Exception as exc:
                failed[f"{country}:{variable}"] = f"{type(exc).__name__}: {exc}"
                continue
            frames.append(normalize_vintage_frame(
                long, country="DEU", variable=variable, provider="bundesbank",
                series_id=BBK_RTD_KEYS[variable],
                units="index" if BBK_RTD_KEYS[variable].endswith(".I")
                      else "level",
            ))
    df = (pd.concat(frames, ignore_index=True) if frames
          else pd.DataFrame(columns=[
              "country", "variable", "date", "vintage", "value", "provider",
              "series_id", "units"]))
    return VintagePanel(df=df, metadata={"provider": "bundesbank",
                                         "failed": failed})


def _register() -> None:
    register_catalog("bundesbank", {
        "DEU": {
            var: SeriesSpec(
                key, "index" if key.endswith(".I") else "level",
                "bundesbank", "Gerda real-time database; true release dates")
            for var, key in BBK_RTD_KEYS.items()
        }
    })
    register_provider("bundesbank", fetch_bundesbank_panel, ["DEU"])


__all__ = [
    "BBK_RTD_URL",
    "BBK_RTD_KEYS",
    "BBK_RTD_LEGACY_KEYS",
    "UNKNOWN_DAY_CODES",
    "parse_bbk_rtd_csv",
    "parse_bbk_vintage_date",
    "fetch_bundesbank_vintages",
    "fetch_bundesbank_panel",
]
