"""Spain — INE's archived Contabilidad Nacional Trimestral vintages.

The live CNTR starts in 1995Q1, which is exactly where the OECD spine
starts, so the current tables buy nothing. Spain's depth is in two
archived products that INE still serves:

===================  ==========  ==============================================
segment              span        route
===================  ==========  ==============================================
``ine_base1995``     1980Q1-     JSON API, archived table 3157 (demand),
                     2004Q4      3158 (income), 3156 (supply)
``ine_base1986``     1970Q1-     one legacy workbook, ``cntrb86.xls``
                     1998Q4
===================  ==========  ==============================================

Together they take Spain from 1995 back to **1970Q1** — 100 extra
quarters — and both joins are unusually well determined: base-1995
overlaps the current vintage by 40 quarters and base-1986 overlaps
base-1995 by 76.

TWO THINGS THE JSON API WILL NOT TELL YOU
-----------------------------------------
``TABLAS_OPERACION`` returns an **empty list** for the archived
operations (CNTR2000, CNTR2008, CNE), so the archived tables are
undiscoverable through the documented route even though the tables
themselves are live and queryable by numeric id. Hence the hardcoded
ids above; there is no API call that would find them.

``det=2`` is **required**. Without it the payload has no ``Unidad``,
``Escala``, ``Periodo`` or ``NombrePeriodo`` keys at all, and a parser
written against the default response raises ``KeyError`` on the first
observation.

DATES
-----
Use ``Anyo`` plus ``Periodo.Valor``, never the ``Fecha`` field:
``Fecha`` is a local-timezone millisecond epoch of the quarter *start*
and shifts by a day across a DST boundary.

UNITS
-----
Base-1995 is already millions of euro. Base-1986 is **thousands of
millions of pesetas** and is converted here at the irrevocable rate
166.386 ESP/EUR. For a ratio splice the conversion cancels out
entirely, but converting keeps the pre-splice levels readable and the
overlap ratio near one, where a drifting ratio is easy to see.
"""
from __future__ import annotations

import io
import json

import pandas as pd

from ..._http import safe_get_bytes, safe_get_bytes_cached


INE_SERIES_URL = "https://servicios.ine.es/wstempus/js/ES/DATOS_SERIE/{cod}?nult={n}&det=2"
INE_TABLE_URL = "https://servicios.ine.es/wstempus/js/ES/DATOS_TABLA/{table}?nult={n}&det=2"
CNTRB86_URL = "https://www.ine.es/daco/daco42/daco4214/cntrb86.xls"

_UA = "puremacro (long national accounts panel)"

#: Irrevocable peseta/euro conversion rate.
ESP_PER_EUR = 166.386

#: puremacro column -> INE base-1995 series code(s), current prices,
#: seasonally and calendar adjusted, levels. A tuple is summed:
#: ``cons_hh`` must be households **plus NPISH** to match the OECD's
#: S1M sector, which is households + NPISH, not households alone.
INE_BASE1995_SERIES: dict[str, tuple[str, ...]] = {
    "gdp":       ("CTA1527",),
    "cons_hh":   ("CTA1525", "CTA1523"),   # hogares + ISFLSH = S1M
    "cons_gov":  ("CTA1522",),
    "inv":       ("CTA1520",),             # FBCF = P51G
    "capform":   ("CTA1521",),             # FBC = P5
    "exports":   ("CTA1515",),
    "imports":   ("CTA1512",),
    # The asset split this vintage publishes is coarser than the modern
    # one: equipment / construction / other products, where the current
    # accounts separate dwellings from other structures. Construction is
    # therefore exposed under its own name rather than mapped onto
    # `inv_struct`, which would silently drop dwellings into it.
    "inv_equip":  ("CTA1519",),
    "inv_constr": ("CTA1518",),
    "inv_other":  ("CTA1517",),
}

#: puremacro column -> column index in ``cntrb86.xls``, sheet
#: "demanda a precios corrientes". Verified: the expenditure identity
#: closes exactly on these at 1970Q1
#: (407.9 + 63 + 176.4 + 8.99 + 79.174 - 92.764 = 642.7 = GDP).
CNTRB86_COLUMNS: dict[str, int] = {
    "gdp": 2,
    "cons_hh": 3,        # consumo privado NACIONAL (residents), not interior
    "cons_gov": 5,
    "inv": 6,            # FBCF
    "inv_equip": 7,
    "inv_constr": 8,
    "inventories": 9,
    "exports": 11,
    "imports": 15,
}

CNTRB86_SHEET = "demanda a precios corrientes"


def _get(url: str, *, timeout: float, use_cache: bool) -> bytes:
    getter = safe_get_bytes_cached if use_cache else safe_get_bytes
    return getter(url, timeout, user_agent=_UA)


def parse_ine_series(payload: dict | str | bytes) -> pd.Series:
    """Parse one INE ``DATOS_SERIE`` payload into a quarterly Series.

    Pure function. Indexed by quarter **start**, matching
    :func:`puremacro.fetch.qna_panel`. Null observations are dropped —
    the API emits them (base-1986 GDP at 1970T1, for instance) and they
    are absence, not zero.
    """
    data: dict = json.loads(payload) if isinstance(payload, (bytes, str)) else payload
    rows = data.get("Data") or []
    out: dict[pd.Timestamp, float] = {}
    for o in rows:
        if o.get("Valor") is None:
            continue
        per = o.get("Periodo") or {}
        year, q = o.get("Anyo"), per.get("Valor")
        if year is None or q is None:
            continue
        out[pd.Period(f"{int(year)}Q{int(q)}", freq="Q").to_timestamp()] = \
            float(o["Valor"])
    s = pd.Series(out, dtype=float).sort_index()
    s.name = data.get("COD")
    return s


def fetch_ine_base1995(
    *, timeout: float = 90.0, use_cache: bool = True, n: int = 600,
) -> pd.DataFrame:
    """Spain's base-1995 CNTR: current prices, SA, millions of euro.

    Covers 1980Q1-2004Q4, which overlaps the live vintage by 40
    quarters.
    """
    cols: dict[str, pd.Series] = {}
    for name, codes in INE_BASE1995_SERIES.items():
        parts = []
        for cod in codes:
            raw = _get(INE_SERIES_URL.format(cod=cod, n=n),
                       timeout=timeout, use_cache=use_cache)
            parts.append(parse_ine_series(raw))
        if not parts:
            continue
        # Summed components must both be present in a quarter; min_count
        # stops a missing NPISH quarter silently becoming households-only.
        cols[name] = (pd.concat(parts, axis=1)
                      .sum(axis=1, min_count=len(parts)))
    return pd.DataFrame(cols).sort_index()


def parse_cntrb86_workbook(
    raw: bytes, *, sheet: str = CNTRB86_SHEET, esp_per_eur: float = ESP_PER_EUR,
) -> pd.DataFrame:
    """Parse ``cntrb86.xls`` into millions of euro, quarter-start index.

    Pure apart from the lazy ``xlrd`` import pandas needs for ``.xls``.
    The year appears only on the Q1 row and is carried forward; the
    source is in thousands of millions of pesetas.
    """
    df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet, header=None)
    year = pd.to_numeric(df[0], errors="coerce").ffill()
    quarter = pd.to_numeric(df[1], errors="coerce")
    ok = year.notna() & quarter.between(1, 4)
    idx = pd.PeriodIndex(
        [f"{int(y)}Q{int(q)}" for y, q in zip(year[ok], quarter[ok])],
        freq="Q").to_timestamp()

    scale = 1000.0 / esp_per_eur          # 10^9 ESP -> 10^6 EUR
    out: dict[str, pd.Series] = {}
    for name, col in CNTRB86_COLUMNS.items():
        if col >= df.shape[1]:
            continue
        vals = pd.to_numeric(df[col][ok], errors="coerce").to_numpy(dtype=float)
        out[name] = pd.Series(vals * scale, index=idx)
    frame = pd.DataFrame(out).sort_index()
    # P5 = P51G + inventories, so the coarse vintage can still fill the
    # `capform` column the modern panel carries.
    if {"inv", "inventories"} <= set(frame.columns):
        frame["capform"] = frame[["inv", "inventories"]].sum(
            axis=1, min_count=2)
    return frame


def fetch_ine_base1986(
    *, timeout: float = 120.0, use_cache: bool = True,
) -> pd.DataFrame:
    """Spain's base-1986 CNTR from the legacy workbook: 1970Q1-1998Q4."""
    return parse_cntrb86_workbook(
        _get(CNTRB86_URL, timeout=timeout, use_cache=use_cache))


def spain_segments(
    *, timeout: float = 120.0, use_cache: bool = True,
) -> list[tuple[str, pd.DataFrame]]:
    """The archived Spanish segments, oldest last.

    Returned in the order :func:`.._splice.ratio_splice` expects —
    newest first — but **without** a spine: the caller supplies the
    OECD panel as the first element, so that the modern levels are the
    published ones.
    """
    return [
        ("ine_base1995", fetch_ine_base1995(timeout=timeout,
                                            use_cache=use_cache)),
        ("ine_base1986", fetch_ine_base1986(timeout=timeout,
                                            use_cache=use_cache)),
    ]


__all__ = [
    "INE_SERIES_URL",
    "CNTRB86_URL",
    "ESP_PER_EUR",
    "INE_BASE1995_SERIES",
    "CNTRB86_COLUMNS",
    "parse_ine_series",
    "parse_cntrb86_workbook",
    "fetch_ine_base1995",
    "fetch_ine_base1986",
    "spain_segments",
]
