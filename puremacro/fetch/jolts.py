"""BLS JOLTS fetcher (key-free FRED fredgraph mirror).

Job Openings and Labor Turnover Survey: openings, hires, quits,
layoffs & discharges and total separations — level (thousands) and
rate (percent) — monthly from 2000-12, seasonally adjusted or not,
total nonfarm or by JOLTS industry supersector.

Series ID layout (fredgraph shorthand of the BLS JOLTS ids):
    JTS<industry:4|empty><element:2><measure:1>       (SA)
    JTU<industry:4|empty><element:2><measure:1>       (NSA)
Elements: JO=openings, HI=hires, QU=quits, LD=layoffs & discharges,
TS=total separations. Measures: L=level (thousands), R=rate (%).
Total nonfarm omits the industry code entirely (``JTSJOL``).

The industry map below ships ONLY codes verified live against the
key-free fredgraph endpoint (2026-07-21); several BLS supersectors
(e.g. 5400 professional & business services, 1100 mining & logging)
are not mirrored under this shorthand and are deliberately absent —
pass a raw 4-digit code at your own risk, the fetcher will raise if
the mirror 404s. State-level JOLTS is not exposed key-free and is out
of scope here.
"""
from __future__ import annotations

from io import BytesIO
from typing import Iterable

import pandas as pd

from ._http import cached_get

_FREDGRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

_ELEMENTS: dict[str, str] = {
    "openings": "JO",
    "hires": "HI",
    "quits": "QU",
    "layoffs": "LD",
    "separations": "TS",
}

#: JOLTS industry supersectors verified on fredgraph (2026-07-21).
#: 'total' (nonfarm) maps to the empty code.
INDUSTRY_CODES: dict[str, str] = {
    "total": "",
    "total_private": "1000",
    "construction": "2300",
    "manufacturing": "3000",
    "durable_goods": "3200",
    "nondurable_goods": "3400",
    "trade_transport_utilities": "4000",
    "retail_trade": "4400",
    "private_education_health": "6000",
    "leisure_hospitality": "7000",
    "arts_entertainment": "7100",
    "accommodation_food": "7200",
    "government": "9000",
    "state_local_government": "9200",
}

__all__ = ["fetch_jolts", "INDUSTRY_CODES"]


def _series(sid: str, *, refresh: bool) -> pd.Series:
    raw = cached_get(_FREDGRAPH.format(sid), refresh=refresh)
    df = pd.read_csv(BytesIO(raw))
    s = pd.to_numeric(df[df.columns[1]], errors="coerce")
    s.index = pd.to_datetime(df[df.columns[0]])
    return s.dropna()


def fetch_jolts(
    *,
    elements: Iterable[str] = ("openings", "hires", "quits", "layoffs",
                               "separations"),
    industries: Iterable[str] = ("total",),
    sa: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Tidy JOLTS panel: one row per (date, industry, element).

    Parameters
    ----------
    elements : iterable of {'openings','hires','quits','layoffs',
        'separations'}.
    industries : iterable of :data:`INDUSTRY_CODES` keys, or raw
        4-digit JOLTS supersector codes.
    sa : True for seasonally adjusted (JTS...), False for NSA (JTU...).
    refresh : bypass the on-disk HTTP cache.

    Returns
    -------
    DataFrame ``[date, industry, element, level, rate]`` — level in
    thousands, rate in percent, monthly from 2000-12.
    """
    prefix = "JTS" if sa else "JTU"
    frames = []
    for ind in industries:
        code = INDUSTRY_CODES.get(ind)
        if code is None:
            if not (isinstance(ind, str) and ind.isdigit()
                    and len(ind) == 4):
                raise ValueError(
                    f"fetch_jolts: unknown industry {ind!r}; use one of "
                    f"{sorted(INDUSTRY_CODES)} or a raw 4-digit code")
            code = ind
        for el in elements:
            if el not in _ELEMENTS:
                raise ValueError(
                    f"fetch_jolts: unknown element {el!r}; use one of "
                    f"{sorted(_ELEMENTS)}")
            base = f"{prefix}{code}{_ELEMENTS[el]}"
            level = _series(base + "L", refresh=refresh)
            rate = _series(base + "R", refresh=refresh)
            df = pd.DataFrame({"level": level, "rate": rate})
            df.index.name = "date"
            df = df.reset_index()
            df.insert(1, "industry", ind)
            df.insert(2, "element", el)
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["industry", "element", "date"],
                           ignore_index=True)
