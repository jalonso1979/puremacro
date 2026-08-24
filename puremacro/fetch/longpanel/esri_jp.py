"""Japan — Cabinet Office (ESRI) Quarterly Estimates of GDP.

The current Quarterly Estimates start in 1994Q1, exactly where the OECD
spine does. Japan's depth is in the archived releases, which ESRI still
serves as static CSV at URLs frozen since 2012:

====================  ================  =====================================
segment               span              basis
====================  ================  =====================================
``jp_qe_current``     1994Q1-           2008SNA, 2020 benchmark
``jp_retro2015``      1980Q1-1995Q1     2008SNA, 2015 benchmark (retroactive)
``jp_sna93``          1980Q1-2004Q2     93SNA, 1995 benchmark
``jp_sna68``          1955Q2-2001Q1     68SNA, 1990 benchmark
====================  ================  =====================================

The chain used by default is OECD -> ``jp_sna93`` -> ``jp_sna68``,
chosen for overlap rather than recency: 93SNA overlaps the modern
vintage by 42 quarters and 68SNA overlaps 93SNA by 85, where the
retroactive file overlaps the spine by only five. Long overlaps are
what make a splice ratio estimable, and the drift diagnostic in
:mod:`.._splice` is only as good as the overlap it is measured on.

That reaches **1955Q2 — 155 quarters before OECD**.

FOUR WAYS THIS FILE FORMAT BITES
--------------------------------
1. **Encoding is cp932 (Shift-JIS), not UTF-8.** ``b.decode('utf-8')``
   raises outright. Unchanged across 25 years of releases.
2. **The header block is ragged**, 6-8 rows depending on vintage, and
   the English labels are spread across two or three physical rows
   because of merged spreadsheet cells — some containing embedded
   newlines inside quoted fields. Parse with the :mod:`csv` module,
   which honours quoted newlines; ``splitlines()`` corrupts them. The
   first data row is found by regex, never by a fixed ``skiprows``.
3. **Column order changed between SNA vintages.** In the 68SNA files
   GDP is the *twelfth* column, after the components; in 93SNA and
   2008SNA it is the second. Reading column 1 in a 68SNA file yields
   private consumption while looking entirely plausible.
4. **Units are billions of yen at ANNUAL RATES**, where the OECD spine
   is millions per quarter — a factor of 1000/4. A ratio splice absorbs
   that automatically, because it is a constant, and the spliced output
   therefore comes out in the spine's units throughout. This is the
   design working rather than luck: constant factors (units,
   annualisation, rebasing) disappear into the ratio, while a
   non-constant difference still surfaces as drift. Do not compare a
   raw segment from this module against the spliced panel and expect
   the same numbers.
5. **Only the Q1 row carries the year.** Q2-Q4 rows carry a bare month
   range, with inconsistent leading whitespace between vintages
   (``' 4- 6.'`` vs ``'4- 6.'``), so the year must be carried forward.
"""
from __future__ import annotations

import csv
import io
import re

import pandas as pd

from ..._http import safe_get_bytes, safe_get_bytes_cached


_UA = "puremacro (long national accounts panel)"

_ESRI = "https://www.esri.cao.go.jp"

#: segment label -> (url, layout key). See :data:`JP_LAYOUTS`.
JP_FILES: dict[str, tuple[str, str]] = {
    "jp_qe_current": (
        f"{_ESRI}/jp/sna/data/data_list/sokuhou/files/2026/qe262/tables/"
        "gaku-mk2621.csv", "modern"),
    "jp_retro2015": (
        f"{_ESRI}/jp/sna/data/data_list/h27_retroactive/tables/"
        "gaku-mk_2780.csv", "modern"),
    "jp_sna93": (
        f"{_ESRI}/en/sna/data/sokuhou/files/2004/qe042_2/__icsFiles/"
        "afieldfile/2012/02/28/gaku_mk0422.csv", "modern"),
    "jp_sna68": (
        f"{_ESRI}/en/sna/data/sokuhou/files/2001/qe011/__icsFiles/"
        "afieldfile/2012/02/28/gaku_mk01168.csv", "sna68"),
}

#: layout -> puremacro column -> source column index (or tuple to sum).
#: The two layouts are NOT interchangeable: note where ``gdp`` sits.
JP_LAYOUTS: dict[str, dict[str, tuple[int, ...]]] = {
    "modern": {
        "gdp":        (1,),
        "cons_hh":    (2,),          # private final consumption (incl. NPISH)
        "cons_gov":   (8,),
        "inv":        (5, 6, 9),     # residential + non-resi + public
        "inventories": (7, 10),
        "exports":    (12,),
        "imports":    (13,),
    },
    "sna68": {
        "gdp":        (11,),         # <- twelfth column, after the components
        "cons_hh":    (1,),
        "cons_gov":   (5,),
        "inv":        (2, 3, 6),
        "inventories": (4, 7),
        "exports":    (9,),
        "imports":    (10,),
    },
}

#: The default splice chain, newest first, spine excluded. Chosen for
#: overlap length, not recency -- see the module docstring.
JP_DEFAULT_CHAIN: tuple[str, ...] = ("jp_sna93", "jp_sna68")

_YEAR_MONTH = re.compile(r"^\s*((?:19|20)\d\d)/\s*(\d{1,2})")
_MONTH_ONLY = re.compile(r"^\s*(\d{1,2})\s*-")


def _to_float(cell: str) -> float:
    txt = (cell or "").replace(",", "").replace("△", "-").strip()
    if txt in {"", "-", "...", "…", "***"}:
        return float("nan")
    try:
        return float(txt)
    except ValueError:
        return float("nan")


def parse_esri_csv(raw: bytes | str, layout: str = "modern") -> pd.DataFrame:
    """Parse an ESRI quarterly CSV into puremacro columns.

    Pure function: bytes in, DataFrame out, indexed by quarter start,
    in billions of yen. ``layout`` selects the column map — passing the
    wrong one silently returns private consumption as GDP, so it is
    required rather than guessed.

    The 68SNA files carry more than one table in a single CSV, so
    parsing stops at the first repeated quarter rather than
    concatenating a second block onto the first.
    """
    if layout not in JP_LAYOUTS:
        raise ValueError(
            f"unknown ESRI layout {layout!r}; expected one of "
            f"{sorted(JP_LAYOUTS)}. The 68SNA files put GDP in a different "
            "column from the 93SNA/2008SNA ones, so this cannot be inferred."
        )
    if isinstance(raw, bytes):
        text = raw.decode("cp932", errors="strict")
    else:
        text = raw
    rows = list(csv.reader(io.StringIO(text)))

    colmap = JP_LAYOUTS[layout]
    needed = max(i for idx in colmap.values() for i in idx)

    year: int | None = None
    seen: set[pd.Timestamp] = set()
    records: dict[pd.Timestamp, dict[str, float]] = {}
    for r in rows:
        if not r:
            continue
        label = r[0] or ""
        m = _YEAR_MONTH.match(label)
        if m:
            year, month = int(m.group(1)), int(m.group(2))
        else:
            m2 = _MONTH_ONLY.match(label)
            if not m2 or year is None:
                continue
            month = int(m2.group(1))
        if not 1 <= month <= 12 or len(r) <= needed:
            continue
        ts = pd.Period(f"{year}Q{(month - 1) // 3 + 1}", freq="Q").to_timestamp()
        if ts in seen:
            break                      # a second table has started
        vals = {name: sum(_to_float(r[i]) for i in idx)
                for name, idx in colmap.items()}
        if pd.isna(vals.get("gdp")):
            continue
        seen.add(ts)
        records[ts] = vals

    frame = pd.DataFrame.from_dict(records, orient="index").sort_index()
    if not frame.empty and {"inv", "inventories"} <= set(frame.columns):
        frame["capform"] = frame[["inv", "inventories"]].sum(axis=1,
                                                             min_count=2)
    return frame


def fetch_esri_segment(
    label: str, *, timeout: float = 120.0, use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch and parse one archived ESRI release."""
    if label not in JP_FILES:
        raise ValueError(
            f"unknown ESRI segment {label!r}; available: {sorted(JP_FILES)}")
    url, layout = JP_FILES[label]
    getter = safe_get_bytes_cached if use_cache else safe_get_bytes
    return parse_esri_csv(getter(url, timeout, user_agent=_UA), layout)


def japan_segments(
    chain: tuple[str, ...] = JP_DEFAULT_CHAIN,
    *, timeout: float = 120.0, use_cache: bool = True,
) -> list[tuple[str, pd.DataFrame]]:
    """The archived Japanese segments, newest first, spine excluded."""
    return [(label, fetch_esri_segment(label, timeout=timeout,
                                       use_cache=use_cache))
            for label in chain]


__all__ = [
    "JP_FILES",
    "JP_LAYOUTS",
    "JP_DEFAULT_CHAIN",
    "parse_esri_csv",
    "fetch_esri_segment",
    "japan_segments",
]
