"""World Bank Pink Sheet — monthly commodity benchmarks since 1960.

Source: https://www.worldbank.org/en/research/commodity-markets
Single Excel file, sheet "Monthly Prices". Columns include 'Crude oil, Brent',
'Natural gas, Europe', 'Coal, Australian' etc. Returns long-form schema with
code='WLD' (a single global series per variable).

The ``value`` column is the *price level* in USD (per barrel for oil, per
mmbtu for gas, per metric ton for coal). Downstream consumers that need
log prices must take ``np.log`` themselves.
"""
from __future__ import annotations

import re
import warnings
from io import BytesIO

import numpy as np
import pandas as pd

from ._http import cached_get

#: The page that always links to the current workbook. The World Bank re-issues
#: the Pink Sheet under a **new document id every month**, so a pinned id is a
#: snapshot, not a source.
_LANDING = "https://www.worldbank.org/en/research/commodity-markets"

#: Any ``thedocs.worldbank.org`` link to the monthly historical workbook.
_XLSX_RE = re.compile(
    r"https://thedocs\.worldbank\.org/[^\"'\s<>\\]*?CMO-Historical-Data-Monthly\.xlsx")

#: Last known good document id, used only when the landing page cannot be read
#: (offline, or the page markup changed). This exact id is the one the module
#: was pinned to, and it is **frozen**: its workbook's last row is 2024M12, so
#: `refresh=True` re-downloaded the same stale edition forever and every energy
#: benchmark -- `oil_brent_m`, `oil_wti_m`, `gas_eu_m`, `coal_au_m` and the
#: rest -- silently stopped there while merging into the panel as if current.
_FALLBACK_URL = ("https://thedocs.worldbank.org/en/doc/"
                 "5d903e848db1d1b83e0ec8f744e55570-0350012021/related/"
                 "CMO-Historical-Data-Monthly.xlsx")

#: Warn when the newest observation is older than this. The Pink Sheet is
#: published in the first week of each month, so anything beyond two quarters
#: means the resolved workbook is not the current one.
_STALE_AFTER_DAYS = 185


def _resolve_url(*, refresh: bool = False) -> str:
    """The current workbook URL, from the landing page; fallback if unreadable."""
    try:
        page = cached_get(_LANDING, refresh=refresh, timeout=60)
    except Exception:
        return _FALLBACK_URL
    try:
        text = page.decode("utf-8", errors="ignore")
    except Exception:
        return _FALLBACK_URL
    hit = _XLSX_RE.search(text)
    return hit.group(0) if hit else _FALLBACK_URL


def _warn_if_stale(out: pd.DataFrame) -> None:
    """Say so when the resolved workbook has stopped advancing."""
    if out.empty:
        return
    last = pd.Timestamp(out["date"].max())
    age = (pd.Timestamp.today().normalize() - last).days
    if age > _STALE_AFTER_DAYS:
        warnings.warn(
            f"wb_pink_sheet: the resolved workbook ends at "
            f"{last.date()}, {age} days ago. The World Bank re-issues the Pink "
            f"Sheet under a new document id monthly; this one has stopped "
            f"advancing, so every commodity benchmark below is frozen there "
            f"while merging into the panel as if current. Check "
            f"{_LANDING} for the current file.",
            UserWarning, stacklevel=2)

# Column-name substring -> output variable name. Match is case-insensitive.
_COL_MAP: dict[str, str] = {
    "crude oil, average":          "oil_avg_m",
    "crude oil, brent":            "oil_brent_m",
    "crude oil, dubai":            "oil_dubai_m",
    "crude oil, wti":              "oil_wti_m",
    "coal, australian":            "coal_au_m",
    "coal, south african":         "coal_za_m",
    "natural gas, us":             "gas_us_m",
    "natural gas, europe":         "gas_eu_m",
    "liquefied natural gas, japan": "gas_jp_lng_m",
}

_EMPTY = pd.DataFrame(columns=["code", "date", "variable", "value", "sa_source", "source"])


def fetch(refresh: bool = False) -> pd.DataFrame:
    """Return Pink Sheet monthly benchmarks in long-form schema (code='WLD')."""
    url = _resolve_url(refresh=refresh)
    try:
        content = cached_get(url, refresh=refresh, timeout=120)
    except Exception as e:
        print(f"[wb_pink_sheet] download failed: {e}")
        return _EMPTY.copy()
    try:
        df = pd.read_excel(BytesIO(content), sheet_name="Monthly Prices",
                           skiprows=4, header=0)
    except Exception as e:
        print(f"[wb_pink_sheet] excel parse failed: {e}")
        return _EMPTY.copy()
    if df.empty:
        return _EMPTY.copy()

    # First col is date (YYYYMMM, e.g. 1960M01). The skiprows=4 + header=0 setup
    # leaves a units row at index 0 — drop it.
    date_col = df.columns[0]
    df = df.iloc[1:].copy()

    # Coerce date to string and keep only proper YYYYM## rows.
    df[date_col] = df[date_col].astype(str).str.strip()
    df = df[df[date_col].str.match(r"^\d{4}M\d{2}$")]
    df["date"] = pd.to_datetime(df[date_col].str.replace("M", "-") + "-01")

    rows = []
    for col in df.columns:
        if col == date_col or col == "date":
            continue
        col_str = str(col).strip().lower()
        var_name = None
        for needle, name in _COL_MAP.items():
            if needle in col_str:
                var_name = name
                break
        if var_name is None:
            continue
        s = pd.to_numeric(
            df[col].replace({"…": np.nan, "..": np.nan, "...": np.nan, "n.a.": np.nan}),
            errors="coerce",
        )
        sub = pd.DataFrame({
            "code": "WLD",
            "date": df["date"].values,
            "variable": var_name,
            "value": s.values,
            "sa_source": "none",
            "source": "WorldBank:PinkSheet:MonthlyPrices",
        }).dropna(subset=["value"])
        if not sub.empty:
            rows.append(sub)
    if not rows:
        return _EMPTY.copy()
    out = pd.concat(rows, ignore_index=True)
    # In case the workbook ships a duplicate column for the same commodity,
    # keep the first occurrence per (variable, date).
    out = out.drop_duplicates(subset=["variable", "date"], keep="first")
    out = out.reset_index(drop=True)
    _warn_if_stale(out)
    return out


__all__ = ["fetch"]
