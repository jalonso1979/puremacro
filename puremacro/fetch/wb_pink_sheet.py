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

from io import BytesIO

import numpy as np
import pandas as pd

from ._http import cached_get

_URL = ("https://thedocs.worldbank.org/en/doc/"
        "5d903e848db1d1b83e0ec8f744e55570-0350012021/related/"
        "CMO-Historical-Data-Monthly.xlsx")

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
    try:
        content = cached_get(_URL, refresh=refresh, timeout=120)
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
    return out.reset_index(drop=True)


__all__ = ["fetch"]
