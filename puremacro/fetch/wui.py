"""World Uncertainty Index (Ahir-Bloom-Furceri, IMF) fetcher.

Paper: Ahir, Bloom, and Furceri (2018). "The World Uncertainty Index."
Home:  https://worlduncertaintyindex.com/

Data layout (as of 2025-08):
  Quarterly file  — sheet "T2": first col "year" (format "1952q1"), remaining cols are ISO-3.
  Monthly file    — sheet "T1": first col "date" (datetime), remaining cols are ISO-3.

URL notes:
  _WUI_Q_URL  returns HTTP 200 as of 2025-08.
  _WUI_M_URL  returned HTTP 404; local Drive copy used as fallback.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

from ._http import cached_get

_WUI_Q_URL = "https://worlduncertaintyindex.com/wp-content/uploads/2024/10/WUI_Data.xlsx"
# Monthly URL was 404 as of 2025-08; local fallback is used automatically when the URL fails.
_WUI_M_URL = "https://worlduncertaintyindex.com/wp-content/uploads/2024/10/WUI_M_dataset_2025_08.xlsx"

_LOCAL_Q = Path(
    "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com"
    "/My Drive/MAV/WUI_Data.xlsx"
)
_LOCAL_M = Path(
    "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com"
    "/My Drive/MAV/WUI_M_dataset_2025_08.xlsx"
)


def _parse(content: bytes, var_name: str, sheet: str, freq: str) -> pd.DataFrame:
    raw = pd.read_excel(BytesIO(content), sheet_name=sheet)
    first_col = raw.columns[0]
    long = raw.melt(id_vars=[first_col], var_name="code", value_name="value")
    long = long.rename(columns={first_col: "date"})
    if freq == "Q":
        # Date column is like "1952q1" — parse via PeriodIndex.
        long["date"] = pd.PeriodIndex(long["date"].astype(str), freq="Q").to_timestamp()
    else:
        long["date"] = pd.to_datetime(long["date"])
    long["code"] = long["code"].astype(str).str.upper().str.strip()
    # Only keep rows whose code is exactly 3 characters (ISO-3 country codes).
    long = long[long["code"].str.len() == 3]
    long = long.dropna(subset=["value"])
    long["variable"] = var_name
    long["sa_source"] = "none"
    long["source"] = "WUI"
    return long[["code", "date", "variable", "value", "sa_source", "source"]]


def fetch_q(refresh: bool = False) -> pd.DataFrame:
    """Return quarterly WUI in long form.

    Sheet "T2" of WUI_Data.xlsx has ISO-3 country columns; data starts 1952Q1.
    """
    try:
        content = cached_get(_WUI_Q_URL, refresh=refresh)
    except Exception:
        if _LOCAL_Q.exists():
            content = _LOCAL_Q.read_bytes()
        else:
            raise
    return _parse(content, var_name="wui_q", sheet="T2", freq="Q")


def fetch_m(refresh: bool = False) -> pd.DataFrame:
    """Return monthly WUI in long form.

    Sheet "T1" of WUI_M_dataset_*.xlsx has ISO-3 country columns; data starts 2008-01.
    Falls back to local Drive copy when the online URL is unavailable.
    """
    try:
        content = cached_get(_WUI_M_URL, refresh=refresh)
    except Exception:
        if _LOCAL_M.exists():
            content = _LOCAL_M.read_bytes()
        else:
            raise
    return _parse(content, var_name="wui_m", sheet="T1", freq="M")
