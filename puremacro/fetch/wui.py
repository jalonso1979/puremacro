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
import os
from pathlib import Path

import pandas as pd

from ._http import cached_get

_WUI_Q_URL = "https://worlduncertaintyindex.com/wp-content/uploads/2024/10/WUI_Data.xlsx"
# Monthly URL was 404 as of 2025-08; local fallback is used automatically when the URL fails.
_WUI_M_URL = "https://worlduncertaintyindex.com/wp-content/uploads/2024/10/WUI_M_dataset_2025_08.xlsx"

#: Where the offline copies live when the upstream URL is unavailable. This
#: used to be an absolute path into one person's Google Drive, so on every other
#: machine the fallback silently could not fire and `fetch_m` — whose URL has
#: 404'd since 2025-08 — always raised. Inside `build_all` that is caught and
#: printed, so the monthly uncertainty composite was quietly built from six
#: proxies instead of seven. Same environment variable as the examples use
#: (`puremacro/examples/*.py`), so one setting covers both.
_MAV_ROOT = Path(os.environ.get(
    "PUREMACRO_MAV_ROOT",
    Path.home() / "Library" / "CloudStorage" / "My Drive" / "MAV"))
_LOCAL_Q = _MAV_ROOT / "WUI_Data.xlsx"
_LOCAL_M = _MAV_ROOT / "WUI_M_dataset_2025_08.xlsx"


def _fallback(local: Path, url: str, exc: Exception) -> bytes:
    """Read the offline copy, or say precisely why there is none.

    Re-raising the network error alone sent the caller after a URL that is
    simply gone; naming the path and the environment variable that sets it is
    the difference between an actionable failure and a puzzle.
    """
    if local.exists():
        return local.read_bytes()
    raise FileNotFoundError(
        f"{url} could not be fetched ({exc!r}) and no offline copy exists at "
        f"{local}. Download the workbook from https://worlduncertaintyindex.com "
        f"and put it there, or point PUREMACRO_MAV_ROOT at the directory "
        f"holding it."
    ) from exc


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
    except Exception as exc:
        content = _fallback(_LOCAL_Q, _WUI_Q_URL, exc)
    return _parse(content, var_name="wui_q", sheet="T2", freq="Q")


def fetch_m(refresh: bool = False) -> pd.DataFrame:
    """Return monthly WUI in long form.

    Sheet "T1" of WUI_M_dataset_*.xlsx has ISO-3 country columns; data starts 2008-01.
    Falls back to local Drive copy when the online URL is unavailable.
    """
    try:
        content = cached_get(_WUI_M_URL, refresh=refresh)
    except Exception as exc:
        content = _fallback(_LOCAL_M, _WUI_M_URL, exc)
    return _parse(content, var_name="wui_m", sheet="T1", freq="M")
