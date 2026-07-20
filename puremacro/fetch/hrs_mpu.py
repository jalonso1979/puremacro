"""Husted-Rogers-Sun (2020) monetary policy uncertainty index.

Source: Husted, Lucas, John Rogers, and Bo Sun (2020), "Monetary Policy
Uncertainty," *Journal of Monetary Economics* 115:20-36. Index mirrored
at Iacoviello's site (he co-publishes complementary measures).

CSV columns: ``Date, USA, GBR, DEU, FRA, ITA, ESP, JPN, CAN, SWE, EA``.

The aggregate code ``EA`` (Eurozone) is dropped via ``src._codes`` so the
panel keeps the 9 pure-country series.
"""
from __future__ import annotations

import io

import pandas as pd

from .._codes import is_country
from ._http import cached_get

_HRS_URL = "https://www.matteoiacoviello.com/mpu_files/hrs_mpu.csv"

_EMPTY = pd.DataFrame(
    columns=["code", "date", "variable", "value", "sa_source", "source"]
)


def _cached_get_text(url: str, refresh: bool = False) -> str:
    """Wrap cached_get to return decoded text (test seam)."""
    return cached_get(url, refresh=refresh).decode("utf-8")


def _parse_csv_text(text: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text))
    if "Date" not in raw.columns:
        raise ValueError(f"HRS MPU CSV missing Date column. Got: {list(raw.columns)}")
    raw["date"] = pd.to_datetime(raw["Date"])
    code_cols = [c for c in raw.columns if c not in ("Date", "date") and is_country(c)]
    long = raw[["date"] + code_cols].melt(id_vars="date", var_name="code", value_name="value")
    long = long.dropna(subset=["value"])
    long["variable"] = "mpu_m"
    long["sa_source"] = "none"
    long["source"] = "HRS MPU (Husted-Rogers-Sun 2020)"
    return long[["code", "date", "variable", "value", "sa_source", "source"]].reset_index(drop=True)


def fetch(refresh: bool = False) -> pd.DataFrame:
    """Return long-form HRS monetary policy uncertainty (9 countries since 1985-01)."""
    try:
        text = _cached_get_text(_HRS_URL, refresh=refresh)
        return _parse_csv_text(text)
    except Exception as exc:
        print(f"[hrs_mpu] fetch failed: {exc}")
        return _EMPTY.copy()


__all__ = ["fetch"]
