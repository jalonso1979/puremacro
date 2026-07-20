"""BIS Effective Exchange Rate (NEER) loader.

Fetches the BIS nominal broad-basket effective exchange rate (monthly,
base 2020=100) for one or more countries and aggregates to quarterly.

Endpoint pattern::

    https://stats.bis.org/api/v2/data/dataflow/BIS/WS_EER/1.0/M.N.B.<CC>?format=csv

where ``M`` = monthly, ``N`` = nominal, ``B`` = broad basket,
``<CC>`` = BIS 2-letter country code.

Network failures return an empty DataFrame with columns
``[code, qdate, neer_q]`` rather than raising — callers should check
for emptiness and fall back to bundled data if needed.
"""
from __future__ import annotations

import io
from typing import Iterable, Optional

import pandas as pd

from puremacro._http import safe_get_bytes


_BIS_NEER_URL = (
    "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_EER/1.0/"
    "M.N.B.{cc}?format=csv"
)

_BIS_TO_ISO3 = {
    "US": "USA", "DE": "DEU", "FR": "FRA", "IT": "ITA", "ES": "ESP",
    "GB": "GBR", "JP": "JPN", "KR": "KOR", "CA": "CAN", "AU": "AUS",
    "NZ": "NZL", "MX": "MEX", "BR": "BRA", "CL": "CHL", "AR": "ARG",
    "TR": "TUR", "IL": "ISR", "ZA": "ZAF", "IN": "IND", "CN": "CHN",
    "CH": "CHE", "SE": "SWE", "NO": "NOR", "DK": "DNK", "FI": "FIN",
    "AT": "AUT", "BE": "BEL", "NL": "NLD", "PT": "PRT", "GR": "GRC",
    "IE": "IRL", "PL": "POL", "CZ": "CZE", "HU": "HUN", "SK": "SVK",
    "SI": "SVN", "EE": "EST", "LV": "LVA", "LT": "LTU", "LU": "LUX",
}

_EMPTY = pd.DataFrame(columns=["code", "qdate", "neer_q"])


def fetch_bis_neer(
    bis_code: str,
    *,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch BIS nominal broad-basket NEER for one country, quarterly.

    Parameters
    ----------
    bis_code : str
        BIS 2-letter country code (e.g. ``"US"``, ``"DE"``).
    timeout : float
        HTTP request timeout in seconds.

    Returns
    -------
    pd.DataFrame with columns ``[code, qdate, neer_q]``:

    * ``code``   — ISO-3 country code (e.g. ``"USA"``).
    * ``qdate``  — ``pd.Timestamp`` at quarter-start (e.g. 2020-01-01).
    * ``neer_q`` — mean of monthly NEER values within the quarter
                   (base 2020 = 100).

    Returns an empty DataFrame (same columns) on network failure, HTTP
    error, or unknown ``bis_code``.
    """
    iso3 = _BIS_TO_ISO3.get(bis_code.upper())
    if iso3 is None:
        return _EMPTY.copy()

    url = _BIS_NEER_URL.format(cc=bis_code.upper())
    try:
        raw = safe_get_bytes(url, timeout=timeout)
    except Exception:
        return _EMPTY.copy()

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        return _EMPTY.copy()

    # Locate TIME_PERIOD and OBS_VALUE columns (case-insensitive fallback).
    cols_upper = {c.upper(): c for c in df.columns}
    time_col = cols_upper.get("TIME_PERIOD") or cols_upper.get("TIME")
    val_col = cols_upper.get("OBS_VALUE") or cols_upper.get("VALUE")
    if time_col is None or val_col is None:
        return _EMPTY.copy()

    df = df[[time_col, val_col]].copy()
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col])
    if df.empty:
        return _EMPTY.copy()

    # Parse monthly dates (format "YYYY-MM") and assign quarter-start.
    df["_month"] = pd.to_datetime(df[time_col], format="%Y-%m", errors="coerce")
    df = df.dropna(subset=["_month"])
    df["qdate"] = df["_month"].dt.to_period("Q").dt.to_timestamp()

    # Aggregate to quarterly via mean of months in the quarter.
    agg = (
        df.groupby("qdate", as_index=False)[val_col]
        .mean()
        .rename(columns={val_col: "neer_q"})
    )
    agg["code"] = iso3
    agg = agg.sort_values("qdate").reset_index(drop=True)
    return agg[["code", "qdate", "neer_q"]]


def fetch_bis_neer_panel(
    codes: Optional[Iterable[str]] = None,
    *,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch BIS NEER panel for multiple countries.

    Parameters
    ----------
    codes : iterable of str, optional
        BIS 2-letter country codes.  Defaults to all keys in
        ``_BIS_TO_ISO3``.
    timeout : float
        HTTP request timeout per country, in seconds.

    Returns
    -------
    pd.DataFrame with columns ``[code, qdate, neer_q]``, one row per
    (ISO-3 code, quarter-start Timestamp).  Returns an empty DataFrame
    with those columns if every fetch fails.
    """
    if codes is None:
        codes = list(_BIS_TO_ISO3.keys())

    frames = []
    for cc in codes:
        df = fetch_bis_neer(cc, timeout=timeout)
        if not df.empty:
            frames.append(df)

    if not frames:
        return _EMPTY.copy()

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["code", "qdate"]).reset_index(drop=True)
    return panel


__all__ = ["fetch_bis_neer", "fetch_bis_neer_panel", "_BIS_TO_ISO3"]
