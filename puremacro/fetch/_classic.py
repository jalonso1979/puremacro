"""Pyodide-friendly data fetcher for FRED / ALFRED via the public CSV
endpoint.

These endpoints don't require an API key and work over plain HTTP from
``urllib.request`` — which Pyodide proxies through the browser fetch.
For other sources, the same pattern (``urllib.request.urlopen`` plus
``pd.read_csv``) extends straightforwardly.

If the network is unavailable, callers should be prepared for an
``OSError`` / ``urllib.error.URLError`` and fall back to bundled data.
The example ``puremacro.examples.fetch_fred_demo`` shows the pattern.

Note on Pyodide
---------------
``urllib.request`` works inside Pyodide via the JS-fetch proxy. SSL
fallback for older endpoints with stale CA bundles is handled by the
underlying :func:`puremacro._http.safe_get_bytes` (see its docstring
and ``puremacro/narrative/sources/RETRY_POLICY.md`` for the full
contract).
"""
from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..vintages import AlfredVintageStore


_FREDGRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
_ALFRED   = ("https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={}")


def _safe_urlopen(url: str, timeout: float = 30.0) -> bytes:
    """Fetch ``url`` and return raw bytes.

    Uses a non-Mozilla UA because FRED CSV's WAF rate-limits Mozilla-like
    UAs (silent timeout). Honest UA gets 0.1s response.
    """
    from .._http import safe_get_bytes
    return safe_get_bytes(
        url,
        timeout=timeout,
        user_agent="puremacro/0.9 (FRED CSV reader; +https://github.com/jalonso1979/uncertainty_examples)",
    )


def fetch_fred(
    series_id: str,
    *,
    timeout: float = 30.0,
) -> pd.Series:
    """Download a FRED series as a date-indexed pd.Series.

    Parameters
    ----------
    series_id : str — e.g. ``"GDPC1"`` for real GDP, ``"UNRATE"``,
                       ``"CPIAUCSL"``, etc.

    Returns
    -------
    pd.Series indexed by date, dtype float (with NaN where ``"."``).
    """
    url = _FREDGRAPH.format(series_id)
    raw = _safe_urlopen(url, timeout=timeout)
    df = pd.read_csv(io.BytesIO(raw))
    # FRED's column is the series id; ALFRED variant is the same.
    date_col = df.columns[0]
    val_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    s = df.set_index(date_col)[val_col]
    s.name = series_id
    return s


def _fetch_fred_alfred_raw_api(series_id: str, *, timeout: float = 60.0) -> pd.DataFrame:
    """Live ALFRED CSV fetch. Returns long DataFrame ['date', 'vintage', 'value']."""
    url = _ALFRED.format(series_id)
    raw = _safe_urlopen(url, timeout=timeout)
    df = pd.read_csv(io.BytesIO(raw))
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    long = df.melt(id_vars="date", var_name="vintage_col", value_name="value")
    # vintage_col looks like "GDPC1_2013-04-26"; extract the trailing date.
    long["vintage"] = pd.to_datetime(
        long["vintage_col"].str.split("_").str[-1], errors="coerce"
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["vintage", "value"])
    return long[["date", "vintage", "value"]].reset_index(drop=True)


def fetch_fred_alfred(
    series_id: str,
    *,
    timeout: float = 60.0,
    store: "AlfredVintageStore | None" = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download a FRED-ALFRED real-time vintage panel as long DataFrame.

    Returns columns ``[date, vintage, value]``. Backwards-compat with
    0.65.0: calling without ``store=`` is identical to the pre-0.66.0
    behaviour.

    The ALFRED CSV has columns:
        observation_date, <SERIES_ID>_YYYY-MM-DD (one column per vintage).
    We melt that to long form.

    Parameters
    ----------
    store : optional AlfredVintageStore (0.66.0+). When provided:
        - if ``refresh`` is False and the store already has data for
          ``series_id``, read from the store (no API call);
        - otherwise call the live API and write the rows back to the
          store via ``store.put_many``.
    refresh : if True, ignore the store on read and force a fresh API
        call. Rows still get written back. Useful when ALFRED has
        published new vintages since the store was filled.
    """
    if store is not None and not refresh and store.has_series(series_id):
        df_store = store.get(series_id)
        return df_store.rename(columns={
            "observation_date": "date",
            "vintage_date":     "vintage",
        })[["date", "vintage", "value"]].reset_index(drop=True)

    df = _fetch_fred_alfred_raw_api(series_id, timeout=timeout)

    if store is not None and not df.empty:
        store_rows = df.assign(series_id=series_id).rename(columns={
            "date":    "observation_date",
            "vintage": "vintage_date",
        })[["series_id", "observation_date", "vintage_date", "value"]]
        store.put_many(store_rows)

    return df


__all__ = ["fetch_fred", "fetch_fred_alfred"]
