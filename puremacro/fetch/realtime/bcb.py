"""Banco Central do Brasil (BCB) SGS API real-time connector.

Retrieves Brazilian macroeconomic time series (quarterly GDP 4380, IPCA inflation 433,
Selic policy rate 432, IBC-Br economic activity 24363) from the BCB SGS API:
    https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?formato=json

Public open REST service (no authentication token required). Captures snapshots into
the persistent SQLite `realtime_vintages` cache table.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import urllib.error
import urllib.request
import warnings
from typing import Any

import pandas as pd

from ..._cache_db import query_realtime_vintages, record_connector_event, store_realtime_vintages
from ._base import (
    VINTAGE_COLUMNS,
    VintagePanel,
    normalize_vintage_frame,
    register_provider,
)
from .canary import SchemaCanary, SchemaDriftError
from .catalog import BCB_SERIES, SeriesSpec, register_catalog

BCB_SGS_URL = (
    "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?formato=json"
)

_UA = "puremacro (real-time vintage reader)"


def parse_bcb_json(
    raw: bytes | str | list | dict,
    *,
    series_id: str = "",
    vintage_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Parse BCB SGS API JSON response into tidy [date, vintage, value] DataFrame.

    Parameters
    ----------
    raw : bytes | str | list | dict
        The JSON response from BCB SGS API.
    series_id : str
        The SGS series identifier.
    vintage_date : str | pd.Timestamp | None
        Vintage date to stamp. Defaults to current date.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["date", "vintage", "value"].
    """
    if isinstance(raw, (bytes, bytearray)):
        text = raw.decode("utf-8-sig", errors="ignore")
        if not text.strip():
            return pd.DataFrame(columns=["date", "vintage", "value"])
        data = json.loads(text)
    elif isinstance(raw, str):
        if not raw.strip():
            return pd.DataFrame(columns=["date", "vintage", "value"])
        data = json.loads(raw)
    elif isinstance(raw, (list, dict)):
        data = raw
    else:
        return pd.DataFrame(columns=["date", "vintage", "value"])

    if data is None or (isinstance(data, (list, dict)) and len(data) == 0):
        return pd.DataFrame(columns=["date", "vintage", "value"])

    ok, reason = SchemaCanary.validate_bcb(data)
    if not ok:
        raise SchemaDriftError(f"BCB schema drift: {reason}")

    records = []
    for item in data:
        if not isinstance(item, dict):
            continue
        d_str = item.get("data", "").strip()
        v_str = item.get("valor", "").strip()
        if not d_str or not v_str:
            continue
        try:
            val = float(v_str.replace(",", "."))
        except (ValueError, TypeError):
            continue

        try:
            d = pd.to_datetime(d_str, format="%d/%m/%Y")
        except Exception:
            try:
                d = pd.to_datetime(d_str)
            except Exception:
                continue
        records.append((d, val))

    if not records:
        return pd.DataFrame(columns=["date", "vintage", "value"])

    if vintage_date is not None:
        v_stamp = pd.to_datetime(vintage_date)
    else:
        v_stamp = pd.Timestamp.now(tz=None).normalize()

    rows = [(d, v_stamp, val) for d, val in records]
    df = pd.DataFrame(rows, columns=["date", "vintage", "value"])
    return (
        df.drop_duplicates(subset=["date", "vintage"], keep="last")
        .sort_values(["date", "vintage"])
        .reset_index(drop=True)
    )


def fetch_bcb_vintages(
    series_id: str,
    *,
    vintage_date: str | None = None,
    timeout: float = 60.0,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch time series from BCB SGS API or retrieve cached vintages.

    Parameters
    ----------
    series_id : str
        BCB SGS series ID (e.g. '4380' for GDP, '433' for IPCA, '432' for Selic).
    vintage_date : str | None
        Vintage date stamp to assign.
    timeout : float
        HTTP request timeout in seconds.
    use_cache : bool
        Whether to check and update the SQLite cache.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["date", "vintage", "value"].
    """
    url = BCB_SGS_URL.format(series_id=series_id)
    headers = {"User-Agent": _UA}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
        df = parse_bcb_json(body, series_id=series_id, vintage_date=vintage_date)
        if df.empty and use_cache:
            cached = query_realtime_vintages("bcb", "BRA", series_id)
            if not cached.empty:
                warnings.warn(
                    f"fetch_bcb_vintages received empty observations; falling back to cached vintages.",
                    UserWarning,
                    stacklevel=2,
                )
                record_connector_event("bcb", "fallback", "sqlite_cache")
                return cached[["date", "vintage", "value"]]
        if use_cache and not df.empty:
            store_df = df.copy()
            store_df["provider"] = "bcb"
            store_df["country"] = "BRA"
            store_df["series_id"] = series_id
            store_realtime_vintages(store_df)
            record_connector_event("bcb", "success", "none")
        return df
    except Exception as exc:
        cached = query_realtime_vintages("bcb", "BRA", series_id)
        if not cached.empty:
            warnings.warn(
                f"fetch_bcb_vintages failed ({exc}); falling back to cached vintages.",
                UserWarning,
                stacklevel=2,
            )
            record_connector_event("bcb", "fallback", "sqlite_cache")
            return cached[["date", "vintage", "value"]]
        raise


def fetch_bcb_panel(
    countries,
    variables,
    *,
    timeout: float = 60.0,
    use_cache: bool = True,
    **_ignored,
) -> VintagePanel:
    """Registry entry point for Banco Central do Brasil."""
    frames = []
    failed = {}
    for country in countries:
        if str(country).upper() != "BRA":
            continue
        for variable in variables:
            spec = BCB_SERIES.get(variable)
            if spec is None:
                continue
            series_id = spec.series_id
            try:
                long = fetch_bcb_vintages(
                    series_id,
                    timeout=timeout,
                    use_cache=use_cache,
                )
            except Exception as exc:
                failed[f"{country}:{variable}"] = f"{type(exc).__name__}: {exc}"
                continue
            frames.append(
                normalize_vintage_frame(
                    long,
                    country="BRA",
                    variable=variable,
                    provider="bcb",
                    series_id=series_id,
                    units=spec.units,
                )
            )
    df = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=VINTAGE_COLUMNS)
    )
    return VintagePanel(
        df=df,
        metadata={"provider": "bcb", "failed": failed},
    )


def _register() -> None:
    """Register BCB catalog and provider."""
    register_catalog("bcb", {"BRA": BCB_SERIES})
    register_provider("bcb", fetch_bcb_panel, ["BRA"])


__all__ = [
    "BCB_SGS_URL",
    "parse_bcb_json",
    "fetch_bcb_vintages",
    "fetch_bcb_panel",
]
