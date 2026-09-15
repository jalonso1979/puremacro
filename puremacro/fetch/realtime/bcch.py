"""Banco Central de Chile (BCCh) SIETE API real-time connector.

Retrieves Chilean macroeconomic time series (quarterly real GDP F032.PIB.VOL.Z.Z.18.Z.Z.0.Q,
CPI inflation F073.IPC.VAR.Z.Z.C.M, TPM policy rate F022.TPM.TPO.D001.NO.Z.D,
IMACEC economic activity F032.IMC.IND.Z.Z.EP18.Z.Z.0.M) from the BCCh SIETE REST API:
    https://si3.bcentral.cl/SieteRestWS/SieteRestWS.asmx/GetSeries

Captures snapshots into the persistent SQLite `realtime_vintages` cache table,
enabling offline reproducibility and historical vintage tracking.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import urllib.error
import urllib.request
import warnings
from typing import Any

import pandas as pd

from ... import credentials
from ..._cache_db import query_realtime_vintages, record_connector_event, store_realtime_vintages
from ._base import (
    VINTAGE_COLUMNS,
    VintagePanel,
    normalize_vintage_frame,
    register_provider,
)
from .canary import SchemaCanary, SchemaDriftError
from .catalog import BCCH_SERIES, SeriesSpec, register_catalog

BCCH_SIETE_URL = (
    "https://si3.bcentral.cl/SieteRestWS/SieteRestWS.asmx/GetSeries?"
    "user={user}&password={password}&timeseries={series_id}&function=GetSeries"
)

_UA = "puremacro (real-time vintage reader)"


def parse_bcch_json(
    raw: bytes | str | dict,
    *,
    series_id: str = "",
    vintage_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Parse BCCh SIETE API JSON response into tidy [date, vintage, value] DataFrame.

    Parameters
    ----------
    raw : bytes | str | dict
        The JSON response from BCCh SIETE API.
    series_id : str
        The SIETE series identifier.
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
    elif isinstance(raw, dict):
        data = raw
    else:
        return pd.DataFrame(columns=["date", "vintage", "value"])

    if not data:
        return pd.DataFrame(columns=["date", "vintage", "value"])

    ok, reason = SchemaCanary.validate_bcch(data)
    if not ok:
        raise SchemaDriftError(f"BCCh schema drift: {reason}")

    series_data = data.get("Series", {})
    if isinstance(series_data, dict):
        obs_list = series_data.get("obs", [])
    elif "obs" in data:
        obs_list = data["obs"]
    else:
        obs_list = []

    if not obs_list:
        return pd.DataFrame(columns=["date", "vintage", "value"])

    records = []
    for obs in obs_list:
        d_str = obs.get("indexDateString", "").strip()
        v_str = obs.get("value", "")
        if not d_str or v_str is None:
            continue
        try:
            val = float(str(v_str).replace(",", "."))
        except (ValueError, TypeError):
            continue

        # Parse date: DD-MM-YYYY or YYYY-MM-DD
        try:
            if "-" in d_str and len(d_str.split("-")[0]) == 2:
                d = pd.to_datetime(d_str, format="%d-%m-%Y")
            else:
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


def fetch_bcch_vintages(
    series_id: str,
    *,
    user: str | None = None,
    password: str | None = None,
    vintage_date: str | None = None,
    timeout: float = 60.0,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch time series from BCCh SIETE API or retrieve cached vintages.

    Parameters
    ----------
    series_id : str
        BCCh series ID (e.g. 'F032.PIB.VOL.Z.Z.18.Z.Z.0.Q' for GDP).
    user : str | None
        SIETE user email. Resolved from env/credentials if omitted.
    password : str | None
        SIETE user password. Resolved from env/credentials if omitted.
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
    u = user or os.environ.get("BCCH_API_USER") or credentials.get("bcch")
    p = password or os.environ.get("BCCH_API_PASS") or ""

    if not u:
        cached = query_realtime_vintages("bcch", "CHL", series_id)
        if not cached.empty:
            return cached[["date", "vintage", "value"]]
        credentials.require("bcch")

    url = BCCH_SIETE_URL.format(user=u, password=p, series_id=series_id)
    headers = {"User-Agent": _UA}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
        df = parse_bcch_json(body, series_id=series_id, vintage_date=vintage_date)
        if df.empty and use_cache:
            cached = query_realtime_vintages("bcch", "CHL", series_id)
            if not cached.empty:
                warnings.warn(
                    f"fetch_bcch_vintages received empty observations; falling back to cached vintages.",
                    UserWarning,
                    stacklevel=2,
                )
                record_connector_event("bcch", "fallback", "sqlite_cache")
                return cached[["date", "vintage", "value"]]
        if use_cache and not df.empty:
            store_df = df.copy()
            store_df["provider"] = "bcch"
            store_df["country"] = "CHL"
            store_df["series_id"] = series_id
            store_realtime_vintages(store_df)
            record_connector_event("bcch", "success", "none")
        return df
    except Exception as exc:
        cached = query_realtime_vintages("bcch", "CHL", series_id)
        if not cached.empty:
            warnings.warn(
                f"fetch_bcch_vintages failed ({exc}); falling back to cached vintages.",
                UserWarning,
                stacklevel=2,
            )
            record_connector_event("bcch", "fallback", "sqlite_cache")
            return cached[["date", "vintage", "value"]]
        raise


def fetch_bcch_panel(
    countries,
    variables,
    *,
    user: str | None = None,
    password: str | None = None,
    timeout: float = 60.0,
    use_cache: bool = True,
    **_ignored,
) -> VintagePanel:
    """Registry entry point for Banco Central de Chile."""
    frames = []
    failed = {}
    for country in countries:
        if str(country).upper() != "CHL":
            continue
        for variable in variables:
            spec = BCCH_SERIES.get(variable)
            if spec is None:
                continue
            series_id = spec.series_id
            try:
                long = fetch_bcch_vintages(
                    series_id,
                    user=user,
                    password=password,
                    timeout=timeout,
                    use_cache=use_cache,
                )
            except Exception as exc:
                failed[f"{country}:{variable}"] = f"{type(exc).__name__}: {exc}"
                continue
            frames.append(
                normalize_vintage_frame(
                    long,
                    country="CHL",
                    variable=variable,
                    provider="bcch",
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
        metadata={"provider": "bcch", "failed": failed},
    )


def _register() -> None:
    """Register BCCh catalog and provider."""
    register_catalog("bcch", {"CHL": BCCH_SERIES})
    register_provider("bcch", fetch_bcch_panel, ["CHL"])


__all__ = [
    "BCCH_SIETE_URL",
    "parse_bcch_json",
    "fetch_bcch_vintages",
    "fetch_bcch_panel",
]
