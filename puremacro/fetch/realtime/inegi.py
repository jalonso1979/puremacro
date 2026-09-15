"""INEGI (Instituto Nacional de Estadística y Geografía) real-time connector.

Retrieves Mexican national statistics (quarterly real GDP 735848, CPI inflation 628197,
and IGAE economic activity 736184) from the INEGI BIE / Banco de Indicadores API:
    https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/BISEventOp1/{series_id}/es/0700/true/IP/2.0/{token}?type=json

Captures snapshots into the persistent SQLite `realtime_vintages` cache table,
enabling offline reproducibility and historical vintage analysis.
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

from ... import credentials
from ..._cache_db import query_realtime_vintages, record_connector_event, store_realtime_vintages
from ._base import (
    VINTAGE_COLUMNS,
    VintagePanel,
    normalize_vintage_frame,
    register_provider,
)
from .canary import SchemaCanary, SchemaDriftError
from .catalog import INEGI_SERIES, SeriesSpec, register_catalog

INEGI_SERIES_URL = (
    "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/"
    "BISEventOp1/{series_id}/es/0700/true/IP/2.0/{token}?type=json"
)

_UA = "puremacro (real-time vintage reader)"


def _parse_inegi_period(tp: str, is_quarterly: bool = False) -> pd.Timestamp | None:
    """Convert INEGI TIME_PERIOD string to Timestamp."""
    token = str(tp).strip()
    if not token:
        return None

    # Handle "YYYY/MM" or "YYYY/QQ"
    if "/" in token:
        parts = token.split("/")
        if len(parts) == 2:
            try:
                year = int(parts[0])
                num = int(parts[1])
                if is_quarterly:
                    # Quarter 1 -> Jan 1, 2 -> Apr 1, 3 -> Jul 1, 4 -> Oct 1
                    q_month = {1: 1, 2: 4, 3: 7, 4: 10}.get(num, num)
                    return pd.Timestamp(year, q_month, 1)
                else:
                    # Monthly
                    return pd.Timestamp(year, num, 1)
            except (ValueError, TypeError):
                pass

    # Handle standard formats: YYYY-MM-DD, YYYY-MM, YYYY-Q#
    try:
        if "Q" in token or "q" in token:
            return pd.Period(token.upper(), freq="Q").to_timestamp()
        return pd.to_datetime(token)
    except Exception:
        return None


def parse_inegi_json(
    raw: bytes | str | dict,
    *,
    series_id: str = "",
    vintage_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Parse INEGI BIE API JSON response into tidy [date, vintage, value] DataFrame.

    Parameters
    ----------
    raw : bytes | str | dict
        The JSON response from INEGI API.
    series_id : str
        The indicator identifier.
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

    ok, reason = SchemaCanary.validate_inegi(data)
    if not ok:
        raise SchemaDriftError(f"INEGI schema drift: {reason}")

    series_list = data.get("Series", [])
    if not series_list:
        return pd.DataFrame(columns=["date", "vintage", "value"])

    first_series = series_list[0]
    freq_desc = str(first_series.get("FREQ", "")).lower()
    is_quarterly = "trimestral" in freq_desc or series_id == "735848"

    obs_list = first_series.get("OBSERVATIONS", [])
    if not obs_list:
        return pd.DataFrame(columns=["date", "vintage", "value"])

    # If freq not explicitly stated, check if max period number is <= 4
    if not is_quarterly:
        p_nums = []
        for obs in obs_list:
            tp = str(obs.get("TIME_PERIOD", ""))
            if "/" in tp:
                parts = tp.split("/")
                if len(parts) == 2 and parts[1].isdigit():
                    p_nums.append(int(parts[1]))
        if p_nums and max(p_nums) <= 4 and series_id == "735848":
            is_quarterly = True

    records = []
    for obs in obs_list:
        tp = obs.get("TIME_PERIOD")
        v_str = obs.get("OBS_VALUE")
        if tp is None or v_str is None:
            continue
        try:
            val = float(str(v_str).replace(",", ""))
        except (ValueError, TypeError):
            continue

        d = _parse_inegi_period(tp, is_quarterly=is_quarterly)
        if d is None:
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


def fetch_inegi_vintages(
    series_id: str,
    *,
    token: str | None = None,
    vintage_date: str | None = None,
    timeout: float = 60.0,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch time series from INEGI BIE API or retrieve cached vintages.

    Parameters
    ----------
    series_id : str
        INEGI series ID (e.g. '735848' for GDP, '628197' for CPI).
    token : str | None
        INEGI API token. Resolved from `puremacro.credentials` if omitted.
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
    tok = token or credentials.get("inegi")
    if not tok:
        cached = query_realtime_vintages("inegi", "MEX", series_id)
        if not cached.empty:
            return cached[["date", "vintage", "value"]]
        credentials.require("inegi")

    url = INEGI_SERIES_URL.format(series_id=series_id, token=tok)
    headers = {"User-Agent": _UA}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
        df = parse_inegi_json(body, series_id=series_id, vintage_date=vintage_date)
        if df.empty and use_cache:
            cached = query_realtime_vintages("inegi", "MEX", series_id)
            if not cached.empty:
                warnings.warn(
                    f"fetch_inegi_vintages received empty observations; falling back to cached vintages.",
                    UserWarning,
                    stacklevel=2,
                )
                record_connector_event("inegi", "fallback", "sqlite_cache")
                return cached[["date", "vintage", "value"]]
        if use_cache and not df.empty:
            store_df = df.copy()
            store_df["provider"] = "inegi"
            store_df["country"] = "MEX"
            store_df["series_id"] = series_id
            store_realtime_vintages(store_df)
            record_connector_event("inegi", "success", "none")
        return df
    except Exception as exc:
        cached = query_realtime_vintages("inegi", "MEX", series_id)
        if not cached.empty:
            warnings.warn(
                f"fetch_inegi_vintages failed ({exc}); falling back to cached vintages.",
                UserWarning,
                stacklevel=2,
            )
            record_connector_event("inegi", "fallback", "sqlite_cache")
            return cached[["date", "vintage", "value"]]
        raise


def fetch_inegi_panel(
    countries,
    variables,
    *,
    token: str | None = None,
    timeout: float = 60.0,
    use_cache: bool = True,
    **_ignored,
) -> VintagePanel:
    """Registry entry point for INEGI."""
    frames = []
    failed = {}
    for country in countries:
        if str(country).upper() != "MEX":
            continue
        for variable in variables:
            spec = INEGI_SERIES.get(variable)
            if spec is None:
                continue
            series_id = spec.series_id
            try:
                long = fetch_inegi_vintages(
                    series_id,
                    token=token,
                    timeout=timeout,
                    use_cache=use_cache,
                )
            except Exception as exc:
                failed[f"{country}:{variable}"] = f"{type(exc).__name__}: {exc}"
                continue
            frames.append(
                normalize_vintage_frame(
                    long,
                    country="MEX",
                    variable=variable,
                    provider="inegi",
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
        metadata={"provider": "inegi", "failed": failed},
    )


def _register() -> None:
    """Register INEGI catalog and provider."""
    register_catalog("inegi", {"MEX": INEGI_SERIES})
    register_provider("inegi", fetch_inegi_panel, ["MEX"])


__all__ = [
    "INEGI_SERIES_URL",
    "parse_inegi_json",
    "fetch_inegi_vintages",
    "fetch_inegi_panel",
]
