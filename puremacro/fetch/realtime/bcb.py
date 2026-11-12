"""Banco Central do Brasil (BCB) SGS API real-time connector.

Retrieves Brazilian series from the BCB SGS open REST service (no
token required)::

    https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?formato=json

The JSON body is a list of ``{"data": "dd/mm/yyyy", "valor": "1.23"}``
objects (dot decimal). Catalogued series: quarterly real GDP index
22099, IPCA monthly % change 433, Selic target 432 (daily), IBC-Br
activity index 24363 (monthly) — see
:data:`puremacro.fetch.realtime.catalog.BCB_SERIES`.

SGS overwrites series in place and carries no vintage field, so a
*vintage* here is a **snapshot date**: every fetch is stored in the
local SQLite ``realtime_vintages`` table stamped with the day it was
taken, and later calls return every stored snapshot as one vintage
each. Revision history therefore starts with the first local snapshot.
"""
from __future__ import annotations

import urllib.request

import pandas as pd

from ._base import (
    VINTAGE_COLUMNS,
    VintagePanel,
    normalize_vintage_frame,
    register_provider,
)
from ._snapshot import (
    empty_snapshot,
    fetch_snapshot_vintages,
    finish_snapshot,
    load_json,
    warn_skipped,
)
from .canary import SchemaCanary
from .catalog import BCB_SERIES, register_catalog

BCB_SGS_URL = (
    "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?formato=json"
)

_UA = "puremacro (real-time vintage reader)"
_DAILY_BCB_IDS = frozenset(
    spec.series_id for spec in BCB_SERIES.values() if spec.freq == "D"
)


def _format_bcb_date_param(value: str | pd.Timestamp) -> str:
    """Format a date bound as ``dd/mm/yyyy`` for BCB SGS query parameters."""
    if isinstance(value, str):
        text = value.strip()
        parts = text.split("/")
        if len(parts) == 3 and len(parts[0]) == 2 and len(parts[1]) == 2 and len(parts[2]) == 4:
            return text
    return pd.Timestamp(value).strftime("%d/%m/%Y")


def _build_bcb_url(
    series_id: str,
    *,
    vintage_date: str | pd.Timestamp | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> str:
    """Construct the BCB SGS request URL, applying a <=10-year window for daily series."""
    url = BCB_SGS_URL.format(series_id=series_id)
    sid = str(series_id).strip()
    needs_window = (
        start_date is not None
        or end_date is not None
        or sid in _DAILY_BCB_IDS
    )
    if not needs_window:
        return url
    if end_date is not None:
        end_str = _format_bcb_date_param(end_date)
        end_ts = pd.to_datetime(end_str, format="%d/%m/%Y")
    elif vintage_date is not None:
        end_ts = pd.Timestamp(vintage_date).normalize()
        end_str = end_ts.strftime("%d/%m/%Y")
    else:
        end_ts = pd.Timestamp.today().normalize()
        end_str = end_ts.strftime("%d/%m/%Y")
    if start_date is not None:
        start_str = _format_bcb_date_param(start_date)
    else:
        start_ts = end_ts - pd.DateOffset(years=10) + pd.Timedelta(days=1)
        start_str = start_ts.strftime("%d/%m/%Y")
    return f"{url}&dataInicial={start_str}&dataFinal={end_str}"


def _parse_bcb_date(text: str) -> pd.Timestamp | None:
    """``dd/mm/yyyy`` as documented, with a permissive fallback for drift."""
    try:
        return pd.to_datetime(text, format="%d/%m/%Y")
    except (ValueError, ArithmeticError, Exception):
        pass
    try:
        return pd.to_datetime(text)
    except (ValueError, ArithmeticError, Exception):
        return None


def parse_bcb_json(
    raw: bytes | str | list | dict,
    *,
    series_id: str = "",
    vintage_date: str | pd.Timestamp | None = None,
    on_drift: str = "raise",
) -> pd.DataFrame:
    """Parse a BCB SGS JSON response into a tidy ``[date, vintage, value]`` frame.

    Parameters
    ----------
    raw : bytes | str | list | dict
        The JSON response from the BCB SGS API.
    series_id : str
        The SGS series identifier (for messages only).
    vintage_date : str | pd.Timestamp | None
        Snapshot date to stamp on every row. Defaults to today.
    on_drift : {"raise", "warn", "ignore"}
        Schema-canary policy; see :mod:`puremacro.fetch.realtime.canary`.

    Returns
    -------
    pd.DataFrame
        Columns ``["date", "vintage", "value"]``. Observations whose
        date or value cannot be read are dropped with a warning
        counting them; empty values are dropped silently.
    """
    data = load_json(raw)
    if data is None or (isinstance(data, (list, dict)) and len(data) == 0):
        return empty_snapshot()

    SchemaCanary.check("bcb", data, on_drift=on_drift)

    items = data if isinstance(data, list) else []
    records: list[tuple[pd.Timestamp, float]] = []
    skipped = 0
    for item in items:
        if not isinstance(item, dict):
            skipped += 1
            continue
        d_str = str(item.get("data") or "").strip()
        v_str = str(item.get("valor") or "").strip()
        if not d_str or not v_str:
            continue
        try:
            val = float(v_str.replace(",", "."))
        except ValueError:
            skipped += 1
            continue
        d = _parse_bcb_date(d_str)
        if d is None:
            skipped += 1
            continue
        records.append((d, val))
    warn_skipped("parse_bcb_json", skipped, len(items))
    return finish_snapshot(records, vintage_date)


def fetch_bcb_vintages(
    series_id: str,
    *,
    vintage_date: str | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    timeout: float = 60.0,
    use_cache: bool = True,
    history: bool = True,
    on_drift: str = "raise",
) -> pd.DataFrame:
    """Fetch one SGS series and return its locally stored snapshot vintages.

    Parameters
    ----------
    series_id : str
        BCB SGS series ID (e.g. ``'22099'`` for the real GDP index,
        ``'433'`` for IPCA, ``'432'`` for the Selic target).
    vintage_date : str | None
        Snapshot date to stamp on *this* fetch. Defaults to today. It is
        a capture date, not a publication date.
    start_date, end_date : str | pd.Timestamp | None
        Optional query window bounds (formatted as ``dd/mm/yyyy`` in the
        SGS query string). Daily SGS series (such as ``'432'``, Selic)
        require a query window of at most 10 years; when omitted on a
        catalogued daily series, ``fetch_bcb_vintages`` defaults to the
        10-year window ending at ``vintage_date`` (or today).
    timeout : float
        HTTP request timeout in seconds.
    use_cache : bool
        ``True`` stores the snapshot in the SQLite ``realtime_vintages``
        table and falls back to stored snapshots when the fetch fails.
        ``False`` neither reads nor writes the cache: the live snapshot
        is returned, and a failure is raised.
    history : bool
        ``True`` (default) returns every snapshot stored locally for
        this series — one vintage per snapshot date, today's included —
        which is what a revision test needs. ``False`` returns only the
        snapshot just fetched.
    on_drift : {"raise", "warn", "ignore"}
        Schema-canary policy. With ``"raise"`` a drifted payload falls
        back to the cached snapshots (with a ``SchemaDriftWarning``)
        when any exist.

    Returns
    -------
    pd.DataFrame
        Columns ``["date", "vintage", "value"]``.
    """
    url = _build_bcb_url(
        series_id,
        vintage_date=vintage_date,
        start_date=start_date,
        end_date=end_date,
    )
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    return fetch_snapshot_vintages(
        provider="bcb", country="BRA", series_id=series_id, request=req,
        parser=parse_bcb_json, timeout=timeout, use_cache=use_cache,
        history=history, on_drift=on_drift, vintage_date=vintage_date,
    )


def fetch_bcb_panel(
    countries,
    variables,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    timeout: float = 60.0,
    use_cache: bool = True,
    history: bool = True,
    on_drift: str = "raise",
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
                    start_date=start_date,
                    end_date=end_date,
                    timeout=timeout,
                    use_cache=use_cache,
                    history=history,
                    on_drift=on_drift,
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
