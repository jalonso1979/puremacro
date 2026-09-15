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


def _parse_bcb_date(text: str) -> pd.Timestamp | None:
    """``dd/mm/yyyy`` as documented, with a permissive fallback for drift."""
    try:
        return pd.to_datetime(text, format="%d/%m/%Y")
    except Exception:
        pass
    try:
        return pd.to_datetime(text)
    except Exception:
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
    if data is None or len(data) == 0:
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
    url = BCB_SGS_URL.format(series_id=series_id)
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
