"""Banco de México (Banxico) SIE API real-time connector.

Retrieves Mexican series from the Banxico SIE REST API (token in the
``Bmx-Token`` header)::

    https://www.banxico.org.mx/SieAPIRest/service/v1/series/{series_id}/datos

The body is ``{"bmx": {"series": [{"idSerie": .., "titulo": ..,
"datos": [{"fecha": "dd/mm/yyyy", "dato": "1,234.56"}]}]}}`` with
``"N/E"`` where an observation is missing. Catalogued series: policy
rate SF61745 (daily), INPC SP1 (monthly), IGAE activity SR17631
(monthly) — see :data:`puremacro.fetch.realtime.catalog.BANXICO_SERIES`.

SIE overwrites series in place, so a *vintage* here is a **snapshot
date**: every fetch is stored in the local SQLite ``realtime_vintages``
table stamped with the day it was taken, and later calls return every
stored snapshot as one vintage each. Revision history therefore starts
with the first local snapshot.
"""
from __future__ import annotations

import urllib.request

import pandas as pd

from ... import credentials
from ._base import (
    VINTAGE_COLUMNS,
    VintagePanel,
    normalize_vintage_frame,
    register_provider,
)
from ._snapshot import (
    cached_snapshots,
    empty_snapshot,
    fetch_snapshot_vintages,
    finish_snapshot,
    load_json,
    warn_skipped,
)
from .canary import SchemaCanary
from .catalog import BANXICO_SERIES, register_catalog

#: Why Mexico quarterly GDP had no native vintage provider originally.
MEXICO_VINTAGE_NOTE = (
    "Neither Banxico SIE nor INEGI BIE retains previously published "
    "editions of quarterly GDP — both overwrite in place and their "
    "payloads carry no vintage field. Use provider 'oecd_stes', which "
    "archives INEGI's series with 329 monthly editions from 1999-02."
)

#: Same determination for Spain's national statistics office.
SPAIN_VINTAGE_NOTE = (
    "INE / Banco de España publish no machine-readable vintage archive "
    "for quarterly GDP. Use provider 'oecd_stes' (331 editions from "
    "1999-02)."
)

BANXICO_SIE_BASE = "https://www.banxico.org.mx/SieAPIRest/service/v1/"
INEGI_BIE_BASE = (
    "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/"
)
BANXICO_SERIES_URL = (
    "https://www.banxico.org.mx/SieAPIRest/service/v1/series/{series_id}/datos"
)

_UA = "puremacro (real-time vintage reader)"

#: ``dato`` values SIE uses for a missing observation.
_MISSING_MARKERS = frozenset({"N/E", "NaN", "null", ""})


def parse_banxico_json(
    raw: bytes | str | dict,
    *,
    series_id: str = "",
    vintage_date: str | pd.Timestamp | None = None,
    on_drift: str = "raise",
) -> pd.DataFrame:
    """Parse a Banxico SIE JSON response into a tidy ``[date, vintage, value]`` frame.

    Parameters
    ----------
    raw : bytes | str | dict
        The JSON response from Banxico's SIE API.
    series_id : str
        The series identifier (for messages only).
    vintage_date : str | pd.Timestamp | None
        Snapshot date to stamp on every row. Defaults to today.
    on_drift : {"raise", "warn", "ignore"}
        Schema-canary policy; see :mod:`puremacro.fetch.realtime.canary`.

    Returns
    -------
    pd.DataFrame
        Columns ``["date", "vintage", "value"]``. ``"N/E"`` observations
        are dropped silently; rows whose date or value cannot be read
        are dropped with a warning counting them.
    """
    data = load_json(raw)
    if not data:
        return empty_snapshot()

    SchemaCanary.check("banxico", data, on_drift=on_drift)

    bmx = data.get("bmx") if isinstance(data, dict) else None
    series_list = bmx.get("series") if isinstance(bmx, dict) else None
    if (not isinstance(series_list, list) or not series_list
            or not isinstance(series_list[0], dict)):
        return empty_snapshot()

    datos = series_list[0].get("datos")
    if not isinstance(datos, list):
        return empty_snapshot()
    records: list[tuple[pd.Timestamp, float]] = []
    skipped = 0
    for item in datos:
        if not isinstance(item, dict):
            skipped += 1
            continue
        f_str = str(item.get("fecha") or "").strip()
        v_str = str(item.get("dato") or "").strip()
        if not f_str or v_str in _MISSING_MARKERS:
            continue
        try:
            val = float(v_str.replace(",", ""))
        except ValueError:
            skipped += 1
            continue
        # SIE sends DD/MM/YYYY; anything else is drift the canary reports.
        try:
            if "/" in f_str:
                d = pd.to_datetime(f_str, format="%d/%m/%Y")
            else:
                d = pd.to_datetime(f_str)
        except Exception:
            skipped += 1
            continue
        records.append((d, val))
    warn_skipped("parse_banxico_json", skipped, len(datos))
    return finish_snapshot(records, vintage_date)


def fetch_banxico_vintages(
    series_id: str,
    *,
    token: str | None = None,
    vintage_date: str | None = None,
    timeout: float = 60.0,
    use_cache: bool = True,
    history: bool = True,
    on_drift: str = "raise",
) -> pd.DataFrame:
    """Fetch one SIE series and return its locally stored snapshot vintages.

    Parameters
    ----------
    series_id : str
        Banxico series ID (e.g. ``'SF61745'`` for the policy rate,
        ``'SP1'`` for the INPC).
    token : str | None
        Banxico API token. Resolved from :mod:`puremacro.credentials`
        (``BANXICO_API_KEY`` / ``BMX_TOKEN`` / ``[banxico].api_key``)
        if omitted. Without a token the stored snapshots are returned
        when there are any (and ``use_cache`` is True); otherwise
        ``MissingCredentialError``.
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
        this series — one vintage per snapshot date, today's included.
        ``False`` returns only the snapshot just fetched.
    on_drift : {"raise", "warn", "ignore"}
        Schema-canary policy. With ``"raise"`` a drifted payload falls
        back to the cached snapshots (with a ``SchemaDriftWarning``)
        when any exist.

    Returns
    -------
    pd.DataFrame
        Columns ``["date", "vintage", "value"]``.
    """
    tok = token or credentials.get("banxico")
    if not tok:
        if use_cache:
            cached = cached_snapshots("banxico", "MEX", series_id)
            if not cached.empty:
                return cached
        credentials.require("banxico")

    url = BANXICO_SERIES_URL.format(series_id=series_id)
    req = urllib.request.Request(
        url, headers={"User-Agent": _UA, "Bmx-Token": tok},
    )
    return fetch_snapshot_vintages(
        provider="banxico", country="MEX", series_id=series_id, request=req,
        parser=parse_banxico_json, timeout=timeout, use_cache=use_cache,
        history=history, on_drift=on_drift, vintage_date=vintage_date,
    )


def fetch_banxico_panel(
    countries,
    variables,
    *,
    token: str | None = None,
    timeout: float = 60.0,
    use_cache: bool = True,
    history: bool = True,
    on_drift: str = "raise",
    **_ignored,
) -> VintagePanel:
    """Registry entry point for Banxico."""
    frames = []
    failed = {}
    for country in countries:
        if str(country).upper() != "MEX":
            continue
        for variable in variables:
            spec = BANXICO_SERIES.get(variable)
            if spec is None:
                continue
            series_id = spec.series_id
            try:
                long = fetch_banxico_vintages(
                    series_id,
                    token=token,
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
                    country="MEX",
                    variable=variable,
                    provider="banxico",
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
        metadata={"provider": "banxico", "failed": failed},
    )


def _register() -> None:
    """Register Banxico catalog and provider."""
    register_catalog(
        "banxico",
        {"MEX": BANXICO_SERIES},
        gaps={"ESP": SPAIN_VINTAGE_NOTE},
    )
    register_provider("banxico", fetch_banxico_panel, ["MEX"])


__all__ = [
    "MEXICO_VINTAGE_NOTE",
    "SPAIN_VINTAGE_NOTE",
    "BANXICO_SIE_BASE",
    "INEGI_BIE_BASE",
    "BANXICO_SERIES_URL",
    "parse_banxico_json",
    "fetch_banxico_vintages",
    "fetch_banxico_panel",
]
