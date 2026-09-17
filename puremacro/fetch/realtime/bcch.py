"""Banco Central de Chile (BCCh) SIETE API real-time connector.

Retrieves Chilean series from the Base de Datos Estadísticos SIETE
REST service (user + password in the query string)::

    https://si3.bcentral.cl/SieteRestWS/SieteRestWS.ashx?user=..&pass=..&function=GetSeries&timeseries=..&firstdate=..&lastdate=..

The body is ``{"Codigo": 0, "Descripcion": "", "Series": {"descripEsp":
.., "seriesId": .., "Obs": [{"indexDateString": "dd-MM-yyyy", "value":
"123.4", "statusCode": "OK"}]}}`` with ``value`` ``"NaN"`` where the
observation is missing. Catalogued series (TPM policy rate, IMACEC,
IPC, quarterly GDP) are in
:data:`puremacro.fetch.realtime.catalog.BCCH_SERIES`.

SIETE overwrites series in place, so a *vintage* here is a **snapshot
date**: every fetch is stored in the local SQLite ``realtime_vintages``
table stamped with the day it was taken, and later calls return every
stored snapshot as one vintage each. Revision history therefore starts
with the first local snapshot.

Credentials are a two-part pair resolved by :mod:`puremacro.credentials`:
the user from ``BCCH_API_USER`` (or ``[bcch].user`` in the TOML file)
and the password from ``BCCH_API_PASS`` (or ``[bcch].password``). The
password travels in the query string because the API requires it; this
module scrubs it from the warnings it emits and from the URL urllib
attaches to its errors.
"""
from __future__ import annotations

import math
import urllib.parse
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
from .canary import SchemaCanary, _bcch_observations
from .catalog import BCCH_SERIES, register_catalog

BCCH_SIETE_ENDPOINT = "https://si3.bcentral.cl/SieteRestWS/SieteRestWS.ashx"

#: The documented ``GetSeries`` request. Requests are built with
#: :func:`_build_bcch_url` so every value is URL-encoded; the template
#: records the parameter names.
BCCH_SIETE_URL = (
    BCCH_SIETE_ENDPOINT + "?user={user}&pass={password}&function=GetSeries"
    "&timeseries={series_id}&firstdate={firstdate}&lastdate={lastdate}"
)

_UA = "puremacro (real-time vintage reader)"


def _build_bcch_url(
    series_id: str,
    user: str,
    password: str,
    firstdate: str | None = None,
    lastdate: str | None = None,
) -> str:
    """The ``GetSeries`` URL with every parameter URL-encoded.

    ``firstdate`` / ``lastdate`` are ``yyyy-mm-dd``; the official client
    always sends both and leaves them empty for the whole series.
    """
    query = urllib.parse.urlencode({
        "user": user,
        "pass": password,
        "function": "GetSeries",
        "timeseries": series_id,
        "firstdate": firstdate or "",
        "lastdate": lastdate or "",
    })
    return f"{BCCH_SIETE_ENDPOINT}?{query}"


def _redactor(user: str, password: str):
    """Scrub the password (raw and URL-encoded) and the user from text."""
    secrets = []
    for value in (password, user):
        if value:
            secrets.append(value)
            for encoded in (urllib.parse.quote_plus(value), urllib.parse.quote(value)):
                if encoded != value:
                    secrets.append(encoded)

    def redact(text: str) -> str:
        for secret in secrets:
            text = text.replace(secret, "***")
        return text

    return redact


def parse_bcch_json(
    raw: bytes | str | dict,
    *,
    series_id: str = "",
    vintage_date: str | pd.Timestamp | None = None,
    on_drift: str = "raise",
) -> pd.DataFrame:
    """Parse a BCCh SIETE JSON response into a tidy ``[date, vintage, value]`` frame.

    Parameters
    ----------
    raw : bytes | str | dict
        The JSON response from the BCCh SIETE API.
    series_id : str
        The SIETE series identifier (for messages only).
    vintage_date : str | pd.Timestamp | None
        Snapshot date to stamp on every row. Defaults to today.
    on_drift : {"raise", "warn", "ignore"}
        Schema-canary policy; see :mod:`puremacro.fetch.realtime.canary`.

    Returns
    -------
    pd.DataFrame
        Columns ``["date", "vintage", "value"]``. Observations reported
        as ``"NaN"`` (missing) are dropped silently; rows whose date or
        value cannot be read are dropped with a warning counting them.
        Both ``Series.Obs`` (what SIETE emits) and the lowercase ``obs``
        spelling are read.
    """
    data = load_json(raw)
    if not data:
        return empty_snapshot()

    SchemaCanary.check("bcch", data, on_drift=on_drift)

    obs_list, _key = _bcch_observations(data) if isinstance(data, dict) else (None, None)
    if not isinstance(obs_list, list) or not obs_list:
        return empty_snapshot()

    records: list[tuple[pd.Timestamp, float]] = []
    skipped = 0
    for obs in obs_list:
        if not isinstance(obs, dict):
            skipped += 1
            continue
        d_str = str(obs.get("indexDateString") or "").strip()
        v_raw = obs.get("value")
        if not d_str or v_raw is None or str(v_raw).strip() == "":
            continue
        try:
            val = float(str(v_raw).replace(",", "."))
        except (ValueError, TypeError, Exception):
            skipped += 1
            continue
        if math.isnan(val):
            continue                        # SIETE's marker for no data

        # Parse date: DD-MM-YYYY (documented) or ISO
        try:
            if "-" in d_str and len(d_str.split("-")[0]) == 2:
                d = pd.to_datetime(d_str, format="%d-%m-%Y")
            else:
                d = pd.to_datetime(d_str)
        except (ValueError, ArithmeticError, Exception):
            skipped += 1
            continue
        records.append((d, val))
    warn_skipped("parse_bcch_json", skipped, len(obs_list))
    return finish_snapshot(records, vintage_date)


def fetch_bcch_vintages(
    series_id: str,
    *,
    user: str | None = None,
    password: str | None = None,
    firstdate: str | None = None,
    lastdate: str | None = None,
    vintage_date: str | None = None,
    timeout: float = 60.0,
    use_cache: bool = True,
    history: bool = True,
    on_drift: str = "raise",
) -> pd.DataFrame:
    """Fetch one SIETE series and return its locally stored snapshot vintages.

    Parameters
    ----------
    series_id : str
        BCCh series ID (e.g. ``'F022.TPM.TPO.D001.NO.Z.D'`` for the TPM).
    user : str | None
        SIETE user (the registered e-mail). Resolved from
        :mod:`puremacro.credentials` (``BCCH_API_USER`` or
        ``[bcch].user``) if omitted.
    password : str | None
        SIETE password. Resolved from ``BCCH_API_PASS`` or
        ``[bcch].password`` if omitted. Without both parts the stored
        snapshots are returned when there are any (and ``use_cache`` is
        True); otherwise ``MissingCredentialError``.
    firstdate, lastdate : str | None
        Optional ``yyyy-mm-dd`` window; empty means the whole series.
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
    u = credentials.get("bcch", explicit=user)
    p = credentials.get_password("bcch", explicit=password)
    if not (u and p):
        if use_cache:
            cached = cached_snapshots("bcch", "CHL", series_id)
            if not cached.empty:
                return cached
        credentials.require("bcch", explicit=user)

    url = _build_bcch_url(series_id, u, p, firstdate, lastdate)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    return fetch_snapshot_vintages(
        provider="bcch", country="CHL", series_id=series_id, request=req,
        parser=parse_bcch_json, timeout=timeout, use_cache=use_cache,
        history=history, on_drift=on_drift, vintage_date=vintage_date,
        redact=_redactor(u, p),
    )


def fetch_bcch_panel(
    countries,
    variables,
    *,
    user: str | None = None,
    password: str | None = None,
    timeout: float = 60.0,
    use_cache: bool = True,
    history: bool = True,
    on_drift: str = "raise",
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
                    history=history,
                    on_drift=on_drift,
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
    "BCCH_SIETE_ENDPOINT",
    "BCCH_SIETE_URL",
    "parse_bcch_json",
    "fetch_bcch_vintages",
    "fetch_bcch_panel",
]
