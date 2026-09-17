"""INEGI (Instituto Nacional de Estadística y Geografía) real-time connector.

Retrieves Mexican national statistics from the INEGI Indicadores API,
BIE data source (token in the path)::

    https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/INDICATOR/{series_id}/es/00/false/BIE/2.0/{token}?type=json

The path segments are the documented constructor: ``INDICATOR``,
indicator id, language ``es``, geography ``00`` (national, BIE
coding), ``false`` for the whole series rather than the latest datum
only, data source ``BIE``, API version ``2.0``, token. The body is
``{"Series": [{"INDICADOR": .., "FREQ": .., "OBSERVATIONS":
[{"TIME_PERIOD": "2024/01", "OBS_VALUE": "1.2"}]}]}``; ``TIME_PERIOD``
is ``YYYY/MM`` for monthly series and ``YYYY/N`` (N = 1..4) for
quarterly ones. Catalogued series are in
:data:`puremacro.fetch.realtime.catalog.INEGI_SERIES`.

INEGI overwrites series in place, so a *vintage* here is a **snapshot
date**: every fetch is stored in the local SQLite ``realtime_vintages``
table stamped with the day it was taken, and later calls return every
stored snapshot as one vintage each. Revision history therefore starts
with the first local snapshot.
"""
from __future__ import annotations

import functools
import urllib.request
from typing import Iterable

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
from .catalog import INEGI_SERIES, register_catalog

INEGI_SERIES_URL = (
    "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/"
    "INDICATOR/{series_id}/es/00/false/BIE/2.0/{token}?type=json"
)

_UA = "puremacro (real-time vintage reader)"

#: Spanish frequency words INEGI uses in ``FREQ`` on some payloads.
_INEGI_FREQ_WORDS = {
    "anual": "A", "semestral": "S", "trimestral": "Q", "mensual": "M",
    "quincenal": "SM", "semanal": "W", "diaria": "D",
}

#: Numeric ``CL_FREQ`` codes as published in the API catalogue. Used as
#: a hint only: the structure of ``TIME_PERIOD`` (see
#: :func:`_infer_inegi_quarterly`) overrides it whenever it is decisive,
#: so a code that turns out to be mis-mapped cannot mis-date a series
#: long enough to tell. Verify against the live catalogue when in doubt.
_INEGI_FREQ_CODES = {
    "3": "A", "4": "S", "5": "Q", "6": "M", "7": "SM", "8": "W", "9": "D",
}


def _freq_hint(freq_field: object) -> str | None:
    """``FREQ`` field -> ``"Q"`` / ``"M"`` / ... or None when unknown."""
    token = str(freq_field or "").strip().lower()
    if not token:
        return None
    if token in _INEGI_FREQ_CODES:
        return _INEGI_FREQ_CODES[token]
    for word, code in _INEGI_FREQ_WORDS.items():
        if word in token:
            return code
    return None


def _infer_inegi_quarterly(
    freq_field: object,
    periods: Iterable[object],
    explicit_freq: str | None = None,
) -> bool:
    """Decide whether ``YYYY/NN`` periods are quarters or months.

    Structural evidence wins. A period number above 4 rules out
    quarters. Period numbers that never exceed 4 across two or more
    years, or across five or more observations, rule out months (a
    monthly series is contiguous, so five months always reach May).
    Only when the payload is too short to tell does the caller's
    ``freq`` decide, then the ``FREQ`` field; failing all of those, the
    series is treated as monthly.
    """
    nums: list[int] = []
    years: set[int] = set()
    for tp in periods:
        text = str(tp).strip()
        if "/" not in text:
            continue
        year, _, num = text.partition("/")
        if year.isdigit() and num.isdigit():
            years.add(int(year))
            nums.append(int(num))
    if nums:
        if max(nums) > 4:
            return False
        if len(years) >= 2 or len(nums) >= 5:
            return True
    if explicit_freq:
        return str(explicit_freq).strip().upper().startswith("Q")
    hint = _freq_hint(freq_field)
    return hint == "Q"


def _parse_inegi_period(tp: str, is_quarterly: bool = False) -> pd.Timestamp | None:
    """Convert INEGI TIME_PERIOD string to Timestamp."""
    token = str(tp).strip()
    if not token:
        return None

    # Handle "YYYY/MM" or "YYYY/Q"
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
            except (ValueError, TypeError, Exception):
                pass

    # Handle standard formats: YYYY-MM-DD, YYYY-MM, YYYY-Q#
    try:
        if "Q" in token or "q" in token:
            return pd.Period(token.upper(), freq="Q").to_timestamp()
        return pd.to_datetime(token)
    except (ValueError, ArithmeticError, Exception):
        return None


def parse_inegi_json(
    raw: bytes | str | dict,
    *,
    series_id: str = "",
    vintage_date: str | pd.Timestamp | None = None,
    freq: str | None = None,
    on_drift: str = "raise",
) -> pd.DataFrame:
    """Parse an INEGI Indicadores JSON response into a tidy ``[date, vintage, value]`` frame.

    Parameters
    ----------
    raw : bytes | str | dict
        The JSON response from the INEGI API.
    series_id : str
        The indicator identifier (for messages only).
    vintage_date : str | pd.Timestamp | None
        Snapshot date to stamp on every row. Defaults to today.
    freq : str | None
        The catalogue's frequency for this indicator (``"Q"`` or
        ``"M"``). It only decides when the payload is too short for
        its ``TIME_PERIOD`` pattern to tell quarters from months; see
        :func:`_infer_inegi_quarterly`.
    on_drift : {"raise", "warn", "ignore"}
        Schema-canary policy; see :mod:`puremacro.fetch.realtime.canary`.

    Returns
    -------
    pd.DataFrame
        Columns ``["date", "vintage", "value"]``. Quarterly periods are
        dated to the first day of the quarter, monthly ones to the
        first of the month. Rows whose period or value cannot be read
        are dropped with a warning counting them.
    """
    data = load_json(raw)
    if not data:
        return empty_snapshot()

    SchemaCanary.check("inegi", data, on_drift=on_drift)

    series_list = data.get("Series") if isinstance(data, dict) else None
    if (not isinstance(series_list, list) or not series_list
            or not isinstance(series_list[0], dict)):
        return empty_snapshot()

    first_series = series_list[0]
    obs_list = first_series.get("OBSERVATIONS")
    if not isinstance(obs_list, list) or not obs_list:
        return empty_snapshot()

    is_quarterly = _infer_inegi_quarterly(
        first_series.get("FREQ"),
        (o.get("TIME_PERIOD") for o in obs_list if isinstance(o, dict)),
        explicit_freq=freq,
    )

    records: list[tuple[pd.Timestamp, float]] = []
    skipped = 0
    for obs in obs_list:
        if not isinstance(obs, dict):
            skipped += 1
            continue
        tp = obs.get("TIME_PERIOD")
        v_str = obs.get("OBS_VALUE")
        if tp is None or v_str is None or str(v_str).strip() == "":
            continue
        try:
            val = float(str(v_str).replace(",", ""))
        except (ValueError, TypeError, Exception):
            skipped += 1
            continue
        d = _parse_inegi_period(tp, is_quarterly=is_quarterly)
        if d is None:
            skipped += 1
            continue
        records.append((d, val))
    warn_skipped("parse_inegi_json", skipped, len(obs_list))
    return finish_snapshot(records, vintage_date)


def fetch_inegi_vintages(
    series_id: str,
    *,
    token: str | None = None,
    vintage_date: str | None = None,
    timeout: float = 60.0,
    use_cache: bool = True,
    history: bool = True,
    on_drift: str = "raise",
    freq: str | None = None,
) -> pd.DataFrame:
    """Fetch one INEGI indicator and return its locally stored snapshot vintages.

    Parameters
    ----------
    series_id : str
        INEGI BIE indicator ID (see
        :data:`puremacro.fetch.realtime.catalog.INEGI_SERIES`).
    token : str | None
        INEGI API token. Resolved from :mod:`puremacro.credentials`
        (``INEGI_API_KEY`` / ``[inegi].api_key``) if omitted. Without a
        token the stored snapshots are returned when there are any
        (and ``use_cache`` is True); otherwise ``MissingCredentialError``.
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
    freq : str | None
        Frequency of the indicator (``"Q"`` / ``"M"``) when the caller
        knows it; the catalogue supplies it for catalogued series. See
        :func:`parse_inegi_json`.

    Returns
    -------
    pd.DataFrame
        Columns ``["date", "vintage", "value"]``.
    """
    tok = token or credentials.get("inegi")
    if not tok:
        if use_cache:
            cached = cached_snapshots("inegi", "MEX", series_id)
            if not cached.empty:
                return cached
        credentials.require("inegi")

    url = INEGI_SERIES_URL.format(series_id=series_id, token=tok)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    return fetch_snapshot_vintages(
        provider="inegi", country="MEX", series_id=series_id, request=req,
        parser=functools.partial(parse_inegi_json, freq=freq),
        timeout=timeout, use_cache=use_cache, history=history,
        on_drift=on_drift, vintage_date=vintage_date,
        redact=lambda text: text.replace(tok, "***"),
    )


def fetch_inegi_panel(
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
                    history=history,
                    on_drift=on_drift,
                    freq=spec.freq or None,
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
