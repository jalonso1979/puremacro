"""Fetch, parse, store and fall back — the loop the snapshot connectors share.

Banxico, INEGI, BCB and BCCh overwrite their series in place and carry
no vintage field, so the only revision history available for them is
the one this package accumulates locally: every fetch is stored in the
``realtime_vintages`` SQLite table stamped with its *snapshot date*,
and a later call returns every stored snapshot as one vintage each.

The four connectors differ only in how they build the request and
parse the body; everything after that (schema canary policy, cache
write, history read-back, fallback, telemetry) lives here so a fix
lands in all of them at once.

Private: nothing here is part of the public API.
"""
from __future__ import annotations

import json
import urllib.request
import warnings
from typing import Any, Callable

import pandas as pd

from ... import _cache_db
from .canary import SchemaDriftError, SchemaDriftWarning

#: The tidy shape every ``parse_*_json`` returns.
SNAPSHOT_COLUMNS = ["date", "vintage", "value"]


def empty_snapshot() -> pd.DataFrame:
    return pd.DataFrame(columns=SNAPSHOT_COLUMNS)


def load_json(raw: Any) -> Any:
    """Decode ``raw`` (bytes / str / already-parsed JSON) or return None.

    ``None`` means "nothing to parse" (empty body, unsupported type);
    the caller returns an empty snapshot for it. A body that is not
    valid JSON raises ``json.JSONDecodeError`` so the fetch layer can
    treat it as a failed fetch rather than as an empty series.
    """
    if isinstance(raw, (bytes, bytearray)):
        text = raw.decode("utf-8-sig", errors="ignore")
        return json.loads(text) if text.strip() else None
    if isinstance(raw, str):
        return json.loads(raw) if raw.strip() else None
    if isinstance(raw, (list, dict)):
        return raw
    return None


def snapshot_stamp(vintage_date: Any) -> pd.Timestamp:
    """The vintage stamp for one snapshot: the caller's date, else today."""
    if vintage_date is not None:
        return pd.to_datetime(vintage_date)
    return pd.Timestamp.now(tz=None).normalize()


def finish_snapshot(
    records: list[tuple[pd.Timestamp, float]], vintage_date: Any,
) -> pd.DataFrame:
    """``[(date, value)]`` -> tidy ``[date, vintage, value]``, deduplicated."""
    if not records:
        return empty_snapshot()
    stamp = snapshot_stamp(vintage_date)
    df = pd.DataFrame(
        [(d, stamp, v) for d, v in records], columns=SNAPSHOT_COLUMNS,
    )
    return (
        df.drop_duplicates(subset=["date", "vintage"], keep="last")
        .sort_values(["date", "vintage"])
        .reset_index(drop=True)
    )


def warn_skipped(parser: str, skipped: int, total: int) -> None:
    """Report observations a parser dropped because they were malformed.

    Documented missing-value markers (``N/E``, ``NaN``, empty strings)
    are not counted: they are data, not drift. Rows whose date or value
    could not be read at all are, so a payload that is silently losing
    half its observations does not pass as a short series.
    """
    if skipped > 0:
        warnings.warn(
            f"{parser} skipped {skipped} of {total} observations it could "
            "not parse (malformed date or value); the rest were kept.",
            UserWarning, stacklevel=3,
        )


def record_event(provider: str, outcome: str, fallback_used: str) -> None:
    """Telemetry that cannot break the fetch that emitted it."""
    try:
        _cache_db.record_connector_event(provider, outcome, fallback_used)
    except Exception as exc:                          # pragma: no cover
        warnings.warn(
            f"{provider}: could not record connector telemetry ({exc})",
            UserWarning, stacklevel=3,
        )


def cached_snapshots(provider: str, country: str, series_id: str) -> pd.DataFrame:
    """Every locally stored snapshot of one series, as ``[date, vintage, value]``.

    A cache that cannot be read is reported and treated as empty; the
    caller decides whether that means "raise the original error".
    """
    try:
        cached = _cache_db.query_realtime_vintages(provider, country, series_id)
    except Exception as exc:
        warnings.warn(
            f"{provider}: could not read the realtime_vintages cache ({exc})",
            UserWarning, stacklevel=3,
        )
        return empty_snapshot()
    if cached is None or cached.empty:
        return empty_snapshot()
    return cached[SNAPSHOT_COLUMNS].reset_index(drop=True)


def _redact_exception(exc: BaseException, redact: Callable[[str], str] | None) -> None:
    """Scrub credentials from the URL attributes urllib hangs on its errors."""
    if redact is None:
        return
    for attr in ("url", "filename"):
        value = getattr(exc, attr, None)
        if isinstance(value, str):
            try:
                setattr(exc, attr, redact(value))
            except (ValueError, ArithmeticError, Exception):                         # pragma: no cover
                pass


def fetch_snapshot_vintages(
    *,
    provider: str,
    country: str,
    series_id: str,
    request: urllib.request.Request,
    parser: Callable[..., pd.DataFrame],
    timeout: float,
    use_cache: bool = True,
    history: bool = True,
    on_drift: str = "raise",
    vintage_date: Any = None,
    redact: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """GET ``request``, parse it, store the snapshot, return the vintages.

    Parameters
    ----------
    parser : callable
        ``parser(body, series_id=..., vintage_date=..., on_drift=...)``
        returning ``[date, vintage, value]``.
    use_cache : bool
        ``False`` neither reads nor writes ``realtime_vintages``: the
        live snapshot is returned as parsed and any failure is raised.
    history : bool
        ``True`` (default) returns every snapshot stored locally for
        this series, today's included, one vintage per snapshot date.
        ``False`` returns only the snapshot just fetched.
    on_drift : {"raise", "warn", "ignore"}
        Schema-canary policy handed to the parser. ``"raise"`` turns
        drift into a failed fetch, which falls back to the cache when
        it holds anything.
    redact : callable, optional
        Applied to the URL stored on urllib errors and to the failure
        message, for providers that carry credentials in the query.

    Failure semantics
    -----------------
    A network error, a non-JSON body, or schema drift under
    ``on_drift="raise"`` falls back to the cached snapshots with a
    warning (``SchemaDriftWarning`` for drift, ``UserWarning``
    otherwise) and a ``fallback`` telemetry event; with nothing cached,
    or with ``use_cache=False``, the error propagates. A snapshot that
    cannot be *stored* is still returned — the cache is a convenience,
    the live data is the point — and telemetry never raises.
    """
    name = f"fetch_{provider}_vintages"
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = resp.read()
        df = parser(
            body, series_id=series_id, vintage_date=vintage_date,
            on_drift=on_drift,
        )
    except Exception as exc:
        _redact_exception(exc, redact)
        if not use_cache:
            raise
        cached = cached_snapshots(provider, country, series_id)
        if cached.empty:
            raise
        detail = str(exc)
        if redact is not None:
            detail = redact(detail)
        category = (SchemaDriftWarning if isinstance(exc, SchemaDriftError)
                    else UserWarning)
        warnings.warn(
            f"{name} failed ({detail}); falling back to cached vintages.",
            category, stacklevel=3,
        )
        record_event(provider, "fallback", "sqlite_cache")
        return cached

    if not use_cache:
        return df

    if df.empty:
        cached = cached_snapshots(provider, country, series_id)
        if cached.empty:
            return df
        warnings.warn(
            f"{name} received empty observations; falling back to cached "
            "vintages.",
            UserWarning, stacklevel=3,
        )
        record_event(provider, "fallback", "sqlite_cache")
        return cached

    store_df = df.assign(provider=provider, country=country, series_id=series_id)
    try:
        _cache_db.store_realtime_vintages(store_df)
    except Exception as exc:
        warnings.warn(
            f"{name}: could not store this snapshot in realtime_vintages "
            f"({exc}); returning the live snapshot only.",
            UserWarning, stacklevel=3,
        )
        record_event(provider, "success", "none")
        return df
    record_event(provider, "success", "none")

    if not history:
        return df
    cached = cached_snapshots(provider, country, series_id)
    return cached if not cached.empty else df
