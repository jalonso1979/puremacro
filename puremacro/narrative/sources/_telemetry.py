"""Per-connector health telemetry (0.67.0+).

Each fetch / parse attempt by a participating connector produces one
row in the ``connector_events`` SQLite table:

    (ts, source, outcome, fallback_used)

Emitters:
- ``puremacro.narrative.sources._fallback.fetch_with_fallback`` — one
  event per stage attempt (live, wayback, playwright).
- The 8 Slice-A schema-checked iter_<source> wrappers — one event per
  ``ParserSchemaMismatchError`` catch (outcome="parser_schema_mismatch",
  fallback_used="none").

Researchers introspect with ``connector_health(window=...)``.

Kill-switch: ``PUREMACRO_NARRATIVE_TELEMETRY=0`` disables event logging
without breaking any caller. Telemetry must NEVER raise — DB failures
emit a ``UserWarning`` and silently no-op.
"""
from __future__ import annotations

import os
import sqlite3
import time
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


VALID_OUTCOMES: frozenset[str] = frozenset({
    "success", "404", "timeout", "ssl_fail", "server_5xx",
    "wayback_no_snapshot", "playwright_unavailable",
    "parser_schema_mismatch", "other_network_error",
})

VALID_FALLBACK_USED: frozenset[str] = frozenset({
    "live", "wayback", "playwright", "none",
})


def telemetry_enabled() -> bool:
    """True unless ``PUREMACRO_NARRATIVE_TELEMETRY=0`` is set."""
    return os.environ.get("PUREMACRO_NARRATIVE_TELEMETRY", "1") != "0"


def log_event(
    *,
    source: str,
    outcome: str,
    fallback_used: str = "none",
) -> None:
    """Insert one row into connector_events.

    Validates ``outcome`` and ``fallback_used`` against the registries
    above (raises ``ValueError`` on miss — programmer error). Skipped
    silently if ``telemetry_enabled()`` is False. DB failures emit a
    ``UserWarning`` and no-op — the calling fetch path MUST keep going.
    """
    if outcome not in VALID_OUTCOMES:
        raise ValueError(
            f"log_event: outcome {outcome!r} not in {sorted(VALID_OUTCOMES)}"
        )
    if fallback_used not in VALID_FALLBACK_USED:
        raise ValueError(
            f"log_event: fallback_used {fallback_used!r} not in "
            f"{sorted(VALID_FALLBACK_USED)}"
        )
    if not telemetry_enabled():
        return
    # _cache_db lives at puremacro._cache_db (NOT
    # puremacro.narrative._cache_db). Local import keeps cold-import time
    # cheap and avoids a hard module-load dependency that could affect
    # Pyodide guards.
    from puremacro import _cache_db as _db
    try:
        _db.get_conn().execute(
            "INSERT INTO connector_events "
            "(ts, source, outcome, fallback_used) VALUES (?, ?, ?, ?)",
            (int(time.time()), source, outcome, fallback_used),
        )
    except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
        warnings.warn(
            f"puremacro.narrative.sources._telemetry.log_event failed "
            f"({source}/{outcome}): {e}",
            UserWarning, stacklevel=2,
        )


def connector_health(
    *,
    window: pd.Timedelta | None = None,
    sources: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate connector_events over the last ``window`` (default 7 days).

    Returns one row per source with columns:
      source, n_total, n_success, success_rate,
      n_fallback, fallback_rate, last_seen
    Where:
      - n_total      = rows for the source in the window
      - n_success    = rows with outcome='success'
      - success_rate = n_success / n_total
      - n_fallback   = rows with fallback_used != 'live'
                       (i.e., served by wayback or playwright,
                       or — for parser_schema_mismatch — 'none')
      - fallback_rate = n_fallback / n_total
      - last_seen    = max(ts) for the source

    Empty DataFrame (with the expected 7 columns) if no events match.
    DB failures emit a UserWarning and return the empty shape.
    """
    import pandas as pd
    from puremacro import _cache_db as _db

    expected_cols = ["source", "n_total", "n_success", "success_rate",
                     "n_fallback", "fallback_rate", "last_seen"]
    if window is None:
        window = pd.Timedelta(days=7)

    cutoff = int(time.time() - window.total_seconds())
    sql = (
        "SELECT source, COUNT(*) AS n_total, "
        "SUM(CASE WHEN outcome='success' THEN 1 ELSE 0 END) AS n_success, "
        "SUM(CASE WHEN fallback_used <> 'live' THEN 1 ELSE 0 END) AS n_fallback, "
        "MAX(ts) AS last_seen_ts "
        "FROM connector_events WHERE ts >= ?"
    )
    params: list = [cutoff]
    if sources:
        placeholders = ",".join("?" * len(sources))
        sql += f" AND source IN ({placeholders})"
        params.extend(sources)
    sql += " GROUP BY source ORDER BY source"

    try:
        rows = _db.get_conn().execute(sql, params).fetchall()
    except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
        warnings.warn(
            f"connector_health query failed: {e}",
            UserWarning, stacklevel=2,
        )
        return pd.DataFrame(columns=expected_cols)

    if not rows:
        return pd.DataFrame(columns=expected_cols)

    df = pd.DataFrame(rows, columns=["source", "n_total", "n_success",
                                       "n_fallback", "last_seen_ts"])
    df["n_total"] = df["n_total"].astype("int64")
    df["n_success"] = df["n_success"].astype("int64")
    df["n_fallback"] = df["n_fallback"].astype("int64")
    df["success_rate"] = df["n_success"] / df["n_total"]
    df["fallback_rate"] = df["n_fallback"] / df["n_total"]
    df["last_seen"] = pd.to_datetime(df["last_seen_ts"], unit="s")
    df = df[["source", "n_total", "n_success", "success_rate",
             "n_fallback", "fallback_rate", "last_seen"]]
    return df


__all__ = [
    "VALID_OUTCOMES",
    "VALID_FALLBACK_USED",
    "telemetry_enabled",
    "log_event",
    "connector_health",
]
