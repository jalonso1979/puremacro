"""SQLite connection manager + schema bootstrap for puremacro's data
infrastructure.

A single SQLite file at ``~/.cache/puremacro/cache.db`` (overridable
via ``$PUREMACRO_HTTP_CACHE_DIR``) hosts five tables:

- ``http_cache`` — replaces the flat-file HTTP cache; backs
  ``puremacro._http_cache.cache_read``/``cache_write``.
- ``alfred_vintages`` — persistent FRED-ALFRED vintage panel; backs
  ``puremacro.vintages.AlfredVintageStore``.
- ``connector_events`` — fetch telemetry (``record_connector_event``,
  ``puremacro.narrative.sources._telemetry.log_event``).
- ``realtime_vintages`` — local snapshot history of the central-bank
  connectors (Banxico, INEGI, BCB, BCCh), one row per (series,
  observation, snapshot date).
- ``schema_version`` — registry for future migrations.

Bootstrap is additive (``CREATE TABLE IF NOT EXISTS`` + ``INSERT OR
IGNORE``), so a cache file written by an earlier release gains the
newer tables on first open and keeps its rows.

WAL journal mode is enabled so multiple notebooks against the same DB
do not block each other on writes. One ``sqlite3.Connection`` per
process is kept alive in a module-level singleton. A cache file the
process cannot write (read-only permissions, a mounted snapshot) is
opened read-only: reads work, writes raise ``sqlite3.OperationalError``
as usual, and the callers that write only opportunistically (telemetry,
snapshot storage) degrade to a warning.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

_DDL_HTTP_CACHE = """
CREATE TABLE IF NOT EXISTS http_cache (
    key            TEXT PRIMARY KEY,
    url            TEXT NOT NULL,
    fetched_at     INTEGER NOT NULL,
    content_type   TEXT,
    body           BLOB NOT NULL
);
"""

_DDL_HTTP_CACHE_IDX = (
    "CREATE INDEX IF NOT EXISTS http_cache_fetched_at_idx " "ON http_cache(fetched_at);"
)

_DDL_ALFRED_VINTAGES = """
CREATE TABLE IF NOT EXISTS alfred_vintages (
    series_id        TEXT NOT NULL,
    observation_date TEXT NOT NULL,
    vintage_date     TEXT NOT NULL,
    value            REAL,
    PRIMARY KEY (series_id, observation_date, vintage_date)
);
"""

_DDL_ALFRED_VINTAGES_IDX = (
    "CREATE INDEX IF NOT EXISTS alfred_vintages_series_vintage_idx "
    "ON alfred_vintages(series_id, vintage_date);"
)

_DDL_CONNECTOR_EVENTS = """
CREATE TABLE IF NOT EXISTS connector_events (
    ts             INTEGER NOT NULL,
    source         TEXT NOT NULL,
    outcome        TEXT NOT NULL,
    fallback_used  TEXT NOT NULL
);
"""

_DDL_CONNECTOR_EVENTS_IDX = (
    "CREATE INDEX IF NOT EXISTS connector_events_ts_source_idx "
    "ON connector_events(ts, source);"
)

_DDL_REALTIME_VINTAGES = """
CREATE TABLE IF NOT EXISTS realtime_vintages (
    provider         TEXT NOT NULL,
    country          TEXT NOT NULL,
    series_id        TEXT NOT NULL,
    observation_date TEXT NOT NULL,
    vintage_date     TEXT NOT NULL,
    value            REAL,
    PRIMARY KEY (provider, country, series_id, observation_date, vintage_date)
);
"""

_DDL_REALTIME_VINTAGES_IDX = (
    "CREATE INDEX IF NOT EXISTS realtime_vintages_idx "
    "ON realtime_vintages(provider, country, series_id, vintage_date);"
)

_DDL_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
    component TEXT PRIMARY KEY,
    version   INTEGER NOT NULL
);
"""

_SCHEMA_SEED = [
    ("http_cache", 1),
    ("alfred_vintages", 1),
    ("connector_events", 1),
    ("realtime_vintages", 1),
]


def default_db_path() -> Path:
    """Resolve the canonical cache DB path.

    ``$PUREMACRO_HTTP_CACHE_DIR`` overrides everything:
      - if it ends with ``.db``, return that path verbatim;
      - otherwise treat as a directory and return ``<dir>/cache.db``.
    Default: ``~/.cache/puremacro/cache.db``.
    """
    env = os.environ.get("PUREMACRO_HTTP_CACHE_DIR")
    if env:
        p = Path(env)
        if p.suffix == ".db":
            return p
        return p / "cache.db"
    return Path.home() / ".cache" / "puremacro" / "cache.db"


def bootstrap_schema(conn: sqlite3.Connection) -> None:
    """Idempotent: create tables + seed schema_version rows if missing."""
    cur = conn.cursor()
    cur.execute(_DDL_HTTP_CACHE)
    cur.execute(_DDL_HTTP_CACHE_IDX)
    cur.execute(_DDL_ALFRED_VINTAGES)
    cur.execute(_DDL_ALFRED_VINTAGES_IDX)
    cur.execute(_DDL_CONNECTOR_EVENTS)
    cur.execute(_DDL_CONNECTOR_EVENTS_IDX)
    cur.execute(_DDL_REALTIME_VINTAGES)
    cur.execute(_DDL_REALTIME_VINTAGES_IDX)
    cur.execute(_DDL_SCHEMA_VERSION)
    for component, version in _SCHEMA_SEED:
        cur.execute(
            "INSERT OR IGNORE INTO schema_version (component, version) "
            "VALUES (?, ?)",
            (component, version),
        )
    conn.commit()


_CONN: sqlite3.Connection | None = None
_CONN_PATH: Path | None = None


def _connect_readonly(target: Path) -> sqlite3.Connection:
    """Open ``target`` read-only (SQLite URI ``mode=ro``).

    No WAL switch and no schema bootstrap: both are writes. Reads work;
    writes raise ``sqlite3.OperationalError`` exactly as they would on
    any read-only database.
    """
    uri = target.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(
        uri,
        uri=True,
        timeout=30.0,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_conn(db_path: Path | None = None) -> sqlite3.Connection:
    """Return the module-level singleton SQLite connection.

    Lazily opens on first call. Enables WAL mode and bootstraps the
    schema. If called twice with different ``db_path`` arguments
    (e.g., in tests), closes the previous connection and opens a new one.

    A database file the process cannot write is opened read-only
    instead of failing on the WAL switch, so a cache on a read-only
    volume can still be queried.
    """
    global _CONN, _CONN_PATH
    target = Path(db_path) if db_path is not None else default_db_path()
    if _CONN is not None and _CONN_PATH == target:
        return _CONN
    if _CONN is not None:
        _CONN.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not os.access(target, os.W_OK):
        conn = _connect_readonly(target)
    else:
        conn = sqlite3.connect(
            target,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            bootstrap_schema(conn)
        except sqlite3.OperationalError as exc:
            conn.close()
            if "readonly" not in str(exc).lower():
                raise
            conn = _connect_readonly(target)
    _CONN = conn
    _CONN_PATH = target
    return _CONN


def close_conn() -> None:
    """Close the singleton connection (used by tests)."""
    global _CONN, _CONN_PATH
    if _CONN is not None:
        try:
            _CONN.close()
        finally:
            _CONN = None
            _CONN_PATH = None


def migrate_from_flat_files(
    conn: sqlite3.Connection,
    flat_cache_dir: Path,
    *,
    remove: bool = False,
) -> int:
    """Walk ``flat_cache_dir`` for ``*.bin`` + ``*.json`` sidecar pairs.

    Inserts each into ``http_cache`` with ``INSERT OR IGNORE`` semantics
    (idempotent). If ``remove=True``, unlinks each migrated file pair
    after successful insert. Returns the count actually migrated.

    Failures on a per-file basis are warned-and-skipped, not raised —
    migration is opportunistic.
    """
    import json
    import warnings

    if not flat_cache_dir.exists():
        return 0
    migrated = 0
    for bin_path in flat_cache_dir.glob("*.bin"):
        sidecar = bin_path.with_suffix(".json")
        if not sidecar.exists():
            continue
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            url = meta["url"]
            fetched_at = int(float(meta["fetched_at"]))
            content_type = meta.get("content_type")
            body = bin_path.read_bytes()
            key = bin_path.stem
            cur = conn.execute(
                "INSERT OR IGNORE INTO http_cache "
                "(key, url, fetched_at, content_type, body) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, url, fetched_at, content_type, body),
            )
            if cur.rowcount > 0:
                migrated += 1
                if remove:
                    bin_path.unlink(missing_ok=True)
                    sidecar.unlink(missing_ok=True)
        except (OSError, ValueError, KeyError) as e:
            warnings.warn(
                f"puremacro._cache_db: skipping {bin_path.name}: {e}",
                UserWarning,
                stacklevel=2,
            )
            continue
    conn.commit()
    return migrated


def store_realtime_vintages(
    df,
    *,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Insert or replace rows into `realtime_vintages`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: provider, country, series_id,
        date (or observation_date), vintage (or vintage_date), and value.
    conn : sqlite3.Connection | None
        SQLite connection. Defaults to `get_conn()`.

    Returns
    -------
    int
        Number of rows inserted or updated.
    """
    if df is None or len(df) == 0:
        return 0
    import pandas as pd

    c = conn or get_conn()
    date_col = "date" if "date" in df.columns else "observation_date"
    vin_col = "vintage" if "vintage" in df.columns else "vintage_date"
    required = {"provider", "country", "series_id", date_col, vin_col, "value"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"store_realtime_vintages missing columns: {sorted(missing)}")

    import numpy as np

    # Filter invalid values matching iterrows logic
    is_na = df["value"].isna()
    val_float = pd.to_numeric(df["value"], errors="coerce")

    # We want values that were NA natively to be NaN/None (handled later),
    # but values that failed parsing (not natively NA but coerced to NA) should be skipped.
    valid_val_mask = is_na | val_float.notna()

    try:
        parsed_obs = pd.to_datetime(df[date_col], errors="coerce")
        parsed_vin = pd.to_datetime(df[vin_col], errors="coerce")
    except Exception:
        # Fallback to failing all dates if completely unparseable
        parsed_obs = pd.Series(pd.NaT, index=df.index)
        parsed_vin = pd.Series(pd.NaT, index=df.index)

    valid_mask = valid_val_mask & parsed_obs.notna() & parsed_vin.notna()

    if not valid_mask.any():
        return 0

    df_valid = df[valid_mask]
    obs_d = parsed_obs[valid_mask].dt.strftime("%Y-%m-%d")
    vin_d = parsed_vin[valid_mask].dt.strftime("%Y-%m-%d")
    val_float_valid = val_float[valid_mask]
    # replace NaN with None for SQLite
    val_float_list = [None if pd.isna(x) else float(x) for x in val_float_valid]

    records = list(
        zip(
            df_valid["provider"].astype(str),
            df_valid["country"].astype(str).str.upper(),
            df_valid["series_id"].astype(str),
            obs_d,
            vin_d,
            val_float_list,
        )
    )
    c.executemany(
        "INSERT OR REPLACE INTO realtime_vintages "
        "(provider, country, series_id, observation_date, vintage_date, value) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        records,
    )
    c.commit()
    return len(records)


def query_realtime_vintages(
    provider: str,
    country: str,
    series_id: str,
    *,
    vintage_date: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> Any:
    """Query `realtime_vintages` for a given provider, country, and series_id.

    Every stored snapshot is returned, one ``vintage`` per snapshot
    date; ``vintage_date`` is an *as-of* cutoff that keeps snapshots
    taken on or before it.

    Returns DataFrame with columns:
    date, vintage, value, provider, country, series_id
    """
    import pandas as pd

    c = conn or get_conn()
    c_code = str(country).upper()
    if vintage_date is not None:
        vin_str = pd.to_datetime(vintage_date).strftime("%Y-%m-%d")
        cur = c.execute(
            "SELECT observation_date, vintage_date, value, provider, country, series_id "
            "FROM realtime_vintages "
            "WHERE provider = ? AND country = ? AND series_id = ? AND vintage_date <= ? "
            "ORDER BY observation_date, vintage_date",
            (provider, c_code, series_id, vin_str),
        )
    else:
        cur = c.execute(
            "SELECT observation_date, vintage_date, value, provider, country, series_id "
            "FROM realtime_vintages "
            "WHERE provider = ? AND country = ? AND series_id = ? "
            "ORDER BY observation_date, vintage_date",
            (provider, c_code, series_id),
        )
    rows = cur.fetchall()
    cols = ["date", "vintage", "value", "provider", "country", "series_id"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows, columns=cols)
    out["date"] = pd.to_datetime(out["date"])
    out["vintage"] = pd.to_datetime(out["vintage"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out


def record_connector_event(
    source: str,
    outcome: str,
    fallback_used: str,
    *,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Log an event into the ``connector_events`` table.

    Telemetry never breaks a fetch: a locked, read-only or unreachable
    database emits a ``UserWarning`` and returns. The
    ``PUREMACRO_NARRATIVE_TELEMETRY=0`` kill-switch turns the call into
    a no-op (explicit ``conn`` included), matching
    ``puremacro.narrative.sources._telemetry.log_event``.
    """
    import time
    import warnings

    if os.environ.get("PUREMACRO_NARRATIVE_TELEMETRY", "1") == "0":
        return
    ts = int(time.time())
    try:
        c = conn or get_conn()
        c.execute(
            "INSERT INTO connector_events (ts, source, outcome, fallback_used) "
            "VALUES (?, ?, ?, ?)",
            (ts, source, outcome, fallback_used),
        )
        c.commit()
    except (sqlite3.Error, OSError) as exc:
        warnings.warn(
            f"puremacro._cache_db.record_connector_event failed "
            f"({source}/{outcome}): {exc}",
            UserWarning,
            stacklevel=2,
        )


__all__ = [
    "default_db_path",
    "bootstrap_schema",
    "get_conn",
    "close_conn",
    "migrate_from_flat_files",
    "store_realtime_vintages",
    "query_realtime_vintages",
    "record_connector_event",
]
