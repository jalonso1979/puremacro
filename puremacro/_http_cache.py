"""On-disk HTTP cache, backed by SQLite (0.66.0+).

Used by the opt-in ``safe_get_*_cached`` helpers in :mod:`puremacro._http`.

Public API preserved from 0.65.0:
- ``cache_key(url) -> str``
- ``default_cache_dir() -> Path``  (now returns the DB's parent dir)
- ``cache_read(cache_dir, url, ttl_seconds=...) -> bytes | None``
- ``cache_write(cache_dir, url, body, content_type=None) -> None``

Behavior unchanged from the caller's perspective. Internal storage
switches from one flat file per URL to a single SQLite DB at
``cache_dir/cache.db`` (created on first write).

Cache failures (OperationalError, disk full, etc.) MUST NOT break the
caller — ``cache_read`` returns None and ``cache_write`` no-ops, both
emitting a UserWarning so the failure is visible without being fatal.

A one-time lazy migration runs on the first call after upgrade: if the
legacy ``cache_dir/*.bin`` files exist and ``http_cache`` is empty, the
flat-file entries are inserted into the DB (without deleting the files).
A UserWarning points to ``tools/cache_migrate.py --apply --rm`` for users
who also want to delete the flat files.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import time
import warnings
from pathlib import Path

from . import _cache_db


def cache_key(url: str) -> str:
    """SHA-256 hex of the URL — used as the http_cache primary key."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def default_cache_dir() -> Path:
    """Return the default cache directory.

    For backwards compatibility with 0.65.0 callers, this returns a
    *directory* path; the SQLite DB lives at ``<dir>/cache.db``.
    ``$PUREMACRO_HTTP_CACHE_DIR`` overrides; otherwise ``~/.cache/puremacro``.
    """
    env = os.environ.get("PUREMACRO_HTTP_CACHE_DIR")
    if env:
        p = Path(env)
        # If the env var points at a .db file, return its parent dir.
        return p.parent if p.suffix == ".db" else p
    return Path.home() / ".cache" / "puremacro"


def _resolve_db_path(cache_dir: Path) -> Path:
    """Map a 0.65.0-style cache_dir argument to the actual DB path."""
    p = Path(cache_dir)
    if p.suffix == ".db":
        return p
    return p / "cache.db"


_MIGRATION_ATTEMPTED = False


def _maybe_migrate_flat_files(cache_dir: Path, conn: sqlite3.Connection) -> None:
    """First-call lazy migration: walk cache_dir/*.bin into http_cache."""
    global _MIGRATION_ATTEMPTED
    if _MIGRATION_ATTEMPTED:
        return
    _MIGRATION_ATTEMPTED = True
    flat = Path(cache_dir)
    if not flat.exists():
        return
    # Only run if there are flat files AND http_cache is empty.
    bin_files = list(flat.glob("*.bin"))
    if not bin_files:
        return
    row = conn.execute("SELECT COUNT(*) FROM http_cache").fetchone()
    if row and row[0] > 0:
        return
    try:
        n = _cache_db.migrate_from_flat_files(conn, flat, remove=False)
    except Exception as e:  # defensive; migration must not fail callers
        warnings.warn(
            f"puremacro._http_cache: lazy migration failed: {e}",
            UserWarning, stacklevel=3,
        )
        return
    if n > 0:
        warnings.warn(
            f"puremacro._http_cache: migrated {n} flat-file cache entries "
            f"to SQLite at {_resolve_db_path(flat)}. Run "
            f"`python tools/cache_migrate.py --apply --rm` if you also "
            f"want to delete the original flat files.",
            UserWarning, stacklevel=3,
        )


def cache_read(
    cache_dir: Path,
    url: str,
    ttl_seconds: int = 30 * 24 * 3600,
) -> bytes | None:
    """Return cached bytes for ``url`` if fresh, else None.

    Returns None on: miss, stale entry, or any DB failure. Cache
    failures emit a UserWarning but never raise.
    """
    try:
        conn = _cache_db.get_conn(_resolve_db_path(cache_dir))
        _maybe_migrate_flat_files(cache_dir, conn)
        row = conn.execute(
            "SELECT body, fetched_at FROM http_cache WHERE key = ?",
            (cache_key(url),),
        ).fetchone()
    except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
        warnings.warn(
            f"puremacro._http_cache: cache_read failed for {url!r}: {e}",
            UserWarning, stacklevel=2,
        )
        return None
    if row is None:
        return None
    body, fetched_at = row
    if time.time() - int(fetched_at) >= ttl_seconds:
        return None
    return body


def cache_write(
    cache_dir: Path,
    url: str,
    body: bytes,
    content_type: str | None = None,
) -> None:
    """Write ``body`` under the cache key for ``url``.

    Silently no-ops on any DB failure (emits a UserWarning).
    """
    try:
        conn = _cache_db.get_conn(_resolve_db_path(cache_dir))
        _maybe_migrate_flat_files(cache_dir, conn)
        conn.execute(
            "INSERT OR REPLACE INTO http_cache "
            "(key, url, fetched_at, content_type, body) VALUES (?, ?, ?, ?, ?)",
            (cache_key(url), url, int(time.time()), content_type, body),
        )
    except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
        warnings.warn(
            f"puremacro._http_cache: cache_write failed for {url!r}: {e}",
            UserWarning, stacklevel=2,
        )


__all__ = ["cache_key", "default_cache_dir", "cache_read", "cache_write"]
