"""Disk cache abstraction (roadmap item D2).

Provides a unified namespace-keyed cache backed by parquet (DataFrame /
Series) and JSON (everything else) under ``~/.cache/puremacro/``. The
goal is to replace the ad-hoc ``notebooks/data_cache/*.parquet`` files
scattered through notebook builders with a single entry point that
callers can use *anywhere* — not just inside a notebook.

API:

    from puremacro.cache import disk_cache, disk_cache_path

    df = disk_cache('bbd_aieu_monthly', loader=lambda: pd.read_csv(...))
    # Subsequent calls hit the cache.

    # Or imperatively:
    if disk_cache_path('mything').exists():
        ...

Env vars:
  PUREMACRO_CACHE_DIR  — override the default location.
  PUREMACRO_CACHE_DISABLE=1  — bypass the cache (always call loader).

Per-call TTL is optional — pass ``ttl_seconds=N`` to force re-fetch
when the cached file is older than that.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd


T = TypeVar("T")


def _default_root() -> Path:
    return Path(
        os.environ.get(
            "PUREMACRO_CACHE_DIR",
            str(Path.home() / ".cache" / "puremacro"),
        )
    )


def disk_cache_path(key: str, *, namespace: str = "default",
                    suffix: str = ".parquet") -> Path:
    """Return the canonical path for a (namespace, key) cache entry."""
    safe_key = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    return _default_root() / namespace / f"{safe_key}{suffix}"


def _is_stale(path: Path, ttl_seconds: int | None) -> bool:
    if ttl_seconds is None:
        return False
    if not path.exists():
        return True
    return (time.time() - path.stat().st_mtime) > ttl_seconds


def disk_cache(
    key: str,
    loader: Callable[[], T],
    *,
    namespace: str = "default",
    ttl_seconds: int | None = None,
    refetch: bool = False,
) -> T:
    """Cache the return value of ``loader()`` keyed by (namespace, key).

    Storage format chosen by inspecting the loader's return type on
    cache-miss:
      - ``pd.DataFrame`` / ``pd.Series`` → parquet
      - everything else → pickle (covers dicts, lists, np.ndarray, etc.)

    Set ``refetch=True`` (or env var ``PUREMACRO_CACHE_DISABLE=1``) to
    bypass the cache and always call the loader.
    """
    disabled = (
        os.environ.get("PUREMACRO_CACHE_DISABLE") == "1" or refetch
    )
    pq = disk_cache_path(key, namespace=namespace, suffix=".parquet")
    pkl = disk_cache_path(key, namespace=namespace, suffix=".pkl")

    if not disabled:
        if pq.exists() and not _is_stale(pq, ttl_seconds):
            return pd.read_parquet(pq)
        if pkl.exists() and not _is_stale(pkl, ttl_seconds):
            with pkl.open("rb") as f:
                return pickle.load(f)

    value = loader()
    if isinstance(value, (pd.DataFrame, pd.Series)):
        pq.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, pd.Series):
            value.to_frame(name=value.name or "value").to_parquet(pq, index=True)
        else:
            value.to_parquet(pq, index=True)
    else:
        pkl.parent.mkdir(parents=True, exist_ok=True)
        with pkl.open("wb") as f:
            pickle.dump(value, f)
    return value


# ── HTTP-cache introspection (0.66.0+) ───────────────────────────────
# These functions read from the SQLite http_cache table populated by
# puremacro._http_cache.cache_write. They are distinct from the
# disk_cache / disk_cache_path helpers above (which back a DataFrame /
# JSON cache under a different namespace).

def http_list_urls() -> list[str]:
    """Return all URLs currently in the HTTP cache, sorted alphabetically."""
    from . import _cache_db
    conn = _cache_db.get_conn()
    rows = conn.execute("SELECT url FROM http_cache ORDER BY url").fetchall()
    return [r[0] for r in rows]


def http_cache_size_bytes() -> int:
    """Total bytes stored in the http_cache.body column."""
    from . import _cache_db
    conn = _cache_db.get_conn()
    row = conn.execute(
        "SELECT COALESCE(SUM(LENGTH(body)), 0) FROM http_cache"
    ).fetchone()
    return int(row[0])


def http_cache_clear(older_than: "pd.Timedelta | None" = None) -> int:
    """Delete entries older than ``older_than`` (None = delete all).

    Returns the number of rows deleted. Issues a VACUUM after large
    deletions (>1000 rows) so the SQLite file actually shrinks on disk.
    """
    import time
    from . import _cache_db
    conn = _cache_db.get_conn()
    if older_than is None:
        cur = conn.execute("DELETE FROM http_cache")
    else:
        cutoff = int(time.time() - older_than.total_seconds())
        cur = conn.execute(
            "DELETE FROM http_cache WHERE fetched_at < ?", (cutoff,)
        )
    deleted = cur.rowcount or 0
    if deleted > 1000:
        conn.execute("VACUUM")
    return deleted


__all__ = ["disk_cache", "disk_cache_path", "http_list_urls",
           "http_cache_size_bytes", "http_cache_clear"]
