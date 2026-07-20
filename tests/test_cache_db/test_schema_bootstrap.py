"""F2.1 — Schema bootstrap is idempotent + WAL mode is enabled."""
from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Force every call in this test to use a tmp_path DB; reset the singleton."""
    db = tmp_path / "cache.db"
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(db.parent))
    import puremacro._cache_db as M
    M.close_conn()
    yield db
    M.close_conn()


def test_default_db_path_uses_env_directory(monkeypatch, tmp_path):
    from puremacro._cache_db import default_db_path
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    assert default_db_path() == tmp_path / "cache.db"


def test_default_db_path_accepts_explicit_db_file(monkeypatch, tmp_path):
    from puremacro._cache_db import default_db_path
    explicit = tmp_path / "custom.db"
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(explicit))
    assert default_db_path() == explicit


def test_default_db_path_fallback(monkeypatch, tmp_path):
    from puremacro._cache_db import default_db_path
    from pathlib import Path
    monkeypatch.delenv("PUREMACRO_HTTP_CACHE_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert default_db_path() == tmp_path / ".cache" / "puremacro" / "cache.db"


def test_bootstrap_creates_three_tables(fresh_db):
    from puremacro._cache_db import get_conn
    conn = get_conn(fresh_db)
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert {"http_cache", "alfred_vintages", "schema_version",
            "connector_events"}.issubset(tables)


def test_bootstrap_seeds_schema_version(fresh_db):
    from puremacro._cache_db import get_conn
    conn = get_conn(fresh_db)
    rows = dict(conn.execute("SELECT component, version FROM schema_version"))
    assert rows == {"http_cache": 1, "alfred_vintages": 1,
                    "connector_events": 1}


def test_bootstrap_is_idempotent(fresh_db):
    from puremacro._cache_db import get_conn, bootstrap_schema
    conn = get_conn(fresh_db)
    bootstrap_schema(conn)
    bootstrap_schema(conn)  # second call must not raise or duplicate
    rows = list(conn.execute("SELECT component, version FROM schema_version"))
    assert len(rows) == 3


def test_wal_mode_enabled(fresh_db):
    from puremacro._cache_db import get_conn
    conn = get_conn(fresh_db)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_singleton_returns_same_connection(fresh_db):
    from puremacro._cache_db import get_conn
    a = get_conn(fresh_db)
    b = get_conn(fresh_db)
    assert a is b
