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
    assert rows == {
        "http_cache": 1,
        "alfred_vintages": 1,
        "connector_events": 1,
        "realtime_vintages": 1,
    }


def test_bootstrap_is_idempotent(fresh_db):
    from puremacro._cache_db import get_conn, bootstrap_schema
    conn = get_conn(fresh_db)
    bootstrap_schema(conn)
    bootstrap_schema(conn)  # second call must not raise or duplicate
    rows = list(conn.execute("SELECT component, version FROM schema_version"))
    assert len(rows) == 4


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


def _write_330_cache_file(db) -> None:
    """A cache.db exactly as puremacro 3.3.0 wrote it: four tables, three
    schema_version rows, no realtime_vintages."""
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE http_cache (
            key TEXT PRIMARY KEY, url TEXT NOT NULL, fetched_at INTEGER NOT NULL,
            content_type TEXT, body BLOB NOT NULL);
        CREATE INDEX http_cache_fetched_at_idx ON http_cache(fetched_at);
        CREATE TABLE alfred_vintages (
            series_id TEXT NOT NULL, observation_date TEXT NOT NULL,
            vintage_date TEXT NOT NULL, value REAL,
            PRIMARY KEY (series_id, observation_date, vintage_date));
        CREATE TABLE connector_events (
            ts INTEGER NOT NULL, source TEXT NOT NULL, outcome TEXT NOT NULL,
            fallback_used TEXT NOT NULL);
        CREATE TABLE schema_version (component TEXT PRIMARY KEY, version INTEGER NOT NULL);
        INSERT INTO schema_version VALUES ('http_cache', 1), ('alfred_vintages', 1),
            ('connector_events', 1);
        INSERT INTO http_cache VALUES ('k', 'https://example.com/a', 1700000000, 'text/plain', X'6869');
        INSERT INTO alfred_vintages VALUES ('GDPC1', '2020-01-01', '2020-04-29', 100.5);
    """)
    conn.commit()
    conn.close()


def test_330_cache_file_opens_and_gains_realtime_vintages(fresh_db):
    """A 3.3.0-era cache.db migrates in place on first open: the new table
    and schema_version row appear, existing rows are untouched."""
    from puremacro._cache_db import (
        get_conn, query_realtime_vintages, store_realtime_vintages,
    )
    import pandas as pd
    _write_330_cache_file(fresh_db)
    conn = get_conn(fresh_db)
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "realtime_vintages" in tables
    assert dict(conn.execute("SELECT component, version FROM schema_version")) == {
        "http_cache": 1, "alfred_vintages": 1,
        "connector_events": 1, "realtime_vintages": 1,
    }
    assert conn.execute("SELECT url, body FROM http_cache").fetchall() == [
        ("https://example.com/a", b"hi")]
    assert conn.execute("SELECT value FROM alfred_vintages").fetchall() == [(100.5,)]
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    store_realtime_vintages(pd.DataFrame([{
        "provider": "bcb", "country": "BRA", "series_id": "432",
        "date": "2026-01-01", "vintage": "2026-02-01", "value": 12.25}]), conn=conn)
    assert len(query_realtime_vintages("bcb", "BRA", "432", conn=conn)) == 1


def test_readonly_cache_file_opens_read_only(fresh_db):
    """A cache file the process cannot write is opened read-only instead of
    failing on the WAL switch / schema bootstrap."""
    import os
    from puremacro._cache_db import close_conn, get_conn
    _write_330_cache_file(fresh_db)
    os.chmod(fresh_db, 0o444)
    if os.access(fresh_db, os.W_OK):
        os.chmod(fresh_db, 0o644)
        pytest.skip("file permissions do not restrict this account (root?)")
    try:
        conn = get_conn(fresh_db)
        assert conn.execute("SELECT count(*) FROM http_cache").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("CREATE TABLE IF NOT EXISTS realtime_vintages (x)")
    finally:
        close_conn()
        os.chmod(fresh_db, 0o644)


def test_query_realtime_vintages_type_hints_resolve():
    """Regression: the return annotation named ``Any`` without importing it,
    so ``typing.get_type_hints`` raised NameError (and mypy failed)."""
    import typing
    from puremacro import _cache_db
    hints = typing.get_type_hints(_cache_db.query_realtime_vintages)
    assert hints["return"] is typing.Any


def test_record_connector_event_swallows_db_errors_and_honours_kill_switch(fresh_db, monkeypatch):
    import warnings
    from puremacro._cache_db import get_conn, record_connector_event
    closed = sqlite3.connect(":memory:")
    closed.close()
    with pytest.warns(UserWarning, match="record_connector_event failed"):
        record_connector_event("bcb", "success", "none", conn=closed)
    conn = get_conn(fresh_db)
    monkeypatch.setenv("PUREMACRO_NARRATIVE_TELEMETRY", "0")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        record_connector_event("bcb", "success", "none")
    assert conn.execute("SELECT count(*) FROM connector_events").fetchone()[0] == 0
    monkeypatch.delenv("PUREMACRO_NARRATIVE_TELEMETRY")
    record_connector_event("bcb", "success", "none")
    assert conn.execute("SELECT count(*) FROM connector_events").fetchone()[0] == 1
