"""F2.1 — cache_read / cache_write roundtrip against the SQLite backend."""
from __future__ import annotations

import time

import pytest


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    yield tmp_path
    M.close_conn()


def test_write_then_read_returns_bytes(fresh_cache):
    from puremacro._http_cache import cache_read, cache_write
    cache_write(fresh_cache, "https://example.com/a", b"hello", content_type="text/plain")
    assert cache_read(fresh_cache, "https://example.com/a") == b"hello"


def test_read_miss_returns_none(fresh_cache):
    from puremacro._http_cache import cache_read
    assert cache_read(fresh_cache, "https://example.com/never-cached") is None


def test_stale_entry_returns_none(fresh_cache):
    from puremacro._http_cache import cache_read, cache_write
    cache_write(fresh_cache, "https://example.com/b", b"old")
    # Force a 1-second TTL and sleep past it.
    time.sleep(1.1)
    assert cache_read(fresh_cache, "https://example.com/b", ttl_seconds=1) is None


def test_overwrite_updates_body(fresh_cache):
    from puremacro._http_cache import cache_read, cache_write
    cache_write(fresh_cache, "https://example.com/c", b"v1")
    cache_write(fresh_cache, "https://example.com/c", b"v2")
    assert cache_read(fresh_cache, "https://example.com/c") == b"v2"


def test_cache_key_stable_for_same_url():
    from puremacro._http_cache import cache_key
    assert cache_key("https://example.com/x") == cache_key("https://example.com/x")
    assert cache_key("https://example.com/x") != cache_key("https://example.com/y")


def test_default_cache_dir_returns_db_parent(monkeypatch, tmp_path):
    from puremacro._http_cache import default_cache_dir
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path / "x"))
    # When env points to a non-.db dir, returns that dir verbatim
    # (callers may pass `cache_dir` and we use it; the db lives at
    # cache_dir/cache.db).
    assert default_cache_dir() == tmp_path / "x"
