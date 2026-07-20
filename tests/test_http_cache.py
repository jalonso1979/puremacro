"""Tests for puremacro._http_cache — on-disk cache layer for HTTP responses.

Updated for 0.66.0: storage is SQLite-backed (_cache_db). Flat-file
sidecar tests removed; replaced with SQLite-compatible equivalents.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_cache_db(tmp_path, monkeypatch):
    """Isolate each test: fresh tmp dir + reset singleton connection."""
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    yield
    M.close_conn()


class TestCacheKey:
    def test_key_is_sha256_of_url(self):
        from puremacro._http_cache import cache_key
        k = cache_key("https://example.com/foo")
        # SHA-256 hex is 64 chars
        assert len(k) == 64
        assert all(c in "0123456789abcdef" for c in k)

    def test_key_is_deterministic(self):
        from puremacro._http_cache import cache_key
        assert cache_key("https://x/y") == cache_key("https://x/y")

    def test_different_urls_different_keys(self):
        from puremacro._http_cache import cache_key
        assert cache_key("https://x/y") != cache_key("https://x/z")


class TestCacheRoundtrip:
    def test_write_then_read_returns_bytes(self, tmp_path):
        from puremacro._http_cache import cache_write, cache_read
        cache_write(tmp_path, "https://example.com/a", b"hello world")
        assert cache_read(tmp_path, "https://example.com/a") == b"hello world"

    def test_read_miss_returns_none(self, tmp_path):
        from puremacro._http_cache import cache_read
        assert cache_read(tmp_path, "https://example.com/missing") is None

    def test_read_expired_returns_none(self, tmp_path):
        """Stale entry: backdated via direct DB update; short TTL should miss."""
        import puremacro._cache_db as db_mod
        from puremacro._http_cache import cache_write, cache_read, cache_key
        cache_write(tmp_path, "https://example.com/b", b"old")
        # Backdate fetched_at directly in the SQLite DB.
        conn = db_mod.get_conn(tmp_path / "cache.db")
        old_ts = int(time.time()) - 100  # 100s ago
        conn.execute(
            "UPDATE http_cache SET fetched_at = ? WHERE key = ?",
            (old_ts, cache_key("https://example.com/b")),
        )
        # Read with TTL=10s — should miss.
        assert cache_read(tmp_path, "https://example.com/b", ttl_seconds=10) is None

    def test_write_stores_in_sqlite(self, tmp_path):
        """After cache_write, a single cache.db file exists (no .bin files)."""
        from puremacro._http_cache import cache_write, cache_key
        cache_write(tmp_path, "https://example.com/c", b"data")
        # SQLite DB must exist
        assert (tmp_path / "cache.db").exists()
        # No stale flat files should be created by the new implementation
        assert list(tmp_path.glob("*.bin")) == []


class TestCacheDir:
    def test_default_cache_dir_under_home(self, monkeypatch, tmp_path):
        # Pretend HOME points to tmp_path
        monkeypatch.delenv("PUREMACRO_HTTP_CACHE_DIR", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        from puremacro._http_cache import default_cache_dir
        d = default_cache_dir()
        assert "puremacro" in str(d)
        # 0.66.0: default dir is ~/.cache/puremacro (no /http suffix)
        assert d == tmp_path / ".cache" / "puremacro"

    def test_env_var_overrides_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path / "my-cache"))
        from puremacro._http_cache import default_cache_dir
        d = default_cache_dir()
        assert d == tmp_path / "my-cache"
