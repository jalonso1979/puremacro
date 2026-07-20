"""F2.1 — http_list_urls / http_cache_size_bytes / http_cache_clear."""
from __future__ import annotations

import time

import pandas as pd
import pytest


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    yield tmp_path
    M.close_conn()


def _seed(tmp_path, entries):
    """Insert (url, body) pairs via cache_write."""
    from puremacro._http_cache import cache_write
    for url, body in entries:
        cache_write(tmp_path, url, body)


def test_http_list_urls_returns_sorted(fresh_cache):
    _seed(fresh_cache, [
        ("https://b.example/x", b"x"),
        ("https://a.example/y", b"y"),
        ("https://c.example/z", b"z"),
    ])
    from puremacro.cache import http_list_urls
    assert http_list_urls() == [
        "https://a.example/y",
        "https://b.example/x",
        "https://c.example/z",
    ]


def test_http_cache_size_bytes(fresh_cache):
    _seed(fresh_cache, [("https://a/", b"x" * 100), ("https://b/", b"y" * 200)])
    from puremacro.cache import http_cache_size_bytes
    assert http_cache_size_bytes() == 300


def test_http_cache_clear_all(fresh_cache):
    _seed(fresh_cache, [("https://a/", b"x"), ("https://b/", b"y")])
    from puremacro.cache import http_cache_clear, http_list_urls
    assert http_cache_clear() == 2
    assert http_list_urls() == []


def test_http_cache_clear_older_than(fresh_cache):
    from puremacro._http_cache import cache_write
    from puremacro.cache import http_cache_clear, http_list_urls
    cache_write(fresh_cache, "https://old/", b"o")
    time.sleep(2.1)
    cache_write(fresh_cache, "https://new/", b"n")
    deleted = http_cache_clear(older_than=pd.Timedelta(seconds=1))
    assert deleted == 1
    assert http_list_urls() == ["https://new/"]


def test_disk_cache_helpers_still_present():
    """Verify the existing disk_cache / disk_cache_path API in cache.py
    is untouched by the http_* additions."""
    import puremacro.cache as C
    assert callable(C.disk_cache)
    assert callable(C.disk_cache_path)
