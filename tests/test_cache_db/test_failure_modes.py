"""F2.1 — cache_read / cache_write must never raise; warn + None / no-op."""
from __future__ import annotations

import sqlite3
import warnings
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    yield tmp_path
    M.close_conn()


def test_cache_read_on_db_error_returns_none(fresh_cache, monkeypatch):
    from puremacro import _http_cache, _cache_db

    def _raising_get_conn(*a, **kw):
        mock = MagicMock()
        mock.execute.side_effect = sqlite3.OperationalError("simulated DB error")
        return mock

    monkeypatch.setattr(_cache_db, "get_conn", _raising_get_conn)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _http_cache.cache_read(fresh_cache, "https://example.com/x")
    assert result is None
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_cache_write_on_db_error_is_noop(fresh_cache, monkeypatch):
    from puremacro import _http_cache, _cache_db

    def _raising_get_conn(*a, **kw):
        mock = MagicMock()
        mock.execute.side_effect = sqlite3.OperationalError("simulated DB error")
        return mock

    monkeypatch.setattr(_cache_db, "get_conn", _raising_get_conn)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _http_cache.cache_write(fresh_cache, "https://example.com/x", b"body")
    assert any(issubclass(w.category, UserWarning) for w in caught)
    # No exception leaked.
