"""F2.2 — store failures warn + degrade; never raise to caller.

Note: Python 3.13 makes sqlite3.Connection.execute read-only, so we cannot
use monkeypatch.setattr(conn, "execute", ...) directly. Instead, we patch
_cache_db.get_conn to return a mock connection whose execute/executemany
raise OperationalError.
"""
from __future__ import annotations

import sqlite3
import warnings
from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    from puremacro.vintages import AlfredVintageStore
    s = AlfredVintageStore()
    yield s
    M.close_conn()


def test_get_on_db_error_returns_empty(store, monkeypatch):
    from puremacro import _cache_db

    mock_conn = MagicMock()
    mock_conn.execute.side_effect = sqlite3.OperationalError("simulated")
    monkeypatch.setattr(_cache_db, "get_conn", lambda *a, **kw: mock_conn)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = store.get("GDPC1")
    assert isinstance(out, pd.DataFrame) and out.empty
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_put_many_on_db_error_is_noop(store, monkeypatch):
    from puremacro import _cache_db

    mock_conn = MagicMock()
    mock_conn.executemany.side_effect = sqlite3.OperationalError("simulated")
    monkeypatch.setattr(_cache_db, "get_conn", lambda *a, **kw: mock_conn)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        n = store.put_many(pd.DataFrame({
            "series_id": ["GDPC1"], "observation_date": ["2020-01-01"],
            "vintage_date": ["2020-04-29"], "value": [21000.0],
        }))
    assert n == 0
    assert any(issubclass(w.category, UserWarning) for w in caught)
