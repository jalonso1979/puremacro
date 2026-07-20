"""F2.2 — has_series / series_list introspection."""
from __future__ import annotations

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


def test_has_series_false_when_empty(store):
    assert store.has_series("GDPC1") is False


def test_has_series_true_after_put(store):
    store.put("GDPC1", "2020-01-01", "2020-04-29", 21000.0)
    assert store.has_series("GDPC1") is True


def test_series_list_returns_sorted_distinct(store):
    store.put("UNRATE", "2020-04-01", "2020-05-08", 14.7)
    store.put("GDPC1", "2020-01-01", "2020-04-29", 21000.0)
    store.put("GDPC1", "2020-04-01", "2020-07-30", 19500.0)
    assert store.series_list() == ["GDPC1", "UNRATE"]
