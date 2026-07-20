"""F2.2 — AlfredVintageStore put_many / get roundtrip."""
from __future__ import annotations

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


def test_put_many_then_get_returns_same_rows(store):
    rows = pd.DataFrame({
        "series_id":        ["GDPC1", "GDPC1"],
        "observation_date": ["2020-01-01", "2020-04-01"],
        "vintage_date":     ["2020-04-29", "2020-04-29"],
        "value":            [21000.0, 19500.0],
    })
    assert store.put_many(rows) == 2
    out = store.get("GDPC1")
    assert len(out) == 2
    assert set(out.columns) >= {"observation_date", "vintage_date", "value"}


def test_put_single_row(store):
    store.put("UNRATE", "2020-04-01", "2020-05-08", 14.7)
    out = store.get("UNRATE")
    assert len(out) == 1
    assert out["value"].iloc[0] == 14.7


def test_put_or_replace_overwrites(store):
    store.put("GDPC1", "2020-01-01", "2020-04-29", 21000.0)
    store.put("GDPC1", "2020-01-01", "2020-04-29", 21001.5)
    out = store.get("GDPC1")
    assert len(out) == 1
    assert out["value"].iloc[0] == 21001.5


def test_get_missing_series_returns_empty_dataframe(store):
    out = store.get("DOES_NOT_EXIST")
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_observation_and_vintage_dates_returned_as_timestamps(store):
    store.put("GDPC1", "2020-01-01", "2020-04-29", 21000.0)
    out = store.get("GDPC1")
    assert pd.api.types.is_datetime64_any_dtype(out["observation_date"])
    assert pd.api.types.is_datetime64_any_dtype(out["vintage_date"])
