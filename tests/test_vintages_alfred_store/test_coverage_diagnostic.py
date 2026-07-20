"""F2.2 — coverage() diagnostic returns expected dict."""
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


def test_coverage_returns_none_for_missing_series(store):
    assert store.coverage("DOES_NOT_EXIST") is None


def test_coverage_returns_expected_fields(store):
    store.put_many(pd.DataFrame({
        "series_id":        ["GDPC1"] * 3,
        "observation_date": ["2020-01-01", "2020-04-01", "2020-07-01"],
        "vintage_date":     ["2020-04-29", "2020-07-30", "2020-10-29"],
        "value":            [21000.0, 19500.0, 20100.0],
    }))
    c = store.coverage("GDPC1")
    assert c is not None
    assert c["n_rows"] == 3
    assert c["first_obs"] == pd.Timestamp("2020-01-01")
    assert c["last_obs"] == pd.Timestamp("2020-07-01")
    assert c["first_vintage"] == pd.Timestamp("2020-04-29")
    assert c["last_vintage"] == pd.Timestamp("2020-10-29")
    assert c["n_vintages"] == 3
