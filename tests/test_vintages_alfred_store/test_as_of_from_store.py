"""F2.2 — as_of_from_store end-to-end."""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def populated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    from puremacro.vintages import AlfredVintageStore
    s = AlfredVintageStore()
    s.put_many(pd.DataFrame({
        "series_id":        ["GDPC1"] * 4,
        "observation_date": ["2020-01-01", "2020-01-01",
                              "2020-04-01", "2020-04-01"],
        "vintage_date":     ["2020-04-29", "2020-07-30",
                              "2020-07-30", "2020-10-29"],
        "value":            [21000.0, 21010.0, 19500.0, 19520.0],
    }))
    yield s
    M.close_conn()


def test_as_of_returns_latest_vintage_known_at_date(populated_store):
    from puremacro.vintages import as_of_from_store
    # As of 2020-08-01, both observations have a vintage on/before that date.
    s = as_of_from_store("GDPC1", "2020-08-01", populated_store)
    # Expect the 2020-07-30 vintages: GDPC1[2020-01-01] = 21010.0,
    # GDPC1[2020-04-01] = 19500.0.
    assert s.loc[pd.Timestamp("2020-01-01")] == 21010.0
    assert s.loc[pd.Timestamp("2020-04-01")] == 19500.0


def test_as_of_excludes_future_vintages(populated_store):
    from puremacro.vintages import as_of_from_store
    # As of 2020-05-01, only the 2020-04-29 vintage of obs=2020-01-01 is known.
    s = as_of_from_store("GDPC1", "2020-05-01", populated_store)
    assert s.loc[pd.Timestamp("2020-01-01")] == 21000.0
    # The 2020-04-01 observation's earliest vintage is 2020-07-30 > 2020-05-01.
    assert pd.Timestamp("2020-04-01") not in s.index


def test_missing_series_returns_empty_series(populated_store):
    from puremacro.vintages import as_of_from_store
    s = as_of_from_store("DOES_NOT_EXIST", "2020-08-01", populated_store)
    assert isinstance(s, pd.Series)
    assert s.empty
