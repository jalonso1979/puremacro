"""F2.2 — store.get(vintage_until=...) filters by vintage_date."""
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


def test_vintage_until_filters_correctly(populated_store):
    out = populated_store.get("GDPC1", vintage_until="2020-07-31")
    assert len(out) == 3  # 3 vintages on or before 2020-07-31


def test_vintage_until_strict_bound(populated_store):
    out = populated_store.get("GDPC1", vintage_until="2020-07-30")
    # The boundary vintage IS included (vintage_date <= vintage_until).
    assert len(out) == 3
    out = populated_store.get("GDPC1", vintage_until="2020-07-29")
    assert len(out) == 1


def test_vintage_until_none_returns_all(populated_store):
    out = populated_store.get("GDPC1", vintage_until=None)
    assert len(out) == 4
