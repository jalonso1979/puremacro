"""F2.2 — fetch_fred_alfred(store=, refresh=) gap-fill semantics."""
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


def _mock_api_rows():
    """Synthetic ALFRED rows in the shape fetch_fred_alfred returns:
       columns [date, vintage, value]."""
    return pd.DataFrame({
        "date":    [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-01")],
        "vintage": [pd.Timestamp("2020-04-29"), pd.Timestamp("2020-07-30")],
        "value":   [21000.0, 19500.0],
    })


def test_with_empty_store_calls_api_and_populates(store, monkeypatch):
    from puremacro.fetch import _classic

    calls = []

    def _fake_api(series_id, *, timeout):
        calls.append(series_id)
        return _mock_api_rows()

    # Patch the underlying live-API helper (the one that does the HTTP).
    # The new code path calls the original raw fetch when needed.
    monkeypatch.setattr(_classic, "_fetch_fred_alfred_raw_api", _fake_api,
                        raising=False)

    df = _classic.fetch_fred_alfred("GDPC1", store=store)
    assert calls == ["GDPC1"]
    assert len(df) == 2
    assert store.has_series("GDPC1")


def test_refetch_when_store_has_no_series(store, monkeypatch):
    from puremacro.fetch import _classic

    calls = []
    monkeypatch.setattr(
        _classic,
        "_fetch_fred_alfred_raw_api",
        lambda series_id, *, timeout: (calls.append(series_id) or _mock_api_rows()),
        raising=False,
    )
    _classic.fetch_fred_alfred("GDPC1", store=store)
    assert len(calls) == 1


def test_no_refetch_when_store_has_data(store, monkeypatch):
    from puremacro.fetch import _classic
    # Pre-populate store.
    store.put_many(pd.DataFrame({
        "series_id":        ["GDPC1", "GDPC1"],
        "observation_date": ["2020-01-01", "2020-04-01"],
        "vintage_date":     ["2020-04-29", "2020-07-30"],
        "value":            [21000.0, 19500.0],
    }))
    calls = []
    monkeypatch.setattr(
        _classic,
        "_fetch_fred_alfred_raw_api",
        lambda series_id, *, timeout: (calls.append(series_id) or _mock_api_rows()),
        raising=False,
    )
    df = _classic.fetch_fred_alfred("GDPC1", store=store)
    assert calls == []  # store hit; no API call
    assert len(df) == 2


def test_refresh_true_forces_api(store, monkeypatch):
    from puremacro.fetch import _classic
    store.put_many(pd.DataFrame({
        "series_id":        ["GDPC1"],
        "observation_date": ["2020-01-01"],
        "vintage_date":     ["2020-04-29"],
        "value":            [21000.0],
    }))
    calls = []
    monkeypatch.setattr(
        _classic,
        "_fetch_fred_alfred_raw_api",
        lambda series_id, *, timeout: (calls.append(series_id) or _mock_api_rows()),
        raising=False,
    )
    _classic.fetch_fred_alfred("GDPC1", store=store, refresh=True)
    assert calls == ["GDPC1"]


def test_no_store_no_behavior_change(monkeypatch):
    """Backwards-compat: calling without store= must behave exactly
    like 0.65.0 (use the live API)."""
    from puremacro.fetch import _classic
    calls = []
    monkeypatch.setattr(
        _classic,
        "_fetch_fred_alfred_raw_api",
        lambda series_id, *, timeout: (calls.append(series_id) or _mock_api_rows()),
        raising=False,
    )
    df = _classic.fetch_fred_alfred("GDPC1")
    assert calls == ["GDPC1"]
    assert list(df.columns) == ["date", "vintage", "value"]
