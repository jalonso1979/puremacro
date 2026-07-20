"""F2.5 — connector_health aggregation shape + math + filters."""
from __future__ import annotations

import time

import pandas as pd
import pytest


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("PUREMACRO_NARRATIVE_TELEMETRY", raising=False)
    import puremacro._cache_db as M
    M.close_conn()
    yield tmp_path
    M.close_conn()


_EXPECTED_COLUMNS = {
    "source", "n_total", "n_success", "success_rate",
    "n_fallback", "fallback_rate", "last_seen",
}


def _seed(events):
    """Insert (ts_offset_seconds_ago, source, outcome, fallback_used) rows."""
    from puremacro import _cache_db
    conn = _cache_db.get_conn()
    now = int(time.time())
    for offset, source, outcome, fb in events:
        conn.execute(
            "INSERT INTO connector_events "
            "(ts, source, outcome, fallback_used) VALUES (?, ?, ?, ?)",
            (now - offset, source, outcome, fb),
        )
    conn.commit()


def test_connector_health_returns_expected_columns(fresh_db):
    from puremacro.narrative.sources._telemetry import connector_health
    _seed([(60, "x", "success", "live")])
    df = connector_health(window=pd.Timedelta(days=1))
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == _EXPECTED_COLUMNS


def test_connector_health_aggregation_math(fresh_db):
    from puremacro.narrative.sources._telemetry import connector_health
    _seed([
        # 7 successes (5 live, 2 wayback), 3 failures (1 live, 2 wayback).
        (10, "eu_eurlex", "success",         "live"),
        (20, "eu_eurlex", "success",         "live"),
        (30, "eu_eurlex", "success",         "live"),
        (40, "eu_eurlex", "success",         "live"),
        (50, "eu_eurlex", "success",         "live"),
        (60, "eu_eurlex", "success",         "wayback"),
        (70, "eu_eurlex", "success",         "wayback"),
        (80, "eu_eurlex", "timeout",         "live"),
        (90, "eu_eurlex", "wayback_no_snapshot", "wayback"),
        (100,"eu_eurlex", "other_network_error", "wayback"),
    ])
    df = connector_health(window=pd.Timedelta(hours=1)).set_index("source")
    row = df.loc["eu_eurlex"]
    assert row["n_total"] == 10
    assert row["n_success"] == 7
    assert row["success_rate"] == 0.7
    # n_fallback = events where fallback_used != "live" = 4
    assert row["n_fallback"] == 4
    assert row["fallback_rate"] == 0.4


def test_connector_health_window_filter(fresh_db):
    from puremacro.narrative.sources._telemetry import connector_health
    _seed([
        (60,      "x", "success", "live"),       # 60s ago, within 5min
        (8 * 24 * 3600, "x", "success", "live"), # 8 days ago, outside 7d window
    ])
    df_5min = connector_health(window=pd.Timedelta(minutes=5)).set_index("source")
    assert df_5min.loc["x", "n_total"] == 1
    df_7d = connector_health(window=pd.Timedelta(days=7)).set_index("source")
    assert df_7d.loc["x", "n_total"] == 1   # 8-day-old row still excluded
    df_30d = connector_health(window=pd.Timedelta(days=30)).set_index("source")
    assert df_30d.loc["x", "n_total"] == 2


def test_connector_health_sources_filter(fresh_db):
    from puremacro.narrative.sources._telemetry import connector_health
    _seed([
        (60, "eu_eurlex", "success", "live"),
        (60, "rba",       "success", "live"),
    ])
    df = connector_health(window=pd.Timedelta(days=1), sources=["eu_eurlex"])
    assert set(df["source"]) == {"eu_eurlex"}


def test_connector_health_empty_db_returns_empty_df(fresh_db):
    from puremacro.narrative.sources._telemetry import connector_health
    df = connector_health(window=pd.Timedelta(days=7))
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert set(df.columns) == _EXPECTED_COLUMNS


def test_connector_health_last_seen_is_timestamp(fresh_db):
    from puremacro.narrative.sources._telemetry import connector_health
    _seed([(60, "x", "success", "live")])
    df = connector_health(window=pd.Timedelta(days=1))
    assert pd.api.types.is_datetime64_any_dtype(df["last_seen"])
