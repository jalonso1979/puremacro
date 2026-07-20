"""F2.5 — log_event roundtrip, validation, failure mode, kill-switch."""
from __future__ import annotations

import sqlite3
import warnings
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("PUREMACRO_NARRATIVE_TELEMETRY", raising=False)
    import puremacro._cache_db as M
    M.close_conn()
    yield tmp_path
    M.close_conn()


def test_log_event_inserts_row(fresh_db):
    from puremacro.narrative.sources._telemetry import log_event
    from puremacro import _cache_db
    log_event(source="eu_eurlex", outcome="success", fallback_used="live")
    conn = _cache_db.get_conn()
    rows = conn.execute(
        "SELECT source, outcome, fallback_used FROM connector_events"
    ).fetchall()
    assert rows == [("eu_eurlex", "success", "live")]


def test_log_event_fallback_used_defaults_to_none(fresh_db):
    from puremacro.narrative.sources._telemetry import log_event
    from puremacro import _cache_db
    log_event(source="beige_book", outcome="parser_schema_mismatch")
    conn = _cache_db.get_conn()
    rows = conn.execute(
        "SELECT fallback_used FROM connector_events"
    ).fetchall()
    assert rows == [("none",)]


def test_log_event_rejects_invalid_outcome(fresh_db):
    from puremacro.narrative.sources._telemetry import log_event
    with pytest.raises(ValueError, match="outcome"):
        log_event(source="x", outcome="not_a_valid_outcome",
                  fallback_used="live")


def test_log_event_rejects_invalid_fallback_used(fresh_db):
    from puremacro.narrative.sources._telemetry import log_event
    with pytest.raises(ValueError, match="fallback_used"):
        log_event(source="x", outcome="success",
                  fallback_used="not_a_valid_fallback")


def test_log_event_db_failure_warns_and_no_ops(fresh_db, monkeypatch):
    from puremacro.narrative.sources._telemetry import log_event
    from puremacro import _cache_db

    # Replace get_conn with a mock whose execute raises (Python 3.13's
    # sqlite3.Connection.execute is read-only, so we patch the resolver).
    fake_conn = MagicMock()
    fake_conn.execute.side_effect = sqlite3.OperationalError("simulated")
    monkeypatch.setattr(_cache_db, "get_conn", lambda *a, **kw: fake_conn)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log_event(source="x", outcome="success", fallback_used="live")
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_log_event_kill_switch_skips_insert(fresh_db, monkeypatch):
    monkeypatch.setenv("PUREMACRO_NARRATIVE_TELEMETRY", "0")
    from puremacro.narrative.sources._telemetry import log_event
    from puremacro import _cache_db
    log_event(source="x", outcome="success", fallback_used="live")
    conn = _cache_db.get_conn()
    rows = conn.execute(
        "SELECT COUNT(*) FROM connector_events"
    ).fetchone()
    assert rows[0] == 0


def test_telemetry_enabled_reflects_env(monkeypatch):
    from puremacro.narrative.sources._telemetry import telemetry_enabled
    monkeypatch.delenv("PUREMACRO_NARRATIVE_TELEMETRY", raising=False)
    assert telemetry_enabled() is True
    monkeypatch.setenv("PUREMACRO_NARRATIVE_TELEMETRY", "0")
    assert telemetry_enabled() is False
    monkeypatch.setenv("PUREMACRO_NARRATIVE_TELEMETRY", "1")
    assert telemetry_enabled() is True
