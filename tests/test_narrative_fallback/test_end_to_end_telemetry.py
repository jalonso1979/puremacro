"""F2.4 — fetch_with_fallback emits the expected telemetry events."""
from __future__ import annotations

import socket

import pytest


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("PUREMACRO_NARRATIVE_TELEMETRY", raising=False)
    import puremacro._cache_db as M
    M.close_conn()
    yield tmp_path
    M.close_conn()


def test_happy_path_emits_single_success_event(fresh_db, monkeypatch):
    from puremacro.narrative.sources import _fallback
    from puremacro import _cache_db
    monkeypatch.setattr(
        _fallback, "_stage_live",
        lambda url, *, timeout, use_cache: "<html>ok</html>",
    )
    _fallback.fetch_with_fallback(
        "https://example.com/a", policy=("live",), source="eu_eurlex",
    )
    rows = _cache_db.get_conn().execute(
        "SELECT source, outcome, fallback_used FROM connector_events"
    ).fetchall()
    assert rows == [("eu_eurlex", "success", "live")]


def test_live_fails_wayback_succeeds_emits_two_events(fresh_db, monkeypatch):
    from puremacro.narrative.sources import _fallback
    from puremacro import _cache_db

    def _live_raises(url, *, timeout, use_cache):
        raise socket.timeout("boom")

    monkeypatch.setattr(_fallback, "_stage_live", _live_raises)
    monkeypatch.setattr(
        _fallback, "_stage_wayback",
        lambda url, *, timeout, use_cache: "<html>wb</html>",
    )
    _fallback.fetch_with_fallback(
        "https://example.com/a", policy=("live", "wayback"), source="eu_eurlex",
    )
    rows = _cache_db.get_conn().execute(
        "SELECT outcome, fallback_used FROM connector_events ORDER BY ts, fallback_used"
    ).fetchall()
    assert ("timeout", "live") in rows
    assert ("success", "wayback") in rows


def test_all_stages_fail_emits_failures_only(fresh_db, monkeypatch):
    from puremacro.narrative.sources import _fallback
    from puremacro import _cache_db

    monkeypatch.setattr(
        _fallback, "_stage_live",
        lambda url, *, timeout, use_cache: (_ for _ in ()).throw(socket.timeout("a")),
    )
    monkeypatch.setattr(
        _fallback, "_stage_wayback",
        lambda url, *, timeout, use_cache: (_ for _ in ()).throw(socket.timeout("b")),
    )
    with pytest.raises(_fallback.FallbackExhaustedError):
        _fallback.fetch_with_fallback(
            "https://example.com/a", policy=("live", "wayback"), source="eu_eurlex",
        )
    rows = _cache_db.get_conn().execute(
        "SELECT outcome FROM connector_events"
    ).fetchall()
    outcomes = [r[0] for r in rows]
    assert outcomes.count("timeout") == 2
    assert "success" not in outcomes
