"""F2.4 — fetch_with_fallback happy path, exhausted, classify."""
from __future__ import annotations

import socket
import ssl
import urllib.error

import pytest


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("PUREMACRO_NARRATIVE_TELEMETRY", raising=False)
    import puremacro._cache_db as M
    M.close_conn()
    yield tmp_path
    M.close_conn()


def test_live_stage_succeeds(fresh_db, monkeypatch):
    from puremacro.narrative.sources import _fallback
    monkeypatch.setattr(
        _fallback, "_stage_live",
        lambda url, *, timeout, use_cache: "<html>live body</html>",
    )
    body = _fallback.fetch_with_fallback(
        "https://example.com/", policy=("live",), source="x",
    )
    assert body == "<html>live body</html>"


def test_live_fails_wayback_succeeds(fresh_db, monkeypatch):
    from puremacro.narrative.sources import _fallback

    def _live_raises(url, *, timeout, use_cache):
        raise urllib.error.HTTPError(url, 500, "boom", {}, None)

    monkeypatch.setattr(_fallback, "_stage_live", _live_raises)
    monkeypatch.setattr(
        _fallback, "_stage_wayback",
        lambda url, *, timeout, use_cache: "<html>wayback body</html>",
    )
    body = _fallback.fetch_with_fallback(
        "https://example.com/", policy=("live", "wayback"), source="x",
    )
    assert body == "<html>wayback body</html>"


def test_all_stages_fail_raises_exhausted(fresh_db, monkeypatch):
    from puremacro.narrative.sources import _fallback

    def _raises(url, *, timeout, use_cache=None):
        raise socket.timeout("boom")

    monkeypatch.setattr(_fallback, "_stage_live", _raises)
    monkeypatch.setattr(_fallback, "_stage_wayback", _raises)
    with pytest.raises(_fallback.FallbackExhaustedError):
        _fallback.fetch_with_fallback(
            "https://example.com/", policy=("live", "wayback"), source="x",
        )


@pytest.mark.parametrize("exc, expected_outcome", [
    (socket.timeout("t"), "timeout"),
    (urllib.error.HTTPError("u", 404, "nf", {}, None), "404"),
    (urllib.error.HTTPError("u", 500, "isr", {}, None), "server_5xx"),
    (urllib.error.HTTPError("u", 503, "unavail", {}, None), "server_5xx"),
    (ssl.SSLError("ssl boom"), "ssl_fail"),
    (TimeoutError("t"), "timeout"),
    (RuntimeError("other"), "other_network_error"),
])
def test_classify_maps_exceptions(exc, expected_outcome):
    from puremacro.narrative.sources._fallback import _classify
    assert _classify(exc) == expected_outcome


def test_classify_wayback_no_snapshot():
    from puremacro.narrative.sources._fallback import (
        _classify, FallbackStageUnavailable,
    )
    assert _classify(
        FallbackStageUnavailable("wayback_no_snapshot")
    ) == "wayback_no_snapshot"


def test_classify_playwright_unavailable():
    from puremacro.narrative.sources._fallback import (
        _classify, FallbackStageUnavailable,
    )
    assert _classify(
        FallbackStageUnavailable("playwright_unavailable")
    ) == "playwright_unavailable"
