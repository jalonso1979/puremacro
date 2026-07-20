"""Tests for the shared Wayback Machine CDX helper."""
from __future__ import annotations

import pytest


class TestWaybackSnapshotUrl:
    def test_returns_snapshot_on_cdx_hit(self, monkeypatch):
        from puremacro.narrative.sources import _wayback
        cdx_response = '[["statuscode","timestamp"],["200","20240115123456"]]'
        monkeypatch.setattr(
            _wayback, "safe_get_text",
            lambda url, *a, **kw: cdx_response,
        )
        result = _wayback.wayback_snapshot_url("https://example.com/foo")
        assert result == (
            "https://web.archive.org/web/20240115123456/"
            "https://example.com/foo"
        )

    def test_returns_none_on_empty_cdx(self, monkeypatch):
        from puremacro.narrative.sources import _wayback
        monkeypatch.setattr(_wayback, "safe_get_text",
                            lambda url, *a, **kw: "")
        assert _wayback.wayback_snapshot_url("https://example.com/foo") is None

    def test_returns_none_on_malformed_json(self, monkeypatch):
        from puremacro.narrative.sources import _wayback
        monkeypatch.setattr(_wayback, "safe_get_text",
                            lambda url, *a, **kw: "not json")
        assert _wayback.wayback_snapshot_url("https://example.com/foo") is None

    def test_url_encodes_target(self, monkeypatch):
        """The CDX query must URL-encode the target so colons/slashes
        don't get interpreted as part of the CDX URL itself."""
        from puremacro.narrative.sources import _wayback
        seen_urls: list[str] = []
        monkeypatch.setattr(
            _wayback, "safe_get_text",
            lambda url, *a, **kw: (seen_urls.append(url), "")[1] or "",
        )
        _wayback.wayback_snapshot_url("https://example.com/foo?bar=baz")
        assert seen_urls, "safe_get_text should have been called once"
        assert "%3A" in seen_urls[0]  # encoded colon from https:
        assert "%2F" in seen_urls[0]  # encoded slash
        assert "%3F" in seen_urls[0]  # encoded question-mark

    def test_returns_none_on_network_error(self, monkeypatch):
        from puremacro.narrative.sources import _wayback
        def boom(url, *a, **kw):
            raise ConnectionError("no network")
        monkeypatch.setattr(_wayback, "safe_get_text", boom)
        assert _wayback.wayback_snapshot_url("https://example.com/foo") is None
