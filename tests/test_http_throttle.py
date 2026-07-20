"""Tests for per-host rate limit + cached HTTP fetchers."""
from __future__ import annotations

import time

import pytest


class TestPerHostThrottle:
    def test_first_call_no_sleep(self, monkeypatch):
        from puremacro import _http
        sleeps = []
        monkeypatch.setattr(_http.time, "sleep", lambda s: sleeps.append(s))
        _http._reset_throttle_state()  # test helper
        _http._throttle("example.com", 0.5)
        assert sleeps == []  # first call to a host: no sleep

    def test_back_to_back_calls_sleep(self, monkeypatch):
        from puremacro import _http
        sleeps = []
        # Fake monotonic that advances deterministically by 0.05s per call.
        clock = {"t": 100.0}
        monkeypatch.setattr(_http.time, "monotonic", lambda: clock["t"])
        monkeypatch.setattr(_http.time, "sleep",
                            lambda s: (sleeps.append(s), clock.__setitem__("t", clock["t"] + s)))
        _http._reset_throttle_state()
        _http._throttle("a.com", 0.5)   # first → no sleep
        _http._throttle("a.com", 0.5)   # second → must sleep ~0.5s
        assert sleeps and sleeps[-1] > 0

    def test_different_hosts_dont_interfere(self, monkeypatch):
        from puremacro import _http
        sleeps = []
        clock = {"t": 100.0}
        monkeypatch.setattr(_http.time, "monotonic", lambda: clock["t"])
        monkeypatch.setattr(_http.time, "sleep",
                            lambda s: sleeps.append(s))
        _http._reset_throttle_state()
        _http._throttle("a.com", 0.5)
        _http._throttle("b.com", 0.5)   # different host — no sleep
        assert sleeps == []


class TestCachedFetchers:
    def test_cached_get_text_hits_network_first_call(self, monkeypatch, tmp_path):
        from puremacro import _http
        monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
        monkeypatch.delenv("PUREMACRO_HTTP_NO_CACHE", raising=False)
        calls = []
        monkeypatch.setattr(_http, "_request",
                            lambda url, timeout, user_agent=None:
                            calls.append(url) or b"<html>x</html>")
        out = _http.safe_get_text_cached("https://test.invalid/page",
                                          rate_limit_seconds=0)
        assert out == "<html>x</html>"
        assert calls == ["https://test.invalid/page"]

    def test_cached_get_text_uses_cache_on_second_call(self, monkeypatch, tmp_path):
        from puremacro import _http
        monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
        monkeypatch.delenv("PUREMACRO_HTTP_NO_CACHE", raising=False)
        calls = []
        monkeypatch.setattr(_http, "_request",
                            lambda url, timeout, user_agent=None:
                            calls.append(url) or b"<html>x</html>")
        _http.safe_get_text_cached("https://test.invalid/page",
                                    rate_limit_seconds=0)
        _http.safe_get_text_cached("https://test.invalid/page",
                                    rate_limit_seconds=0)
        assert len(calls) == 1  # second call served from cache

    def test_no_cache_env_var_bypasses(self, monkeypatch, tmp_path):
        from puremacro import _http
        monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("PUREMACRO_HTTP_NO_CACHE", "1")
        calls = []
        monkeypatch.setattr(_http, "_request",
                            lambda url, timeout, user_agent=None:
                            calls.append(url) or b"<html>x</html>")
        _http.safe_get_text_cached("https://test.invalid/page",
                                    rate_limit_seconds=0)
        _http.safe_get_text_cached("https://test.invalid/page",
                                    rate_limit_seconds=0)
        assert len(calls) == 2  # cache bypassed

    def test_cached_get_bytes_returns_raw_bytes(self, monkeypatch, tmp_path):
        from puremacro import _http
        monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
        monkeypatch.delenv("PUREMACRO_HTTP_NO_CACHE", raising=False)
        monkeypatch.setattr(_http, "_request",
                            lambda url, timeout, user_agent=None:
                            b"\x00\x01\x02\x03")
        out = _http.safe_get_bytes_cached("https://test.invalid/blob",
                                           rate_limit_seconds=0)
        assert out == b"\x00\x01\x02\x03"

    def test_fetch_failure_not_cached(self, monkeypatch, tmp_path):
        """Transient failures must not poison the cache."""
        from puremacro import _http
        import urllib.error
        monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
        monkeypatch.delenv("PUREMACRO_HTTP_NO_CACHE", raising=False)
        def boom(url, timeout, user_agent=None):
            raise urllib.error.URLError("boom")
        monkeypatch.setattr(_http, "_request", boom)
        with pytest.raises(urllib.error.URLError):
            _http.safe_get_text_cached("https://test.invalid/page",
                                        rate_limit_seconds=0)
        # Nothing written to cache dir
        assert list(tmp_path.glob("*.bin")) == []
