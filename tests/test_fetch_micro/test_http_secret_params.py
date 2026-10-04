"""cached_get must never write an API key to disk or into an error."""
from __future__ import annotations

import json

import pytest

from puremacro.fetch import _http

KEY = "abc123SECRET"


class _Resp:
    def __init__(self, status=200, content=b"ok"):
        self.status_code, self.content = status, content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code} Client Error for url: {self.url}")


class _Requests:
    def __init__(self, status=200):
        self.urls, self.status = [], status

    def get(self, url, headers=None, timeout=None):
        self.urls.append(url)
        r = _Resp(self.status)
        r.url = url
        return r


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(_http, "CACHE_ROOT", tmp_path)
    return tmp_path


def test_key_is_sent_but_not_cached(cache, monkeypatch):
    fake = _Requests()
    monkeypatch.setattr(_http, "requests", fake)
    out = _http.cached_get("https://api.example.gov/data", params={"get": "A,B"},
                           secret_params={"key": KEY})
    assert out == b"ok"
    assert fake.urls == [f"https://api.example.gov/data?get=A,B&key={KEY}"]
    for p in cache.rglob("*"):
        assert KEY not in str(p)
        if p.is_file():
            assert KEY.encode() not in p.read_bytes()
    manifest = json.loads((cache / "_manifest.json").read_text())
    assert any(v["url"] == "https://api.example.gov/data?get=A,B" for v in manifest.values())


def test_cache_hit_ignores_the_key(cache, monkeypatch):
    fake = _Requests()
    monkeypatch.setattr(_http, "requests", fake)
    _http.cached_get("https://h/x", params={"a": 1}, secret_params={"key": KEY})
    _http.cached_get("https://h/x", params={"a": 1}, secret_params={"key": "other"})
    assert len(fake.urls) == 1


def test_errors_are_redacted(cache, monkeypatch):
    monkeypatch.setattr(_http, "requests", _Requests(status=400))
    with pytest.raises(RuntimeError) as ei:
        _http.cached_get("https://h/x", secret_params={"key": KEY})
    assert KEY not in str(ei.value) and "***" in str(ei.value)


def test_params_append_to_an_existing_query(cache, monkeypatch):
    fake = _Requests()
    monkeypatch.setattr(_http, "requests", fake)
    _http.cached_get("https://h/x?y=1", params={"z": 2})
    assert fake.urls == ["https://h/x?y=1&z=2"]


def test_spaces_are_percent_encoded(cache, monkeypatch):
    """Census geography names contain spaces ("public use microdata area")."""
    fake = _Requests()
    monkeypatch.setattr(_http, "requests", fake)
    _http.cached_get("https://h/x", params={"for": "public use microdata area:*"})
    assert fake.urls == ["https://h/x?for=public%20use%20microdata%20area:*"]
