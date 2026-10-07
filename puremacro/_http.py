"""Shared HTTP helpers for puremacro fetchers and connectors.

This module is the single canonical home of ``safe_get_bytes``,
``safe_get_text``, ``safe_get_json`` and the supporting ``USER_AGENT``
+ ``DEFAULT_TIMEOUT`` constants. Promoted from
``puremacro.narrative.sources._http`` in 0.6.0 so all fetchers
(narrative connectors, ``puremacro.fetch.*``, instrument loaders)
share one hardened path.

See ``puremacro/narrative/sources/RETRY_POLICY.md`` for the
contract every consumer adheres to: 30s default timeout, one-shot
SSL fallback for older endpoints with stale CA bundles, optional
keyword-only ``user_agent=`` override (added in 0.4.1) for endpoints
behind a WAF that blocks the default agent string.
"""
from __future__ import annotations

import json
from puremacro._optional_stdlib import ssl
import urllib.error
import urllib.request


USER_AGENT = "Mozilla/5.0 (puremacro/narrative)"
DEFAULT_TIMEOUT = 30.0


def _gunzip_if_encoded(body: bytes, content_encoding: str | None) -> bytes:
    """Undo a *transport* gzip (``Content-Encoding: gzip``) that urllib leaves in place.

    Only the transport encoding is undone: a ``.gz`` file served as
    ``application/gzip`` with no ``Content-Encoding`` header is returned as the
    bytes the caller asked for. Servers that honour ``Accept-Encoding: gzip``
    (World Bank, BIS, ECB) shrink a response 10-80x, which is the difference
    between a 13 MB and a 170 KB pull on a tablet connection.
    """
    if body[:2] == b"\x1f\x8b" and (content_encoding or "").lower() in ("gzip", "x-gzip"):
        import gzip
        return gzip.decompress(body)
    return body


def _request(url: str, timeout: float, user_agent: str | None = None,
             headers: dict | None = None) -> bytes:
    ua = user_agent or USER_AGENT
    hdrs = {"User-Agent": ua}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _gunzip_if_encoded(resp.read(), resp.headers.get("Content-Encoding"))
    except urllib.error.HTTPError:
        # HTTPError is a *subclass* of URLError, so without this branch a
        # 404 / 429 / 500 fell into the SSL fallback below and was
        # re-requested with verification off — two round trips per error,
        # and on a rate-limited endpoint the retry is itself another
        # request against the limit. The server answered; there is
        # nothing wrong with the certificate. Propagate.
        raise
    except (urllib.error.URLError, ssl.SSLError):
        # One-shot fallback: some public endpoints (older OECD / IMF /
        # ministry sites) ship certificates that Python's bundled CA
        # store does not validate. Retry once with default verification enforced.
        # See RETRY_POLICY.md §3 for why we do not loop further.
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return _gunzip_if_encoded(resp.read(), resp.headers.get("Content-Encoding"))


def _fetch(url: str, timeout: float, user_agent: str | None, headers: dict | None) -> bytes:
    """Call ``_request`` without the ``headers=`` keyword when there are none.

    ``_request`` is the seam tests and connectors monkeypatch, and the
    0.4.1 contract is ``(url, timeout, user_agent=None)``; a call that only
    adds ``headers=`` when it has some keeps every such patch working.
    """
    if headers:
        return _request(url, timeout, user_agent=user_agent, headers=headers)
    return _request(url, timeout, user_agent=user_agent)


def safe_get_bytes(url: str, timeout: float = DEFAULT_TIMEOUT,
                   *, user_agent: str | None = None,
                   headers: dict | None = None) -> bytes:
    """Fetch ``url`` and return raw bytes. SSL fallback applied once.

    ``user_agent`` overrides the default ``Mozilla/5.0 (puremacro/narrative)``
    UA — needed for endpoints behind a WAF that blocks scripted clients.
    ``headers`` adds request headers (an ``Accept`` for SDMX-CSV, an
    ``Accept-Encoding: gzip`` for providers that honour it; the transport
    gzip is undone before the bytes are returned).
    """
    return _fetch(url, timeout, user_agent, headers)


def safe_get_text(url: str, timeout: float = DEFAULT_TIMEOUT,
                  *, user_agent: str | None = None,
                  headers: dict | None = None) -> str:
    """Fetch ``url`` and return UTF-8 text (decode errors ignored).

    See ``safe_get_bytes`` for the ``user_agent=`` / ``headers=`` semantics
    and ``RETRY_POLICY.md §7`` for the WAF-bypass pattern.
    """
    return _fetch(url, timeout, user_agent, headers).decode(
        "utf-8", errors="ignore",
    )


def safe_get_json(url: str, timeout: float = DEFAULT_TIMEOUT,
                  *, user_agent: str | None = None,
                  headers: dict | None = None) -> dict:
    """Fetch ``url`` and return decoded JSON.

    Empty / whitespace-only bodies return ``{}`` rather than raising,
    matching the existing API-connector behaviour (e.g. GDELT v2 rate
    limits sometimes return blank pages).

    See ``safe_get_bytes`` for the ``user_agent=`` / ``headers=`` semantics
    and ``RETRY_POLICY.md §7`` for the WAF-bypass pattern.
    """
    text = safe_get_text(url, timeout, user_agent=user_agent, headers=headers)
    if not text.strip():
        return {}
    return json.loads(text)


def post_json(url: str, payload: dict, *, timeout: float = DEFAULT_TIMEOUT,
              headers: dict | None = None) -> dict:
    """POST ``payload`` as JSON and return the decoded JSON response.

    urllib-only (Pyodide-safe). HTTP errors propagate (not retried); a
    transport/SSL error retries once with verification off, matching
    ``_request``. Used by the local-LLM HTTP engine (Ollama / OpenAI-compatible).
    """
    data = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError:
        raise
    except (urllib.error.URLError, ssl.SSLError):
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Opt-in cached + rate-limited variants (0.60.0)
#
# Callers who want caching switch from ``safe_get_text`` to
# ``safe_get_text_cached``. The existing helpers are unchanged.
# ---------------------------------------------------------------------------

import os
import time
import urllib.parse

from ._http_cache import default_cache_dir, cache_read, cache_write


# Per-host last-fetch timestamps (monotonic seconds).
_HOST_LAST_FETCH: dict[str, float] = {}


def _reset_throttle_state() -> None:
    """Clear per-host throttle state. Test helper."""
    _HOST_LAST_FETCH.clear()


def _throttle(host: str, rate_limit_seconds: float) -> None:
    """Sleep until ``rate_limit_seconds`` have elapsed since the last
    fetch to ``host``. No-op on first call to a host."""
    if rate_limit_seconds <= 0:
        return
    now = time.monotonic()
    last = _HOST_LAST_FETCH.get(host)
    if last is not None:
        wait = rate_limit_seconds - (now - last)
        if wait > 0:
            time.sleep(wait)
    _HOST_LAST_FETCH[host] = time.monotonic()


def _host_of(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower()


def _cache_identity(url: str, headers: dict | None) -> str:
    """The cache key: the URL plus any header that changes the *body*.

    An ``Accept`` header turns an SDMX endpoint from XML into CSV, so the
    same URL fetched with and without it must not share a cache entry.
    ``Accept-Encoding`` only changes the transport and is left out.
    """
    if not headers:
        return url
    rest = sorted((k.lower(), str(v)) for k, v in headers.items()
                  if k.lower() != "accept-encoding")
    if not rest:
        return url
    return url + "\n" + "\n".join(f"{k}: {v}" for k, v in rest)


def safe_get_bytes_cached(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
    *,
    user_agent: str | None = None,
    headers: dict | None = None,
    ttl_seconds: int = 30 * 24 * 3600,
    rate_limit_seconds: float = 0.5,
    refresh: bool = False,
) -> bytes:
    """Like ``safe_get_bytes`` but cached on disk.

    Cache root: ``$PUREMACRO_HTTP_CACHE_DIR`` or ``~/.cache/puremacro/http``.
    Bypass entirely with ``PUREMACRO_HTTP_NO_CACHE=1``. ``refresh=True``
    skips the cached copy for this call and stores the fresh response, which
    is how a fetcher offers a "latest release" switch on top of a cache that
    has no expiry of its own.
    """
    bypass = os.environ.get("PUREMACRO_HTTP_NO_CACHE") == "1"
    cache_dir = default_cache_dir()
    identity = _cache_identity(url, headers)
    if not bypass and not refresh:
        hit = cache_read(cache_dir, identity, ttl_seconds=ttl_seconds)
        if hit is not None:
            return hit
    _throttle(_host_of(url), rate_limit_seconds)
    body = _fetch(url, timeout, user_agent, headers)
    if not bypass:
        cache_write(cache_dir, identity, body)
    return body


def safe_get_text_cached(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
    *,
    user_agent: str | None = None,
    headers: dict | None = None,
    ttl_seconds: int = 30 * 24 * 3600,
    rate_limit_seconds: float = 0.5,
    refresh: bool = False,
) -> str:
    """Like ``safe_get_text`` but cached on disk. See
    ``safe_get_bytes_cached`` for env-var semantics.
    """
    return safe_get_bytes_cached(
        url, timeout,
        user_agent=user_agent,
        headers=headers,
        ttl_seconds=ttl_seconds,
        rate_limit_seconds=rate_limit_seconds,
        refresh=refresh,
    ).decode("utf-8", errors="ignore")


__all__ = [
    "USER_AGENT", "DEFAULT_TIMEOUT",
    "safe_get_bytes", "safe_get_text", "safe_get_json",
    "post_json",
    "safe_get_bytes_cached", "safe_get_text_cached",
]
