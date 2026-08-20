"""HTTP that works where there are no sockets.

Under Pyodide there is no TCP stack, so `urllib` and `requests` both
fail — which is why every ``puremacro.fetch.*`` call is dead on an iPad
even though the estimator core imports perfectly. The browser can still
make requests; it just does it through JavaScript. This module is the
bridge, plus a one-call switch that routes the package's existing
fetchers over it:

    >>> import puremacro as pm
    >>> runtime.enable_browser_network()      # doctest: +SKIP
    'js-fetch'
    >>> from puremacro.fetch import fetch_xrate_monthly
    >>> fx = fetch_xrate_monthly(["MEX"])        # doctest: +SKIP

Nothing in ``fetch`` or ``narrative.sources`` changes: the switch swaps
the two chokepoints those modules already funnel through — the urllib
call in :mod:`puremacro._http` and the ``requests`` module object in
:mod:`puremacro.fetch._http`.

Two browser limits are worth knowing before you rely on this:

**CORS.** The browser refuses cross-origin responses that don't carry
``Access-Control-Allow-Origin``. Some public statistical endpoints send
it (OECD SDMX, the FRED CSV path); many don't (most WAF-fronted
government sites). A blocked request raises :class:`TransportError`
naming CORS as the likely cause. Pass ``proxy=`` to route through a
CORS proxy you control.

**Timeouts and headers.** A synchronous ``XMLHttpRequest`` on the main
thread cannot set a timeout, and the browser forbids setting
``User-Agent`` and friends. Both are accepted and ignored rather than
raising, so the existing call sites keep working unmodified — which
also means the WAF-bypass user-agent trick in
``narrative/sources/RETRY_POLICY.md`` §7 does not work in a browser.
"""
from __future__ import annotations

import json as _json
from dataclasses import dataclass

from puremacro.runtime._capabilities import capabilities

__all__ = [
    "TransportError",
    "available",
    "get_bytes",
    "get_text",
    "get_json",
    "enable_browser_network",
    "disable_browser_network",
]


class TransportError(RuntimeError):
    """A request failed, or no transport is available at all."""


# Headers the browser will not let script set on an XHR. Sending them
# raises a JS SecurityError, so they are dropped with the rest of the
# request left intact.
_FORBIDDEN_HEADERS = frozenset({
    "accept-charset", "accept-encoding", "access-control-request-headers",
    "access-control-request-method", "connection", "content-length",
    "cookie", "cookie2", "date", "dnt", "expect", "host", "keep-alive",
    "origin", "referer", "te", "trailer", "transfer-encoding", "upgrade",
    "user-agent", "via",
})

_PROXY: str | None = None


def available() -> str:
    """Which transport this environment can use.

    Returns ``"sockets"``, ``"js-fetch"`` or ``"none"``.
    """
    caps = capabilities()
    if caps.sockets:
        return "sockets"
    if caps.js_fetch:
        return "js-fetch"
    return "none"


def _proxied(url: str) -> str:
    return f"{_PROXY}{url}" if _PROXY else url


def _xhr_get(url: str, headers: dict | None = None) -> bytes:
    """Synchronous XHR, returning raw bytes. Pyodide only."""
    try:  # pragma: no cover - only reachable under Pyodide
        from js import XMLHttpRequest  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise TransportError(
            "no JavaScript bridge available; puremacro.runtime.transport "
            "needs either sockets (CPython) or a Pyodide kernel"
        ) from exc

    xhr = XMLHttpRequest.new()
    xhr.open("GET", _proxied(url), False)  # False => synchronous
    for name, value in (headers or {}).items():
        if name.lower() in _FORBIDDEN_HEADERS:
            continue
        xhr.setRequestHeader(name, value)
    # Keeps the browser from mangling bytes into UTF-8: each response
    # byte arrives as one code point in the low byte of a character.
    xhr.overrideMimeType("text/plain; charset=x-user-defined")
    try:
        xhr.send(None)
    except Exception as exc:  # JS exceptions surface as generic Python ones
        raise TransportError(
            f"browser refused the request to {url!r}. The usual cause is "
            f"CORS: the endpoint did not send Access-Control-Allow-Origin. "
            f"Retry via a proxy: "
            f"puremacro.runtime.enable_browser_network(proxy=...). "
            f"Underlying error: {exc}"
        ) from exc

    status = int(xhr.status)
    if status >= 400:
        raise TransportError(f"HTTP {status} for {url!r}")
    return bytes(ord(ch) & 0xFF for ch in xhr.responseText)


def get_bytes(url: str, timeout: float = 30.0, *,
              headers: dict | None = None,
              user_agent: str | None = None) -> bytes:
    """Fetch ``url`` as bytes over whichever transport works here.

    On a socket host this delegates to :func:`puremacro._http.safe_get_bytes`
    so the SSL-fallback and user-agent behaviour documented in
    ``narrative/sources/RETRY_POLICY.md`` is preserved exactly. In a
    browser it goes through a synchronous ``XMLHttpRequest``, where
    ``timeout`` and ``user_agent`` are accepted and ignored (see the
    module docstring).
    """
    mode = available()
    if mode == "sockets":
        from puremacro._http import safe_get_bytes

        return safe_get_bytes(url, timeout, user_agent=user_agent)
    if mode == "js-fetch":
        merged = dict(headers or {})
        if user_agent:
            merged.setdefault("User-Agent", user_agent)  # dropped downstream
        return _xhr_get(url, merged)
    raise TransportError(
        "no HTTP transport: this environment has neither sockets nor a "
        "JavaScript bridge. Load the data from a cartridge instead — see "
        "puremacro.pocket."
    )


def get_text(url: str, timeout: float = 30.0, *,
             headers: dict | None = None,
             user_agent: str | None = None,
             encoding: str = "utf-8") -> str:
    """Fetch ``url`` and decode it (errors ignored, as elsewhere in the package)."""
    raw = get_bytes(url, timeout, headers=headers, user_agent=user_agent)
    return raw.decode(encoding, errors="ignore")


def get_json(url: str, timeout: float = 30.0, *,
             headers: dict | None = None,
             user_agent: str | None = None) -> dict:
    """Fetch ``url`` and parse JSON. Empty bodies return ``{}``."""
    body = get_text(url, timeout, headers=headers, user_agent=user_agent).strip()
    if not body:
        return {}
    return _json.loads(body)


# ---------------------------------------------------------------------
# Routing the existing fetch layer over the bridge
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class _Response:
    """The slice of ``requests.Response`` the fetch layer actually uses."""

    url: str
    status_code: int
    content: bytes

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="ignore")

    def json(self):
        return _json.loads(self.text)

    def raise_for_status(self) -> None:
        if not self.ok:
            raise TransportError(f"HTTP {self.status_code} for {self.url!r}")


class _RequestsShim:
    """A ``requests``-shaped façade over the browser bridge.

    Only ``get`` is implemented — it is the only verb the fetch layer
    uses. ``post`` raises rather than silently doing nothing.
    """

    def get(self, url, *, headers=None, timeout=None, params=None, **_):
        if params:
            from urllib.parse import urlencode

            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urlencode(params)}"
        return _Response(url=url, status_code=200,
                         content=_xhr_get(url, headers))

    def post(self, *_a, **_k):
        raise TransportError(
            "puremacro's browser transport is GET-only; POST needs the "
            "async pyodide.http.pyfetch API, which cannot be called from "
            "synchronous estimator code."
        )


_ORIGINAL: dict = {}


def enable_browser_network(*, proxy: str | None = None) -> str:
    """Route the package's HTTP through the browser. Idempotent.

    Parameters
    ----------
    proxy : str, optional
        Prefix prepended to every URL, for a CORS proxy you control
        (e.g. ``"https://my-worker.example.dev/?url="``). Persists until
        :func:`disable_browser_network`.

    Returns
    -------
    str
        The transport now in force: ``"sockets"`` when the host already
        had them (nothing is patched), otherwise ``"js-fetch"``.

    Raises
    ------
    TransportError
        If the environment has neither sockets nor a JS bridge.
    """
    global _PROXY

    mode = available()
    if mode == "none":
        # Validate before recording the proxy, so a failed call leaves no
        # state behind.
        raise TransportError(
            "cannot enable browser networking: no JavaScript bridge found "
            "(this is not a Pyodide kernel)."
        )
    _PROXY = proxy
    if mode == "sockets":
        return "sockets"

    if _ORIGINAL:  # already installed
        return "js-fetch"

    import puremacro._http as core_http
    import puremacro.fetch._http as fetch_http

    _ORIGINAL["core_request"] = core_http._request
    _ORIGINAL["fetch_requests"] = fetch_http.requests

    def _request(url: str, timeout: float, user_agent: str | None = None) -> bytes:
        headers = {"User-Agent": user_agent} if user_agent else None
        return _xhr_get(url, headers)

    core_http._request = _request          # type: ignore[assignment]
    fetch_http.requests = _RequestsShim()  # type: ignore[assignment]
    return "js-fetch"


def disable_browser_network() -> None:
    """Undo :func:`enable_browser_network`. Safe to call when not installed."""
    global _PROXY
    _PROXY = None
    if not _ORIGINAL:
        return
    import puremacro._http as core_http
    import puremacro.fetch._http as fetch_http

    core_http._request = _ORIGINAL.pop("core_request")      # type: ignore[assignment]
    fetch_http.requests = _ORIGINAL.pop("fetch_requests")   # type: ignore[assignment]
    _ORIGINAL.clear()
