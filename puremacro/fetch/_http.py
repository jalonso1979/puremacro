"""Tiny HTTP cache for fetch modules.

Each URL is mirrored into CACHE_ROOT/<host>/<path-with-underscores>. A manifest
tracks fetch timestamps for reproducibility.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Optional
from urllib.parse import quote, quote_plus, urlencode, urlparse

try:
    import requests
except ImportError:                     # tablet / no-socket build
    # The name must stay bound: puremacro.runtime.transport rebinds it to a
    # browser-fetch shim and restores it afterwards, and `None` reads fine.
    requests = None                     # type: ignore[assignment]

CACHE_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
_MANIFEST_FILE = "_manifest.json"


def _path_for(url: str) -> Path:
    u = urlparse(url)
    safe_path = re.sub(r"[^A-Za-z0-9._-]", "_", (u.path + ("?" + u.query if u.query else "")))
    safe_path = safe_path.strip("_") or "root"
    # Truncate long paths and append hash to disambiguate.
    if len(safe_path) > 120:
        digest = hashlib.sha1(safe_path.encode()).hexdigest()[:8]
        safe_path = safe_path[:100] + "_" + digest
    return CACHE_ROOT / u.netloc / safe_path


def _load_manifest() -> dict:
    f = CACHE_ROOT / _MANIFEST_FILE
    try:
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError, RuntimeError, ConnectionRefusedError, Exception, KeyError, LookupError, AssertionError, TypeError, ConnectionError):
        pass
    return {}


def _write_manifest(d: dict) -> None:
    try:
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        (CACHE_ROOT / _MANIFEST_FILE).write_text(
            json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def _with_query(url: str, params: Optional[dict]) -> str:
    if not params:
        return url
    return url + ("&" if urlparse(url).query else "?") + urlencode(params, safe=",:*", quote_via=quote)


def _redact(exc: BaseException, secret_params: Optional[dict]) -> None:
    secrets = [str(v) for v in (secret_params or {}).values() if v]
    secrets += [quote_plus(v) for v in secrets]
    if not secrets:
        return
    def clean(a):
        if not isinstance(a, str):
            a = str(a)
        for v in secrets:
            a = a.replace(v, "***")
        return a
    exc.args = tuple(clean(a) for a in exc.args)


def cached_get(
    url: str,
    *,
    refresh: bool = False,
    headers: Optional[dict] = None,
    timeout: int = 60,
    params: Optional[dict] = None,
    secret_params: Optional[dict] = None,
) -> bytes:
    """GET the URL, cache the bytes on disk, return bytes.

    Re-fetches if ``refresh=True`` or the cache file is missing.
    Updates the manifest with the fetch timestamp each time bytes are written.

    ``params`` are appended to the query string and are part of the cache
    identity. ``secret_params`` (an API key) are sent with the request but
    never reach the cache path or the manifest: the on-disk file name and
    ``_manifest.json`` are derived from ``url`` + ``params`` alone, so a key
    passed this way is not written anywhere under ``data/raw``. Never put a
    key in ``url`` or ``params`` directly.
    """
    if requests is None:
        raise RuntimeError(
            "puremacro.fetch needs `requests` for a live fetch; install it, or "
            "work from the on-disk cache under data/raw/.")
    url = _with_query(url, params)
    request_url = _with_query(url, secret_params)
    target = _path_for(url)
    # The cache is a convenience, never a requirement: on a read-only or
    # sandboxed install (an iPad, a container with the package baked into the
    # image) the directory may be unwritable, or even unreadable — note that
    # pathlib only swallows ENOENT/ENOTDIR/EBADF/ELOOP, so a permission denial
    # raises out of `.exists()` rather than returning False. Degrade to a plain
    # uncached fetch instead of failing.
    cacheable = True
    try:
        if target.exists() and not refresh:
            return target.read_bytes()
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        cacheable = False

    # Many federal data hosts (BLS Akamai, FRB) block default python-requests UA.
    # Set a research-identifying default; callers can override via headers=.
    final_headers = {"User-Agent": "uncertainty_examples/1.0 (research@itam.mx)"}
    if headers:
        final_headers.update(headers)
    try:
        resp = requests.get(request_url, headers=final_headers, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        # requests puts the full URL in its error text; keep the key out of
        # tracebacks and logs.
        _redact(exc, secret_params)
        raise
    if not cacheable:
        return resp.content
    try:
        target.write_bytes(resp.content)
    except OSError:
        return resp.content

    manifest = _load_manifest()
    manifest[str(target.relative_to(CACHE_ROOT))] = {
        "url": url,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "size_bytes": len(resp.content),
    }
    _write_manifest(manifest)
    return resp.content
