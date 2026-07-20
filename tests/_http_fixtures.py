"""HTTP fixture cache for narrative connector offline tests.

Mechanism
---------

* Replay mode (default): the three ``puremacro.narrative.sources._http``
  helpers (``safe_get_bytes``, ``safe_get_text``, ``safe_get_json``) are
  monkey-patched to read from a per-request JSON file under
  ``tests/fixtures/http/``. Cache miss in replay mode raises a clear
  ``FileNotFoundError`` telling the developer to re-record.
* Record mode (``PUREMACRO_RECORD_HTTP=1``): the helpers fire the real
  HTTP call, return the result, and persist the response into the
  cache before yielding.

Cache key
---------

SHA256 of (URL + sorted-headers JSON). For the current connectors,
headers are not custom-supplied per call (the underlying ``urllib``
``Request`` always uses the module's ``USER_AGENT`` constant), so the
key is effectively a content-addressed hash of the URL.

JSON payload
------------

::

    {
      "url": "...",
      "status": 200,
      "content_type": "application/json",
      "encoding": "text" | "base64",
      "body":  "<text body>"          # when encoding == "text"
      "body_b64": "<base64 bytes>"   # when encoding == "base64"
    }

The dual-encoding scheme lets us record HTML/JSON bodies as plain text
(easier to diff in the fixture file) while still supporting binary
payloads from ``safe_get_bytes``.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "http"


def _record_mode() -> bool:
    """Read the env var fresh each call so tests can flip it via
    ``monkeypatch.setenv`` without re-importing this module."""
    return os.getenv("PUREMACRO_RECORD_HTTP") == "1"


# Public alias kept for spec compatibility. Note that this is only
# evaluated at import time; tests should call ``_record_mode()`` for
# the live value.
RECORD_MODE = _record_mode()


def _key(url: str, headers: dict | None = None) -> str:
    payload = url + "|" + json.dumps(
        sorted((headers or {}).items()), separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fixture_path(url: str, headers: dict | None = None) -> Path:
    """Path to the JSON fixture for a given URL+headers pair."""
    return FIXTURE_DIR / f"{_key(url, headers)}.json"


def read_fixture(url: str, headers: dict | None = None) -> tuple[int, bytes, str]:
    """Load a recorded fixture. Returns (status, body_bytes, content_type).

    Raises ``FileNotFoundError`` if no fixture is recorded for the
    requested URL — the message points at how to re-record.
    """
    p = fixture_path(url, headers)
    if not p.exists():
        raise FileNotFoundError(
            f"HTTP fixture missing for URL: {url!r} "
            f"(expected at {p}). Re-run the test with "
            f"PUREMACRO_RECORD_HTTP=1 to record."
        )
    with p.open("r", encoding="utf-8") as f:
        rec = json.load(f)
    encoding = rec.get("encoding", "text")
    if encoding == "base64":
        body = base64.b64decode(rec["body_b64"])
    else:
        body = rec.get("body", "").encode("utf-8")
    status = int(rec.get("status", 200))
    content_type = rec.get("content_type", "")
    return status, body, content_type


def write_fixture(
    url: str,
    headers: dict | None,
    status: int,
    body: bytes,
    content_type: str = "",
) -> Path:
    """Write a recorded HTTP response to the fixture cache.

    Text-decodable bodies are stored as plain text (easier to diff);
    binary bodies fall back to base64. Returns the path written.
    """
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    p = fixture_path(url, headers)
    rec: dict = {
        "url": url,
        "status": int(status),
        "content_type": content_type,
    }
    try:
        text = body.decode("utf-8")
        # Sanity-check: text round-trip must not lose bytes (rules out
        # mojibake from invalid UTF-8 that happened to decode).
        if text.encode("utf-8") == body:
            rec["encoding"] = "text"
            rec["body"] = text
        else:
            raise UnicodeDecodeError("utf-8", body, 0, len(body), "round-trip")
    except UnicodeDecodeError:
        rec["encoding"] = "base64"
        rec["body_b64"] = base64.b64encode(body).decode("ascii")
    with p.open("w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2, sort_keys=True)
    return p


# ---------------------------------------------------------------------------
# Monkey-patch helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def install_fixture_patches() -> Iterator[None]:
    """Patch the three ``_http`` helpers so they consult the fixture
    cache. Real HTTP only fires when ``PUREMACRO_RECORD_HTTP=1``.

    Replay mode behaviour
    ---------------------
    * ``safe_get_bytes`` returns the recorded body bytes.
    * ``safe_get_text`` returns the body decoded as UTF-8 (errors
      ignored, matching the production helper's contract).
    * ``safe_get_json`` parses the body as JSON; an empty body returns
      ``{}`` (matching the production behaviour for rate-limited
      pages).

    Record mode behaviour
    ---------------------
    The patched functions delegate to the original helpers, capture the
    raw bytes (and inferred content-type), persist them to the cache,
    and return the same values the originals returned.
    """
    from puremacro.narrative.sources import _http as _http_mod

    orig_bytes = _http_mod.safe_get_bytes
    orig_text = _http_mod.safe_get_text
    orig_json = _http_mod.safe_get_json
    record = _record_mode()

    def _patched_bytes(url, timeout=_http_mod.DEFAULT_TIMEOUT, **_kwargs):
        if record:
            body = orig_bytes(url, timeout=timeout)
            try:
                write_fixture(url, None, 200, body, "")
            except Exception:
                # Recording failures should never break the call.
                pass
            return body
        _, body, _ = read_fixture(url)
        return body

    def _patched_text(url, timeout=_http_mod.DEFAULT_TIMEOUT, **_kwargs):
        if record:
            text = orig_text(url, timeout=timeout)
            try:
                write_fixture(url, None, 200, text.encode("utf-8"), "")
            except Exception:
                pass
            return text
        _, body, _ = read_fixture(url)
        return body.decode("utf-8", errors="ignore")

    def _patched_json(url, timeout=_http_mod.DEFAULT_TIMEOUT, **_kwargs):
        if record:
            data = orig_json(url, timeout=timeout)
            # Re-serialise to capture exact bytes for replay.
            try:
                serialised = json.dumps(data).encode("utf-8")
                write_fixture(url, None, 200, serialised, "application/json")
            except Exception:
                pass
            return data
        _, body, _ = read_fixture(url)
        text = body.decode("utf-8", errors="ignore")
        if not text.strip():
            return {}
        return json.loads(text)

    _http_mod.safe_get_bytes = _patched_bytes
    _http_mod.safe_get_text = _patched_text
    _http_mod.safe_get_json = _patched_json
    # Also patch the names already imported into other modules. Each
    # connector imports its helper directly (``from ._http import
    # safe_get_bytes``), so we need to update the module globals where
    # those names live.
    from puremacro.narrative.sources import (
        _rss as _rss_mod,
        us_treasury as _us_treasury,
        us_federal_register as _us_fr,
        us_dod_contracts as _us_dod,
        imf_articleiv as _imf_a4,
        oecd_surveys as _oecd,
        news_api as _news_api,
        eu_ecfin as _eu_ecfin,
    )
    from puremacro.narrative.replication import (
        ramey_2011_defense as _ramey_mod,
        romer_romer_2010 as _rr10_mod,
        romer_romer_2017 as _rr17_mod,
        mertens_ravn_2013 as _mr_mod,
        cloyne_2013_uk as _cloyne_mod,
        dglp_2011_consolidations as _dglp_mod,
    )

    # Modules that re-imported the helpers; map module → set(name).
    targets = [
        (_rss_mod,           {"safe_get_bytes": _patched_bytes}),
        (_us_treasury,       {"safe_get_text": _patched_text}),
        (_us_fr,             {"safe_get_json":  _patched_json}),
        (_us_dod,            {"safe_get_text":  _patched_text}),
        (_imf_a4,            {"safe_get_text":  _patched_text}),
        (_oecd,              {"safe_get_text":  _patched_text}),
        (_news_api,          {"safe_get_json":  _patched_json}),
        # eu_ecfin imports inside the function body, so it always
        # sees the patched module-level _http (covered by the
        # _http_mod patch above), but patch defensively anyway.
        (_eu_ecfin,          {}),
        (_ramey_mod,         {"safe_get_bytes": _patched_bytes}),
        (_rr10_mod,          {"safe_get_bytes": _patched_bytes}),
        (_rr17_mod,          {"safe_get_bytes": _patched_bytes}),
        (_mr_mod,            {"safe_get_bytes": _patched_bytes}),
        (_cloyne_mod,        {"safe_get_bytes": _patched_bytes}),
        (_dglp_mod,          {"safe_get_bytes": _patched_bytes}),
    ]
    saved: list[tuple[object, str, object]] = []
    for mod, name_map in targets:
        for name, patch in name_map.items():
            if hasattr(mod, name):
                saved.append((mod, name, getattr(mod, name)))
                setattr(mod, name, patch)

    try:
        yield
    finally:
        _http_mod.safe_get_bytes = orig_bytes
        _http_mod.safe_get_text = orig_text
        _http_mod.safe_get_json = orig_json
        for mod, name, original in saved:
            setattr(mod, name, original)


__all__ = [
    "FIXTURE_DIR",
    "RECORD_MODE",
    "fixture_path",
    "read_fixture",
    "write_fixture",
    "install_fixture_patches",
]
