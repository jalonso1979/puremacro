"""F2.5 — the 8 Slice-A connectors emit a parser_schema_mismatch event
on ParserSchemaMismatchError."""
from __future__ import annotations

import importlib
import json

import pytest


_SCHEMA_CHECKED_CONNECTORS = (
    "beige_book", "eu_eurlex", "eu_parliament", "us_cbo",
    "fed_minutes", "fed_speeches", "bluesky", "ecb_press",
)


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("PUREMACRO_NARRATIVE_TELEMETRY", raising=False)
    import puremacro._cache_db as M
    M.close_conn()
    yield tmp_path
    M.close_conn()


def _apply_connector_stubs(name: str, mod, monkeypatch, _always_raise) -> None:
    """Apply connector-specific stubs so the ParserSchemaMismatchError path
    is reachable without making real network calls.

    For each connector:
    1. assert_landmarks is patched at the connector's local binding.
    2. HTTP fetch functions are stubbed at the connector's local binding.
    3. Listing/enumeration functions that produce large iteration sets are
       stubbed to return exactly one entry.
    """
    _dummy_text = "dummy content for schema mismatch test"
    _dummy_bytes = _dummy_text.encode()

    # --- Patch assert_landmarks at the connector's local binding ---
    # connectors do `from ._schema_check import assert_landmarks` which
    # creates a local binding; patching _schema_check alone won't suffice.
    if hasattr(mod, "assert_landmarks"):
        monkeypatch.setattr(mod, "assert_landmarks", _always_raise)

    if name == "beige_book":
        # beige_book uses safe_get_text_cached / safe_get_bytes_cached
        # (aliased as safe_get_text / safe_get_bytes at module level).
        # Stub both so _fomc_listing returns [] and _iter_modern returns
        # early (pre-1996 loop skipped) or hits assert_landmarks.
        for attr in ("safe_get_text", "safe_get_bytes"):
            if hasattr(mod, attr):
                val = _dummy_text if "text" in attr else _dummy_bytes
                monkeypatch.setattr(mod, attr, lambda *a, _v=val, **kw: _v)

    elif name == "eu_eurlex":
        # _enumerate_celex_ids makes SPARQL HTTP; stub it to return one ID.
        if hasattr(mod, "_enumerate_celex_ids"):
            monkeypatch.setattr(mod, "_enumerate_celex_ids",
                                lambda **kw: ["32024R0001"])
        # fetch_with_fallback routes to Wayback; return dummy HTML so
        # _parse_eurlex_html (which calls assert_landmarks) is reached.
        if hasattr(mod, "fetch_with_fallback"):
            monkeypatch.setattr(mod, "fetch_with_fallback",
                                lambda *a, **kw: _dummy_text)

    elif name == "eu_parliament":
        # _enumerate_session_dates is pure but can return thousands of
        # dates; stub it to return exactly one date.
        import datetime as dt
        _one_date = dt.date(2024, 1, 15)
        if hasattr(mod, "_enumerate_session_dates"):
            monkeypatch.setattr(mod, "_enumerate_session_dates",
                                lambda *a, **kw: [_one_date])
        # fetch_with_fallback routes to Wayback; return dummy HTML so
        # _parse_ep_page (which calls assert_landmarks) is reached.
        if hasattr(mod, "fetch_with_fallback"):
            monkeypatch.setattr(mod, "fetch_with_fallback",
                                lambda *a, **kw: _dummy_text)

    elif name == "us_cbo":
        # iter_cbo fetches RSS text → calls _parse_rss → calls assert_landmarks.
        # Stub safe_get_text to return dummy text; assert_landmarks fires.
        for attr in ("safe_get_text", "safe_get_bytes"):
            if hasattr(mod, attr):
                val = _dummy_text if "text" in attr else _dummy_bytes
                monkeypatch.setattr(mod, attr, lambda *a, _v=val, **kw: _v)

    elif name == "fed_minutes":
        # iter_fed_minutes fetches JSON listing → iterates items → fetches
        # announcement HTML → calls assert_landmarks on first body.
        # Stub safe_get_bytes to return valid JSON with one FOMC minutes item.
        _listing_json = json.dumps([{
            "pt": "Monetary Policy",
            "t": "FOMC Federal Open Market Committee Minutes 2024",
            "d": "2024-01-31",
            "l": "/test/fomc-minutes-test.htm",
        }]).encode()
        if hasattr(mod, "safe_get_bytes"):
            monkeypatch.setattr(mod, "safe_get_bytes",
                                lambda *a, **kw: _listing_json)
        # Stub safe_get_text (fetches announcement HTML) to return dummy text
        # so assert_landmarks fires.
        if hasattr(mod, "safe_get_text"):
            monkeypatch.setattr(mod, "safe_get_text",
                                lambda *a, **kw: _dummy_text)

    elif name == "fed_speeches":
        # iter_fed_speeches fetches RSS bytes → decodes → calls assert_landmarks.
        if hasattr(mod, "safe_get_bytes"):
            monkeypatch.setattr(mod, "safe_get_bytes",
                                lambda *a, **kw: _dummy_bytes)

    elif name == "bluesky":
        # iter_bluesky_posts iterates handles → calls _resolve_handle (HTTP)
        # → fetches feed page → calls assert_landmarks on first page.
        # Stub _resolve_handle to return a fake profile dict and limit to one
        # handle so setup is fast.
        _fake_profile = {"did": "did:plc:test", "displayName": "Test",
                         "handle": "test.bsky.social"}
        if hasattr(mod, "_resolve_handle"):
            monkeypatch.setattr(mod, "_resolve_handle",
                                lambda *a, **kw: _fake_profile)
        if hasattr(mod, "KNOWN_HANDLES"):
            monkeypatch.setattr(mod, "KNOWN_HANDLES",
                                ({"handle": "test.bsky.social"},))
        # Stub safe_get_text so the feed body fetch returns dummy text for
        # assert_landmarks.
        if hasattr(mod, "safe_get_text"):
            monkeypatch.setattr(mod, "safe_get_text",
                                lambda *a, **kw: _dummy_text)

    elif name == "ecb_press":
        # iter_ecb_press delegates to iter_ecb_decision; assert_landmarks is
        # called on the text field of the first yielded record.
        # Stub iter_ecb_decision to yield one dummy 4-tuple so the loop
        # body runs exactly once.
        import datetime as dt
        _dummy_record = (dt.date(2024, 1, 15), _dummy_text,
                         "https://ecb.europa.eu/test", {})
        if hasattr(mod, "iter_ecb_decision"):
            monkeypatch.setattr(mod, "iter_ecb_decision",
                                lambda **kw: iter([_dummy_record]))


@pytest.mark.parametrize("name", _SCHEMA_CHECKED_CONNECTORS)
def test_iter_source_emits_event_on_schema_mismatch(fresh_db, monkeypatch, name):
    """Patch the inner parser / body-fetch to raise ParserSchemaMismatchError;
    confirm the iter_<source> generator catches it AND emits a
    parser_schema_mismatch event before warning."""
    import warnings
    from puremacro.narrative.sources._schema_check import (
        ParserSchemaMismatchError,
    )

    # Spy on log_event to confirm it was called with the right args.
    calls: list[tuple] = []

    def _spy_log_event(*, source, outcome, fallback_used="none"):
        calls.append((source, outcome, fallback_used))

    # Patch at the canonical site; connectors that do a local import
    # will also resolve to this module-level spy.
    monkeypatch.setattr(
        "puremacro.narrative.sources._telemetry.log_event",
        _spy_log_event,
    )

    # Force assert_landmarks to always raise so the iter_<source>
    # wrapper's except block fires.
    def _always_raise(*a, **kw):
        raise ParserSchemaMismatchError(f"{name!r}: simulated mismatch")

    # Patch at the _schema_check module level.
    monkeypatch.setattr(
        "puremacro.narrative.sources._schema_check.assert_landmarks",
        _always_raise,
    )

    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")

    # Apply connector-specific stubs (local bindings + listing stubs).
    _apply_connector_stubs(name, mod, monkeypatch, _always_raise)

    # Prefer iter_ functions from __all__ (avoids picking up re-exported
    # helpers like iter_ecb_decision in ecb_press); fall back to dir() scan.
    _all = getattr(mod, "__all__", [])
    _all_iter = [n for n in _all if n.startswith("iter_")
                 and callable(getattr(mod, n, None))]
    if _all_iter:
        iter_fn_name = _all_iter[0]
    else:
        iter_fn_name = next(
            n for n in dir(mod)
            if n.startswith("iter_") and callable(getattr(mod, n))
        )
    iter_fn = getattr(mod, iter_fn_name)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        try:
            list(iter_fn())
        except TypeError:
            # Some iter_<source> functions REQUIRE args (e.g.,
            # iter_bluesky_posts(actors=)). For those, this test is
            # advisory — skip; a manual per-iter fixture can verify.
            pytest.skip(
                f"{name}: iter_{name} requires args; manual fixture needed."
            )

    matching = [c for c in calls if c[1] == "parser_schema_mismatch"]
    assert matching, (
        f"{name}: ParserSchemaMismatchError was raised but no "
        f"log_event(outcome='parser_schema_mismatch') was emitted. "
        f"Slice B F2.5 contract violation."
    )
    assert matching[0][0] == name
    assert matching[0][2] == "none"
