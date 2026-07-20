"""Federal Reserve FOMC minutes.

Resolution strategy (3-tier, robust across all eras):
  1. Fetch the announcement page (URL from JSON `l` field).
  2. Parse it for the actual minutes body link
     (``/fomc/minutes/{meeting-date}.htm`` for pre-2014;
      ``/monetarypolicy/fomcminutes{release-date}.htm`` for post-2014).
  3. Fetch + extract the body. Fall back to the announcement page text
     if the body URL fails or the extraction is short.
"""
from __future__ import annotations

import json
import re
import warnings
from typing import Iterator

import pandas as pd

from ..._http import safe_get_bytes, safe_get_text
from ._extractors import extract_body
from ._schema_check import assert_landmarks, ParserSchemaMismatchError


PARSER_SCHEMA_VERSION = 1


_LISTING_URL = "https://www.federalreserve.gov/json/ne-press.json"
_BASE = "https://www.federalreserve.gov"
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_BODY_LINK_RX = re.compile(
    r'<a\b[^>]*\bhref="(/(?:fomc/minutes|monetarypolicy/fomcminutes)[^"]+\.htm)"',
    flags=re.IGNORECASE,
)


def _extract_minutes_body_link(announcement_html: str) -> str | None:
    """Find the first link to a minutes body inside the announcement page.

    Looks for ``<a href="/fomc/minutes/...html">`` (pre-2014) or
    ``<a href="/monetarypolicy/fomcminutes...html">`` (post-2014).
    Returns the href as found; caller prepends ``_BASE`` if relative.
    Returns ``None`` if no match.
    """
    m = _BODY_LINK_RX.search(announcement_html)
    if not m:
        return None
    return m.group(1)


def iter_fed_minutes() -> Iterator[tuple]:
    """Yield (date, text, url, metadata) for FOMC meeting minutes."""
    try:
        body = safe_get_bytes(_LISTING_URL, user_agent=_UA)
    except Exception:
        return
    try:
        obj = json.loads(body.decode("utf-8-sig", errors="ignore"))
    except json.JSONDecodeError:
        return
    items = obj if isinstance(obj, list) else obj.get("refData", [])
    _checked = False
    for item in items:
        if (item.get("pt") or "").lower() != "monetary policy":
            continue
        title = (item.get("t") or item.get("ti") or "").lower()
        if "minutes" not in title:
            continue
        if "discount rate" in title:
            continue
        if "fomc" not in title and "federal open market committee" not in title:
            continue
        try:
            date = pd.Timestamp(item.get("d"))
        except Exception:
            continue
        href = item.get("l", "")
        if not href:
            continue
        announcement_url = _BASE + href if href.startswith("/") else href

        # Step 1+2: fetch announcement, parse for body link.
        try:
            announcement_html = safe_get_text(announcement_url, user_agent=_UA)
        except Exception:
            continue

        if not _checked:
            try:
                assert_landmarks(
                    announcement_html, source="fed_minutes",
                    expected_version=PARSER_SCHEMA_VERSION,
                    landmarks=["Federal Open Market Committee", "Minutes"],
                )
            except ParserSchemaMismatchError as e:
                from ._telemetry import log_event
                log_event(source="fed_minutes", outcome="parser_schema_mismatch",
                          fallback_used="none")
                warnings.warn(
                    f"puremacro.narrative.sources.fed_minutes: schema mismatch "
                    f"on first body: {e}",
                    UserWarning, stacklevel=2,
                )
                return
            _checked = True

        body_text = ""
        chosen_url = announcement_url
        body_href = _extract_minutes_body_link(announcement_html)
        if body_href:
            body_url = _BASE + body_href if body_href.startswith("/") else body_href
            try:
                body_html = safe_get_text(body_url, user_agent=_UA)
                body_text = extract_body(body_html, bank_code="FED")
                if body_text and len(body_text) >= 5000:
                    chosen_url = body_url
                else:
                    body_text = ""  # too short, fall back
            except Exception:
                body_text = ""

        # Step 3: fall back to announcement-page extraction if body fetch
        # failed or was too short.
        if not body_text:
            body_text = extract_body(announcement_html, bank_code="FED")
            chosen_url = announcement_url

        if not body_text:
            continue
        yield (date, body_text, chosen_url, {
            "doctype": "minutes", "language": "en",
            "bank_code": "FED", "country": "USA",
        })


__all__ = ["iter_fed_minutes", "PARSER_SCHEMA_VERSION"]
