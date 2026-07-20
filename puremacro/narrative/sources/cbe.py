"""Central Bank of Egypt (CBE) — monetary-policy decisions.

Live: https://www.cbe.org.eg
Decision source (Sitecore JSON REST API):
  https://www.cbe.org.eg/api/listing/News?pageNo=0&pageSize=500

The CBE website (cbe.org.eg) is protected by an F5 WAF for most paths,
returning a "Request Rejected" HTML page to plain HTTP clients.  However,
the news-listing API endpoint ``/api/listing/News`` is publicly accessible
without WAF challenge.  This endpoint is used by the SPA component on the
``/en/news-publications/news`` page, which passes ``data-apiUrl="/api/listing/News"``
to a Sitecore listing component.

The endpoint returns up to 753 news items (all categories) ordered by date
descending.  The CBE uses a Sitecore CMS; each news item carries a list of
``categories`` array with ``{key, value}`` objects.  MPC rate decisions are
tagged with category key ``7185e116d73040b78de697623070479e``
("MPC Press Release").  This connector fetches a large page (pageSize=500)
and filters client-side to keep only the MPC-tagged entries.

The ``/api/sitecore/MPCMeetings/GetMPCMeetingsDetails`` endpoint referenced in
``data-api-url`` on the MPC schedule page is also present, but it is protected
by the F5 WAF (returns "Request Rejected" 403) even with browser-like headers.

Speeches: The hypothesised speeches URL
(cbe.org.eg/en/media-center/speeches) returns HTTP 404.  The "Interviews,
Talks and Speeches" news category (key ``fe1b250c3ce8432da678d4e7c83f7a97``)
exists in the same news listing API, but returns a small number of results
that are mixed interviews and press appearances — not a clean governor-speech
archive.  ``iter_cbe_speeches`` is intentionally omitted.

Date field: ``customDate`` (ISO-8601 datetime string, e.g. ``"2026-05-21T00:00:00"``).
Link field: relative path (``/en/news-publications/news/<slug>``).

Verified 2026-05-26 by the F1 Slice A rollout.
FALLBACK_POLICY = ("live",)  — the JSON listing API is public, no WAF,
no JS rendering required.
"""
from __future__ import annotations

import datetime
import json
import warnings
from typing import Iterator

from ._fallback import fetch_with_fallback, FallbackExhaustedError
from ._schema_check import assert_landmarks, ParserSchemaMismatchError
from ._telemetry import log_event


PARSER_SCHEMA_VERSION = 1
FALLBACK_POLICY: tuple[str, ...] = ("live",)

_BASE_URL = "https://www.cbe.org.eg"
_API_BASE = f"{_BASE_URL}/api/listing/News"

# MPC Press Release category key — verified from live API 2026-05-26.
# All rate-decision communiques carry this Sitecore category tag.
_MPC_CATEGORY_KEY = "7185e116d73040b78de697623070479e"

# Default page size: large enough to cover the full news archive in one call.
# The total news count as of 2026-05-26 is 753 items; 500 covers ~3 years
# of decisions (51 MPC entries).  Set _PAGE_SIZE below the hard API limit
# to allow a safety margin.
_PAGE_SIZE = 500

# Landmarks verified from live API response captured 2026-05-26.
# The JSON body always contains these substring markers.
_DECISION_LANDMARKS = ['"results"', '"totalResultsCount"', '"customDate"', '"MPC Press Release"']


def _decision_url(*, page_no: int = 0, page_size: int = _PAGE_SIZE) -> str:
    """Build the news-listing API URL."""
    return f"{_API_BASE}?pageNo={page_no}&pageSize={page_size}"


def _parse_news_json(body: str) -> Iterator[tuple]:
    """Parse the /api/listing/News response; yield MPC decision 4-tuples.

    Filters to entries whose ``categories`` list contains the MPC Press
    Release category key.  All other news items are silently skipped.
    """
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        warnings.warn(
            f"cbe: JSON decode error in decision response: {e}",
            UserWarning, stacklevel=3,
        )
        return

    results = payload.get("results")
    if not isinstance(results, list):
        warnings.warn(
            f"cbe: unexpected 'results' type {type(results)} in response "
            f"(expected list)",
            UserWarning, stacklevel=3,
        )
        return

    seen: set[str] = set()
    for entry in results:
        # Filter: keep only MPC Press Release entries.
        cats = entry.get("categories") or []
        cat_keys = {c.get("key") for c in cats}
        if _MPC_CATEGORY_KEY not in cat_keys:
            continue

        # Use customDate for precision (ISO datetime); fall back to date string.
        raw_date = (entry.get("customDate") or entry.get("date") or "").strip()
        if not raw_date or raw_date in seen:
            continue
        seen.add(raw_date)

        try:
            # customDate format: "2026-05-21T00:00:00"
            date = datetime.datetime.fromisoformat(raw_date[:19])
        except (ValueError, TypeError):
            # Fallback to the human-readable date field ("21 May 2026")
            try:
                date = datetime.datetime.strptime(raw_date, "%d %b %Y")
            except (ValueError, TypeError):
                continue

        title = (entry.get("title") or "").strip()
        if not title:
            continue

        link = (entry.get("url") or "").strip()
        url = (f"{_BASE_URL}{link}" if link.startswith("/") else link) or _BASE_URL

        item_id = (entry.get("itemId") or "").strip()

        yield (date, title, url, {
            "bank_code": "CBE",
            "country": "EGY",
            "doctype": "decision",
            "language": "en",
            "item_id": item_id,
        })


def iter_cbe_decision(
    *,
    page_size: int = _PAGE_SIZE,
    fetch_body: bool = False,
) -> Iterator[tuple]:
    """Yield (date, text, url, metadata) tuples for CBE MPC rate decisions.

    Fetches the CBE public Sitecore news-listing API and filters client-side
    to entries tagged with the "MPC Press Release" category.  Returns one
    tuple per MPC meeting communique, ordered by date descending.

    Parameters
    ----------
    page_size : number of news items to request per API call (default 500,
        which covers the full news history as of 2026).  Increase if the
        total news count exceeds this value and older decisions are needed.
    fetch_body : unused (kept for API symmetry; the API already returns
        structured title data as the primary text, and the linked HTML pages
        require JS rendering to display the full communique text).
    """
    url = _decision_url(page_no=0, page_size=page_size)
    try:
        listing = fetch_with_fallback(
            url, policy=FALLBACK_POLICY, source="cbe",
        )
    except FallbackExhaustedError as e:
        warnings.warn(
            f"cbe.iter_cbe_decision: fetch failed: {e}",
            UserWarning, stacklevel=2,
        )
        return

    try:
        assert_landmarks(
            listing, source="cbe", expected_version=PARSER_SCHEMA_VERSION,
            landmarks=_DECISION_LANDMARKS,
        )
    except ParserSchemaMismatchError as e:
        log_event(source="cbe", outcome="parser_schema_mismatch",
                  fallback_used="none")
        warnings.warn(
            f"cbe.iter_cbe_decision: schema mismatch: {e}",
            UserWarning, stacklevel=2,
        )
        return

    yield from _parse_news_json(listing)


# iter_cbe_speeches is intentionally omitted:
# - The hypothesised speeches URL (cbe.org.eg/en/media-center/speeches)
#   returns HTTP 404.
# - The "Interviews, Talks and Speeches" news category
#   (key fe1b250c3ce8432da678d4e7c83f7a97) exists in the news API but
#   returns mixed press appearances, not a clean governor-speech archive.
# Ship when CBE adds a dedicated speeches listing or API endpoint.

__all__ = ["iter_cbe_decision"]
