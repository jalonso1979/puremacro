"""Bangko Sentral ng Pilipinas (BSP) — monetary-policy decisions.

Live: https://www.bsp.gov.ph
Decision source (SharePoint REST API):
  https://www.bsp.gov.ph/_api/web/lists/getbytitle('Media Releases and Advisories')/items
  ?$select=ID,Title,PDate,Tag,Status,OData__ModerationStatus
  &$filter=(Tag eq 'Media Releases') and OData__ModerationStatus eq 0 and Status eq '2'
       and PDate ge '<from_date>T00:00:00Z'
  &$top=5000&$orderby=PDate desc

The BSP website (bsp.gov.ph/SitePages/MediaAndResearch/MonetaryPolicyDecisions.aspx)
returns HTTP 404.  The live press-release listing
(bsp.gov.ph/SitePages/MediaAndResearch/MediaList.aspx?TabId=1) is a
SharePoint/Angular SPA — its HTML shell is returned by a plain GET, but
the listing data is fetched at runtime via PnP.js calling the SharePoint
REST API.  That REST endpoint is publicly accessible without authentication
or WAF challenge.

The endpoint returns all press releases under the "Media Releases" tag.
Monetary-policy rate decisions are identified client-side by filtering on
titles that contain "Monetary Board" together with a rate-action keyword
("target rrp", "policy settings", "policy stance", "basis points",
"vigilance against inflation").  Non-rate Monetary Board releases (e.g.,
registration cancellations, foreign borrowing approvals) are excluded.

Speeches archive: BSP governor speeches are stored in a separate
SharePoint list ("Speeches list"), also accessible via the REST API at
/_api/web/lists/getbytitle('Speeches list')/items.  This connector ships
``iter_bsp_speeches`` because the endpoint is publicly accessible and
returns clean structured data.  The SpeechesDisp.aspx page URL pattern is
used for linking back to the full text.

Verified 2026-05-26 by the F1 Slice A rollout.
FALLBACK_POLICY = ("live",)  — the SharePoint REST API is public, no WAF,
no JS rendering required.
"""
from __future__ import annotations

import datetime
import json
import re
import warnings
from typing import Iterator

from ._fallback import fetch_with_fallback, FallbackExhaustedError
from ._schema_check import assert_landmarks, ParserSchemaMismatchError
from ._telemetry import log_event


PARSER_SCHEMA_VERSION = 1
FALLBACK_POLICY: tuple[str, ...] = ("live",)

_API_BASE = "https://www.bsp.gov.ph/_api/web/lists"

# Decision endpoint: Media Releases and Advisories list
# Returns all "Media Releases" tagged items; we filter to MB rate decisions
# client-side.  Using a 3-year look-back by default.
_DECISION_LIST = "Media Releases and Advisories"

# Speeches endpoint: separate list
_SPEECHES_LIST = "Speeches list"

# Landmarks verified from live API responses captured 2026-05-26.
_DECISION_LANDMARKS = ['"d"', '"results"', '"PDate"', '"Media Releases"']
_SPEECHES_LANDMARKS = ['"d"', '"results"', '"SDate"', 'Speeches_x0020_listListItem']

# Rate-decision title patterns — English-language MB releases about the RRP rate.
# Deliberately conservative to avoid capturing registration cancellations,
# foreign borrowing approvals, and membership announcements.
_DECISION_TITLE_RX = re.compile(
    r"monetary board.{0,80}?("
    r"target\s+rrp|policy\s+settings|policy\s+stance"
    r"|basis\s+points|vigilance\s+against\s+inflation"
    r"|maintains\s+target|reduces\s+target|raises\s+target"
    r")",
    re.IGNORECASE,
)

_BASE_URL = "https://www.bsp.gov.ph"


def _decision_url(*, lookback_years: int = 3) -> str:
    """Build the SharePoint REST API URL for Media Releases."""
    from_date = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(days=365 * lookback_years)
    ).strftime("%Y-%m-%dT00:00:00Z")
    list_encoded = _DECISION_LIST.replace(" ", "%20")
    return (
        f"{_API_BASE}/getbytitle('{list_encoded}')/items"
        f"?$select=ID,Title,PDate,Tag,Status,OData__ModerationStatus"
        f"&$filter=(Tag%20eq%20'Media%20Releases')"
        f"%20and%20OData__ModerationStatus%20eq%200"
        f"%20and%20Status%20eq%20'2'"
        f"%20and%20PDate%20ge%20'{from_date}'"
        f"&$top=5000&$orderby=PDate%20desc"
    )


def _speeches_url(*, lookback_years: int = 3) -> str:
    """Build the SharePoint REST API URL for Speeches."""
    from_date = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(days=365 * lookback_years)
    ).strftime("%Y-%m-%dT00:00:00Z")
    list_encoded = _SPEECHES_LIST.replace(" ", "%20")
    return (
        f"{_API_BASE}/getbytitle('{list_encoded}')/items"
        f"?$select=ID,Title,SDate,OData__ModerationStatus"
        f"&$filter=OData__ModerationStatus%20eq%200"
        f"%20and%20SDate%20ge%20'{from_date}'"
        f"&$top=5000&$orderby=SDate%20desc"
    )


def _parse_decision_json(body: str) -> Iterator[tuple]:
    """Parse Media Releases API response; yield rate-decision 4-tuples."""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        warnings.warn(
            f"bsp: JSON decode error in decision response: {e}",
            UserWarning, stacklevel=3,
        )
        return

    results = payload.get("d", {}).get("results", [])
    if not isinstance(results, list):
        warnings.warn(
            f"bsp: unexpected 'results' type {type(results)} in decision response",
            UserWarning, stacklevel=3,
        )
        return

    seen: set[str] = set()
    for entry in results:
        title = (entry.get("Title") or "").strip()
        if not title:
            continue

        # Filter to rate decisions only
        if not _DECISION_TITLE_RX.search(title):
            continue

        # Skip Tagalog/Filipino versions (contain non-ASCII or commas that
        # split the MB name from a Tagalog verb phrase)
        if not title.isascii():
            continue

        raw_date = entry.get("PDate", "")
        if not raw_date or raw_date in seen:
            continue
        seen.add(raw_date)

        try:
            date = datetime.datetime.strptime(raw_date[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            continue

        item_id = entry.get("ID") or entry.get("Id")
        url = (
            f"{_BASE_URL}/SitePages/MediaAndResearch/MediaDisp.aspx"
            f"?ItemId={item_id}&MType=MediaReleases"
            if item_id else _BASE_URL
        )

        yield (date, title, url, {
            "bank_code": "BSP",
            "country": "PHL",
            "doctype": "decision",
            "language": "en",
            "item_id": item_id,
        })


def _parse_speeches_json(body: str) -> Iterator[tuple]:
    """Parse Speeches list API response; yield speech 4-tuples."""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        warnings.warn(
            f"bsp: JSON decode error in speeches response: {e}",
            UserWarning, stacklevel=3,
        )
        return

    results = payload.get("d", {}).get("results", [])
    if not isinstance(results, list):
        warnings.warn(
            f"bsp: unexpected 'results' type {type(results)} in speeches response",
            UserWarning, stacklevel=3,
        )
        return

    for entry in results:
        title = (entry.get("Title") or "").strip()
        if not title:
            continue

        raw_date = entry.get("SDate", "")
        if not raw_date:
            continue

        try:
            date = datetime.datetime.strptime(raw_date[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            continue

        item_id = entry.get("ID") or entry.get("Id")
        url = (
            f"{_BASE_URL}/SitePages/MediaAndResearch/SpeechesDisp.aspx"
            f"?ItemId={item_id}"
            if item_id else _BASE_URL
        )

        yield (date, title, url, {
            "bank_code": "BSP",
            "country": "PHL",
            "doctype": "speech",
            "language": "en",
            "item_id": item_id,
        })


def iter_bsp_decision(
    *,
    lookback_years: int = 3,
    fetch_body: bool = False,
) -> Iterator[tuple]:
    """Yield (date, text, url, metadata) tuples for BSP Monetary Board decisions.

    Queries the BSP SharePoint REST API for "Media Releases" tagged press
    releases, then filters client-side to keep only Monetary Board rate
    decisions (Target RRP changes, holds, and policy-stance statements).

    Parameters
    ----------
    lookback_years : how many years of history to retrieve (default 3).
    fetch_body : unused (kept for API symmetry; the API already returns
        structured data with titles as the primary text).
    """
    url = _decision_url(lookback_years=lookback_years)
    try:
        listing = fetch_with_fallback(
            url, policy=FALLBACK_POLICY, source="bsp",
        )
    except FallbackExhaustedError as e:
        warnings.warn(
            f"bsp.iter_bsp_decision: fetch failed: {e}",
            UserWarning, stacklevel=2,
        )
        return

    try:
        assert_landmarks(
            listing, source="bsp", expected_version=PARSER_SCHEMA_VERSION,
            landmarks=_DECISION_LANDMARKS,
        )
    except ParserSchemaMismatchError as e:
        log_event(source="bsp", outcome="parser_schema_mismatch",
                  fallback_used="none")
        warnings.warn(
            f"bsp.iter_bsp_decision: schema mismatch: {e}",
            UserWarning, stacklevel=2,
        )
        return

    yield from _parse_decision_json(listing)


def iter_bsp_speeches(
    *,
    lookback_years: int = 3,
    fetch_body: bool = False,
) -> Iterator[tuple]:
    """Yield (date, text, url, metadata) tuples for BSP Governor speeches.

    Queries the BSP SharePoint REST API "Speeches list" endpoint, which
    returns all governor and senior-official speeches ordered by date.

    Parameters
    ----------
    lookback_years : how many years of history to retrieve (default 3).
    fetch_body : unused (kept for API symmetry).
    """
    url = _speeches_url(lookback_years=lookback_years)
    try:
        listing = fetch_with_fallback(
            url, policy=FALLBACK_POLICY, source="bsp",
        )
    except FallbackExhaustedError as e:
        warnings.warn(
            f"bsp.iter_bsp_speeches: fetch failed: {e}",
            UserWarning, stacklevel=2,
        )
        return

    try:
        assert_landmarks(
            listing, source="bsp", expected_version=PARSER_SCHEMA_VERSION,
            landmarks=_SPEECHES_LANDMARKS,
        )
    except ParserSchemaMismatchError as e:
        log_event(source="bsp", outcome="parser_schema_mismatch",
                  fallback_used="none")
        warnings.warn(
            f"bsp.iter_bsp_speeches: schema mismatch: {e}",
            UserWarning, stacklevel=2,
        )
        return

    yield from _parse_speeches_json(listing)


__all__ = ["iter_bsp_decision", "iter_bsp_speeches"]
