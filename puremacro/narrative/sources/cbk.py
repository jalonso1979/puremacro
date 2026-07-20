"""Central Bank of Kenya (CBK) — monetary-policy decisions.

Live: https://www.centralbank.go.ke
Decision source (WordPress REST API):
  https://www.centralbank.go.ke/wp-json/wp/v2/posts?categories=27&per_page=100&orderby=date&order=desc

The CBK website runs on WordPress with the Sucuri WAF in front.  The
WordPress REST API (``/wp-json/wp/v2/posts``) is publicly accessible
without WAF challenge when a ``Referer`` header is set (the
:mod:`._http` helper sends browser-like headers by default).

The API returns all posts in the "Monetary Policy" category (ID 27),
which includes rate decisions, survey posts (CEOs, Market Perceptions,
Agriculture), media briefings, and meeting announcements.  This
connector filters client-side to keep only rate-decision posts — those
whose title matches ``_DECISION_TITLE_RX`` (e.g.
"MPC retains/lowers/raises (the) CBR at/to X.XX percent").

As of 2026-05-26 the Monetary Policy category holds 224+ posts;
requesting ``per_page=100`` covers roughly three years of posts (10+
decisions per year) in a single call.  Increase ``_PAGE_SIZE`` if the
full history is needed.

The older speeches archive at
``/releases/speeches/`` is a static HTML table (no wp-json route;
entries link to PDF files in ``/uploads/speeches/``).  The table covers
speeches from 2009 through mid-2024 with no consistent structured
metadata.  ``iter_cbk_speeches`` is intentionally omitted pending a
cleaner archive.

Date field: ``date`` (ISO-8601 datetime string, e.g.
``"2026-04-08T18:31:37"``).
Link field: full URL (``https://www.centralbank.go.ke/<year>/<month>/<day>/<slug>/``).

Verified 2026-05-26 by the F1 Slice A rollout.
FALLBACK_POLICY = ("live",)  — the WordPress REST API is public; the
Sucuri WAF does not block API calls with standard browser headers.
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

_BASE_URL = "https://www.centralbank.go.ke"

# Monetary Policy category ID — verified from
# /wp-json/wp/v2/categories?per_page=100 on 2026-05-26.
_MONETARY_POLICY_CATEGORY = 27

# Default page size: large enough to cover several years of decisions
# in a single API call.  Total posts in category 27 as of 2026-05-26:
# 224.  per_page=100 covers ~3-4 years of the most recent posts.
_PAGE_SIZE = 100


def _decision_url(*, page_size: int = _PAGE_SIZE) -> str:
    """Build the WordPress REST API URL for Monetary Policy posts."""
    return (
        f"{_BASE_URL}/wp-json/wp/v2/posts"
        f"?categories={_MONETARY_POLICY_CATEGORY}"
        f"&per_page={page_size}"
        f"&orderby=date&order=desc"
    )


# Matches rate-decision post titles.  CBK uses three verbs and two
# slightly varying preposition patterns (with/without "the"):
#   "MPC retains the CBR at 8.75 percent"   (newer format)
#   "MPC retains CBR at 13.00 percent"       (older format)
#   "MPC lowers the CBR to 8.75 percent"
#   "MPC raises CBR to 13.00 percent"
_DECISION_TITLE_RX = re.compile(
    r"\bMPC\b.+\bCBR\b",
    re.IGNORECASE,
)

# Landmarks verified from live API response captured 2026-05-26.
# The JSON array body always contains these substring markers for
# category 27 with at least one rate decision.
# Note: "CBR" appears as a value inside title strings (not as a JSON key),
# so we match it unquoted; "centralbank.go.ke" appears in all link URLs.
_DECISION_LANDMARKS = ['"title"', '"categories"', 'CBR', 'centralbank.go.ke']


def _parse_posts_json(body: str) -> Iterator[tuple]:
    """Parse the /wp-json/wp/v2/posts response; yield decision 4-tuples.

    Filters to entries whose ``title.rendered`` matches
    ``_DECISION_TITLE_RX`` (MPC rate decisions).  All other posts —
    survey reports, media briefings, meeting announcements — are
    silently skipped.
    """
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        warnings.warn(
            f"cbk: JSON decode error in decision response: {e}",
            UserWarning, stacklevel=3,
        )
        return

    if not isinstance(payload, list):
        warnings.warn(
            f"cbk: unexpected top-level type {type(payload)} in response "
            f"(expected list)",
            UserWarning, stacklevel=3,
        )
        return

    seen: set[int] = set()
    for entry in payload:
        item_id = entry.get("id")
        if item_id is None or item_id in seen:
            continue
        seen.add(item_id)

        title_obj = entry.get("title") or {}
        title = (title_obj.get("rendered") or "").strip()
        if not title:
            continue

        # Keep only MPC rate-decision posts.
        if not _DECISION_TITLE_RX.search(title):
            continue

        raw_date = (entry.get("date") or "").strip()
        if not raw_date:
            continue

        try:
            # WP date format: "2026-04-08T18:31:37"
            date = datetime.datetime.fromisoformat(raw_date[:19])
        except (ValueError, TypeError):
            continue

        url = (entry.get("link") or "").strip() or _BASE_URL

        yield (date, title, url, {
            "bank_code": "CBK",
            "country": "KEN",
            "doctype": "decision",
            "language": "en",
            "item_id": item_id,
        })


def iter_cbk_decision(
    *,
    page_size: int = _PAGE_SIZE,
    fetch_body: bool = False,
) -> Iterator[tuple]:
    """Yield (date, text, url, metadata) tuples for CBK MPC rate decisions.

    Fetches the CBK WordPress REST API (``/wp-json/wp/v2/posts``,
    category 27 "Monetary Policy") and filters client-side to entries
    whose title matches "MPC (retains|lowers|raises) (the) CBR
    (at|to) X.XX percent".  Survey posts, media briefings, and meeting
    announcements are excluded.

    Parameters
    ----------
    page_size : number of posts to request per API call (default 100,
        covering several years of the most recent posts).  Increase if
        older decisions are needed.
    fetch_body : unused (kept for API symmetry; the API already returns
        the post title as the primary text, and the linked web pages
        use Divi Builder shortcodes that render poorly as raw HTML).
    """
    url = _decision_url(page_size=page_size)
    try:
        listing = fetch_with_fallback(
            url, policy=FALLBACK_POLICY, source="cbk",
        )
    except FallbackExhaustedError as e:
        warnings.warn(
            f"cbk.iter_cbk_decision: fetch failed: {e}",
            UserWarning, stacklevel=2,
        )
        return

    try:
        assert_landmarks(
            listing, source="cbk", expected_version=PARSER_SCHEMA_VERSION,
            landmarks=_DECISION_LANDMARKS,
        )
    except ParserSchemaMismatchError as e:
        log_event(source="cbk", outcome="parser_schema_mismatch",
                  fallback_used="none")
        warnings.warn(
            f"cbk.iter_cbk_decision: schema mismatch: {e}",
            UserWarning, stacklevel=2,
        )
        return

    yield from _parse_posts_json(listing)


# iter_cbk_speeches is intentionally omitted:
# - The speeches archive at /releases/speeches/ is a static HTML table
#   with no wp-json route (PDFs linked from /uploads/speeches/).
# - The table spans 2009-2024 but lacks consistent structured metadata
#   (no category keys, no machine-readable date+title pairs beyond what
#   a regex could extract from raw HTML).
# - Ship when CBK adds a WordPress-based speeches post type or a clean
#   structured API.

__all__ = ["iter_cbk_decision"]
