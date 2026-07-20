"""Bank Indonesia (BI) — monetary-policy decisions + governor speeches.

Live: https://www.bi.go.id/en
Decision listing: https://www.bi.go.id/en/publikasi/ruang-media/news-release/Default.aspx
Speech archive:   https://www.bi.go.id/en/publikasi/ruang-media/pidato-dewan-gubernur/Default.aspx

Both pages are publicly accessible without JS rendering (plain HTTP GET
returns the full listing HTML). The hypothesised RSS URL
(bi.go.id/en/publikasi/RSS/RSS-news.xml) returns a 404.

Verified 2026-05-26 by the F1 Slice A rollout.
FALLBACK_POLICY = ("live",)

Each listing page renders at most one page of entries (the current page).
Each entry is parsed from the HTML pattern:
  <a href="/en/publikasi/ruang-media/.../Pages/<slug>.aspx"
     class="mt-0 media__title ...">TITLE</a>
  ...
  <div class="media__subtitle">DD Month YYYY• ...
"""
from __future__ import annotations

import re
import warnings
from typing import Iterator

import pandas as pd

from ._fallback import fetch_with_fallback, FallbackExhaustedError
from ._ratedoc import strip_html
from ._schema_check import assert_landmarks, ParserSchemaMismatchError
from ._telemetry import log_event


PARSER_SCHEMA_VERSION = 1
FALLBACK_POLICY: tuple[str, ...] = ("live",)

_DECISION_URL = (
    "https://www.bi.go.id/en/publikasi/ruang-media/news-release/Default.aspx"
)
_SPEECHES_URL = (
    "https://www.bi.go.id/en/publikasi/ruang-media/"
    "pidato-dewan-gubernur/Default.aspx"
)

# Landmarks verified from live fixture captured 2026-05-26
_DECISION_LANDMARKS = ["Bank Indonesia", "news-release", "media__subtitle"]
_SPEECHES_LANDMARKS = ["Bank Indonesia", "pidato-dewan-gubernur", "media__subtitle"]

# Matches listing entries on both the decisions and speeches pages.
# The URL segment differs (news-release vs pidato-dewan-gubernur) so
# _ENTRY_RX is parameterised via the url_segment argument.
_ENTRY_RX_TEMPLATE = (
    r'<a\s+href="(?P<href>/en/publikasi/ruang-media/{segment}/Pages/[^"]+)"'
    r'[^>]*class="mt-0 media__title[^"]*"[^>]*>\s*(?P<title>[^<]+?)\s*</a>'
    r'.{{0,800}}?<div class="media__subtitle">\s*'
    r'(?P<date>\d{{1,2}}\s+\w+\s+\d{{4}})'
)


def _make_entry_rx(segment: str) -> re.Pattern:
    return re.compile(
        _ENTRY_RX_TEMPLATE.format(segment=re.escape(segment)),
        re.DOTALL,
    )


_DECISION_ENTRY_RX = _make_entry_rx("news-release")
_SPEECHES_ENTRY_RX = _make_entry_rx("pidato-dewan-gubernur")

_BASE_URL = "https://www.bi.go.id"


def _parse_listing(
    html: str,
    *,
    entry_rx: re.Pattern,
    bank_code: str,
    country: str,
    doctype: str,
    language: str,
    fetch_body: bool,
) -> Iterator[tuple]:
    for m in entry_rx.finditer(html):
        try:
            date = pd.to_datetime(m.group("date"))
        except (ValueError, TypeError):
            continue
        href = m.group("href")
        title = m.group("title").strip()
        url = href if href.startswith("http") else f"{_BASE_URL}{href}"
        text = title
        if fetch_body and url:
            try:
                body_html = fetch_with_fallback(
                    url, policy=FALLBACK_POLICY, source="bi",
                )
                body_text = strip_html(body_html)
                if body_text and len(body_text) > 100:
                    text = body_text
            except FallbackExhaustedError:
                pass
        yield (date, text, url, {
            "bank_code": bank_code,
            "country": country,
            "doctype": doctype,
            "language": language,
        })


def iter_bi_decision(*, fetch_body: bool = False) -> Iterator[tuple]:
    """Yield (date, text, url, metadata) tuples for BI monetary-policy decisions.

    Parses the BI news-release listing page
    (bi.go.id/en/publikasi/ruang-media/news-release/Default.aspx).
    The listing shows the current page of releases; to fetch older entries
    use the BI website's pagination (not yet supported).
    """
    try:
        listing = fetch_with_fallback(
            _DECISION_URL, policy=FALLBACK_POLICY, source="bi",
        )
    except FallbackExhaustedError as e:
        warnings.warn(
            f"bi.iter_bi_decision: listing fetch failed: {e}",
            UserWarning, stacklevel=2,
        )
        return

    try:
        assert_landmarks(
            listing, source="bi", expected_version=PARSER_SCHEMA_VERSION,
            landmarks=_DECISION_LANDMARKS,
        )
    except ParserSchemaMismatchError as e:
        log_event(source="bi", outcome="parser_schema_mismatch",
                  fallback_used="none")
        warnings.warn(
            f"bi.iter_bi_decision: schema mismatch: {e}",
            UserWarning, stacklevel=2,
        )
        return

    yield from _parse_listing(
        listing,
        entry_rx=_DECISION_ENTRY_RX,
        bank_code="BI",
        country="IDN",
        doctype="decision",
        language="en",
        fetch_body=fetch_body,
    )


def iter_bi_speeches(*, fetch_body: bool = False) -> Iterator[tuple]:
    """Yield (date, text, url, metadata) tuples for BI Board of Governor speeches.

    Parses the BI governor-speeches listing page
    (bi.go.id/en/publikasi/ruang-media/pidato-dewan-gubernur/Default.aspx).
    The page contains English-language speeches from the Board of Governors.
    """
    try:
        listing = fetch_with_fallback(
            _SPEECHES_URL, policy=FALLBACK_POLICY, source="bi",
        )
    except FallbackExhaustedError as e:
        warnings.warn(
            f"bi.iter_bi_speeches: listing fetch failed: {e}",
            UserWarning, stacklevel=2,
        )
        return

    try:
        assert_landmarks(
            listing, source="bi", expected_version=PARSER_SCHEMA_VERSION,
            landmarks=_SPEECHES_LANDMARKS,
        )
    except ParserSchemaMismatchError as e:
        log_event(source="bi", outcome="parser_schema_mismatch",
                  fallback_used="none")
        warnings.warn(
            f"bi.iter_bi_speeches: schema mismatch: {e}",
            UserWarning, stacklevel=2,
        )
        return

    yield from _parse_listing(
        listing,
        entry_rx=_SPEECHES_ENTRY_RX,
        bank_code="BI",
        country="IDN",
        doctype="speech",
        language="en",
        fetch_body=fetch_body,
    )


__all__ = ["iter_bi_decision", "iter_bi_speeches"]
