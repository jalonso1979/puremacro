"""EU Parliament plenary verbatim debates.

europarl.europa.eu is AWS-WAF-protected (HTTP 202 +
x-amzn-waf-action: challenge to all clients). This connector routes
all fetches through the Wayback Machine via the CDX API — same
pattern as Task 1's EUR-Lex connector. Coverage is necessarily
degraded: only plenary sessions snapshotted by Wayback (with
status=200) are reachable.

Plenary verbatim records (CRE) live at
``europarl.europa.eu/doceo/document/CRE-{term}-{YYYY}-{MM}-{DD}_{LANG}.html``.
The connector emits one record per (session × language). Per-speech
parsing is NOT in scope — session-level body gives the LUI kernel
enough text to compute a per-session uncertainty signal.

Records are 4-tuples ``(date, text, source_url, metadata)`` with
metadata: ``language``, ``session_date`` (ISO), ``term`` (8/9/10),
``source`` ('wayback').
"""
from __future__ import annotations

import re
import warnings
from datetime import date as _date
from typing import Iterator

from bs4 import BeautifulSoup

from ._fallback import fetch_with_fallback, FallbackExhaustedError
from ._http import safe_get_text
from ._schema_check import assert_landmarks, ParserSchemaMismatchError
from ._wayback import wayback_snapshot_url

# eu_parliament is currently Wayback-only — verbatim plenary CRE pages
# are JS-gated on the live endpoint (europarl.europa.eu returns HTTP 202 +
# x-amzn-waf-action: challenge to all programmatic clients). If WAF is
# lifted in the future, change to ("live", "wayback").
FALLBACK_POLICY: tuple[str, ...] = ("wayback",)

PARSER_SCHEMA_VERSION = 1


_EP_DIRECT_URL = (
    "https://www.europarl.europa.eu/doceo/document/"
    "CRE-{term}-{yyyy_mm_dd}_{lang}.html"
)

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)

# Term boundaries.
_TERM_BOUNDARIES = (
    (10, _date(2024, 7, 16)),
    (9,  _date(2019, 7, 2)),
    (8,  _date(2014, 7, 1)),
    (7,  _date(2009, 7, 14)),
)

# EP verbatim records (CRE) in doceo format start from term 7 (July 2009).
# As of 0.62.0 (Track A.3), Wayback has 200-OK snapshots for ~2/3 of
# probed Term-7 plenary dates; coverage is partial but useful. Requests
# for pre-Term-7 dates have no usable Wayback coverage under this URL
# scheme and are skipped immediately.
_EP_EARLIEST_DATE = _date(2009, 7, 14)

# Body selector candidates.  The EP verbatim pages use an old-style
# table-based layout.  After removing Wayback chrome, the outermost
# ``<table>`` is ``body > table``.  Newer EP templates may use
# ``div.doceo-content`` or similar — keep them as fallbacks.
_BODY_SELECTORS = (
    "body > table",
    "div.doceo-content",
    "div#content",
    "main",
    "div.contents",
    "div#mainContent",
)


def _term_for_date(d: _date) -> int:
    """Return the parliamentary term that includes ``d``."""
    for term, start in _TERM_BOUNDARIES:
        if d >= start:
            return term
    return 7


def _strip_wayback_chrome(html: str) -> str:
    """Remove Wayback toolbar / header chrome before parsing."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(id=re.compile(r"^(wm-|wayback-)")):
        tag.decompose()
    for cls in ("wb-autocomplete-suggestions", "wb_header"):
        for tag in soup.find_all(class_=cls):
            tag.decompose()
    return str(soup)


def _parse_ep_page(html: str, *, session_date: _date, language: str,
                    source_url: str, term: int) -> tuple | None:
    """Parse one EP plenary verbatim page into a 4-tuple record."""
    assert_landmarks(
        html, source="eu_parliament",
        expected_version=PARSER_SCHEMA_VERSION,
        landmarks=["European Parliament", "plenary"],
    )
    cleaned = _strip_wayback_chrome(html)
    soup = BeautifulSoup(cleaned, "html.parser")
    for sel in _BODY_SELECTORS:
        el = soup.select_one(sel)
        if el:
            body = el.get_text(" ", strip=True)
            if len(body) > 1000:
                metadata = {
                    "language": language.lower(),
                    "session_date": session_date.isoformat(),
                    "term": term,
                    "source": "wayback",
                }
                return (session_date, body, source_url, metadata)
    return None


def _fetch_ep_session(session_date: _date, language: str) -> tuple | None:
    """Fetch one plenary session via the governed fallback layer.

    Returns None on failure (CDX miss, fetch error, or parser failure).
    """
    term = _term_for_date(session_date)
    yyyy_mm_dd = session_date.strftime("%Y-%m-%d")
    direct = _EP_DIRECT_URL.format(
        term=term, yyyy_mm_dd=yyyy_mm_dd, lang=language.upper())
    try:
        html = fetch_with_fallback(
            direct, policy=FALLBACK_POLICY, source="eu_parliament",
        )
    except FallbackExhaustedError:
        return None
    if not html.strip():
        return None
    return _parse_ep_page(html, session_date=session_date,
                          language=language, source_url=direct, term=term)


def _enumerate_session_dates(since: _date, until: _date) -> list[_date]:
    """Enumerate plausible EP plenary session dates in [since, until]."""
    import datetime as dt
    out: list[_date] = []
    d = since
    while d <= until:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def iter_ep_debates(
    *,
    since: str = "2014-07-01",
    until: str | None = None,
    refetch: bool = False,
    languages: tuple[str, ...] = ("en", "de", "fr"),
    max_sessions: int | None = None,
) -> Iterator[tuple]:
    """Iterate EU Parliament plenary verbatim records.

    Yields one 4-tuple per (session × language) via Wayback Machine.
    Coverage is sparse — only Wayback-snapshotted sessions reachable.
    """
    import datetime as dt
    since_dt = dt.date.fromisoformat(since)
    until_dt = dt.date.fromisoformat(until) if until else dt.date.today()

    # CRE doceo records only exist from term 8 (2014-07-01). Skip earlier
    # windows entirely to avoid unnecessary CDX probes.
    if until_dt < _EP_EARLIEST_DATE:
        return
    since_dt = max(since_dt, _EP_EARLIEST_DATE)

    candidate_dates = _enumerate_session_dates(since_dt, until_dt)
    yielded = 0
    for session_date in candidate_dates:
        for lang in languages:
            try:
                rec = _fetch_ep_session(session_date, lang)
            except ParserSchemaMismatchError as e:
                from ._telemetry import log_event
                log_event(source="eu_parliament", outcome="parser_schema_mismatch",
                          fallback_used="none")
                warnings.warn(
                    f"puremacro.narrative.sources.eu_parliament: schema mismatch "
                    f"in batch {session_date.isoformat()!r} ({lang}): {e}",
                    UserWarning, stacklevel=2,
                )
                continue
            if rec is None:
                continue
            yield rec
            yielded += 1
            if max_sessions is not None and yielded >= max_sessions:
                return


__all__ = ["iter_ep_debates"]
