"""US Congressional Budget Office (CBO) publications.

CBO publishes ~50-100 reports/year on budget, economic outlook,
specific legislation, and topics. Source:

  https://www.cbo.gov/publications/all/rss.xml

Each RSS item links to a per-publication HTML page on cbo.gov; each
HTML page links to one or more PDF body files. The connector fetches
the RSS, follows per-publication links, downloads PDFs, and extracts
body text via pdfplumber.

Per-document granularity only — CBO section structure varies too
much across topic categories to canonicalize cheaply. Metadata carries
CBO's RSS-supplied title + publication ID + topic if discoverable
from the publication page.

Coverage: 2005-present (~20 years; cap configurable via ``since``).
Pre-2005 backfill via GovInfo or FRASER queued for a follow-up release.

Bot-protection note
-------------------
As of 2025-2026, cbo.gov deploys DataDome bot protection that blocks
plain HTTP clients on publication HTML pages and PDF downloads (HTTP 403
regardless of User-Agent). The connector handles this gracefully:

- RSS feed: always accessible (no DataDome on RSS endpoint).
- Publication HTML + PDF: attempted via ``safe_get_*``; on 403/failure
  the record is skipped and iteration continues to the next item.

In controlled environments (local fetch with a browser session or
Playwright with a valid DataDome cookie), callers may monkey-patch
``safe_get_text`` / ``safe_get_bytes`` with a wrapper that injects the
cookie before passing to ``iter_cbo``.
"""
from __future__ import annotations

import io
import re
import warnings
import xml.etree.ElementTree as ET
from datetime import date as _date, datetime
from typing import Iterator

import pdfplumber

from ._fallback import fetch_with_fallback, FallbackExhaustedError
from ._http import safe_get_bytes, safe_get_text
from ._schema_check import assert_landmarks, ParserSchemaMismatchError
from ._wayback import wayback_snapshot_url

# us_cbo: the RSS feed works live; per-publication HTML pages and PDF
# links are DataDome-blocked on direct fetch and sometimes need Wayback.
FALLBACK_POLICY: tuple[str, ...] = ("live", "wayback")

PARSER_SCHEMA_VERSION = 1


_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)

_CBO_RSS_URL = "https://www.cbo.gov/publications/all/rss.xml"
_CBO_BASE = "https://www.cbo.gov"


def _parse_rss(xml_text: str) -> list[dict]:
    """Parse CBO RSS XML into a list of publication dicts.

    Each dict: {title, link, description, pubDate (datetime), pub_id (int)}.
    """
    assert_landmarks(
        xml_text, source="us_cbo",
        expected_version=PARSER_SCHEMA_VERSION,
        landmarks=["<rss", "Congressional Budget Office"],
    )
    root = ET.fromstring(xml_text)
    channel = root.find("channel")
    if channel is None:
        return []
    items: list[dict] = []
    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = (item.findtext("description") or "").strip()
        pub_date_text = (item.findtext("pubDate") or "").strip()
        try:
            pub_date = datetime.strptime(
                pub_date_text, "%a, %d %b %Y %H:%M:%S %z")
        except ValueError:
            pub_date = None
        pub_id_m = re.search(r"/publication/(\d+)", link)
        pub_id = int(pub_id_m.group(1)) if pub_id_m else None
        items.append({
            "title": title, "link": link, "description": description,
            "pubDate": pub_date, "pub_id": pub_id,
        })
    return items


def _find_pdf_link(pub_html: str, pub_id: int | None = None) -> str | None:
    """Scrape per-publication page HTML for the body PDF link.

    Handles both direct cbo.gov hrefs (``/system/files/...``) and
    Wayback Machine wrapper URLs (``/web/TIMESTAMP/https://www.cbo.gov/...``)
    that appear in Internet Archive snapshots used as test fixtures.

    Preference order:
    1. A PDF whose URL contains the publication ID and is not an executive
       summary or appendix (i.e. the main body document).
    2. The first PDF link found on the page.
    """
    pdfs = re.findall(r'href="([^"]+\.pdf)"', pub_html, re.I)
    if not pdfs:
        return None

    # Normalize Wayback Machine wrappers: /web/TIMESTAMP/https://... -> https://...
    normalized: list[str] = []
    for href in pdfs:
        m_wb = re.match(r"^/web/\d+/(https?://.+)$", href)
        if m_wb:
            normalized.append(m_wb.group(1))
        elif href.startswith("http"):
            normalized.append(href)
        else:
            normalized.append(_CBO_BASE + href)

    # Prefer the main body document: contains pub_id and is not a summary/appendix
    if pub_id is not None:
        pub_id_str = str(pub_id)
        for href in normalized:
            basename = href.rsplit("/", 1)[-1].lower()
            if pub_id_str in href and "summary" not in basename and "appendix" not in basename:
                return href

    # Fall back to first link
    return normalized[0]


def _extract_cbo_pdf_text(pdf_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(pages).strip()


def _try_fetch_text(url: str) -> str:
    """Wrapper: return text on success, empty string on any failure."""
    try:
        return safe_get_text(url)
    except Exception:
        return ""


def _try_fetch_bytes(url: str) -> bytes:
    """Wrapper: return bytes on success, empty bytes on any failure."""
    try:
        return safe_get_bytes(url)
    except Exception:
        return b""


def _fetch_publication_body(pub_url: str,
                             pub_id: int | None = None) -> tuple[str, str]:
    """Return (body_text, pdf_url) for one publication. Empty body on failure.

    On 2025-2026, cbo.gov publication HTML + PDFs are DataDome-blocked
    (direct ``safe_get_text`` returns an empty body). The governed fallback
    layer (FALLBACK_POLICY = ("live", "wayback")) handles the live→Wayback
    retry for the HTML page. PDF bytes are fetched separately (bytes-only
    path; fetch_with_fallback returns text) with an inline Wayback retry.
    """
    try:
        html = fetch_with_fallback(
            pub_url, policy=FALLBACK_POLICY, source="us_cbo",
        )
    except FallbackExhaustedError:
        return ("", "")
    if not html.strip():
        return ("", "")
    pdf_url = _find_pdf_link(html, pub_id=pub_id)
    if not pdf_url:
        return ("", "")
    pdf_bytes = _try_fetch_bytes(pdf_url)
    if not pdf_bytes:
        wb_pdf = wayback_snapshot_url(pdf_url, user_agent=_USER_AGENT)
        if wb_pdf:
            pdf_bytes = _try_fetch_bytes(wb_pdf)
    if not pdf_bytes:
        return ("", pdf_url)
    return (_extract_cbo_pdf_text(pdf_bytes), pdf_url)


def iter_cbo(
    *,
    since: str = "2005-01-01",
    until: str | None = None,
    refetch: bool = False,
    max_items: int | None = None,
) -> Iterator[tuple]:
    """Iterate CBO publication records.

    Parameters
    ----------
    since, until : ISO date strings. ``until=None`` means present.
        First reliable RSS coverage starts ~2005.
    refetch : reserved.
    max_items : optional cap (useful for tests / dev).

    Yields
    ------
    4-tuples ``(date, text, source_url, metadata)``.

    Notes
    -----
    cbo.gov deploys DataDome bot protection on publication HTML pages and
    PDF downloads. In environments where the HTTP fetcher is blocked (HTTP
    403), ``iter_cbo`` silently skips inaccessible records and yields
    nothing rather than raising. The RSS feed itself is always accessible.
    """
    import datetime as dt
    since_dt = dt.date.fromisoformat(since)
    until_dt = dt.date.fromisoformat(until) if until else dt.date.today()

    try:
        rss_text = safe_get_text(_CBO_RSS_URL)
    except Exception:
        return
    try:
        items = _parse_rss(rss_text)
    except ParserSchemaMismatchError as e:
        from ._telemetry import log_event
        log_event(source="us_cbo", outcome="parser_schema_mismatch",
                  fallback_used="none")
        warnings.warn(
            f"puremacro.narrative.sources.us_cbo: schema mismatch "
            f"in RSS feed {_CBO_RSS_URL!r}: {e}",
            UserWarning, stacklevel=2,
        )
        return

    yielded = 0
    for item in items:
        if item["pubDate"] is None:
            continue
        pub_d = item["pubDate"].date()
        if pub_d < since_dt or pub_d > until_dt:
            continue
        try:
            body, pdf_url = _fetch_publication_body(
                item["link"], pub_id=item["pub_id"]
            )
        except Exception:
            continue
        if not body.strip():
            continue
        metadata = {
            "title": item["title"],
            "pub_id": item["pub_id"],
            "description": item["description"],
            "pdf_url": pdf_url,
        }
        yield (pub_d, body, item["link"], metadata)
        yielded += 1
        if max_items is not None and yielded >= max_items:
            return


__all__ = ["iter_cbo"]
