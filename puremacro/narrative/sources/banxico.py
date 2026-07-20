"""Banco de México (Banxico) — monetary-policy announcements (Spanish).

The legacy RSS feed (``rss/feeds/comunicados.xml``) was retired in 2025.
We instead scrape the canonical announcement listing page, which renders
all decisions in HTML with structured date + title + PDF link rows:

    https://www.banxico.org.mx/publicaciones-y-prensa/
        anuncios-de-las-decisiones-de-politica-monetaria/
        anuncios-politica-monetaria-t.html

The listing covers ~235 decisions back to the early-2000s. Each row has:
- A ``HREF`` to the PDF announcement
- A title in the surrounding ``aria-label`` attribute
- A short Spanish date string ``dd/mm/yy``

Each linked file is a PDF (the listing page is the only HTML surface;
there are no per-decision HTML landing pages). When ``fetch_body=True``
(default), we mirror the α-6 BCB PDF-pivot: download the linked PDF
via ``safe_get_bytes`` and extract text with ``extract_body_from_pdf``,
falling back to the listing-derived title on any failure.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Iterator

from ..._http import safe_get_text
from ._extractors import extract_body_from_pdf
from ._http import safe_get_bytes


_LISTING_URL = (
    "https://www.banxico.org.mx/publicaciones-y-prensa/"
    "anuncios-de-las-decisiones-de-politica-monetaria/"
    "anuncios-politica-monetaria-t.html"
)
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)
_PDF_RX = re.compile(
    r'HREF="(/publicaciones-y-prensa/anuncios-de-las-decisiones-de-politica-monetaria/\{[^"]+\}\.pdf)"',
    re.I,
)
_DATE_RX = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4})")
_TAG_RX = re.compile(r"<[^>]+>")


def _strip(text: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RX.sub(" ", text)).strip()


def _parse_date(raw: str) -> datetime | None:
    """Banxico dates are 'dd/mm/yy' (2-digit year). Years < 50 → 20yy,
    years 50-99 → 19yy (decisions don't predate 1995, so safe)."""
    try:
        d, m, y = (int(x) for x in raw.split("/"))
    except ValueError:
        return None
    if y < 100:
        y = 2000 + y if y < 50 else 1900 + y
    try:
        return datetime(y, m, d)
    except ValueError:
        return None


def _maybe_fetch_pdf_body(url: str, fetch_body: bool) -> str | None:
    """Fetch + extract text from a Banxico PDF URL.

    Returns the extracted text if successful, or None when:
    - fetch_body is False
    - url doesn't end in .pdf
    - fetch fails (network)
    - pypdf isn't installed or PDF parse fails
    - extracted text is shorter than the minimum threshold.
    """
    if not fetch_body or not url:
        return None
    if not url.lower().endswith(".pdf"):
        return None
    try:
        return extract_body_from_pdf(safe_get_bytes(url))
    except Exception:
        return None


def iter_banxico_decision(*, fetch_body: bool = True) -> Iterator[tuple]:
    """Yield (date, text, url, metadata) for Banxico monetary-policy
    decisions.

    When ``fetch_body=True`` (default), each linked PDF is downloaded
    and parsed via ``extract_body_from_pdf``; on any failure the
    listing-derived title is returned instead.
    """
    try:
        html = safe_get_text(_LISTING_URL, user_agent=_USER_AGENT)
    except Exception:
        return
    for m in _PDF_RX.finditer(html):
        href = m.group(1)
        ctx = html[max(0, m.start() - 600):m.start()]
        ctx_text = _strip(ctx)
        date_m = _DATE_RX.search(ctx_text)
        if not date_m:
            continue
        dt = _parse_date(date_m.group(1))
        if dt is None:
            continue
        # Title: last sentence-ish chunk before the link.
        title = ctx_text[-300:].strip()
        title = re.sub(r"^.*?(\d{1,2}/\d{1,2}/\d{2,4})\s*", "", title)
        full_url = "https://www.banxico.org.mx" + href
        text = title
        pdf_text = _maybe_fetch_pdf_body(full_url, fetch_body)
        if pdf_text:
            text = pdf_text
        yield (dt, text, full_url, {
            "doctype": "decision",
            "language": "es",
            "bank_code": "BANXICO",
            "country": "MEX",
        })


__all__ = ["iter_banxico_decision"]
