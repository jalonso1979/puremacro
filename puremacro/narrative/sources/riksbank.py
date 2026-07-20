"""Sveriges Riksbank — monetary-policy press releases (English).

Legacy RSS retired in 2025; site is SPA-rendered. We scrape the
monetary-policy news category page via stealth Chromium:

    https://www.riksbank.se/en-gb/press-and-published/notices-and-press-releases/
        news-about-monetary-policy/

Each item appears as:
    <a href="/.../(notices|press-releases)/<yyyy>/<slug>/">
      <span class="date-and-category">dd/mm/yyyy CATEGORY</span>
      <h2>...TITLE...</h2>
    </a>

Only the most recent ~10 items are listed per fetch (pagination is JS-
driven and not followed).
"""
from __future__ import annotations

import re
import warnings
from datetime import datetime
from typing import Iterator

from ._fallback import fetch_with_fallback, FallbackExhaustedError
from ._extractors import extract_body

# riksbank is currently Playwright-only — the Riksbank site is SPA-rendered
# and requires JS execution to render content.
FALLBACK_POLICY: tuple[str, ...] = ("playwright",)


_INDEX_URL = (
    "https://www.riksbank.se/en-gb/press-and-published/"
    "notices-and-press-releases/news-about-monetary-policy/"
)
_ITEM_RX = re.compile(
    r'<a\s+href="(?P<href>/en-gb/press-and-published/notices-and-press-releases/'
    r'(?:notices|press-releases)/\d{4}/[^"]+)"[^>]*>\s*'
    r'<span\s+class="date-and-category">(?P<date>\d{2}/\d{2}/\d{4})\s+(?P<cat>[A-Za-z ]+)</span>\s*'
    r'<h2[^>]*>(?P<title>.+?)</h2>',
    re.S,
)
_TAG_RX = re.compile(r"<[^>]+>")


def _strip(text: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RX.sub(" ", text)).strip()


def iter_riksbank_decision(*, fetch_body: bool = True) -> Iterator[tuple]:
    """Yield (date, title, url, metadata) for Riksbank monetary-policy
    notices and press releases.

    When ``fetch_body`` is True (default), each row's linked HTML page is
    fetched via stealth-Playwright and its body extracted via
    :func:`extract_body` (bank_code ``"RIKSBANK"``); on success the body
    text replaces the title, otherwise the title-only fallback is preserved.
    """
    try:
        html = fetch_with_fallback(_INDEX_URL, policy=FALLBACK_POLICY, source="riksbank")
    except FallbackExhaustedError:
        warnings.warn("riksbank: fallback exhausted for index page; skipping",
                      RuntimeWarning, stacklevel=2)
        return
    for m in _ITEM_RX.finditer(html):
        href = m.group("href")
        title = _strip(m.group("title"))
        category = m.group("cat").strip()
        try:
            dt = datetime.strptime(m.group("date"), "%d/%m/%Y")
        except ValueError:
            continue
        full_url = "https://www.riksbank.se" + href
        text = title  # title-only fallback
        if fetch_body and full_url:
            try:
                body_html = fetch_with_fallback(
                    full_url, policy=FALLBACK_POLICY, source="riksbank",
                )
                body_text = extract_body(body_html, bank_code="RIKSBANK")
                if body_text and len(body_text) > 200:
                    text = body_text
            except FallbackExhaustedError:
                pass  # fall back to title
        yield (dt, text, full_url, {
            "doctype": "decision",
            "language": "en",
            "bank_code": "RIKSBANK",
            "country": "SWE",
            "category": category,
        })


__all__ = ["iter_riksbank_decision"]
