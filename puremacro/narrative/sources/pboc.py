"""People's Bank of China (PBoC) — English mirror.

PBoC does not publish RSS. We scrape the English news / speeches
archive index pages. URLs updated 2026-05 (the old ``3688110/3688215``
prefix was retired during a 2025 site reorg).

    http://www.pbc.gov.cn/english/130721/index.html  — news / press releases
    http://www.pbc.gov.cn/english/130724/index.html  — speeches

Each index page exposes ~30 most-recent items as table rows containing
an <a href="..." title="..."> link followed by a YYYY-MM-DD date cell.
Pagination is JS-driven and not followed.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Iterator

from ._ratedoc import safe_get_text
from ._extractors import extract_body


_BASE = "http://www.pbc.gov.cn"
_INDEX_URLS = {
    "press":    f"{_BASE}/english/130721/index.html",
    "speeches": f"{_BASE}/english/130724/index.html",
}

_ROW_RX = re.compile(
    r"<a\s+href=\"(?P<href>[^\"]+\.html)\"[^>]*\stitle=\"(?P<title>[^\"]+)\"[^>]*>"
    r".*?<td[^>]*>(?P<date>20\d{2}-\d{2}-\d{2})</td>",
    re.S | re.I,
)


def _yield_from(doctype: str, url: str, *, fetch_body: bool) -> Iterator[tuple]:
    try:
        html = safe_get_text(url)
    except Exception:
        return
    for m in _ROW_RX.finditer(html):
        href = m.group("href")
        title = m.group("title").strip()
        date_raw = m.group("date")
        try:
            dt = datetime.strptime(date_raw, "%Y-%m-%d")
        except ValueError:
            continue
        full_url = href if href.startswith("http") else _BASE + href
        text = title  # title-only fallback
        if fetch_body and full_url:
            try:
                body_html = safe_get_text(full_url)
                body_text = extract_body(body_html, bank_code="PBOC")
                if body_text and len(body_text) > 200:
                    text = body_text
            except Exception:
                pass  # fall back to title
        yield (dt, text, full_url, {
            "doctype": doctype,
            "language": "en",
            "bank_code": "PBoC",
            "country": "CHN",
        })


def iter_pboc_decision(*, fetch_body: bool = True) -> Iterator[tuple]:
    """Yield (date, title, url, metadata) for PBoC English press / news items.

    When ``fetch_body`` is True (default), each row's linked HTML page is
    fetched and its body extracted via :func:`extract_body` (bank_code
    ``"PBOC"``); on success the body text replaces the title, otherwise
    the title-only fallback is preserved.
    """
    yield from _yield_from("decision", _INDEX_URLS["press"],
                           fetch_body=fetch_body)


def iter_pboc_speeches(*, fetch_body: bool = True) -> Iterator[tuple]:
    """Yield (date, title, url, metadata) for PBoC English speeches.

    When ``fetch_body`` is True (default), each row's linked HTML page is
    fetched and its body extracted via :func:`extract_body` (bank_code
    ``"PBOC"``); on success the body text replaces the title, otherwise
    the title-only fallback is preserved.
    """
    yield from _yield_from("speech", _INDEX_URLS["speeches"],
                           fetch_body=fetch_body)


__all__ = ["iter_pboc_decision", "iter_pboc_speeches"]
