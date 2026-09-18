"""Bank of Japan speeches — HTML archive scraper.

BoJ retired the whatsnew_e.xml RSS feed during a 2025 site reorg.
Speeches now live in static HTML archive pages, one per year:

    https://www.boj.or.jp/en/about/press/koen_<yyyy>/index.htm

Each year-index renders a table of (date, speaker, title→link) rows.
This connector parses those tables and yields 4-tuples with the title
+ speaker as text. Body fetching (fetch_body=True) would follow the
linked URL and strip navigation chrome — left as a future enhancement.

Years are scraped from most-recent backward until ``min_year``; default
covers the LTUI / LWUI z-score base period (2017+).
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Iterator

from ._ratedoc import safe_get_text
from ._extractors import extract_body


_INDEX_URL = "https://www.boj.or.jp/en/about/press/koen_{year}/index.htm"

_ROW_RX = re.compile(
    r"<tr[^>]*>\s*"
    r"<td[^>]*>(?P<date>[^<]+?)</td>\s*"
    r"<td[^>]*>(?P<speaker>[^<]+?)</td>\s*"
    r"<td[^>]*>\s*<a\s+href=\"(?P<href>[^\"]+\.htm)\"[^>]*>(?P<title>.+?)</a>",
    re.S | re.I,
)
_TAG_RX = re.compile(r"<[^>]+>")
_NBSP_RX = re.compile(r"\xa0|&nbsp;", re.I)
_DATE_PARSE_FMTS = ("%b. %d, %Y", "%b %d, %Y", "%B %d, %Y")


def _parse_boj_date(raw: str) -> datetime | None:
    cleaned = _NBSP_RX.sub(" ", raw).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    for fmt in _DATE_PARSE_FMTS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def _strip_html(s: str) -> str:
    return _NBSP_RX.sub(" ", _TAG_RX.sub("", s)).strip()


def iter_boj_speeches(
    *,
    min_year: int = 2017,
    max_year: int | None = None,
    fetch_body: bool = True,
) -> Iterator[tuple]:
    """Yield (date, title+speaker, url, metadata) for BoJ speeches from
    ``min_year`` to ``max_year`` (default: 2017 → current year).

    When ``fetch_body`` is True (default), each row's linked HTML page is
    fetched and its body extracted via :func:`extract_body` (bank_code
    ``"BOJ"``); on success the body text replaces the speaker+title
    stand-in, otherwise the title-only fallback is preserved.
    """
    if max_year is None:
        max_year = datetime.utcnow().year
    for year in range(max_year, min_year - 1, -1):
        url = _INDEX_URL.format(year=year)
        try:
            html = safe_get_text(url)
        except Exception:
            continue  # year archive may not exist yet
        for m in _ROW_RX.finditer(html):
            date_raw = m.group("date")
            href = m.group("href")
            title = _strip_html(m.group("title"))
            speaker = _strip_html(m.group("speaker"))
            dt = _parse_boj_date(date_raw)
            if dt is None:
                continue
            full_url = href if href.startswith("http") else "https://www.boj.or.jp" + href
            text = f"{speaker}. {title}" if speaker else title  # title-only fallback
            if fetch_body and full_url:
                try:
                    body_html = safe_get_text(full_url)
                    body_text = extract_body(body_html, bank_code="BOJ")
                    if body_text and len(body_text) > 200:
                        text = body_text
                except Exception:
                    pass  # fall back to title
            yield (dt, text, full_url, {
                "doctype": "speech",
                "language": "en",
                "bank_code": "BoJ",
                "country": "JPN",
                "speaker": speaker,
            })


__all__ = ["iter_boj_speeches"]
