"""Reserve Bank of Australia (RBA) — monetary-policy decisions + speeches.

Original RSS feeds (``/feeds/rss.xml`` and ``/feeds/speeches/rss.xml``)
return 403 (Akamai bot wall) for plain ``requests``. The HTML archive
pages at ``/media-releases/<yyyy>/`` and ``/speeches/<yyyy>/`` are
publicly listed and admit a stealth-Playwright client.

This connector enumerates year archive pages from ``max_year`` down to
``min_year`` and parses the ``<li class="item rss-mr-item">`` rows for
date + title + URL. Each year-page is cached process-locally via
``_playwright_helper.fetch_with_playwright``.
"""
from __future__ import annotations

import re
import warnings
from datetime import datetime
from typing import Iterator

from ._fallback import fetch_with_fallback, FallbackExhaustedError
from ._extractors import extract_body

# rba is currently Playwright-only — the RBA speech pages require JS to render.
FALLBACK_POLICY: tuple[str, ...] = ("playwright",)


_DECISION_INDEX = "https://www.rba.gov.au/media-releases/{year}/"
_SPEECHES_INDEX = "https://www.rba.gov.au/speeches/{year}/"

_ITEM_RX = re.compile(
    r'<a\s+href="(?P<href>/(?:media-releases|speeches)/\d{4}/[^"]+\.html)"[^>]*>'
    r'\s*<span[^>]*itemprop="headline">(?P<title>[^<]+)</span>.*?'
    r'<time[^>]+datetime="(?P<date>\d{4}-\d{2}-\d{2})',
    re.S,
)


def _yield_year(
    year: int, *, kind: str, country: str = "AUS", bank: str = "RBA",
    fetch_body: bool = False,
) -> Iterator[tuple]:
    url = (_DECISION_INDEX if kind == "decision" else _SPEECHES_INDEX).format(year=year)
    try:
        html = fetch_with_fallback(url, policy=FALLBACK_POLICY, source="rba")
    except FallbackExhaustedError:
        warnings.warn(f"rba: fallback exhausted for {url}; skipping year {year}",
                      RuntimeWarning, stacklevel=3)
        return
    for m in _ITEM_RX.finditer(html):
        href = m.group("href")
        title = m.group("title").strip()
        date_str = m.group("date")
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        full_url = "https://www.rba.gov.au" + href
        text = title  # title-only fallback
        if fetch_body and full_url:
            try:
                body_html = fetch_with_fallback(
                    full_url, policy=FALLBACK_POLICY, source="rba",
                )
                body_text = extract_body(body_html, bank_code="RBA")
                if body_text and len(body_text) > 200:
                    text = body_text
            except FallbackExhaustedError:
                pass  # fall back to title
        yield (dt, text, full_url, {
            "doctype": kind,
            "language": "en",
            "bank_code": bank,
            "country": country,
        })


def iter_rba_decision(
    *, min_year: int = 2017, max_year: int | None = None,
    fetch_body: bool = True,
) -> Iterator[tuple]:
    """Yield (date, title, url, metadata) for RBA media releases.

    When ``fetch_body`` is True (default), each row's linked HTML page is
    fetched via stealth-Playwright and its body extracted via
    :func:`extract_body` (bank_code ``"RBA"``); on success the body text
    replaces the title, otherwise the title-only fallback is preserved.
    """
    if max_year is None:
        max_year = datetime.utcnow().year
    for year in range(max_year, min_year - 1, -1):
        yield from _yield_year(year, kind="decision", fetch_body=fetch_body)


def iter_rba_speeches(
    *, min_year: int = 2017, max_year: int | None = None,
    fetch_body: bool = True,
) -> Iterator[tuple]:
    """Yield (date, title, url, metadata) for RBA speeches.

    When ``fetch_body`` is True (default), each row's linked HTML page is
    fetched via stealth-Playwright and its body extracted via
    :func:`extract_body` (bank_code ``"RBA"``); on success the body text
    replaces the title, otherwise the title-only fallback is preserved.
    """
    if max_year is None:
        max_year = datetime.utcnow().year
    for year in range(max_year, min_year - 1, -1):
        yield from _yield_year(year, kind="speech", fetch_body=fetch_body)


__all__ = ["iter_rba_decision", "iter_rba_speeches"]
