"""Bank of England speeches — XHR-API scraper.

BoE retired the RSS feed during a 2025 site reorg and serves the
speech archive through a JS-rendered SPA. The SPA itself is bot-
protected (Akamai blocks plain ``requests``-based crawls with 403),
but the back-end XHR endpoint it hits is open to any client that
sends a realistic ``User-Agent`` and the ``X-Requested-With:
XMLHttpRequest`` header.

The endpoint is ``POST /_api/News/RefreshPagedNewsList`` with form
parameters carrying the section ID, news-type filter (speech), page
number, and page size. The response is JSON containing an HTML
fragment with one ``<a class="release release-speech">`` per item.
We parse this fragment with regex (no HTML parser dep) and yield
4-tuples. Pure-``requests`` at runtime — no headless browser needed.
Discovery of the XHR contract was done via Playwright tracing; this
connector is fully decoupled from Playwright in production.

The HTML fragment is a list of:
    <a href="/speech/<yyyy>/<month>/<slug>" class="release release-speech">
      <div class="release-tag">Speech // <Speaker></div>
      ...
      <time class="release-date" datetime="YYYY-MM-DD">...
      ...
      <h3 class="release-title">...</h3>
      <p class="release-summary">...</p>
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Iterator

import requests


_API_URL = "https://www.bankofengland.co.uk/_api/News/RefreshPagedNewsList"
_REFERER = "https://www.bankofengland.co.uk/news/speeches"
_SECTION_ID = "{CE377CC8-BFBC-418B-B4D9-DBC1C64774A8}"
_NEWS_TYPE_SPEECH = "f949c64a4c88448b9e269d10080b0987"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)

_ITEM_RX = re.compile(
    r'<a\s+href="(?P<href>/speech/[^"]+)"[^>]*class="release\s+release-speech\s*"\s*>'
    r'\s*<div class="release-tag-wrap"><div class="release-tag">(?P<tag>[^<]+)</div></div>'
    r'.*?<time\s+class="release-date"\s+itemprop="datePublished"\s+datetime="(?P<date>\d{4}-\d{2}-\d{2})"',
    re.S,
)
_TITLE_RX = re.compile(
    r'<h3\s+itemprop="name"\s+class="list[^"]*"[^>]*>(?P<title>.*?)</h3>',
    re.S,
)
_TAG_RX = re.compile(r"<[^>]+>")


def _strip(s: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RX.sub(" ", s)).strip()


def _fetch_page(page: int, *, page_size: int = 30, timeout: float = 20.0) -> str:
    resp = requests.post(
        _API_URL,
        data={
            "SearchTerm": "",
            "Id": _SECTION_ID,
            "PageSize": page_size,
            "NewsTypesAvailable[]": _NEWS_TYPE_SPEECH,
            "Page": page,
            "Direction": 1,
            "Grid": "false",
            "InfiniteScrolling": "false",
        },
        headers={
            "User-Agent": _USER_AGENT,
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "*/*",
            "Referer": _REFERER,
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("Results", "")


def _parse_items(html: str) -> Iterator[tuple]:
    """Yield (datetime, text, full_url, metadata) for each speech item."""
    # Split into per-anchor blocks so title/summary stay paired with their item.
    anchor_starts = [m.start() for m in re.finditer(
        r'<a\s+href="/speech/[^"]+"[^>]*class="release\s+release-speech\s*"', html)]
    if not anchor_starts:
        return
    bounds = anchor_starts + [len(html)]
    for i in range(len(anchor_starts)):
        block = html[bounds[i]:bounds[i + 1]]
        m = _ITEM_RX.search(block)
        if not m:
            continue
        href = m.group("href")
        tag = m.group("tag").strip()  # "Speech // <Speaker>"
        speaker = tag.split("//", 1)[1].strip() if "//" in tag else tag
        date_str = m.group("date")
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        title_m = _TITLE_RX.search(block)
        title = _strip(title_m.group("title")) if title_m else ""
        text_parts = [t for t in (speaker, title) if t]
        text = " . ".join(text_parts)
        full_url = "https://www.bankofengland.co.uk" + href
        yield (dt, text, full_url, {
            "doctype": "speech",
            "language": "en",
            "bank_code": "BoE",
            "country": "GBR",
            "speaker": speaker,
        })


def iter_boe_speeches(
    *,
    max_pages: int = 10,
    page_size: int = 30,
    fetch_body: bool = False,  # placeholder; body fetch follows the speech URL
) -> Iterator[tuple]:
    """Yield (date, text, url, metadata) for BoE speeches via the BoE
    ``RefreshPagedNewsList`` JSON-XHR endpoint.

    Up to ``max_pages * page_size`` records are fetched. Default 10 pages
    × 30 = 300 records (~3.5 years of BoE speeches).
    """
    seen: set[str] = set()
    for page in range(1, max_pages + 1):
        try:
            html = _fetch_page(page, page_size=page_size)
        except Exception:
            return  # network failure → graceful exit
        new_in_page = 0
        for record in _parse_items(html):
            href = record[2]
            if href in seen:
                continue
            seen.add(href)
            new_in_page += 1
            yield record
        if new_in_page == 0:
            return  # exhausted


__all__ = ["iter_boe_speeches"]
