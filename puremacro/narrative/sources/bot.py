"""Bank of Thailand (BoT) — English news + MPC decisions.

Legacy RSS feed (``/content/bot/en/_jcr_content.feed``) returns 404
after the 2025 Adobe-AEM redeploy. The replacement is an AEM
``newsListing`` JSON endpoint that returns to plain ``requests`` as
long as a ``Referer`` header pointing at the news landing page is
provided:

    /content/bot/en/news-and-media/news/jcr:content/root/container/
      newslisting_copy.newsListingResults.<page_size>.p<page_index>.
      descending.<min_year>%7C<max_year>.sk0.true.json

The response payload is ``{"success": true, "results": [...]}`` with
each item carrying ``listingTitle``, ``issueDt`` (e.g. ``"29 Apr 2026"``),
``pagePath``, ``tags``, and ``releaseNumber``.

Discovered by XHR-tracing the BoT SPA with stealth-Playwright.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterator

import requests


_BASE = "https://www.bot.or.th"
_LISTING_URL_TPL = (
    "{base}/content/bot/en/news-and-media/news/jcr:content/root/container/"
    "newslisting_copy.newsListingResults.{page_size}.p{page_index}."
    "descending.{min_year}%7C{max_year}.sk0.true.json"
)
_REFERER = "https://www.bot.or.th/en/news-and-media/news.html"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)
_DATE_FMT = "%d %b %Y"


def _fetch_page(page_index: int = 0, page_size: int = 500,
                min_year: int = 1997, max_year: int | None = None) -> list[dict]:
    if max_year is None:
        max_year = datetime.utcnow().year
    url = _LISTING_URL_TPL.format(
        base=_BASE, page_size=page_size, page_index=page_index,
        min_year=min_year, max_year=max_year,
    )
    try:
        r = requests.get(
            url,
            headers={"User-Agent": _USER_AGENT, "Referer": _REFERER},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("results", []) if data.get("success") else []
    except Exception:
        return []


def _parse_date(raw: str) -> datetime | None:
    try:
        return datetime.strptime(raw, _DATE_FMT)
    except (ValueError, TypeError):
        return None


def _is_mpc(item: dict) -> bool:
    """Heuristic: an MPC decision has 'Monetary Policy Committee' in
    its title or 'MPC' in tags."""
    title = (item.get("listingTitle") or "").lower()
    tags = [t.lower() for t in (item.get("tags") or [])]
    return (
        "monetary policy committee" in title
        or "mpc" in title.split()
        or any("monetary policy" in t for t in tags)
    )


def iter_bot_news(
    *, page_size: int = 500, min_year: int = 1997,
    fetch_body: bool = False,
) -> Iterator[tuple]:
    """Yield (date, title, url, metadata) for all BoT English news items."""
    for it in _fetch_page(page_size=page_size, min_year=min_year):
        dt = _parse_date(it.get("issueDt", ""))
        if dt is None:
            continue
        title = (it.get("listingTitle") or "").strip()
        url = it.get("pagePath") or ""
        tags = it.get("tags") or []
        release = it.get("releaseNumber") or ""
        yield (dt, title, url, {
            "doctype": "news",
            "language": "en",
            "bank_code": "BoT",
            "country": "THA",
            "tags": tags,
            "release_number": release,
        })


def iter_bot_decision(
    *, page_size: int = 500, min_year: int = 1997,
    fetch_body: bool = False,
) -> Iterator[tuple]:
    """Yield (date, title, url, metadata) for BoT MPC monetary-policy
    decisions only. Filters the general news feed by title/tag heuristics."""
    for it in _fetch_page(page_size=page_size, min_year=min_year):
        if not _is_mpc(it):
            continue
        dt = _parse_date(it.get("issueDt", ""))
        if dt is None:
            continue
        title = (it.get("listingTitle") or "").strip()
        url = it.get("pagePath") or ""
        yield (dt, title, url, {
            "doctype": "decision",
            "language": "en",
            "bank_code": "BoT",
            "country": "THA",
            "tags": it.get("tags") or [],
            "release_number": it.get("releaseNumber") or "",
        })


__all__ = ["iter_bot_decision", "iter_bot_news"]
