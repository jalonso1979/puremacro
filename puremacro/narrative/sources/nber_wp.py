"""NBER Working Papers — RSS feed (roadmap item A-ext.2a).

The National Bureau of Economic Research publishes the most-recent
working papers as an RSS 2.0 feed at:

    https://www.nber.org/rss/new.xml

Each item contains the paper title (with authors after `--`), the
abstract in the description, and a URL to the working-paper page
(``nber.org/papers/wNNNNN``). Roughly 30 papers/week appear in the
feed; pagination isn't supported (NBER's archive listing pages can be
scraped separately if historical depth is needed).

This is a *leading-indicator* text corpus for the LTUI / LWUI line:
research literature mentions emerging risks 1–4 quarters before they
appear in central-bank communication.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Iterator

from ..._http import safe_get_bytes


_FEED = "https://www.nber.org/rss/new.xml"


def iter_nber_wp(*, fetch_body: bool = False) -> Iterator[tuple]:
    """Yield (date, title+abstract, url, metadata) for the latest NBER WP.

    NBER's RSS feed does not include per-item ``<pubDate>`` elements.
    As a pragmatic workaround we stamp every record with the current
    UTC date — the feed is by construction "the latest" papers (~30
    items, refreshed weekly), so for quarterly LTUI/LWUI aggregation
    this places them in the current quarter, which is correct ±1 week.

    Future enhancement: follow each ``link`` to the paper page and
    extract ``citation_publication_date`` from the HTML head meta tags
    (one extra GET per paper).

    Title format: ``"<Title> -- by <Author>, <Author>, ..."``. The
    author list is preserved in metadata under ``"authors"``.
    """
    try:
        body = safe_get_bytes(_FEED)
    except Exception:
        return
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return
    today = datetime.utcnow()
    for item in root.findall(".//item"):
        title = item.findtext("title", default="") or ""
        desc = item.findtext("description", default="") or ""
        link = item.findtext("link", default="") or ""
        if not (title or desc):
            continue
        m = re.match(r"^(?P<title>.+?)\s+--\s+by\s+(?P<authors>.+?)$",
                     title.strip(), re.S)
        if m:
            paper_title = m.group("title").strip()
            authors = m.group("authors").strip()
        else:
            paper_title = title.strip()
            authors = ""
        text = f"{paper_title}. {desc}" if desc else paper_title
        yield (today, text, link, {
            "doctype": "working_paper",
            "language": "en",
            "bank_code": "NBER",
            "country": "MULTI",
            "authors": authors,
        })


__all__ = ["iter_nber_wp"]
