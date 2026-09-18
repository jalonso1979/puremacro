"""US Treasury press-release feed.

The Treasury publishes press releases at
``https://home.treasury.gov/news/press-releases``. The Atom/RSS feed
that earlier puremacro versions targeted (``/rss/press-releases.xml``)
went dead in 2026; this module instead scrapes the listing page HTML.
The listing markup is stable across redesigns and stdlib regex parsing
is sufficient — no new dependencies introduced.

For history beyond what the live listing exposes, the user can supply
a local CSV of Treasury releases (see :mod:`local_csv`).
"""
from __future__ import annotations

import re
from typing import Iterator

import pandas as pd

from ._http import safe_get_text


_BASE = "https://home.treasury.gov/news/press-releases"


# The listing renders each release as a card with a date, a headline
# anchor, and an excerpt. The exact tag wrapping rotates between
# Drupal redesigns, so we anchor on the most stable element: the
# ``<time datetime="...">`` element with a sibling/descendant anchor.
_ITEM_RX = re.compile(
    r"<article[^>]*>(.*?)</article>", flags=re.IGNORECASE | re.DOTALL
)
_TITLE_RX = re.compile(
    r"<h[23][^>]*>\s*<a[^>]*>(.*?)</a>", flags=re.IGNORECASE | re.DOTALL
)
_DATE_RX = re.compile(
    r"<time[^>]*datetime=\"([0-9\-T:Z+.]+)\"", flags=re.IGNORECASE
)
_HREF_RX = re.compile(
    r"<a[^>]*href=\"(/news/press-releases/[^\"]+)\"", flags=re.IGNORECASE
)


def iter_treasury_press(*, max_pages: int = 5) -> Iterator[tuple]:
    """Yield (pubdate, title, link) records for recent Treasury releases.

    Pagination uses the Drupal default ``?page=N`` query parameter
    (zero-indexed on this site).
    """
    for page in range(0, max_pages):
        url = _BASE + (f"?page={page}" if page > 0 else "")
        try:
            html = safe_get_text(url)
        except Exception:
            return
        items = _ITEM_RX.findall(html)
        if not items:
            return
        for item in items:
            title_m = _TITLE_RX.search(item)
            date_m = _DATE_RX.search(item)
            href_m = _HREF_RX.search(item)
            if not title_m or not date_m:
                continue
            title = re.sub(r"<[^>]+>", " ", title_m.group(1)).strip()
            try:
                date = pd.Timestamp(date_m.group(1))
            except Exception:
                continue
            link = href_m.group(1) if href_m else ""
            if link.startswith("/"):
                link = "https://home.treasury.gov" + link
            yield date, title, link


__all__ = ["iter_treasury_press"]
