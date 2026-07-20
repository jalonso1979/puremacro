"""US Department of Defense daily contracts.

The DoD publishes daily contract awards at
``https://www.defense.gov/News/Contracts/``. This is rich source
material for *defense investment* events (procurement) and *defense
consumption* events (operating expenditure). The site does not expose
a structured API; instead we parse the listing HTML to extract dates
and contract abstracts. The HTML is stable enough that simple regex
parsing works, and stdlib only.

For historical batch use, point ``iter_dod_contracts`` at a local CSV
that already contains the parsed data (see ``local_csv``).
"""
from __future__ import annotations

import re
import urllib.parse
from typing import Iterator

import pandas as pd

from ._http import safe_get_text


_BASE = "https://www.defense.gov/News/Contracts/Filter-Contracts/"

# defense.gov WAF blocks the default Mozilla/5.0 (puremacro/narrative)
# string; pass a realistic browser UA. See narrative/sources/RETRY_POLICY.md §7.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# Contracts page items follow a "<a class=...><h3>Title</h3></a>" template
# and date headers like "Sept. 25, 2024". We pluck items via regex; a
# more robust parser would use html.parser, but for the listing page
# regex is enough.
_ITEM_RX = re.compile(
    r"<article[^>]*>(.*?)</article>", flags=re.IGNORECASE | re.DOTALL
)
_TITLE_RX = re.compile(r"<h3[^>]*>(.*?)</h3>", flags=re.IGNORECASE | re.DOTALL)
_DATE_RX = re.compile(
    r"<time[^>]*datetime=\"([0-9\-T:Z+]+)\"", flags=re.IGNORECASE
)
_HREF_RX = re.compile(r"<a[^>]*href=\"([^\"]+)\"", flags=re.IGNORECASE)


def iter_dod_contracts(*, max_pages: int = 5) -> Iterator[tuple]:
    """Yield (date, title-as-text, link) for recent DoD contract pages.

    Pagination uses the site's standard ``?Page=N`` query parameter.
    """
    for page in range(1, max_pages + 1):
        url = _BASE + (f"?Page={page}" if page > 1 else "")
        try:
            html = safe_get_text(url, user_agent=_BROWSER_UA)
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
                link = "https://www.defense.gov" + link
            yield date, title, link


__all__ = ["iter_dod_contracts"]
