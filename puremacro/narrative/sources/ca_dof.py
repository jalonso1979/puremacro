"""Canadian Department of Finance / Parliamentary Budget Officer (PBO).

Canada's Department of Finance maintains an Atom feed under
https://www.canada.ca/en/department-finance/news.atom and the PBO
publishes at https://www.pbo-dpb.ca/en/blog. This connector aggregates
the DoF Atom feed.
"""
from __future__ import annotations

from typing import Iterator

from ._rss import iter_atom


# Government of Canada's IO API is the canonical news endpoint
# (``api.io.canada.ca``); the older ``canada.ca/.../news.atom`` URLs
# silently return zero items. The dept slug is ``departmentfinance``
# (no "of"), and a date floor must be supplied. Verified live, May 2026.
_FEED = (
    "https://api.io.canada.ca/io-server/gc/news/en/v2"
    "?dept=departmentfinance&type=newsreleases"
    "&sort=publishedDate&orderBy=desc"
    "&publishedDate%3E=2015-01-01&pick=200&format=atom"
)


def iter_dof_press(*, feed_url: str | None = None) -> Iterator[tuple]:
    """Yield Canadian Department of Finance press items.

    The Canadian government has migrated press feeds across multiple
    URLs over the years; if the default 404s, look for the current
    Atom URL at ``https://www.canada.ca/en/department-finance/news.html``
    and pass via ``feed_url=``.
    """
    yield from iter_atom(feed_url or _FEED)


__all__ = ["iter_dof_press"]
