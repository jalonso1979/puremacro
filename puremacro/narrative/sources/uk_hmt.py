"""UK HM Treasury press-release feed.

The UK government's gov.uk publishes Treasury news at
``https://www.gov.uk/government/organisations/hm-treasury`` with an
Atom feed at ``hm-treasury.atom``. This connector pulls the most recent
items; for historical batch use, supply a local CSV of pre-fetched
items.
"""
from __future__ import annotations

from typing import Iterator

from ._rss import iter_atom


_FEED = "https://www.gov.uk/government/organisations/hm-treasury.atom"


def iter_hmt_press(*, feed_url: str | None = None) -> Iterator[tuple]:
    yield from iter_atom(feed_url or _FEED)


__all__ = ["iter_hmt_press"]
