"""UK Office for Budget Responsibility (OBR) feed.

The OBR publishes Economic and Fiscal Outlooks, Forecast Evaluation
Reports, Fiscal Sustainability Reports, and policy commentary. The RSS
feed is at https://obr.uk/feed/.
"""
from __future__ import annotations

from typing import Iterator

from ._rss import iter_rss


_FEED = "https://obr.uk/feed/"


def iter_obr_publications(*, feed_url: str | None = None) -> Iterator[tuple]:
    yield from iter_rss(feed_url or _FEED)


__all__ = ["iter_obr_publications"]
