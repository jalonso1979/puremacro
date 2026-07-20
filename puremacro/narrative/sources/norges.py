"""Norges Bank — news feed (English mirror)."""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://www.norges-bank.no/en/news-events/news-publications/?rss=true"


def iter_norges_decision(*, fetch_body: bool = True) -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _FEED, bank_code="NORGES", country="NOR",
        doctype="decision", language="en",
        title_keywords=["policy rate", "monetary policy", "key policy"],
        fetch_body=fetch_body,
    )


__all__ = ["iter_norges_decision"]
