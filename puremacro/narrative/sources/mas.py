"""Monetary Authority of Singapore (MAS) — news RSS feed (English)."""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://www.mas.gov.sg/news/rss"


def iter_mas_decision(*, fetch_body: bool = False) -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _FEED, bank_code="MAS", country="SGP",
        doctype="decision", language="en",
        title_keywords=["monetary policy"],
        fetch_body=fetch_body,
    )


__all__ = ["iter_mas_decision"]
