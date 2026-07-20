"""Reserve Bank of New Zealand (RBNZ) — combined news feed (English).

RBNZ's main RSS covers monetary-policy decisions, FSR releases, and
speeches in one feed. We default to filtering by title for monetary
policy items.
"""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://www.rbnz.govt.nz/rss/news.xml"


def iter_rbnz_decision(*, fetch_body: bool = False) -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _FEED, bank_code="RBNZ", country="NZL",
        doctype="decision", language="en",
        title_keywords=["monetary policy", "ocr", "official cash rate"],
        fetch_body=fetch_body,
    )


__all__ = ["iter_rbnz_decision"]
