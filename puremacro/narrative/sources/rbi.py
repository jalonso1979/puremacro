"""Reserve Bank of India (RBI) — main press-release RSS feed (English)."""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://rbi.org.in/Scripts/RSS.aspx"


def iter_rbi_decision(*, fetch_body: bool = True) -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _FEED, bank_code="RBI", country="IND",
        doctype="decision", language="en",
        title_keywords=["monetary policy", "repo rate", "mpc"],
        fetch_body=fetch_body,
    )


__all__ = ["iter_rbi_decision"]
