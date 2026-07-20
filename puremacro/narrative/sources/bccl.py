"""Banco Central de Chile (BCCh) — monetary-policy press feed (Spanish)."""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://www.bcentral.cl/-/rss-feed-prensa"


def iter_bccl_decision(*, fetch_body: bool = False) -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _FEED, bank_code="BCCH", country="CHL",
        doctype="decision", language="es",
        fetch_body=fetch_body,
    )


__all__ = ["iter_bccl_decision"]
