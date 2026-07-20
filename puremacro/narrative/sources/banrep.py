"""Banco de la República (Colombia, BanRep) — press releases (Spanish)."""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://www.banrep.gov.co/rss-comunicados"


def iter_banrep_decision(*, fetch_body: bool = False) -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _FEED, bank_code="BANREP", country="COL",
        doctype="decision", language="es",
        fetch_body=fetch_body,
    )


__all__ = ["iter_banrep_decision"]
