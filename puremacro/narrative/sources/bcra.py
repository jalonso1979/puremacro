"""Banco Central de la República Argentina (BCRA) — press releases (Spanish)."""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://www.bcra.gob.ar/rss/Prensa.aspx"


def iter_bcra_decision(*, fetch_body: bool = False) -> Iterator[tuple]:
    yield from iter_rss_filtered(
        _FEED, bank_code="BCRA", country="ARG",
        doctype="decision", language="es",
        fetch_body=fetch_body,
    )


__all__ = ["iter_bcra_decision"]
