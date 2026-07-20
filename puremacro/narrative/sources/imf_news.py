"""IMF press releases / news.

The official IMF news landing at ``www.imf.org/en/News`` is WAF-blocked
to ``urllib`` clients. As a stable substitute we reuse the IMF's open
``elibrary.imf.org`` search endpoint, which returns the same press
release / blog content with HTTP 200 to a plain client. Items are
scoped by a free-text query.
"""
from __future__ import annotations

import urllib.parse
from typing import Iterator

from .imf_articleiv import iter_imf_articleiv


def iter_imf_news(
    query: str,
    *,
    max_results: int = 25,
    fetch_abstracts: bool = False,
) -> Iterator[tuple]:
    """Yield (date, text, url) for IMF Staff Country Reports whose
    title contains ``query`` (case-insensitive substring search).

    Internally a thin wrapper over :func:`iter_imf_articleiv` with a
    free-text title filter and the Article-IV requirement disabled.
    Useful for surfacing IMF reports on a specific topic across
    countries (e.g. ``query="fiscal consolidation"``).
    """
    yield from iter_imf_articleiv(
        country_name=query,
        fetch_abstracts=fetch_abstracts,
        require_phrase="",
        max_items=max_results,
    )


__all__ = ["iter_imf_news"]
