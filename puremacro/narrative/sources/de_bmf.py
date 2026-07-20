"""German Federal Ministry of Finance (BMF) news feed.

BMF publishes press releases at https://www.bundesfinanzministerium.de/
with English and German RSS feeds. We use the English-language feed by
default; pass ``language="de"`` for the German-language feed (then
LLM-based scoring is needed since the keyword lexicon is English-only).
"""
from __future__ import annotations

from typing import Iterator

from ._rss import iter_rss


# BMF only publishes a German-language press feed; the historical
# English-language RSS endpoints have all been retired (verified May 2026).
# For English-language scoring of these items, run the LLM scorer with
# its prompt in German or pre-translate.
_FEED_DE = (
    "https://www.bundesfinanzministerium.de/SiteGlobals/Functions/RSSFeed/"
    "DE/Pressemitteilungen/RSSPressemitteilungen.xml"
)
_FEED_EN = _FEED_DE  # graceful alias; same content (German)


def iter_bmf_press(
    *,
    language: str = "en",
    feed_url: str | None = None,
) -> Iterator[tuple]:
    """Yield BMF press items.

    BMF restructures its RSS endpoints periodically; if the default URL
    returns 404, find the current feed at
    ``https://www.bundesfinanzministerium.de/Web/EN/Press/press.html``
    and pass it as ``feed_url=``.
    """
    url = feed_url or (_FEED_EN if language == "en" else _FEED_DE)
    yield from iter_rss(url)


__all__ = ["iter_bmf_press"]
