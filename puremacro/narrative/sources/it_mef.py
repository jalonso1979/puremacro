"""Italian Ministry of Economy and Finance (MEF) press feed.

MEF publishes news at https://www.mef.gov.it/. The English-language
RSS feed is at https://www.mef.gov.it/en/feeds/news.html and the
Italian feed at https://www.mef.gov.it/feeds/news.html.
"""
from __future__ import annotations

from typing import Iterator

from ._rss import iter_rss


# MEF's RSS index lives at https://www.mef.gov.it/comunica-con-noi/
# seguici/social-mef.html. Categories are exposed by ``t`` integer:
#   t=4: Comunicati stampa (press releases) — primary
#   t=3: Interviste / interventi (speeches & interviews)
#   t=5: Note (technical notes) — fiscal-policy heavy
#   t=8: Comunicati delle Direzioni (departmental notes)
_FEED_PRESS = "https://www.mef.gov.it/rss/rss.asp?t=4"
_FEED_NOTES = "https://www.mef.gov.it/rss/rss.asp?t=5"


def iter_mef_press(
    *,
    category: str = "press",
    feed_url: str | None = None,
) -> Iterator[tuple]:
    """Yield MEF items in Italian.

    ``category`` selects ``"press"`` (t=4, default) or ``"notes"`` (t=5).
    Override ``feed_url=`` for any other ``t=N``. Items are in
    Italian — combine with LLM scoring.
    """
    if feed_url is None:
        feed_url = _FEED_NOTES if category == "notes" else _FEED_PRESS
    yield from iter_rss(feed_url)


__all__ = ["iter_mef_press"]
