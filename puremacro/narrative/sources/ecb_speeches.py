"""ECB Executive Board speeches (RSS).

ECB consolidated its RSS endpoints around 2025 — the old per-doctype
feeds (sp.html / mopo.html) now 404. We pull from the unified press
feed and filter items by URL prefix:
    /press/key/   -> speeches (default)
    /press/inter/ -> interviews
"""
from __future__ import annotations

from typing import Iterator

from ._speeches import iter_speeches_rss


_FEED_BY_LANG = {
    "en": "https://www.ecb.europa.eu/rss/press.html",
    "de": "https://www.ecb.europa.eu/rss/press.de.html",
    "fr": "https://www.ecb.europa.eu/rss/press.fr.html",
}
_PATH_FILTERS = ("/press/key/", "/press/inter/")


def iter_ecb_speeches(*, language: str = "en", fetch_body: bool = False) -> Iterator[tuple]:
    """Yield ECB speeches and interviews from the unified press feed.

    The unified feed mixes speeches, interviews, press releases, and
    governing-council decisions; this iterator returns only the speech /
    interview items by URL prefix.
    """
    url = _FEED_BY_LANG.get(language, _FEED_BY_LANG["en"])
    for record in iter_speeches_rss(
        url, bank_code="ECB", country="EA20", language=language,
        fetch_body=fetch_body,
    ):
        link = record[2] if len(record) >= 3 else ""
        if any(p in (link or "") for p in _PATH_FILTERS):
            yield record


__all__ = ["iter_ecb_speeches"]
