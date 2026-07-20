"""European Central Bank press releases (monetary-policy decisions feed).

The ECB publishes monetary-policy decisions and other press releases
on a single RSS feed, in 6 languages. Renamed from ``ecb_press.py`` in
0.6.1; ``ecb_press.py`` survives as a deprecation shim that re-exports
``iter_ecb_press = iter_ecb_decision``.

Feed URLs:
    https://www.ecb.europa.eu/rss/press.html        (English; default)
    https://www.ecb.europa.eu/rss/press.de.html     (German)
    https://www.ecb.europa.eu/rss/press.fr.html     (French)
    https://www.ecb.europa.eu/rss/press.es.html     (Spanish)
    https://www.ecb.europa.eu/rss/press.it.html     (Italian)
    https://www.ecb.europa.eu/rss/press.pt.html     (Portuguese)
"""
from __future__ import annotations

from typing import Iterator

from ._rss import iter_rss
from . import _ratedoc
from ._ratedoc import strip_html
from ._extractors import extract_body


_FEED_BY_LANG = {
    "en": "https://www.ecb.europa.eu/rss/press.html",
    "de": "https://www.ecb.europa.eu/rss/press.de.html",
    "fr": "https://www.ecb.europa.eu/rss/press.fr.html",
    "es": "https://www.ecb.europa.eu/rss/press.es.html",
    "it": "https://www.ecb.europa.eu/rss/press.it.html",
    "pt": "https://www.ecb.europa.eu/rss/press.pt.html",
}


_DECISION_PATH_FILTERS = ("/press/pr/", "/press/govcdec/")


def iter_ecb_decision(
    *, language: str = "en", feed_url: str | None = None,
    fetch_body: bool = False,
) -> Iterator[tuple]:
    """Yield (date, text, url, metadata) for ECB press releases and
    governing-council decisions.

    The unified press feed mixes these with speeches and interviews; we
    filter to ``/press/pr/`` and ``/press/govcdec/`` URL prefixes so that
    iter_ecb_speeches (which returns ``/press/key/`` and ``/press/inter/``)
    does not double-count.
    """
    url = feed_url or _FEED_BY_LANG.get(language, _FEED_BY_LANG["en"])
    for date, title_desc, link in iter_rss(url):
        if link and not any(p in link for p in _DECISION_PATH_FILTERS):
            continue
        clean = strip_html(title_desc) if "<" in title_desc else title_desc
        if fetch_body and link:
            try:
                body_html = _ratedoc.safe_get_text(link)
                body_text = extract_body(body_html, bank_code="ECB")
                if body_text:
                    clean = body_text
            except Exception:
                pass
        yield (date, clean, link, {
            "doctype": "decision", "language": language,
            "bank_code": "ECB", "country": "EA20",
        })


__all__ = ["iter_ecb_decision"]
