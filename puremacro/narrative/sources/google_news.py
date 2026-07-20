"""Google News RSS — universal, multilingual, no API key required.

Google News exposes a permanent RSS endpoint at
``https://news.google.com/rss/search`` that accepts a query plus locale
parameters and returns an RSS 2.0 feed. This works for every language
Google indexes (de, fr, es, it, ja, zh, …) and gives the package a
country-agnostic news source out of the box.

Caveats:
* Google's free RSS is rate-limited (a few hundred queries/hour) and
  returns at most 100 items per query.
* Article URLs are Google-tracking redirects; ``urllib`` follows them
  to the original outlet.
* Google's title-only RSS items are short, so this connector is best
  paired with LLM scoring that can identify fiscal events from a
  headline + outlet name.
"""
from __future__ import annotations

import urllib.parse
from typing import Iterator

from ._rss import iter_rss


_BASE = "https://news.google.com/rss/search"

# (hl, gl, ceid) tuples per country / language.
LOCALE_BY_COUNTRY = {
    "USA": ("en-US", "US", "US:en"),
    "GBR": ("en-GB", "GB", "GB:en"),
    "DEU": ("de",     "DE", "DE:de"),
    "FRA": ("fr",     "FR", "FR:fr"),
    "ITA": ("it",     "IT", "IT:it"),
    "ESP": ("es-419", "ES", "ES:es"),
    "JPN": ("ja",     "JP", "JP:ja"),
    "CAN": ("en-CA",  "CA", "CA:en"),
    "MEX": ("es-419", "MX", "MX:es"),
    "BRA": ("pt-BR",  "BR", "BR:pt-419"),
    "AUS": ("en-AU",  "AU", "AU:en"),
    "NLD": ("nl",     "NL", "NL:nl"),
    "POL": ("pl",     "PL", "PL:pl"),
    "SWE": ("sv",     "SE", "SE:sv"),
    "CHN": ("zh-CN",  "CN", "CN:zh-Hans"),
    "KOR": ("ko",     "KR", "KR:ko"),
}


def iter_google_news(
    query: str,
    *,
    country: str = "USA",
    hl: str | None = None,
    gl: str | None = None,
    ceid: str | None = None,
) -> Iterator[tuple]:
    """Yield (date, title+description, link) for matching news items.

    Parameters
    ----------
    query : Google News search query (boolean operators supported).
    country : ISO3 code mapped to a default (hl, gl, ceid) triple.
    hl, gl, ceid : optional explicit locale overrides.
    """
    if hl is None or gl is None or ceid is None:
        d_hl, d_gl, d_ceid = LOCALE_BY_COUNTRY.get(
            country, ("en-US", "US", "US:en"),
        )
        hl = hl or d_hl
        gl = gl or d_gl
        ceid = ceid or d_ceid
    params = {"q": query, "hl": hl, "gl": gl, "ceid": ceid}
    url = _BASE + "?" + urllib.parse.urlencode(params)
    yield from iter_rss(url)


__all__ = ["iter_google_news", "LOCALE_BY_COUNTRY"]
