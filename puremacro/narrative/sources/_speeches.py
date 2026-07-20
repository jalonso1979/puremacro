"""Shared parser for CB speech archives (RSS-style)."""
from __future__ import annotations

from typing import Iterator

from ._rss import iter_rss
from . import _ratedoc
from ._ratedoc import strip_html
from ._extractors import extract_body


def iter_speeches_rss(
    url: str,
    *,
    bank_code: str,
    country: str,
    language: str = "en",
    fetch_body: bool = False,
) -> Iterator[tuple]:
    """Wrap an RSS speech feed and emit 4-tuple SourceRecords."""
    for date, title_desc, link in iter_rss(url):
        clean = strip_html(title_desc) if "<" in title_desc else title_desc
        if fetch_body and link:
            try:
                body_html = _ratedoc.safe_get_text(link)
                body_text = extract_body(body_html, bank_code=bank_code)
                if body_text:
                    clean = body_text
            except Exception:
                pass
        yield (date, clean, link, {
            "doctype": "speech", "language": language,
            "bank_code": bank_code, "country": country,
        })


__all__ = ["iter_speeches_rss"]
