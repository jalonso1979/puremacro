"""ECB monetary-policy account (minutes-equivalent).

ECB does not publish "minutes" in the FOMC sense; it publishes a
"monetary policy account" of each Governing Council monetary-policy
meeting, on a delayed schedule. We pull from the monetary-policy RSS
feed and filter on title.
"""
from __future__ import annotations

from typing import Iterator

from ._rss import iter_rss
from . import _ratedoc
from ._ratedoc import strip_html
from ._extractors import extract_body


_FEED = "https://www.ecb.europa.eu/rss/press.html"  # mopo.html now 404; unified feed since 2025


def iter_ecb_minutes(*, language: str = "en", fetch_body: bool = False) -> Iterator[tuple]:
    for date, title_desc, link in iter_rss(_FEED):
        clean = strip_html(title_desc) if "<" in title_desc else title_desc
        if "account" not in clean.lower():
            continue
        if fetch_body and link:
            try:
                body_html = _ratedoc.safe_get_text(link)
                body_text = extract_body(body_html, bank_code="ECB")
                if body_text:
                    clean = body_text
            except Exception:
                pass
        yield (date, clean, link, {
            "doctype": "minutes", "language": language,
            "bank_code": "ECB", "country": "EA20",
        })


__all__ = ["iter_ecb_minutes"]
