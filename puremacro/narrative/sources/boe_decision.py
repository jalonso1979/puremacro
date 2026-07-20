"""Bank of England MPC decision RSS feed."""
from __future__ import annotations

from typing import Iterator

from ._rss import iter_rss
from . import _ratedoc
from ._ratedoc import strip_html
from ._extractors import extract_body


_FEED = "https://www.bankofengland.co.uk/rss/news/monetary-policy"


def iter_boe_decision(*, fetch_body: bool = False) -> Iterator[tuple]:
    for date, title_desc, link in iter_rss(_FEED):
        clean = strip_html(title_desc) if "<" in title_desc else title_desc
        if "bank rate" not in clean.lower() and "mpc" not in clean.lower():
            continue
        if fetch_body and link:
            try:
                body_html = _ratedoc.safe_get_text(link)
                body_text = extract_body(body_html, bank_code="BOE")
                if body_text:
                    clean = body_text
            except Exception:
                pass
        yield (date, clean, link, {
            "doctype": "decision", "language": "en",
            "bank_code": "BOE", "country": "GBR",
        })


__all__ = ["iter_boe_decision"]
