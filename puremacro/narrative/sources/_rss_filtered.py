"""Shared RSS-feed wrapper with optional title-keyword filtering and
optional body-fetch.
"""
from __future__ import annotations

from typing import Iterator

from ._rss import iter_rss
from . import _ratedoc
from ._ratedoc import strip_html
from ._extractors import extract_body


def iter_rss_filtered(
    url: str,
    *,
    bank_code: str,
    country: str,
    doctype: str,
    language: str = "en",
    title_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None,
    fetch_body: bool = False,
) -> Iterator[tuple]:
    """Wrap an RSS feed and emit 4-tuple SourceRecords.

    Parameters
    ----------
    url : RSS feed URL.
    bank_code : short tag for ``metadata["bank_code"]`` (e.g. ``"RBA"``).
    country : ISO3 (e.g. ``"AUS"``).
    doctype : ``"decision"`` | ``"minutes"`` | ``"speech"`` | ``"press"`` | ``"fsr"``.
    language : ISO-639-1.
    title_keywords : if given (non-empty), the title must contain at
        least one of these (case-insensitive). If ``None`` or empty,
        no filter — all items pass.
    exclude_keywords : if given (non-empty), items whose title contains
        any of these are dropped (case-insensitive).
    fetch_body : default ``False``. If ``True``, fetch the link target
        URL for each item and replace the RSS-summary text with the
        extracted body. Body fetch failures (or empty bodies) fall
        back to the RSS summary so the connector never yields empty
        text. Doubles HTTP calls per item; opt-in.
    """
    for date, title_desc, link in iter_rss(url):
        clean = strip_html(title_desc) if "<" in title_desc else title_desc
        low = clean.lower()
        if title_keywords and not any(kw.lower() in low for kw in title_keywords):
            continue
        if exclude_keywords and any(kw.lower() in low for kw in exclude_keywords):
            continue
        if fetch_body and link:
            try:
                body_html = _ratedoc.safe_get_text(link)
                body_text = extract_body(body_html, bank_code=bank_code)
                if body_text:
                    clean = body_text
            except Exception:
                pass
        yield (date, clean, link, {
            "doctype": doctype, "language": language,
            "bank_code": bank_code, "country": country,
        })


__all__ = ["iter_rss_filtered"]
