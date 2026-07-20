"""Shared parser for central-bank decision / minutes pages.

Most CB decisions follow the same shape: a listing page with one entry
per meeting linking to a statement HTML or PDF, plus an optional
language-tagged variant. This helper wraps:
  - listing-page fetch (RSS, Atom, or HTML)
  - per-entry fetch + body extraction
into a uniform 4-tuple SourceRecord stream.
"""
from __future__ import annotations

import re
from typing import Callable, Iterator

import pandas as pd

from ..._http import safe_get_bytes, safe_get_text


_HTML_TEXT_RX = re.compile(r"<[^>]+>")


def strip_html(html: str) -> str:
    """Crude HTML→text. Keeps line breaks at block tags, drops tags."""
    txt = re.sub(r"</?(p|div|li|br|h[1-6])[^>]*>", "\n", html, flags=re.I)
    txt = _HTML_TEXT_RX.sub("", txt)
    txt = re.sub(r"\n\s*\n+", "\n\n", txt)
    return txt.strip()


def iter_ratedoc_listing(
    url: str,
    *,
    parse_listing: Callable[[bytes], list[tuple[pd.Timestamp, str]]],
    fetch_body: Callable[[str], str] | None = None,
    bank_code: str,
    country: str,
    doctype: str,
    language: str = "en",
    user_agent: str | None = None,
) -> Iterator[tuple]:
    """Yield SourceRecord 4-tuples for a CB decision/minutes listing.

    Parameters
    ----------
    url : listing-page URL.
    parse_listing : callable bytes → list[(date, item_url)]. Bank-specific.
    fetch_body : callable url → text. Default uses ``safe_get_text``;
        connectors may override to handle PDFs.
    bank_code : short tag stamped into metadata (e.g. ``"FED"``, ``"ECB"``).
    country : ISO3 of the bank's jurisdiction.
    doctype : ``"decision"`` | ``"minutes"`` | ``"press_conf"`` | ``"fsr"``.
    language : ISO-639-1 of the listing.
    user_agent : optional UA override for WAF-protected sites.
    """
    try:
        body = (safe_get_bytes(url, user_agent=user_agent)
                if user_agent else safe_get_bytes(url))
    except Exception:
        return
    try:
        entries = parse_listing(body)
    except Exception:
        return
    fetcher = fetch_body or (
        (lambda u: safe_get_text(u, user_agent=user_agent))
        if user_agent else safe_get_text
    )
    for date, item_url in entries:
        if pd.isna(date) or not item_url:
            continue
        try:
            text = fetcher(item_url)
        except Exception:
            continue
        clean = strip_html(text) if "<" in text and ">" in text else text
        if not clean:
            continue
        yield (date, clean, item_url, {
            "doctype": doctype, "language": language,
            "bank_code": bank_code, "country": country,
        })


__all__ = ["iter_ratedoc_listing", "strip_html"]
