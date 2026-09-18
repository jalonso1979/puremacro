"""Federal Reserve FOMC decision statements.

Listing endpoint: federalreserve.gov publishes a JSON index of press
releases at /json/ne-press.json. The endpoint serves UTF-8 with a BOM
and the top-level shape is a list (not a dict). Each entry has keys
``d`` (date), ``t`` (title), ``pt`` (press type), ``l`` (relative URL).
We filter to ``pt == "Monetary Policy"`` AND title containing ``fomc
statement`` and fetch the linked HTML body.
"""
from __future__ import annotations

import json
from typing import Iterator

import pandas as pd

from ..._http import safe_get_bytes, safe_get_text
from ._extractors import extract_body


_LISTING_URL = "https://www.federalreserve.gov/json/ne-press.json"
_BASE = "https://www.federalreserve.gov"
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _parse_listing(raw: bytes) -> list[tuple[pd.Timestamp, str]]:
    """Parse the Fed press JSON listing into ``(date, item_url)`` pairs.

    The endpoint returns a UTF-8-with-BOM document containing a JSON
    list of items. Schema-tolerant: also accepts ``{"refData": [...]}``
    shape in case the endpoint changes.
    """
    try:
        text = raw.decode("utf-8-sig", errors="ignore")
        obj = json.loads(text)
    except json.JSONDecodeError:
        return []
    items = obj if isinstance(obj, list) else obj.get("refData", [])
    out: list[tuple[pd.Timestamp, str]] = []
    for item in items:
        if (item.get("pt") or "").lower() != "monetary policy":
            continue
        # Schema uses "t" for title; tolerate legacy "ti".
        title = (item.get("t") or item.get("ti") or "").lower()
        if "statement" not in title:
            continue
        if "fomc" not in title and "federal open market committee" not in title:
            continue
        try:
            date = pd.Timestamp(item.get("d"))
        except Exception:
            continue
        href = item.get("l", "")
        if not href:
            continue
        out.append((date, _BASE + href if href.startswith("/") else href))
    return out


def iter_fed_decision() -> Iterator[tuple]:
    """Yield (date, text, url, metadata) for FOMC statement releases."""
    try:
        body = safe_get_bytes(_LISTING_URL, user_agent=_UA)
    except Exception:
        return
    for date, item_url in _parse_listing(body):
        try:
            html = safe_get_text(item_url, user_agent=_UA)
        except Exception:
            continue
        text = extract_body(html, bank_code="FED")
        if not text:
            continue
        yield (date, text, item_url, {
            "doctype": "decision", "language": "en",
            "bank_code": "FED", "country": "USA",
        })


__all__ = ["iter_fed_decision"]
