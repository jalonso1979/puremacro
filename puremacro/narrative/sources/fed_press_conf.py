"""FOMC chair press-conference transcripts.

Listing page: /monetarypolicy/fomcpresconf.htm. Anchor hrefs to PDFs
named FOMCpresconfYYYYMMDD.pdf. We extract the date from the filename
and fetch the PDF as bytes (the LLM pipeline tolerates raw PDF noise).
"""
from __future__ import annotations

import re
from typing import Iterator

import pandas as pd

from ..._http import safe_get_bytes, safe_get_text


_LISTING_URL = "https://www.federalreserve.gov/monetarypolicy/fomcpresconf.htm"
_BASE = "https://www.federalreserve.gov"
_FNAME_RX = re.compile(r"FOMCpresconf(\d{8})\.pdf", re.I)
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def iter_fed_press_conf() -> Iterator[tuple]:
    try:
        html = safe_get_text(_LISTING_URL, user_agent=_UA)
    except Exception:
        return
    seen: set[str] = set()
    for m in _FNAME_RX.finditer(html):
        ymd = m.group(1)
        try:
            date = pd.Timestamp(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}")
        except Exception:
            continue
        href_start = max(0, m.start() - 200)
        snippet = html[href_start:m.end() + 5]
        href_m = re.search(r'href="([^"]+\.pdf)"', snippet, re.I)
        if not href_m:
            continue
        item_url = href_m.group(1)
        if item_url.startswith("/"):
            item_url = _BASE + item_url
        if item_url in seen:
            continue
        seen.add(item_url)
        try:
            pdf_bytes = safe_get_bytes(item_url, user_agent=_UA)
        except Exception:
            continue
        text = pdf_bytes.decode("latin-1", errors="ignore")
        text = re.sub(r"[^\x20-\x7e\n]+", " ", text)
        text = re.sub(r"\s{3,}", "\n\n", text)
        if len(text) < 200:
            continue
        yield (date, text[:30000], item_url, {
            "doctype": "press_conf", "language": "en",
            "bank_code": "FED", "country": "USA",
        })


__all__ = ["iter_fed_press_conf"]
