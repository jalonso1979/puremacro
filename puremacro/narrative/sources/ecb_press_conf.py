"""ECB press conferences (after Governing Council monetary decisions).

Press-conference transcripts live at /press/pressconf/{year}/html/. We
crawl the most-recent year's listing and yield 4-tuple records.
"""
from __future__ import annotations

import re
from typing import Iterator

import pandas as pd

from ..._http import safe_get_text
from ._extractors import extract_body


_LISTING_FMT = "https://www.ecb.europa.eu/press/pressconf/{year}/html/index.en.html"
# ECB filenames look like .../is220721~973616afa9.en.html for 2022-07-21.
_FILENAME_RX = re.compile(r"is(\d{6})[^/\"]*\.html")


def iter_ecb_press_conf(*, year: int | None = None) -> Iterator[tuple]:
    if year is None:
        from datetime import date as _d
        year = _d.today().year
    url = _LISTING_FMT.format(year=year)
    try:
        html = safe_get_text(url)
    except Exception:
        return
    seen: set[str] = set()
    for m in _FILENAME_RX.finditer(html):
        href = m.group(0)
        ymd = m.group(1)
        if href in seen:
            continue
        seen.add(href)
        try:
            date = pd.Timestamp(f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:6]}")
        except Exception:
            continue
        item_url = f"https://www.ecb.europa.eu/press/pressconf/{year}/html/{href}"
        try:
            body_html = safe_get_text(item_url)
        except Exception:
            continue
        text = extract_body(body_html, bank_code="ECB")
        if len(text) < 200:
            continue
        yield (date, text[:30000], item_url, {
            "doctype": "press_conf", "language": "en",
            "bank_code": "ECB", "country": "EA20",
        })


__all__ = ["iter_ecb_press_conf"]
