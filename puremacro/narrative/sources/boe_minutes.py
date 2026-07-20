"""Bank of England Monetary Policy Summary & MPC Minutes — URL-enumerator.

BoE retired the RSS feed in 2025 and the MPC archive is not directly
indexed. URL pattern is consistent though:

    https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/
        <yyyy>/<month>-<yyyy>

where ``<month>`` is the full lowercase English month name. MPC meets
approximately 8 times per year (Feb/Mar/May/Jun/Aug/Sep/Nov/Dec since
2016). We HEAD each candidate URL and yield the ones that resolve.

The MPC meeting date is mid-month, but for quarterly aggregation we
stamp the first-of-month date from the URL (precise day not needed).

Body fetch (``fetch_body=True``) downloads the full HTML page and
strips boilerplate; default is title-only.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Iterator

import requests


_BASE = "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)
_MPC_MONTHS = [
    ("february", 2),
    ("march",    3),
    ("may",      5),
    ("june",     6),
    ("august",   8),
    ("september",9),
    ("november",11),
    ("december",12),
]
_TITLE_RX = re.compile(r"<title[^>]*>([^<]+)</title>", re.I)
_TAG_RX = re.compile(r"<[^>]+>")
_SCRIPT_RX = re.compile(r"<(?:script|style)[^>]*>.*?</(?:script|style)>", re.S | re.I)


def _strip(html: str) -> str:
    txt = _SCRIPT_RX.sub(" ", html)
    txt = _TAG_RX.sub(" ", txt)
    txt = re.sub(r"&nbsp;", " ", txt)
    txt = re.sub(r"&\w+;", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def _head_or_get(url: str, *, want_body: bool, timeout: float = 15.0):
    """Return (status_code, body_or_none)."""
    method = "GET" if want_body else "HEAD"
    headers = {"User-Agent": _USER_AGENT,
               "Accept": "text/html,application/xhtml+xml",
               "Referer": "https://www.bankofengland.co.uk/monetary-policy"}
    try:
        r = requests.request(method, url, headers=headers, timeout=timeout,
                             allow_redirects=True)
    except Exception:
        return 0, None
    if r.status_code != 200:
        return r.status_code, None
    return 200, (r.text if want_body else None)


def iter_boe_minutes(
    *,
    min_year: int = 2017,
    max_year: int | None = None,
    fetch_body: bool = False,
) -> Iterator[tuple]:
    """Yield (date, text, url, metadata) for BoE Monetary Policy Summary
    + MPC Minutes pages from ``min_year`` to ``max_year`` (default 2017+)."""
    if max_year is None:
        max_year = datetime.utcnow().year
    for year in range(max_year, min_year - 1, -1):
        for month_name, month_idx in _MPC_MONTHS:
            url = f"{_BASE}/{year}/{month_name}-{year}"
            status, body = _head_or_get(url, want_body=fetch_body)
            if status != 200:
                continue
            dt = datetime(year, month_idx, 1)
            if fetch_body and body:
                title_m = _TITLE_RX.search(body)
                title = title_m.group(1).strip() if title_m else ""
                text = _strip(body)
            else:
                title = f"Monetary Policy Summary and Minutes — {month_name.title()} {year}"
                text = title
            yield (dt, text, url, {
                "doctype": "minutes",
                "language": "en",
                "bank_code": "BoE",
                "country": "GBR",
            })


__all__ = ["iter_boe_minutes"]
