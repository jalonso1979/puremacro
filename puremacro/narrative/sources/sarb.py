"""South African Reserve Bank (SARB) — MPC statements (English).

Legacy RSS feed (``/en/home/publications/rss``) returned 404 after the
2025 site reorg. The SARB site is SPA-rendered but admits stealth-
Chromium clients; we scrape the MPC-statements listing page.

Each MPC statement appears as:
    <a href="/en/home/publications/publication-detail-pages/statements/
              monetary-policy-statements/<yyyy>/<month-slug>">
      <span class="publications__resultListItem__title"><b>
        Statement of the Monetary Policy Committee <Month> <Year>
      </b></span>
    </a>

The month + year are parseable from the title.
"""
from __future__ import annotations

import re
import warnings
from datetime import datetime
from typing import Iterator

from ._fallback import fetch_with_fallback, FallbackExhaustedError

# sarb is currently Playwright-only — the SARB site is SPA-rendered and
# requires JS execution to render content.
FALLBACK_POLICY: tuple[str, ...] = ("playwright",)


_INDEX_URL = "https://www.resbank.co.za/en/home/publications/statements/mpc-statements"
_BASE = "https://www.resbank.co.za"
_ITEM_RX = re.compile(
    r'<a\s+href="(?P<href>/en/home/publications/publication-detail-pages/'
    r'statements/monetary-policy-statements/[^"]+)"[^>]*>\s*'
    r'<span\s+class="publications__resultListItem__title">\s*<b>\s*'
    r'(?P<title>[^<]+)</b>',
    re.S,
)
_MONTH_RX = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
    re.I,
)


def _parse_month_year(title: str) -> datetime | None:
    m = _MONTH_RX.search(title)
    if not m:
        return None
    try:
        return datetime.strptime(f"1 {m.group(1)} {m.group(2)}", "%d %B %Y")
    except ValueError:
        return None


def iter_sarb_decision(*, fetch_body: bool = False) -> Iterator[tuple]:
    """Yield (date, title, url, metadata) for SARB MPC statements.

    The page renders ~25 most-recent statements (covers ~10 years of
    MPC meetings since SARB meets 6×/year)."""
    try:
        html = fetch_with_fallback(_INDEX_URL, policy=FALLBACK_POLICY, source="sarb")
    except FallbackExhaustedError:
        warnings.warn("sarb: fallback exhausted for index page; skipping",
                      RuntimeWarning, stacklevel=2)
        return
    for m in _ITEM_RX.finditer(html):
        href = m.group("href")
        title = re.sub(r"\s+", " ", m.group("title")).strip()
        dt = _parse_month_year(title)
        if dt is None:
            continue
        full_url = _BASE + href if href.startswith("/") else href
        yield (dt, title, full_url, {
            "doctype": "decision",
            "language": "en",
            "bank_code": "SARB",
            "country": "ZAF",
        })


__all__ = ["iter_sarb_decision"]
