"""Bank of Korea (BoK) — English mirror, monetary-policy decisions.

Legacy RSS endpoint (``/eng/rss/RssBokService.do?menuNo=400069``)
returns 404 after the 2025 site reorg. The current archive is at:

    https://www.bok.or.kr/eng/singl/newsDataEng/list.do?menuNo=400022
        — Monetary Policy Decisions
    https://www.bok.or.kr/eng/singl/newsDataEng/list.do?menuNo=400021
        — Monetary Policy Board Meeting Minutes

Each item appears as:
    <li class="bbsRowCls">
      <span class="date">YYYY.MM.DD</span>
      <a href="/eng/bbs/E0000627/view.do?nttId=<id>&...">
        <Title>
      </a>
    </li>

Items are rendered by JS (not visible in the static HTML), so we go
through the stealth-Playwright helper.
"""
from __future__ import annotations

import re
import warnings
from datetime import datetime
from typing import Iterator

from ._fallback import fetch_with_fallback, FallbackExhaustedError
from ._extractors import extract_body

# bok is currently Playwright-only — the BoK list pages require JS execution
# to render content.
FALLBACK_POLICY: tuple[str, ...] = ("playwright",)


_DECISIONS_URL = "https://www.bok.or.kr/eng/singl/newsDataEng/list.do?menuNo=400022"
_MINUTES_URL = "https://www.bok.or.kr/eng/singl/newsDataEng/list.do?menuNo=400021"
_BASE = "https://www.bok.or.kr"
_ITEM_RX = re.compile(
    r'<li\s+class="bbsRowCls">.*?'
    r'<span\s+class="date">(?P<date>\d{4}\.\d{1,2}\.\d{1,2})</span>.*?'
    r'<a\s+href="(?P<href>/eng/bbs/[^"]+)"[^>]*>\s*(?P<title>[^<]+?)\s*</a>',
    re.S,
)


def _yield_from(url: str, doctype: str, *, fetch_body: bool = False) -> Iterator[tuple]:
    try:
        html = fetch_with_fallback(url, policy=FALLBACK_POLICY, source="bok")
    except FallbackExhaustedError:
        warnings.warn(f"bok: fallback exhausted for {url}; skipping",
                      RuntimeWarning, stacklevel=3)
        return
    for m in _ITEM_RX.finditer(html):
        try:
            dt = datetime.strptime(m.group("date"), "%Y.%m.%d")
        except ValueError:
            continue
        title = re.sub(r"\s+", " ",
                       m.group("title").replace("&amp;", "&")).strip()
        href = m.group("href").replace("&amp;", "&")
        full_url = _BASE + href if href.startswith("/") else href
        text = title  # title-only fallback
        if fetch_body and full_url:
            try:
                body_html = fetch_with_fallback(
                    full_url, policy=FALLBACK_POLICY, source="bok",
                )
                body_text = extract_body(body_html, bank_code="BOK")
                if body_text and len(body_text) > 200:
                    text = body_text
            except FallbackExhaustedError:
                pass  # fall back to title
        yield (dt, text, full_url, {
            "doctype": doctype,
            "language": "en",
            "bank_code": "BoK",
            "country": "KOR",
        })


def iter_bok_decision(*, fetch_body: bool = True) -> Iterator[tuple]:
    """Yield (date, title, url, metadata) for BoK monetary-policy decisions.

    When ``fetch_body`` is True (default), each row's linked HTML page is
    fetched via stealth-Playwright and its body extracted via
    :func:`extract_body` (bank_code ``"BOK"``); on success the body text
    replaces the title, otherwise the title-only fallback is preserved.
    """
    yield from _yield_from(_DECISIONS_URL, "decision", fetch_body=fetch_body)


def iter_bok_minutes(*, fetch_body: bool = True) -> Iterator[tuple]:
    """Yield (date, title, url, metadata) for BoK Monetary Policy Board
    meeting minutes.

    When ``fetch_body`` is True (default), each row's linked HTML page is
    fetched via stealth-Playwright and its body extracted via
    :func:`extract_body` (bank_code ``"BOK"``); on success the body text
    replaces the title, otherwise the title-only fallback is preserved.
    """
    yield from _yield_from(_MINUTES_URL, "minutes", fetch_body=fetch_body)


__all__ = ["iter_bok_decision", "iter_bok_minutes"]
