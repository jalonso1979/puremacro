"""Banco Central do Brasil (BCB) — monetary-policy announcements
(Portuguese + English).

BCB migrated its public site to an SPA backed by a JSON content API
in 2025. The legacy ``/api/feed/site/...`` endpoints now return 405
(method-not-allowed) and the front-end is WAF-protected. The
underlying content APIs are open to plain ``requests`` with a
realistic User-Agent, though, and serve all Copom decisions + minutes
through pagination-free batch endpoints. We discovered the endpoints
by XHR-tracing the SPA with stealth-Playwright and then dropped to
pure-requests in production.

Endpoints (all GET, JSON):
    /api/servico/sitebcb/comunicadoscopompublicacao/ultimas
        ?quantidade=N&filtro=
        → Copom rate decisions (Portuguese). ~230 records.
    /api/servico/sitebcb/atascopom/ultimas?quantidade=N&filtro=
        → Copom minutes (Portuguese). ~260 records.
    /api/servico/sitebcb/copomminutes/ultimas?quantidade=N&filtro=
        → Copom minutes (English mirror). ~240 records.

Body text: BCB's HTML landing pages are SPA shells (no scrapable body),
so per-record ``Url`` fields point to canonical PDFs. When
``fetch_body=True``, we fetch those PDFs and extract text via
``extract_body_from_pdf``; on any failure we fall back to title +
subtitle.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterator
from urllib.parse import quote, urlsplit, urlunsplit

import requests

from ._extractors import extract_body_from_pdf
from ._http import safe_get_bytes


_BASE = "https://www.bcb.gov.br"
_API_TEMPLATE = "{base}/api/servico/sitebcb/{section}/ultimas?quantidade={n}&filtro="
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)


def _fetch_section(section: str, quantidade: int = 1000) -> list[dict]:
    url = _API_TEMPLATE.format(base=_BASE, section=section, n=quantidade)
    try:
        r = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=30)
        r.raise_for_status()
        return r.json().get("conteudo", [])
    except Exception:
        return []


def _parse_date(raw: str) -> datetime | None:
    """BCB API uses ISO-8601 with timezone; parse and strip tz to keep
    parity with the rest of the connectors (all return tz-naive)."""
    if not raw:
        return None
    try:
        s = raw.replace("Z", "").split("+")[0][:19]
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(microsecond=0)
    except ValueError:
        try:
            return datetime.strptime(raw[:10], "%Y-%m-%d")
        except ValueError:
            return None


def _absolutize(link: str) -> str:
    if not link:
        return ""
    return _BASE + link if link.startswith("/") else link


def _encode_url(url: str) -> str:
    """Percent-encode unsafe characters in the path of ``url``.

    BCB occasionally returns PDF URLs whose filenames contain spaces or
    other characters that ``requests`` refuses to send raw (e.g.
    ``/content/copom/copomminutes/Minutes 275.pdf``). We split, quote
    only the path component, and rebuild — leaving scheme/host/query
    untouched. Idempotent for already-encoded URLs."""
    if not url:
        return url
    parts = urlsplit(url)
    if not parts.path:
        return url
    # ``safe`` keeps reserved characters that may already be in the path
    # (slashes, percent-escapes, etc.) so the call is idempotent.
    safe_path = quote(parts.path, safe="/%:@!$&'()*+,;=~-._")
    return urlunsplit((parts.scheme, parts.netloc, safe_path,
                       parts.query, parts.fragment))


def _maybe_fetch_pdf_body(item: dict, *, fetch_body: bool) -> str | None:
    """If ``fetch_body`` and the record carries a PDF URL, fetch & parse it.
    Returns body text on success, None otherwise (caller falls back to
    title + subtitle)."""
    if not fetch_body:
        return None
    pdf_url = (item.get("Url") or item.get("UrlPdf") or "").strip()
    if not pdf_url or not pdf_url.lower().endswith(".pdf"):
        return None
    pdf_url = _encode_url(_absolutize(pdf_url))
    try:
        pdf_bytes = safe_get_bytes(pdf_url)
    except Exception:
        return None
    return extract_body_from_pdf(pdf_bytes)


def _yield_records(
    items: list[dict], *, doctype: str, language: str, fetch_body: bool,
    bank_code: str = "BCB", country: str = "BRA",
) -> Iterator[tuple]:
    for it in items:
        date_raw = it.get("DataReferencia") or it.get("Data") or ""
        dt = _parse_date(date_raw)
        if dt is None:
            continue
        title = (it.get("Titulo") or "").strip()
        subtitle = (it.get("Subtitulo") or "").strip()
        text = f"{title}. {subtitle}" if subtitle else title

        body = _maybe_fetch_pdf_body(it, fetch_body=fetch_body)
        if body:
            text = body

        # Prefer the landing-page URL for identity (stable, human-readable),
        # but fall back to the PDF URL when the SPA page is missing.
        link = it.get("LinkPagina") or it.get("Url") or ""
        full_url = _absolutize(link)
        yield (dt, text, full_url, {
            "doctype": doctype,
            "language": language,
            "bank_code": bank_code,
            "country": country,
        })


def iter_bcb_decision(
    *, language: str = "pt", quantidade: int = 1000,
    fetch_body: bool = True,
) -> Iterator[tuple]:
    """Yield (date, title+subtitle, url, metadata) for BCB Copom rate
    decisions (Portuguese only — API is single-language).

    When ``fetch_body=True`` (default), any record carrying a PDF URL
    has its body extracted and substituted for the title+subtitle text.
    """
    items = _fetch_section("comunicadoscopompublicacao", quantidade=quantidade)
    yield from _yield_records(items, doctype="decision", language="pt",
                              fetch_body=fetch_body)


def iter_bcb_minutes(
    *, language: str = "pt", quantidade: int = 1000,
    fetch_body: bool = True,
) -> Iterator[tuple]:
    """Yield (date, title, url, metadata) for BCB Copom minutes.

    ``language="pt"`` (default) pulls the Portuguese minutes; ``"en"``
    pulls the English mirror. When ``fetch_body=True`` (default), the
    linked Minutes PDF is fetched and parsed for body text.
    """
    section = "copomminutes" if language == "en" else "atascopom"
    items = _fetch_section(section, quantidade=quantidade)
    yield from _yield_records(items, doctype="minutes", language=language,
                              fetch_body=fetch_body)


__all__ = ["iter_bcb_decision", "iter_bcb_minutes"]
