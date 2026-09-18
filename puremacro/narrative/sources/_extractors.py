"""HTML body extraction for CB pages (per-bank registry + generic fallback).

Replaces the crude ``_ratedoc.strip_html`` for connectors that fetch
HTML body pages. Per-bank precise extractors find a known container
(e.g. Fed's ``<div id="article">``); the generic fallback picks the
largest text-dense block.

Pure stdlib (``re``); Pyodide-clean. PDF support (``extract_body_from_pdf``)
is optional and degrades gracefully when ``pypdf`` is not installed.
"""
from __future__ import annotations

import re
from io import BytesIO
from typing import Callable

from ._ratedoc import strip_html


# ---------------------------------------------------------------------------
# Whole-document preprocessing: remove obvious chrome before any per-bank
# matching. Order matters (script/style first to avoid grabbing JS strings).
# ---------------------------------------------------------------------------
_CHROME_TAG_RX = re.compile(
    r"<(script|style|nav|header|footer|aside)\b[^>]*>.*?</\1\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _drop_chrome(html: str) -> str:
    return _CHROME_TAG_RX.sub("", html)


# ---------------------------------------------------------------------------
# Balanced-tag matching helper. Real-world CB pages nest the same tag
# (e.g. ``<div id="article">`` contains heading/body/share-buttons divs),
# so a non-greedy ``</div>`` match would close on the first inner div
# instead of the outer container. Track depth on the same tag name.
# ---------------------------------------------------------------------------
def _extract_balanced_tag(html: str, open_tag_pattern: str) -> str | None:
    """Find a tag matching ``open_tag_pattern``, then return the inner
    content up to the matching balanced close tag.

    ``open_tag_pattern`` is a regex matching the opening tag (e.g.
    ``r'<div\\b[^>]*\\bid="article"[^>]*>'``). The first character of
    each opening tag inside the regex (``<div``, ``<main``, etc.)
    determines the matching close-tag pattern.

    Uses depth counting on the same tag name (``div``, ``main``, etc.)
    so nested tags of the same kind are tracked correctly.

    Returns the inner content (between matched open and close), or
    None if the open tag isn't found or the document doesn't have a
    balanced close.
    """
    open_match = re.search(open_tag_pattern, html, re.IGNORECASE | re.DOTALL)
    if not open_match:
        return None

    # Extract tag name from the matched open tag (first <\w+>).
    name_match = re.match(r"<(\w+)", open_match.group(0))
    if not name_match:
        return None
    tag_name = name_match.group(1)

    open_rx = re.compile(rf"<{tag_name}\b[^>]*>", re.IGNORECASE)
    close_rx = re.compile(rf"</{tag_name}\s*>", re.IGNORECASE)

    pos = open_match.end()
    depth = 1
    while depth > 0 and pos < len(html):
        next_open = open_rx.search(html, pos)
        next_close = close_rx.search(html, pos)
        if next_close is None:
            return None  # unbalanced
        if next_open and next_open.start() < next_close.start():
            depth += 1
            pos = next_open.end()
        else:
            depth -= 1
            content_end = next_close.start()
            pos = next_close.end()
            if depth == 0:
                return html[open_match.end():content_end]
    return None


# ---------------------------------------------------------------------------
# Per-bank extractors. Each returns either body text or None (caller
# falls back to the generic on None).
# ---------------------------------------------------------------------------
def _extract_fed_body(html: str) -> str | None:
    inner = _extract_balanced_tag(html, r'<div\b[^>]*\bid="article"[^>]*>')
    if not inner:
        return None
    return strip_html(inner).strip() or None


def _extract_ecb_body(html: str) -> str | None:
    for pat in (
        r'<main\b[^>]*\bid="main-wrapper"[^>]*>',
        r'<div\b[^>]*\bclass="[^"]*\bsection\b[^"]*"[^>]*>',
    ):
        inner = _extract_balanced_tag(html, pat)
        if inner:
            text = strip_html(inner).strip()
            if text:
                return text
    return None


def _extract_bis_body(html: str) -> str | None:
    # BIS review pages wrap the speech body in <div id='cmsContent'>
    # (single-quoted attrs). Fall back to a class="...article..."
    # container for legacy/alternate layouts, then to <main>.
    for pat in (
        r'''<div\b[^>]*\bid=['"]cmsContent['"][^>]*>''',
        r'''<div\b[^>]*\bclass=['"][^'"]*\barticle\b[^'"]*['"][^>]*>''',
        r'<main\b[^>]*>',
    ):
        inner = _extract_balanced_tag(html, pat)
        if inner:
            text = strip_html(inner).strip()
            if text and len(text) > 200:
                return text
    return None


def _extract_boj_body(html: str) -> str | None:
    # BoJ speech pages wrap the speech body in <main id="contents">.
    # Inside that main, the actual prose lives in a <div class="outline mod_outer">.
    # Try the inner-most container first (narrower → less chrome), then
    # fall back to the wider <main>. Quote-agnostic per α-2 precedent.
    cleaned = _drop_chrome(html)
    for pat in (
        r'''<div\b[^>]*\bclass=['"][^'"]*\boutline\s+mod_outer\b[^'"]*['"][^>]*>''',
        r'''<main\b[^>]*\bid=['"]contents['"][^>]*>''',
        r'''<div\b[^>]*\bid=['"]contents['"][^>]*>''',
        r'''<div\b[^>]*\bclass=['"][^'"]*\bContentBlock\b[^'"]*['"][^>]*>''',
    ):
        inner = _extract_balanced_tag(cleaned, pat)
        if inner:
            text = strip_html(inner).strip()
            if text and len(text) > 200:
                return text
    return None


def _extract_pboc_body(html: str) -> str | None:
    # PBoC English mirror wraps article prose in <div id="zoom"> nested
    # inside a <td class="content">. The td_box_left container is a legacy
    # layout from the pre-2025 site. Try the inner-most container first
    # (narrower → less chrome). Quote-agnostic per α-2 precedent.
    cleaned = _drop_chrome(html)
    for pat in (
        r'''<div\b[^>]*\bid=['"]zoom['"][^>]*>''',
        r'''<td\b[^>]*\bclass=['"][^'"]*\bcontent\b[^'"]*['"][^>]*>''',
        r'''<div\b[^>]*\bclass=['"][^'"]*\btd_box_left\b[^'"]*['"][^>]*>''',
        r'''<div\b[^>]*\bclass=['"][^'"]*\bmain_text\b[^'"]*['"][^>]*>''',
    ):
        inner = _extract_balanced_tag(cleaned, pat)
        if inner:
            text = strip_html(inner).strip()
            if text and len(text) > 200:
                return text
    return None


def _extract_rba_body(html: str) -> str | None:
    # RBA speech/media-release pages wrap article prose in
    # <div id="content" role="main">. Strip script/style/nav/header/
    # footer/aside chrome first (the page ships extensive GA + share
    # widgets inside the content container). Quote-agnostic per α-2
    # precedent; fall back to <main> for any layout variants.
    cleaned = _drop_chrome(html)
    for pat in (
        r'''<div\b[^>]*\bid=['"]content['"][^>]*>''',
        r'''<main\b[^>]*\bid=['"]content['"][^>]*>''',
        r'<main\b[^>]*>',
    ):
        inner = _extract_balanced_tag(cleaned, pat)
        if inner:
            text = strip_html(inner).strip()
            if text and len(text) > 200:
                return text
    return None


def _extract_bcb_body(html: str) -> str | None:
    for pat in (
        r'<div\b[^>]*\bclass="[^"]*\bnews-detail\b[^"]*"[^>]*>',
        r'<main\b[^>]*>',
    ):
        inner = _extract_balanced_tag(html, pat)
        if inner:
            text = strip_html(inner).strip()
            if text and len(text) > 200:
                return text
    return None


def _extract_banxico_body(html: str) -> str | None:
    inner = _extract_balanced_tag(html, r'<div\b[^>]*\bid="MainContent"[^>]*>')
    if not inner:
        return None
    text = strip_html(inner).strip()
    return text if (text and len(text) > 200) else None


def _extract_bok_body(html: str) -> str | None:
    # BoK English-mirror view pages wrap the press-release prose in
    # <div class="dbdata"> (narrow, prose-only) nested inside a
    # <div class="bd-view"> (wider, also includes attachments / metadata).
    # Try the inner-most prose container first; fall back to bd-view,
    # then to the legacy view_con class from older layouts, then to
    # <main id="main-container">. Quote-agnostic per α-2 precedent.
    cleaned = _drop_chrome(html)
    for pat in (
        r'''<div\b[^>]*\bclass=['"][^'"]*\bdbdata\b[^'"]*['"][^>]*>''',
        r'''<div\b[^>]*\bclass=['"][^'"]*\bbd-view\b[^'"]*['"][^>]*>''',
        r'''<div\b[^>]*\bclass=['"][^'"]*\bview_con\b[^'"]*['"][^>]*>''',
        r'''<main\b[^>]*\bid=['"]main-container['"][^>]*>''',
    ):
        inner = _extract_balanced_tag(cleaned, pat)
        if inner:
            text = strip_html(inner).strip()
            if text and len(text) > 200:
                return text
    return None


def _extract_riksbank_body(html: str) -> str | None:
    # Riksbank press / notice pages wrap the prose in a single ``<article>``
    # tag nested inside ``<main id="main-content">``; older layouts and
    # some non-press pages instead used ``rb-article-text`` or
    # ``page-base__main__body``. Try the live container first, then fall
    # back through legacy/wrapper classes. Quote-agnostic per α-2.
    cleaned = _drop_chrome(html)
    for pat in (
        r'''<article\b[^>]*>''',
        r'''<div\b[^>]*\bclass=['"][^'"]*\bpage-base__main__body\b[^'"]*['"][^>]*>''',
        r'''<div\b[^>]*\bclass=['"][^'"]*\brb-article-text\b[^'"]*['"][^>]*>''',
        r'''<main\b[^>]*\bid=['"]main-content['"][^>]*>''',
    ):
        inner = _extract_balanced_tag(cleaned, pat)
        if inner:
            text = strip_html(inner).strip()
            if text and len(text) > 200:
                return text
    return None


def _extract_rbi_body(html: str) -> str | None:
    # RBI press-release pages wrap the prose table in ``<div id="doublescroll">``
    # (innermost, prose-only). Outer wrappers (``<div role="main"
    # class="container_12">``) also include breadcrumbs and chrome. The
    # legacy ``pnlPRViewLeft`` id was a placeholder from α-1 and is not
    # present on the live site. Quote-agnostic per α-2 precedent.
    cleaned = _drop_chrome(html)
    for pat in (
        r'''<div\b[^>]*\bid=['"]doublescroll['"][^>]*>''',
        r'''<div\b[^>]*\bid=['"]pnlPRViewLeft['"][^>]*>''',
        r'''<div\b[^>]*\brole=['"]main['"][^>]*>''',
    ):
        inner = _extract_balanced_tag(cleaned, pat)
        if inner:
            text = strip_html(inner).strip()
            if text and len(text) > 200:
                return text
    return None


def _extract_norges_body(html: str) -> str | None:
    # Norges Bank press / news pages wrap the article prose in
    # ``<div class="article__main-body">`` (innermost, prose-only) nested
    # inside ``<article class="article">``. The legacy α-1 placeholder
    # targeted ``<div class="page-content">`` but on the live site that
    # class lives on the outer ``<main>`` and includes Javascript-disabled
    # boilerplate. Try the narrow body first, then fall back through the
    # article tag and the main wrapper. Quote-agnostic per α-2 precedent.
    cleaned = _drop_chrome(html)
    for pat in (
        r'''<div\b[^>]*\bclass=['"][^'"]*\barticle__main-body\b[^'"]*['"][^>]*>''',
        r'''<article\b[^>]*\bclass=['"][^'"]*\barticle\b[^'"]*['"][^>]*>''',
        r'''<main\b[^>]*\bid=['"]main-content['"][^>]*>''',
        r'''<div\b[^>]*\bclass=['"][^'"]*\bpage-content\b[^'"]*['"][^>]*>''',
    ):
        inner = _extract_balanced_tag(cleaned, pat)
        if inner:
            text = strip_html(inner).strip()
            if text and len(text) > 200:
                return text
    return None


# ---------------------------------------------------------------------------
# Generic fallback: pick the largest text-dense container.
# ---------------------------------------------------------------------------
_BODY_RX = re.compile(r"<body\b[^>]*>(.*?)</body\s*>",
                      flags=re.IGNORECASE | re.DOTALL)
_BLOCK_RX = re.compile(
    r"<(main|article|div)\b[^>]*>(.*?)</\1\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _default_extract_body(html: str) -> str:
    """Generic heuristic: drop chrome, find largest text-dense container."""
    if not html:
        return ""
    cleaned = _drop_chrome(html)
    body_match = _BODY_RX.search(cleaned)
    scope = body_match.group(1) if body_match else cleaned

    best_text = ""
    for m in _BLOCK_RX.finditer(scope):
        candidate = strip_html(m.group(2)).strip()
        if len(candidate) > len(best_text):
            best_text = candidate

    if best_text:
        return best_text
    # Last-resort: strip the entire scope.
    return strip_html(scope).strip()


# ---------------------------------------------------------------------------
# Registry + dispatcher
# ---------------------------------------------------------------------------
BODY_EXTRACTORS: dict[str, Callable[[str], str | None]] = {
    "FED":      _extract_fed_body,
    "ECB":      _extract_ecb_body,
    "BIS":      _extract_bis_body,
    "BOJ":      _extract_boj_body,
    "PBOC":     _extract_pboc_body,
    "RBA":      _extract_rba_body,
    "BCB":      _extract_bcb_body,
    "BANXICO":  _extract_banxico_body,
    "BOK":      _extract_bok_body,
    "RIKSBANK": _extract_riksbank_body,
    "RBI":      _extract_rbi_body,
    "NORGES":   _extract_norges_body,
}


def extract_body(html: str, *, bank_code: str | None = None) -> str:
    """Extract main body text from a CB HTML page.

    Looks up ``bank_code`` in ``BODY_EXTRACTORS``; if registered, tries
    the per-bank extractor and falls back to the generic on None.
    Unknown bank codes go straight to the generic fallback.
    """
    if not html:
        return ""
    extractor = BODY_EXTRACTORS.get(bank_code or "")
    if extractor is not None:
        result = extractor(html)
        if result:
            return result
    return _default_extract_body(html)


# ---------------------------------------------------------------------------
# PDF body extraction (optional pypdf dep). Used by SPA-only connectors
# (BCB) whose long-form text lives only in linked PDFs.
# ---------------------------------------------------------------------------
def extract_body_from_pdf(pdf_bytes: bytes) -> str | None:
    """Extract text from a PDF, returning None on parse error or when
    the extracted text is too short to be meaningful (< 200 chars).

    Returns None if ``pypdf`` is not installed — keeps the rest of this
    module usable in pypdf-less environments (Pyodide, minimal installs).
    """
    if not pdf_bytes:
        return None
    try:
        from pypdf import PdfReader  # optional dep
    except ImportError:
        return None
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        chunks = []
        for page in reader.pages:
            try:
                chunks.append(page.extract_text() or "")
            except Exception:
                continue
        text = "\n\n".join(chunks).strip()
    except Exception:
        return None
    return text if (text and len(text) > 200) else None


__all__ = ["extract_body", "extract_body_from_pdf", "BODY_EXTRACTORS"]
