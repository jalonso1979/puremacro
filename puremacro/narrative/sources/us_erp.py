"""US Economic Report of the President (ERP).

Annual narrative report from the Council of Economic Advisers,
covering economic conditions of the prior year. ~80 reports
(1947-present). Source: GovInfo collection
(https://www.govinfo.gov/app/collection/erp).

Each ERP is published as a large PDF (~5-15 MB, 300-500 pages). The
connector splits each PDF into per-chapter records via a page-level
chapter-header detection; falls back to per-document if chapter
parsing fails.

Chapter detection strategy
--------------------------
Modern ERPs (mid-1990s onwards) open each chapter on a new page whose
first non-empty line is exactly "Chapter N" (e.g. "Chapter 1"), with
the chapter title on the immediately following non-empty line.  Older
ERPs may use an inline "CHAPTER N: Title" format on a single line.
Both patterns are detected:

1. Page-start pattern (primary): first non-empty line of a page matches
   ``^Chapter\\s+\\d+$`` — the chapter number and title are extracted
   from that page's first two non-empty lines.
2. Inline pattern (fallback): a line matches the inline regex
   ``^\\s*(?:CHAPTER|Chapter)\\s+(\\d+|[IVX]+)\\s*[:\\.\\-]?\\s*(.*)$``
   and the page-start pattern has not fired for this page.

The TOC also contains "Chapter N: Title ........ pg" lines, which would
cause false chapter splits if matched inline.  To suppress TOC false
positives the connector bundles all pre-first-chapter content (including
the TOC) as "front matter" and only starts a new chapter when the page-
start pattern fires or (for old-format ERPs) when the inline pattern fires
on what looks like an actual chapter-opening page (i.e. very few other
lines on that page).
"""
from __future__ import annotations

import io
import re
from datetime import date as _date
from typing import Iterator

import pdfplumber

from ._http import safe_get_bytes, safe_get_text


_GOVINFO_COLLECTION = "https://www.govinfo.gov/app/collection/erp"
_ERP_PDF_TEMPLATE: str = (
    "https://www.govinfo.gov/content/pkg/ERP-{year}/pdf/ERP-{year}.pdf"
)

# Page-start pattern: first non-empty line is exactly "Chapter N" or
# "CHAPTER N" (no title on same line).
_PAGE_START_RX = re.compile(r"^(?:CHAPTER|Chapter)\s+(\d+|[IVX]+)\s*$")

# Inline pattern: used as a fallback for older ERP formats where the
# chapter number and title appear on the same line.
_INLINE_CHAPTER_RX = re.compile(
    r"^\s*(?:CHAPTER|Chapter)\s+(\d+|[IVX]+)\s*[:\.\-]?\s*(.*)\s*$",
    re.MULTILINE,
)


def _extract_erp_pages(pdf_bytes: bytes) -> list[str]:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return [p.extract_text() or "" for p in pdf.pages]


def _page_nonblank_lines(page_text: str) -> list[str]:
    return [l.strip() for l in page_text.splitlines() if l.strip()]


def _split_chapters(pages: list[str]) -> list[tuple[str, str]]:
    """Return [(chapter_label, body_text), ...].

    Walks pages, detects chapter-opening pages via the page-start pattern
    (first non-empty line is exactly "Chapter N").  Older-format ERPs that
    use a single inline "CHAPTER N: Title" line are handled via the inline
    fallback when the page-start pattern never fires.

    Pre-chapter content (front matter, executive summary, TOC) is bundled
    under "front matter".
    """
    # --- Pass 1: try page-start detection ---
    chapter_page_indices: list[tuple[int, str]] = []  # (page_index, label)
    for idx, page_text in enumerate(pages):
        lines = _page_nonblank_lines(page_text)
        if not lines:
            continue
        m = _PAGE_START_RX.match(lines[0])
        if m:
            ch_num = m.group(1)
            title = lines[1] if len(lines) > 1 else ""
            label = f"Chapter {ch_num}" + (f": {title}" if title else "")
            chapter_page_indices.append((idx, label))

    # --- Pass 2 (fallback): if no page-start chapters found, try inline ---
    if not chapter_page_indices:
        for idx, page_text in enumerate(pages):
            lines = _page_nonblank_lines(page_text)
            if not lines:
                continue
            # Only match inline if the matching line is one of the first 3
            # non-blank lines (avoids TOC-page false positives).
            for li, line in enumerate(lines[:3]):
                m = _INLINE_CHAPTER_RX.match(line)
                if m:
                    ch_num, ch_title = m.group(1), m.group(2).strip()
                    # Strip trailing dotted-page-number artefacts from TOC
                    ch_title = re.sub(r"\.{2,}\s*\d+\s*$", "", ch_title).strip()
                    label = f"Chapter {ch_num}" + (
                        f": {ch_title}" if ch_title else ""
                    )
                    chapter_page_indices.append((idx, label))
                    break

    # --- Assemble chapter bodies ---
    chapters: list[tuple[str, str]] = []
    if not chapter_page_indices:
        # No chapter structure found; return full document as a single record
        full = " ".join(
            line
            for page_text in pages
            for line in _page_nonblank_lines(page_text)
        ).strip()
        if full:
            chapters.append(("document", full))
        return chapters

    # Build boundary list: (start_page, end_page_exclusive, label)
    first_chapter_page = chapter_page_indices[0][0]
    boundaries: list[tuple[int, int, str]] = []

    # Front matter: everything before the first chapter
    if first_chapter_page > 0:
        boundaries.append((0, first_chapter_page, "front matter"))

    for i, (start, label) in enumerate(chapter_page_indices):
        end = (
            chapter_page_indices[i + 1][0]
            if i + 1 < len(chapter_page_indices)
            else len(pages)
        )
        boundaries.append((start, end, label))

    for start, end, label in boundaries:
        body_tokens: list[str] = []
        for page_text in pages[start:end]:
            body_tokens.extend(_page_nonblank_lines(page_text))
        body = " ".join(body_tokens).strip()
        if body:
            chapters.append((label, body))

    return chapters


def _iter_erp_year(year: int, *, granularity: str = "per_chapter"
                   ) -> Iterator[tuple]:
    """Fetch and parse one annual ERP PDF."""
    url = _ERP_PDF_TEMPLATE.format(year=year)
    pdf_bytes = safe_get_bytes(url)
    if not pdf_bytes:
        return
    pages = _extract_erp_pages(pdf_bytes)
    full_text = "\n".join(pages).strip()
    if not full_text:
        return
    release_date = _date(year, 1, 15)  # ERPs typically released mid-January

    if granularity == "per_document":
        yield (release_date, full_text, url, {"year": year})
        return

    chapters = _split_chapters(pages)
    if not chapters:
        yield (release_date, full_text, url, {"year": year})
        return

    for chapter_label, body in chapters:
        if not body:
            continue
        yield (release_date, body, url,
               {"chapter": chapter_label, "year": year})


def iter_erp(
    *,
    since: str = "1947-01-01",
    until: str | None = None,
    refetch: bool = False,
    granularity: str = "per_chapter",
) -> Iterator[tuple]:
    """Iterate ERP records.

    Parameters
    ----------
    since, until : ISO date strings for the release window.
        ``until=None`` defaults to today.
    refetch : ignored (reserved for future caching layer).
    granularity : ``"per_chapter"`` (default) yields one record per
        parsed chapter; ``"per_document"`` yields one record per year.

    Yields
    ------
    (date, text, source_url, metadata) 4-tuples where metadata carries
    at minimum ``{"year": int}`` and, for per-chapter records,
    ``{"chapter": str, "year": int}``.
    """
    import datetime as dt
    since_dt = dt.date.fromisoformat(since)
    until_dt = dt.date.fromisoformat(until) if until else dt.date.today()
    for year in range(max(1947, since_dt.year), until_dt.year + 1):
        release = _date(year, 1, 15)
        if release < since_dt or release > until_dt:
            continue
        try:
            yield from _iter_erp_year(year, granularity=granularity)
        except Exception:
            continue


__all__ = ["iter_erp"]
