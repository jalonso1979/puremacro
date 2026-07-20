"""Fed Beige Book corpus (federalreserve.gov modern + FRASER historical).

Each release is a national summary plus 12 district reports. Each
district report is split into canonical sections via a synonym table.
The connector emits one record per (release, district, canonical_section)
tuple at the default ``granularity='per_section'``.

Records are 6-tuples ``(date, district, section, text, source_url, metadata)``
where ``metadata`` is a dict carrying parse-quality flags and the
original section header string. Older 3-tuple consumers can positionally
unpack the first three fields.
"""
from __future__ import annotations

import re
import warnings
from datetime import date as _date
from typing import Iterator

from ._schema_check import assert_landmarks, ParserSchemaMismatchError

PARSER_SCHEMA_VERSION = 1


# Canonical sections — 6 plus "other" catch-all. Each district report's
# raw section headers map onto these via _SECTION_SYNONYMS below.
CANONICAL_SECTIONS: tuple[str, ...] = (
    "overall", "labor", "prices", "consumer",
    "manufacturing", "realestate", "other",
)

# Synonym matching is case-insensitive prefix matching: a header is
# canonicalised to the first section whose synonym set has any element
# that appears as a substring of the lowercased header.
_SECTION_SYNONYMS: dict[str, frozenset[str]] = {
    "overall": frozenset({
        "summary", "overall economic activity", "overall activity",
        "current economic conditions", "general economic conditions",
        "highlights", "economic activity",
    }),
    "labor": frozenset({
        "employment", "labor market", "labor markets",
        "wages", "wage", "workforce", "hiring", "jobs",
        "labor",  # catches "Labor Costs", "Labor Conditions", etc.
    }),
    "prices": frozenset({
        "prices", "inflation",
    }),
    "consumer": frozenset({
        "consumer spending", "consumer", "retail",
        "tourism", "travel",
    }),
    "manufacturing": frozenset({
        "manufacturing", "production", "industry", "industrial",
    }),
    "realestate": frozenset({
        "real estate", "construction", "housing",
        "commercial real estate", "residential",
    }),
}

# Twelve Fed districts (as appear in modern Beige Book release index pages)
DISTRICTS: tuple[str, ...] = (
    "Boston", "New York", "Philadelphia", "Cleveland", "Richmond",
    "Atlanta", "Chicago", "St. Louis", "Minneapolis", "Kansas City",
    "Dallas", "San Francisco",
)


def canonical_section(raw_header: str) -> str:
    """Map a raw district section header to a canonical section.

    Returns one of ``CANONICAL_SECTIONS``. Empty/None header → ``"other"``.
    Match is case-insensitive substring against the synonym tables in
    a deterministic CANONICAL_SECTIONS order.
    """
    if not raw_header:
        return "other"
    h = raw_header.lower().strip()
    for canon in CANONICAL_SECTIONS:
        if canon == "other":
            continue
        for syn in _SECTION_SYNONYMS[canon]:
            if syn in h:
                return canon
    return "other"


# ---------------------------------------------------------------------------
# Modern backend (federalreserve.gov, 1996-present)
#
# Three sub-layouts, auto-detected per release:
#
# (a) Per-district pages with sidebar links (Jan 2024-present):
#   Index:   beigebook{YYYYMM}.htm         — <a class="list-group-item">
#                                            links to all district pages
#   Summary: beigebook{YYYYMM}-summary.htm — national summary
#   District: beigebook{YYYYMM}-{slug}.htm — per-district content
#   Structure within each district page:
#     h3  = "Federal Reserve Bank of <District>"
#     h4  = section header (e.g. "Summary of Economic Activity",
#            "Labor Markets", "Prices", "Retail and Tourism", ...)
#     p   = paragraph content under each h4
#
# (b) Single-page inline layout (~2017 through Nov 2023): the index page
#   itself carries all twelve district reports —
#     h4               = "Federal Reserve Bank of <District>"
#     p > strong       = section header, body text in the same <p>
#   Only the National Summary has its own -summary.htm sub-page (also in
#   the p > strong section style). No -{slug}.htm district pages exist.
#
# (c) Pre-2017 layout: index page with no parseable links; per-district
#   pages exist at the canonical beigebook{YYYYMM}-{slug}.htm pattern and
#   are enumerated directly (see _enumerate_pre2017_district_urls).
# ---------------------------------------------------------------------------

from bs4 import BeautifulSoup  # noqa: E402 (after __future__ imports OK here)

# 0.60.0: opt into the cached + rate-limited variants. The uncached
# helpers remain available; other connectors still use them.
from ..._http import (
    safe_get_text_cached as safe_get_text,
    safe_get_bytes_cached as safe_get_bytes,
)  # noqa: E402


_MODERN_BASE = "https://www.federalreserve.gov/monetarypolicy/beigebook{yyyymm}"
_BASE_URL = "https://www.federalreserve.gov"

# Map canonical district name → URL slug used by federalreserve.gov
_DISTRICT_SLUGS: dict[str, str] = {
    "Boston": "boston",
    "New York": "new-york",
    "Philadelphia": "philadelphia",
    "Cleveland": "cleveland",
    "Richmond": "richmond",
    "Atlanta": "atlanta",
    "Chicago": "chicago",
    "St. Louis": "st-louis",
    "Minneapolis": "minneapolis",
    "Kansas City": "kansas-city",
    "Dallas": "dallas",
    "San Francisco": "san-francisco",
}


def _modern_index_url(year: int, month: int) -> str:
    return _MODERN_BASE.format(yyyymm=f"{year}{month:02d}") + ".htm"


def _modern_summary_url(year: int, month: int) -> str:
    return _MODERN_BASE.format(yyyymm=f"{year}{month:02d}") + "-summary.htm"


def _modern_district_url(year: int, month: int, slug: str) -> str:
    return _MODERN_BASE.format(yyyymm=f"{year}{month:02d}") + f"-{slug}.htm"


def _enumerate_pre2017_district_urls(year: int, month: int) -> list[tuple[str, str]]:
    """Return [(name, url)] for a pre-2017 release using the canonical
    URL pattern. ``name`` is either ``"__summary__"`` or the canonical
    district name (key of ``_DISTRICT_SLUGS``).

    Used by ``_parse_modern_html`` as a fallback when the index page
    doesn't expose ``list-group-item`` anchors (the pre-2017 layout).
    """
    out: list[tuple[str, str]] = [
        ("__summary__", _modern_summary_url(year, month)),
    ]
    for district, slug in _DISTRICT_SLUGS.items():
        out.append((district, _modern_district_url(year, month, slug)))
    return out


def _parse_summary_page(html: str, *, release_date: _date,
                        source_url: str) -> list[tuple]:
    """Parse the national summary page into records.

    The summary page has h4 section headers (Overall Economic Activity,
    Labor Markets, Prices) followed by paragraphs, then an h4
    'Highlights by Federal Reserve District' with h5 district names
    and short blurb paragraphs.

    Yields one record per h4 section in the national summary, plus one
    'overall' record for the aggregate blurb if present.
    """
    soup = BeautifulSoup(html, "html.parser")
    records: list[tuple] = []

    # Find all h4 tags that are before the 'Highlights' section
    h4_tags = soup.find_all("h4")
    highlights_idx = None
    for i, h4 in enumerate(h4_tags):
        if "highlights" in h4.get_text(strip=True).lower():
            highlights_idx = i
            break

    summary_h4s = h4_tags[:highlights_idx] if highlights_idx is not None else h4_tags

    for h4 in summary_h4s:
        raw_header = h4.get_text(strip=True)
        chunks: list[str] = []
        cur = h4.find_next_sibling()
        while cur is not None and cur.name not in {"h4", "h3", "h2"}:
            if cur.name in {"p", "div", "li"}:
                t = cur.get_text(" ", strip=True)
                if t:
                    chunks.append(t)
            cur = cur.find_next_sibling()
        text = " ".join(chunks).strip()
        if text:
            records.append((
                release_date, "National", canonical_section(raw_header),
                text, source_url, {"raw_header": raw_header},
            ))

    # ~2017-2023 layout: sections are <p><strong>Header</strong> body</p>
    # (no h4 headers). Stop at the 'Highlights by Federal Reserve
    # District' blurb block, mirroring the h4 path above.
    if not records:
        container = soup.find(id="article") or soup
        for p in container.find_all("p"):
            strong = p.find("strong")
            if strong is None:
                continue
            raw_header = strong.get_text(" ", strip=True)
            if not raw_header:
                continue
            if "highlights" in raw_header.lower():
                break
            strong.extract()  # body = paragraph text minus the header
            text = p.get_text(" ", strip=True)
            if text:
                records.append((
                    release_date, "National", canonical_section(raw_header),
                    text, source_url, {"raw_header": raw_header},
                ))

    # If no structured sections found, fall back to all paragraphs
    if not records:
        paras = []
        for p in soup.find_all("p"):
            t = p.get_text(" ", strip=True)
            if len(t) > 80:
                paras.append(t)
        if paras:
            records.append((
                release_date, "National", "overall",
                " ".join(paras), source_url, {"raw_header": "Summary"},
            ))

    return records


def _parse_district_page(html: str, *, district: str, release_date: _date,
                         source_url: str,
                         granularity: str = "per_section") -> list[tuple]:
    """Parse one district page into records.

    Structure: h4 section headers with paragraphs as content.
    """
    soup = BeautifulSoup(html, "html.parser")
    records: list[tuple] = []

    h4_tags = soup.find_all("h4")
    if not h4_tags:
        return records

    # Collect (raw_header, text) pairs
    sections: list[tuple[str, str]] = []
    for h4 in h4_tags:
        raw_header = h4.get_text(strip=True)
        chunks: list[str] = []
        cur = h4.find_next_sibling()
        while cur is not None and cur.name not in {"h4", "h3", "h2"}:
            if cur.name in {"p", "div", "li"}:
                t = cur.get_text(" ", strip=True)
                if t:
                    chunks.append(t)
            cur = cur.find_next_sibling()
        text = " ".join(chunks).strip()
        if text:
            sections.append((raw_header, text))

    if granularity == "per_district":
        all_text = " ".join(t for _, t in sections).strip()
        if all_text:
            records.append((
                release_date, district, "all", all_text,
                source_url, {"raw_header": "district-bundle"},
            ))
    else:
        for raw_header, text in sections:
            records.append((
                release_date, district, canonical_section(raw_header),
                text, source_url, {"raw_header": raw_header},
            ))

    return records


_BOILERPLATE_PREFIX = "for more information about district economic conditions"


def _parse_inline_district_reports(
    soup: "BeautifulSoup", *, release_date: _date, source_url: str,
    granularity: str = "per_section",
) -> list[tuple]:
    """Parse the ~2017-2023 single-page layout where all twelve district
    reports live on the release index page itself.

    Structure: ``<h4>Federal Reserve Bank of <District></h4>`` followed by
    ``<p><strong>Section Header</strong> body text...</p>`` paragraphs
    until the next h4. Returns [] when the page carries no inline
    district reports (so callers can fall through to other layouts).
    """
    container = soup.find(id="article") or soup
    records: list[tuple] = []

    for h4 in container.find_all("h4"):
        header_text = h4.get_text(" ", strip=True)
        if "federal reserve bank of" not in header_text.lower():
            continue
        district = None
        for cand in DISTRICTS:
            if cand.lower() in header_text.lower():
                district = cand
                break
        if district is None:
            continue

        # Collect (raw_header, text) section pairs until the next h2-h4.
        sections: list[tuple[str, str]] = []
        current_header: str | None = None
        current_chunks: list[str] = []

        def _flush() -> None:
            nonlocal current_header, current_chunks
            body = " ".join(current_chunks).strip()
            if current_header is not None and body:
                sections.append((current_header, body))
            current_chunks = []

        cur = h4.find_next_sibling()
        while cur is not None and cur.name not in {"h4", "h3", "h2"}:
            if cur.name in {"p", "div", "li"}:
                strong = cur.find("strong") if cur.name == "p" else None
                if strong is not None and strong.get_text(strip=True):
                    _flush()
                    current_header = strong.get_text(" ", strip=True)
                    strong.extract()  # leave only the body in the <p>
                t = cur.get_text(" ", strip=True)
                if t and not t.lower().startswith(_BOILERPLATE_PREFIX):
                    if current_header is None:
                        current_header = "Summary"
                    current_chunks.append(t)
            cur = cur.find_next_sibling()
        _flush()

        if granularity == "per_district":
            all_text = " ".join(t for _, t in sections).strip()
            if all_text:
                records.append((
                    release_date, district, "all", all_text,
                    source_url, {"raw_header": "district-bundle"},
                ))
        else:
            for raw_header, text in sections:
                records.append((
                    release_date, district, canonical_section(raw_header),
                    text, source_url, {"raw_header": raw_header},
                ))

    return records


def _parse_modern_html(html: str, *, release_date: _date, source_url: str,
                       granularity: str = "per_section",
                       year: int | None = None,
                       month: int | None = None) -> Iterator[tuple]:
    """Parse a federalreserve.gov modern Beige Book index HTML.

    For post-2017 releases the index page has ``<a class="list-group-item">``
    anchors pointing to per-district pages. For pre-2017 releases that
    markup is absent — fall back to URL-pattern enumeration via
    ``_enumerate_pre2017_district_urls``. ``year``/``month`` are only
    needed for the fallback branch.

    Yields (date, district, section, text, source_url, metadata) tuples.
    """
    assert_landmarks(
        html, source="beige_book",
        expected_version=PARSER_SCHEMA_VERSION,
        landmarks=["Beige Book"],
    )
    soup = BeautifulSoup(html, "html.parser")
    district_links: list[tuple[str, str]] = []

    for a in soup.find_all("a", class_="list-group-item"):
        href = str(a.get("href", ""))
        text = a.get_text(strip=True)
        if not href or not text:
            continue
        full_url = href if href.startswith("http") else _BASE_URL + href
        if text == "National Summary":
            district_links.append(("__summary__", full_url))
        else:
            for district in DISTRICTS:
                if district.lower() in text.lower():
                    district_links.append((district, full_url))
                    break

    # ~2017-2023 single-page layout: the index page links only the
    # National Summary; all twelve district reports are inline on the
    # index page itself (h4 bank headers + p>strong sections).
    if not any(name != "__summary__" for name, _ in district_links):
        inline_records = _parse_inline_district_reports(
            soup, release_date=release_date, source_url=source_url,
            granularity=granularity,
        )
        if inline_records:
            for district_name, url in district_links:  # summary link(s)
                page_html = safe_get_text(url)
                if page_html.strip():
                    yield from _parse_summary_page(
                        page_html, release_date=release_date, source_url=url,
                    )
            yield from inline_records
            return

    # Pre-2017 fallback: enumerate via the canonical URL pattern.
    if not district_links and year is not None and month is not None:
        district_links = _enumerate_pre2017_district_urls(year, month)

    for district_name, url in district_links:
        page_html = safe_get_text(url)
        if not page_html.strip():
            continue
        if district_name == "__summary__":
            yield from _parse_summary_page(
                page_html, release_date=release_date, source_url=url,
            )
        else:
            yield from _parse_district_page(
                page_html, district=district_name,
                release_date=release_date, source_url=url,
                granularity=granularity,
            )


def _iter_modern(year: int, month: int, *,
                 granularity: str = "per_section") -> Iterator[tuple]:
    """Fetch and parse one modern Beige Book release.

    Fetches the index page then follows links to the summary and each
    district sub-page. When the modern index does not exist (the pattern
    only goes back to 2017 on the live site — older requests 404), falls
    back to the archived layouts via ``_iter_archive``.
    """
    url = _modern_index_url(year, month)
    try:
        html = safe_get_text(url)
    except Exception:
        html = ""
    if not html.strip():
        yield from _iter_archive(year, month, granularity=granularity)
        return
    yield from _parse_modern_html(
        html, release_date=_date(year, month, 1), source_url=url,
        granularity=granularity, year=year, month=month,
    )


# ---------------------------------------------------------------------------
# Fed FOMC historical PDF backend (federalreserve.gov, 1983-1995)
#
# Pre-1996 Beige Books (titled "Summary of Commentary on Current Economic
# Conditions by Federal Reserve Districts") are archived on the Federal
# Reserve Board's FOMC historical materials pages:
#
#   Index: https://www.federalreserve.gov/monetarypolicy/fomchistorical{YEAR}.htm
#   PDF:   https://www.federalreserve.gov/monetarypolicy/files/fomc{DATE}beige{DATE}.pdf
#
# Note: This was originally scoped as a "FRASER backend" but exhaustive FRASER
# search found no national Beige Book series there. FRASER only hosts
# district-level Beige Books from individual Reserve Banks.
#
# Available range: mid-1983 through 1995.
# Pre-1983: not available in digital PDF form on the Fed website.
#
# PDF structure (consistent across 1983-1995):
#   Page 1: Cover (3 lines: title, subtitle, date)
#   Page 2: Table of Contents
#   Pages 3-5: National summary (topical sections: Overview, Retailing,
#              Manufacturing and Industry, Construction and Real Estate,
#              Agriculture, Finance)
#   Pages 6+: Per-district sections, each starting with a header line
#              of the form "FIRST DISTRICT - BOSTON" or
#              "SECOND DISTRICT--NEW YORK" (dash or double-dash variants).
#              Within each district, section sub-headers are short
#              title-cased or all-caps lines (1-5 words).
# ---------------------------------------------------------------------------

import io as _io

# Try pdfplumber (preferred — already in the env alongside other narrative deps).
# Fall back gracefully if not installed.
try:
    import pdfplumber as _pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _pdfplumber = None  # type: ignore[assignment]
    _PDFPLUMBER_AVAILABLE = False

_FOMC_HISTORICAL_BASE = (
    "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
)
_FOMC_PDF_BASE = "https://www.federalreserve.gov/monetarypolicy/files/"

# Ordinal → canonical district name
_ORDINAL_TO_DISTRICT: dict[str, str] = {
    "FIRST": "Boston",
    "SECOND": "New York",
    "THIRD": "Philadelphia",
    "FOURTH": "Cleveland",
    "FIFTH": "Richmond",
    "SIXTH": "Atlanta",
    "SEVENTH": "Chicago",
    "EIGHTH": "St. Louis",
    "NINTH": "Minneapolis",
    "TENTH": "Kansas City",
    "ELEVENTH": "Dallas",
    "TWELFTH": "San Francisco",
}

_ORDINAL_PAT = re.compile(
    r"^(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH)"
    r"\s+DISTRICT\s*[-–—]+",
    re.IGNORECASE,
)

# Page-number artifacts to skip: e.g. "II-1", "IV-2", "iii", "iv", "viii"
_PAGE_NUMBER_PAT = re.compile(r"^[IVXivx]+-\d*[a-z]?$|^[ivxlc]+$", re.IGNORECASE)


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract concatenated text from all pages of a Beige Book PDF.

    Uses pdfplumber which preserves layout. Empty pages return empty strings.
    Returns an empty string if pdfplumber is not installed.
    """
    if not _PDFPLUMBER_AVAILABLE:
        return ""
    with _pdfplumber.open(_io.BytesIO(pdf_bytes)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(pages).strip()


def _parse_fomc_pdf(text: str, *, release_date: _date, source_url: str,
                    granularity: str = "per_section") -> Iterator[tuple]:
    """Parse extracted FOMC historical Beige Book PDF text into records.

    The national summary and per-district sections are identified by line-level
    heuristics:
    - District boundaries: lines matching ``ORDINAL DISTRICT - CityName``
    - Section boundaries within a district: short (1-5 word) lines that start
      with an uppercase letter, do not end with sentence punctuation, and are
      not page-number artifacts.

    Yields (date, district, section, text, source_url, metadata) 6-tuples.
    """
    lines = [ln.strip() for ln in text.splitlines()]

    current_district = "National"
    current_section_header = "Summary"
    current_chunks: list[str] = []

    def _flush() -> Iterator[tuple]:
        nonlocal current_chunks
        body = " ".join(current_chunks).strip()
        if body:
            yield (
                release_date,
                current_district,
                canonical_section(current_section_header),
                body,
                source_url,
                {"raw_header": current_section_header},
            )
        current_chunks = []

    for ln in lines:
        if not ln:
            continue

        # Skip cover/TOC artifacts: page-number codes, dotted TOC lines
        if _PAGE_NUMBER_PAT.match(ln):
            continue
        if "....." in ln or "-----" in ln:
            continue

        # --- District boundary detection ---
        dist_match = _ORDINAL_PAT.match(ln)
        if dist_match:
            ordinal = dist_match.group(1).upper()
            new_district = _ORDINAL_TO_DISTRICT.get(ordinal)
            if new_district:
                yield from _flush()
                current_district = new_district
                current_section_header = "Summary"
                continue

        # --- Section header detection ---
        # A section header is a short line (1-5 words) that:
        # 1. Starts with uppercase
        # 2. Does not end with sentence punctuation
        # 3. Is not a page-number artifact
        # 4. Is not a bare district name (running page header in PDFs)
        words = ln.split()
        word_count = len(words)
        if (1 <= word_count <= 5
                and ln[0].isupper()
                and ln[-1] not in ".,:;"
                and not _PAGE_NUMBER_PAT.match(ln)):
            # Reject bare district names — they appear as running page headers
            # in historical PDFs and would cause false section splits.
            if ln.strip() in DISTRICTS:
                current_chunks.append(ln)
                continue
            # Check against synonym table — if it matches a canonical section,
            # treat it as a section header
            candidate_canon = canonical_section(ln)
            if candidate_canon != "other":
                yield from _flush()
                current_section_header = ln
                continue
            # Also accept "other"-mapped headers that look header-like
            # (title-cased or all-caps, no lowercase start after first word)
            if (ln.istitle() or ln.isupper()
                    or (word_count <= 3 and all(w[0].isupper() for w in words))):
                yield from _flush()
                current_section_header = ln
                continue

        current_chunks.append(ln)

    yield from _flush()


def _fetch_fomc_pdf(pdf_url: str, *, release_date: _date,
                    granularity: str = "per_section") -> Iterator[tuple]:
    """Fetch and parse a single FOMC-historical Beige Book PDF.

    Takes a pre-resolved ``pdf_url`` (typically from ``_fomc_listing``).
    No re-scraping of the year index. Yields 6-tuples.
    """
    if not _PDFPLUMBER_AVAILABLE:
        return
    pdf_bytes = safe_get_bytes(pdf_url)
    if not pdf_bytes:
        return
    text = _extract_pdf_text(pdf_bytes)
    if not text.strip():
        return
    yield from _parse_fomc_pdf(
        text, release_date=release_date, source_url=pdf_url,
        granularity=granularity,
    )


def _iter_fomc_historical(year: int, month: int, *,
                          granularity: str = "per_section") -> Iterator[tuple]:
    """Thin backwards-compat wrapper: fetch and parse one Beige Book
    issue by (year, month). Re-scrapes the year listing — prefer
    ``_fetch_fomc_pdf(pdf_url, ...)`` when you already have the URL.
    """
    if not _PDFPLUMBER_AVAILABLE:
        return
    issues = _fomc_listing(year)
    candidates = [(y, m, u) for (y, m, u) in issues if y == year]
    if not candidates:
        return
    best = min(candidates, key=lambda t: abs(t[1] - month))
    _, _, pdf_url = best
    yield from _fetch_fomc_pdf(
        pdf_url, release_date=_date(year, month, 1), granularity=granularity,
    )


def _fomc_listing(year: int) -> list[tuple[int, int, str]]:
    """Scrape the FOMC historical index page for (year, month, pdf_url) triples.

    Returns an empty list if pdfplumber is not installed or the page is
    unreachable.
    """
    if not _PDFPLUMBER_AVAILABLE:
        return []
    index_url = _FOMC_HISTORICAL_BASE.format(year=year)
    html = safe_get_text(index_url)
    if not html.strip():
        return []
    soup = BeautifulSoup(html, "html.parser")
    issues: list[tuple[int, int, str]] = []
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        m = re.search(r"fomc(\d{4})(\d{2})\d{2}beige(\d{4})(\d{2})\d{2}\.pdf",
                      href, re.IGNORECASE)
        if m:
            ry, rm = int(m.group(3)), int(m.group(4))
            if 1 <= rm <= 12:
                full_url = (
                    "https://www.federalreserve.gov" + href
                    if href.startswith("/") else href
                )
                issues.append((ry, rm, full_url))
    return issues


# ---------------------------------------------------------------------------
# Archived pre-2017 layouts (live federalreserve.gov archive)
#
# The modern index pattern /monetarypolicy/beigebook{YYYYMM}.htm only
# exists from 2017 onward; requesting it for older releases returns 404.
# Older releases remain live on federalreserve.gov under two archived
# layouts, discovered via the per-year archive index pages
# (/monetarypolicy/beigebook{YYYY}.htm, linked from
# /monetarypolicy/beige-book-archive.htm):
#
# (d) ~2011-2016: /monetarypolicy/beigebook/beigebook{YYYYMM}.htm — the
#     full report on one page. District boundaries are <h2> headers of
#     the form "First District--Boston"; within a district, section
#     headers are <strong> leaders inside <p> tags; the national summary
#     follows an <h2> "Summary of Commentary on Current Economic
#     Conditions by Federal Reserve District".
# (e) 1996-2010: /fomc/beigebook/{YYYY}/{YYYYMMDD}/FullReport.htm — same
#     content, but district boundaries are ALSO <strong> leaders
#     ("First District--Boston"; 1996 uses "First District - Boston"),
#     told apart from section headers by the ordinal pattern.
#
# Months found under neither (d) nor (e) fall back to the
# FOMC-historical PDF listing (backend above) — e.g. Jan-Sep 1996,
# whose issues are only published as PDFs on the FOMC pages.
# ---------------------------------------------------------------------------

_ARCHIVE_PAGE_BASE = (
    "https://www.federalreserve.gov/monetarypolicy/beigebook/beigebook{yyyymm}.htm"
)
_ARCHIVE_YEAR_INDEX = (
    "https://www.federalreserve.gov/monetarypolicy/beigebook{year}.htm"
)
_FOMC_ERA_RELEASE_PAT = re.compile(
    r"/fomc/beigebook/(\d{4})/(\d{4})(\d{2})(\d{2})/default\.htm", re.IGNORECASE
)
_ARCHIVE_SUMMARY_MARKER = "summary of commentary"
# Section headers are short <strong> leaders ("Labor Markets", "Agriculture
# and Natural Resources"); longer leaders ("Prepared at the Federal Reserve
# Bank of ...") are body text.
_ARCHIVE_MAX_HEADER_WORDS = 6


def _archive_page_url(year: int, month: int) -> str:
    return _ARCHIVE_PAGE_BASE.format(yyyymm=f"{year}{month:02d}")


def _archive_fomc_era_listing(year: int) -> dict[int, str]:
    """Enumerate the fomc-era releases of ``year`` -> FullReport.htm URLs.

    Returns {release_month: FullReport.htm URL}. Scrapes the Beige Book
    year archive index (/monetarypolicy/beigebook{year}.htm) first, then
    fills months it omits from the FOMC historical materials page for
    the same year — the two listings are inconsistently maintained
    (e.g. the 2003 year index omits the September 3, 2003 release that
    fomchistorical2003.htm lists). Empty dict when neither page lists
    /fomc/beigebook/ releases (the ~2011+ years, whose releases live at
    the (d) pattern instead).
    """
    out: dict[int, str] = {}
    for index_url in (_ARCHIVE_YEAR_INDEX.format(year=year),
                      _FOMC_HISTORICAL_BASE.format(year=year)):
        try:
            html = safe_get_text(index_url)
        except Exception:
            continue
        if not html.strip():
            continue
        for m in _FOMC_ERA_RELEASE_PAT.finditer(html):
            ry, rm = int(m.group(2)), int(m.group(3))
            if ry == year and 1 <= rm <= 12 and rm not in out:
                out[rm] = (
                    "https://www.federalreserve.gov"
                    + m.group(0)[: -len("default.htm")] + "FullReport.htm"
                )
    return out


def _parse_archive_report(html: str, *, release_date: _date, source_url: str,
                          granularity: str = "per_section") -> list[tuple]:
    """Parse a single-page archived full report (layouts (d) and (e)).

    Walks <h2>/<h3>/<p> elements in document order. District boundaries
    are ordinal headers ("First District--Boston") appearing either as
    h2/h3 text (layout d) or as the <strong> leader of a <p> (layout e);
    the national summary starts at the "Summary of Commentary ..."
    marker. Short <strong> leaders inside <p> are section headers.
    Content before the first marker (nav, title) is dropped.
    """
    assert_landmarks(
        html, source="beige_book",
        expected_version=PARSER_SCHEMA_VERSION,
        landmarks=["District"],
    )
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find(id="article") or soup

    sections: list[tuple[str, str, str]] = []  # (district, raw_header, body)
    current_district: str | None = None
    current_header = "Summary"
    current_chunks: list[str] = []

    def _flush() -> None:
        nonlocal current_chunks
        body = " ".join(current_chunks).strip()
        if current_district is not None and body:
            sections.append((current_district, current_header, body))
        current_chunks = []

    def _district_of(text: str) -> str | None:
        m = _ORDINAL_PAT.match(text.strip())
        if m:
            return _ORDINAL_TO_DISTRICT.get(m.group(1).upper())
        return None

    for el in container.find_all(["h2", "h3", "p"]):
        if el.name in ("h2", "h3"):
            txt = el.get_text(" ", strip=True)
            district = _district_of(txt)
            if district is not None:
                _flush()
                current_district = district
                current_header = "Summary"
            elif _ARCHIVE_SUMMARY_MARKER in txt.lower():
                _flush()
                current_district = "National"
                current_header = "Summary"
            continue

        # <p>: a <strong> leader is a district boundary, the national-
        # summary marker, or a section header; anything else is body.
        p_text = el.get_text(" ", strip=True)
        strong = el.find(["strong", "b"])
        if strong is not None:
            lead = strong.get_text(" ", strip=True)
            is_leader = bool(lead) and p_text.startswith(lead)
            if is_leader:
                district = _district_of(lead)
                if district is not None:
                    _flush()
                    current_district = district
                    current_header = "Summary"
                    strong.extract()
                elif _ARCHIVE_SUMMARY_MARKER in lead.lower():
                    _flush()
                    current_district = "National"
                    current_header = "Summary"
                    strong.extract()
                elif (current_district is not None
                        and len(lead.split()) <= _ARCHIVE_MAX_HEADER_WORDS):
                    _flush()
                    current_header = lead
                    strong.extract()
                p_text = el.get_text(" ", strip=True)

        if current_district is None:
            continue
        low = p_text.lower()
        if (p_text and not low.startswith(_BOILERPLATE_PREFIX)
                and low != "return to top"):
            current_chunks.append(p_text)
    _flush()

    records: list[tuple] = []
    if granularity == "per_district":
        order: list[str] = []
        bundle: dict[str, list[str]] = {}
        for district, _, body in sections:
            if district not in bundle:
                order.append(district)
                bundle[district] = []
            bundle[district].append(body)
        for district in order:
            records.append((
                release_date, district, "all",
                " ".join(bundle[district]).strip(),
                source_url, {"raw_header": "district-bundle"},
            ))
    else:
        for district, raw_header, body in sections:
            records.append((
                release_date, district, canonical_section(raw_header),
                body, source_url, {"raw_header": raw_header},
            ))
    return records


def _archive_fullreport_text(html: str) -> str:
    """Linear text of a fomc-era FullReport page, trimmed to the report.

    Drops everything before the first content marker (the "Summary of
    Commentary ..." title line or the first ordinal district header) so
    banner/nav text stays out of the National summary record.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "title"]):
        tag.extract()
    lines = (soup.find(id="article") or soup).get_text("\n").splitlines()
    start = 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if _ARCHIVE_SUMMARY_MARKER in s.lower() or _ORDINAL_PAT.match(s):
            start = i
            break
    return "\n".join(lines[start:])


def _archive_records_substantive(records: list[tuple]) -> bool:
    """True when a parsed archive page carries real report content.

    Pre-2011 pages at the (d) URL pattern are table-of-contents stubs
    (h2 district headers + "Return to top" links, no body paragraphs);
    they parse to nearly nothing and must fall through to layout (e).
    A real full report covers >= 6 districts with substantial text.
    """
    districts = {r[1] for r in records if r[1] != "National"}
    total_chars = sum(len(r[3]) for r in records)
    return len(districts) >= 6 and total_chars >= 5000


def _iter_archive(year: int, month: int, *,
                  granularity: str = "per_section") -> Iterator[tuple]:
    """Fetch and parse one archived (pre-2017) release, trying layout (d),
    then (e), then the FOMC-historical PDF listing."""
    # (d) ~2011-2016 single-page archive pattern. Pre-2011 the same URL
    # serves a TOC stub — only accept a substantive parse.
    try:
        html = safe_get_text(_archive_page_url(year, month))
    except Exception:
        html = ""
    if html.strip():
        # The (d) probe is speculative: a landmark mismatch here (e.g. a
        # generic stub served for a month the archive does not list)
        # must fall through to (e)/PDF, not abort the month.
        try:
            records = _parse_archive_report(
                html, release_date=_date(year, month, 1),
                source_url=_archive_page_url(year, month),
                granularity=granularity,
            )
        except ParserSchemaMismatchError:
            records = []
        if _archive_records_substantive(records):
            yield from records
            return

    # (e) 1996-2010 fomc-era FullReport pages, via the year archive index.
    # These pages have unclosed-tag table markup that defeats a DOM walk
    # (wrapper <p> elements swallow the document), so parse the *linear
    # text* with the FOMC-PDF line state machine — the content model is
    # identical (ordinal district header lines + short section-header
    # lines).
    full_url = _archive_fomc_era_listing(year).get(month)
    if full_url:
        try:
            html = safe_get_text(full_url)
        except Exception:
            html = ""
        if html.strip():
            assert_landmarks(
                html, source="beige_book",
                expected_version=PARSER_SCHEMA_VERSION,
                landmarks=["District"],
            )
            yield from _parse_fomc_pdf(
                _archive_fullreport_text(html),
                release_date=_date(year, month, 1),
                source_url=full_url, granularity=granularity,
            )
            return

    # PDF-only issues (e.g. Jan-Sep 1996) on the FOMC historical pages.
    for ry, rm, pdf_url in _fomc_listing(year):
        if ry == year and rm == month:
            yield from _fetch_fomc_pdf(
                pdf_url, release_date=_date(year, month, 1),
                granularity=granularity,
            )
            return


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def iter_beige_book(
    *,
    since: str = "1983-01-01",
    until: str | None = None,
    refetch: bool = False,
    granularity: str = "per_section",
) -> Iterator[tuple]:
    """Iterate Beige Book records from ``since`` to ``until`` (inclusive).

    Routes modern (1996+) requests to the federalreserve.gov per-release
    pages and pre-1996 (1983-1995) to the FOMC historical materials
    pages. Each release contributes one National Summary record plus
    per-district records (per-section by default).

    Parameters
    ----------
    since, until : ISO date strings. ``until=None`` means present.
        Note: pre-1983 is not digitally available; ``since`` is clamped
        to 1983-01-01.
    refetch : currently a no-op (kept for backwards-compat). To bypass
        the on-disk HTTP cache (live as of 0.60.0) set the env var
        ``PUREMACRO_HTTP_NO_CACHE=1``.
    granularity : ``'per_section'`` (default) or ``'per_district'``.

    Yields
    ------
    6-tuples ``(date, district, section, text, source_url, metadata)``.

    Notes
    -----
    Modern release URLs are ``beigebook<YYYYMM>.htm`` with 8 releases/year;
    the release month drifts across years, so all 12 months are probed
    and nonexistent ones skipped (they 404).
    Pre-1996 issues live on the FOMC historical materials pages —
    discovered via ``_fomc_listing(year)`` which scrapes the year-index
    page for available PDFs.
    """
    import datetime as dt
    since_dt = dt.date.fromisoformat(since)
    until_dt = dt.date.fromisoformat(until) if until else dt.date.today()

    # Pre-1996: FOMC historical materials (1983-1995).
    if since_dt.year < 1996:
        for year in range(max(1983, since_dt.year), min(1996, until_dt.year + 1)):
            for year_, month, pdf_url in _fomc_listing(year):
                release = _date(year_, month, 1)
                if release < since_dt or release > until_dt:
                    continue
                try:
                    yield from _fetch_fomc_pdf(
                        pdf_url, release_date=release,
                        granularity=granularity,
                    )
                except ParserSchemaMismatchError as e:
                    from ._telemetry import log_event
                    log_event(source="beige_book", outcome="parser_schema_mismatch",
                              fallback_used="none")
                    warnings.warn(
                        f"puremacro.narrative.sources.beige_book: schema mismatch "
                        f"in batch {pdf_url!r}: {e}",
                        UserWarning, stacklevel=2,
                    )
                    continue
                except Exception:
                    continue

    # Modern 1996+ releases — probe every month. The Beige Book comes out
    # 8 times/year but the release month drifts (e.g. Nov 30 2022 and
    # May 31 2023 instead of the canonical Dec/Jun); a fixed 8-month list
    # silently drops those releases. Nonexistent months 404 quickly and
    # are skipped by the per-release error handling below.
    modern_months = tuple(range(1, 13))
    for year in range(max(1996, since_dt.year), until_dt.year + 1):
        for month in modern_months:
            release = _date(year, month, 1)
            if release < since_dt or release > until_dt:
                continue
            try:
                yield from _iter_modern(year, month, granularity=granularity)
            except ParserSchemaMismatchError as e:
                from ._telemetry import log_event
                log_event(source="beige_book", outcome="parser_schema_mismatch",
                          fallback_used="none")
                warnings.warn(
                    f"puremacro.narrative.sources.beige_book: schema mismatch "
                    f"in batch {year}-{month:02d}: {e}",
                    UserWarning, stacklevel=2,
                )
                continue
            except Exception:
                continue


__all__ = [
    "iter_beige_book",
    "CANONICAL_SECTIONS", "DISTRICTS",
    "canonical_section",
]
