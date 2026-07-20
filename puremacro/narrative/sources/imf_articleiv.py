"""IMF Article IV consultation reports + Staff Country Reports.

The IMF's main ``www.imf.org`` portal is fronted by a WAF that returns
403/blocked to scripted clients (verified May 2026). The same Staff
Country Reports (series ``imfscr``) are mirrored on RePEc, where the
series listing page (and per-paper pages) return clean HTML to a
generic UA.

Strategy:
  1. Fetch the RePEc series listing
     ``https://ideas.repec.org/s/imf/imfscr.html`` — gives the most
     recent ~50 staff country reports as ``<href>title</a>`` rows.
  2. Match each row's ``YYYY-NNN`` paper id and title; filter by
     country name (case-insensitive substring of the title).
  3. Optionally follow each paper to its RePEc page to recover the
     exact publication date and abstract via ``<meta name="citation_*">``.

Article IV consultation reports contain rich fiscal-policy commentary
on public-investment programmes, fiscal-consolidation paths, and tax
reforms — an excellent text source for narrative scoring.
"""
from __future__ import annotations

import re
import time
import urllib.parse
from typing import Iterator

import pandas as pd

from ._http import safe_get_text


# RePEc paginates the imfscr series across ~52 pages (verified 2026):
#   page 1:  imfscr.html
#   page 2:  imfscr2.html
#   ...
#   page N:  imfscr{N}.html
# Each page lists ~200 papers, oldest at the highest page number.
_SERIES_LISTING_TPL = "https://ideas.repec.org/s/imf/imfscr{n}.html"
_PAPER_TPL = "https://ideas.repec.org/p/imf/imfscr/{year}-{number}.html"
_PAGE_SLEEP = 0.3


# Match a RePEc IMF Staff Country Report listing row.
_LISTING_ROW_RX = re.compile(
    r"/p/imf/imfscr/(\d{4})-(\d+)\.html\">([^<]+)"
)
_REPEC_TITLE_RX = re.compile(
    r"<meta\s+name=\"citation_title\"\s+content=\"([^\"]+)\"",
    flags=re.IGNORECASE,
)
_REPEC_ABSTRACT_RX = re.compile(
    r"<meta\s+name=\"citation_abstract\"\s+content=\"([^\"]+)\"",
    flags=re.IGNORECASE | re.DOTALL,
)
_REPEC_DATE_RX = re.compile(
    r"<meta\s+name=\"citation_publication_date\"\s+content=\"([0-9/-]+)\"",
    flags=re.IGNORECASE,
)


def _fetch_repec_metadata(year: str, number: str) -> dict:
    url = _PAPER_TPL.format(year=year, number=str(number).zfill(3))
    try:
        html = safe_get_text(url)
    except Exception:
        return {}
    out: dict[str, str] = {"repec_url": url}
    if (m := _REPEC_TITLE_RX.search(html)):
        out["title"] = m.group(1).strip()
    if (m := _REPEC_ABSTRACT_RX.search(html)):
        out["abstract"] = re.sub(r"\s+", " ", m.group(1)).strip()
    if (m := _REPEC_DATE_RX.search(html)):
        out["date"] = m.group(1).strip()
    return out


# Module-level cache so repeated country queries within a session don't
# re-crawl the 52-page listing. Reset by calling ``clear_listing_cache()``.
_LISTING_CACHE: list[tuple[str, str, str]] | None = None


def clear_listing_cache() -> None:
    """Invalidate the in-memory IMF listing cache (forces a fresh crawl
    on the next call)."""
    global _LISTING_CACHE
    _LISTING_CACHE = None


def _iter_listing_pages(
    *,
    max_pages: int = 60,
    timeout: float = 60.0,
    use_cache: bool = True,
) -> Iterator[tuple[str, str, str]]:
    """Walk the paginated RePEc series listing, yielding ``(year, number,
    title)`` tuples in order. Stops at the first empty page (RePEc grew
    to ~52 pages as of 2026; ``max_pages=60`` future-proofs a few years
    of growth). Caches results within the process for instant re-use.
    """
    global _LISTING_CACHE
    if use_cache and _LISTING_CACHE is not None:
        yield from _LISTING_CACHE
        return

    collected: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for page in range(1, max_pages + 1):
        url = _SERIES_LISTING_TPL.format(n="" if page == 1 else page)
        try:
            html = safe_get_text(url, timeout=timeout)
        except Exception:
            break
        any_match = False
        for m in _LISTING_ROW_RX.finditer(html):
            year, number, title = m.group(1), m.group(2), m.group(3).strip()
            title = re.sub(r"\s+", " ", title)
            key = (year, number)
            if key in seen:
                continue
            seen.add(key)
            any_match = True
            collected.append((year, number, title))
            yield year, number, title
        if not any_match:
            break
        time.sleep(_PAGE_SLEEP)

    if use_cache:
        _LISTING_CACHE = collected


def iter_imf_articleiv(
    country_name: str | None = None,
    *,
    fetch_abstracts: bool = True,
    require_phrase: str = "Article IV",
    max_items: int = 50,
    max_pages: int = 60,
    year_min: int | None = None,
    year_max: int | None = None,
    timeout: float = 60.0,
) -> Iterator[tuple]:
    """Yield (date, title+abstract, url) for IMF Staff Country Reports.

    Walks the paginated RePEc series listing automatically — covers the
    full ~10,000 Staff Country Reports back to 1994.

    Parameters
    ----------
    country_name : substring match against the report title
        (e.g. ``"United States"``, ``"Mexico"``). When None, returns all
        reports without a country filter.
    fetch_abstracts : if True, follow each match to its RePEc page to
        recover the exact publication date + abstract; otherwise yield
        title-only with the year as date (faster). With per-paper
        fetches the rate is ~2-3 reports/sec; without, ~200/sec.
    require_phrase : keep only reports whose title contains this phrase
        (default ``"Article IV"``). Pass ``""`` to keep everything.
    max_items : stop after yielding this many matching reports.
    max_pages : safety cap on listing pages to walk (52 currently;
        60 buys a few years of forward compatibility).
    year_min, year_max : optional inclusive year bounds applied before
        ``fetch_abstracts``. Use these to skip the slow per-paper
        metadata fetches for irrelevant years.
    """
    n_yielded = 0
    for year, number, title in _iter_listing_pages(
        max_pages=max_pages, timeout=timeout,
    ):
        if n_yielded >= max_items:
            break
        if country_name and country_name.lower() not in title.lower():
            continue
        if require_phrase and require_phrase.lower() not in title.lower():
            continue
        y_int = int(year)
        if year_min is not None and y_int < year_min:
            continue
        if year_max is not None and y_int > year_max:
            continue

        date = pd.Timestamp(f"{year}-01-01")
        text = title
        url = _PAPER_TPL.format(year=year, number=str(number).zfill(3))

        if fetch_abstracts:
            meta = _fetch_repec_metadata(year, number)
            if meta.get("title"):
                text = meta["title"]
            if meta.get("abstract"):
                text = f"{text}. {meta['abstract']}"
            if meta.get("date"):
                try:
                    date = pd.Timestamp(meta["date"])
                except Exception:
                    pass

        yield date, text, url
        n_yielded += 1


def iter_imf_listing(
    *,
    max_pages: int = 60,
    use_cache: bool = True,
    timeout: float = 60.0,
) -> Iterator[tuple]:
    """Yield (year, number, title) for every IMF Staff Country Report on
    the RePEc series listing. Result is cached within the process.

    Useful when you want to filter the full ~10,000-report corpus in
    Python (e.g., by your own multi-country regex) rather than re-crawling
    once per country.
    """
    yield from _iter_listing_pages(max_pages=max_pages, timeout=timeout,
                                     use_cache=use_cache)


__all__ = ["iter_imf_articleiv", "iter_imf_listing", "clear_listing_cache"]
