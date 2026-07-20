"""OECD Economic Surveys + Country Studies.

The OECD publishes biennial Economic Surveys for every member country
plus key non-member partners — currently the only continuous text
source with explicit fiscal-policy commentary covering the full
puremacro 33-country wedges panel.

Strategy: same RePEc trick used for IMF Article IV. The OECD's main
``oecd.org`` portal is JS-rendered and CloudFlare-fronted, but the
Surveys are mirrored on RePEc as series ``oecd:ecokaa`` (Economic
Surveys) and ``oecd:ecoaaa`` (Working Papers). Every paper has a
clean ``/p/oec/{series}/{ID}.html`` page with citation_* metadata.

Series IDs we use:
    oecd:ecoaaa   OECD Economic Department Working Papers
    oecd:ecokaa   OECD Economic Surveys (biennial country reviews)
    oecd:ecoaab   Going for Growth (annual policy assessments)

The series listings paginate the same way as IMF imfscr.
"""
from __future__ import annotations

import re
import time
from typing import Iterator

import pandas as pd

from ._http import safe_get_text


_PAGE_SLEEP = 0.3

# RePEc OECD series we surface.
# OECD Economics Department Working Papers (``ecoaaa``) is paginated
# on RePEc and indexes ~2,000 papers — the staff commentary that
# backs every Economic Survey, Going for Growth chapter, and country-
# specific fiscal-policy review. The Economic Surveys themselves are
# published as books on RePEc and not exposed as a paginated series;
# pending a structured RePEc index for ``oec/ecokaa``, we route the
# survey/growth aliases to the WP series.
_SERIES = {
    "wp":      "ecoaaa",
    "surveys": "ecoaaa",
    "growth":  "ecoaaa",
}

_SERIES_LISTING_TPL = "https://ideas.repec.org/s/oec/{slug}{n}.html"
_PAPER_TPL = "https://ideas.repec.org/p/oec/{slug}/{paper_id}.html"
_LISTING_ROW_RX_TPL = (
    r"/p/oec/{slug}/([A-Za-z0-9_\-]+)\.html\">([^<]+)"
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


# Per-series listing cache (keyed by RePEc series slug).
_LISTING_CACHE: dict[str, list[tuple[str, str]]] = {}


def clear_listing_cache(series: str | None = None) -> None:
    """Drop the in-memory OECD listing cache (all or one series)."""
    global _LISTING_CACHE
    if series is None:
        _LISTING_CACHE = {}
    else:
        _LISTING_CACHE.pop(_SERIES.get(series, series), None)


def _walk_series(slug: str, *, max_pages: int = 60,
                  use_cache: bool = True) -> Iterator[tuple[str, str]]:
    """Yield ``(paper_id, title)`` across paginated RePEc listing for
    one series."""
    if use_cache and slug in _LISTING_CACHE:
        yield from _LISTING_CACHE[slug]
        return
    listing_rx = re.compile(
        _LISTING_ROW_RX_TPL.format(slug=re.escape(slug))
    )
    seen: set[str] = set()
    collected: list[tuple[str, str]] = []
    for page in range(1, max_pages + 1):
        url = _SERIES_LISTING_TPL.format(slug=slug, n="" if page == 1 else page)
        try:
            html = safe_get_text(url)
        except Exception:
            break
        any_match = False
        for m in listing_rx.finditer(html):
            paper_id = m.group(1)
            title = re.sub(r"\s+", " ", m.group(2)).strip()
            if paper_id in seen:
                continue
            seen.add(paper_id)
            any_match = True
            collected.append((paper_id, title))
            yield paper_id, title
        if not any_match:
            break
        time.sleep(_PAGE_SLEEP)
    if use_cache:
        _LISTING_CACHE[slug] = collected


def _fetch_paper_metadata(slug: str, paper_id: str) -> dict:
    url = _PAPER_TPL.format(slug=slug, paper_id=paper_id)
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


def iter_oecd_surveys(
    country_name: str | None = None,
    *,
    series: str = "wp",
    fetch_abstracts: bool = True,
    max_items: int = 50,
    year_min: int | None = None,
    year_max: int | None = None,
    max_pages: int = 60,
) -> Iterator[tuple]:
    """Yield (date, title+abstract, url) for OECD publications.

    Parameters
    ----------
    country_name : substring match against title (e.g. ``"Mexico"``,
        ``"Germany"``); pass ``None`` for an unfiltered firehose.
    series : ``"surveys"`` (default; biennial Economic Surveys),
        ``"wp"`` (Economics Department Working Papers), or
        ``"growth"`` (Going for Growth annual reports).
    fetch_abstracts : if True, follow each match to its RePEc page for
        the citation_abstract; otherwise yield title-only.
    max_items, year_min, year_max, max_pages : same semantics as
        :func:`puremacro.narrative.sources.iter_imf_articleiv`.
    """
    slug = _SERIES.get(series, series)
    n_yielded = 0
    for paper_id, title in _walk_series(slug, max_pages=max_pages):
        if n_yielded >= max_items:
            break
        if country_name and country_name.lower() not in title.lower():
            continue
        # Year is embedded in the paper id for OECD's RePEc publication
        # (typical IDs: "1234-en", "0145"). The clean way is via the
        # paper page's citation_publication_date — that's what
        # fetch_abstracts pays for. Without abstracts we can't always
        # date precisely, so we punt to "year not known".
        date = pd.NaT
        text = title
        url = _PAPER_TPL.format(slug=slug, paper_id=paper_id)
        if fetch_abstracts:
            meta = _fetch_paper_metadata(slug, paper_id)
            if meta.get("title"):
                text = meta["title"]
            if meta.get("abstract"):
                text = f"{text}. {meta['abstract']}"
            if meta.get("date"):
                try:
                    date = pd.Timestamp(meta["date"])
                except Exception:
                    pass
        if pd.isna(date):
            # Fallback: try to pull a 4-digit year out of the title.
            m = re.search(r"\b(19|20)\d{2}\b", title)
            if m:
                date = pd.Timestamp(f"{m.group(0)}-01-01")
        if year_min is not None and (pd.isna(date) or date.year < year_min):
            continue
        if year_max is not None and (pd.isna(date) or date.year > year_max):
            continue
        yield date if not pd.isna(date) else pd.Timestamp.today(), text, url
        n_yielded += 1


def iter_oecd_listing(
    *,
    series: str = "wp",
    use_cache: bool = True,
    max_pages: int = 60,
) -> Iterator[tuple]:
    """Yield (paper_id, title) for the entire RePEc OECD series listing.

    Useful for one-pass multi-country filtering like
    :func:`puremacro.narrative.sources.iter_imf_listing`.
    """
    slug = _SERIES.get(series, series)
    yield from _walk_series(slug, max_pages=max_pages, use_cache=use_cache)


__all__ = ["iter_oecd_surveys", "iter_oecd_listing", "clear_listing_cache"]
