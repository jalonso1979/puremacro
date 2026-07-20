"""US State of the Union (SOTU) addresses.

Annual addresses delivered to Congress. ~250 SOTUs (1790-present),
plus written annual messages in early years.

Source: UCSB Presidency Project (https://www.presidency.ucsb.edu/documents/).
The category-listing page enumerates per-document URLs; each document
page renders the full transcript with consistent HTML structure.

Per-document granularity only (speeches are linear; no canonical
sub-sections). Records are 4-tuples ``(date, text, source_url, metadata)``
with metadata carrying president name + year.

Discovery notes
---------------
The plan's original approach (keyword-anchor heuristic on the guidebook
listing page) did not yield individual document links because the guidebook
page uses a link format without SOTU keywords in the slug.

Instead we use the UCSB Presidency Project category pages directly:

- **Spoken** addresses (73 records, all on one page with
  ``items_per_page=100``):
  ``/documents/app-categories/spoken-addresses-and-remarks/presidential/
  state-the-union-addresses``

- **Written** messages (≈145 records, two pages at ``items_per_page=100``):
  ``/documents/app-categories/citations/presidential/
  state-the-union-written-messages``

Together these two categories cover all ~218 SOTU/annual-message records
available on the site.

Each document page uses Drupal-rendered HTML.  Confirmed selectors
(verified against 2024 fixture):

- Body: ``div.field-docs-content``  (falls back down the ``_BODY_SELECTORS``
  chain if the Drupal theme changes).
- Date: ``span.date-display-single``
- President: ``h3.diet-title`` (first ``<h3>`` before the body div with
  class ``diet-title``), falling back to ``div.field-docs-person``.
"""
from __future__ import annotations

import re
from datetime import date as _date, datetime
from typing import Iterator

from bs4 import BeautifulSoup

from ._http import safe_get_text


_UCSB_BASE = "https://www.presidency.ucsb.edu"

# Two category listing pages that together cover all ~218 SOTU records.
# Each page is fetched with items_per_page=100 to minimise HTTP calls;
# a second request (page=1) is made when the pager indicates more pages.
_CATEGORY_URLS = [
    # Spoken SOTU addresses (~73 records, fits one page at 100/page)
    (
        _UCSB_BASE
        + "/documents/app-categories/spoken-addresses-and-remarks/presidential"
        "/state-the-union-addresses"
    ),
    # Written annual messages (~145 records, two pages at 100/page)
    (
        _UCSB_BASE
        + "/documents/app-categories/citations/presidential"
        "/state-the-union-written-messages"
    ),
]

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)

_BODY_SELECTORS = (
    "div.field-docs-content",
    "div.field-body",
    "div.docs-content",
    "div.node__content",
)

# URL fragments that are NOT individual document pages
_SKIP_FRAGS = frozenset(
    ["archive-guidebook", "app-categories", "category-attributes",
     "items_per_page", "?page"]
)


def _parse_sotu_page(html: str, *, source_url: str) -> tuple | None:
    soup = BeautifulSoup(html, "html.parser")

    body = None
    for sel in _BODY_SELECTORS:
        el = soup.select_one(sel)
        if el:
            body = el.get_text(" ", strip=True)
            break
    if not body or len(body) < 200:
        return None

    date_text = None
    for sel in ("span.date-display-single", "time",
                ".field-name-field-docs-start-date-time"):
        el = soup.select_one(sel)
        if el:
            date_text = el.get_text(strip=True)
            break
    parsed_date = _parse_date(date_text)
    if not parsed_date:
        return None

    # President name: prefer `h3.diet-title` (Drupal-rendered byline),
    # fall back to `div.field-docs-person`, then h1 prefix heuristic.
    president = ""
    pres_el = soup.select_one("h3.diet-title") or soup.select_one(
        "div.field-docs-person"
    )
    if pres_el:
        # The element may include the title line "46th President of...";
        # take only text before the first newline / digit / "President of".
        raw = pres_el.get_text(" ", strip=True)
        raw = re.split(r"\d|President of", raw, maxsplit=1)[0].strip()
        president = raw.strip(",. ")
    if not president:
        title_el = soup.select_one("h1") or soup.title
        if title_el:
            title_text = title_el.get_text(" ", strip=True)
            m = re.match(r"^([A-Z][a-zA-Z\.\s]+?):", title_text)
            if m:
                president = m.group(1).strip()

    metadata = {"president": president, "year": parsed_date.year}
    return (parsed_date, body, source_url, metadata)


def _parse_date(text: str | None) -> _date | None:
    if not text:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _doc_links_from_category(base_url: str) -> list[str]:
    """Scrape one category listing page (all pages) for document URLs.

    Fetches with ``items_per_page=100``; follows page=N links until the
    pager's ``last`` link disappears or we see no new rows.
    """
    urls: list[str] = []
    seen: set[str] = set()
    page = 0
    while True:
        sep = "&" if "?" in base_url else "?"
        url = f"{base_url}?items_per_page=100" + (
            f"&page={page}" if page > 0 else ""
        )
        html = safe_get_text(url, user_agent=_USER_AGENT)
        if not html.strip():
            break
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.find_all(class_="views-row")
        if not rows:
            break
        for row in rows:
            a = row.find("a", href=True)
            if not a:
                continue
            href: str = str(a["href"])
            if "/documents/" not in href:
                continue
            if any(frag in href for frag in _SKIP_FRAGS):
                continue
            full = href if href.startswith("http") else _UCSB_BASE + href
            if full not in seen:
                seen.add(full)
                urls.append(full)
        # Check whether there is a "next page" link
        next_link = soup.select_one("li.pager-next a, li.next a")
        if not next_link:
            break
        page += 1
        if page > 20:  # safety guard
            break
    return urls


def _sotu_listing() -> list[str]:
    """Return de-duplicated SOTU document URLs from both category pages."""
    all_urls: list[str] = []
    seen: set[str] = set()
    for cat_url in _CATEGORY_URLS:
        for u in _doc_links_from_category(cat_url):
            if u not in seen:
                seen.add(u)
                all_urls.append(u)
    return all_urls


def iter_sotu(
    *,
    since: str = "1790-01-01",
    until: str | None = None,
    refetch: bool = False,
) -> Iterator[tuple]:
    """Iterate SOTU records from ``since`` to ``until`` (inclusive).

    Parameters
    ----------
    since, until : ISO date strings bounding the address date.
        ``until=None`` defaults to today.
    refetch : ignored (reserved for future caching layer).

    Yields
    ------
    (date, text, source_url, metadata) 4-tuples where metadata carries
    ``{"president": str, "year": int}``.
    """
    import datetime as dt
    since_dt = dt.date.fromisoformat(since)
    until_dt = dt.date.fromisoformat(until) if until else dt.date.today()

    for url in _sotu_listing():
        try:
            html = safe_get_text(url, user_agent=_USER_AGENT)
            record = _parse_sotu_page(html, source_url=url)
        except Exception:
            continue
        if record is None:
            continue
        doc_date, *_ = record
        if doc_date < since_dt or doc_date > until_dt:
            continue
        yield record


__all__ = ["iter_sotu"]
