"""EUR-Lex: binding EU legislative acts (regulations, directives, decisions).

Per-act content endpoints at
``https://eur-lex.europa.eu/legal-content/{LANG}/TXT/?uri=CELEX:{celex_id}``.
The CELEX ID encodes year + sector + act type. The search endpoint
enumerates acts by date and type.

This connector emits one record per (act × language). The same CELEX
ID yields multiple records — one for each requested language — with
the body text in that language.

Records are 4-tuples ``(date, text, source_url, metadata)`` where
``metadata`` carries:
  - ``language``: 'en' / 'de' / 'fr' (lower-case)
  - ``celex_id``: e.g., '32024R0903'
  - ``act_type``: 'regulation' / 'directive' / 'decision'
  - ``subject_codes``: list[str] from CELEX subject classification

**WAF note (2026-05-25):** eur-lex.europa.eu now returns HTTP 202 +
``x-amzn-waf-action: challenge`` to all programmatic clients, matching
the europarl.europa.eu WAF. All fetches therefore route through the
Wayback Machine using CDX-located 200-OK snapshots. Coverage is
necessarily historical; the most recent acts (last few months) may not
be Wayback-snapshotted yet.
"""
from __future__ import annotations

import json
import re
import warnings
from datetime import date as _date, datetime
from typing import Iterator

from bs4 import BeautifulSoup

from ._fallback import fetch_with_fallback, FallbackExhaustedError
from ._http import safe_get_text
from ._schema_check import assert_landmarks, ParserSchemaMismatchError
from ._wayback import wayback_snapshot_url

# eu_eurlex is currently Wayback-only — the live EUR-Lex endpoint is
# AWS-WAF-blocked (HTTP 202 + x-amzn-waf-action: challenge). If WAF is
# lifted in the future, change to ("live", "wayback").
FALLBACK_POLICY: tuple[str, ...] = ("wayback",)

PARSER_SCHEMA_VERSION = 1


_LEGAL_CONTENT_URL = (
    "https://eur-lex.europa.eu/legal-content/{lang}/TXT/?uri=CELEX:{celex}"
)
_SEARCH_URL = "https://eur-lex.europa.eu/search.html"

# CELEX type codes: R=regulation, L=directive, D=decision (modern era).
_CELEX_TYPE_CODES = {"R": "regulation", "L": "directive", "D": "decision"}

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)

# Body selector candidates (EUR-Lex rotates templates).
_BODY_SELECTORS = (
    "#document1",
    "#docHtml",
    "#text-html",
    "div.eli-main-title",
)


def _parse_eurlex_html(html: str, *, celex_id: str, language: str,
                       source_url: str) -> tuple | None:
    """Parse one EUR-Lex per-act HTML page into a 4-tuple record.

    Date is extracted from ``<meta property="eli:date_document">`` which
    contains the adoption date in ISO YYYY-MM-DD format. This is more
    reliable than regex on body text (body dates refer to cited older acts).

    Body is extracted from the first matching selector in ``_BODY_SELECTORS``.
    """
    assert_landmarks(
        html, source="eu_eurlex",
        expected_version=PARSER_SCHEMA_VERSION,
        landmarks=["CELEX", "EUR-Lex"],
    )
    soup = BeautifulSoup(html, "html.parser")

    body = None
    for sel in _BODY_SELECTORS:
        el = soup.select_one(sel)
        if el:
            candidate = el.get_text(" ", strip=True)
            if len(candidate) > 500:
                body = candidate
                break
    if not body:
        return None

    # Adoption date from ELI metadata (property="eli:date_document").
    # EUR-Lex pages embed RDFa metadata in <meta property="..."> tags.
    meta_date = soup.find("meta", property="eli:date_document")
    if meta_date and meta_date.get("content"):
        date_str = str(meta_date["content"]).strip()
        try:
            adoption_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            adoption_date = None
    else:
        adoption_date = None

    if adoption_date is None:
        # Fallback: look for "DD.MM.YYYY" in the OJ header line
        m = re.search(r"(?:OJ|Series L)[^.]*?(\d{1,2}\.\d{1,2}\.\d{4})", body[:500])
        if m:
            try:
                adoption_date = datetime.strptime(m.group(1), "%d.%m.%Y").date()
            except ValueError:
                pass

    if adoption_date is None:
        return None

    # Act type from CELEX. The pattern is 3<YYYY><type><number>, so:
    #   celex_id[0]   = '3' (sector: EU law)
    #   celex_id[1:5] = 4-digit year (e.g. '2024')
    #   celex_id[5]   = type code (R/L/D)
    # e.g. '32024R0903' → type code at index 5 = 'R' = regulation.
    act_type = "unknown"
    if len(celex_id) >= 6:
        type_code = celex_id[5]
        act_type = _CELEX_TYPE_CODES.get(type_code, "unknown")

    # Subject codes (best-effort — often in metadata table)
    subject_codes: list[str] = []
    full_text = soup.get_text(" ", strip=True)
    subj_m = re.search(r"Subject matter[:\s]*([^\n<]+)", full_text)
    if subj_m:
        subject_codes = [s.strip() for s in subj_m.group(1).split(";") if s.strip()]

    metadata = {
        "language": language.lower(),
        "celex_id": celex_id,
        "act_type": act_type,
        "subject_codes": subject_codes,
    }
    return (adoption_date, body, source_url, metadata)


def _fetch_celex_in_language(celex_id: str, language: str) -> tuple | None:
    """Fetch one CELEX act in one language via the governed fallback layer.

    EUR-Lex is WAF-blocked (HTTP 202 + x-amzn-waf-action: challenge) as
    of 2026-05-25. FALLBACK_POLICY is ("wayback",) so all fetches route
    through the Wayback Machine CDX index. Returns None on failure
    (CDX miss, fetch error, or parser failure).
    """
    direct_url = _LEGAL_CONTENT_URL.format(
        lang=language.upper(), celex=celex_id)
    try:
        html = fetch_with_fallback(
            direct_url, policy=FALLBACK_POLICY, source="eu_eurlex",
        )
    except FallbackExhaustedError:
        return None
    if not html.strip():
        return None
    return _parse_eurlex_html(html, celex_id=celex_id,
                               language=language, source_url=direct_url)


_SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"

# Cellar ontology resource-type URIs for binding acts.
_SPARQL_RESOURCE_TYPES = {
    "regulation": "REG",
    "directive": "DIR",
    "decision": "DEC",
}

# Map from Cellar resource-type codes to CELEX type letters.
# (Cellar's DIR does not correspond to CELEX's D; it corresponds to CELEX's L.)
_SPARQL_RTYPE_TO_CELEX_LETTER = {
    "REG": "R",
    "DIR": "L",  # directive in Cellar is DIR but CELEX uses L (from French Loi)
    "DEC": "D",
}

_SPARQL_QUERY_TEMPLATE = """PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT DISTINCT ?celex WHERE {{
  ?work cdm:work_id_document ?celex ;
        cdm:work_date_document ?date ;
        cdm:work_has_resource-type <http://publications.europa.eu/resource/authority/resource-type/{rtype}> .
  FILTER (?date >= "{start}"^^xsd:date && ?date <= "{end}"^^xsd:date)
}}
ORDER BY ?date
LIMIT 1000"""


def _celex_via_sparql(*, since: _date, until: _date,
                       act_types: tuple[str, ...]) -> list[str]:
    """Enumerate CELEX IDs via the EUR-Lex Cellar SPARQL endpoint.

    One SPARQL call per (act_type) covering the full date window.
    Returns CELEX IDs matching the expected ``3<YYYY><type-letter><number>``
    shape, deduplicated.
    """
    import urllib.parse as _up
    found: list[str] = []
    seen: set[str] = set()
    for act_type in act_types:
        rtype = _SPARQL_RESOURCE_TYPES.get(act_type)
        if not rtype:
            continue
        celex_letter = _SPARQL_RTYPE_TO_CELEX_LETTER.get(rtype)
        if not celex_letter:
            continue
        query = _SPARQL_QUERY_TEMPLATE.format(
            rtype=rtype, start=since.isoformat(), end=until.isoformat(),
        )
        post_body = _up.urlencode({
            "query": query,
            "format": "application/sparql-results+json",
        })
        url = f"{_SPARQL_ENDPOINT}?{post_body}"
        try:
            text = safe_get_text(url, user_agent=_USER_AGENT)
        except Exception:
            continue
        if not text.strip():
            continue
        try:
            data = json.loads(text)
        except Exception:
            continue
        bindings = data.get("results", {}).get("bindings", [])
        for b in bindings:
            celex = b.get("celex", {}).get("value", "")
            # Expected shape: 3<YYYY><letter><number>
            if re.match(rf"^3\d{{4}}{celex_letter}\d+$", celex) and celex not in seen:
                seen.add(celex)
                found.append(celex)
    return found


def _celex_via_html_scrape(*, since: _date, until: _date,
                            act_types: tuple[str, ...]) -> list[str]:
    """Legacy HTML-scrape enumeration (kept as fallback when SPARQL empty).

    EUR-Lex's search interface accepts year + type filters via the
    advanced-search URL. CELEX references are extracted from the
    result HTML. Currently degraded since the search page is also
    WAF-blocked.
    """
    type_letters = {
        "regulation": "R", "directive": "L", "decision": "D",
    }
    wanted_letters = {type_letters[t] for t in act_types if t in type_letters}
    if not wanted_letters:
        return []
    found: list[str] = []
    for year in range(since.year, until.year + 1):
        for letter in wanted_letters:
            url = (
                "https://eur-lex.europa.eu/search.html"
                f"?type=advanced&DTS_DOM=ALL&DTS_SUBDOM=ALL&DTA={year}"
                f"&DB_TYPE_OF_ACT={letter}&qid=1&page=1"
            )
            try:
                html = safe_get_text(url, user_agent=_USER_AGENT)
            except Exception:
                continue
            for celex in re.findall(rf"CELEX:?(3{year}{letter}\d+)", html):
                if celex not in found:
                    found.append(celex)
    return found


def _enumerate_celex_ids(*, since: _date, until: _date,
                          act_types: tuple[str, ...]) -> list[str]:
    """Enumerate CELEX IDs for binding acts in the date window.

    Tries SPARQL against the public EUR-Lex Cellar endpoint first.
    Falls back to HTML-scrape on empty SPARQL response (currently a
    no-op since EUR-Lex search HTML is WAF-blocked, but keeps the
    fallback path live for future site changes).
    """
    via_sparql = _celex_via_sparql(since=since, until=until,
                                    act_types=act_types)
    if via_sparql:
        return via_sparql
    return _celex_via_html_scrape(since=since, until=until,
                                    act_types=act_types)


def iter_eurlex(
    *,
    since: str = "2010-01-01",
    until: str | None = None,
    refetch: bool = False,
    act_types: tuple[str, ...] = ("regulation", "directive", "decision"),
    languages: tuple[str, ...] = ("en", "de", "fr"),
    max_acts_per_year: int | None = None,
) -> Iterator[tuple]:
    """Iterate EUR-Lex binding-act records.

    Yields one 4-tuple per (act × language). Records lacking a Wayback
    snapshot or language translation are silently skipped (graceful
    partial coverage).

    All fetches route through Wayback Machine because eur-lex.europa.eu
    is WAF-blocked (HTTP 202 + x-amzn-waf-action: challenge) as of
    2026-05-25.
    """
    import datetime as dt
    since_dt = dt.date.fromisoformat(since)
    until_dt = dt.date.fromisoformat(until) if until else dt.date.today()

    celex_ids = _enumerate_celex_ids(since=since_dt, until=until_dt,
                                      act_types=act_types)
    if max_acts_per_year is not None:
        # Group by year, cap each year
        from collections import defaultdict
        by_year: dict[str, list[str]] = defaultdict(list)
        for cid in celex_ids:
            y = cid[1:5]
            if len(by_year[y]) < max_acts_per_year:
                by_year[y].append(cid)
        celex_ids = [cid for ids in by_year.values() for cid in ids]

    for celex in celex_ids:
        for lang in languages:
            try:
                rec = _fetch_celex_in_language(celex, lang)
            except ParserSchemaMismatchError as e:
                from ._telemetry import log_event
                log_event(source="eu_eurlex", outcome="parser_schema_mismatch",
                          fallback_used="none")
                warnings.warn(
                    f"puremacro.narrative.sources.eu_eurlex: schema mismatch "
                    f"in batch {celex!r} ({lang}): {e}",
                    UserWarning, stacklevel=2,
                )
                continue
            if rec is None:
                continue
            date, *_ = rec
            if date < since_dt or date > until_dt:
                continue
            yield rec


__all__ = ["iter_eurlex"]
