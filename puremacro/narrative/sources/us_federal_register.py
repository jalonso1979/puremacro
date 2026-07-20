"""US Federal Register documents API.

Public endpoint, no auth required:
    https://www.federalregister.gov/api/v1/documents.json

Useful for retrieving "presidential documents" (executive orders,
proclamations on spending packages) and rules / notices from the
Treasury, OMB, and DoD.
"""
from __future__ import annotations

import urllib.parse
from typing import Iterator

import pandas as pd

from ._http import safe_get_json


_BASE = "https://www.federalregister.gov/api/v1/documents.json"


def iter_federal_register(
    *,
    since: str = "2008-01-01",
    until: str | None = None,
    agencies: tuple[str, ...] = ("treasury-department", "defense-department"),
    document_types: tuple[str, ...] = ("PRESDOCU", "RULE", "NOTICE"),
    per_page: int = 200,
    max_pages: int = 5,
) -> Iterator[tuple]:
    """Yield (date, abstract+title, html_url) records for matching documents.

    Parameters
    ----------
    since, until : ISO date strings; ``until=None`` defaults to today.
    agencies : Federal Register agency slugs (lowercase, hyphenated).
        See ``https://www.federalregister.gov/api/v1/agencies.json`` for
        the canonical list. Note: the slug for OMB has historically
        rotated and may not be ``office-of-management-and-budget``;
        verify before adding it back to the default.
    document_types : ``"PRESDOCU"``, ``"RULE"``, ``"NOTICE"``.
    per_page : page size; FR API max is 1000 per call.
    max_pages : safety cap.
    """
    until = until or pd.Timestamp.today().strftime("%Y-%m-%d")
    for page in range(1, max_pages + 1):
        params = {
            "conditions[publication_date][gte]": since,
            "conditions[publication_date][lte]": until,
            "conditions[type][]": list(document_types),
            "conditions[agencies][]": list(agencies),
            "per_page": per_page,
            "page": page,
            "fields[]": ["title", "abstract", "publication_date",
                          "html_url", "type", "agencies"],
        }
        url = _BASE + "?" + urllib.parse.urlencode(params, doseq=True)
        try:
            data = safe_get_json(url)
        except Exception:
            return
        results = data.get("results", [])
        if not results:
            return
        for r in results:
            text_blob = (r.get("title") or "") + " . " + (r.get("abstract") or "")
            yield (
                pd.Timestamp(r["publication_date"]),
                text_blob,
                r.get("html_url", ""),
            )
        if len(results) < per_page:
            return


__all__ = ["iter_federal_register"]
