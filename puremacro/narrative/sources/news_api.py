"""Generic news connector — currently wraps the GDELT 2.0 DOC API.

GDELT's DOC 2.0 API is free and key-less, scraping news articles
across thousands of outlets in 65 languages. It exposes an HTTP
endpoint at ``https://api.gdeltproject.org/api/v2/doc/doc`` returning
JSON for a given query and date range.

Usage:
    for date, text, url in iter_gdelt_v2(
            query="\"infrastructure bill\" OR \"recovery act\"",
            start="2008-01-01"):
        ...

For NYT / FT / Reuters paywalled APIs, add a sibling connector that
respects their auth.
"""
from __future__ import annotations

import urllib.parse
from typing import Iterator

import pandas as pd

from ._http import safe_get_json


_GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"


def iter_gdelt_v2(
    *,
    query: str,
    start: str,
    end: str | None = None,
    max_records: int = 250,
    source_country: str = "US",
) -> Iterator[tuple]:
    """Yield (date, title-blob, url) for GDELT-DOC-API hits.

    Parameters
    ----------
    query : GDELT 2.0 query syntax (the ``?query=`` param).
    start : ISO date.
    end : ISO date; defaults to today.
    max_records : per-call cap; GDELT 2.0 supports up to 250.
    source_country : restrict to news from a given country (ISO2-like).
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    full_query = f"({query}) sourcecountry:{source_country}"
    params = {
        "query": full_query,
        "mode": "ArtList",
        "format": "JSON",
        "startdatetime": start.replace("-", "") + "000000",
        "enddatetime": end.replace("-", "") + "235959",
        "maxrecords": str(max_records),
    }
    url = _GDELT_BASE + "?" + urllib.parse.urlencode(params)
    try:
        body = safe_get_json(url)
    except Exception:
        # Network failure or non-JSON rate-limit HTML — treat as empty.
        return
    arts = body.get("articles", [])
    for a in arts:
        try:
            date = pd.Timestamp(a["seendate"])
        except Exception:
            continue
        title = a.get("title", "")
        link = a.get("url", "")
        yield date, title, link


__all__ = ["iter_gdelt_v2"]
