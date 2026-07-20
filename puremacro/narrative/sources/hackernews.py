"""Hacker News — public search API (roadmap item A-ext.1b).

Hacker News exposes two free public APIs:
  - **Firebase REST** (``hacker-news.firebaseio.com``): canonical raw
    item store, no rate limit, but no search — only sequential IDs.
  - **Algolia search** (``hn.algolia.com/api/v1/search_by_date``): full
    text search by tag/query/date, capped at 1000 items per query but
    paginated via ``page=`` parameter.

For the LTUI / LWUI line we want query-driven retrieval (e.g. items
mentioning "layoff" or "AI") rather than full firehose ingest, so we
default to the Algolia endpoint. Each record carries ``score`` and
``num_comments`` in metadata — useful as kernel-side weights.

Tags + query combine as a logical AND in Algolia's grammar.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterator

import requests


_API = "https://hn.algolia.com/api/v1/search_by_date"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)


def iter_hackernews(
    query: str,
    *,
    tags: str = "story",
    hits_per_page: int = 100,
    max_pages: int = 10,
    min_score: int | None = None,
    fetch_body: bool = False,
) -> Iterator[tuple]:
    """Yield (date, title+text, url, metadata) for HN items matching ``query``.

    Parameters
    ----------
    query : Algolia query string (boolean ops via space = AND).
    tags : Algolia tag filter; default ``"story"`` (excludes comments).
        Use ``"comment"`` for top-level comment search, ``"poll"``,
        ``"job"``, or combined like ``"story,front_page"``.
    hits_per_page : max 100 per request (Algolia limit).
    max_pages : pagination cap; default 10 → up to 1000 items.
    min_score : optional minimum HN score; items below are dropped.
        ``None`` keeps everything.
    fetch_body : placeholder (HN stories don't carry body text via
        Algolia by default; story_text is exposed when available).
    """
    for page in range(max_pages):
        params: dict[str, str | int] = {
            "query": query,
            "tags": tags,
            "hitsPerPage": hits_per_page,
            "page": page,
        }
        try:
            r = requests.get(
                _API, params=params,
                headers={"User-Agent": _USER_AGENT}, timeout=20,
            )
            r.raise_for_status()
            data = r.json()
        except Exception:
            return
        hits = data.get("hits", [])
        if not hits:
            return
        for h in hits:
            ts = h.get("created_at_i")
            if ts is None:
                continue
            try:
                dt = datetime.utcfromtimestamp(int(ts))
            except (ValueError, TypeError):
                continue
            title = (h.get("title") or "").strip()
            story_text = (h.get("story_text") or "").strip()
            comment_text = (h.get("comment_text") or "").strip()
            url = h.get("url") or (
                f"https://news.ycombinator.com/item?id={h.get('objectID')}"
            )
            score = h.get("points")
            num_comments = h.get("num_comments")
            if min_score is not None and (score or 0) < min_score:
                continue
            text = title or story_text or comment_text
            if story_text and story_text != text:
                text = f"{text}. {story_text}"
            elif comment_text and comment_text != text:
                text = f"{text}. {comment_text}"
            if not text:
                continue
            yield (dt, text, url, {
                "doctype": "story" if tags.startswith("story") else "comment",
                "language": "en",
                "bank_code": "HN",
                "country": "MULTI",
                "score": int(score) if score is not None else None,
                "num_comments": int(num_comments) if num_comments is not None else None,
                "author": h.get("author"),
                "query": query,
            })


__all__ = ["iter_hackernews"]
