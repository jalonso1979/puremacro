"""Reddit — public JSON API (roadmap item A-ext.1a).

Reddit exposes a read-only JSON API at ``reddit.com/r/<sub>/new.json``
and ``reddit.com/r/<sub>/top.json``. No OAuth required for read-only.
Rate-limited to ~60 requests/minute per IP — we don't paginate
aggressively to stay well under that cap.

For the LTUI / LWUI line we focus on labor-anxiety subreddits:
``r/LayOffs``, ``r/jobs``, ``r/recruitinghell``, ``r/unemployment``,
``r/antiwork``, ``r/cscareerquestions``, ``r/ExperiencedDevs``,
``r/wallstreetbets``, ``r/Economics``.

Each post carries ``score`` (upvote-downvote net), ``num_comments``,
``selftext`` (body for self-posts; empty for link-posts) in metadata
— suitable as kernel-side weights.

User-Agent: Reddit requires a distinctive UA per their docs; using
the puremacro UA satisfies the policy.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterator

import requests


_USER_AGENT = "puremacro/0.28 (https://github.com/jalonso1979/uncertainty_examples)"


def iter_reddit(
    subreddit: str,
    *,
    sort: str = "new",
    limit: int = 100,
    time_filter: str = "month",
    min_score: int | None = 1,
    fetch_body: bool = False,
) -> Iterator[tuple]:
    """Yield (date, title+selftext, url, metadata) for posts in ``r/<subreddit>``.

    Parameters
    ----------
    subreddit : subreddit name (no leading ``r/``).
    sort : ``"new"`` | ``"top"`` | ``"hot"`` | ``"rising"``.
    limit : posts to fetch (Reddit caps at 100 per request; we don't
        paginate to keep within the per-IP rate budget).
    time_filter : applies when ``sort="top"`` — ``"hour"`` | ``"day"`` |
        ``"week"`` | ``"month"`` | ``"year"`` | ``"all"``.
    min_score : drop posts below this score (filters spam / very-fresh
        items). ``None`` keeps everything.
    fetch_body : placeholder; Reddit ``.json`` already includes
        ``selftext`` for self-posts.
    """
    if sort not in ("new", "top", "hot", "rising"):
        raise ValueError(f"sort {sort!r} not supported")
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
    params: dict[str, int | str] = {"limit": min(int(limit), 100)}
    if sort == "top":
        params["t"] = time_filter
    try:
        r = requests.get(
            url, params=params,
            headers={"User-Agent": _USER_AGENT}, timeout=20,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return
    children = data.get("data", {}).get("children", [])
    for ch in children:
        post = ch.get("data") or {}
        ts = post.get("created_utc")
        if ts is None:
            continue
        try:
            dt = datetime.utcfromtimestamp(int(ts))
        except (ValueError, TypeError):
            continue
        score = post.get("score") or 0
        if min_score is not None and score < min_score:
            continue
        title = (post.get("title") or "").strip()
        selftext = (post.get("selftext") or "").strip()
        permalink = post.get("permalink") or ""
        post_url = (
            f"https://www.reddit.com{permalink}"
            if permalink else (post.get("url") or "")
        )
        text = f"{title}. {selftext}" if selftext else title
        if not text:
            continue
        yield (dt, text, post_url, {
            "doctype": "social_post",
            "language": "en",
            "bank_code": "REDDIT",
            "country": "MULTI",
            "subreddit": subreddit,
            "score": int(score),
            "num_comments": int(post.get("num_comments") or 0),
            "author": post.get("author"),
        })


__all__ = ["iter_reddit"]
