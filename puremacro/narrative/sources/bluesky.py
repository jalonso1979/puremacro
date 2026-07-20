"""Bluesky archive of central-bank governors + finance ministers.

Fetches posts from a hand-curated list of handles via the AT Protocol
public XRPC endpoint (no auth required):

  GET https://public.api.bsky.app/xrpc/app.bsky.actor.getProfile?actor={handle}
  GET https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed?actor={did}&limit=100&cursor={cursor}

Per-actor flow: resolve handle → DID → paginate feed → emit 4-tuples.
Handles that don't resolve (HTTP 404 or empty response) are silently
skipped — many seeded handles will not have Bluesky accounts.

Records are 4-tuples ``(date, text, source_url, metadata)``:
  - date: post.record.createdAt (date only, UTC)
  - text: post.record.text (≤300 chars)
  - source_url: ``https://bsky.app/profile/{handle}/post/{rkey}``
  - metadata: {handle, did, name, role, country, actor_class,
    post_uri, langs}
"""
from __future__ import annotations

import json
import warnings
from datetime import date as _date, datetime
from typing import Iterator

from ._http import safe_get_json, safe_get_text
from ._schema_check import assert_landmarks, ParserSchemaMismatchError


PARSER_SCHEMA_VERSION = 1


_PROFILE_URL = (
    "https://public.api.bsky.app/xrpc/app.bsky.actor.getProfile?actor={actor}"
)
_FEED_URL = (
    "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"
    "?actor={actor}&limit={limit}{cursor}"
)
_POST_WEB_URL = "https://bsky.app/profile/{handle}/post/{rkey}"


# Hand-curated seed list of ~30 actors. Each entry is a dict with
# handle, name, role, country (ISO3), actor_class.
# Handles that do not resolve on Bluesky will be silently skipped by
# the connector; we ship a broad seed list and let runtime resolve
# which are actually active.
KNOWN_HANDLES: tuple[dict, ...] = (
    # Institutions (10)
    {"handle": "federalreserve.gov", "name": "Federal Reserve",
     "role": "central bank", "country": "USA", "actor_class": "institution"},
    {"handle": "ecb.europa.eu", "name": "European Central Bank",
     "role": "central bank", "country": "EUR", "actor_class": "institution"},
    {"handle": "bankofengland.co.uk", "name": "Bank of England",
     "role": "central bank", "country": "GBR", "actor_class": "institution"},
    {"handle": "boj.or.jp", "name": "Bank of Japan",
     "role": "central bank", "country": "JPN", "actor_class": "institution"},
    {"handle": "rba.gov.au", "name": "Reserve Bank of Australia",
     "role": "central bank", "country": "AUS", "actor_class": "institution"},
    {"handle": "bankofcanada.ca", "name": "Bank of Canada",
     "role": "central bank", "country": "CAN", "actor_class": "institution"},
    {"handle": "treasury.gov", "name": "US Treasury",
     "role": "treasury", "country": "USA", "actor_class": "institution"},
    {"handle": "imf.org", "name": "International Monetary Fund",
     "role": "multilateral", "country": "WORLD",
     "actor_class": "institution"},
    {"handle": "worldbank.org", "name": "World Bank",
     "role": "multilateral", "country": "WORLD",
     "actor_class": "institution"},
    {"handle": "bis.org", "name": "Bank for International Settlements",
     "role": "multilateral", "country": "WORLD",
     "actor_class": "institution"},
    # Governors (~20). Handles guessed by name conventions; many will
    # not resolve. Discovery (Task 1 Step 5) reports which do.
    {"handle": "jerompowell.bsky.social", "name": "Jerome Powell",
     "role": "Fed Chair", "country": "USA", "actor_class": "governor"},
    {"handle": "lagarde.bsky.social", "name": "Christine Lagarde",
     "role": "ECB President", "country": "EUR", "actor_class": "governor"},
    {"handle": "ueda.bsky.social", "name": "Kazuo Ueda",
     "role": "BoJ Governor", "country": "JPN", "actor_class": "governor"},
    {"handle": "abailey.bsky.social", "name": "Andrew Bailey",
     "role": "BoE Governor", "country": "GBR", "actor_class": "governor"},
    {"handle": "mbullock.bsky.social", "name": "Michele Bullock",
     "role": "RBA Governor", "country": "AUS", "actor_class": "governor"},
    {"handle": "tmacklem.bsky.social", "name": "Tiff Macklem",
     "role": "BoC Governor", "country": "CAN", "actor_class": "governor"},
    {"handle": "shaktikantadas.bsky.social", "name": "Shaktikanta Das",
     "role": "RBI Governor (recent)", "country": "IND",
     "actor_class": "governor"},
    {"handle": "rcamposneto.bsky.social", "name": "Roberto Campos Neto",
     "role": "BCB Governor (recent)", "country": "BRA",
     "actor_class": "governor"},
    {"handle": "victoriarodriguez.bsky.social",
     "name": "Victoria Rodríguez Ceja", "role": "Banxico Governor",
     "country": "MEX", "actor_class": "governor"},
    # Finance ministers (~20). Handles also guessed.
    {"handle": "bessent.bsky.social", "name": "Scott Bessent",
     "role": "US Treasury Secretary", "country": "USA",
     "actor_class": "minister"},
    {"handle": "rachelreeves.bsky.social", "name": "Rachel Reeves",
     "role": "UK Chancellor (recent)", "country": "GBR",
     "actor_class": "minister"},
    {"handle": "klingbeil.bsky.social", "name": "Lars Klingbeil",
     "role": "Germany Finance Min (recent)", "country": "DEU",
     "actor_class": "minister"},
    {"handle": "lemaire.bsky.social", "name": "Bruno Le Maire",
     "role": "France MEF (recent)", "country": "FRA",
     "actor_class": "minister"},
    {"handle": "giorgetti.bsky.social", "name": "Giancarlo Giorgetti",
     "role": "Italy MEF", "country": "ITA", "actor_class": "minister"},
    {"handle": "kkato.bsky.social", "name": "Katsunobu Kato",
     "role": "Japan MoF (recent)", "country": "JPN",
     "actor_class": "minister"},
    {"handle": "francoisphilippe.bsky.social",
     "name": "François-Philippe Champagne",
     "role": "Canada Finance Min", "country": "CAN",
     "actor_class": "minister"},
    {"handle": "edharguindeguy.bsky.social",
     "name": "Edmundo Sasso Harguindeguy",
     "role": "Argentina MoF (recent)", "country": "ARG",
     "actor_class": "minister"},
    {"handle": "fhaddad.bsky.social", "name": "Fernando Haddad",
     "role": "Brazil MoF", "country": "BRA", "actor_class": "minister"},
    {"handle": "rogelioramirez.bsky.social", "name": "Rogelio Ramírez de la O",
     "role": "Mexico SHCP (recent)", "country": "MEX",
     "actor_class": "minister"},
)


def _post_to_record(post_json: dict, *, actor_meta: dict,
                     languages: tuple[str, ...] = ("en",)) -> tuple | None:
    """Convert one feed item's post object to a 4-tuple record.

    Returns None for reposts, quote-posts, or records missing required
    fields.
    """
    if not isinstance(post_json, dict):
        return None
    post = post_json.get("post")
    if not isinstance(post, dict):
        return None
    record = post.get("record")
    if not isinstance(record, dict):
        return None
    # Skip reposts/quote-posts: feed item carries a "reason" field for
    # reposts; original posts have reason=None. Also skip posts whose
    # record type isn't app.bsky.feed.post.
    if post_json.get("reason") is not None:
        return None
    if record.get("$type") != "app.bsky.feed.post":
        return None
    # Skip posts with embedded record (quote-posts) — keep only
    # original text-bearing posts.
    embed = record.get("embed", {}) or {}
    if embed.get("$type") == "app.bsky.embed.record":
        return None

    text = record.get("text", "").strip()
    if not text:
        return None

    created_at = record.get("createdAt")
    if not isinstance(created_at, str):
        return None
    try:
        # Bluesky uses ISO 8601 with 'Z' suffix
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    rec_date = dt.date()

    langs = record.get("langs") or []
    # Language filter: accept if the post is tagged with at least one
    # of the requested languages, OR if the post has no langs tag at all
    # (Bluesky clients sometimes omit it; we don't penalize that).
    if langs:
        accepted = set(languages)
        if not (set(langs) & accepted):
            return None

    # Build canonical web URL from at:// URI
    post_uri = post.get("uri", "")
    if not post_uri.startswith("at://"):
        return None
    parts = post_uri.split("/")
    if len(parts) < 5:
        return None
    rkey = parts[-1]
    handle = actor_meta.get("handle", "")
    source_url = _POST_WEB_URL.format(handle=handle, rkey=rkey)

    metadata = {
        "handle": handle,
        "did": actor_meta.get("did", ""),
        "name": actor_meta.get("name", ""),
        "role": actor_meta.get("role", ""),
        "country": actor_meta.get("country", ""),
        "actor_class": actor_meta.get("actor_class", ""),
        "post_uri": post_uri,
        "langs": list(langs),
    }
    return (rec_date, text, source_url, metadata)


def _resolve_handle(handle: str) -> dict | None:
    """Resolve a handle to its DID + display name via getProfile.

    Returns None on 404 / non-200 / empty response.
    """
    url = _PROFILE_URL.format(actor=handle)
    try:
        resp = safe_get_json(url)
    except Exception:
        return None
    if not isinstance(resp, dict) or "did" not in resp:
        return None
    return {
        "did": resp.get("did", ""),
        "handle": resp.get("handle", handle),
        "displayName": resp.get("displayName", ""),
    }


def _iter_actor_feed(did: str, *, max_posts: int) -> Iterator[dict]:
    """Paginate getAuthorFeed for one DID, yielding raw feed items."""
    cursor = ""
    yielded = 0
    while yielded < max_posts:
        cursor_qs = f"&cursor={cursor}" if cursor else ""
        url = _FEED_URL.format(actor=did, limit=100, cursor=cursor_qs)
        try:
            resp = safe_get_json(url)
        except Exception:
            return
        if not isinstance(resp, dict):
            return
        feed = resp.get("feed", []) or []
        if not feed:
            return
        for item in feed:
            yield item
            yielded += 1
            if yielded >= max_posts:
                return
        cursor = resp.get("cursor", "")
        if not cursor:
            return


def iter_bluesky_posts(
    *,
    handles: tuple[dict, ...] | tuple[str, ...] = KNOWN_HANDLES,
    since: str = "2024-01-01",
    until: str | None = None,
    refetch: bool = False,
    max_posts_per_actor: int = 500,
    languages: tuple[str, ...] = ("en",),
) -> Iterator[tuple]:
    """Iterate Bluesky posts from the given handles.

    Parameters
    ----------
    handles : tuple of dicts (with at least 'handle' key) or tuple of
        bare handle strings. Default ``KNOWN_HANDLES``.
    since, until : ISO date strings; records outside this window are
        skipped. ``since`` clamps to 2024-01-01 (Bluesky public launch).
    refetch : reserved for future cache layer.
    max_posts_per_actor : pagination cap.
    languages : tuple of language codes ('en', 'de', 'fr', ...) — posts
        tagged with any of these are kept. Posts with no `langs` tag are
        always kept (Bluesky clients sometimes omit it). Default ``("en",)``
        preserves pre-0.63.0 behavior.

    Yields
    ------
    4-tuples ``(date, text, source_url, metadata)``.
    """
    import datetime as dt
    since_dt = dt.date.fromisoformat(since)
    until_dt = dt.date.fromisoformat(until) if until else dt.date.today()

    _checked = False
    for h in handles:
        # Normalize: bare string → dict with handle only
        if isinstance(h, str):
            actor_meta = {"handle": h}
        else:
            actor_meta = dict(h)
        handle = actor_meta.get("handle", "")
        if not handle:
            continue
        profile = _resolve_handle(handle)
        if profile is None:
            continue
        actor_meta["did"] = profile["did"]
        if not actor_meta.get("name"):
            actor_meta["name"] = profile.get("displayName", "")

        if not _checked:
            # Landmark check: fetch the raw JSON text of the first feed
            # page before entering normal iteration.
            _first_url = _FEED_URL.format(
                actor=profile["did"], limit=1, cursor="",
            )
            try:
                _raw_text = safe_get_text(_first_url)
            except Exception:
                _raw_text = ""
            try:
                assert_landmarks(
                    _raw_text, source="bluesky",
                    expected_version=PARSER_SCHEMA_VERSION,
                    landmarks=['"$type":', "app.bsky.feed.post"],
                )
            except ParserSchemaMismatchError as e:
                from ._telemetry import log_event
                log_event(source="bluesky", outcome="parser_schema_mismatch",
                          fallback_used="none")
                warnings.warn(
                    f"puremacro.narrative.sources.bluesky: schema mismatch "
                    f"on first body: {e}",
                    UserWarning, stacklevel=2,
                )
                return
            _checked = True

        for feed_item in _iter_actor_feed(profile["did"],
                                           max_posts=max_posts_per_actor):
            rec = _post_to_record(feed_item, actor_meta=actor_meta,
                                    languages=languages)
            if rec is None:
                continue
            date, *_ = rec
            if date < since_dt or date > until_dt:
                continue
            yield rec


__all__ = ["iter_bluesky_posts", "KNOWN_HANDLES", "PARSER_SCHEMA_VERSION"]
