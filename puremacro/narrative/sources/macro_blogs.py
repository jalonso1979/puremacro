"""Macroeconomic Blogs, Research Portals, and Substack Aggregator.

Ingests real-time and historical narrative commentary from premier macroeconomics
blogs, research hubs, and economist discussions:
- **VoxEU / CEPR Policy Portal**: Academic policy debates and empirical findings.
- **Apricitas Economics (Joseph Politano)**: High-frequency macro/labor/inflation dives.
- **Marginal Revolution (Cowen & Tabarrok)**: Daily macro, trade, and regulatory commentary.
- **The Macro Compass (Alfonso Peccatiello)**: Global liquidity and monetary plumbing.
- **NBER Digest**: Digest of new macroeconomic working papers.

Follows the standard puremacro source generator interface:
``yield (date, title_and_text, url, metadata)``
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterator

from ._rss import parse_rss_feed

# Curated macroeconomic RSS endpoints
MACRO_BLOG_FEEDS = {
    "voxeu": {
        "url": "https://cepr.org/voxeu/rss.xml",
        "name": "VoxEU / CEPR Policy Portal",
        "category": "academic_policy",
    },
    "apricitas": {
        "url": "https://www.apricitas.io/feed",
        "name": "Apricitas Economics",
        "category": "applied_macro",
    },
    "marginal_revolution": {
        "url": "https://feeds.feedburner.com/marginalrevolution",
        "name": "Marginal Revolution",
        "category": "general_economics",
    },
    "macro_compass": {
        "url": "https://themacrocompass.substack.com/feed",
        "name": "The Macro Compass",
        "category": "monetary_plumbing",
    },
    "nber_digest": {
        "url": "https://www.nber.org/rss/digest.xml",
        "name": "NBER Digest",
        "category": "academic_research",
    },
}


def iter_macro_blogs(
    source_key: str = "all",
    *,
    limit: int = 50,
) -> Iterator[tuple[datetime, str, str, dict]]:
    """Yield (date, text, url, metadata) for macroeconomic blog and policy posts.

    Parameters
    ----------
    source_key : str, default 'all'
        One of 'voxeu', 'apricitas', 'marginal_revolution', 'macro_compass', 'nber_digest', or 'all'.
    limit : int, default 50
        Maximum items to fetch per source.

    Yields
    ------
    (date, text, url, metadata)
    """
    if source_key == "all":
        keys = list(MACRO_BLOG_FEEDS.keys())
    elif source_key in MACRO_BLOG_FEEDS:
        keys = [source_key]
    else:
        raise ValueError(
            f"Unknown source_key {source_key!r}. Available: {sorted(MACRO_BLOG_FEEDS.keys())} or 'all'"
        )

    for k in keys:
        cfg = MACRO_BLOG_FEEDS[k]
        try:
            items = parse_rss_feed(cfg["url"], limit=limit)
            for item in items:
                date = item["date"]
                title = item.get("title", "")
                desc = item.get("description", "")
                text = f"{title}. {desc}".strip()
                url = item.get("link", "")
                metadata = {
                    "source": k,
                    "source_name": cfg["name"],
                    "category": cfg["category"],
                }
                yield (date, text, url, metadata)
        except Exception:
            # Resilient network fallback: continue to next source
            continue


__all__ = [
    "MACRO_BLOG_FEEDS",
    "iter_macro_blogs",
]
