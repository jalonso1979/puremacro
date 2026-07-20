"""European Commission DG ECFIN news feed.

DG ECFIN publishes news under https://economy-finance.ec.europa.eu/
with an Atom feed for press releases. Useful for NextGenerationEU,
recovery-and-resilience-plan, and SGP-clause announcements.
"""
from __future__ import annotations

from typing import Iterator

from ._rss import iter_atom


# DG ECFIN does not expose its own dedicated feed. The Commission-wide
# press corner is the canonical source; ECFIN releases come tagged with
# <category>POLICY_AREA=ECFIN</category> and we filter client-side.
_FEED = "https://ec.europa.eu/commission/presscorner/api/rss?language=en"


def iter_ecfin_press(
    *,
    feed_url: str | None = None,
    filter_ecfin: bool = True,
) -> Iterator[tuple]:
    """Yield Commission press items, optionally filtered to DG ECFIN.

    ``filter_ecfin=True`` (default) keeps only items whose XML carries
    a ``POLICY_AREA=ECFIN`` category tag. Set to False to receive the
    full Commission press corner (covers all DGs).
    ``language``: pass ``feed_url=`` with ``?language=de|fr|it|es|pt``
    etc. to switch the press-corner language.
    """
    from ._http import safe_get_bytes
    import xml.etree.ElementTree as ET
    import pandas as pd
    url = feed_url or _FEED
    try:
        body = safe_get_bytes(url)
    except Exception:
        return
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return
    for item in root.findall(".//item"):
        if filter_ecfin:
            cats = [c.text for c in item.findall("category") if c.text]
            if not any("ECFIN" in (c or "") for c in cats):
                continue
        title = item.findtext("title", default="")
        desc = item.findtext("description", default="")
        link = item.findtext("link", default="")
        pub = item.findtext("pubDate", default=None)
        try:
            date = pd.to_datetime(pub) if pub else pd.NaT
        except Exception:
            date = pd.NaT
        if pd.isna(date):
            continue
        yield date, f"{title} . {desc}", link


__all__ = ["iter_ecfin_press"]
