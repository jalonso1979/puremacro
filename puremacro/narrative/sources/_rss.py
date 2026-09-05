"""Common RSS / Atom helper used by country-specific source connectors.

Stdlib-only (``urllib`` + ``xml.etree.ElementTree``). Returns a list of
``(date, title+description, link)`` tuples.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Iterator

import pandas as pd

from ._http import safe_get_bytes


_RSS10_NS = "http://purl.org/rss/1.0/"
_DC_NS = "http://purl.org/dc/elements/1.1/"


def iter_rss(url: str) -> Iterator[tuple]:
    """Yield (pubdate, title+description, link) records from an RSS feed.

    Handles RSS 2.0 (unqualified ``<item>``) and RSS 1.0 / RDF
    (``<rdf:RDF>`` with namespaced ``{http://purl.org/rss/1.0/}item``,
    Dublin Core ``<dc:date>``).
    """
    try:
        body = safe_get_bytes(url)
    except Exception:
        return
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return
    items = root.findall(".//item")
    if items:
        for item in items:
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
        return
    for item in root.findall(f".//{{{_RSS10_NS}}}item"):
        title = item.findtext(f"{{{_RSS10_NS}}}title", default="")
        desc = item.findtext(f"{{{_RSS10_NS}}}description", default="")
        link = item.findtext(f"{{{_RSS10_NS}}}link", default="")
        date_str = (
            item.findtext(f"{{{_DC_NS}}}date", default="")
            or item.findtext("pubDate", default="")
        )
        try:
            date = pd.to_datetime(date_str) if date_str else pd.NaT
        except Exception:
            date = pd.NaT
        if pd.isna(date):
            continue
        yield date, f"{title} . {desc}", link


def iter_atom(url: str) -> Iterator[tuple]:
    """Yield (updated, title+summary, link) records from an Atom feed."""
    try:
        body = safe_get_bytes(url)
    except Exception:
        return
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns) or root.findall(".//entry")
    for entry in entries:
        title = entry.findtext("atom:title", default="", namespaces=ns) or \
                 entry.findtext("title", default="")
        summary = (entry.findtext("atom:summary", default="", namespaces=ns)
                    or entry.findtext("summary", default="")
                    or entry.findtext("atom:content", default="", namespaces=ns)
                    or "")
        # link can be a sub-element with href attribute or text.
        link_el = entry.find("atom:link", ns) or entry.find("link")
        link = ""
        if link_el is not None:
            link = link_el.get("href") or (link_el.text or "")
        updated = (entry.findtext("atom:updated", default="", namespaces=ns)
                    or entry.findtext("updated", default="")
                    or entry.findtext("atom:published", default="", namespaces=ns)
                    or entry.findtext("published", default=""))
        try:
            date = pd.to_datetime(updated) if updated else pd.NaT
        except Exception:
            date = pd.NaT
        if pd.isna(date):
            continue
        yield date, f"{title} . {summary}", link


def parse_rss_feed(url: str, limit: int = 50) -> list[dict[str, Any]]:
    """Parse an RSS or Atom feed into a list of item dictionaries."""
    results: list[dict[str, Any]] = []
    for date, text, link in iter_rss(url):
        results.append({
            "date": date,
            "title": text,
            "description": "",
            "link": link,
        })
        if len(results) >= limit:
            break
    if not results:
        for date, text, link in iter_atom(url):
            results.append({
                "date": date,
                "title": text,
                "description": "",
                "link": link,
            })
            if len(results) >= limit:
                break
    return results


__all__ = ["iter_rss", "iter_atom", "parse_rss_feed"]
