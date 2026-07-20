"""Shared Wayback Machine CDX-API helper for narrative connectors that
need to route around WAF-blocked endpoints (EUR-Lex, EU Parliament, CBO).

Consolidated 2026-05-25 (puremacro 0.61.0) from duplicated inline
copies in :mod:`puremacro.narrative.sources.eu_eurlex` and
:mod:`puremacro.narrative.sources.eu_parliament`. The CBO connector
became the third consumer at the same time.
"""
from __future__ import annotations

import json
import urllib.parse as _up

from ._http import safe_get_text


_CDX_URL = (
    "https://web.archive.org/cdx/search/cdx?url={target}"
    "&filter=statuscode:200&output=json&limit=1"
)
_WAYBACK_URL = "https://web.archive.org/web/{timestamp}/{target}"


def wayback_snapshot_url(target: str, *,
                          user_agent: str | None = None) -> str | None:
    """Return the most-recent 200-OK Wayback snapshot URL for ``target``,
    or None if CDX has no such snapshot.

    Parameters
    ----------
    target : the original URL we want an archived snapshot of.
    user_agent : optional override for the CDX query — some clients
        prefer a browser-style chrome string.

    Returns
    -------
    A ``https://web.archive.org/web/<timestamp>/<target>`` URL, or
    ``None`` on any failure (network error, empty CDX body, malformed
    JSON, no rows).
    """
    cdx_url = _CDX_URL.format(target=_up.quote(target, safe=""))
    try:
        text = safe_get_text(cdx_url, user_agent=user_agent)
    except Exception:
        return None
    if not text.strip():
        return None
    try:
        records = json.loads(text)
    except Exception:
        return None
    if not isinstance(records, list) or len(records) < 2:
        return None
    data_row = records[1]
    if len(data_row) < 2:
        return None
    timestamp = data_row[1]
    return _WAYBACK_URL.format(timestamp=timestamp, target=target)


__all__ = ["wayback_snapshot_url"]
