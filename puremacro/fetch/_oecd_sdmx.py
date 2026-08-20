"""Shared OECD SDMX helper with on-disk caching + retry-with-backoff.

OECD's public SDMX endpoint rate-limits aggressively (~20 reqs/hour/IP).
Wrapping every CSV-with-labels call in this helper means:

1. First successful response is persisted on disk via ``cached_get``.
2. Subsequent calls return the cached bytes — bulletproof against future
   rate limits and survives ``build_all`` re-runs.
3. On 429 / connection error, retry once after 60s (handles a single
   transient blip without manual intervention).

Returns an empty DataFrame on persistent failure so callers can treat
"no data" and "rate-limited" symmetrically.
"""
from __future__ import annotations

import io
import time

import pandas as pd

_BASE = "https://sdmx.oecd.org/public/rest/data"
_FMT = "csvfilewithlabels"
_TIMEOUT = 180


def get_sdmx_csv(agency_flow: str, key: str, start_period: str,
                 *, refresh: bool = False, retry_sleep: int = 60) -> pd.DataFrame:
    """Fetch an OECD SDMX endpoint as a DataFrame, cached on disk.

    Parameters
    ----------
    agency_flow : e.g. ``"OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,"``.
    key : SDMX key string (dim values separated by dots).
    start_period : e.g. ``"1995"``.
    refresh : if True, bypass the cache and re-fetch.
    retry_sleep : seconds to wait between the first try and the single retry.

    Returns
    -------
    DataFrame parsed from the CSV-with-labels response. Empty DataFrame on
    HTTP error / 429 / parse error / empty response (after one retry), and
    also when there is no HTTP stack to fetch with at all.

    Notes
    -----
    ``requests`` is imported here rather than at module scope, and its absence
    is treated as "the download failed" rather than allowed to propagate. On a
    tablet build (Juno) or under Pyodide the scraper stack may not be installed
    at all, and every caller in this package documents an empty frame — not an
    exception — as the failure mode. Letting ``ImportError`` out of a fetch
    takes down callers that have a perfectly good frozen snapshot to fall back
    to, which is exactly the situation the course notebooks are in.
    """
    try:
        import requests

        from ._http import cached_get
    except ImportError:
        return pd.DataFrame()

    url = f"{_BASE}/{agency_flow}/{key}?startPeriod={start_period}&format={_FMT}"
    for attempt in (1, 2):
        try:
            content = cached_get(url, refresh=refresh, timeout=_TIMEOUT)
            if not content or len(content) < 200:
                return pd.DataFrame()
            return pd.read_csv(io.BytesIO(content), low_memory=False)
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 429 and attempt == 1:
                time.sleep(retry_sleep)
                continue
            return pd.DataFrame()
        except requests.RequestException:
            if attempt == 1:
                time.sleep(retry_sleep)
                continue
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


__all__ = ["get_sdmx_csv"]
