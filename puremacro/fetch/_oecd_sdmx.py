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
        except (ValueError, ArithmeticError, Exception):
            return pd.DataFrame()
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# urllib transport (4.7): the route every new cross-country panel builder
# takes. No ``requests``, no write into the repository: the on-disk cache
# is the SQLite one under ``$PUREMACRO_HTTP_CACHE_DIR`` / ``~/.cache/puremacro``.
# ---------------------------------------------------------------------------

#: Reference areas the OECD publishes alongside countries in its national
#: accounts, labour and short-term flows. A panel builder drops them before
#: anything is counted as a country. (``DEU_F`` is pre-unification West
#: Germany, published as a separate area.)
OECD_AGGREGATES: frozenset[str] = frozenset({
    "EA", "EA12", "EA19", "EA20", "EU", "EU15", "EU27_2020", "EU28", "EEA",
    "G7", "G20", "NAFTA", "USMCA", "OECD", "OECD26", "OECDE", "OECDXEA",
    "W", "WLD", "DEU_F", "A5M", "G4E", "SDR",
})

_AVAIL_BASE = "https://sdmx.oecd.org/public/rest/availableconstraint"
_AVAIL_ACCEPT = "application/vnd.sdmx.structure+json;version=1.0"


def oecd_key(dimensions, **pins) -> str:
    """Build an SDMX key from a DSD's dimension list and a few pinned values.

    ``oecd_key(["FREQ", "REF_AREA", "SECTOR"], FREQ="A", SECTOR="S1+S13")``
    gives ``"A..S1+S13"``: every dimension not pinned is left open. The
    dot count is then right by construction, which matters because the
    OECD answers a key with the wrong number of dimensions with HTTP 403
    ("expecting 12 got 11") rather than with an empty frame. A pinned value
    may be a string (``"V+Y"``) or any iterable of codes.
    """
    unknown = sorted(set(pins) - set(dimensions))
    if unknown:
        raise ValueError(f"pinned dimensions not in the DSD: {unknown}; "
                         f"dimensions are {list(dimensions)}")
    parts = []
    for dim in dimensions:
        value = pins.get(dim, "")
        if value is None:
            value = ""
        elif not isinstance(value, str):
            value = "+".join(str(v) for v in value)
        parts.append(value)
    return ".".join(parts)


def _empty_with_status(url: str, status: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out.attrs.update(url=url, status=status)
    return out


def oecd_csv(agency_flow: str, key: str, *, start_period: str | None = None,
             end_period: str | None = None, first_n: int | None = None,
             last_n: int | None = None, labels: bool = False,
             refresh: bool = False, pause: float = 5.0, timeout: float = 180.0,
             retries: int = 3, retry_sleep: float = 90.0) -> pd.DataFrame:
    """One OECD SDMX data request as a DataFrame, through urllib, paced and cached.

    ``agency_flow`` is ``"OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE1,2.0"`` (agency,
    DSD@flow, version; a blank version also works, ``latest`` does not).
    ``key`` is the dotted SDMX key (see :func:`oecd_key`). The response is
    ``format=csvfile`` (codes only, 3-4x smaller than the labelled variant)
    unless ``labels=True``.

    Behaviour the panel builders rely on:

    - Requests to ``sdmx.oecd.org`` are spaced at least ``pause`` seconds
      apart across the whole process (the endpoint throttles by response
      size, roughly twenty data requests a minute), and the transport asks
      for gzip.
    - An HTTP 429 sleeps ``retry_sleep`` seconds and tries again, up to
      ``retries`` attempts; a transport error retries after a short pause.
    - It never raises on a provider failure: the result is an EMPTY frame
      whose ``attrs["status"]`` says why (``"HTTP 404"`` is the OECD's way of
      saying "nothing for that key"; ``"HTTP 429"`` after the retries means
      "throttled", which a caller must not confuse with "publishes nothing").
      A successful frame carries ``attrs["status"] == "ok"`` and ``attrs["url"]``.
    - Successful responses are cached on disk without expiry; ``refresh=True``
      re-fetches and replaces the cached copy.
    """
    import http.client
    import time
    import urllib.error

    from .._http import safe_get_bytes_cached

    params = [f"format={'csvfilewithlabels' if labels else 'csvfile'}"]
    if start_period is not None:
        params.append(f"startPeriod={start_period}")
    if end_period is not None:
        params.append(f"endPeriod={end_period}")
    if first_n is not None:
        params.append(f"firstNObservations={int(first_n)}")
    if last_n is not None:
        params.append(f"lastNObservations={int(last_n)}")
    url = f"{_BASE}/{agency_flow}/{key}?{'&'.join(params)}"
    headers = {"Accept-Encoding": "gzip"}
    status = "no attempt"
    for attempt in range(1, retries + 1):
        try:
            body = safe_get_bytes_cached(url, timeout, headers=headers,
                                         rate_limit_seconds=pause, refresh=refresh)
        except urllib.error.HTTPError as exc:
            status = f"HTTP {exc.code}"
            if exc.code == 429 and attempt < retries:
                time.sleep(retry_sleep)
                continue
            return _empty_with_status(url, status)
        except (OSError, ValueError, EOFError, http.client.HTTPException) as exc:
            # URLError, timeout, bad SSL, a truncated body (IncompleteRead) or a
            # truncated gzip: all transient transport failures, retried alike.
            status = type(exc).__name__
            if attempt < retries:
                time.sleep(min(retry_sleep, 30.0))
                continue
            return _empty_with_status(url, status)
        if not body or len(body) < 40:
            return _empty_with_status(url, "empty response")
        try:
            frame = pd.read_csv(io.BytesIO(body), low_memory=False)
        except (ValueError, pd.errors.ParserError) as exc:
            return _empty_with_status(url, f"parse error: {exc}")
        frame.attrs.update(url=url, status="ok")
        return frame
    return _empty_with_status(url, status)


def oecd_availability(agency_flow: str, key: str = "all", *, mode: str = "exact",
                      end_period: str | None = None, refresh: bool = False,
                      timeout: float = 60.0, pause: float = 2.0) -> dict:
    """The ``availableconstraint`` answer for a flow and key, as the raw JSON dict.

    The cheap way to learn which reference areas (and which codes of any
    other dimension) carry data for a key before asking for the data itself.
    Returns ``{}`` when the endpoint fails; it is not throttled like the
    data endpoint.
    """
    import urllib.error

    from .._http import safe_get_bytes_cached

    params = [f"mode={mode}", "format=jsondata"]
    if end_period is not None:
        params.append(f"endPeriod={end_period}")
    url = f"{_AVAIL_BASE}/{agency_flow}/{key}?{'&'.join(params)}"
    try:
        body = safe_get_bytes_cached(url, timeout, headers={"Accept": _AVAIL_ACCEPT,
                                                            "Accept-Encoding": "gzip"},
                                     rate_limit_seconds=pause, refresh=refresh)
    except (urllib.error.HTTPError, OSError, ValueError, EOFError, Exception):
        return {}
    try:
        import json
        return json.loads(body.decode("utf-8"))
    except ValueError:
        return {}


__all__ = ["OECD_AGGREGATES", "get_sdmx_csv", "oecd_availability", "oecd_csv", "oecd_key"]
