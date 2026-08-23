"""ALFRED (Archival FRED) — first-class vintage accessor.

ALFRED archives every edition of every series FRED publishes, foreign
series included, which makes it the widest single source of
international GDP vintages.

TWO ROUTES, AND WHY BOTH EXIST
------------------------------
1. **The API** (needs a free FRED key). One request returns the whole
   revision triangle::

       series/observations?series_id=X&output_type=2
           &realtime_start=1776-07-04&realtime_end=9999-12-31

   ``output_type=2`` is "observations by vintage date, all
   observations": one row per reference period, one column per
   edition. For German real GDP that is 142 periods x 61 editions in a
   single call.

2. **Keyless** (no key, but N requests). The graph CSV
   ``alfredgraph.csv?id=X`` serves exactly **one** vintage per
   request — the current one unless ``&vintage_date=`` names another,
   and a comma-separated list 404s. So the keyless route enumerates
   the available vintage dates from the ALFRED download page and then
   fetches one CSV per edition. Correct, cached, and much slower.

That single-vintage behaviour of the bare graph CSV is worth stating
loudly, because it is invisible: the response is a perfectly
well-formed CSV, so code that melts it produces a tidy
``[date, vintage, value]`` frame carrying exactly one edition, and a
revision study built on it silently finds no revisions at all.

WHAT THE VINTAGE DATE MEANS HERE
--------------------------------
Not a single thing — a documented three-step fallback. ALFRED's own
release-dates note (e.g.
``https://alfred.stlouisfed.org/release/downloaddates?rid=267``) states
it verbatim:

    "When available, the actual release date as provided by the source
    is entered into the ALFRED database. If a release date is not
    available from the source, the release date given by our data
    providers is used. If a release date cannot be determined either
    from the source or from our data providers, the date that the
    series was first available in the St. Louis Fed's FRED database is
    used."

So for a well-documented foreign release the vintage usually *is* the
source's release date; where the source supplies none it degrades to a
provider date and finally to FRED's own ingestion date. The rule is
applied per release, not per series, and the payload does not say which
of the three branches produced any given date.

Practical reading:

* Ordering editions (first, second, latest) — faithful either way.
* Measuring revision magnitudes — faithful, the values are as published.
* Dating a release *event* — check that release's download-dates page
  first, or use a provider whose vintage is unambiguously a national
  release date (Bundesbank, StatCan). See ``VINTAGE_SEMANTICS`` in
  :mod:`puremacro.fetch.realtime._base`.
"""
from __future__ import annotations

import io
import json
import re
import urllib.parse
import warnings

import pandas as pd

from ..._http import safe_get_bytes, safe_get_bytes_cached
from ...vintages import AlfredVintageStore
from ._base import VintagePanel, normalize_vintage_frame, register_provider


FRED_API_BASE = "https://api.stlouisfed.org/fred/"
ALFRED_GRAPH_URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={series_id}"
ALFRED_DOWNLOAD_URL = "https://alfred.stlouisfed.org/series/downloaddata?seid={series_id}"

# FRED's CSV WAF rate-limits Mozilla-like agents into silent timeouts;
# an honest agent string answers in a fraction of a second. Same
# reasoning as puremacro.fetch._classic._safe_urlopen.
_ALFRED_UA = (
    "puremacro (real-time vintage reader; "
    "+https://github.com/jalonso1979/puremacro)"
)

# Vintage suffixes appear as YYYY-MM-DD or YYYYMMDD depending on which
# ALFRED endpoint produced the payload; accept both.
_VINTAGE_SUFFIX = re.compile(r"_(\d{4}-\d{2}-\d{2}|\d{8})$")

#: Above this many editions the keyless route warns before issuing one
#: HTTP request per edition.
KEYLESS_REQUEST_WARN_THRESHOLD = 25


def _parse_vintage_token(token: str) -> pd.Timestamp:
    return pd.to_datetime(
        token, format="%Y-%m-%d" if "-" in token else "%Y%m%d", errors="coerce",
    )


def _melt_vintage_columns(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Melt a wide ``SERIES_<vintage>`` frame into long form.

    Columns whose header carries no parseable vintage suffix are
    skipped rather than guessed at.
    """
    dates = pd.to_datetime(df[date_col], errors="coerce")
    parts = []
    for col in df.columns:
        if col == date_col:
            continue
        m = _VINTAGE_SUFFIX.search(str(col))
        if m is None:
            continue
        vintage = _parse_vintage_token(m.group(1))
        if pd.isna(vintage):
            continue
        parts.append(pd.DataFrame({
            "date": dates,
            "vintage": vintage,
            "value": pd.to_numeric(df[col], errors="coerce"),
        }))
    if not parts:
        return pd.DataFrame(columns=["date", "vintage", "value"])
    out = pd.concat(parts, ignore_index=True).dropna(
        subset=["date", "vintage", "value"])
    return (out.sort_values(["date", "vintage"])
               .reset_index(drop=True)[["date", "vintage", "value"]])


# ---------------------------------------------------------------------------
# Pure parsers (no network — these are what the offline tests exercise)
# ---------------------------------------------------------------------------
def parse_alfredgraph_csv(raw: bytes | str) -> pd.DataFrame:
    """Parse an ALFRED graph CSV into long ``[date, vintage, value]``.

    A plain FRED CSV (no vintage suffix on the value column) parses to
    an empty frame rather than silently claiming one bogus edition.
    """
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not raw or not raw.strip():
        return pd.DataFrame(columns=["date", "vintage", "value"])
    df = pd.read_csv(io.BytesIO(raw))
    if df.empty or len(df.columns) < 2:
        return pd.DataFrame(columns=["date", "vintage", "value"])
    return _melt_vintage_columns(df, df.columns[0])


def parse_alfred_api_observations(payload: dict | str | bytes) -> pd.DataFrame:
    """Parse a FRED ``series/observations&output_type=2`` payload.

    Each observation is ``{date, realtime_start, realtime_end,
    SERIES_YYYYMMDD: value, ...}``; missing values arrive as ``"."``.
    """
    if isinstance(payload, (bytes, str)):
        payload = json.loads(payload)
    obs = payload.get("observations") or []
    if not obs:
        return pd.DataFrame(columns=["date", "vintage", "value"])
    df = pd.DataFrame(obs)
    df = df.drop(columns=[c for c in ("realtime_start", "realtime_end")
                          if c in df.columns])
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date", "vintage", "value"])
    return _melt_vintage_columns(df, "date")


def parse_alfred_vintage_dates_html(html: str) -> list[pd.Timestamp]:
    """Pull the available vintage dates out of ALFRED's download page.

    The page carries a ``form[selected_vintage_dates][]`` multi-select
    whose options are the archived edition dates.
    """
    dates = re.findall(r'<option[^>]*value="(\d{4}-\d{2}-\d{2})"', html or "")
    out = sorted({pd.Timestamp(d) for d in dates})
    return out


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
def _get(url: str, timeout: float, use_cache: bool) -> bytes:
    if use_cache:
        return safe_get_bytes_cached(url, timeout, user_agent=_ALFRED_UA)
    return safe_get_bytes(url, timeout, user_agent=_ALFRED_UA)


def _resolve_key(api_key: str | None) -> str | None:
    if api_key is not None:
        return api_key
    try:
        from ...credentials import get as _get_cred
        return _get_cred("fred")
    except Exception:                                    # pragma: no cover
        return None


def alfred_vintage_dates(
    series_id: str,
    *,
    api_key: str | None = None,
    timeout: float = 60.0,
    use_cache: bool = True,
) -> list[pd.Timestamp]:
    """Every edition date ALFRED holds for ``series_id``.

    Uses the API when a FRED key is available, and otherwise scrapes
    the public download page — both return the same list.
    """
    key = _resolve_key(api_key)
    if key:
        url = FRED_API_BASE + "series/vintagedates?" + urllib.parse.urlencode({
            "series_id": series_id, "api_key": key,
            "file_type": "json", "limit": 10000,
        })
        payload = json.loads(_get(url, timeout, use_cache))
        return sorted(pd.Timestamp(d) for d in payload.get("vintage_dates", []))
    html = _get(ALFRED_DOWNLOAD_URL.format(series_id=series_id),
                timeout, use_cache).decode("utf-8", errors="ignore")
    return parse_alfred_vintage_dates_html(html)


def _fetch_via_api(
    series_id: str, key: str, *,
    start_vintage: str | None, end_vintage: str | None,
    timeout: float, use_cache: bool,
) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "output_type": 2,           # all observations, one column per vintage
        "realtime_start": str(start_vintage)[:10] if start_vintage else "1776-07-04",
        "realtime_end": str(end_vintage)[:10] if end_vintage else "9999-12-31",
    }
    url = FRED_API_BASE + "series/observations?" + urllib.parse.urlencode(params)
    return parse_alfred_api_observations(_get(url, timeout, use_cache))


def _fetch_via_graph_csv(
    series_id: str, *,
    start_vintage: str | None, end_vintage: str | None,
    timeout: float, use_cache: bool,
) -> pd.DataFrame:
    """Keyless route: one request per edition."""
    dates = alfred_vintage_dates(series_id, api_key=None, timeout=timeout,
                                 use_cache=use_cache)
    if start_vintage is not None:
        dates = [d for d in dates if d >= pd.Timestamp(start_vintage)]
    if end_vintage is not None:
        dates = [d for d in dates if d <= pd.Timestamp(end_vintage)]
    if not dates:
        return pd.DataFrame(columns=["date", "vintage", "value"])
    if len(dates) > KEYLESS_REQUEST_WARN_THRESHOLD:
        warnings.warn(
            f"alfred_vintages({series_id!r}) is using the keyless route, "
            f"which needs one request per edition — {len(dates)} of them. "
            "A free FRED API key (https://fred.stlouisfed.org/docs/api/"
            "api_key.html, then `export FRED_API_KEY=...`) collapses this "
            "to a single request. Responses are cached, so this is a "
            "one-off cost per series.",
            UserWarning, stacklevel=3,
        )
    base = ALFRED_GRAPH_URL.format(series_id=series_id)
    frames, dropped = [], []
    for d in dates:
        url = f"{base}&vintage_date={d.strftime('%Y-%m-%d')}"
        try:
            part = parse_alfredgraph_csv(_get(url, timeout, use_cache))
        except Exception as exc:
            dropped.append((d, f"{type(exc).__name__}: {exc}"))
            continue
        if part.empty:
            dropped.append((d, "empty response"))
            continue
        frames.append(part)
    if dropped:
        # One request per edition means one chance per edition to fail.
        # Silently dropping them would hand back a gap-toothed archive
        # that looks complete -- and, worse, get it written to the
        # persistent store as if it were.
        warnings.warn(
            f"alfred_vintages({series_id!r}): {len(dropped)} of "
            f"{len(dates)} editions could not be retrieved "
            f"(first: {dropped[0][0].date()} -> {dropped[0][1]}). The "
            "result is incomplete and will not be cached.",
            UserWarning, stacklevel=3,
        )
    out = (pd.concat(frames, ignore_index=True)
             .drop_duplicates(subset=["date", "vintage"], keep="last")
             .sort_values(["date", "vintage"]).reset_index(drop=True)
           if frames else pd.DataFrame(columns=["date", "vintage", "value"]))
    out.attrs["n_editions_dropped"] = len(dropped)
    return out


def _to_wide(long: pd.DataFrame) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame()
    wide = long.pivot_table(index="date", columns="vintage", values="value")
    wide = wide.sort_index().reindex(columns=sorted(wide.columns))
    wide.columns.name = "vintage"
    return wide


def alfred_vintages(
    series_id: str,
    *,
    vintages: str | list = "all",
    output: str = "long",
    start_vintage: str | None = None,
    end_vintage: str | None = None,
    api_key: str | None = None,
    store: AlfredVintageStore | None = None,
    refresh: bool = False,
    use_cache: bool = True,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Every archived edition of an ALFRED series.

    Parameters
    ----------
    series_id : FRED/ALFRED series id, e.g. ``"CLVMNACSCAB1GQDE"``
        (real GDP, Germany) or ``"GDPC1"`` (real GDP, United States).
    vintages : ``"all"`` (default) for the whole archive, ``"first"``
        for the earliest edition of each period, ``"latest"`` for the
        most recent, or an explicit list of edition dates to keep.
    output : ``"long"`` gives ``[date, vintage, value]``; ``"wide"``
        gives the reference-period x edition revision triangle.
    start_vintage, end_vintage : inclusive bounds on the edition date.
    api_key : FRED key. Defaults to
        :func:`puremacro.credentials.get`\\ ``("fred")``; without one
        the slower keyless route is used automatically.
    store : an :class:`~puremacro.vintages.AlfredVintageStore` for
        persistent SQLite caching across kernel restarts.
    refresh : ignore any cached copy and refetch.
    use_cache : when False, bypass both the SQLite store and the HTTP
        disk cache.

    Returns
    -------
    pd.DataFrame — long or wide per ``output``.

    Notes
    -----
    The ``vintage`` column is ALFRED's release date, which is the
    source's own release date when the source supplies one and
    otherwise falls back to a provider date or to FRED's ingestion
    date. See the module docstring; the distinction matters whenever a
    vintage date is used as an event date rather than as an ordering
    key.
    """
    if output not in {"long", "wide"}:
        raise ValueError(f"output must be 'long' or 'wide', got {output!r}")
    if isinstance(vintages, str) and vintages not in {"all", "first", "latest"}:
        raise ValueError(
            "vintages must be 'all', 'first', 'latest', or a list of dates; "
            f"got {vintages!r}"
        )

    if store is None and use_cache:
        try:
            store = AlfredVintageStore()
        except Exception:                                # pragma: no cover
            store = None

    long: pd.DataFrame | None = None
    if store is not None and not refresh and store.has_series(series_id):
        cached = store.get(series_id)
        if not cached.empty:
            long = cached.rename(columns={
                "observation_date": "date", "vintage_date": "vintage",
            })[["date", "vintage", "value"]]

    if long is None:
        key = _resolve_key(api_key)
        if key:
            long = _fetch_via_api(
                series_id, key, start_vintage=start_vintage,
                end_vintage=end_vintage, timeout=timeout, use_cache=use_cache,
            )
        else:
            long = _fetch_via_graph_csv(
                series_id, start_vintage=start_vintage,
                end_vintage=end_vintage, timeout=timeout, use_cache=use_cache,
            )
        # Persist ONLY a complete archive. Both retrieval routes push the
        # vintage bounds down to the server, so a bounded fetch returns a
        # partial archive -- and the store is keyed by series_id alone,
        # with `has_series` as the hit test. Writing a bounded result, or
        # a keyless sweep that lost editions to failed requests, would
        # make every later unbounded call read that fragment back as if
        # it were the whole archive, silently and permanently.
        complete = (start_vintage is None and end_vintage is None
                    and not long.attrs.get("n_editions_dropped"))
        if store is not None and not long.empty and complete:
            store.put_many(
                long.assign(series_id=series_id).rename(columns={
                    "date": "observation_date", "vintage": "vintage_date",
                })[["series_id", "observation_date", "vintage_date", "value"]]
            )

    long = long.copy()
    if start_vintage is not None:
        long = long[long["vintage"] >= pd.Timestamp(start_vintage)]
    if end_vintage is not None:
        long = long[long["vintage"] <= pd.Timestamp(end_vintage)]

    if isinstance(vintages, str):
        if vintages == "first" and not long.empty:
            long = (long.sort_values(["date", "vintage"])
                        .groupby("date", as_index=False).first())
        elif vintages == "latest" and not long.empty:
            long = (long.sort_values(["date", "vintage"])
                        .groupby("date", as_index=False).last())
    else:
        keep = {pd.Timestamp(v) for v in vintages}
        long = long[long["vintage"].isin(keep)]

    long = long.sort_values(["date", "vintage"]).reset_index(drop=True)
    return _to_wide(long) if output == "wide" else long


def fetch_alfred_panel(
    countries,
    variables,
    *,
    catalog=None,
    store: AlfredVintageStore | None = None,
    refresh: bool = False,
    use_cache: bool = True,
    timeout: float = 60.0,
    api_key: str | None = None,
    **_ignored,
) -> VintagePanel:
    """Registry entry point: assemble an ALFRED-backed VintagePanel."""
    from .catalog import resolve_spec

    frames, resolved, failed = [], {}, {}
    for country in countries:
        for variable in variables:
            spec = resolve_spec("alfred", country, variable, catalog=catalog)
            if spec is None:
                continue
            sid = spec.series_id
            resolved[f"{country}:{variable}"] = sid
            try:
                long = alfred_vintages(
                    sid, store=store, refresh=refresh, use_cache=use_cache,
                    timeout=timeout, api_key=api_key,
                )
            except Exception as exc:
                failed[f"{country}:{variable}"] = f"{type(exc).__name__}: {exc}"
                continue
            frames.append(normalize_vintage_frame(
                long, country=country, variable=variable,
                provider="alfred", series_id=sid, units=spec.units,
            ))
    df = (pd.concat(frames, ignore_index=True) if frames
          else pd.DataFrame(columns=["country", "variable", "date", "vintage",
                                     "value", "provider", "series_id",
                                     "units"]))
    return VintagePanel(df=df, metadata={
        "provider": "alfred", "series_ids": resolved, "failed": failed,
    })


def _register() -> None:
    from .catalog import provider_country_codes
    register_provider("alfred", fetch_alfred_panel,
                      provider_country_codes("alfred"))


__all__ = [
    "FRED_API_BASE",
    "ALFRED_GRAPH_URL",
    "ALFRED_DOWNLOAD_URL",
    "KEYLESS_REQUEST_WARN_THRESHOLD",
    "alfred_vintages",
    "alfred_vintage_dates",
    "parse_alfredgraph_csv",
    "parse_alfred_api_observations",
    "parse_alfred_vintage_dates_html",
    "fetch_alfred_panel",
]
