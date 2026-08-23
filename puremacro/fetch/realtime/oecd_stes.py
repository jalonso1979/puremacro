"""OECD Short-Term Economic Statistics — original release and revisions.

The widest keyless cross-country vintage archive there is: **42
economies**, monthly editions from February 1999, quarterly national
accounts plus the expenditure components, employment and the current
account.

FINDING THE DATAFLOW
--------------------
It is easy to conclude this archive did not survive the move to the
OECD Data Explorer, because the obvious identifier does not resolve.
``OECD.SDD.STES,DSD_STES@DF_VINTAGES`` returns nothing for any country.
The archive is alive under a different id::

    OECD.SDD.STES,DSD_STES_REVISIONS@DF_STES_REVISIONS,4.0

KEY STRUCTURE
-------------
Six dimensions, so a key carries five dots::

    REF_AREA . FREQ . MEASURE . UNIT_MEASURE . ACTIVITY . EDITION
    MEX.Q.B1GQ_Q...                 -> every edition
    MEX.Q.B1GQ_Q...202608           -> the August 2026 edition

Dot-counting is unforgiving: a key with the wrong number of parts
returns HTTP 422 with an empty body.

WHAT THE EDITION MEANS
----------------------
``EDITION`` is ``YYYYMM`` — the month of the **OECD's snapshot**, not
the national statistical office's release date. The OECD ingests with a
lag, so an edition can still carry the previous month's published
figure. Fine for ordering editions and measuring revisions; not a
release date. For true national release dates use the Bundesbank (DE),
ONS (UK) or StatCan (CA) providers.

REBASING, AND WHY GROWTH RATES SAVE YOU
---------------------------------------
Levels are not comparable across distant editions: each rebasing moves
the whole column, and for Mexico editions 200807-200911 sit on a
different unit scale entirely — roughly 4.5x. This is exactly why the
revision tests here transform **within** each vintage column. A
rescaling of a whole edition, by any constant, cancels out of that
edition's growth rates: ``log(k x_t) - log(k x_{t-1})`` does not depend
on ``k``. Compare levels across editions and the same rebasing shows up
as a spectacular fake revision.
"""
from __future__ import annotations

import io

import pandas as pd

from ._base import (
    VintagePanel,
    fetch_with_backoff,
    normalize_vintage_frame,
    register_provider,
)
from .catalog import SeriesSpec, register_catalog


DATAFLOW = "OECD.SDD.STES,DSD_STES_REVISIONS@DF_STES_REVISIONS,4.0"
OECD_STES_URL = (
    "https://sdmx.oecd.org/public/rest/data/{dataflow}/{key}"
    "?dimensionAtObservation=AllDimensions&format=csvfile"
)

_UA = "puremacro (real-time vintage reader)"

#: Canonical variable -> STES ``MEASURE`` code, with the units the
#: measure is published in.
OECD_STES_MEASURES: dict[str, tuple[str, str]] = {
    "gdp_real": ("B1GQ_Q", "level"),
    "gdp_nom": ("B1GQ_V", "level"),
    "deflator": ("B1GQ_D", "index"),
    "con_real": ("P3_S1M_Q", "level"),
    "govcon_real": ("P3_S13_Q", "level"),
    "gfcf_real": ("P51G_Q", "level"),
    "exports_real": ("P6_Q", "level"),
    "imports_real": ("P7_Q", "level"),
    "employment": ("EMP", "level"),
}

#: The 42 economies this archive covers, verified against the dataflow.
OECD_STES_COUNTRIES: tuple[str, ...] = (
    "AUS", "AUT", "BEL", "BRA", "CAN", "CHE", "CHL", "COL", "CRI", "CZE",
    "DEU", "DNK", "EA20", "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HUN",
    "IDN", "IRL", "ISL", "ISR", "ITA", "JPN", "KOR", "LTU", "LUX", "LVA",
    "MEX", "NLD", "NOR", "NZL", "POL", "PRT", "SVK", "SVN", "SWE", "TUR",
    "USA", "ZAF",
)


def build_key(country: str, measure: str, *, freq: str = "Q",
              edition: str | None = None) -> str:
    """Assemble a six-part STES-revisions key.

    ``edition=None`` leaves the last slot empty, which asks for every
    edition.
    """
    return (f"{str(country).upper()}.{freq}.{measure}..."
            f"{edition or ''}")


def parse_stes_revisions_csv(
    raw: bytes | str, *, keep_country: bool = False,
) -> pd.DataFrame:
    """Parse an SDMX-CSV STES-revisions response into ``[date, vintage, value]``.

    Pure function. ``EDITION`` is ``YYYYMM`` and becomes the first of
    that month; ``TIME_PERIOD`` is ``YYYY-Qn`` and becomes the quarter
    start. Rows come back in no particular order and are sorted here.

    ``keep_country=True`` adds a ``country`` column, which is what makes
    the batched multi-country request usable: one response covers
    several ``REF_AREA`` values and has to be split back apart.
    """
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not raw or not raw.strip():
        return pd.DataFrame(columns=["date", "vintage", "value"])
    df = pd.read_csv(io.BytesIO(raw))
    needed = {"EDITION", "TIME_PERIOD", "OBS_VALUE"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"OECD STES response missing {sorted(missing)}; got "
            f"{sorted(df.columns)}"
        )
    period = df["TIME_PERIOD"].astype(str)
    dates = pd.PeriodIndex(period, freq="Q").to_timestamp()
    cols = {
        "date": dates,
        "vintage": pd.to_datetime(df["EDITION"].astype(str),
                                  format="%Y%m", errors="coerce"),
        "value": pd.to_numeric(df["OBS_VALUE"], errors="coerce"),
    }
    subset = ["date", "vintage"]
    if keep_country:
        if "REF_AREA" not in df.columns:
            raise ValueError(
                "keep_country=True needs a REF_AREA column; got "
                f"{sorted(df.columns)}"
            )
        cols["country"] = df["REF_AREA"].astype(str)
        subset = ["country", "date", "vintage"]
    out = pd.DataFrame(cols).dropna(subset=["date", "vintage", "value"])
    ordered = subset + ["value"] if keep_country else ["date", "vintage", "value"]
    return (out.drop_duplicates(subset=subset, keep="last")
               .sort_values(subset).reset_index(drop=True)[ordered])


def fetch_oecd_stes_vintages(
    country: str,
    variable: str = "gdp_real",
    *,
    freq: str = "Q",
    start_period: str | None = None,
    timeout: float = 180.0,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Long ``[date, vintage, value]`` for one (country, variable)."""
    if variable not in OECD_STES_MEASURES:
        raise ValueError(
            f"no OECD STES measure mapped for {variable!r}; available: "
            f"{sorted(OECD_STES_MEASURES)}"
        )
    measure, _units = OECD_STES_MEASURES[variable]
    url = OECD_STES_URL.format(
        dataflow=DATAFLOW, key=build_key(country, measure, freq=freq))
    if start_period:
        url += f"&startPeriod={start_period}"
    return parse_stes_revisions_csv(fetch_with_backoff(
        url, timeout=timeout, use_cache=use_cache, user_agent=_UA))


#: Countries per request when batching. The archive returns roughly
#: 3 MB per country, and the endpoint rate-limits (HTTP 429) long
#: before it size-limits, so batching is what makes a 42-country sweep
#: feasible at all: 42 requests reliably trip the limiter, 7 do not.
BATCH_SIZE = 6

#: The endpoint rate-limits hard and recovers slowly, so the retry
#: schedule here is deliberately patient rather than snappy: a 429 on a
#: bulk archive is worth waiting out, because the alternative is a
#: panel that silently omits a country. Responses are disk-cached, so
#: the cost is paid once per series.
RETRY_ATTEMPTS = 6
RETRY_BASE_DELAY = 5.0
RATE_LIMIT_SECONDS = 2.0


def fetch_oecd_stes_panel(
    countries, variables, *, timeout: float = 180.0, use_cache: bool = True,
    batch_size: int = BATCH_SIZE, **_ignored,
) -> VintagePanel:
    """Registry entry point. Fetches countries in batched requests."""
    frames, failed = [], {}
    wanted = [str(c).upper() for c in countries
              if str(c).upper() in OECD_STES_COUNTRIES]
    for variable in variables:
        if variable not in OECD_STES_MEASURES:
            continue
        measure, units = OECD_STES_MEASURES[variable]
        for i in range(0, len(wanted), max(1, batch_size)):
            batch = wanted[i:i + max(1, batch_size)]
            key = build_key("+".join(batch), measure)
            url = OECD_STES_URL.format(dataflow=DATAFLOW, key=key)
            try:
                raw = fetch_with_backoff(
                    url, timeout=timeout, use_cache=use_cache,
                    user_agent=_UA, attempts=RETRY_ATTEMPTS,
                    base_delay=RETRY_BASE_DELAY,
                    rate_limit_seconds=RATE_LIMIT_SECONDS)
                long = parse_stes_revisions_csv(raw, keep_country=True)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                for c in batch:
                    failed[f"{c}:{variable}"] = reason
                continue
            for c in batch:
                part = long[long["country"] == c]
                if part.empty:
                    failed[f"{c}:{variable}"] = (
                        "no rows for this REF_AREA in the batched response")
                    continue
                frames.append(normalize_vintage_frame(
                    part, country=c, variable=variable, provider="oecd_stes",
                    series_id=build_key(c, measure), units=units,
                ))
    df = (pd.concat(frames, ignore_index=True) if frames
          else pd.DataFrame(columns=[
              "country", "variable", "date", "vintage", "value", "provider",
              "series_id", "units"]))
    return VintagePanel(df=df, metadata={"provider": "oecd_stes",
                                         "failed": failed})


def _register() -> None:
    register_catalog("oecd_stes", {
        c: {var: SeriesSpec(build_key(c, m), units, "oecd_stes",
                            "STES revisions archive; monthly editions "
                            "from 1999-02")
            for var, (m, units) in OECD_STES_MEASURES.items()}
        for c in OECD_STES_COUNTRIES
    })
    register_provider("oecd_stes", fetch_oecd_stes_panel, OECD_STES_COUNTRIES)


__all__ = [
    "DATAFLOW",
    "OECD_STES_URL",
    "OECD_STES_MEASURES",
    "OECD_STES_COUNTRIES",
    "build_key",
    "parse_stes_revisions_csv",
    "fetch_oecd_stes_vintages",
    "fetch_oecd_stes_panel",
]
