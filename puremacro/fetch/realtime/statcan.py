"""Statistics Canada — genuine vintage tables (CODR "real-time" tables).

StatCan is unusual: it publishes previously-released values as a
first-class product rather than leaving you to reconstruct them. Table
**36-10-0431**, "Vintages of releases of gross domestic product,
expenditure-based", carries one full history per release date.

HOW THE VINTAGE IS ENCODED
--------------------------
Not as a timestamp on the observation. The vintage is an ordinary cube
**dimension** called ``Release`` whose members are Statistics Canada
*Daily* release dates ("May 29, 2026"). Every
(GEO x Prices x Seasonal adjustment x Estimates x Release) cell is its
own CODR series with its own vector id, holding the aggregate exactly
as published on that date. Fixing the Release coordinate gives one
vintage; sweeping it gives the triangle.

THE TRAP
--------
The WDS JSON carries a ``releaseTime`` field on every datapoint. **It
is not the vintage date.** It is the cube's last-refresh timestamp,
and it is identical across vintages — the 2015 vintage vector and the
2026 vintage vector both report the same ``releaseTime``. A connector
that read it as the vintage would produce a panel in which every
edition claims the same publication date, collapsing the triangle
without raising anything. The vintage lives only in the ``Release``
dimension member.

LEVELS ARE NOT COMPARABLE ACROSS VINTAGES
-----------------------------------------
Every Release member is labelled "Chained (2017) dollars", including
vintages published when the base year was 2007 or 2012. The values are
as-published, so the label lags the data. Level differences across
distant vintages therefore mix genuine revisions with rebasing, which
is one more reason the revision tests here default to growth rates.
"""
from __future__ import annotations

import io
import json
import zipfile

import pandas as pd

from ..._http import safe_get_bytes, safe_get_bytes_cached
from ._base import VintagePanel, normalize_vintage_frame, register_provider
from .catalog import SeriesSpec, register_catalog


_UA = "puremacro (real-time vintage reader)"

WDS_FULL_TABLE_URL = (
    "https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/{pid}/en"
)

#: Canonical variable -> (product id, ``Estimates`` member, ``Prices`` member).
STATCAN_VINTAGE_TABLES: dict[str, tuple[str, str, str]] = {
    "gdp_real": ("36100431", "Gross domestic product at market prices",
                 "Chained (2017) dollars"),
    "gdp_nom": ("36100431", "Gross domestic product at market prices",
                "Current prices"),
    "con_real": ("36100431", "Household final consumption expenditure",
                 "Chained (2017) dollars"),
    "gfcf_real": ("36100431", "Gross fixed capital formation",
                  "Chained (2017) dollars"),
    "exports_real": ("36100431", "Exports of goods and services",
                     "Chained (2017) dollars"),
    "imports_real": ("36100431", "Less: imports of goods and services",
                     "Chained (2017) dollars"),
}

_SEASONAL = "Seasonally adjusted at annual rates"

_USECOLS = ["REF_DATE", "GEO", "Prices", "Seasonal adjustment", "Estimates",
            "Release", "VALUE"]


def parse_statcan_vintage_csv(
    raw: bytes | str,
    *,
    estimate: str,
    prices: str,
    seasonal: str = _SEASONAL,
    geo: str = "Canada",
) -> pd.DataFrame:
    """Parse a StatCan real-time table CSV into ``[date, vintage, value]``.

    Pure function — takes the CSV bytes, does no I/O. ``REF_DATE`` is
    ``YYYY-MM`` where the month is the quarter's first month, so
    ``"2025-01"`` is 2025Q1; parsing it as monthly would silently
    produce a series with three-month gaps.

    Cells that were not yet published in a given vintage arrive with an
    empty ``VALUE`` and ``STATUS`` of ``".."``. They are dropped — that
    absence is what marks each vintage's truncation edge, not an error.
    """
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not raw or not raw.strip():
        return pd.DataFrame(columns=["date", "vintage", "value"])

    df = pd.read_csv(
        io.BytesIO(raw), usecols=lambda c: c.strip('﻿"') in _USECOLS,
        dtype=str, encoding="utf-8-sig",
    )
    df.columns = [c.strip('﻿"') for c in df.columns]
    missing = set(_USECOLS) - set(df.columns)
    if missing:
        raise ValueError(
            f"StatCan CSV missing expected columns {sorted(missing)}; got "
            f"{sorted(df.columns)}"
        )

    sub = df[
        (df["GEO"] == geo)
        & (df["Prices"] == prices)
        & (df["Seasonal adjustment"] == seasonal)
        & (df["Estimates"] == estimate)
    ]
    if sub.empty:
        return pd.DataFrame(columns=["date", "vintage", "value"])

    out = pd.DataFrame({
        "date": pd.to_datetime(sub["REF_DATE"] + "-01", errors="coerce"),
        # "May 29, 2026" / "March 01, 2013" — zero-padded day.
        "vintage": pd.to_datetime(sub["Release"], format="%B %d, %Y",
                                  errors="coerce"),
        "value": pd.to_numeric(sub["VALUE"], errors="coerce"),
    }).dropna(subset=["date", "vintage", "value"])
    return (out.sort_values(["date", "vintage"])
               .reset_index(drop=True)[["date", "vintage", "value"]])


def _download_table_csv(pid: str, *, timeout: float, use_cache: bool) -> bytes:
    """Resolve the JSON envelope, fetch the zip, return the data CSV bytes."""
    getter = safe_get_bytes_cached if use_cache else safe_get_bytes
    env = json.loads(getter(
        WDS_FULL_TABLE_URL.format(pid=pid), timeout, user_agent=_UA))
    if env.get("status") != "SUCCESS" or not env.get("object"):
        raise RuntimeError(
            f"StatCan getFullTableDownloadCSV({pid}) returned {env!r}"
        )
    zip_bytes = getter(env["object"], max(timeout, 180.0), user_agent=_UA)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = [n for n in z.namelist()
                 if n.lower().endswith(".csv")
                 and "metadata" not in n.lower()]
        if not names:
            raise RuntimeError(
                f"StatCan zip for {pid} has no data CSV; members={z.namelist()}"
            )
        return z.read(names[0])


def fetch_statcan_vintages(
    variable: str = "gdp_real",
    *,
    timeout: float = 180.0,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Long ``[date, vintage, value]`` for one Canadian variable."""
    if variable not in STATCAN_VINTAGE_TABLES:
        raise ValueError(
            f"no StatCan vintage table mapped for {variable!r}; available: "
            f"{sorted(STATCAN_VINTAGE_TABLES)}"
        )
    pid, estimate, prices = STATCAN_VINTAGE_TABLES[variable]
    csv_bytes = _download_table_csv(pid, timeout=timeout, use_cache=use_cache)
    return parse_statcan_vintage_csv(csv_bytes, estimate=estimate,
                                     prices=prices)


def fetch_statcan_panel(
    countries, variables, *, timeout: float = 180.0, use_cache: bool = True,
    **_ignored,
) -> VintagePanel:
    """Registry entry point."""
    frames, failed = [], {}
    for country in countries:
        if str(country).upper() != "CAN":
            continue
        for variable in variables:
            if variable not in STATCAN_VINTAGE_TABLES:
                continue
            pid, estimate, _prices = STATCAN_VINTAGE_TABLES[variable]
            try:
                long = fetch_statcan_vintages(
                    variable, timeout=timeout, use_cache=use_cache)
            except Exception as exc:
                failed[f"{country}:{variable}"] = f"{type(exc).__name__}: {exc}"
                continue
            frames.append(normalize_vintage_frame(
                long, country="CAN", variable=variable, provider="statcan",
                series_id=f"{pid}:{estimate}", units="level",
            ))
    df = (pd.concat(frames, ignore_index=True) if frames
          else pd.DataFrame(columns=[
              "country", "variable", "date", "vintage", "value", "provider",
              "series_id", "units"]))
    return VintagePanel(df=df, metadata={"provider": "statcan",
                                         "failed": failed})


def _register() -> None:
    register_catalog("statcan", {
        "CAN": {
            var: SeriesSpec(f"{pid}:{est}", "level", "statcan",
                            "CODR real-time table; vintages from 2012-11-30")
            for var, (pid, est, _p) in STATCAN_VINTAGE_TABLES.items()
        }
    })
    register_provider("statcan", fetch_statcan_panel, ["CAN"])


__all__ = [
    "WDS_FULL_TABLE_URL",
    "STATCAN_VINTAGE_TABLES",
    "parse_statcan_vintage_csv",
    "fetch_statcan_vintages",
    "fetch_statcan_panel",
]
