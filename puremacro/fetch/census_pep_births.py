"""Census Bureau Population Estimates Program — county and state births.

Why this exists
---------------
Census PEP publishes per-county BIRTHS columns in its "alldata" vintage
files at https://www2.census.gov/programs-surveys/popest/datasets/. These
estimates are derived from NCHS natality data but are released **without**
the post-1989 "100K+ counties only" suppression that affects the public
NCHS microdata, because Census uses births as an input to population
estimates rather than a primary vital-statistics product. So Census PEP
is the cleanest public source of county-year births for **2000-2025**.

Three vintage files cover non-overlapping windows:
    2000-2009: co-est2009-alldata.csv  (BIRTHS2000..BIRTHS2009)
    2010-2019: co-est2019-alldata.csv  (BIRTHS2010..BIRTHS2019)
    2020-2025: co-est2025-alldata.csv  (BIRTHS2020..BIRTHS2025)

Within each file, SUMLEV='040' rows are state totals (COUNTY='000');
SUMLEV='050' rows are county totals.

Public API:
    fetch_county_year() -> pd.DataFrame[county_fips, year, births]
    fetch_state_year()  -> pd.DataFrame[state_fips, year, births]
"""
from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pandas as pd

from ._http import cached_get

logger = logging.getLogger(__name__)


_VINTAGES = (
    ("2000-2009", "co-est2009-alldata.csv", range(2000, 2010)),
    ("2010-2019", "co-est2019-alldata.csv", range(2010, 2020)),
    ("2020-2025", "co-est2025-alldata.csv", range(2020, 2026)),
)
_BASE = "https://www2.census.gov/programs-surveys/popest/datasets/{vintage}/counties/totals/{filename}"


def _read_vintage(vintage: str, filename: str, *, refresh: bool = False) -> pd.DataFrame:
    url = _BASE.format(vintage=vintage, filename=filename)
    blob = cached_get(url, refresh=refresh)
    df = pd.read_csv(
        StringIO(blob.decode("latin-1")),
        dtype={"STATE": str, "COUNTY": str, "SUMLEV": str},
    )
    df["STATE"] = df["STATE"].str.zfill(2)
    df["COUNTY"] = df["COUNTY"].str.zfill(3)
    return df


def _melt_births(df: pd.DataFrame, years: range) -> pd.DataFrame:
    """Reshape BIRTHS[YYYY] wide columns to long-form."""
    birth_cols = [f"BIRTHS{y}" for y in years if f"BIRTHS{y}" in df.columns]
    keep = ["SUMLEV", "STATE", "COUNTY"] + birth_cols
    long = df[keep].melt(
        id_vars=["SUMLEV", "STATE", "COUNTY"],
        value_vars=birth_cols,
        var_name="year_str",
        value_name="births",
    )
    long["year"] = long["year_str"].str.removeprefix("BIRTHS").astype(int)
    long["births"] = pd.to_numeric(long["births"], errors="coerce")
    return long.drop(columns=["year_str"])


# Minimum national total for a calendar-year row to be considered complete.
# Real US total is always > 3M; vintage-base rows sum to ~25% of a year.
_PARTIAL_YEAR_THRESHOLD = 2_500_000


def _is_partial_year(year: int, total_us_births: float) -> bool:
    """Census PEP's first vintage year (e.g. BIRTHS2010 in the 2010-19
    vintage) covers only April 1 - June 30 of that year, so its total is
    ~25% of a calendar year. Recent provisional years can also be partial
    (release lag).
    """
    # Read the module global at call time so monkeypatch in tests works.
    import sys as _sys
    threshold = getattr(_sys.modules[__name__], "_PARTIAL_YEAR_THRESHOLD",
                        _PARTIAL_YEAR_THRESHOLD)
    return total_us_births < threshold


def fetch_county_year(*, refresh: bool = False) -> pd.DataFrame:
    """Return (county_fips, year, births) for all US counties.

    Drops state-total rows (SUMLEV='040', COUNTY='000') and partial-year
    columns (vintage base April-June, plus any current year still being
    released). Effective coverage is 2001-2023 calendar-year totals.
    """
    frames = []
    for vintage, fname, years in _VINTAGES:
        try:
            raw = _read_vintage(vintage, fname, refresh=refresh)
        except Exception as e:
            logger.warning("Census PEP %s fetch failed: %s", vintage, e)
            continue
        long = _melt_births(raw, years)
        # County rows only.
        county = long[long["SUMLEV"] == "050"].copy()
        county["county_fips"] = county["STATE"] + county["COUNTY"]
        county = county.dropna(subset=["births"])
        county["births"] = county["births"].astype(int)
        # Drop partial vintage-base / lagging years.
        for y, total in county.groupby("year")["births"].sum().items():
            if _is_partial_year(int(y), float(total)):
                logger.info(
                    "Census PEP %s: dropping year %d (partial; US total=%s)",
                    vintage, y, f"{int(total):,}",
                )
                county = county[county["year"] != y]
        county["source"] = f"census_pep_{vintage}"
        frames.append(county[["county_fips", "year", "births", "source"]])
    if not frames:
        return pd.DataFrame(columns=["county_fips", "year", "births", "source"])
    return pd.concat(frames, ignore_index=True).sort_values(
        ["county_fips", "year"]
    ).reset_index(drop=True)


def fetch_state_year(*, refresh: bool = False) -> pd.DataFrame:
    """Return (state_fips, year, births) for 50 states + DC, 2000-2025."""
    frames = []
    for vintage, fname, years in _VINTAGES:
        try:
            raw = _read_vintage(vintage, fname, refresh=refresh)
        except Exception as e:
            logger.warning("Census PEP %s fetch failed: %s", vintage, e)
            continue
        long = _melt_births(raw, years)
        state = long[long["SUMLEV"] == "040"].copy()
        state = state.rename(columns={"STATE": "state_fips"})
        state = state.dropna(subset=["births"])
        state["births"] = state["births"].astype(int)
        for y, total in state.groupby("year")["births"].sum().items():
            if _is_partial_year(int(y), float(total)):
                state = state[state["year"] != y]
        state["source"] = f"census_pep_{vintage}"
        frames.append(state[["state_fips", "year", "births", "source"]])
    if not frames:
        return pd.DataFrame(columns=["state_fips", "year", "births", "source"])
    return pd.concat(frames, ignore_index=True).sort_values(
        ["state_fips", "year"]
    ).reset_index(drop=True)
