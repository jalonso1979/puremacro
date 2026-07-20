"""BLS LAUS state and county fetcher.

State series are pulled from `la.data.1.CurrentS` (seasonally adjusted) with
prefix `LASST`. County series come from `la.data.64.County` (only published
unadjusted; prefix `LAUCN`); panel users should add seasonal dummies or
apply X-13 if county-level seasonality biases their estimates.

Series ID layout (after right-stripping trailing whitespace, length 20):
  LASST<state_fips:2><area_code:13><measure:2>           (state, SA)
  LAUCN<state_fips:2><county_fips:3><area_code:10><measure:2>   (county, NSA)

Measures: 03=urate, 04=lf, 05=emp, 06=une.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import IO, Literal, Union

import pandas as pd

from ._http import cached_get

# State: seasonally adjusted (la.data.1.CurrentS, prefix LASST)
_LAUS_STATE_URL = "https://download.bls.gov/pub/time.series/la/la.data.1.CurrentS"
# County: only available unadjusted (la.data.64.County, prefix LAUCN)
_LAUS_COUNTY_URL = "https://download.bls.gov/pub/time.series/la/la.data.64.County"
_MEASURE_MAP = {"03": "urate", "04": "lf", "05": "emp", "06": "une"}


def _read_laus(source: Union[str, Path, IO[bytes]]) -> pd.DataFrame:
    return pd.read_csv(source, sep="\t", dtype=str)


def _pivot_laus(raw: pd.DataFrame, geography: Literal["state", "county"]) -> pd.DataFrame:
    raw = raw.copy()
    raw.columns = [c.strip() for c in raw.columns]
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    # Real BLS LAUS series IDs are 30-char fixed-width with trailing whitespace.
    raw["series_id"] = raw["series_id"].str.strip()

    # SA state series use prefix LASST; NSA county series use LAUCN.
    prefix = "LASST" if geography == "state" else "LAUCN"
    raw = raw[raw["series_id"].str.startswith(prefix)]

    sid = raw["series_id"].str
    raw["measure"] = sid[-2:].map(_MEASURE_MAP)
    raw = raw[raw["measure"].notna()]

    if geography == "state":
        raw["area_fips"] = sid[5:7]
        area_col = "state_fips"
    else:
        raw["area_fips"] = sid[5:10]
        area_col = "county_fips"

    raw["period"] = raw["period"].str.strip()
    raw = raw[raw["period"].str.startswith("M") & (raw["period"] != "M13")]
    raw["month"] = raw["period"].str[1:].astype(int)
    raw["date"] = pd.to_datetime({"year": raw["year"].astype(int),
                                   "month": raw["month"],
                                   "day": 1})

    wide = raw.pivot(
        index=["area_fips", "date"],
        columns="measure",
        values="value",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"area_fips": area_col})
    wide[area_col] = wide[area_col].astype(object)
    wide["source"] = "bls_laus"
    return wide


def parse_laus_series(
    path: Union[str, Path, IO[bytes]],
    geography: Literal["state", "county"] = "state",
) -> pd.DataFrame:
    """Parse a LAUS flat file (path or BytesIO) into wide frame."""
    return _pivot_laus(_read_laus(path), geography)


def fetch_state(*, refresh: bool = False) -> pd.DataFrame:
    return _pivot_laus(_read_laus(BytesIO(cached_get(_LAUS_STATE_URL, refresh=refresh))), "state")


def fetch_county(*, refresh: bool = False) -> pd.DataFrame:
    return _pivot_laus(_read_laus(BytesIO(cached_get(_LAUS_COUNTY_URL, refresh=refresh))), "county")
