"""BLS CES state employment, hours, earnings by supersector (NAICS-2 rollup).

Series ID layout (SMS prefix, seasonally adjusted, 20 chars):
  SMS<state_fips:2><area_code:5><supersector:2><industry:6><datatype:2>

Datatype codes used here: 01=emp, 03=hrs, 11=ahe.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import IO, Union

import pandas as pd

from ._http import cached_get

_CES_STATE_URL = "https://download.bls.gov/pub/time.series/sm/sm.data.1.AllData"
_DATATYPE_MAP = {"01": "emp", "03": "hrs", "11": "ahe"}
_PREFIX = "SMS"


def _read_ces(source: Union[str, Path, IO[bytes]]) -> pd.DataFrame:
    return pd.read_csv(source, sep="\t", dtype=str)


def _pivot_ces(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw.columns = [c.strip() for c in raw.columns]
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    # Real BLS sm.data.1.AllData ships 30-char fixed-width series IDs with trailing
    # whitespace; strip before any slicing.
    raw["series_id"] = raw["series_id"].str.strip()
    raw = raw[raw["series_id"].str.startswith(_PREFIX)]

    sid = raw["series_id"].str
    raw["state_fips"] = sid[3:5]
    raw["area"] = sid[5:10]
    raw["supersector"] = sid[10:12]
    raw["industry"] = sid[12:18]
    # Keep state-level (area "00000") supersector-aggregate (industry "000000") rows.
    # BLS sm.data.1.AllData also ships MSA detail and within-supersector industry
    # detail; both would create duplicate (state,supersector,date,measure) keys.
    raw = raw[(raw["area"] == "00000") & (raw["industry"] == "000000")]
    raw["datatype"] = sid[-2:]
    raw["measure"] = raw["datatype"].map(_DATATYPE_MAP)
    raw = raw[raw["measure"].notna()]

    raw["period"] = raw["period"].str.strip()
    raw = raw[raw["period"].str.startswith("M") & (raw["period"] != "M13")]
    raw["month"] = raw["period"].str[1:].astype(int)
    raw["date"] = pd.to_datetime({"year": raw["year"].astype(int),
                                   "month": raw["month"],
                                   "day": 1})

    wide = raw.pivot(
        index=["state_fips", "supersector", "date"],
        columns="measure",
        values="value",
    ).reset_index()
    wide.columns.name = None
    wide["state_fips"] = wide["state_fips"].astype(object)
    wide["supersector"] = wide["supersector"].astype(object)
    wide["source"] = "bls_ces_states"
    return wide


def parse_ces_states(path: Union[str, Path, IO[bytes]]) -> pd.DataFrame:
    """Parse a CES state flat file (path or BytesIO) into wide frame."""
    return _pivot_ces(_read_ces(path))


def fetch(*, refresh: bool = False) -> pd.DataFrame:
    return _pivot_ces(_read_ces(BytesIO(cached_get(_CES_STATE_URL, refresh=refresh))))
