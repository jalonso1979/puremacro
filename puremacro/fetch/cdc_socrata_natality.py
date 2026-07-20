"""CDC NCHS state × month natality via the Socrata SODA API (2023+).

Dataset: VSRR — State and National Provisional Counts for Live Births,
Deaths, and Infant Deaths. Socrata id `hmz2-vwda`.

  https://data.cdc.gov/resource/hmz2-vwda.json
  https://data.cdc.gov/NCHS/VSRR-State-and-National-Provisional-Counts-for-Liv/hmz2-vwda

This is the only public, programmatic source of state × month live-
birth counts post-2002. CDC WONDER's API rejects state-level grouping
for natality, NCHS NVSR "Births: Final Data" supplements never cross
state with month, and NCHS All-County Natality is RDC-restricted. So
between this Socrata endpoint (2023+) and the NBER mirror used by
`cdc_births_state` (1968–2002), there is a programmatic **gap from 2003
through 2022** that can only be filled via manual WONDER export.

Public API:
    fetch_state_month() -> pd.DataFrame[state_fips, date, births_state_m]
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


_SOCRATA_URL = "https://data.cdc.gov/resource/hmz2-vwda.json"
_PAGE_LIMIT = 5000

_STATE_NAME_TO_FIPS = {
    "ALABAMA": "01", "ALASKA": "02", "ARIZONA": "04", "ARKANSAS": "05",
    "CALIFORNIA": "06", "COLORADO": "08", "CONNECTICUT": "09",
    "DELAWARE": "10", "DISTRICT OF COLUMBIA": "11", "FLORIDA": "12",
    "GEORGIA": "13", "HAWAII": "15", "IDAHO": "16", "ILLINOIS": "17",
    "INDIANA": "18", "IOWA": "19", "KANSAS": "20", "KENTUCKY": "21",
    "LOUISIANA": "22", "MAINE": "23", "MARYLAND": "24",
    "MASSACHUSETTS": "25", "MICHIGAN": "26", "MINNESOTA": "27",
    "MISSISSIPPI": "28", "MISSOURI": "29", "MONTANA": "30",
    "NEBRASKA": "31", "NEVADA": "32", "NEW HAMPSHIRE": "33",
    "NEW JERSEY": "34", "NEW MEXICO": "35", "NEW YORK": "36",
    "NORTH CAROLINA": "37", "NORTH DAKOTA": "38", "OHIO": "39",
    "OKLAHOMA": "40", "OREGON": "41", "PENNSYLVANIA": "42",
    "RHODE ISLAND": "44", "SOUTH CAROLINA": "45", "SOUTH DAKOTA": "46",
    "TENNESSEE": "47", "TEXAS": "48", "UTAH": "49", "VERMONT": "50",
    "VIRGINIA": "51", "WASHINGTON": "53", "WEST VIRGINIA": "54",
    "WISCONSIN": "55", "WYOMING": "56",
}

_MONTH_NAME_TO_NUM = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5,
    "June": 6, "July": 7, "August": 8, "September": 9, "October": 10,
    "November": 11, "December": 12,
}


def _socrata_get(params: dict, *, timeout: int = 60) -> list[dict]:
    """Page through SODA results until exhausted."""
    rows: list[dict] = []
    offset = 0
    while True:
        q = dict(params)
        q["$limit"] = _PAGE_LIMIT
        q["$offset"] = offset
        resp = requests.get(_SOCRATA_URL, params=q, timeout=timeout,
                            headers={"User-Agent": "uncertainty_examples/1.0 (research@itam.mx)"})
        resp.raise_for_status()
        chunk = resp.json()
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < _PAGE_LIMIT:
            break
        offset += _PAGE_LIMIT
    return rows


def fetch_state_month() -> pd.DataFrame:
    """Return a (state_fips, date, births_state_m) frame for state-level live
    births, monthly. National-level rows ("UNITED STATES", "PUERTO RICO") are
    dropped — we only return 50 states + DC.
    """
    raw = _socrata_get({
        "indicator": "Number of Live Births",
        "period": "Monthly",
    })
    if not raw:
        return pd.DataFrame(columns=["state_fips", "date", "births_state_m"])
    df = pd.DataFrame(raw)
    df["state_fips"] = df["state"].str.upper().map(_STATE_NAME_TO_FIPS)
    df["month"] = df["month"].map(_MONTH_NAME_TO_NUM)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["births_state_m"] = pd.to_numeric(df["data_value"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["state_fips", "year", "month", "births_state_m"])
    df["date"] = pd.to_datetime(
        {"year": df["year"], "month": df["month"], "day": 1}
    )
    return df[["state_fips", "date", "births_state_m"]].sort_values(
        ["state_fips", "date"]
    ).reset_index(drop=True)
