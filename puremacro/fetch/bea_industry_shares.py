"""BEA state-by-industry employment shares.

Primary source (when available): BEA Regional dataset via
``https://apps.bea.gov/api/data``. Table `SAEMP25N` is the historical
reference but has been replaced by newer tables with more granular
LineCode schemas that are painful to aggregate back to 2-digit NAICS.

Bundled fallback: ``data/raw/bea/state_industry_shares_bundle.csv`` — a
51 states × 20 NAICS × 3 base years synthetic-but-structured CSV used
by the T5 teaching notebook. See `data/raw/bea/README.md` for
provenance. The fetcher prefers the bundle when it exists (the usual
case for pedagogical runs); the BEA API path is kept for users with a
valid key and working table access.

Output: tidy CSV with columns (state, year, industry_code, share).
Three base years supported: 1980, 1990, 2000.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_CACHE = Path(__file__).resolve().parents[2] / "data" / "raw" / "www.bea.gov" / "state_industry_shares.csv"
_BUNDLE = Path(__file__).resolve().parents[2] / "data" / "raw" / "bea" / "state_industry_shares_bundle.csv"
_API = "https://apps.bea.gov/api/data"


_STATE_NAME_TO_CODE = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


def _fetch_raw_year(year: int, api_key: str | None = None) -> pd.DataFrame:
    """Fetch raw BEA SAEMP25N for a single year. Requires BEA_API_KEY env."""
    import requests
    from puremacro import credentials
    key = credentials.require("bea", explicit=api_key)
    params = {
        "UserID": key, "method": "GetData", "datasetname": "Regional",
        "TableName": "SAEMP25N", "LineCode": "ALL", "GeoFips": "STATE",
        "Year": str(year), "ResultFormat": "json",
    }
    r = requests.get(_API, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json().get("BEAAPI", {}).get("Results", {})
    if "Error" in payload:
        raise RuntimeError(f"BEA API error: {payload['Error']}")
    data = payload.get("Data", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "GeoName": "state_name",
        "Code": "industry_code",
        "DataValue": "emp",
    })
    df["state"] = df["state_name"].map(_STATE_NAME_TO_CODE)
    df["emp"] = pd.to_numeric(df["emp"].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna(subset=["state", "industry_code", "emp"])[
        ["state", "industry_code", "emp"]
    ].assign(year=year)


def fetch_shares(years=(1980, 1990, 2000), refresh: bool = False) -> pd.DataFrame:
    """Return state x industry employment shares per base year.

    Resolution order:
      1. If `_CACHE` exists and not refresh → return cached API result.
      2. If `_BUNDLE` exists → return the bundled synthetic shares,
         filtered to the requested years.
      3. Otherwise fall back to the BEA API (requires BEA_API_KEY).
    """
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    if _CACHE.exists() and not refresh:
        return pd.read_csv(_CACHE)

    if _BUNDLE.exists():
        df = pd.read_csv(_BUNDLE)
        df = df[df["year"].isin(years)]
        # Normalize to the four-column contract (state, year, industry_code, share).
        return df[["state", "year", "industry_code", "share"]].reset_index(drop=True)

    parts = [_fetch_raw_year(y) for y in years]
    raw = pd.concat(parts, ignore_index=True)
    raw["share"] = raw.groupby(["state", "year"])["emp"].transform(
        lambda s: s / s.sum()
    )
    out = raw[["state", "year", "industry_code", "share"]]
    out.to_csv(_CACHE, index=False)
    return out
