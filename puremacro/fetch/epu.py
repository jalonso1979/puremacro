"""Economic Policy Uncertainty (Baker-Bloom-Davis) country indices.

Home: https://www.policyuncertainty.com/
Data: All_Country_Data.xlsx — columns: Year, Month, <country names>
      Country names include GEPU aggregates (skipped) and individual countries.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd

from ._http import cached_get

_EPU_URL = "https://www.policyuncertainty.com/media/All_Country_Data.xlsx"

# Mapping from column names as they appear in the file to ISO-3 codes.
# GEPU_current and GEPU_ppp are global aggregates — excluded (no ISO-3 code).
_NAME_TO_ISO3: dict[str, str] = {
    "Australia": "AUS",
    "Brazil": "BRA",
    "Canada": "CAN",
    "Chile": "CHL",
    "China": "CHN",
    "Mainland China": "CHN",
    "SCMP China": "CHN",
    "France": "FRA",
    "Germany": "DEU",
    "Greece": "GRC",
    "India": "IND",
    "Ireland": "IRL",
    "Italy": "ITA",
    "Japan": "JPN",
    "Korea": "KOR",
    "Mexico": "MEX",
    "Pakistan": "PAK",
    "Russia": "RUS",
    "Singapore": "SGP",
    "Spain": "ESP",
    "Sweden": "SWE",
    "UK": "GBR",
    "US": "USA",
    # Extended variants
    "United States": "USA",
    "United Kingdom": "GBR",
    "South Korea": "KOR",
    "Hong Kong": "HKG",
    "Netherlands": "NLD",
    "Belgium": "BEL",
    "Colombia": "COL",
    "South Africa": "ZAF",
    "New Zealand": "NZL",
}


def fetch(refresh: bool = False) -> pd.DataFrame:
    """Return EPU monthly indices in long form.

    Columns: code, date, variable, value, sa_source, source.
    variable is always 'epu_m'.  GEPU aggregate columns are dropped.
    """
    content = cached_get(_EPU_URL, refresh=refresh)
    raw = pd.read_excel(BytesIO(content))

    # Verify expected Year/Month structure.
    if "Year" not in raw.columns or "Month" not in raw.columns:
        raise ValueError(
            f"EPU file missing Year/Month columns. Got: {list(raw.columns)}"
        )

    # Drop rows with missing Year or Month (trailing/metadata rows in the file).
    raw = raw.dropna(subset=["Year", "Month"])
    raw["date"] = pd.to_datetime(
        raw["Year"].astype(int).astype(str)
        + "-"
        + raw["Month"].astype(int).astype(str)
        + "-01"
    )

    country_cols = [c for c in raw.columns if c not in ("Year", "Month", "date")]
    long = raw[["date"] + country_cols].melt(
        id_vars="date", var_name="name", value_name="value"
    )
    long["code"] = long["name"].map(_NAME_TO_ISO3)
    # Drop rows with no ISO-3 mapping (GEPU aggregates) or no value.
    long = long.dropna(subset=["code", "value"])
    long["variable"] = "epu_m"
    long["sa_source"] = "none"
    long["source"] = "EPU"
    return long[["code", "date", "variable", "value", "sa_source", "source"]].reset_index(drop=True)
