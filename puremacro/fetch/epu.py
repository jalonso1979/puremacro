"""Economic Policy Uncertainty (Baker-Bloom-Davis) country indices.

Home: https://www.policyuncertainty.com/
Data: All_Country_Data.xlsx — columns: Year, Month, <country names>
      Country names include GEPU aggregates (skipped) and individual countries.
"""

from __future__ import annotations

import warnings
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


#: Columns that are deliberately not countries: the two global aggregates,
#: which have no ISO-3 code and are excluded by design.
_GEPU_COLS: frozenset[str] = frozenset({"GEPU_current", "GEPU_ppp"})

#: The ISO-3 codes this workbook is *known* to carry, verified against the
#: 2026-09 edition. This is deliberately NOT ``set(_NAME_TO_ISO3.values())``:
#: that map is a broad name->code translation carrying variant spellings
#: ("US" and "United States") and codes this particular file has never
#: published (BEL, COL, HKG, NLD, NZL, ZAF), so measuring against it would
#: report six absences that are not news on every single call.
#:
#: Measured against a verified snapshot instead, a difference in either
#: direction is a real change in the source and worth a human decision.
#: Sweden was in this workbook until the 2026-09 edition and is not any more,
#: which is why SWE is absent below -- `_NAME_TO_ISO3` still maps it, so if it
#: comes back the guard reports that too rather than letting it slip in
#: unnoticed.
_EXPECTED_CODES: frozenset[str] = frozenset({
    "AUS", "BRA", "CAN", "CHL", "CHN", "DEU", "ESP", "FRA", "GBR", "GRC",
    "IND", "IRL", "ITA", "JPN", "KOR", "MEX", "PAK", "RUS", "SGP", "USA",
})


def _report_coverage(file_cols: list[str], produced: set[str]) -> None:
    """Say so when the workbook stops -- or starts -- carrying a country.

    `fetch` builds its country list from the columns the file actually has,
    never from `_NAME_TO_ISO3`, so a country that disappears upstream simply
    disappears from `epu_m`: no error, no warning, and a panel quietly one
    country short. That is how Sweden left without comment. This is the
    diagnostic that makes the next one audible.
    """
    gone = sorted(_EXPECTED_CODES - produced)
    if gone:
        warnings.warn(
            f"EPU: the workbook no longer carries {', '.join(gone)}. `epu_m` "
            f"is returned without {'them' if len(gone) > 1 else 'it'} and "
            f"nothing downstream will say so -- `build_panel` merges whatever "
            f"arrives. If this is the source's own decision, drop the code(s) "
            f"from `_EXPECTED_CODES`; if not, the download is incomplete.",
            UserWarning, stacklevel=3)

    new = sorted(produced - _EXPECTED_CODES)
    if new:
        warnings.warn(
            f"EPU: the workbook now carries {', '.join(new)}, which "
            f"`_EXPECTED_CODES` did not list. The data are included. Add the "
            f"code(s) to `_EXPECTED_CODES` once you have checked the series.",
            UserWarning, stacklevel=3)

    unmapped = [c for c in file_cols
                if c not in _NAME_TO_ISO3 and c not in _GEPU_COLS]
    if unmapped:
        warnings.warn(
            f"EPU: {len(unmapped)} column(s) have no ISO-3 mapping and were "
            f"dropped: {', '.join(map(str, unmapped))}. If any is a country, "
            f"add it to `_NAME_TO_ISO3` -- unmapped columns are discarded "
            f"silently, which is how a newly published country would be lost.",
            UserWarning, stacklevel=3)


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
    _report_coverage(country_cols, set(long["code"].unique()))
    long["variable"] = "epu_m"
    long["sa_source"] = "none"
    long["source"] = "EPU"
    return long[["code", "date", "variable", "value", "sa_source", "source"]].reset_index(drop=True)
