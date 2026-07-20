"""BLS QCEW county x NAICS x quarter fetcher.

Pulls quarterly single-file CSVs from the QCEW Open Data Access site:
https://www.bls.gov/cew/downloadable-data-files.htm

Subnational fetchers use a flat per-cell schema, not the country-level
(code, date, variable, value) schema used by `epu.py`, `wui.py`, etc.

Columns returned: county_fips, naics, year, qtr, emp_avg, total_wages,
aww, estabs, suppressed, source
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd

from ._http import cached_get

_QCEW_SINGLE_FILE_URL = (
    "https://data.bls.gov/cew/data/files/{year}/csv/{year}_qtrly_singlefile.zip"
)


def _project(raw: pd.DataFrame) -> pd.DataFrame:
    """Filter + project a raw QCEW frame into the long-form panel shape.

    Filters to county-level private ownership (own_code=5). Keeps four agglvl
    codes:
      71 = county × industry-total (industry='10', post-2001 NAICS)
      73 = county × supersector (pre-2001 SIC pattern)
      74 = county × NAICS-3 (post-2001)
      75 = county × SIC-3 (pre-2001)
    """
    raw = raw[raw["agglvl_code"].isin([71, 73, 74, 75])]
    raw = raw[raw["own_code"] == 5]

    # Use Python bool (not numpy.bool_) so identity checks like `is True` work.
    suppressed = (
        raw["disclosure_code"].astype(str).str.contains("N")
        .map(bool).astype(object)
    )

    out = pd.DataFrame({
        "county_fips": raw["area_fips"].astype(object),
        "naics": raw["industry_code"].astype(object),
        "year": raw["year"].astype(int),
        "qtr": raw["qtr"].astype(int),
        "emp_avg": raw[["month1_emplvl", "month2_emplvl", "month3_emplvl"]]
                        .mean(axis=1, skipna=False),
        "total_wages": raw["total_qtrly_wages"],
        "aww": raw["avg_wkly_wage"],
        "estabs": raw["qtrly_estabs"],
        "suppressed": suppressed,
        "source": "bls_qcew",
    })
    # Suppressed rows have no values. Cast object-dtype mask to bool because
    # recent pandas rejects object-dtype boolean masks in `.loc[...]`.
    out.loc[out["suppressed"].astype(bool),
            ["emp_avg", "total_wages", "aww", "estabs"]] = pd.NA
    return out.reset_index(drop=True)


def parse_qcew_csv(path: str | Path) -> pd.DataFrame:
    """Parse a QCEW single-file CSV into the long-form panel shape."""
    raw = pd.read_csv(path, dtype={"area_fips": str, "industry_code": str})
    return _project(raw)


def fetch(years: Iterable[int], *, refresh: bool = False) -> pd.DataFrame:
    """Download QCEW single-file zips for the given years and return long-form."""
    frames = []
    for year in years:
        content = cached_get(_QCEW_SINGLE_FILE_URL.format(year=year), refresh=refresh)
        raw = pd.read_csv(
            BytesIO(content),
            compression="zip",
            dtype={"area_fips": str, "industry_code": str},
        )
        frames.append(_project(raw))
    return pd.concat(frames, ignore_index=True)
