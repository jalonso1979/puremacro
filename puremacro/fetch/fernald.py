"""Fernald utilization-adjusted TFP (FRBSF, US quarterly).

Home: https://www.frbsf.org/economic-research/indicators-data/total-factor-productivity-tfp/

The workbook has a note row before the header, so we read with header=1.
The date column contains year:quarter strings like "1947:Q1"; we parse these
with pd.PeriodIndex.  The column dtfp_util holds annualized quarterly log
growth (% at annual rate); we divide by 400 to get quarterly log changes and
cumulate to build a level index anchored at the first observation.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd

from ._http import cached_get

_FERNALD_URL = "https://www.frbsf.org/wp-content/uploads/quarterly_tfp.xlsx"


def fetch(refresh: bool = False) -> pd.DataFrame:
    """Return Fernald utilization-adjusted TFP level series in long form.

    Columns: code, date, variable, value, sa_source, source.
    Variable: tfp_fernald (cumulated log-level, anchored at first observation).
    Dates are quarter-start timestamps (freq Q -> to_timestamp()).
    """
    content = cached_get(_FERNALD_URL, refresh=refresh)
    # Row 0 is a note; actual column headers are in row 1 (0-indexed).
    raw = pd.read_excel(BytesIO(content), sheet_name="quarterly", header=1)

    date_col = raw.columns[0]  # labeled "date" after skipping the note row
    if "dtfp_util" not in raw.columns:
        raise ValueError(
            f"Fernald file missing 'dtfp_util' column. Got: {list(raw.columns)}"
        )

    # Keep only rows with a valid "YYYY:Qn" date (the file has summary rows at the end).
    date_mask = raw[date_col].astype(str).str.match(r"^\d{4}:Q\d$")
    raw = raw[date_mask].copy()

    # Date strings are "YYYY:Qn" (e.g. "1947:Q1"); convert to quarter-start timestamps.
    period_strings = raw[date_col].astype(str).str.replace(":", "", regex=False)
    raw["date"] = pd.PeriodIndex(period_strings, freq="Q").to_timestamp()

    # Convert annualized % growth to quarterly log change, then cumulate to a level.
    growth = raw["dtfp_util"].astype(float) / 400.0
    level = growth.fillna(0).cumsum()
    level = level - level.iloc[0]

    out = pd.DataFrame(
        {
            "code": "USA",
            "date": raw["date"],
            "variable": "tfp_fernald",
            "value": level.values,
            "sa_source": "frbsf",
            "source": "Fernald",
        }
    )
    return out.dropna(subset=["value"])
