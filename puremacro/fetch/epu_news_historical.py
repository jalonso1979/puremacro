"""Baker-Bloom-Davis News-based historical Economic Policy Uncertainty index.

Home: https://www.policyuncertainty.com/us_historical.html
Data: US_Historical_EPU_data.xlsx — columns: Year, Month, "News-Based Historical
      Economic Policy Uncertainty" (hyphen, spaces, no underscores).
Coverage: 1900-01 → 2014-02 (historical file; the upstream does not extend past
          February 2014). For post-2014 data use src/fetch/epu.py
          (US_Policy_Uncertainty_Data.xlsx), which includes the News-based
          series from 1985-01 onward and can be spliced with this one.

The file has a trailing source-attribution row where Year is a sentence starting
"Source: ..."; the parser filters it by dropping rows where Year cannot be cast
to an integer.

Returns a tidy long-form DataFrame with columns
(code, date, variable, value, sa_source, source), matching the
convention used by the rest of `src/fetch/`.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd

from ._http import cached_get

_URL = "https://www.policyuncertainty.com/media/US_Historical_EPU_data.xlsx"


def fetch(refresh: bool = False) -> pd.DataFrame:
    """Return News-EPU historical as tidy long-form frame."""
    raw = cached_get(_URL, refresh=refresh)
    xlsx = pd.read_excel(BytesIO(raw))

    # Normalize column names — match on normalized form (hyphens and spaces
    # collapsed to underscores, lowercased) so header variants are tolerated.
    def _norm(c: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in str(c).lower())

    normed = {_norm(c): c for c in xlsx.columns}
    year_col = normed.get("year")
    month_col = normed.get("month")
    value_col = next(
        (c for k, c in normed.items() if "news" in k and "uncert" in k),
        None,
    )
    if not (year_col and month_col and value_col):
        raise RuntimeError(
            f"Unexpected columns in News-EPU historical file: {list(xlsx.columns)}"
        )

    df = xlsx[[year_col, month_col, value_col]].copy()
    # Drop trailing source-attribution row and any other non-numeric rows.
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df[month_col] = pd.to_numeric(df[month_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[year_col, month_col, value_col])

    df["date"] = pd.to_datetime(
        df[year_col].astype(int).astype(str) + "-"
        + df[month_col].astype(int).astype(str) + "-01"
    )
    out = pd.DataFrame(
        {
            "code": "USA",
            "date": df["date"].values,
            "variable": "epu_news_hist",
            "value": df[value_col].astype(float).values,
            "sa_source": "NSA",
            "source": "policyuncertainty.com",
        }
    ).sort_values("date").reset_index(drop=True)

    return out
