"""Baker-Bloom-Davis-Levy state-level EPU (Economic Policy Uncertainty).

Home:  https://www.policyuncertainty.com/state_epu.html
Data:  https://www.policyuncertainty.com/media/State_Policy_Uncertainty.xlsx
       Wide Excel — columns: year, month, EPU_National{ST}, EPU_State{ST},
       EPU_Composite{ST} for each of 50 states + DC. 1985-01 → present.

Three indices per state are published:
  - EPU_National{ST} — share of national news articles mentioning the state
    AND policy AND economy keywords.
  - EPU_State{ST}   — share of state-level news articles mentioning policy
    AND economy keywords.
  - EPU_Composite{ST} — combined index (the canonical state-EPU used in the
    Baker-Bloom-Davis-Levy 2022 paper).

By default this fetcher returns the **Composite** index with
variable='state_epu'. Callers who want a different family can pass
``family='national'`` or ``family='state'``.

Returns a tidy long-form DataFrame with columns
(state, date, variable, value, sa_source, source).
"""
from __future__ import annotations

from io import BytesIO
from typing import Literal

import pandas as pd

from ._http import cached_get

_URL = "https://www.policyuncertainty.com/media/State_Policy_Uncertainty.xlsx"

_STATE_CODES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC",
}

_PREFIX_BY_FAMILY = {
    "composite": "EPU_Composite",
    "national":  "EPU_National",
    "state":     "EPU_State",
}


def fetch(
    refresh: bool = False,
    family: Literal["composite", "national", "state"] = "composite",
) -> pd.DataFrame:
    """Return state-level EPU as tidy long-form frame.

    Parameters
    ----------
    refresh : bool
        Force re-download of the upstream XLSX.
    family : {'composite', 'national', 'state'}
        Which index family to return. Default 'composite' (canonical).
    """
    if family not in _PREFIX_BY_FAMILY:
        raise ValueError(f"family must be one of {list(_PREFIX_BY_FAMILY)}; got {family!r}")
    prefix = _PREFIX_BY_FAMILY[family]

    raw = cached_get(_URL, refresh=refresh)
    xlsx = pd.read_excel(BytesIO(raw))

    cols_lower = {str(c).lower(): c for c in xlsx.columns}
    year_col = cols_lower.get("year")
    month_col = cols_lower.get("month")
    if not (year_col and month_col):
        raise RuntimeError(
            f"Expected 'year' and 'month' columns; got {list(xlsx.columns)[:5]}…"
        )

    # Columns for the chosen family: e.g. 'EPU_CompositeCA', 'EPU_CompositeTX'.
    family_cols = []
    for c in xlsx.columns:
        cs = str(c)
        if cs.startswith(prefix) and cs[len(prefix):].upper() in _STATE_CODES:
            family_cols.append(c)
    if not family_cols:
        raise RuntimeError(
            f"No '{prefix}<ST>' columns found in {list(xlsx.columns)[:10]}…"
        )

    xlsx = xlsx.dropna(subset=[year_col, month_col]).copy()
    xlsx["date"] = pd.to_datetime(
        xlsx[year_col].astype(int).astype(str) + "-"
        + xlsx[month_col].astype(int).astype(str) + "-01"
    )
    long = xlsx.melt(
        id_vars=["date"], value_vars=family_cols,
        var_name="col", value_name="value",
    )
    long["state"] = long["col"].astype(str).str.slice(len(prefix)).str.upper()
    long["variable"] = "state_epu"
    long["sa_source"] = "NSA"
    long["source"] = "policyuncertainty.com"
    long = (long.dropna(subset=["value"])
                .sort_values(["state", "date"])
                .reset_index(drop=True))
    return long[["state", "date", "variable", "value", "sa_source", "source"]]
