"""Time-varying county-level EPU from state EPU × industry shares.

epu_county_bartik_{c,t} = Σ_i w_{c,i,base} × epu_state_{s(c),t}

Because state EPU is industry-agnostic, this collapses to
    (Σ_i w_{c,i}) × epu_state_{s(c),t}
which equals epu_state when shares sum to 1. The shares layer still matters
because residuals from suppressed-and-not-imputable cells leave Σ_i w_{c,i} < 1
in practice.
"""
from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)


def build_county_epu(
    shares: pd.DataFrame,
    state_crosswalk: pd.DataFrame,
    state_epu_monthly: pd.DataFrame,
    *,
    freq: Literal["M", "Q"] = "Q",
) -> pd.DataFrame:
    """Construct county EPU at the requested frequency.

    Parameters
    ----------
    shares : DataFrame with columns (county_fips, naics, share)
    state_crosswalk : DataFrame with columns (county_fips, state_fips)
    state_epu_monthly : DataFrame with columns (state_fips, date, epu_state_m)
    freq : 'M' for monthly, 'Q' for quarterly-averaged.
    """
    if freq not in ("M", "Q"):
        raise ValueError(f"freq must be 'M' or 'Q', got {freq!r}")

    shares = shares.copy()
    shares["county_fips"] = shares["county_fips"].astype(str)

    state_crosswalk = state_crosswalk.copy()
    state_crosswalk["county_fips"] = state_crosswalk["county_fips"].astype(str)
    state_crosswalk["state_fips"] = state_crosswalk["state_fips"].astype(str)

    state_epu_monthly = state_epu_monthly.copy()
    state_epu_monthly["state_fips"] = state_epu_monthly["state_fips"].astype(str)

    county_weight = shares.groupby("county_fips", as_index=False)["share"].sum()
    county_weight = county_weight.rename(columns={"share": "share_sum"})
    cw = state_crosswalk.merge(county_weight, on="county_fips", how="left")

    missing_shares = cw["share_sum"].isna().sum()
    if missing_shares > 0:
        logger.warning(
            "%d counties in crosswalk have no share data; share_sum will be NaN",
            missing_shares,
        )

    if freq == "Q":
        epu = state_epu_monthly.copy()
        epu["period"] = epu["date"].dt.to_period("Q")
        agg = epu.groupby(["state_fips", "period"], as_index=False)["epu_state_m"].mean()
        agg["date"] = agg["period"].dt.to_timestamp(how="start")
        source = agg[["state_fips", "date", "epu_state_m"]]
        value_col = "epu_county_bartik_q"
    else:
        source = state_epu_monthly[["state_fips", "date", "epu_state_m"]].copy()
        value_col = "epu_county_bartik_m"

    merged = cw.merge(source, on="state_fips", how="left")
    merged[value_col] = merged["share_sum"] * merged["epu_state_m"]

    missing_epu = merged[value_col].isna().sum()
    total = len(merged)
    if total > 0 and missing_epu / total > 0.1:
        logger.warning(
            "%d/%d (%.0f%%) county-date rows have no matching state EPU; "
            "epu_county_bartik will be NaN for those rows",
            missing_epu, total, 100 * missing_epu / total,
        )

    return merged[["county_fips", "state_fips", "date", value_col]].reset_index(drop=True)
