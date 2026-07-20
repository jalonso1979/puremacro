"""Base-year and rolling shift-share weights.

w_{c,i,base} = emp_{c,i,base} / Σ_i' emp_{c,i',base}
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _assert_suppressed_dtype(panel: pd.DataFrame) -> None:
    """Ensure the suppressed column has a dtype where ~astype(bool) is safe."""
    dtype = panel["suppressed"].dtype
    assert dtype == bool or dtype == object, (
        f"'suppressed' column must be bool or object dtype, got {dtype}"
    )


def build_base_year_shares(
    industry_panel: pd.DataFrame,
    *,
    base_year: int,
) -> pd.DataFrame:
    """Compute county × NAICS shares from a given base year's four quarters.

    Suppressed cells are dropped before normalization. Counties with zero
    non-suppressed employment in the base year are dropped (and logged at
    WARNING level) rather than producing NaN shares.
    """
    _assert_suppressed_dtype(industry_panel)
    base = industry_panel[industry_panel["year"] == base_year].copy()
    base = base[~base["suppressed"].astype(bool)]
    annual = base.groupby(["county_fips", "naics"], as_index=False)["emp_avg"].mean()
    totals = annual.groupby("county_fips")["emp_avg"].sum().rename("total")
    annual = annual.join(totals, on="county_fips")
    # Drop zero-denominator counties to avoid silent NaN shares
    zero_counties = annual[annual["total"].fillna(0) == 0]["county_fips"].unique()
    if len(zero_counties) > 0:
        logger.warning(
            "Dropping %d counties with zero base-year employment: %s",
            len(zero_counties),
            list(zero_counties)[:5],
        )
        annual = annual[~annual["county_fips"].isin(zero_counties)]
    annual["share"] = annual["emp_avg"] / annual["total"]
    return annual[["county_fips", "naics", "share"]].reset_index(drop=True)


def build_rolling_shares(
    industry_panel: pd.DataFrame,
    *,
    window_quarters: int = 20,
    lag_quarters: int = 4,
    min_county_obs: int = 12,
) -> pd.DataFrame:
    """Rolling 5y-lagged shares for robustness.

    For each quarter t, the weights come from a window ending at t-lag_quarters,
    spanning window_quarters quarters back from that endpoint.

    Counties must have at least ``min_county_obs`` non-suppressed
    quarter × industry observations within the window to receive shares for a
    given period_end. Counties whose window-summed employment is zero are also
    dropped (and would otherwise produce NaN shares).

    Note on zombie counties: Shares may be emitted for a county at period_end
    even if the county has no data *at* period_end itself — the weights are
    historical. Downstream merges on (county_fips, period_end) will therefore
    include counties that are no longer reporting; filter on the current
    panel's active set if that is not desired.
    """
    _assert_suppressed_dtype(industry_panel)
    panel = industry_panel[~industry_panel["suppressed"].astype(bool)].copy()
    panel["period"] = pd.PeriodIndex.from_fields(
        year=panel["year"], quarter=panel["qtr"], freq="Q"
    )
    all_periods = sorted(panel["period"].unique())

    frames = []
    for end_p in all_periods:
        window_start = end_p - (lag_quarters + window_quarters - 1)
        window_end = end_p - lag_quarters
        if window_start < all_periods[0]:
            continue
        win = panel[(panel["period"] >= window_start) & (panel["period"] <= window_end)]
        if win.empty:
            continue
        agg = win.groupby(["county_fips", "naics"], as_index=False)["emp_avg"].mean()

        # Per-county minimum observation check
        obs_per_county = win.groupby("county_fips").size()
        keep_counties = obs_per_county[obs_per_county >= min_county_obs].index
        agg = agg[agg["county_fips"].isin(keep_counties)]
        if agg.empty:
            continue

        # Totals and divide-by-zero filter
        tot = agg.groupby("county_fips")["emp_avg"].sum().rename("total")
        agg = agg.join(tot, on="county_fips")
        zero_counties = agg[agg["total"].fillna(0) == 0]["county_fips"].unique()
        if len(zero_counties) > 0:
            logger.warning(
                "period_end=%s: dropping %d counties with zero window employment",
                end_p,
                len(zero_counties),
            )
            agg = agg[~agg["county_fips"].isin(zero_counties)]
        if agg.empty:
            continue
        agg["share"] = agg["emp_avg"] / agg["total"]
        agg["period_end"] = end_p
        frames.append(agg[["county_fips", "naics", "period_end", "share"]])

    if not frames:
        return pd.DataFrame(columns=["county_fips", "naics", "period_end", "share"])
    return pd.concat(frames, ignore_index=True)
