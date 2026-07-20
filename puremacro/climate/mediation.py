"""Within-year-quintile mediation LP for climate shocks.

Splits regions into quintiles of a mediator's year-over-year growth
(e.g., housing-price growth) computed within each year, then runs two
annual LPs: one baseline and one with top-quintile × shock interactions
added as controls. Reports the per-horizon mediation share.
"""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from .annual_lp import climate_annual_lp


def _within_year_quintile(
    panel: pd.DataFrame,
    mediator_col: str,
    region_col: str,
    year_col: str,
    n_bins: int = 5,
) -> pd.Series:
    """Quintile bucket of the mediator's year-over-year growth per region,
    ranked within each year across regions.

    Returns a Series aligned with ``panel.index`` whose values are the
    integer bin index in ``[0, n_bins-1]`` or NaN when the mediator
    growth is NaN (first year per region has no prior, so it's always NaN).
    Emits a warning when any year with a defined cross-section produces
    fewer than ``n_bins`` distinct buckets.
    """
    df = panel[[region_col, year_col, mediator_col]].copy()
    df = df.sort_values([region_col, year_col])
    df["_g"] = df.groupby(region_col, observed=True)[mediator_col].pct_change()
    df["_q"] = df.groupby(year_col, observed=True)["_g"].transform(
        lambda s: pd.qcut(s, n_bins, labels=False, duplicates="drop")
    )
    # Warn only for years that have non-NaN growth but produced too few bins.
    # (Year zero is universally all-NaN by construction — that's expected, not a warning.)
    counts_per_year = df.groupby(year_col, observed=True)["_g"].count()
    bins_per_year = df.groupby(year_col, observed=True)["_q"].nunique(dropna=True)
    short_years = bins_per_year[(counts_per_year > 0) & (bins_per_year < n_bins)].index.tolist()
    if short_years:
        warnings.warn(
            f"_within_year_quintile: {len(short_years)} year(s) produced "
            f"fewer than {n_bins} quintile bins (first: {short_years[:3]}); "
            "mediator may have too few distinct values.",
            stacklevel=2,
        )
    return df["_q"].reindex(panel.index)


def climate_mediation_lp(
    panel: pd.DataFrame,
    *,
    mediator_col: str,
    response: str,
    cdd_col: str = "annual_cdd",
    hdd_col: str = "annual_hdd",
    horizons: Iterable[int] = range(0, 11),
    n_lags: int = 2,
    region_col: str = "region",
    year_col: str = "year",
    n_bins: int = 5,
    top_quintile_only: bool = True,
) -> dict:
    """Return baseline, interacted, and mediation-share LP results."""
    horizons = list(horizons)
    df = panel.copy()
    df["_q"] = _within_year_quintile(df, mediator_col, region_col, year_col, n_bins=n_bins)
    df["_top"] = (df["_q"] == (n_bins - 1)).astype(float)
    df["_cdd_top"] = df[cdd_col] * df["_top"]
    df["_hdd_top"] = df[hdd_col] * df["_top"]

    baseline = climate_annual_lp(
        df, response=response, cdd_col=cdd_col, hdd_col=hdd_col,
        horizons=horizons, n_lags=n_lags,
        region_col=region_col, year_col=year_col,
    )
    interacted = climate_annual_lp(
        df, response=response, cdd_col=cdd_col, hdd_col=hdd_col,
        horizons=horizons, n_lags=n_lags,
        region_col=region_col, year_col=year_col,
        controls=("_cdd_top", "_hdd_top", mediator_col),
    )

    return {
        "baseline": baseline,
        "interacted": interacted,
        "mediation_share_cdd": _mediation_share(baseline["cdd"], interacted["cdd"]),
        "mediation_share_hdd": _mediation_share(baseline["hdd"], interacted["hdd"]),
    }


def _mediation_share(
    baseline_df: pd.DataFrame, interacted_df: pd.DataFrame
) -> np.ndarray:
    """Compute the per-horizon mediation share ``(b - c) / b``.

    Clamps to 0 when ``|b| < 1e-12`` to avoid division-by-near-zero
    producing NaN or Inf.

    Parameters
    ----------
    baseline_df, interacted_df : DataFrames with a ``'beta'`` column.

    Returns
    -------
    np.ndarray of the same length as the input DataFrames.
    """
    b = baseline_df["beta"].to_numpy()
    c = interacted_df["beta"].to_numpy()
    near_zero = np.abs(b) < 1e-12
    return np.where(near_zero, 0.0, (b - c) / np.where(near_zero, 1.0, b))


__all__ = ["climate_mediation_lp"]
