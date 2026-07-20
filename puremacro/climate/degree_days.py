"""Cooling- and heating-degree-day construction from monthly temperatures."""
from __future__ import annotations

import math

import pandas as pd


def compute_monthly_cdd_hdd(
    df: pd.DataFrame,
    *,
    temp_col: str = "temp_c",
    threshold: float = 18.0,
) -> pd.DataFrame:
    """Return a copy of df with two new columns:

        cdd = max(temp - threshold, 0)
        hdd = max(threshold - temp, 0)

    Parameters
    ----------
    df : DataFrame containing ``temp_col``.
    temp_col : column name in Celsius.
    threshold : base temperature for the degree-day cut-off.

    Raises
    ------
    KeyError if ``temp_col`` is not a column of df.
    ValueError if ``threshold`` is not finite.
    """
    if temp_col not in df.columns:
        raise KeyError(f"compute_monthly_cdd_hdd: expected column {temp_col!r} in df")
    if not math.isfinite(threshold):
        raise ValueError(f"compute_monthly_cdd_hdd: threshold must be finite, got {threshold}")
    out = df.copy()
    out["cdd"] = (out[temp_col] - threshold).clip(lower=0.0)
    out["hdd"] = (threshold - out[temp_col]).clip(lower=0.0)
    return out


def compute_annual_cdd_hdd(
    df: pd.DataFrame,
    *,
    temp_col: str = "temp_c",
    threshold: float = 18.0,
    region_col: str = "region",
    year_col: str = "year",
    month_col: str = "month",
) -> pd.DataFrame:
    """Aggregate monthly degree-days to annual by summing across months.

    If ``cdd``/``hdd`` columns are not already present, compute them via
    ``compute_monthly_cdd_hdd``.

    Returns
    -------
    DataFrame with columns ``[region_col, year_col, annual_cdd, annual_hdd]``.
    """
    if "cdd" not in df.columns or "hdd" not in df.columns:
        df = compute_monthly_cdd_hdd(df, temp_col=temp_col, threshold=threshold)
    for col in (region_col, year_col, month_col):
        if col not in df.columns:
            raise KeyError(f"compute_annual_cdd_hdd: expected column {col!r} in df")
    annual = (
        df.groupby([region_col, year_col], observed=True)[["cdd", "hdd"]]
        .sum()
        .reset_index()
        .rename(columns={"cdd": "annual_cdd", "hdd": "annual_hdd"})
    )
    return annual


__all__ = ["compute_monthly_cdd_hdd", "compute_annual_cdd_hdd"]
