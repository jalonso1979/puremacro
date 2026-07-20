"""Pedagogical shift-share (Bartik) instrument builder.

Core formula:
    bartik_{s,t} = Σ_k  share_{s,k}  ×  shift_{k,t}
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def predict(
    shares: pd.DataFrame,
    shifts: pd.DataFrame,
    *,
    share_state_col: str,
    share_bucket_col: str,
    share_value_col: str,
    shift_bucket_col: str,
    shift_date_col: str,
    shift_value_col: str,
) -> pd.DataFrame:
    """Return tidy `(state, date, bartik)` DataFrame."""
    merged = shares.merge(
        shifts,
        left_on=share_bucket_col,
        right_on=shift_bucket_col,
        how="inner",
    )
    merged["contrib"] = merged[share_value_col] * merged[shift_value_col]
    out = (merged.groupby([share_state_col, shift_date_col])["contrib"]
                 .sum()
                 .reset_index()
                 .rename(columns={
                     share_state_col: "state",
                     shift_date_col: "date",
                     "contrib": "bartik",
                 }))
    return out


def rolling_sigma_of_returns(
    returns: pd.DataFrame, *, window: int = 6, min_periods: int = 3,
    bucket_col: str = "industry_name", date_col: str = "date", value_col: str = "ret",
) -> pd.DataFrame:
    """Convert a monthly-return frame into rolling-σ shifts per industry.

    Returns tidy `(bucket, date, sigma)` — no look-ahead.
    """
    out = returns.sort_values([bucket_col, date_col]).copy()
    out["sigma"] = out.groupby(bucket_col)[value_col].transform(
        lambda s: s.rolling(window, min_periods=min_periods).std()
    )
    return out[[bucket_col, date_col, "sigma"]].dropna()
