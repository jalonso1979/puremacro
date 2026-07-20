"""Per-unit STL seasonal adjustment for panel data.

Why STL (not X-13ARIMA-SEATS): X-13 needs an external binary (`x13as`),
which we cannot rely on across user environments. STL ships with
`statsmodels` and gives a robust seasonal/trend/residual decomposition that
is sufficient for our diagnostic and panel-LP use cases. Counties below
the minimum-observations threshold (`min_obs`, default 24) are returned
as NaN for that variable so downstream regressions can drop them cleanly
without contaminating other units' decompositions.

Public API:
    deseasonalize(df, value_col, by, date_col, freq='Q') -> pd.Series
    residual_seasonality_F(series, freq='Q') -> (F, p, dof_num, dof_den)
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

# statsmodels.tsa.seasonal.STL is lazy-imported inside _stl_one (Pyodide contract).

logger = logging.getLogger(__name__)


_PERIOD = {"Q": 4, "M": 12}


def _stl_one(values: np.ndarray, period: int) -> np.ndarray:
    """Deseasonalize a 1-D series via STL. Returns trend + residual.

    Internal NaNs are linearly interpolated for the decomposition only;
    NaN positions in the input are restored to NaN in the output.
    """
    from statsmodels.tsa.seasonal import STL  # lazy: Pyodide contract

    s = pd.Series(values).copy()
    nan_mask = s.isna()
    if nan_mask.all():
        return values.astype(float)
    # STL needs a contiguous numeric series; interpolate internal NaNs and
    # forward/back-fill edges so the decomposition is well defined. Edge
    # imputations only affect the seasonal estimate, never the returned
    # values at originally-NaN positions.
    s_interp = s.interpolate(limit_direction="both")
    if s_interp.isna().any():
        return values.astype(float)
    try:
        res = STL(s_interp.values, period=period, robust=True).fit()
    except Exception as e:
        logger.debug("STL failed (%s); returning raw series", e)
        return values.astype(float)
    sa = s_interp.values - res.seasonal
    sa = np.where(nan_mask.values, np.nan, sa)
    return sa


def deseasonalize(
    df: pd.DataFrame,
    value_col: str,
    *,
    by: str,
    date_col: str = "date",
    freq: Literal["Q", "M"] = "Q",
    min_obs: int = 24,
) -> pd.Series:
    """Return a seasonally-adjusted Series aligned to ``df.index``.

    Groups by ``by`` (e.g. county_fips) and applies STL within each group.
    Groups with fewer than ``min_obs`` non-NaN observations are returned as
    NaN — the trend/seasonal split is not identified on short series.

    Parameters
    ----------
    df : DataFrame
    value_col : str
    by : str
        Unit column (e.g. 'county_fips' or 'state_fips').
    date_col : str
    freq : 'Q' or 'M'
    min_obs : int
        Minimum non-NaN observations required for STL fit.
    """
    if freq not in _PERIOD:
        raise ValueError(f"freq must be 'Q' or 'M', got {freq!r}")
    period = _PERIOD[freq]

    out = pd.Series(np.nan, index=df.index, dtype=float)
    df_sorted = df[[by, date_col, value_col]].copy()
    df_sorted["_orig_idx"] = df.index
    df_sorted = df_sorted.sort_values([by, date_col])

    for unit, sub in df_sorted.groupby(by, sort=False):
        v = sub[value_col].to_numpy(dtype=float)
        if np.isfinite(v).sum() < min_obs:
            continue
        sa = _stl_one(v, period=period)
        out.loc[sub["_orig_idx"].values] = sa

    return out


def residual_seasonality_F(
    series: pd.Series, *, freq: Literal["Q", "M"] = "Q"
) -> tuple[float, float, int, int]:
    """Test for residual seasonality in an SA series via one-way ANOVA on quarter.

    Returns (F, p_value, dof_num, dof_den). Computed on the second
    difference (so trend doesn't dominate F).
    """
    if freq not in _PERIOD:
        raise ValueError(f"freq must be 'Q' or 'M', got {freq!r}")
    period = _PERIOD[freq]
    s = pd.Series(series).dropna()
    if len(s) < 2 * period:
        return (np.nan, np.nan, 0, 0)
    diff = s.diff().diff().dropna()
    if len(diff) < 2 * period:
        return (np.nan, np.nan, 0, 0)
    # Reuse the existing date index modulo period
    idx_pos = np.arange(len(diff)) % period
    groups = [diff.values[idx_pos == k] for k in range(period)]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return (np.nan, np.nan, 0, 0)
    F, p = stats.f_oneway(*groups)
    dof_num = len(groups) - 1
    dof_den = sum(len(g) for g in groups) - len(groups)
    return (float(F), float(p), int(dof_num), int(dof_den))
