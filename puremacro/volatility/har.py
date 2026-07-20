"""Heterogeneous Autoregressive (HAR) realised-volatility model.

Corsi (2009) HAR-RV regresses one-step-ahead realised volatility on
daily, weekly, and monthly averages of past RV:

    RV_{t+1} = β_0 + β_d RV_t + β_w RV_{t-5..t}^{(5)} + β_m RV_{t-22..t}^{(22)} + ε_{t+1}

The HAR model captures the long-memory-like persistence of RV without
fractional integration: the cascade of horizons (1 / 5 / 22 trading days)
mimics how short-, medium-, and long-horizon traders update their
volatility expectations.

References
----------
Corsi, F. (2009). A simple approximate long-memory model of realized
    volatility. Journal of Financial Econometrics 7(2), 174-196.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..inference._ols_helpers import ols_hac


def har_rv(
    rv: np.ndarray | pd.Series,
    *,
    horizons: Iterable[int] = (1, 5, 22),
    nw_lags: int = 5,
) -> dict:
    """Fit Corsi's HAR-RV with Newey-West HAC SE.

    Parameters
    ----------
    rv : 1-D array-like
        Realised-volatility series (any frequency; daily is canonical).
    horizons : iterable of int, default ``(1, 5, 22)``
        Averaging windows for the daily / weekly / monthly components.
        The first horizon is treated as the daily lag (no averaging).
    nw_lags : int
        Bandwidth for Newey-West SE on the OLS regression.

    Returns
    -------
    dict
        Keys: ``beta`` (intercept + per-horizon coefficients), ``se``,
        ``t``, ``vcov``, ``residuals``, ``n_obs``, ``horizons``.
    """
    rv = np.asarray(rv, dtype=float).ravel()
    horizons = list(horizons)
    if not horizons:
        raise ValueError("horizons must be non-empty")
    max_h = max(horizons)
    if rv.shape[0] <= max_h + 1:
        raise ValueError(
            f"need at least {max_h + 2} obs for horizons up to {max_h}"
        )

    T = rv.shape[0]
    # Build feature columns: for horizon h, the regressor at time t is
    # the mean of rv[t - h + 1 .. t]. y is rv[t + 1].
    y = rv[max_h:]                    # (T - max_h,)
    rows_X = [np.ones_like(y)]
    for h in horizons:
        col = np.array([
            rv[t - h + 1: t + 1].mean()
            for t in range(max_h - 1, T - 1)
        ])
        rows_X.append(col)
    X = np.column_stack(rows_X)
    out = ols_hac(y, X, lags=nw_lags)
    out["horizons"] = tuple(horizons)
    return out


__all__ = ["har_rv"]
