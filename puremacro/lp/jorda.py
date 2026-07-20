"""Single-country Jordà (2005) local projection with Newey-West HAC SE.

Pure-numpy port of src/lp/lp_jorda.py — replaces statsmodels.OLS w/ HAC.
Uses puremacro.inference._ols_helpers.ols_hac under the hood.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..inference._ols_helpers import ols_hac


def lp_hac(
    df: pd.DataFrame,
    y: str,
    x: str,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    controls: Sequence[str] | None = None,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Estimate β_h from
        y_{t+h} - y_{t-1} = α_h + β_h x_t + Σ γ_l z_{t-l} + ε_{t,h}

    HAC bandwidth = h + 1 (Plagborg-Møller-Wolf 2021 recommendation).
    Bands at level (1 - alpha) — default 90 %.

    Returns
    -------
    pd.DataFrame with columns [h, beta, se, t, lo, hi] (one row per horizon).
    """
    horizons = list(horizons)
    ctl = list(controls or [])
    z_crit = norm.ppf(1 - alpha / 2)

    rows = []
    for h in horizons:
        sub = df[[y, x] + ctl].copy()
        sub["dy_h"] = sub[y].shift(-h) - sub[y].shift(1)
        for lag in range(1, n_lags + 1):
            sub[f"{x}_L{lag}"] = sub[x].shift(lag)
            sub[f"{y}_L{lag}"] = sub[y].shift(lag)
            for c in ctl:
                sub[f"{c}_L{lag}"] = sub[c].shift(lag)
        sub = sub.dropna()
        if sub.empty:
            rows.append({"h": h, "beta": np.nan, "se": np.nan,
                         "t": np.nan, "lo": np.nan, "hi": np.nan})
            continue
        n = len(sub)
        regressors = [np.ones(n), sub[x].values]
        for lag in range(1, n_lags + 1):
            regressors.append(sub[f"{x}_L{lag}"].values)
            regressors.append(sub[f"{y}_L{lag}"].values)
            for c in ctl:
                regressors.append(sub[f"{c}_L{lag}"].values)
        for c in ctl:
            regressors.append(sub[c].values)
        X = np.column_stack(regressors)
        out = ols_hac(sub["dy_h"].values, X, lags=h + 1)
        beta_h = float(out["beta"][1])  # x_t coefficient (col 1 after const)
        se_h = float(out["se"][1])
        rows.append({
            "h": h,
            "beta": beta_h,
            "se": se_h,
            "t": beta_h / se_h if se_h > 0 else np.nan,
            "lo": beta_h - z_crit * se_h,
            "hi": beta_h + z_crit * se_h,
        })
    return pd.DataFrame(rows)


__all__ = ["lp_hac"]
