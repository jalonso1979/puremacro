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
    df: pd.DataFrame | np.ndarray,
    y: str | np.ndarray | None = None,
    x: str | None = None,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    controls: Sequence[str] | np.ndarray | None = None,
    alpha: float = 0.10,
    *,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
) -> pd.DataFrame:
    """Estimate local projection with Newey-West HAC standard errors.

    Accepts either DataFrame input `lp_hac(df, y='y_col', x='x_col', ...)`
    or 1D array/series input `lp_hac(y_series, shock_series, horizons=..., lags=...)`.

    HAC bandwidth = h + 1 (Plagborg-Møller-Wolf 2021 recommendation).
    Bands at level (1 - alpha) — default 90 %.

    Returns
    -------
    LPResult (subclass of pd.DataFrame) with columns [h, beta, se, t, lo, hi] indexed by h.
    """
    if lags is not None:
        n_lags = lags
    if horizon is not None:
        horizons = range(0, horizon + 1)
    if ci is not None:
        alpha = 1.0 - ci
    horizons = list(horizons)
    z_crit = norm.ppf(1 - alpha / 2)

    if not (isinstance(df, pd.DataFrame) and isinstance(y, str) and isinstance(x, str)):
        y_vals = np.asarray(df, float)
        x_vals = np.asarray(y, float)
        data = {"y": y_vals, "x": x_vals}
        ctl_names = []
        if controls is not None:
            C = np.asarray(controls, float)
            if C.ndim == 1:
                data["c0"] = C
                ctl_names.append("c0")
            elif C.ndim == 2:
                for j in range(C.shape[1]):
                    cname = f"c{j}"
                    data[cname] = C[:, j]
                    ctl_names.append(cname)
        df = pd.DataFrame(data)
        y = "y"
        x = "x"
        controls = ctl_names if ctl_names else None

    ctl = list(controls or [])
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
    from ._results import LPResult

    res = LPResult(rows)
    res.index = res["h"]
    res.y_name = str(y)
    res.x_name = str(x)
    res.method = "LP-HAC"
    return res


__all__ = ["lp_hac"]
