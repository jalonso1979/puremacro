"""Lag-augmented local projection (Plagborg-Møller-Wolf 2021).

Adds extra lags of x and y beyond what's needed to reduce omitted-
variable bias, which (per PMW 2021) makes the standard
heteroskedasticity-robust SE valid at every horizon h — no HAC, no
Bonferroni, and no horizon-dependent bandwidth. Recommended lag count
is ``p_aug = p + h`` (or just a fixed safe number > p).

This is the "modern default" for LP at long horizons.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .._linalg import inv_xtx
from ._results import LPResult


def _ols_eicker_huber(y: np.ndarray, X: np.ndarray) -> dict:
    """OLS with Eicker-Huber-White (HC0) heteroskedasticity-robust SE."""
    XtX_inv = inv_xtx(X, name="la_lp")
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    meat = (X * u[:, None]).T @ (X * u[:, None])
    V = XtX_inv @ meat @ XtX_inv
    return {"beta": beta, "se": np.sqrt(np.diag(V)), "vcov": V, "resid": u}


def la_lp(
    df: pd.DataFrame,
    y: str,
    x: str,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 4,
    extra_lags: int | None = None,
    controls: Sequence[str] | None = None,
    alpha: float = 0.10,
    *,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
) -> LPResult:
    """PMW (2021) lag-augmented LP with Eicker-Huber-White SE.

    Parameters
    ----------
    n_lags : int
        Baseline number of lags of x and y to include.
    extra_lags : int or None
        Additional lags beyond ``n_lags`` (PMW recommend p_aug ≥ p + h).
        If None, defaults to max(horizons), which makes coverage uniform
        across all reported horizons.
    """
    if lags is not None:
        n_lags = lags
    if horizon is not None:
        horizons = range(0, horizon + 1)
    if ci is not None:
        alpha = 1.0 - ci
    horizons = list(horizons)
    ctl = list(controls or [])
    z_crit = norm.ppf(1 - alpha / 2)
    H = max(horizons) if horizons else 0
    if extra_lags is None:
        extra_lags = H
    p_aug = n_lags + extra_lags

    rows = []
    for h in horizons:
        sub = df[[y, x] + ctl].copy()
        sub["__dy_h__"] = sub[y].shift(-h) - sub[y].shift(1)
        for lag in range(1, p_aug + 1):
            sub[f"__{x}_L{lag}__"] = sub[x].shift(lag)
            sub[f"__{y}_L{lag}__"] = sub[y].shift(lag)
        for c in ctl:
            for lag in range(1, n_lags + 1):
                sub[f"__{c}_L{lag}__"] = sub[c].shift(lag)
        sub = sub.dropna()
        if sub.empty:
            rows.append({"h": h, "beta": np.nan, "se": np.nan,
                         "lo": np.nan, "hi": np.nan, "p_aug": p_aug})
            continue
        n = len(sub)
        cols = [np.ones(n), sub[x].values]
        for lag in range(1, p_aug + 1):
            cols.append(sub[f"__{x}_L{lag}__"].values)
            cols.append(sub[f"__{y}_L{lag}__"].values)
        for c in ctl:
            for lag in range(1, n_lags + 1):
                cols.append(sub[f"__{c}_L{lag}__"].values)
            cols.append(sub[c].values)
        X = np.column_stack(cols)
        out = _ols_eicker_huber(sub["__dy_h__"].values, X)
        beta_h = float(out["beta"][1])
        se_h = float(out["se"][1])
        rows.append({
            "h": h,
            "beta": beta_h,
            "se": se_h,
            "lo": beta_h - z_crit * se_h,
            "hi": beta_h + z_crit * se_h,
            "p_aug": p_aug,
        })
    res = LPResult(rows)
    if "h" in res.columns:
        res.index = res["h"]
    res.y_name = str(y)
    res.x_name = str(x)
    res.method = "la_lp"
    return res


__all__ = ["la_lp"]
