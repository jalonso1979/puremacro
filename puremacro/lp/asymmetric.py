"""Tenreyro-Thwaites (2016) asymmetric LP — separate β_h^+ and β_h^−.

Two interpretations:
- "shock_sign": split based on sign of x_t (positive shocks vs negative shocks)
- "outcome_state": condition on the sign of a state variable

For v1 we implement the first: x_t enters as x_t * 1{x_t > 0} (positive
component) and x_t * 1{x_t < 0} (negative component). The two
coefficients identify the response to positive and negative shocks
separately.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..inference._ols_helpers import ols_hac
from ._results import LPResult


def lp_asymmetric(
    df: pd.DataFrame,
    y: str,
    x: str,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    controls: Sequence[str] | None = None,
    alpha: float = 0.10,
    *,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
) -> LPResult:
    """Estimate
        y_{t+h} - y_{t-1} = α_h
                         + β_h^+ x_t · 1{x_t > 0}
                         + β_h^- x_t · 1{x_t < 0}
                         + Σ γ_l z_{t-l} + ε
    via Newey-West HAC. Bands at level (1-alpha).

    Returns LPResult with columns
        [h, beta_pos, se_pos, lo_pos, hi_pos,
            beta_neg, se_neg, lo_neg, hi_neg].
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
    rows = []
    for h in horizons:
        sub = df[[y, x] + ctl].copy()
        sub["__dy_h__"] = sub[y].shift(-h) - sub[y].shift(1)
        for lag in range(1, n_lags + 1):
            sub[f"__{x}_L{lag}__"] = sub[x].shift(lag)
            sub[f"__{y}_L{lag}__"] = sub[y].shift(lag)
            for c in ctl:
                sub[f"__{c}_L{lag}__"] = sub[c].shift(lag)
        sub = sub.dropna()
        if sub.empty:
            rows.append({"h": h, "beta_pos": np.nan, "se_pos": np.nan,
                         "lo_pos": np.nan, "hi_pos": np.nan,
                         "beta_neg": np.nan, "se_neg": np.nan,
                         "lo_neg": np.nan, "hi_neg": np.nan})
            continue
        x_arr = sub[x].values
        x_pos = x_arr * (x_arr > 0).astype(float)
        x_neg = x_arr * (x_arr < 0).astype(float)
        n = len(sub)
        regressors = [np.ones(n), x_pos, x_neg]
        for lag in range(1, n_lags + 1):
            regressors.append(sub[f"__{x}_L{lag}__"].values)
            regressors.append(sub[f"__{y}_L{lag}__"].values)
            for c in ctl:
                regressors.append(sub[f"__{c}_L{lag}__"].values)
        for c in ctl:
            regressors.append(sub[c].values)
        X = np.column_stack(regressors)
        out = ols_hac(sub["__dy_h__"].values, X, lags=h + 1)
        beta_pos = float(out["beta"][1]); se_pos = float(out["se"][1])
        beta_neg = float(out["beta"][2]); se_neg = float(out["se"][2])
        rows.append({
            "h": h,
            "beta_pos": beta_pos, "se_pos": se_pos,
            "lo_pos": beta_pos - z_crit * se_pos,
            "hi_pos": beta_pos + z_crit * se_pos,
            "beta_neg": beta_neg, "se_neg": se_neg,
            "lo_neg": beta_neg - z_crit * se_neg,
            "hi_neg": beta_neg + z_crit * se_neg,
        })
    res = LPResult(rows)
    if "h" in res.columns:
        res.index = res["h"]
    res.y_name = str(y)
    res.x_name = str(x)
    res.method = "LP-asymmetric"
    return res


__all__ = ["lp_asymmetric"]
