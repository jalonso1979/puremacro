"""Pesaran (2006) Common Correlated Effects panel LP.

Augments the panel LP regression with cross-sectional means of y and x
to absorb common factors. The CCE-pooled (CCEP) estimator is just OLS
on the augmented model with entity FE.

References:
- Pesaran (2006). Estimation and inference in large heterogeneous panels
  with a multifactor error structure. Econometrica 74(4): 967-1012.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from ._results import LPResult


def _entity_fe_ols(sub: pd.DataFrame, y_col: str, x_cols: Sequence[str],
                   entity_level: str) -> dict:
    """Entity-FE-only within OLS with cluster-robust SE.

    The CCE estimator already controls for time via the cross-sectional
    means; adding time FE on top creates perfect collinearity with those
    proxies. We therefore demean only by entity (Pesaran 2006, eq. 12).
    """
    arr = sub[[y_col] + list(x_cols)].values.astype(float)
    entity_idx = sub.index.get_level_values(entity_level).values
    ent_means = pd.DataFrame(arr).groupby(entity_idx).transform("mean").values
    arr = arr - ent_means
    y_d = arr[:, 0]
    X_d = arr[:, 1:]
    beta, *_ = np.linalg.lstsq(X_d, y_d, rcond=None)
    u = y_d - X_d @ beta
    # Cluster-robust SE by entity
    score = X_d * u[:, None]
    S = np.zeros((X_d.shape[1], X_d.shape[1]))
    for c in np.unique(entity_idx):
        s_c = score[entity_idx == c].sum(axis=0)
        S += np.outer(s_c, s_c)
    XtX = X_d.T @ X_d
    XtX_inv = np.linalg.pinv(XtX)
    vcov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    return {"beta": beta, "se": se}


def cce_panel_lp(
    df_wide: pd.DataFrame,
    y: str,
    x: str,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    controls: Sequence[str] = (),
    alpha: float = 0.10,
    entity_level: str = "code",
    time_level: str = "date",
    *,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
) -> LPResult:
    """Estimate panel LP β_h with Pesaran-CCE common-factor correction.

    Augments the regression with the time-by-time cross-sectional means
    of y and x (and any controls) so that common factors entering the
    error term are absorbed. With sufficient cross-section, the CCE
    estimator is consistent under unobserved factor structure.

    Uses entity FE only (not two-way FE): the cross-sectional means
    already serve as proxies for the common time component, so adding
    time dummies would create exact collinearity with those proxies.

    Returns LPResult with [h, beta, se, lo, hi].
    """
    if lags is not None:
        n_lags = lags
    if horizon is not None:
        horizons = range(0, horizon + 1)
    if ci is not None:
        alpha = 1.0 - ci
    horizons = list(horizons); ctl = list(controls)
    z_crit = norm.ppf(1 - alpha / 2)

    df = df_wide.copy()
    df.index = df.index.set_names([entity_level, time_level])
    df = df.sort_index()
    grouped_e = df.groupby(level=entity_level)
    # Pre-compute lags
    for lag in range(1, n_lags + 1):
        df[f"__{x}_L{lag}__"] = grouped_e[x].shift(lag)
        df[f"__{y}_L{lag}__"] = grouped_e[y].shift(lag)
        for c in ctl:
            df[f"__{c}_L{lag}__"] = grouped_e[c].shift(lag)
    # Cross-sectional means by time (CCE augmentation)
    grouped_t = df.groupby(level=time_level)
    df["__cs_mean_y__"] = grouped_t[y].transform("mean")
    df["__cs_mean_x__"] = grouped_t[x].transform("mean")
    for c in ctl:
        df[f"__cs_mean_{c}__"] = grouped_t[c].transform("mean")

    rows = []
    for h in horizons:
        col_dy = f"__dy_h{h}__"
        df[col_dy] = grouped_e[y].shift(-h) - grouped_e[y].shift(1)
        x_cols = [x]
        for lag in range(1, n_lags + 1):
            x_cols += [f"__{x}_L{lag}__", f"__{y}_L{lag}__"]
            for c in ctl:
                x_cols.append(f"__{c}_L{lag}__")
        for c in ctl:
            x_cols.append(c)
        # Add the CCE common-factor proxies (cross-sectional means) as regressors
        x_cols.extend(["__cs_mean_y__", "__cs_mean_x__"])
        for c in ctl:
            x_cols.append(f"__cs_mean_{c}__")
        sub = df[[col_dy] + x_cols].dropna()
        if sub.empty:
            rows.append({"h": h, "beta": np.nan, "se": np.nan,
                         "lo": np.nan, "hi": np.nan})
            continue
        out = _entity_fe_ols(sub, y_col=col_dy, x_cols=x_cols,
                              entity_level=entity_level)
        beta_h = float(out["beta"][0]); se_h = float(out["se"][0])
        rows.append({
            "h": h, "beta": beta_h, "se": se_h,
            "lo": beta_h - z_crit * se_h,
            "hi": beta_h + z_crit * se_h,
        })
    res = LPResult(rows)
    if "h" in res.columns:
        res.index = res["h"]
    res.y_name = str(y)
    res.x_name = str(x)
    res.method = "cce_panel_lp"
    return res


__all__ = ["cce_panel_lp"]
