"""Single-country LP-IV (Stock-Watson 2018; Plagborg-Møller-Wolf 2021).

First stage: x_t = π_z z_t + π'  W_t + ν_t.
Second stage: y_{t+h} - y_{t-1} = α_h + β_h x_t + γ' W_t + ε_{t,h}
              with x_t replaced by its first-stage projection x̂_t.
HAC SE via Newey-West with bandwidth L = h + 1.

`first_stage_f` is the squared t-stat on the instrument's coefficient
in the first stage (exact-identification single-instrument case).
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..inference._ols_helpers import ols_hac


def lp_iv(
    df: pd.DataFrame,
    y: str,
    x: str,
    z: str,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    controls: Sequence[str] | None = None,
    alpha: float = 0.10,
) -> pd.DataFrame:
    horizons = list(horizons)
    ctl = list(controls or [])
    z_crit = norm.ppf(1 - alpha / 2)

    rows = []
    for h in horizons:
        sub = df[[y, x, z] + ctl].copy()
        sub["dy_h"] = sub[y].shift(-h) - sub[y].shift(1)
        for lag in range(1, n_lags + 1):
            sub[f"{x}_L{lag}"] = sub[x].shift(lag)
            sub[f"{y}_L{lag}"] = sub[y].shift(lag)
            for c in ctl:
                sub[f"{c}_L{lag}"] = sub[c].shift(lag)
        sub = sub.dropna()
        if sub.empty:
            rows.append({"h": h, "beta": np.nan, "se": np.nan, "t": np.nan,
                         "lo": np.nan, "hi": np.nan, "first_stage_f": np.nan})
            continue

        n = len(sub)
        # First-stage W matrix: const, z_t, lags of x and y, lags of controls, contemporaneous controls
        W = [np.ones(n), sub[z].values]
        for lag in range(1, n_lags + 1):
            W.append(sub[f"{x}_L{lag}"].values)
            W.append(sub[f"{y}_L{lag}"].values)
            for c in ctl:
                W.append(sub[f"{c}_L{lag}"].values)
        for c in ctl:
            W.append(sub[c].values)
        W_mat = np.column_stack(W)

        # First stage: x ~ z + W (excluding the instrument from W, we keep
        # the instrument as the second column above).
        fs = ols_hac(sub[x].values, W_mat, lags=h + 1)
        x_hat = W_mat @ fs["beta"]
        # First-stage F (single-instrument exactly-identified — use t² on z).
        first_stage_f = float((fs["beta"][1] / fs["se"][1]) ** 2) if fs["se"][1] > 0 else np.nan

        # Second stage: replace x with x_hat; same controls.
        X2 = [np.ones(n), x_hat]
        for lag in range(1, n_lags + 1):
            X2.append(sub[f"{x}_L{lag}"].values)
            X2.append(sub[f"{y}_L{lag}"].values)
            for c in ctl:
                X2.append(sub[f"{c}_L{lag}"].values)
        for c in ctl:
            X2.append(sub[c].values)
        X2_mat = np.column_stack(X2)
        out = ols_hac(sub["dy_h"].values, X2_mat, lags=h + 1)
        beta_h = float(out["beta"][1])
        se_h = float(out["se"][1])
        rows.append({
            "h": h,
            "beta": beta_h,
            "se": se_h,
            "t": beta_h / se_h if se_h > 0 else np.nan,
            "lo": beta_h - z_crit * se_h,
            "hi": beta_h + z_crit * se_h,
            "first_stage_f": first_stage_f,
        })
    return pd.DataFrame(rows)


__all__ = ["lp_iv"]
