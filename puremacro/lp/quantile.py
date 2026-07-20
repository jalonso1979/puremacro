"""Quantile local projection (Adrian-Boyarchenko-Giannone 2019).

For each horizon h and each quantile tau, estimate

    Q_tau[ y_{t+h} - y_{t-1} | x_t, controls ] = alpha_h + beta_h x_t + ...

via standard quantile regression (Koenker-Bassett 1978). Bands come
from a residual block bootstrap by default.

Quantile regression is solved as a linear program through
``scipy.optimize.linprog`` (HiGHS), so this stays pure-numpy/scipy and
runs in Pyodide.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def _qreg(y: np.ndarray, X: np.ndarray, tau: float) -> np.ndarray:
    """Quantile regression of y on X at quantile tau.

    Solves
        min  tau * 1' u_pos + (1 - tau) * 1' u_neg
        s.t. y = X (b_pos - b_neg) + u_pos - u_neg, all >= 0
    """
    n, k = X.shape
    c = np.concatenate([
        np.zeros(2 * k),
        np.full(n, tau),
        np.full(n, 1.0 - tau),
    ])
    A_eq = np.hstack([X, -X, np.eye(n), -np.eye(n)])
    b_eq = y
    bounds = [(0, None)] * (2 * k + 2 * n)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return np.full(k, np.nan)
    z = res.x
    return z[:k] - z[k:2 * k]


def lp_quantile(
    df: pd.DataFrame,
    y: str,
    x: str,
    quantiles: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 0.90),
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    controls: Sequence[str] | None = None,
    n_boot: int = 200,
    alpha: float = 0.10,
    seed: int = 0,
) -> pd.DataFrame:
    """Adrian-Boyarchenko-Giannone "vulnerable growth"-style quantile LP.

    Returns
    -------
    pd.DataFrame with columns [h, tau, beta, lo, hi].
    """
    horizons = list(horizons)
    ctl = list(controls or [])
    rng = np.random.default_rng(seed)
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
            for tau in quantiles:
                rows.append({"h": h, "tau": tau, "beta": np.nan,
                             "lo": np.nan, "hi": np.nan})
            continue
        n = len(sub)
        cols = [np.ones(n), sub[x].values]
        for lag in range(1, n_lags + 1):
            cols.append(sub[f"__{x}_L{lag}__"].values)
            cols.append(sub[f"__{y}_L{lag}__"].values)
            for c in ctl:
                cols.append(sub[f"__{c}_L{lag}__"].values)
        for c in ctl:
            cols.append(sub[c].values)
        X_mat = np.column_stack(cols)
        y_vec = sub["__dy_h__"].values

        for tau in quantiles:
            beta_hat = _qreg(y_vec, X_mat, tau)
            beta_x = float(beta_hat[1])
            boot = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.integers(0, n, size=n)
                bb = _qreg(y_vec[idx], X_mat[idx], tau)
                boot[b] = bb[1]
            lo = float(np.nanpercentile(boot, 100 * alpha / 2))
            hi = float(np.nanpercentile(boot, 100 * (1 - alpha / 2)))
            rows.append({"h": h, "tau": tau, "beta": beta_x,
                         "lo": lo, "hi": hi})

    return pd.DataFrame(rows)


__all__ = ["lp_quantile"]
