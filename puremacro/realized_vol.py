"""Realized variance + bipower variation + Corsi (2009) HAR-RV regression.

Given high-frequency intra-day returns, the daily realized variance is

    RV_t = sum_{i=1..M} r_{t,i}^2

Bipower variation (Barndorff-Nielsen-Shephard 2004) is jump-robust:

    BV_t = (pi/2) * sum_{i=2..M} |r_{t,i-1}| |r_{t,i}|

The relative jump component is RJ_t = max(0, RV_t - BV_t) / RV_t.

The Corsi (2009) HAR-RV regression decomposes daily volatility into
short / mid / long memory components:

    log RV_t = beta_0 + beta_d log RV_{t-1}
                      + beta_w mean(log RV_{t-2..t-5})
                      + beta_m mean(log RV_{t-6..t-22})
                      + e_t

References
----------
Andersen, T.G. and Bollerslev, T. (1998). Answering the skeptics:
    Yes, standard volatility models do provide accurate forecasts. IER.
Barndorff-Nielsen, O.E. and Shephard, N. (2004). Power and bipower
    variation with stochastic volatility and jumps. JFE 1.
Corsi, F. (2009). A simple approximate long-memory model of realized
    volatility. JFE 7(2), 174-196.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .inference._ols_helpers import ols_hac


def realized_variance(returns_intraday: np.ndarray) -> float:
    """RV_t = sum(r_i^2) for one day's intra-day returns."""
    return float(np.sum(np.asarray(returns_intraday, dtype=float) ** 2))


def bipower_variation(returns_intraday: np.ndarray) -> float:
    """BV_t = (pi/2) * sum_{i>=2} |r_{i-1}| |r_i|."""
    r = np.asarray(returns_intraday, dtype=float)
    if len(r) < 2:
        return 0.0
    return float((np.pi / 2.0) * np.sum(np.abs(r[:-1]) * np.abs(r[1:])))


def realized_jump(rv: float, bv: float) -> float:
    """Relative jump RJ = max(0, RV - BV) / RV."""
    if rv <= 0:
        return 0.0
    return max(0.0, (rv - bv) / rv)


def daily_rv_panel(returns_intraday_per_day: list[np.ndarray]) -> pd.DataFrame:
    """Compute (RV, BV, RJ) per day from a list of intra-day return arrays."""
    rows = []
    for day in returns_intraday_per_day:
        rv = realized_variance(day)
        bv = bipower_variation(day)
        rj = realized_jump(rv, bv)
        rows.append({"RV": rv, "BV": bv, "RJ": rj})
    return pd.DataFrame(rows)


def har_rv(rv_series: np.ndarray, log_transform: bool = True,
           hac_lags: int | None = None) -> dict:
    """Estimate the Corsi HAR-RV regression with HAC SE.

    Parameters
    ----------
    rv_series : (T,) ndarray of daily realized variances (T > 30).
    log_transform : if True, regress log RV on log lags (Corsi default).
    hac_lags : Newey-West truncation lag. If None, the Newey-West (1994)
        rule ``floor(4 * (n / 100) ** (2 / 9))`` is used with ``n = T - 22``,
        the number of regression observations (the first 22 days are lost
        to the monthly lag). ``hac_lags=0`` gives White (HC0) SEs.

    Returns
    -------
    dict with keys
        ``intercept``, ``beta_d``, ``beta_w``, ``beta_m`` : float
            OLS coefficients (β_0 in the docstring formula is ``intercept``).
        ``se_intercept``, ``se_d``, ``se_w``, ``se_m`` : float
            Newey-West HAC standard errors.
        ``R2`` : float
        ``fitted``, ``residuals`` : (T-22,) ndarrays
        ``design`` : (T-22, 4) regressor matrix ``[1, x_d, x_w, x_m]``.
        ``log_transform`` : bool (echo of the argument)
        ``hac_lags`` : int (the truncation lag actually used)
    """
    rv = np.asarray(rv_series, dtype=float).ravel()
    if log_transform:
        rv = np.log(np.maximum(rv, 1e-12))
    T = len(rv)
    if T <= 30:
        raise ValueError("HAR-RV needs at least ~30 daily observations.")
    rows = []
    targets = []
    for t in range(22, T):
        x_d = rv[t - 1]
        x_w = rv[t - 5:t - 1].mean()    # 4 obs (t-5..t-2 inclusive)
        x_m = rv[t - 22:t - 5].mean()   # 17 obs
        rows.append([1.0, x_d, x_w, x_m])
        targets.append(rv[t])
    X = np.array(rows)
    y = np.array(targets)
    if hac_lags is None:
        hac_lags = int(np.floor(4 * (len(y) / 100) ** (2 / 9)))
    out = ols_hac(y, X, lags=hac_lags)
    beta = out["beta"]
    se = out["se"]
    fitted = X @ beta
    resid = y - fitted
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    return {
        "intercept": float(beta[0]), "se_intercept": float(se[0]),
        "beta_d": float(beta[1]), "se_d": float(se[1]),
        "beta_w": float(beta[2]), "se_w": float(se[2]),
        "beta_m": float(beta[3]), "se_m": float(se[3]),
        "fitted": fitted,
        "residuals": resid,
        "R2": 1.0 - rss / tss if tss > 0 else 0.0,
        "design": X,
        "log_transform": log_transform,
        "hac_lags": int(hac_lags),
    }


__all__ = ["realized_variance", "bipower_variation", "realized_jump",
           "daily_rv_panel", "har_rv"]
