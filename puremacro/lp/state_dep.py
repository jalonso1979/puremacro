"""State-dependent LP (Granger-Teräsvirta logistic / threshold).

Granger-Teräsvirta (1993) logistic transition:
    F(z_t) = 1 / (1 + exp(-γ z_t))
Auerbach-Gorodnichenko (2013) — z_t is a standardized regime indicator.

Returns separate β_h^L (low-regime, weighted by 1-F) and β_h^H
(high-regime, weighted by F) coefficients with their HAC SE.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..inference._ols_helpers import ols_hac


def _logistic(z: np.ndarray, gamma: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-gamma * z))


def _logistic_threshold(z: np.ndarray, gamma: float, threshold: float) -> np.ndarray:
    """Logistic with explicit threshold c: F(z) = 1 / (1 + exp(-γ(z - c)))."""
    return 1.0 / (1.0 + np.exp(-gamma * (z - threshold)))


def lp_state_dep(
    df: pd.DataFrame,
    y: str,
    x: str,
    state: str,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    transition: str = "logistic",
    gamma: float = 3.0,
    threshold: float = 0.0,
    controls: Sequence[str] | None = None,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Estimate
        y_{t+h} - y_{t-1} = α_h
                         + β_h^H F(z_t) x_t
                         + β_h^L (1-F(z_t)) x_t
                         + Σ γ_l z_{t-l} + ε_{t,h}
    where F(·) is logistic in standardized state z_t, or a threshold
    indicator I{z_t > threshold} when transition="threshold".

    Returns DataFrame with columns
        h, beta_H, se_H, lo_H, hi_H, beta_L, se_L, lo_L, hi_L.
    """
    horizons = list(horizons)
    ctl = list(controls or [])
    z_crit = norm.ppf(1 - alpha / 2)

    rows = []
    for h in horizons:
        sub = df[[y, x, state] + ctl].copy()
        sub["__dy_h__"] = sub[y].shift(-h) - sub[y].shift(1)
        for lag in range(1, n_lags + 1):
            sub[f"__{x}_L{lag}__"] = sub[x].shift(lag)
            sub[f"__{y}_L{lag}__"] = sub[y].shift(lag)
            for c in ctl:
                sub[f"__{c}_L{lag}__"] = sub[c].shift(lag)
        sub = sub.dropna()
        if sub.empty:
            rows.append({"h": h, "beta_H": np.nan, "se_H": np.nan,
                         "lo_H": np.nan, "hi_H": np.nan,
                         "beta_L": np.nan, "se_L": np.nan,
                         "lo_L": np.nan, "hi_L": np.nan})
            continue
        # Standardize state
        s_arr = sub[state].values
        s_std = (s_arr - s_arr.mean()) / (s_arr.std(ddof=0) if s_arr.std(ddof=0) > 0 else 1.0)
        if transition == "logistic":
            F = _logistic(s_std, gamma)
        elif transition == "threshold":
            F = (s_std > threshold).astype(float)
        else:
            raise ValueError(f"unknown transition {transition!r}")

        n = len(sub)
        regressors = [
            np.ones(n),
            F * sub[x].values,            # high-regime
            (1.0 - F) * sub[x].values,    # low-regime
        ]
        for lag in range(1, n_lags + 1):
            regressors.append(sub[f"__{x}_L{lag}__"].values)
            regressors.append(sub[f"__{y}_L{lag}__"].values)
            for c in ctl:
                regressors.append(sub[f"__{c}_L{lag}__"].values)
        for c in ctl:
            regressors.append(sub[c].values)
        X = np.column_stack(regressors)
        out = ols_hac(sub["__dy_h__"].values, X, lags=h + 1)
        beta_H = float(out["beta"][1]); se_H = float(out["se"][1])
        beta_L = float(out["beta"][2]); se_L = float(out["se"][2])
        rows.append({
            "h": h,
            "beta_H": beta_H, "se_H": se_H,
            "lo_H": beta_H - z_crit * se_H, "hi_H": beta_H + z_crit * se_H,
            "beta_L": beta_L, "se_L": se_L,
            "lo_L": beta_L - z_crit * se_L, "hi_L": beta_L + z_crit * se_L,
        })
    return pd.DataFrame(rows)


def lp_smooth_transition_irf(
    df: pd.DataFrame,
    y: str,
    x: str,
    state_var: str,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    gamma: float = 5.0,
    threshold: float = 0.0,
    controls: Sequence[str] | None = None,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Granger-Teräsvirta smooth-transition LP (Auerbach-Gorodnichenko style).

    Estimates, for each horizon h:

        y_{t+h} = alpha_h
                + beta_h^high * F(z_t) * x_t
                + beta_h^low  * (1 - F(z_t)) * x_t
                + lag controls + eps_{t,h}

    where F(z_t) = 1 / (1 + exp(-gamma * (z_t - threshold))) is the logistic
    transition function applied to the standardized ``state_var``.

    Unlike the hard-threshold ``lp_state_dep``, the smooth-transition
    specification allows every observation to contribute to both regimes with
    weight proportional to the continuous transition function.

    Parameters
    ----------
    df : pd.DataFrame
        Wide DataFrame with one row per period.
    y : str
        Outcome variable column name.
    x : str
        Shock / treatment variable column name.
    state_var : str
        Transition variable column name (e.g. lagged output gap, uncertainty
        index). Standardized internally so ``gamma`` and ``threshold`` operate
        on a z-score scale.
    horizons : Iterable[int]
        LP horizons to estimate (e.g. ``range(0, 13)``).
    n_lags : int
        Number of lags of ``x`` and ``y`` to include as controls.
    gamma : float
        Transition speed — steepness of the logistic function. Larger values
        produce a sharper (more threshold-like) transition.
    threshold : float
        Logistic threshold ``c`` on the standardized scale. Default 0.0.
    controls : sequence of str, optional
        Additional control column names (lagged once through ``n_lags``).
    alpha : float
        Two-sided significance level for the HAC confidence intervals.

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with columns:
        ``h``, ``beta_high``, ``se_high``, ``lo_high``, ``hi_high``,
        ``beta_low``, ``se_low``, ``lo_low``, ``hi_low``.

        ``beta_high`` is the coefficient on F(z_t)*x_t (high-state regime);
        ``beta_low`` is the coefficient on (1-F(z_t))*x_t (low-state regime).
    """
    import statsmodels.api as sm  # lazy: Pyodide contract

    horizons = list(horizons)
    ctl = list(controls or [])
    z_crit = norm.ppf(1 - alpha / 2)

    rows = []
    for h in horizons:
        sub = df[[y, x, state_var] + ctl].copy()
        sub["__dep__"] = sub[y].shift(-h)
        for lag in range(1, n_lags + 1):
            sub[f"__{x}_L{lag}__"] = sub[x].shift(lag)
            sub[f"__{y}_L{lag}__"] = sub[y].shift(lag)
            for c in ctl:
                sub[f"__{c}_L{lag}__"] = sub[c].shift(lag)
        sub = sub.dropna()
        if sub.empty:
            rows.append({
                "h": h,
                "beta_high": np.nan, "se_high": np.nan,
                "lo_high": np.nan, "hi_high": np.nan,
                "beta_low": np.nan, "se_low": np.nan,
                "lo_low": np.nan, "hi_low": np.nan,
            })
            continue

        # Standardize transition variable
        tv = sub[state_var].values
        tv_std = (tv - tv.mean()) / (tv.std(ddof=0) if tv.std(ddof=0) > 0 else 1.0)
        F = _logistic_threshold(tv_std, gamma, threshold)   # shape (n,)

        n = len(sub)
        regressors = [
            np.ones(n),
            F * sub[x].values,             # beta_high coefficient
            (1.0 - F) * sub[x].values,     # beta_low coefficient
        ]
        for lag in range(1, n_lags + 1):
            regressors.append(sub[f"__{x}_L{lag}__"].values)
            regressors.append(sub[f"__{y}_L{lag}__"].values)
            for c in ctl:
                regressors.append(sub[f"__{c}_L{lag}__"].values)
        for c in ctl:
            regressors.append(sub[c].values)
        X = np.column_stack(regressors)

        try:
            bw = max(1, int(np.ceil((h + 1) ** (2 / 9) * (n ** (2 / 9)))))
            ols = sm.OLS(sub["__dep__"].values, X).fit()
            hac = ols.get_robustcov_results(
                cov_type="HAC", maxlags=bw, use_correction=True
            )
            coef = hac.params
            cov = hac.cov_params()
            b_high = float(coef[1])
            b_low = float(coef[2])
            se_high = float(np.sqrt(max(cov[1, 1], 0.0)))
            se_low = float(np.sqrt(max(cov[2, 2], 0.0)))
        except Exception:
            b_high = b_low = se_high = se_low = np.nan

        rows.append({
            "h": h,
            "beta_high": b_high, "se_high": se_high,
            "lo_high": b_high - z_crit * se_high,
            "hi_high": b_high + z_crit * se_high,
            "beta_low": b_low, "se_low": se_low,
            "lo_low": b_low - z_crit * se_low,
            "hi_low": b_low + z_crit * se_low,
        })

    return pd.DataFrame(rows)


__all__ = ["lp_state_dep", "lp_smooth_transition_irf"]
