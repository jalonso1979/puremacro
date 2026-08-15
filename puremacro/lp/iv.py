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


def _compute_anderson_rubin_ci(
    y_target: np.ndarray,
    x: np.ndarray,
    W_mat: np.ndarray,
    lags: int,
    alpha: float,
) -> tuple[float, float, str]:
    """Exact closed-form Anderson-Rubin (1949) confidence interval for LP-IV.

    Tests H_0: beta_h = beta_0 by checking whether the instrument coefficient
    gamma(beta_0) in y_target - beta_0 * x ~ z + W is zero using Newey-West HAC errors.
    Inverting AR(beta_0) <= chi2_1(1-alpha) yields a quadratic inequality
    A beta_0^2 + B beta_0 + C <= 0.
    """
    from scipy.stats import chi2

    MtM_inv = np.linalg.pinv(W_mat.T @ W_mat)
    w_z = MtM_inv[1, :]

    res_y = ols_hac(y_target, W_mat, lags=lags)
    gamma_y = float(res_y["beta"][1])
    u_y = y_target - W_mat @ res_y["beta"]

    res_x = ols_hac(x, W_mat, lags=lags)
    gamma_x = float(res_x["beta"][1])
    u_x = x - W_mat @ res_x["beta"]

    h_vec = W_mat @ w_z
    score_y = h_vec * u_y
    score_x = h_vec * u_x

    def _nw_var(s1: np.ndarray, s2: np.ndarray) -> float:
        v = float(np.dot(s1, s2))
        for ell in range(1, lags + 1):
            weight = 1.0 - ell / (lags + 1.0)
            gamma_ell = float(np.dot(s1[ell:], s2[:-ell]) + np.dot(s2[ell:], s1[:-ell]))
            v += weight * gamma_ell
        return v

    V_yy = _nw_var(score_y, score_y)
    V_xx = _nw_var(score_x, score_x)
    V_yx = _nw_var(score_y, score_x)

    crit = float(chi2.ppf(1.0 - alpha, df=1))

    A = gamma_x ** 2 - crit * V_xx
    B = -2.0 * (gamma_y * gamma_x - crit * V_yx)
    C = gamma_y ** 2 - crit * V_yy

    disc = B ** 2 - 4.0 * A * C
    if A > 0:
        if disc >= 0:
            r1 = (-B - np.sqrt(disc)) / (2.0 * A)
            r2 = (-B + np.sqrt(disc)) / (2.0 * A)
            return float(min(r1, r2)), float(max(r1, r2)), "bounded"
        else:
            return np.nan, np.nan, "empty"
    else:
        if disc >= 0:
            r1 = (-B - np.sqrt(disc)) / (2.0 * A)
            r2 = (-B + np.sqrt(disc)) / (2.0 * A)
            return float(max(r1, r2)), float(min(r1, r2)), "unbounded_rays"
        else:
            return -np.inf, np.inf, "all_real"


def lp_iv(
    df: pd.DataFrame,
    y: str,
    x: str,
    z: str,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    controls: Sequence[str] | None = None,
    alpha: float = 0.10,
    *,
    anderson_rubin: bool = False,
) -> pd.DataFrame:
    """Single-country local projections with instrumental variables (LP-IV).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing outcome, endogenous regressor, instrument, and controls.
    y : str
        Outcome variable name.
    x : str
        Endogenous shock / policy variable name.
    z : str
        External instrument name.
    horizons : iterable of int, default range(0, 21)
        Impulse response horizons.
    n_lags : int, default 2
        Number of lags of (x, y, controls) included as control variables.
    controls : sequence of str, optional
        Additional contemporaneous or baseline control variables.
    alpha : float, default 0.10
        Significance level for confidence intervals (0.10 -> 90% CI).
    anderson_rubin : bool, default False
        If True, also computes exact weak-instrument-robust Anderson-Rubin (1949)
        confidence intervals (``ar_lo``, ``ar_hi``, ``ar_set_type``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``[h, beta, se, t, lo, hi, first_stage_f]``
        (plus ``[ar_lo, ar_hi, ar_set_type]`` when ``anderson_rubin=True``).
    """
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
            row = {"h": h, "beta": np.nan, "se": np.nan, "t": np.nan,
                   "lo": np.nan, "hi": np.nan, "first_stage_f": np.nan}
            if anderson_rubin:
                row.update({"ar_lo": np.nan, "ar_hi": np.nan, "ar_set_type": "empty"})
            rows.append(row)
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

        row = {
            "h": h,
            "beta": beta_h,
            "se": se_h,
            "t": beta_h / se_h if se_h > 0 else np.nan,
            "lo": beta_h - z_crit * se_h,
            "hi": beta_h + z_crit * se_h,
            "first_stage_f": first_stage_f,
        }

        if anderson_rubin:
            ar_lo, ar_hi, ar_kind = _compute_anderson_rubin_ci(
                sub["dy_h"].values, sub[x].values, W_mat, lags=h + 1, alpha=alpha
            )
            row.update({"ar_lo": ar_lo, "ar_hi": ar_hi, "ar_set_type": ar_kind})

        rows.append(row)

    return pd.DataFrame(rows)


__all__ = ["lp_iv"]
