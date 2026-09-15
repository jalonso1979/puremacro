"""Single-country LP-IV (Stock-Watson 2018; Plagborg-Møller-Wolf 2021).

First stage: x_t = π_z' z_t + π_w' W_t + ν_t.
Second stage: y_{t+h} - y_{t-1} = α_h + β_h x_t + γ' W_t + ε_{t,h}
              with x_t replaced by its first-stage projection x̂_t.
HAC SE via Newey-West with bandwidth L = h + 1.

Provides Montiel Olea & Pflueger (2013) effective F-statistic (F_eff)
and weak-IV robust confidence sets (Anderson-Rubin / MOP).
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from ..inference._ols_helpers import ols_hac
from ._common import resolve_lp_kwargs


def mop_critical_values(k_z: int, alpha: float = 0.05) -> tuple[float, float]:
    """Return Montiel Olea & Pflueger (2013) critical values for 10% and 20% worst-case bias.

    Parameters
    ----------
    k_z : int
        Number of excluded instruments.
    alpha : float, default 0.05
        Significance level (default 5% test).

    Returns
    -------
    tuple[float, float]
        (cv_10, cv_20) critical values for tau=0.10 and tau=0.20 bias thresholds.
    """
    mop_table_10 = {1: 11.52, 2: 11.12, 3: 10.60, 4: 10.20}
    mop_table_20 = {1: 6.70, 2: 6.00, 3: 5.50, 4: 5.20}
    cv_10 = mop_table_10.get(k_z, 9.80 if k_z >= 5 else 11.52)
    cv_20 = mop_table_20.get(k_z, 4.90 if k_z >= 5 else 6.70)
    return cv_10, cv_20


def compute_mop_effective_f(
    x: np.ndarray,
    Z: np.ndarray,
    W_ctl: np.ndarray,
    lags: int,
) -> float:
    """Compute Montiel Olea & Pflueger (2013) effective F-statistic with Newey-West HAC errors."""
    x_arr = np.asarray(x, dtype=float).ravel()
    Z_arr = np.asarray(Z, dtype=float)
    if Z_arr.ndim == 1:
        Z_arr = Z_arr[:, None]
    W_ctl_arr = np.asarray(W_ctl, dtype=float)

    # Partial out controls W_ctl
    W_inv = np.linalg.pinv(W_ctl_arr.T @ W_ctl_arr)
    x_tilde = x_arr - W_ctl_arr @ (W_inv @ (W_ctl_arr.T @ x_arr))
    Z_tilde = Z_arr - W_ctl_arr @ (W_inv @ (W_ctl_arr.T @ Z_arr))

    ZtZ = Z_tilde.T @ Z_tilde
    ZtZ_inv = np.linalg.pinv(ZtZ)
    pi_hat = ZtZ_inv @ (Z_tilde.T @ x_tilde)
    v_hat = x_tilde - Z_tilde @ pi_hat

    scores = Z_tilde * v_hat[:, None]
    Omega_zz = scores.T @ scores
    for ell in range(1, lags + 1):
        weight = 1.0 - ell / (lags + 1.0)
        gamma_ell = scores[ell:].T @ scores[:-ell]
        Omega_zz += weight * (gamma_ell + gamma_ell.T)

    num = float(pi_hat.T @ ZtZ @ pi_hat)
    denom = float(np.trace(ZtZ_inv @ Omega_zz))
    if denom <= 0:
        return float("inf")
    return float(num / denom)


def _compute_anderson_rubin_ci(
    y_target: np.ndarray,
    x: np.ndarray,
    W_mat: np.ndarray,
    lags: int,
    alpha: float,
) -> tuple[float, float, str]:
    """Exact closed-form Anderson-Rubin (1949) confidence interval for LP-IV."""
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


def _compute_anderson_rubin_multi(
    y_target: np.ndarray,
    x: np.ndarray,
    W_ctl: np.ndarray,
    Z_mat: np.ndarray,
    lags: int,
    alpha: float,
    beta_hat: float = 0.0,
    se_hat: float = 1.0,
) -> tuple[float, float, str]:
    """Weak-IV-robust Anderson-Rubin confidence set for multi-instrument LP-IV."""
    k_z = Z_mat.shape[1]
    crit = float(chi2.ppf(1.0 - alpha, df=k_z))
    W_inv = np.linalg.pinv(W_ctl.T @ W_ctl)
    y_tilde = y_target - W_ctl @ (W_inv @ (W_ctl.T @ y_target))
    x_tilde = x - W_ctl @ (W_inv @ (W_ctl.T @ x))
    Z_tilde = Z_mat - W_ctl @ (W_inv @ (W_ctl.T @ Z_mat))
    ZtZ_inv = np.linalg.pinv(Z_tilde.T @ Z_tilde)

    span = max(10.0 * (se_hat if np.isfinite(se_hat) and se_hat > 0 else 1.0), 10.0)
    grid = np.linspace(beta_hat - span, beta_hat + span, 401)
    mask = np.empty(len(grid), dtype=bool)
    accepted = []

    for i, b0 in enumerate(grid):
        w = y_tilde - b0 * x_tilde
        delta = ZtZ_inv @ (Z_tilde.T @ w)
        e = w - Z_tilde @ delta
        scores = Z_tilde * e[:, None]
        Omega = scores.T @ scores
        for ell in range(1, lags + 1):
            weight = 1.0 - ell / (lags + 1.0)
            gamma_ell = scores[ell:].T @ scores[:-ell]
            Omega += weight * (gamma_ell + gamma_ell.T)
        V_delta = ZtZ_inv @ Omega @ ZtZ_inv
        try:
            stat = float(delta.T @ np.linalg.solve(V_delta, delta))
        except np.linalg.LinAlgError:
            stat = float(delta.T @ np.linalg.pinv(V_delta) @ delta)
        is_acc = stat <= crit
        mask[i] = is_acc
        if is_acc:
            accepted.append(b0)

    if not accepted:
        return np.nan, np.nan, "empty"
    if len(accepted) == len(grid):
        return -np.inf, np.inf, "all_real"

    left_acc = bool(mask[0])
    right_acc = bool(mask[-1])

    if left_acc and right_acc:
        rej_indices = np.where(~mask)[0]
        r1 = float(grid[rej_indices[0] - 1])
        r2 = float(grid[rej_indices[-1] + 1])
        return float(max(r1, r2)), float(min(r1, r2)), "unbounded_rays"
    elif left_acc or right_acc:
        rej_indices = np.where(~mask)[0]
        r1 = float(grid[rej_indices[0] - 1]) if left_acc else float(grid[0])
        r2 = float(grid[rej_indices[-1] + 1]) if right_acc else float(grid[-1])
        return float(max(r1, r2)), float(min(r1, r2)), "unbounded_rays"

    return float(min(accepted)), float(max(accepted)), "bounded"


def lp_iv(
    df: pd.DataFrame,
    y: str,
    x: str,
    z: str | Sequence[str],
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    controls: Sequence[str] | None = None,
    alpha: float = 0.10,
    *,
    anderson_rubin: bool = False,
    weak_iv_robust: bool = False,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
) -> pd.DataFrame:
    """Single-country local projections with instrumental variables (LP-IV).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing outcome, endogenous regressor, instrument(s), and controls.
    y : str
        Outcome variable name.
    x : str
        Endogenous shock / policy variable name.
    z : str or sequence of str
        External instrument name(s).
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
    weak_iv_robust : bool, default False
        Alias for ``anderson_rubin``.
    lags : int, optional
        Alias for ``n_lags``. Standardized keyword.
    horizon : int, optional
        If provided, sets ``horizons = range(0, horizon + 1)``.
    ci : float, optional
        Confidence level in (0, 1), e.g. 0.90 for 90% CI. Sets ``alpha = 1.0 - ci``.

    Returns
    -------
    LPResult
        DataFrame subclass with columns:
        ``[h, beta, se, t, lo, hi, first_stage_f, mop_f, mop_cv_10, mop_cv_20]``
        (plus ``[ar_lo, ar_hi, ar_set_type]`` when ``anderson_rubin=True`` or ``weak_iv_robust=True``).
    """
    horizons, n_lags, alpha = resolve_lp_kwargs(
        horizons, n_lags, alpha, lags=lags, horizon=horizon, ci=ci, name="lp_iv"
    )

    horizons = list(horizons)
    ctl = list(controls or [])
    z_names = [z] if isinstance(z, str) else list(z)
    k_z = len(z_names)
    cv_10, cv_20 = mop_critical_values(k_z)
    z_crit = norm.ppf(1 - alpha / 2)
    do_ar = anderson_rubin or weak_iv_robust

    rows = []
    for h in horizons:
        sub = df[[y, x] + z_names + ctl].copy()
        sub["dy_h"] = sub[y].shift(-h) - sub[y].shift(1)
        for lag in range(1, n_lags + 1):
            sub[f"{x}_L{lag}"] = sub[x].shift(lag)
            sub[f"{y}_L{lag}"] = sub[y].shift(lag)
            for c in ctl:
                sub[f"{c}_L{lag}"] = sub[c].shift(lag)
        sub = sub.dropna()
        if sub.empty:
            row: dict[str, Any] = {
                "h": h,
                "beta": np.nan,
                "se": np.nan,
                "t": np.nan,
                "lo": np.nan,
                "hi": np.nan,
                "first_stage_f": np.nan,
                "mop_f": np.nan,
                "mop_cv_10": cv_10,
                "mop_cv_20": cv_20,
            }
            if do_ar:
                row.update({"ar_lo": np.nan, "ar_hi": np.nan, "ar_set_type": "empty"})
            rows.append(row)
            continue

        n = len(sub)
        # Exogenous controls (without instruments)
        W_ctl_list = [np.ones(n)]
        for lag in range(1, n_lags + 1):
            W_ctl_list.append(sub[f"{x}_L{lag}"].values)
            W_ctl_list.append(sub[f"{y}_L{lag}"].values)
            for c in ctl:
                W_ctl_list.append(sub[f"{c}_L{lag}"].values)
        for c in ctl:
            W_ctl_list.append(sub[c].values)
        W_ctl_mat = np.column_stack(W_ctl_list)

        # Instruments
        Z_mat = sub[z_names].values

        # First stage regressor matrix:
        # If single instrument, keep standard order [ones, z, lags...] for exact backward compat
        if k_z == 1:
            W_mat_list = [np.ones(n), sub[z_names[0]].values]
            for lag in range(1, n_lags + 1):
                W_mat_list.append(sub[f"{x}_L{lag}"].values)
                W_mat_list.append(sub[f"{y}_L{lag}"].values)
                for c in ctl:
                    W_mat_list.append(sub[f"{c}_L{lag}"].values)
            for c in ctl:
                W_mat_list.append(sub[c].values)
            W_mat = np.column_stack(W_mat_list)
        else:
            W_mat = np.column_stack([W_ctl_mat, Z_mat])

        # First stage regression
        fs = ols_hac(sub[x].values, W_mat, lags=h + 1)
        x_hat = W_mat @ fs["beta"]

        # First-stage F & Montiel Olea & Pflueger effective F
        mop_f = compute_mop_effective_f(sub[x].values, Z_mat, W_ctl_mat, lags=h + 1)

        if k_z == 1:
            first_stage_f = (
                float((fs["beta"][1] / fs["se"][1]) ** 2) if fs["se"][1] > 0 else np.nan
            )
        else:
            first_stage_f = mop_f

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
            "mop_f": mop_f,
            "mop_cv_10": cv_10,
            "mop_cv_20": cv_20,
        }

        if do_ar:
            if k_z == 1:
                ar_lo, ar_hi, ar_kind = _compute_anderson_rubin_ci(
                    sub["dy_h"].values, sub[x].values, W_mat, lags=h + 1, alpha=alpha
                )
            else:
                ar_lo, ar_hi, ar_kind = _compute_anderson_rubin_multi(
                    sub["dy_h"].values,
                    sub[x].values,
                    W_ctl_mat,
                    Z_mat,
                    lags=h + 1,
                    alpha=alpha,
                    beta_hat=beta_h,
                    se_hat=se_h,
                )
            row.update({"ar_lo": ar_lo, "ar_hi": ar_hi, "ar_set_type": ar_kind})

        rows.append(row)

    from ._results import LPResult

    res = LPResult(rows)
    if "h" in res.columns:
        res.index = res["h"]
    res.y_name = str(y)
    res.x_name = str(x)
    res.method = "LP-IV"
    res.ci_level = 1.0 - alpha
    return res


__all__ = ["lp_iv", "compute_mop_effective_f", "mop_critical_values"]
