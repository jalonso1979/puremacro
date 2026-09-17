"""Lag-augmented local projection (Plagborg-Møller-Wolf 2021).

Adds extra lags of x and y beyond what's needed to reduce omitted-
variable bias, which (per PMW 2021) makes the standard
heteroskedasticity-robust SE valid at every horizon h — no HAC, no
Bonferroni, and no horizon-dependent bandwidth. Recommended lag count
is ``p_aug = p + h`` (or just a fixed safe number > p).

When an instrumental variable z is supplied, estimates lag-augmented LP-IV
with Eicker-Huber-White (HC0) robust Montiel Olea & Pflueger (2013) effective
F-statistics and White-robust Anderson-Rubin confidence sets.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .._linalg import inv_xtx
from ..inference._ols_helpers import tsls_hac
from ._common import resolve_lp_kwargs
from ._results import LPResult
from .iv import _compute_anderson_rubin_multi, mop_critical_values


def _ols_eicker_huber(y: np.ndarray, X: np.ndarray) -> dict:
    """OLS with Eicker-Huber-White (HC0) heteroskedasticity-robust SE."""
    XtX_inv = inv_xtx(X, name="la_lp")
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    meat = (X * u[:, None]).T @ (X * u[:, None])
    V = XtX_inv @ meat @ XtX_inv
    return {"beta": beta, "se": np.sqrt(np.diag(V)), "vcov": V, "resid": u}


def _compute_white_ar_ci(
    dy: np.ndarray,
    x: np.ndarray,
    Z_mat: np.ndarray,
    W_ctl_mat: np.ndarray,
    alpha: float,
    beta_hat: float = 0.0,
    se_hat: float = 1.0,
) -> tuple[float, float, str]:
    """Eicker-Huber-White robust Anderson-Rubin confidence set.

    ``k_z = 1``: exact quadratic inversion (the White special case of
    :func:`puremacro.lp.iv._compute_anderson_rubin_ci`). ``k_z >= 2``: the
    exact inversion of :func:`puremacro.lp.iv._compute_anderson_rubin_multi`
    with ``lags = 0`` (White ``Omega = S'S`` is the zero-lag Bartlett case).
    """
    k_z = Z_mat.shape[1]
    if k_z != 1:
        return _compute_anderson_rubin_multi(
            dy, x, W_ctl_mat, Z_mat, lags=0, alpha=alpha, beta_hat=beta_hat, se_hat=se_hat
        )

    W_inv = np.linalg.pinv(W_ctl_mat.T @ W_ctl_mat)
    y_tilde = dy - W_ctl_mat @ (W_inv @ (W_ctl_mat.T @ dy))
    x_tilde = x - W_ctl_mat @ (W_inv @ (W_ctl_mat.T @ x))
    Z_tilde = Z_mat - W_ctl_mat @ (W_inv @ (W_ctl_mat.T @ Z_mat))

    z_t = Z_tilde[:, 0]
    ztz = float(z_t @ z_t)
    if ztz <= 1e-14:
        return np.nan, np.nan, "empty"
    gamma_y = float(z_t @ y_tilde / ztz)
    gamma_x = float(z_t @ x_tilde / ztz)
    u_y = y_tilde - z_t * gamma_y
    u_x = x_tilde - z_t * gamma_x
    h_vec = z_t / ztz
    s_y = h_vec * u_y
    s_x = h_vec * u_x
    V_yy = float(np.dot(s_y, s_y))
    V_xx = float(np.dot(s_x, s_x))
    V_yx = float(np.dot(s_y, s_x))
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
        return np.nan, np.nan, "empty"
    if disc >= 0:
        r1 = (-B - np.sqrt(disc)) / (2.0 * A)
        r2 = (-B + np.sqrt(disc)) / (2.0 * A)
        return float(max(r1, r2)), float(min(r1, r2)), "unbounded_rays"
    return -np.inf, np.inf, "all_real"


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
    z: str | Sequence[str] | None = None,
    anderson_rubin: bool = False,
    weak_iv_robust: bool = False,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
) -> LPResult:
    """PMW (2021) lag-augmented LP with Eicker-Huber-White SE.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing outcome, regressor, and controls.
    y : str
        Outcome variable name.
    x : str
        Shock / policy variable name.
    horizons : iterable of int, default range(0, 21)
        Impulse response horizons.
    n_lags : int, default 4
        Baseline number of lags of x and y to include.
    extra_lags : int or None
        Additional lags beyond ``n_lags`` (PMW recommend p_aug ≥ p + h).
        If None, defaults to max(horizons), which makes coverage uniform
        across all reported horizons.
    controls : sequence of str, optional
        Additional control variables.
    alpha : float, default 0.10
        Significance level for confidence intervals (0.10 -> 90% CI).
    z : str or sequence of str, optional
        External instrument(s). If provided, runs lag-augmented LP-IV.
    anderson_rubin : bool, default False
        If True (when z is provided), computes White-robust Anderson-Rubin confidence sets.
    weak_iv_robust : bool, default False
        Alias for anderson_rubin.
    lags : int, optional
        Alias for n_lags.
    horizon : int, optional
        Sets horizons = range(0, horizon + 1).
    ci : float, optional
        Confidence level in (0, 1), e.g. 0.90 for 90% CI.

    Returns
    -------
    LPResult
        DataFrame subclass containing impulse responses and standard errors.
    """
    horizons, n_lags, alpha = resolve_lp_kwargs(
        horizons, n_lags, alpha, lags=lags, horizon=horizon, ci=ci, name="la_lp"
    )
    horizons = list(horizons)
    ctl = list(controls or [])
    z_crit = norm.ppf(1 - alpha / 2)
    H = max(horizons) if horizons else 0
    if extra_lags is None:
        extra_lags = H
    p_aug = n_lags + extra_lags
    do_ar = anderson_rubin or weak_iv_robust

    has_iv = z is not None
    z_names = ([z] if isinstance(z, str) else list(z)) if has_iv else []
    k_z = len(z_names)
    cv_10, cv_20 = mop_critical_values(k_z) if has_iv else (np.nan, np.nan)

    rows = []
    for h in horizons:
        vars_needed = [y, x] + z_names + ctl
        sub = df[vars_needed].copy()
        sub["__dy_h__"] = sub[y].shift(-h) - sub[y].shift(1)
        for lag in range(1, p_aug + 1):
            sub[f"__{x}_L{lag}__"] = sub[x].shift(lag)
            sub[f"__{y}_L{lag}__"] = sub[y].shift(lag)
        for c in ctl:
            for lag in range(1, n_lags + 1):
                sub[f"__{c}_L{lag}__"] = sub[c].shift(lag)
        sub = sub.dropna()
        if sub.empty:
            empty_row: dict[str, Any] = {
                "h": h,
                "beta": np.nan,
                "se": np.nan,
                "lo": np.nan,
                "hi": np.nan,
                "p_aug": p_aug,
            }
            if has_iv:
                empty_row.update(
                    {
                        "first_stage_f": np.nan,
                        "mop_f": np.nan,
                        "mop_cv_10": cv_10,
                        "mop_cv_20": cv_20,
                    }
                )
                if do_ar:
                    empty_row.update({"ar_lo": np.nan, "ar_hi": np.nan, "ar_set_type": "empty"})
            rows.append(empty_row)
            continue

        n = len(sub)
        # Base control columns: constant + augmented lags of x and y + control lags + controls
        ctl_cols = [np.ones(n)]
        for lag in range(1, p_aug + 1):
            ctl_cols.append(sub[f"__{x}_L{lag}__"].values)
            ctl_cols.append(sub[f"__{y}_L{lag}__"].values)
        for c in ctl:
            for lag in range(1, n_lags + 1):
                ctl_cols.append(sub[f"__{c}_L{lag}__"].values)
            ctl_cols.append(sub[c].values)
        W_ctl = np.column_stack(ctl_cols)

        if not has_iv:
            # Standard OLS lag-augmented LP
            X = np.column_stack([W_ctl, sub[x].values])
            out = _ols_eicker_huber(sub["__dy_h__"].values, X)
            beta_h = float(out["beta"][-1])
            se_h = float(out["se"][-1])
            rows.append(
                {
                    "h": h,
                    "beta": beta_h,
                    "se": se_h,
                    "lo": beta_h - z_crit * se_h,
                    "hi": beta_h + z_crit * se_h,
                    "p_aug": p_aug,
                }
            )
        else:
            # IV lag-augmented LP
            Z_mat = sub[z_names].values
            X_fs = np.column_stack([W_ctl, Z_mat])
            out_fs = _ols_eicker_huber(sub[x].values, X_fs)
            x_hat = X_fs @ out_fs["beta"]

            # Compute White-robust Montiel Olea & Pflueger effective F
            W_inv = np.linalg.pinv(W_ctl.T @ W_ctl)
            x_tilde = sub[x].values - W_ctl @ (W_inv @ (W_ctl.T @ sub[x].values))
            Z_tilde = Z_mat - W_ctl @ (W_inv @ (W_ctl.T @ Z_mat))
            ZtZ = Z_tilde.T @ Z_tilde
            ZtZ_inv = np.linalg.pinv(ZtZ)
            pi_hat = ZtZ_inv @ (Z_tilde.T @ x_tilde)
            v_hat = x_tilde - Z_tilde @ pi_hat
            scores = Z_tilde * v_hat[:, None]
            Omega_white = scores.T @ scores

            num = float(pi_hat.T @ ZtZ @ pi_hat)
            denom = float(np.trace(ZtZ_inv @ Omega_white))
            mop_f = float(num / denom) if denom > 0 else float("inf")
            first_stage_f = mop_f

            # Second stage
            X_ss = np.column_stack([W_ctl, x_hat])
            X_actual = np.column_stack([W_ctl, sub[x].values])
            out = tsls_hac(sub["__dy_h__"].values, X_actual, X_ss, lags=0)  # lags=0: HC0
            beta_h = float(out["beta"][-1])
            se_h = float(out["se"][-1])

            row = {
                "h": h,
                "beta": beta_h,
                "se": se_h,
                "lo": beta_h - z_crit * se_h,
                "hi": beta_h + z_crit * se_h,
                "p_aug": p_aug,
                "first_stage_f": first_stage_f,
                "mop_f": mop_f,
                "mop_cv_10": cv_10,
                "mop_cv_20": cv_20,
            }

            if do_ar:
                ar_lo, ar_hi, ar_kind = _compute_white_ar_ci(
                    sub["__dy_h__"].values,
                    sub[x].values,
                    Z_mat,
                    W_ctl,
                    alpha=alpha,
                    beta_hat=beta_h,
                    se_hat=se_h,
                )
                row.update({"ar_lo": ar_lo, "ar_hi": ar_hi, "ar_set_type": ar_kind})

            rows.append(row)

    res = LPResult(rows)
    if "h" in res.columns:
        res.index = res["h"]
    res.y_name = str(y)
    res.x_name = str(x)
    res.method = "la_lp_iv" if has_iv else "la_lp"
    res.ci_level = 1.0 - alpha
    return res


def la_lp_iv(
    df: pd.DataFrame,
    y: str,
    x: str,
    z: str | Sequence[str],
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 4,
    extra_lags: int | None = None,
    controls: Sequence[str] | None = None,
    alpha: float = 0.10,
    *,
    anderson_rubin: bool = False,
    weak_iv_robust: bool = False,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
) -> LPResult:
    """Lag-augmented local projection with instrumental variables (PMW 2021).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing outcome, endogenous regressor, instrument(s), and controls.
    y : str
        Outcome variable name.
    x : str
        Endogenous shock / policy variable name.
    z : str or sequence of str
        Instrument variable name(s).
    horizons : iterable of int, default range(0, 21)
        Impulse response horizons.
    n_lags : int, default 4
        Baseline number of lags.
    extra_lags : int or None
        Additional lags beyond n_lags.
    controls : sequence of str, optional
        Additional control variables.
    alpha : float, default 0.10
        Significance level (default 90% CI).
    anderson_rubin : bool, default False
        Compute White-robust Anderson-Rubin confidence sets.
    weak_iv_robust : bool, default False
        Alias for anderson_rubin.
    lags : int, optional
        Alias for n_lags.
    horizon : int, optional
        Sets horizons = range(0, horizon + 1).
    ci : float, optional
        Confidence level, e.g. 0.90.

    Returns
    -------
    LPResult
        DataFrame subclass containing impulse responses, White-robust MOP effective F,
        and optional AR confidence sets.
    """
    return la_lp(
        df=df,
        y=y,
        x=x,
        horizons=horizons,
        n_lags=n_lags,
        extra_lags=extra_lags,
        controls=controls,
        alpha=alpha,
        z=z,
        anderson_rubin=anderson_rubin,
        weak_iv_robust=weak_iv_robust,
        lags=lags,
        horizon=horizon,
        ci=ci,
    )


__all__ = ["la_lp", "la_lp_iv"]
