"""Diebold-Yilmaz (2009, 2012) spillover index via FEVD.

Total spillover index:
    S(H) = 100 * (sum_{i != j} theta_tilde_{ij}(H)) / N
where theta_tilde_{ij}(H) is the share of variable i's H-step forecast-error
variance attributable to shock j (FEVD), normalised so each row sums to 1.

Directional measures and the pairwise N x N matrix are also returned.

Note: Uses Cholesky-orthogonalised FEVD (variable ordering matters). The
original DY 2012 paper uses generalised FEVD (Pesaran-Shin 1998) for
order-invariance; Cholesky is used here for v1 simplicity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..var.estimate import estimate_var
from ..var.identify.cholesky import cholesky_factor
from ..var.irf import fevd as compute_fevd, gfevd as compute_gfevd


def spillover_index(
    returns_panel: pd.DataFrame,
    var_lags: int = 4,
    fevd_horizon: int = 12,
    window: int | None = None,
    identification: str = "gfevd",
) -> dict:
    """Compute the Diebold-Yilmaz total / directional / pairwise spillover.

    Parameters
    ----------
    returns_panel : pd.DataFrame
        Panel of return / growth-rate series, shape (T, n).
    var_lags : int
        Number of VAR lags p.
    fevd_horizon : int
        Forecast horizon H for FEVD.
    window : int or None
        If given, roll a window of this length across observations and
        attach a time series of total spillover under key ``total_series``.
    identification : {"gfevd", "cholesky"}
        FEVD identification scheme. ``gfevd`` (Pesaran-Shin 1998,
        Diebold-Yilmaz 2012) is order-invariant and the modern default;
        ``cholesky`` reproduces Diebold-Yilmaz 2009 and depends on the
        column order of ``returns_panel``.

    Returns
    -------
    dict with keys:
        total             : float — total spillover index (%)
        directional_to    : (n,) array — to-spillover for each variable (%)
        directional_from  : (n,) array — from-spillover for each variable (%)
        net               : (n,) array — net spillover = to - from
        pairwise_matrix   : (n, n) array — normalised FEVD matrix theta
        total_series      : pd.Series (only when window is not None)
    """
    Y = returns_panel.dropna()
    if window is None:
        return _compute_one(Y.values, var_lags, fevd_horizon, identification)
    # Rolling window
    series = []
    dates = []
    for end in range(window, len(Y) + 1):
        sub = Y.iloc[end - window: end]
        try:
            s = _compute_one(sub.values, var_lags, fevd_horizon, identification)
            series.append(s["total"])
            dates.append(Y.index[end - 1])
        except (np.linalg.LinAlgError, ValueError):
            continue
    out = _compute_one(Y.values, var_lags, fevd_horizon, identification)
    out["total_series"] = pd.Series(series, index=dates)
    return out


def _compute_one(Y: np.ndarray, p: int, H: int, identification: str = "gfevd") -> dict:
    """Estimate VAR, compute FEVD, and return spillover measures."""
    A_list, _c, Sigma, _resid, _X = estimate_var(Y, p)
    if identification == "cholesky":
        B0 = cholesky_factor(Sigma)
        fev = compute_fevd(A_list, B0, H)
        theta = fev[H]
    elif identification == "gfevd":
        gfev = compute_gfevd(A_list, Sigma, H, normalize=True)
        theta = gfev[H]
    else:
        raise ValueError(f"unknown identification {identification!r}")
    n = theta.shape[0]
    # Total: 100 * (off-diagonal sum) / n
    off_diag_sum = float(theta.sum() - np.trace(theta))
    total = 100.0 * off_diag_sum / n
    # Directional to-others: column sums minus diagonal (off-diag col sums) / n
    dir_to = 100.0 * (theta.sum(axis=0) - np.diag(theta)) / n
    # Directional from-others: row sums minus diagonal (off-diag row sums) / n
    dir_from = 100.0 * (theta.sum(axis=1) - np.diag(theta)) / n
    net = dir_to - dir_from
    return {
        "total": float(total),
        "directional_to": dir_to,
        "directional_from": dir_from,
        "net": net,
        "pairwise_matrix": theta,
    }


__all__ = ["spillover_index"]
