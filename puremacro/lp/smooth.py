"""Barnichon-Brownlees (2019) smoothed local projections.

Penalises second differences of β_h across horizons using a B-spline
basis. The smoothing parameter λ is set by GCV by default; can be
passed explicitly via lambda_.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.stats import norm

from .jorda import lp_hac
from ._results import LPResult


def lp_smooth(
    df: pd.DataFrame,
    y: str,
    x: str,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    controls: Sequence[str] | None = None,
    n_knots: int = 6,
    lambda_: float | None = None,
    alpha: float = 0.10,
    *,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
) -> LPResult:
    """Smooth the IRF surface across horizons.

    Internally:
      1. Run unsmoothed LP to get raw {β_h, se_h}.
      2. Build a cubic B-spline basis B over the horizon grid.
      3. Solve β̂_smooth = B (B'B + λ P'P)^{-1} B' β_raw
         where P is the second-difference penalty matrix.
      4. λ chosen by GCV unless supplied.

    Returns LPResult with columns [h, beta, se, lo, hi, lambda].
    """
    if lags is not None:
        n_lags = lags
    if horizon is not None:
        horizons = range(0, horizon + 1)
    if ci is not None:
        alpha = 1.0 - ci
    horizons = list(horizons)
    H = len(horizons)
    h_arr = np.array(horizons, dtype=float)
    z_crit = norm.ppf(1 - alpha / 2)

    raw = lp_hac(df, y=y, x=x, horizons=horizons, n_lags=n_lags,
                  controls=controls, alpha=alpha)
    beta_raw = raw["beta"].values
    se_raw = raw["se"].values

    # B-spline basis on horizons
    knots_inner = np.linspace(h_arr.min(), h_arr.max(), n_knots)
    t_knots = np.r_[[knots_inner[0]] * 3, knots_inner, [knots_inner[-1]] * 3]
    n_basis = n_knots + 2
    Bmat = np.zeros((H, n_basis))
    for j in range(n_basis):
        c = np.zeros(n_basis); c[j] = 1.0
        Bmat[:, j] = BSpline(t_knots, c, 3, extrapolate=False)(h_arr)
    Bmat = np.nan_to_num(Bmat)

    # Second-difference penalty
    D2 = np.diff(np.eye(n_basis), n=2, axis=0)
    P = D2.T @ D2

    if lambda_ is None:
        # GCV grid search
        grid = np.logspace(-3, 4, 30)
        scores = []
        for lam in grid:
            try:
                A = Bmat @ np.linalg.solve(Bmat.T @ Bmat + lam * P, Bmat.T)
                fit = A @ beta_raw
                tr = np.trace(A)
                denom = max(1e-12, (1 - tr / H) ** 2)
                gcv = np.sum((beta_raw - fit) ** 2) / denom
                scores.append(gcv)
            except np.linalg.LinAlgError:
                scores.append(np.inf)
        lambda_ = float(grid[int(np.argmin(scores))])

    A = Bmat @ np.linalg.solve(Bmat.T @ Bmat + lambda_ * P, Bmat.T)
    beta_smooth = A @ beta_raw
    # Smoothed SE: A diag(se_raw^2) A'  → diag
    se_smooth = np.sqrt(np.einsum('ij,j,ij->i', A, se_raw ** 2, A))

    res = LPResult({
        "h": horizons,
        "beta": beta_smooth,
        "se": se_smooth,
        "lo": beta_smooth - z_crit * se_smooth,
        "hi": beta_smooth + z_crit * se_smooth,
        "lambda": [lambda_] * H,
    })
    if "h" in res.columns:
        res.index = res["h"]
    res.y_name = str(y)
    res.x_name = str(x)
    res.method = "LP-smooth"
    return res


__all__ = ["lp_smooth"]
