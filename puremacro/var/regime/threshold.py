"""Self-exciting threshold VAR (Tsay 1998 / Hubrich-Teräsvirta 2013).

Two regimes split by whether a threshold variable z_{t-d} is above or
below a threshold c:

    y_t = c^L + A^L y_{t-1} + ... + A^L_p y_{t-p} + e^L_t   if z_{t-d} <= c
    y_t = c^H + A^H y_{t-1} + ... + A^H_p y_{t-p} + e^H_t   if z_{t-d} >  c

Estimation: grid-search the threshold ``c`` (over inner quantiles of z)
and the delay ``d``; for each (c, d) split the sample, run OLS on each
regime, sum the residual SS, and pick the (c, d) minimising the total.

References
----------
Tsay, R.S. (1998). Testing and modeling multivariate threshold models.
    JASA 93(443), 1188-1202.
Hubrich, K. and Teräsvirta, T. (2013). Thresholds and Smooth
    Transitions in Vector Autoregressive Models. CREATES research paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TVARResult:
    A_low: np.ndarray            # (n, n*p)
    intercept_low: np.ndarray    # (n,)
    Sigma_low: np.ndarray        # (n, n)
    A_high: np.ndarray
    intercept_high: np.ndarray
    Sigma_high: np.ndarray
    threshold: float
    delay: int
    threshold_var_idx: int
    n_low: int
    n_high: int
    rss_total: float
    grid: dict


def _ols_segment(Y_dep, X):
    """OLS with intercept on a segment. Returns (intercept, A.T, Sigma, RSS)."""
    n_obs, _ = Y_dep.shape
    if n_obs <= X.shape[1] + 1:
        return None
    X1 = np.column_stack([np.ones(n_obs), X])
    try:
        B = np.linalg.lstsq(X1, Y_dep, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    resid = Y_dep - X1 @ B
    Sigma = (resid.T @ resid) / max(1, n_obs - X1.shape[1])
    rss = float((resid ** 2).sum())
    return B[0], B[1:].T, Sigma, rss  # intercept, A, Sigma, RSS


def tvar_fit(
    Y: np.ndarray,
    *,
    threshold_var_idx: int = 0,
    p: int = 1,
    delay_grid=(1, 2, 3, 4),
    n_threshold_grid: int = 25,
    trim: float = 0.15,
) -> TVARResult:
    """Fit a self-exciting TVAR(p) by grid search.

    Parameters
    ----------
    Y : (T, n) ndarray.
    threshold_var_idx : column index of z (the threshold variable).
    p : VAR lag order.
    delay_grid : iterable of delay values d to try.
    n_threshold_grid : number of candidate threshold values to test
        (taken from inner quantiles of z respecting the ``trim`` fraction).
    trim : minimum fraction of obs in each regime.
    """
    Y = np.asarray(Y, dtype=float)
    T, n = Y.shape
    Y_dep = Y[p:]
    X_lag = np.column_stack([Y[p - k - 1: T - k - 1] for k in range(p)])

    z_full = Y[:, threshold_var_idx]
    T_eff = Y_dep.shape[0]
    best = None
    grid_log: list[dict] = []

    for d in delay_grid:
        if d > p:
            # Need z_{t-d} for t in [p..T-1]; require d <= p (otherwise we'd
            # still be padding / re-aligning). Keep simple: skip if d > p.
            continue
        # For the slice Y_dep (rows p..T-1), z_{t-d} corresponds to z_full[p-d : T-d].
        z_t_minus_d = z_full[p - d: T - d]
        lo_q, hi_q = trim, 1.0 - trim
        cand_thr = np.quantile(z_t_minus_d, np.linspace(lo_q, hi_q, n_threshold_grid))
        for c in cand_thr:
            mask_lo = z_t_minus_d <= c
            n_lo = int(mask_lo.sum())
            n_hi = T_eff - n_lo
            if n_lo < int(trim * T_eff) or n_hi < int(trim * T_eff):
                continue
            seg_lo = _ols_segment(Y_dep[mask_lo], X_lag[mask_lo])
            seg_hi = _ols_segment(Y_dep[~mask_lo], X_lag[~mask_lo])
            if seg_lo is None or seg_hi is None:
                continue
            rss_total = seg_lo[3] + seg_hi[3]
            grid_log.append({"d": int(d), "c": float(c), "rss": float(rss_total)})
            if best is None or rss_total < best["rss"]:
                best = {
                    "rss": rss_total,
                    "c": float(c),
                    "d": int(d),
                    "low": seg_lo,
                    "high": seg_hi,
                    "n_low": n_lo,
                    "n_high": n_hi,
                }
    if best is None:
        raise RuntimeError("No valid threshold-delay split satisfied the trim.")

    return TVARResult(
        intercept_low=best["low"][0], A_low=best["low"][1], Sigma_low=best["low"][2],
        intercept_high=best["high"][0], A_high=best["high"][1], Sigma_high=best["high"][2],
        threshold=best["c"], delay=best["d"],
        threshold_var_idx=threshold_var_idx,
        n_low=best["n_low"], n_high=best["n_high"],
        rss_total=best["rss"],
        grid={"log": grid_log},
    )


__all__ = ["tvar_fit", "TVARResult"]
