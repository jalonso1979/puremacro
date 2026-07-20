"""Newey-West HAC standard errors."""

from __future__ import annotations

import numpy as np


def newey_west_se(X: np.ndarray, resid: np.ndarray, bw: int) -> np.ndarray:
    """Return Newey-West HAC standard errors of an OLS regression.

    Parameters
    ----------
    X     : (n, k) regressor matrix, with constant if used.
    resid : (n,)   OLS residuals.
    bw    : non-negative integer bandwidth (Bartlett kernel).
    """
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    u = X * resid[:, None]
    S = u.T @ u
    for lag in range(1, bw + 1):
        w = 1 - lag / (bw + 1)
        G = u[lag:].T @ u[:-lag]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.diag(cov))
