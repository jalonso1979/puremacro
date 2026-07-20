"""OLS with Newey-West HAC SE — replaces statsmodels.OLS(cov_type='HAC')."""
from __future__ import annotations

import numpy as np

from .._linalg import inv_xtx


def ols_hac(y, X, lags: int) -> dict:
    """Fit y = X β + u and return Newey-West HAC standard errors.

    Bartlett kernel; bandwidth = ``lags``.

    Returns
    -------
    dict with keys:
        beta : np.ndarray (k,) — point estimates
        se   : np.ndarray (k,) — Newey-West standard errors
        t    : np.ndarray (k,) — beta / se
        vcov : np.ndarray (k, k) — sandwich covariance
        residuals : np.ndarray (T,) — y - X β
        n_obs : int — sample size T
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    T, k = X.shape
    XtX_inv = inv_xtx(X, name="ols_hac")
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta

    # Newey-West sandwich: V = (X'X)^-1 S (X'X)^-1
    # S = sum_{ell = -L..L} w_ell sum_t u_t u_{t-ell} x_t x_{t-ell}'
    # Bartlett weights w_ell = 1 - |ell|/(L+1).
    # ell = 0
    S = (X * u[:, None]).T @ (X * u[:, None])
    for ell in range(1, lags + 1):
        w = 1.0 - ell / (lags + 1.0)
        Gamma = (X[ell:] * u[ell:, None]).T @ (X[:-ell] * u[:-ell, None])
        S += w * (Gamma + Gamma.T)
    vcov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(vcov))
    return {
        "beta": beta,
        "se": se,
        "t": beta / se,
        "vcov": vcov,
        "residuals": u,
        "n_obs": int(T),
    }


__all__ = ["ols_hac"]
