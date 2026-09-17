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


def tsls_hac(y, X, X_hat, lags: int) -> dict:
    """Second stage of two-stage least squares, with Newey-West HAC standard errors.

    ``X_hat`` is ``X`` with each endogenous column replaced by its first-stage fitted
    values. The estimate is the OLS of ``y`` on ``X_hat``, but the residuals, and so the
    variance, use the actual regressors: ``u = y - X beta``. Calling ``ols_hac(y, X_hat)``
    instead builds the variance from ``y - X_hat beta = u + beta (x - x_hat)``, which is
    not the structural error: its variance is off by ``2 beta cov(u, v) + beta^2 var(v)``,
    too wide or too narrow depending on the data.

    Bartlett kernel with bandwidth ``lags``; ``lags = 0`` is the Eicker-Huber-White (HC0)
    variance. Returns the same keys as :func:`ols_hac`.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    X_hat = np.asarray(X_hat, dtype=float)
    if X.shape != X_hat.shape:
        raise ValueError(f"tsls_hac: X {X.shape} and X_hat {X_hat.shape} must have the same shape")
    T, _ = X_hat.shape
    XhXh_inv = inv_xtx(X_hat, name="tsls_hac")
    beta = XhXh_inv @ X_hat.T @ y
    u = y - X @ beta
    scores = X_hat * u[:, None]
    S = scores.T @ scores
    for ell in range(1, lags + 1):
        w = 1.0 - ell / (lags + 1.0)
        Gamma = scores[ell:].T @ scores[:-ell]
        S += w * (Gamma + Gamma.T)
    vcov = XhXh_inv @ S @ XhXh_inv
    se = np.sqrt(np.diag(vcov))
    return {
        "beta": beta,
        "se": se,
        "t": beta / se,
        "vcov": vcov,
        "residuals": u,
        "n_obs": int(T),
    }


__all__ = ["ols_hac", "tsls_hac"]
