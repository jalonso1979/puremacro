"""Modern cointegration: Phillips-Hansen FM-OLS, Stock-Watson DOLS,
Phillips-Ouliaris residual-based tests.

Companion to ``puremacro.var.vecm`` (Engle-Granger + Johansen). These
estimators give consistent, asymptotically normal cointegrating-vector
estimates suitable for publication-grade inference.

References
----------
Phillips, P.C.B. and Hansen, B.E. (1990). Statistical inference in
    instrumental variables regression with I(1) processes. RES 57(1).
Stock, J.H. and Watson, M.W. (1993). A simple estimator of cointegrating
    vectors in higher order integrated systems. Econometrica 61(4).
Phillips, P.C.B. and Ouliaris, S. (1990). Asymptotic properties of
    residual based tests for cointegration. Econometrica 58(1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._linalg import inv_xtx
from .inference._ols_helpers import ols_hac


@dataclass(frozen=True)
class FMOLSResult:
    """Result of :func:`fm_ols` (Phillips-Hansen Fully-Modified OLS).

    Attributes
    ----------
    alpha : float
        Estimated intercept.
    beta : np.ndarray
        Cointegrating coefficients (excluding intercept), shape ``(k,)``.
    se : np.ndarray
        HAC standard errors for ``beta``, shape ``(k,)``.
    residuals : np.ndarray
        Endogeneity-corrected residuals, length ``T - 1``.
    Omega : np.ndarray
        Two-sided long-run covariance, shape ``(k+1, k+1)``.
    Sigma : np.ndarray
        Contemporaneous covariance, shape ``(k+1, k+1)``.
    Lambda : np.ndarray
        One-sided long-run covariance, shape ``(k+1, k+1)``.

    References
    ----------
    Phillips, P.C.B. and Hansen, B.E. (1990). Statistical inference in
        instrumental variables regression with I(1) processes. RES 57(1).
    """

    alpha: float
    beta: np.ndarray
    se: np.ndarray
    residuals: np.ndarray
    Omega: np.ndarray
    Sigma: np.ndarray
    Lambda: np.ndarray

    def summary(self) -> str:
        coefs = ", ".join(f"{b:+.4f}" for b in self.beta)
        ses = ", ".join(f"{s:.4f}" for s in self.se)
        return (
            f"FM-OLS (Phillips-Hansen)\n"
            f"  alpha             : {self.alpha:+.4f}\n"
            f"  beta              : {coefs}\n"
            f"  se(beta)          : {ses}\n"
            f"  n residuals       : {len(self.residuals)}\n"
        )


@dataclass(frozen=True)
class DOLSResult:
    """Result of :func:`dols` (Stock-Watson DOLS).

    Attributes
    ----------
    alpha : float
        Estimated intercept.
    beta : np.ndarray
        Cointegrating coefficients (excluding intercept), shape ``(k,)``.
    se : np.ndarray
        HAC standard errors for ``beta``, shape ``(k,)``.
    alpha_se : float
        HAC standard error for the intercept.
    n_obs : int
        Effective sample size used in the augmented regression
        (lower than T due to lead/lag truncation).

    References
    ----------
    Stock, J.H. and Watson, M.W. (1993). A simple estimator of cointegrating
        vectors in higher order integrated systems. Econometrica 61(4).
    """

    alpha: float
    beta: np.ndarray
    se: np.ndarray
    alpha_se: float
    n_obs: int

    def summary(self) -> str:
        coefs = ", ".join(f"{b:+.4f}" for b in self.beta)
        ses = ", ".join(f"{s:.4f}" for s in self.se)
        return (
            f"DOLS (Stock-Watson)\n"
            f"  alpha             : {self.alpha:+.4f} (se {self.alpha_se:.4f})\n"
            f"  beta              : {coefs}\n"
            f"  se(beta)          : {ses}\n"
            f"  n_obs             : {self.n_obs}\n"
        )


@dataclass(frozen=True)
class PhillipsOuliarisResult:
    """Result of :func:`phillips_ouliaris` (residual-based cointegration test).

    Attributes
    ----------
    z_alpha : float
        Phillips-Ouliaris Z_alpha statistic.
    z_t : float
        Phillips-Ouliaris Z_t statistic. More negative => stronger evidence
        of cointegration. 5% CV ~= -3.37 for k=1 regressor (PO 1990 Table II).
    residuals : np.ndarray
        OLS residuals from the first-stage cointegrating regression.
    rho : float
        First-stage AR(1) coefficient on the residuals.
    lag_used : int
        Bartlett-kernel truncation lag used for long-run variance correction.

    References
    ----------
    Phillips, P.C.B. and Ouliaris, S. (1990). Asymptotic properties of
        residual based tests for cointegration. Econometrica 58(1).
    """

    z_alpha: float
    z_t: float
    residuals: np.ndarray
    rho: float
    lag_used: int

    def summary(self) -> str:
        return (
            f"Phillips-Ouliaris cointegration test\n"
            f"  Z_alpha           : {self.z_alpha:+.4f}\n"
            f"  Z_t               : {self.z_t:+.4f}  (5% CV ~= -3.37, k=1)\n"
            f"  rho               : {self.rho:+.4f}\n"
            f"  lag_used          : {self.lag_used}\n"
        )


# ---------------------------------------------------------------------------
# Long-run covariance via Bartlett kernel
# ---------------------------------------------------------------------------
def _long_run_cov(u: np.ndarray, lags: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Omega, Sigma, Lambda) for u: (T, m).

    Sigma  = E[u_t u_t']             (contemporaneous)
    Lambda = sum_{j=1..L} Gamma_j    (one-sided)
    Omega  = Sigma + Lambda + Lambda' (long-run, two-sided)
    """
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        u = u[:, None]
    T, m = u.shape
    if lags is None:
        lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    Sigma = (u.T @ u) / T
    Lambda = np.zeros((m, m))
    for j in range(1, lags + 1):
        if j >= T:
            break
        w = 1.0 - j / (lags + 1.0)
        Gamma_j = (u[j:].T @ u[:-j]) / T
        Lambda += w * Gamma_j
    Omega = Sigma + Lambda + Lambda.T
    return Omega, Sigma, Lambda


# ---------------------------------------------------------------------------
# Phillips-Hansen FM-OLS
# ---------------------------------------------------------------------------
def fm_ols(y, X, lags: int | None = None) -> FMOLSResult:
    """Phillips-Hansen Fully-Modified OLS estimator of the cointegrating
    vector in y_t = alpha + X_t' beta + u_t with X_t I(1).

    Returns
    -------
    FMOLSResult
        See :class:`FMOLSResult` for fields.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(X if np.asarray(X).ndim > 1 else np.asarray(X)[:, None])
    X = X.astype(float)
    T, k = X.shape
    Z = np.column_stack([np.ones(T), X])
    beta_ols, *_ = np.linalg.lstsq(Z, y, rcond=None)
    u_hat = y - Z @ beta_ols
    dX = np.diff(X, axis=0)
    # Stack residual systems for long-run cov
    stack = np.column_stack([u_hat[1:], dX])
    Omega, Sigma, Lambda = _long_run_cov(stack, lags=lags)
    omega_uy = Omega[0, 1:]
    Omega_xx = Omega[1:, 1:]

    # Endogeneity correction
    try:
        Omega_xx_inv = np.linalg.inv(Omega_xx)
    except np.linalg.LinAlgError:
        Omega_xx_inv = np.linalg.pinv(Omega_xx)
    y_plus = y[1:] - dX @ Omega_xx_inv @ omega_uy

    lambda_uy = Lambda[0, 1:]
    lambda_xx = Lambda[1:, 1:]
    bias = lambda_uy - omega_uy @ Omega_xx_inv @ lambda_xx

    Z_plus = Z[1:]
    Xc = X[1:] - X[1:].mean(axis=0)
    yc = y_plus - y_plus.mean()
    XtX_inv = inv_xtx(Xc, name="fm_ols")
    beta_fm = XtX_inv @ (Xc.T @ yc - (T - 1) * bias)
    alpha_fm = float(y_plus.mean() - X[1:].mean(axis=0) @ beta_fm)

    # Standard errors: omega_uu.x = Omega[0,0] - omega_uy Omega_xx_inv omega_uy
    omega_uu_x = float(Omega[0, 0] - omega_uy @ Omega_xx_inv @ omega_uy)
    var_beta = omega_uu_x * XtX_inv
    se_beta = np.sqrt(np.maximum(np.diag(var_beta), 0.0))

    resid = y[1:] - alpha_fm - X[1:] @ beta_fm
    return FMOLSResult(
        beta=beta_fm,
        alpha=alpha_fm,
        se=se_beta,
        residuals=resid,
        Omega=Omega,
        Sigma=Sigma,
        Lambda=Lambda,
    )


# ---------------------------------------------------------------------------
# Stock-Watson DOLS
# ---------------------------------------------------------------------------
def dols(y, X, leads: int = 2, lags: int = 2, hac_lags: int | None = None) -> DOLSResult:
    """Stock-Watson (1993) DOLS: y_t = alpha + X_t' beta + sum_k gamma_k dX_{t-k} + e_t.

    Long-run-bias-free OLS estimate of the cointegrating vector via
    augmenting the regression with leads and lags of differences of X.
    Uses Newey-West HAC for SE.

    Returns
    -------
    DOLSResult
        See :class:`DOLSResult` for fields.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(X if np.asarray(X).ndim > 1 else np.asarray(X)[:, None])
    X = X.astype(float)
    T, k = X.shape
    dX = np.diff(X, axis=0)  # length T-1
    rows = []
    targets = []
    for t in range(lags, T - leads):
        feat = [1.0]
        feat.extend(X[t])
        for s in range(-lags, leads + 1):
            if s == 0:
                continue
            feat.extend(dX[t + s - 1] if 0 <= t + s - 1 < dX.shape[0]
                         else np.zeros(k))
        rows.append(feat)
        targets.append(y[t])
    Xd = np.array(rows)
    yd = np.array(targets)
    if hac_lags is None:
        hac_lags = int(np.floor(4 * (len(yd) / 100) ** (2 / 9)))
    out = ols_hac(yd, Xd, lags=hac_lags)
    beta = out["beta"]
    se = out["se"]
    return DOLSResult(
        alpha=float(beta[0]),
        beta=beta[1:1 + k],
        se=se[1:1 + k],
        alpha_se=float(se[0]),
        n_obs=len(yd),
    )


# ---------------------------------------------------------------------------
# Phillips-Ouliaris residual cointegration tests
# ---------------------------------------------------------------------------
def phillips_ouliaris(y, X, lags: int | None = None) -> PhillipsOuliarisResult:
    """Phillips-Ouliaris (1990) Z_alpha and Z_t cointegration tests.

    Step 1: OLS y = alpha + X beta + u_hat.
    Step 2: Test ADF-style on u_hat with PP correction.

    Returns
    -------
    PhillipsOuliarisResult
        See :class:`PhillipsOuliarisResult` for fields.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(X if np.asarray(X).ndim > 1 else np.asarray(X)[:, None])
    X = X.astype(float)
    T = len(y)
    Z = np.column_stack([np.ones(T), X])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    u = y - Z @ beta

    du = np.diff(u)
    u_lag = u[:-1]
    rho = float((u_lag @ du) / (u_lag @ u_lag))
    eps = du - rho * u_lag
    n = len(eps)
    sigma2 = float(eps @ eps / n)
    if lags is None:
        lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    g0 = sigma2
    s2 = g0
    for j in range(1, lags + 1):
        if j >= n:
            break
        w = 1.0 - j / (lags + 1.0)
        cov = float(eps[j:] @ eps[:-j]) / n
        s2 += 2.0 * w * cov
    s2 = max(s2, 1e-12)

    # Z_alpha = T * rho - 0.5 * (s2 - g0) / (mean(u^2_lag) / T)
    denom_a = float(u_lag @ u_lag) / T
    z_alpha = T * rho - 0.5 * (s2 - g0) / max(denom_a, 1e-12)
    se_rho = np.sqrt(g0 / (u_lag @ u_lag))
    t_stat = rho / se_rho
    z_t = np.sqrt(g0 / s2) * t_stat - 0.5 * (s2 - g0) * np.sqrt(n) * se_rho / s2

    # Phillips-Ouliaris (1990) Table II 5% CV (constant case, k=1 regressor): -3.37
    # Cite as approximate; for k>1 swap from PO (1990) Table II.
    return PhillipsOuliarisResult(
        z_alpha=float(z_alpha),
        z_t=float(z_t),
        residuals=u,
        rho=rho,
        lag_used=int(lags),
    )


__all__ = [
    "fm_ols", "dols", "phillips_ouliaris",
    "FMOLSResult", "DOLSResult", "PhillipsOuliarisResult",
]
