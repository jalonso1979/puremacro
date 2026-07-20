"""Multivariate GARCH(1,1): BEKK and CCC.

Companion to ``puremacro.garch.dcc`` (DCC, dynamic conditional correlations).

  BEKK (Engle-Kroner 1995):
      H_t = C C' + A' u_{t-1} u_{t-1}' A + B' H_{t-1} B

      Parameters: lower-triangular ``C`` (n(n+1)/2), full ``A``, ``B``
      (each n²). Total = n(n+1)/2 + 2 n² parameters.

  CCC (Bollerslev 1990):
      H_t = D_t · R · D_t,    D_t = diag(h_{1,t}^{1/2}, …),
      h_{i,t} = ω_i + α_i u_{i,t-1}² + β_i h_{i,t-1},
      R       = constant correlation of standardised residuals.

      Parameters: per-equation univariate GARCH(1,1) coefficients
      ``(ω, α, β)`` plus the off-diagonal entries of ``R``.

The constant-correlation CCC is fast and parsimonious; BEKK is more
flexible but the parameter count grows with n². Both share a Gaussian
log-likelihood for the structural shock.

References
----------
Engle, R.F. and Kroner, K.F. (1995). Multivariate simultaneous
    generalized ARCH. Econometric Theory 11(1), 122-150.
Bollerslev, T. (1990). Modelling the coherence in short-run nominal
    exchange rates: a multivariate GARCH approach. RES 72, 498-505.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .._linalg import safe_cholesky
from ..garch.fit import garch11_fit


# ---------------------------------------------------------------------------
# CCC
# ---------------------------------------------------------------------------


def ccc_fit(U: np.ndarray) -> dict:
    """Constant-Conditional-Correlation GARCH (Bollerslev 1990).

    Two-step estimator:
      1. Fit a univariate GARCH(1,1) per series via QMLE.
      2. Compute the unconditional correlation of the standardised
         residuals  ε_t = u_t / h_t^{1/2}; that becomes ``R``.

    Parameters
    ----------
    U : (T, n) ndarray
        Mean-zero residuals or log-returns.

    Returns
    -------
    dict with keys ``omega`` (n,), ``alpha`` (n,), ``beta`` (n,),
    ``R`` (n, n), ``H`` (T, n, n) conditional covariance path,
    ``loglik`` (Gaussian, system).
    """
    U = np.asarray(U, dtype=float)
    T, n = U.shape
    omega = np.empty(n)
    alpha = np.empty(n)
    beta = np.empty(n)
    h = np.empty((T, n))                     # conditional variances
    for i in range(n):
        fit = garch11_fit(U[:, i])
        omega[i] = fit.omega
        alpha[i] = fit.alpha
        beta[i] = fit.beta
        h[:, i] = np.asarray(fit.sigma, dtype=float) ** 2
    eps = U / np.sqrt(np.maximum(h, 1e-12))
    R: np.ndarray = np.asarray(np.corrcoef(eps, rowvar=False))
    R = (R + R.T) / 2.0
    np.fill_diagonal(R, 1.0)

    H = np.empty((T, n, n))
    for t in range(T):
        D = np.diag(np.sqrt(np.maximum(h[t], 1e-12)))
        H[t] = D @ R @ D

    # Gaussian log-lik (per-period)
    loglik = 0.0
    for t in range(T):
        try:
            L = safe_cholesky(H[t], name="ccc_fit H_t", jitter=1e-10)
        except np.linalg.LinAlgError:
            return {
                "omega": omega, "alpha": alpha, "beta": beta,
                "R": R, "H": H, "loglik": float("nan"),
            }
        ldet = 2.0 * np.sum(np.log(np.diag(L)))
        z = np.linalg.solve(L, U[t])
        loglik += -0.5 * (n * np.log(2 * np.pi) + ldet + z @ z)
    return {
        "omega": omega, "alpha": alpha, "beta": beta,
        "R": R, "H": H, "loglik": float(loglik),
    }


# ---------------------------------------------------------------------------
# BEKK
# ---------------------------------------------------------------------------


def _bekk_unpack(theta: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape the parameter vector into (C, A, B) where C is lower
    triangular, A and B are full n×n."""
    n_C = n * (n + 1) // 2
    c_lt = theta[:n_C]
    a_flat = theta[n_C:n_C + n * n]
    b_flat = theta[n_C + n * n:]
    C = np.zeros((n, n))
    idx = np.tril_indices(n)
    C[idx] = c_lt
    A = a_flat.reshape(n, n)
    B = b_flat.reshape(n, n)
    return C, A, B


def _bekk_filter(
    U: np.ndarray, C: np.ndarray, A: np.ndarray, B: np.ndarray,
    H0: np.ndarray,
) -> np.ndarray:
    """Run the BEKK(1,1) recursion forward."""
    T, n = U.shape
    H = np.empty((T, n, n))
    H[0] = H0
    CC = C @ C.T
    for t in range(1, T):
        u_prev = U[t - 1]
        H[t] = CC + A.T @ np.outer(u_prev, u_prev) @ A + B.T @ H[t - 1] @ B
        H[t] = (H[t] + H[t].T) / 2.0
    return H


def _bekk_neg_loglik(theta: np.ndarray, U: np.ndarray, H0: np.ndarray) -> float:
    T, n = U.shape
    C, A, B = _bekk_unpack(theta, n)
    H = _bekk_filter(U, C, A, B, H0)
    nll = 0.0
    for t in range(T):
        try:
            L = safe_cholesky(H[t], name="bekk_filter H_t", jitter=1e-10)
        except np.linalg.LinAlgError:
            return 1e10
        ldet = 2.0 * np.sum(np.log(np.diag(L)))
        z = np.linalg.solve(L, U[t])
        nll += 0.5 * (n * np.log(2 * np.pi) + ldet + z @ z)
    return float(nll)


def bekk_fit(U: np.ndarray, *, max_iter: int = 200) -> dict:
    """Fit a BEKK(1,1) by Gaussian QMLE.

    Initial values follow Engle-Kroner (1995): C from Cholesky of the
    sample covariance scaled down (so its quadratic form gives a small
    intercept), A and B set to scaled identities ``α^{1/2} I`` and
    ``β^{1/2} I`` with α = 0.05, β = 0.90 — a typical persistent-
    GARCH calibration.

    Parameters
    ----------
    U : (T, n) ndarray
        Mean-zero residuals.
    max_iter : int
        Hard cap on L-BFGS-B iterations.

    Returns
    -------
    dict: ``C, A, B, H, loglik, converged, n_iter``.
    """
    U = np.asarray(U, dtype=float)
    T, n = U.shape
    Sigma_sample = np.cov(U, rowvar=False)
    Sigma_sample = (Sigma_sample + Sigma_sample.T) / 2.0
    H0 = Sigma_sample.copy()

    # Initial CC' = (1 - α - β) Σ_sample (variance targeting), so
    # C = chol((1 - α - β) Σ_sample).
    a0, b0 = 0.05, 0.90
    target = max(1.0 - a0 - b0, 1e-3) * Sigma_sample
    target = (target + target.T) / 2.0
    L0 = safe_cholesky(target, name="bekk_fit init", jitter=1e-8)
    A0 = np.sqrt(a0) * np.eye(n)
    B0 = np.sqrt(b0) * np.eye(n)
    n_C = n * (n + 1) // 2
    theta0 = np.empty(n_C + 2 * n * n)
    theta0[:n_C] = L0[np.tril_indices(n)]
    theta0[n_C:n_C + n * n] = A0.ravel()
    theta0[n_C + n * n:] = B0.ravel()

    res = minimize(
        _bekk_neg_loglik, theta0, args=(U, H0),
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    C, A, B = _bekk_unpack(res.x, n)
    H = _bekk_filter(U, C, A, B, H0)
    return {
        "C": C, "A": A, "B": B,
        "H": H,
        "loglik": float(-res.fun),
        "converged": bool(res.success),
        "n_iter": int(res.nit) if hasattr(res, "nit") else None,
    }


__all__ = ["ccc_fit", "bekk_fit"]
