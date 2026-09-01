"""DCC(1,1) — Engle (2002) Dynamic Conditional Correlation.

Two-stage estimator:
    Stage 1: fit GARCH(1,1) per series, obtain sigma_t and standardized
             residuals e_t = u_t / sigma_t.
    Stage 2: maximise the correlation log-likelihood
                ell(a, b) = -1/2 sum_t [ log|R_t| + e_t' R_t^{-1} e_t - e_t' e_t ]
             where
                Q_t = (1 - a - b) Qbar + a e_{t-1} e_{t-1}' + b Q_{t-1}
                R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}.

Conditional covariance: H_t = D_t R_t D_t with D_t = diag(sigma_t).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .fit import garch11_fit
from ._results import DCCResult


def _dcc_recursion(e: np.ndarray, a: float, b: float, Qbar: np.ndarray):
    """Compute Q_t and R_t for t = 0..T-1 given (a, b, Qbar)."""
    T, n = e.shape
    Q = np.empty((T, n, n))
    R = np.empty((T, n, n))
    Q[0] = Qbar.copy()
    inv_sqrt = 1.0 / np.sqrt(np.diag(Q[0]))
    R[0] = (Q[0] * inv_sqrt[:, None]) * inv_sqrt[None, :]
    for t in range(1, T):
        Q[t] = ((1.0 - a - b) * Qbar
                + a * np.outer(e[t - 1], e[t - 1])
                + b * Q[t - 1])
        d = np.sqrt(np.diag(Q[t]))
        inv_sqrt = 1.0 / np.maximum(d, 1e-12)
        R[t] = (Q[t] * inv_sqrt[:, None]) * inv_sqrt[None, :]
    return Q, R


def _dcc_loglik(params: np.ndarray, e: np.ndarray, Qbar: np.ndarray) -> float:
    a, b = params
    if a < 0 or b < 0 or a + b >= 0.999:
        return 1e10
    _, R = _dcc_recursion(e, a, b, Qbar)
    sign, logdet = np.linalg.slogdet(R)
    if np.any(sign <= 0):
        return 1e10
    try:
        x = np.linalg.solve(R, np.expand_dims(e, axis=-1)).squeeze(-1)
    except np.linalg.LinAlgError:
        return 1e10
    quad = (e * x).sum(axis=-1)
    # e @ e for each t is just (e * e).sum(axis=-1)
    ll = -0.5 * np.sum(logdet + quad - (e * e).sum(axis=-1))
    return float(-ll)


def dcc_fit(returns: pd.DataFrame, mean: str = "zero") -> DCCResult:
    """Fit DCC(1,1) on a multivariate return panel.

    Parameters
    ----------
    returns : pd.DataFrame
        T x n panel of (mean-removed) return / shock series.
    mean : {"zero", "constant"}
        Passed through to the per-asset GARCH(1,1) fit.

    Returns
    -------
    DCCResult
        Frozen dataclass with fields ``a``, ``b``, ``Qbar``, ``sigma``
        (pd.DataFrame), ``R``, ``H``, ``garch_params`` (list of per-
        asset dicts), ``loglik``, ``converged``.

    References
    ----------
    Engle, R. (2002). Dynamic conditional correlation: a simple class
        of multivariate generalized autoregressive conditional
        heteroskedasticity models. JBES 20(3), 339-350.
    """
    arr = np.asarray(returns, dtype=float)
    T, n = arr.shape
    sigma = np.empty_like(arr)
    garch_params = []
    for i in range(n):
        gi = garch11_fit(arr[:, i], mean=mean)
        sigma[:, i] = gi.sigma.values
        garch_params.append({k: getattr(gi, k) for k in ("omega", "alpha", "beta",
                                                          "persistence", "loglik")})
    e = arr / np.maximum(sigma, 1e-12)
    Qbar = (e.T @ e) / T

    res = minimize(
        _dcc_loglik, x0=np.array([0.05, 0.90]),
        args=(e, Qbar),
        method="L-BFGS-B",
        bounds=[(1e-6, 0.999), (1e-6, 0.999)],
    )
    a, b = res.x
    _, R = _dcc_recursion(e, a, b, Qbar)
    H = np.empty_like(R)
    for t in range(T):
        D = np.diag(sigma[t])
        H[t] = D @ R[t] @ D
    return DCCResult(
        a=float(a),
        b=float(b),
        Qbar=Qbar,
        sigma=pd.DataFrame(sigma, index=returns.index, columns=returns.columns),
        R=R,
        H=H,
        garch_params=garch_params,
        loglik=float(-res.fun),
        converged=bool(res.success),
    )


__all__ = ["dcc_fit"]
