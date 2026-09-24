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
    ll = 0.0
    for t in range(R.shape[0]):
        sign, logdet = np.linalg.slogdet(R[t])
        if sign <= 0:
            return 1e10
        try:
            quad = e[t] @ np.linalg.solve(R[t], e[t])
        except np.linalg.LinAlgError:
            return 1e10
        ll += -0.5 * (logdet + quad - e[t] @ e[t])
    return float(-ll)


def dcc_fit(returns: pd.DataFrame, mean: str = "zero") -> DCCResult:
    """Fit DCC(1,1) on a multivariate return panel.

    Parameters
    ----------
    returns : pd.DataFrame or ndarray, shape (T, n)
        T x n panel of (mean-removed) return / shock series. A DataFrame
        keeps its index / column labels in ``sigma``; an ndarray gets a
        RangeIndex and integer column labels.
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
    if arr.ndim != 2:
        raise ValueError(
            f"dcc_fit: returns must be a (T, n) panel, got shape {arr.shape}"
        )
    T, n = arr.shape
    if mean not in ("zero", "constant"):
        raise ValueError(f"mean must be 'zero' or 'constant', got {mean!r}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "dcc_fit: returns contain NaN/inf; drop or impute them before fitting"
        )
    # The docstring asks for a DataFrame, but a plain (T, n) ndarray used to
    # crash on ``returns.index`` at the very end (after all the fitting):
    # accept both, defaulting to a RangeIndex and integer column labels.
    if isinstance(returns, pd.DataFrame):
        index, columns = returns.index, returns.columns
    else:
        index, columns = pd.RangeIndex(T), pd.RangeIndex(n)
    # Step 2 of Engle's two-step needs z_t = eps_t / h_t^{1/2} with the SAME
    # eps_t in the numerator whose conditional variance is in the denominator.
    # `garch11_fit(mean="constant")` demeans internally and `GARCH11Result` has
    # no `mu` field, so the demeaned series never came back out: this used to
    # divide the raw `arr` by a sigma fitted to the demeaned one, leaving the
    # mean inside z. The resulting Qbar then measured mu_i * mu_j rather than a
    # correlation — plim m^2/(m^2+1) for two INDEPENDENT series with
    # m = mu/sd, i.e. 0.96 at m = 5. Demeaning once here, and passing the
    # already-demeaned series to a zero-mean GARCH, keeps one definition of
    # eps_t. The `mean="zero"` path is bit-identical to before.
    mu = arr.mean(axis=0) if mean == "constant" else np.zeros(n)
    resid = arr - mu
    sigma = np.empty_like(arr)
    garch_params = []
    for i in range(n):
        gi = garch11_fit(resid[:, i], mean="zero")
        sigma[:, i] = gi.sigma.values
        garch_params.append({k: getattr(gi, k) for k in ("omega", "alpha", "beta",
                                                          "persistence", "loglik")})
    e = resid / np.maximum(sigma, 1e-12)
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
        sigma=pd.DataFrame(sigma, index=index, columns=columns),
        R=R,
        H=H,
        garch_params=garch_params,
        loglik=float(-res.fun),
        converged=bool(res.success),
    )


__all__ = ["dcc_fit"]
