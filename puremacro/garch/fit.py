"""GARCH(1,1) Gaussian MLE — pure numpy/scipy. Replaces arch.arch_model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ._results import GARCH11Result


def _filter_sigma2(eps: np.ndarray, omega: float, alpha: float, beta: float,
                   var0: float) -> np.ndarray:
    """Recursive σ²_t = ω + α u²_{t-1} + β σ²_{t-1} with sigma2[0] = var0."""
    T = len(eps)
    sigma2 = np.empty(T)
    sigma2[0] = var0
    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    return np.maximum(sigma2, 1e-10)


def _neg_loglik(params, eps, var0):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e10
    sigma2 = _filter_sigma2(eps, omega, alpha, beta, var0)
    ll = 0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps ** 2 / sigma2)
    return float(ll)


def garch11_fit(returns, mean: str = "zero") -> GARCH11Result:
    """Fit GARCH(1,1) by Gaussian MLE.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Mean-zero (or mean-removed) returns / shocks.
    mean : {"zero", "constant"}
        If "constant", subtract the sample mean before filtering.

    Returns
    -------
    GARCH11Result
        Frozen dataclass with fields ``omega``, ``alpha``, ``beta``,
        ``sigma`` (pd.Series), ``loglik``, ``converged``, ``persistence``.

    References
    ----------
    Bollerslev, T. (1986). Generalized autoregressive conditional
        heteroskedasticity. Journal of Econometrics 31(3), 307-327.
    """
    if isinstance(returns, pd.Series):
        idx = returns.index
        eps = returns.values.astype(float)
    else:
        idx = None
        eps = np.asarray(returns, dtype=float)
    if mean == "constant":
        eps = eps - eps.mean()
    var0 = float(np.var(eps))
    if var0 <= 0:
        raise ValueError("returns have zero variance — cannot fit GARCH(1,1)")
    x0 = [0.05 * var0, 0.10, 0.85]
    bounds = [(1e-8, None), (0.0, 0.999), (0.0, 0.999)]
    res = minimize(_neg_loglik, x0, args=(eps, var0), method="L-BFGS-B",
                   bounds=bounds)
    omega, alpha, beta = res.x
    sigma2 = _filter_sigma2(eps, omega, alpha, beta, var0)
    sigma = np.sqrt(sigma2)
    sigma_series = pd.Series(sigma, index=idx) if idx is not None else pd.Series(sigma)
    return GARCH11Result(
        omega=float(omega),
        alpha=float(alpha),
        beta=float(beta),
        sigma=sigma_series,
        loglik=float(-res.fun),
        converged=bool(res.success),
        persistence=float(alpha + beta),
    )


__all__ = ["garch11_fit"]
