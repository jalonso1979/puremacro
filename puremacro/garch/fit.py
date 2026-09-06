"""GARCH(1,1) Gaussian MLE — pure numpy/scipy. Replaces arch.arch_model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from ._results import GARCH11Result

_VALID_MEANS = ("zero", "constant")


def _filter_sigma2(eps: np.ndarray, omega: float, alpha: float, beta: float,
                   var0: float) -> np.ndarray:
    """Recursive σ²_t = ω + α u²_{t-1} + β σ²_{t-1} with sigma2[0] = var0.

    The recursion is an AR(1) in σ² driven by ``ω + α u²_{t-1}``, so it is
    evaluated with :func:`scipy.signal.lfilter` (C loop) instead of a Python
    loop: the same arithmetic ``(ω + α u²) + β σ²_{t-1}`` in the same order,
    but ~50x faster, which is what makes the Nelder-Mead polish in
    :func:`garch11_fit` affordable.

    The positivity floor is *relative* to ``var0`` so that the filter is
    exactly scale-equivariant (``σ²(c·u) = c² σ²(u)``); with ω bounded away
    from zero it never binds in practice.
    """
    T = len(eps)
    sigma2 = np.empty(T)
    sigma2[0] = var0
    if T > 1:
        drive = omega + alpha * eps[:-1] ** 2
        sigma2[1:] = lfilter([1.0], [1.0, -beta], drive, zi=np.array([beta * var0]))[0]
    return np.maximum(sigma2, 1e-10 * var0)


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
        Mean-zero (or mean-removed) returns / shocks. Must be finite; a
        series containing NaN/inf raises ``ValueError`` (drop or impute
        missing observations before fitting).
    mean : {"zero", "constant"}
        If "constant", subtract the sample mean before filtering. Any other
        value raises ``ValueError``.

    Returns
    -------
    GARCH11Result
        Frozen dataclass with fields ``omega``, ``alpha``, ``beta``,
        ``sigma`` (pd.Series), ``loglik``, ``converged``, ``persistence``.

    Notes
    -----
    The Gaussian GARCH(1,1) likelihood is scale-equivariant: multiplying the
    series by ``c`` leaves ``alpha`` and ``beta`` unchanged, multiplies
    ``omega`` and every ``sigma_t`` by ``c`` (``omega`` by ``c**2``) and
    shifts the log-likelihood by ``-T log c``. The optimiser therefore works
    on the standardised series ``returns / sd(returns)`` (unit variance, the
    scale the starting values and L-BFGS-B tolerances were designed for)
    and maps ``omega`` back by ``sd**2``. A Nelder-Mead polish with tight
    tolerances follows the L-BFGS-B stage so that the returned point is the
    maximum of its own likelihood whatever the units of the input (decimal
    daily returns with variance ~1e-4, percent returns, raw first
    differences, ...). The reported ``sigma`` and ``loglik`` are evaluated
    on the original scale with ``sigma2[0] = var(returns)``.

    References
    ----------
    Bollerslev, T. (1986). Generalized autoregressive conditional
        heteroskedasticity. Journal of Econometrics 31(3), 307-327.
    """
    if mean not in _VALID_MEANS:
        raise ValueError(
            f"garch11_fit: mean must be one of {_VALID_MEANS}, got {mean!r}"
        )
    if isinstance(returns, pd.Series):
        idx = returns.index
        eps = returns.values.astype(float)
    else:
        idx = None
        eps = np.asarray(returns, dtype=float)
    if eps.ndim != 1:
        if eps.size == max(eps.shape, default=0):
            eps = eps.ravel()
        else:
            raise ValueError(
                f"garch11_fit: returns must be one-dimensional, got shape {eps.shape}"
            )
    if eps.size < 2:
        raise ValueError(
            f"garch11_fit: need at least 2 observations, got {eps.size}"
        )
    if not np.all(np.isfinite(eps)):
        n_bad = int(np.sum(~np.isfinite(eps)))
        raise ValueError(
            f"garch11_fit: returns contain {n_bad} non-finite value(s) (NaN/inf); "
            "drop or impute them before fitting"
        )
    if mean == "constant":
        eps = eps - eps.mean()
    var0 = float(np.var(eps))
    if var0 <= 0:
        raise ValueError("returns have zero variance — cannot fit GARCH(1,1)")

    # --- optimise on the standardised series (unit variance) -----------------
    scale = float(np.sqrt(var0))
    eps_s = eps / scale
    var0_s = float(np.var(eps_s))            # == 1 up to rounding
    x0 = np.array([0.05 * var0_s, 0.10, 0.85])
    bounds = [(1e-8, None), (0.0, 0.999), (0.0, 0.999)]
    res_lb = minimize(_neg_loglik, x0, args=(eps_s, var0_s), method="L-BFGS-B",
                      bounds=bounds, options={"ftol": 1e-12, "gtol": 1e-8,
                                              "maxiter": 1000})
    # Polish: L-BFGS-B's finite-difference gradient and relative ftol stop
    # early on the flat, ill-scaled (omega vs alpha, beta) surface; a
    # derivative-free Nelder-Mead from its endpoint reaches the maximum to
    # ~1e-8 in the parameters at a cost of a few hundred cheap evaluations.
    res_nm = minimize(_neg_loglik, res_lb.x, args=(eps_s, var0_s),
                      method="Nelder-Mead",
                      options={"xatol": 1e-9, "fatol": 1e-10,
                               "maxiter": 5000, "maxfev": 10000})
    if res_nm.fun <= res_lb.fun:
        best_x, converged = res_nm.x, bool(res_nm.success)
    else:                                   # pragma: no cover - NM never worsens
        best_x, converged = res_lb.x, bool(res_lb.success)
    omega_s, alpha, beta = (float(v) for v in best_x)
    omega = omega_s * scale ** 2

    # --- report on the original scale ---------------------------------------
    sigma2 = _filter_sigma2(eps, omega, alpha, beta, var0)
    sigma = np.sqrt(sigma2)
    loglik = -_neg_loglik([omega, alpha, beta], eps, var0)
    if not np.isfinite(loglik) or loglik <= -1e9:
        converged = False
    sigma_series = pd.Series(sigma, index=idx) if idx is not None else pd.Series(sigma)
    return GARCH11Result(
        omega=float(omega),
        alpha=float(alpha),
        beta=float(beta),
        sigma=sigma_series,
        loglik=float(loglik),
        converged=converged,
        persistence=float(alpha + beta),
    )


__all__ = ["garch11_fit"]
