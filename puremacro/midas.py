"""Mixed-frequency regression (MIDAS).

Two flavours:
- ``u_midas``  : unrestricted MIDAS — the K high-frequency lags enter
                  with separate coefficients (Foroni-Marcellino-Schumacher 2015).
                  OLS, easy, but parameter count grows with K.
- ``beta_midas`` : Beta-polynomial MIDAS (Ghysels-Santa-Clara-Valkanov 2007).
                  Imposes a two-parameter lag-weighting kernel
                      w(k) = beta_pdf((k-1)/(K-1); theta1, theta2),
                  so the low-frequency slope is identified from a single
                  scalar ``beta`` regardless of K.

Both functions assume the high-frequency series ``x_hf`` has been
pre-aligned so that ``x_hf[i*K + j]`` is the j-th sub-period of
low-frequency period i. ``y_lf[i]`` is the contemporaneous (or lagged)
low-frequency observation.

References
----------
Ghysels, E., Santa-Clara, P. and Valkanov, R. (2007). MIDAS regressions:
    further results and new directions. Econometric Reviews 26.
Foroni, C., Marcellino, M. and Schumacher, C. (2015). Unrestricted
    mixed data sampling (MIDAS): MIDAS regressions with unrestricted
    lag polynomials. JRSS-A 178(1), 57-82.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist


@dataclass(frozen=True)
class UMidasResult:
    """Result of :func:`u_midas` (unrestricted MIDAS).

    Attributes
    ----------
    intercept : float
        Estimated intercept.
    beta : np.ndarray
        Per-lag MIDAS coefficients, length ``K * (1 + n_low_lags)``.
    fitted : np.ndarray
        In-sample fitted values, length ``n_obs``.
    residuals : np.ndarray
        In-sample residuals, length ``n_obs``.
    R2 : float
        In-sample R².
    n_obs : int
        Effective sample size used in the regression.

    References
    ----------
    Foroni, C., Marcellino, M. and Schumacher, C. (2015). Unrestricted
        mixed data sampling (MIDAS): MIDAS regressions with unrestricted
        lag polynomials. JRSS-A 178(1), 57-82.
    """

    intercept: float
    beta: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    R2: float
    n_obs: int

    def summary(self) -> str:
        coefs = ", ".join(f"{b:+.3f}" for b in self.beta)
        return (
            f"U-MIDAS (Foroni-Marcellino-Schumacher)\n"
            f"  intercept         : {self.intercept:+.4f}\n"
            f"  beta              : {coefs}\n"
            f"  R²                : {self.R2:.4f}\n"
            f"  n_obs             : {self.n_obs}\n"
        )


@dataclass(frozen=True)
class BetaMidasResult:
    """Result of :func:`beta_midas` (Beta-polynomial MIDAS).

    Attributes
    ----------
    intercept : float
        Estimated intercept.
    beta : float
        Single low-frequency slope on the kernel-weighted high-frequency sum.
    theta1, theta2 : float
        Beta-pdf shape parameters (positive).
    weights : np.ndarray
        Kernel weights ``w(k; theta1, theta2)``, length ``K``, sum to 1.
    fitted : np.ndarray
        In-sample fitted values, length ``n_low``.
    residuals : np.ndarray
        In-sample residuals, length ``n_low``.
    R2 : float
        In-sample R².
    converged : bool
        Whether scipy.optimize.minimize converged.

    References
    ----------
    Ghysels, E., Santa-Clara, P. and Valkanov, R. (2007). MIDAS regressions:
        further results and new directions. Econometric Reviews 26.
    """

    intercept: float
    beta: float
    theta1: float
    theta2: float
    weights: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    R2: float
    converged: bool

    def summary(self) -> str:
        w = ", ".join(f"{x:.3f}" for x in self.weights)
        return (
            f"Beta-MIDAS (Ghysels-Santa-Clara-Valkanov)\n"
            f"  intercept         : {self.intercept:+.4f}\n"
            f"  beta              : {self.beta:+.4f}\n"
            f"  theta1, theta2    : {self.theta1:.3f}, {self.theta2:.3f}\n"
            f"  weights           : {w}\n"
            f"  R²                : {self.R2:.4f}\n"
            f"  converged         : {self.converged}\n"
        )


def _stack_hf_lags(x_hf: np.ndarray, K: int, n_periods: int,
                    n_low_lags: int = 0) -> np.ndarray | None:
    """Build the design matrix of high-frequency lags.

    For each low-frequency period i in [n_low_lags, n_periods), the
    matrix row holds ``x_hf`` at sub-periods (i*K - 1, i*K - 2, ..., i*K - K)
    — the K most-recent observations *strictly before* the sub-period
    just begun.  When ``n_low_lags > 0`` we extend to cover n_low_lags
    full prior low-frequency periods.
    """
    rows = []
    total_lags = K * (1 + n_low_lags)
    for i in range(n_low_lags + 1, n_periods + 1):
        end = i * K
        start = end - total_lags
        if start < 0:
            return None
        rows.append(x_hf[start:end][::-1])  # most-recent first
    return np.array(rows)


def u_midas(
    y_lf: np.ndarray,
    x_hf: np.ndarray,
    K: int,
    n_low_lags: int = 0,
) -> UMidasResult:
    """Unrestricted MIDAS (Foroni-Marcellino-Schumacher).

    Parameters
    ----------
    y_lf : (n_low,) ndarray of low-frequency observations.
    x_hf : (n_low * K,) ndarray of aligned high-frequency observations.
    K : int — sub-periods per low-frequency period (e.g. 3 for QM).
    n_low_lags : int — additional low-frequency lags of x_hf to include.

    Returns
    -------
    UMidasResult
        Frozen dataclass with intercept, beta, fitted, residuals, R2, n_obs.
    """
    y_lf = np.asarray(y_lf, dtype=float).ravel()
    x_hf = np.asarray(x_hf, dtype=float).ravel()
    n_low = len(y_lf)
    if len(x_hf) != n_low * K:
        raise ValueError(f"x_hf length must be n_low*K = {n_low * K}; "
                         f"got {len(x_hf)}")
    X = _stack_hf_lags(x_hf, K, n_low, n_low_lags=n_low_lags)
    if X is None:
        raise ValueError("Not enough high-frequency observations for the "
                         "requested n_low_lags.")
    y_aligned = y_lf[n_low_lags:]
    n_obs = len(y_aligned)
    X1 = np.column_stack([np.ones(n_obs), X])
    beta = np.linalg.lstsq(X1, y_aligned, rcond=None)[0]
    fitted = X1 @ beta
    resid = y_aligned - fitted
    rss = float(resid @ resid)
    tss = float(((y_aligned - y_aligned.mean()) ** 2).sum())
    return UMidasResult(
        intercept=float(beta[0]),
        beta=beta[1:],
        fitted=fitted,
        residuals=resid,
        R2=1.0 - rss / tss if tss > 0 else 0.0,
        n_obs=n_obs,
    )


def _beta_weights(K: int, theta1: float, theta2: float) -> np.ndarray:
    """Beta-pdf weights on K nodes (k=1..K), normalised to sum to 1."""
    if theta1 <= 0 or theta2 <= 0:
        return np.full(K, 1.0 / K)
    grid = (np.arange(1, K + 1) - 0.5) / K  # midpoints in (0,1)
    w = grid ** (theta1 - 1) * (1.0 - grid) ** (theta2 - 1)
    return w / w.sum()


def beta_midas(
    y_lf: np.ndarray,
    x_hf: np.ndarray,
    K: int,
    *,
    theta1_init: float = 1.0,
    theta2_init: float = 5.0,
) -> BetaMidasResult:
    """Beta-polynomial MIDAS by NLS.

    Model
    -----
        y_i = alpha + beta * sum_{k=1..K} w(k; theta1, theta2) * x_hf[(i-1)*K + (K-k)]
            + eps_i

    with w(k) the Beta-pdf-style kernel.  alpha, beta are linear; theta1,
    theta2 enter the kernel non-linearly and are estimated by minimising
    the SSR.

    Returns
    -------
    BetaMidasResult
        Frozen dataclass with intercept, beta, theta1, theta2, weights,
        fitted, residuals, R2, converged.
    """
    y_lf = np.asarray(y_lf, dtype=float).ravel()
    x_hf = np.asarray(x_hf, dtype=float).ravel()
    n_low = len(y_lf)
    X_hf = _stack_hf_lags(x_hf, K, n_low, n_low_lags=0)
    if X_hf is None:
        raise ValueError("x_hf too short.")

    def _loss(params):
        t1, t2 = params
        if t1 <= 0 or t2 <= 0:
            return 1e10
        w = _beta_weights(K, t1, t2)
        z = X_hf @ w
        Z1 = np.column_stack([np.ones(len(z)), z])
        try:
            beta_lin, *_ = np.linalg.lstsq(Z1, y_lf, rcond=None)
        except np.linalg.LinAlgError:
            return 1e10
        resid = y_lf - Z1 @ beta_lin
        return float(resid @ resid)

    res = minimize(
        _loss, x0=np.array([theta1_init, theta2_init]),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 1000},
    )
    t1, t2 = res.x
    w_hat = _beta_weights(K, t1, t2)
    z = X_hf @ w_hat
    Z1 = np.column_stack([np.ones(len(z)), z])
    beta_lin, *_ = np.linalg.lstsq(Z1, y_lf, rcond=None)
    fitted = Z1 @ beta_lin
    resid = y_lf - fitted
    rss = float(resid @ resid); tss = float(((y_lf - y_lf.mean()) ** 2).sum())
    return BetaMidasResult(
        intercept=float(beta_lin[0]),
        beta=float(beta_lin[1]),
        theta1=float(t1),
        theta2=float(t2),
        weights=w_hat,
        fitted=fitted,
        residuals=resid,
        R2=1.0 - rss / tss if tss > 0 else 0.0,
        converged=bool(res.success),
    )


__all__ = ["u_midas", "beta_midas", "UMidasResult", "BetaMidasResult"]
