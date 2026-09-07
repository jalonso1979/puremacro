"""Simulation-Extrapolation (SIMEX) estimator for errors-in-variables OLS.

References
----------
Cook, J.R. and Stefanski, L.A. (1994). Simulation-extrapolation estimation in
    parametric measurement error models. JASA 89(428), 1314-1328.
Carroll, R.J., Ruppert, D., Stefanski, L.A., and Crainiceanu, C.M. (2006).
    Measurement Error in Nonlinear Models: A Modern Perspective. CRC Press.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

__all__ = ["SIMEXResult", "simex_ols"]


@dataclass(frozen=True)
class SIMEXResult:
    """Result of :func:`simex_ols`.

    Attributes
    ----------
    beta_simex : float
        Extrapolated slope estimate at lambda = -1.
    beta_naive : float
        Uncorrected OLS slope estimate at lambda = 0.
    beta_mom : float
        Method-of-Moments corrected slope (beta_naive / reliability).
    reliability : float
        Estimated reliability ratio (Var(X) - sigma_u^2) / Var(X).
    sigma_u : float
        Standard deviation of classical measurement error in x.
    curve : np.ndarray
        Mean OLS slope estimate across simulations for each lambda.
    lambdas : np.ndarray
        Array of additional variance multipliers lambda.
    extrapolant_coefs : np.ndarray
        Polynomial coefficients of the extrapolation curve.
    deg : int
        Polynomial extrapolation degree.
    B : int
        Number of simulation replications per lambda.
    """

    beta_simex: float
    beta_naive: float
    beta_mom: float
    reliability: float
    sigma_u: float
    curve: np.ndarray
    lambdas: np.ndarray
    extrapolant_coefs: np.ndarray
    deg: int
    B: int

    def summary(self) -> str:
        return (
            f"SIMEX OLS Estimation (deg={self.deg}, B={self.B})\n"
            f"  sigma_u       : {self.sigma_u:.4f}\n"
            f"  reliability   : {self.reliability:.4f}\n"
            f"  beta (naive)  : {self.beta_naive:+.4f}\n"
            f"  beta (SIMEX)  : {self.beta_simex:+.4f}\n"
            f"  beta (MOM)    : {self.beta_mom:+.4f}\n"
        )


def _ols_slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Slope of single-regressor OLS, vectorised over leading axes."""
    xc = x - x.mean(-1, keepdims=True)
    yc = y - y.mean(-1, keepdims=True)
    denom = (xc * xc).sum(-1)
    zero_mask = denom == 0
    if np.any(zero_mask):
        denom = np.where(zero_mask, 1.0, denom)
        out = (xc * yc).sum(-1) / denom
        return np.where(zero_mask, np.nan, out)
    return (xc * yc).sum(-1) / denom


def simex_ols(
    y: np.ndarray,
    x: np.ndarray,
    sigma_u: float,
    B: int = 100,
    lambdas: np.ndarray | Sequence[float] | None = None,
    deg: int = 2,
    rng: np.random.Generator | int | None = None,
) -> SIMEXResult:
    """Quadratic SIMEX correction for errors-in-variables attenuation in OLS.

    Parameters
    ----------
    y : (N,) ndarray
        Dependent variable.
    x : (N,) ndarray
        Mismeasured independent variable.
    sigma_u : float
        Standard deviation of classical additive measurement error in x.
    B : int, default 100
        Number of simulation replications per lambda.
    lambdas : array-like, optional
        Grid of variance multipliers lambda >= 0. Defaults to
        np.array([0.0, 0.5, 1.0, 1.5, 2.0]).
    deg : int, default 2
        Degree of polynomial extrapolant (1 for linear, 2 for quadratic).
    rng : Generator or int, optional
        Random number generator or seed.

    Returns
    -------
    SIMEXResult
        Object containing beta_simex, beta_naive, beta_mom, reliability, curve, etc.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    x_arr = np.asarray(x, dtype=float).ravel()
    if len(y_arr) != len(x_arr):
        raise ValueError(f"y and x must have same length, got {len(y_arr)} and {len(x_arr)}")
    n_obs = len(y_arr)
    if n_obs < 3:
        raise ValueError("Need at least 3 observations for SIMEX estimation")

    if lambdas is None:
        lam = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    else:
        lam = np.asarray(lambdas, dtype=float).ravel()

    if rng is None:
        gen = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        gen = np.random.default_rng(rng)
    else:
        gen = rng

    # Compute naive estimate
    beta_naive = float(_ols_slope(x_arr[None, :], y_arr[None, :])[0])

    # Reliability calculation
    var_x = float(np.var(x_arr, ddof=1)) if n_obs > 1 else float(np.var(x_arr))
    rel = (var_x - float(sigma_u) ** 2) / var_x if var_x > 0 else 0.0
    beta_mom = beta_naive / rel if rel > 0 else float("nan")

    # Simulation step
    curve = np.empty(len(lam))
    for j, lm in enumerate(lam):
        if lm == 0.0:
            curve[j] = beta_naive
        else:
            noise = gen.normal(0.0, np.sqrt(lm) * sigma_u, (B, n_obs))
            x_sim = x_arr[None, :] + noise
            y_sim = np.broadcast_to(y_arr[None, :], (B, n_obs))
            slopes = _ols_slope(x_sim, y_sim)
            curve[j] = float(np.mean(slopes))

    # Extrapolation step
    fit_deg = min(deg, len(lam) - 1)
    V = np.vander(lam, fit_deg + 1)
    coefs, _, _, _ = np.linalg.lstsq(V, curve, rcond=None)
    v_minus1 = np.vander([-1.0], fit_deg + 1)
    beta_simex = float((v_minus1 @ coefs)[0])

    return SIMEXResult(
        beta_simex=beta_simex,
        beta_naive=beta_naive,
        beta_mom=beta_mom,
        reliability=rel,
        sigma_u=float(sigma_u),
        curve=curve,
        lambdas=lam,
        extrapolant_coefs=coefs,
        deg=fit_deg,
        B=B,
    )
