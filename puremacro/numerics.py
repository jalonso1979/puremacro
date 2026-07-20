"""Numerical-differentiation primitives + sandwich-vcov helper +
maximum-likelihood wrapper.

Lets you write your own log-likelihood in numpy and immediately get
gradient, Hessian, and heteroskedasticity-robust ("sandwich") standard
errors. Pairs naturally with ``puremacro.garch.fit`` (Gaussian GARCH)
or any user-coded MLE.

References
----------
Huber (1967) / White (1980) / Newey-McFadden (1994) — the
sandwich-vcov for misspecified MLEs is

    V = H^{-1} J H^{-1} / T,

where H is (the negative of) the average Hessian of the per-obs
log-likelihood and J is the average outer product of per-obs scores.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize


def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    """Central-difference gradient of scalar f at x."""
    x = np.asarray(x, dtype=float).ravel()
    g = np.zeros_like(x)
    for i in range(len(x)):
        ei = np.zeros_like(x); ei[i] = h
        g[i] = (float(f(x + ei)) - float(f(x - ei))) / (2.0 * h)
    return g


def numerical_hessian(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float = 1e-4,
) -> np.ndarray:
    """Central-difference Hessian via four-point stencil.

    Cost: O(n^2) function evaluations.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros_like(x); ei[i] = h
        for j in range(i, n):
            ej = np.zeros_like(x); ej[j] = h
            f_pp = float(f(x + ei + ej))
            f_pm = float(f(x + ei - ej))
            f_mp = float(f(x - ei + ej))
            f_mm = float(f(x - ei - ej))
            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h)
            H[j, i] = H[i, j]
    return H


def sandwich_vcov(
    neg_loglik_per_obs: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    h_grad: float = 1e-5,
    h_hess: float = 1e-4,
) -> np.ndarray:
    """Quasi-MLE / mis-specification-robust sandwich covariance.

    Parameters
    ----------
    neg_loglik_per_obs : callable theta -> (T,) array of per-obs neg-log-liks.
        Required form so the score outer-product can be computed.
    theta : MLE point estimate.

    Returns
    -------
    V : (k, k) sandwich vcov of theta.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    T = len(neg_loglik_per_obs(theta))

    # Per-obs scores via central differences (numpy-vectorised over t).
    k = len(theta)
    scores = np.zeros((T, k))
    for j in range(k):
        ej = np.zeros_like(theta); ej[j] = h_grad
        # Score on per-obs negative log-lik; flip sign to get score of log-lik
        scores[:, j] = -(
            neg_loglik_per_obs(theta + ej) - neg_loglik_per_obs(theta - ej)
        ) / (2.0 * h_grad)

    # Hessian of *summed* neg-log-lik
    f_total = lambda t: float(np.sum(neg_loglik_per_obs(t)))
    H_total = numerical_hessian(f_total, theta, h=h_hess)

    # H = -d^2/dtheta^2 log-lik (positive-definite at the MLE);
    # H_total is the Hessian of neg-log-lik, so already positive.
    H_per_obs = H_total / T
    J_per_obs = (scores.T @ scores) / T
    try:
        H_inv = np.linalg.inv(H_per_obs)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(H_per_obs)
    return H_inv @ J_per_obs @ H_inv / T


def mle_fit(
    neg_loglik: Callable[[np.ndarray], float],
    theta_init: np.ndarray,
    *,
    bounds=None,
    method: str = "L-BFGS-B",
    sandwich: bool = False,
    neg_loglik_per_obs: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict:
    """Fit by minimising ``neg_loglik`` and (optionally) compute SE.

    Parameters
    ----------
    neg_loglik : scalar function of theta. ``mle_fit`` minimises this.
    theta_init : initial guess.
    bounds : passed to scipy.optimize.minimize.
    method : optimiser.
    sandwich : if True, also compute sandwich SE (requires
        ``neg_loglik_per_obs``).
    neg_loglik_per_obs : per-observation neg-log-lik for sandwich.

    Returns
    -------
    dict with theta, loglik, hessian, vcov_classical, vcov_sandwich (if
    sandwich=True), se_classical, se_sandwich, converged.
    """
    res = minimize(neg_loglik, np.asarray(theta_init, dtype=float).ravel(),
                   method=method, bounds=bounds)
    theta = res.x
    H = numerical_hessian(neg_loglik, theta)
    try:
        vcov_cls = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        vcov_cls = np.linalg.pinv(H)
    out = {
        "theta": theta,
        "loglik": float(-res.fun),
        "hessian": H,
        "vcov_classical": vcov_cls,
        "se_classical": np.sqrt(np.maximum(np.diag(vcov_cls), 0.0)),
        "converged": bool(res.success),
        "n_iter": int(res.nit) if hasattr(res, "nit") else None,
    }
    if sandwich:
        if neg_loglik_per_obs is None:
            raise ValueError("sandwich=True requires neg_loglik_per_obs.")
        V = sandwich_vcov(neg_loglik_per_obs, theta)
        out["vcov_sandwich"] = V
        out["se_sandwich"] = np.sqrt(np.maximum(np.diag(V), 0.0))
    return out


__all__ = [
    "numerical_gradient", "numerical_hessian",
    "sandwich_vcov", "mle_fit",
]
