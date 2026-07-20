"""(Simulated) method-of-moments estimation for puremacro.vfi models.

Estimate parameters by minimizing the weighted distance between model-implied
moments and data moments:

    min_theta  (m(theta) - data)' W (m(theta) - data),

where the user supplies ``moments_at(theta) -> ndarray`` (solve the model and
compute the moments at ``theta``) and ``data_moments``. ``W`` defaults to the
identity. Solved by scipy.optimize.minimize with a gradient-free **Nelder-Mead**
default: simulated / discrete-VFI moments are non-smooth (piecewise-constant in
the parameters), so a gradient optimizer sees a zero finite-difference gradient
below the grid scale and stalls; Nelder-Mead is robust to that and still honors
``bounds``. Pass ``method=`` to override (e.g. L-BFGS-B for smooth moments). This
mirrors the engine's GE/transition seams: the user provides the model-specific
moments, the estimator drives the optimization.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class EstimationResult:
    """A method-of-moments estimate."""
    theta: np.ndarray          # estimated parameter vector
    objective: float           # minimized weighted moment distance
    moments: np.ndarray        # model moments at the estimate, m(theta*)
    success: bool
    n_evals: int


def estimate_method_of_moments(moments_at, data_moments, theta0, *, weight=None,
                               bounds=None, method=None, tol=1e-10,
                               max_iter=10_000):
    """Estimate ``theta`` by minimizing the weighted moment distance.

    ``moments_at(theta) -> ndarray`` returns the model moments at ``theta``;
    ``data_moments`` the targets; ``weight`` the (n_moments, n_moments) matrix
    (default identity). ``bounds`` (list of (lo, hi)) are honored. The default
    optimizer is gradient-free **Nelder-Mead**: simulated / discrete-VFI moments
    are non-smooth step functions, so a gradient method sees a zero
    finite-difference gradient below the grid scale and stalls at the start.
    Pass ``method=`` (e.g. "L-BFGS-B") to override for smooth moments. Returns an
    EstimationResult.
    """
    data = np.asarray(data_moments, dtype=float).reshape(-1)
    theta0 = np.asarray(theta0, dtype=float).reshape(-1)
    n_m = data.shape[0]
    if weight is None:
        W = np.eye(n_m)
    else:
        W = np.asarray(weight, dtype=float)
        if W.shape != (n_m, n_m):
            raise ValueError(
                f"weight must be ({n_m},{n_m}) to match {n_m} moments; got {W.shape}"
            )
    counter = {"n": 0}

    def objective(theta):
        counter["n"] += 1
        m = np.asarray(moments_at(theta), dtype=float).reshape(-1)
        r = m - data
        return float(r @ W @ r)

    if method is None:
        method = "Nelder-Mead"   # gradient-free default; robust to non-smooth moments
    opts = {"maxiter": max_iter}
    if method == "Nelder-Mead":
        opts["xatol"] = tol
        opts["fatol"] = tol
    else:
        opts["gtol"] = tol
    res = minimize(objective, theta0, method=method, bounds=bounds, options=opts)

    theta = np.asarray(res.x, dtype=float)
    moments = np.asarray(moments_at(theta), dtype=float).reshape(-1)
    return EstimationResult(theta=theta, objective=float(res.fun),
                            moments=moments, success=bool(res.success),
                            n_evals=counter["n"])


__all__ = ["estimate_method_of_moments", "EstimationResult"]
