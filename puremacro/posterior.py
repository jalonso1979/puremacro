"""Generic posterior-predictive simulation + fan-chart quantiles.

Pairs with ``puremacro.var.bvar.minnesota_gibbs`` (or any other source
of VAR coefficient + covariance posterior draws) to deliver fan-chart
density forecasts.
"""
from __future__ import annotations

import numpy as np

from ._linalg import safe_cholesky


def simulate_var_paths(
    A_draws: np.ndarray,
    Sigma_draws: np.ndarray,
    intercept_draws: np.ndarray,
    y_init: np.ndarray,
    horizon: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate one path per posterior draw of (A, Sigma, intercept).

    Parameters
    ----------
    A_draws : (D, p, n, n) ndarray — D posterior draws, lag order p.
    Sigma_draws : (D, n, n).
    intercept_draws : (D, n).
    y_init : (p, n) initial conditions (most-recent last).
    horizon : forecast horizon H.

    Returns
    -------
    paths : (D, H+1, n) ndarray. ``paths[d, 0]`` = ``y_init[-1]`` for all d.
    """
    if rng is None:
        rng = np.random.default_rng()
    D, p, n, _ = A_draws.shape
    y_init = np.asarray(y_init, dtype=float)
    if y_init.ndim == 1:
        y_init = y_init[None, :]
    paths = np.empty((D, horizon + 1, n))
    for d in range(D):
        L = safe_cholesky(Sigma_draws[d] + 1e-12 * np.eye(n),
                            name="posterior fan-chart draw")
        history = list(y_init[-p:])
        paths[d, 0] = history[-1]
        for h in range(1, horizon + 1):
            y_t = intercept_draws[d].copy()
            for k in range(p):
                y_t += A_draws[d, k] @ history[-1 - k]
            y_t += L @ rng.standard_normal(n)
            history.append(y_t)
            if len(history) > p:
                history = history[-p:]
            paths[d, h] = y_t
    return paths


def fan_chart_quantiles(
    paths: np.ndarray,
    levels: tuple = (0.10, 0.30, 0.50, 0.70, 0.90),
) -> np.ndarray:
    """Per-horizon, per-variable quantiles of a path bundle.

    Parameters
    ----------
    paths : (D, H+1, n) — output of ``simulate_var_paths``.
    levels : sequence of probabilities in [0, 1].

    Returns
    -------
    quantiles : (len(levels), H+1, n) ndarray.
    """
    pcts = [100.0 * float(p) for p in levels]
    return np.percentile(paths, pcts, axis=0)


__all__ = ["simulate_var_paths", "fan_chart_quantiles"]
