"""Rademacher wild bootstrap for SVAR IRFs (Mertens-Ravn 2013).

Unlike residual bootstrap (which samples WITH replacement from residuals),
wild bootstrap multiplies each residual by ±1 preserving conditional heteroskedasticity.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from numpy.random import default_rng

from .bootstrap import _ols_var, _irf_from_var


def wild_bootstrap(
    residuals: np.ndarray,
    refit_fn: Callable[[np.ndarray], np.ndarray],
    n_boot: int = 999,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Rademacher wild bootstrap for scalar / LP regression inference.

    Multiplies each residual by an i.i.d. Rademacher weight (±1), then
    calls ``refit_fn`` to re-estimate the statistic of interest.

    Parameters
    ----------
    residuals : ndarray of shape (T,) or (T, n)
        Fitted residuals (zero-mean recommended).
    refit_fn : callable
        Signature: ``refit_fn(e_boot) -> ndarray``.  Called ``n_boot`` times.
    n_boot : int
        Number of bootstrap replications.
    rng : numpy Generator or None

    Returns
    -------
    draws : ndarray of shape (n_boot, ...)
        Bootstrap draws of the statistic returned by ``refit_fn``.
    """
    if rng is None:
        rng = default_rng()
    residuals = np.asarray(residuals, dtype=float)
    T = residuals.shape[0]
    draws = []
    for _ in range(n_boot):
        w = rng.choice(np.array([-1.0, 1.0]), size=T)
        if residuals.ndim == 1:
            e_boot = residuals * w
        else:
            e_boot = residuals * w[:, None]
        stat = refit_fn(e_boot)
        draws.append(np.asarray(stat, dtype=float))
    return np.stack(draws, axis=0)


def wild_bootstrap_var(
    Y: np.ndarray,
    *,
    p: int,
    horizon: int,
    impact_fn: Callable[[list, np.ndarray, np.ndarray], np.ndarray],
    n_boot: int = 500,
    ci: float = 0.9,
    seed: int = 0,
):
    rng = default_rng(seed)
    A_list, c, Sigma, resid, X = _ols_var(Y, p)
    B = impact_fn(A_list, Sigma, resid)
    point = _irf_from_var(A_list, horizon, B)

    T, n = Y.shape
    draws = np.empty((n_boot, n, n, horizon + 1))
    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q

    for b in range(n_boot):
        w = rng.choice([-1.0, 1.0], size=len(resid))
        eps_b = resid * w[:, None]
        Yb = np.zeros_like(Y)
        Yb[:p] = Y[:p]
        for t in range(p, T):
            yt = c.copy()
            for l in range(p):
                yt += A_list[l] @ Yb[t - l - 1]
            yt += eps_b[t - p]
            Yb[t] = yt
        A_b, _, Sigma_b, resid_b, _ = _ols_var(Yb, p)
        try:
            B_b = impact_fn(A_b, Sigma_b, resid_b)
            draws[b] = _irf_from_var(A_b, horizon, B_b)
        except np.linalg.LinAlgError:
            draws[b] = point

    return point, np.quantile(draws, lo_q, axis=0), np.quantile(draws, hi_q, axis=0)
