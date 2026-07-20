"""Bootstrap helpers for SVAR / LP inference."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from numpy.random import default_rng


def _ols_var(Y: np.ndarray, p: int):
    """OLS VAR(p) with constant. Returns (A_list, c, Sigma, resid, X)."""
    T, n = Y.shape
    X = np.column_stack([np.ones(T - p)] + [Y[p - l - 1 : T - l - 1] for l in range(p)])
    Yd = Y[p:]
    B = np.linalg.lstsq(X, Yd, rcond=None)[0]
    c = B[0]
    A_list = [B[1 + l * n : 1 + (l + 1) * n].T for l in range(p)]
    resid = Yd - X @ B
    Sigma = resid.T @ resid / (T - p - 1 - n * p)
    return A_list, c, Sigma, resid, X


def _irf_from_var(A_list: list[np.ndarray], horizon: int, impact: np.ndarray) -> np.ndarray:
    n = impact.shape[0]
    p = len(A_list)
    Phi = [np.zeros((n, n)) for _ in range(horizon + 1)]
    Phi[0] = np.eye(n)
    for h in range(1, horizon + 1):
        s = np.zeros((n, n))
        for l in range(min(p, h)):
            s += A_list[l] @ Phi[h - l - 1]
        Phi[h] = s
    ir = np.stack([Phi[h] @ impact for h in range(horizon + 1)], axis=-1)
    return ir


def residual_bootstrap(
    residuals: np.ndarray,
    Y: np.ndarray,
    A_list: list[np.ndarray],
    intercept: np.ndarray,
    n_draws: int = 500,
    horizon: int = 20,
    irf_fn: Optional[Callable] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Generic residual bootstrap for VAR-based identification.

    Resamples residuals i.i.d. (with replacement), simulates Y*, and applies
    a caller-supplied ``irf_fn`` to each bootstrap replication.

    Parameters
    ----------
    residuals : ndarray (T_eff, n)
        Fitted VAR residuals (length T - p).
    Y : ndarray (T, n)
        Original data.
    A_list : list of (n, n) arrays
        Estimated VAR coefficients A_1, ..., A_p.
    intercept : ndarray (n,)
        Estimated VAR intercept.
    n_draws : int
        Number of bootstrap repetitions.
    horizon : int
        IRF horizon passed through to ``irf_fn``.
    irf_fn : callable or None
        Signature: ``irf_fn(Y_star, p, horizon) -> ndarray of shape (H+1, n, n)``.
        If None, returns an identity placeholder.
    rng : numpy Generator or None

    Returns
    -------
    dict with keys:
        "draws" : list of n_draws arrays, each shape (horizon+1, n, n)
    """
    if rng is None:
        rng = np.random.default_rng()

    T_eff, n = residuals.shape
    p = len(A_list)
    T = Y.shape[0]
    Y_init = Y[:p]

    draws = []
    for _ in range(n_draws):
        # i.i.d. resample residuals
        idx = rng.integers(0, T_eff, size=T_eff)
        e_star = residuals[idx]

        # Simulate Y*
        Y_star = np.empty((p + T_eff, n))
        Y_star[:p] = Y_init
        for t in range(T_eff):
            y_new = intercept.copy()
            for lag in range(p):
                y_new = y_new + A_list[lag] @ Y_star[p + t - lag - 1]
            y_new = y_new + e_star[t]
            Y_star[p + t] = y_new

        if irf_fn is not None:
            draw = irf_fn(Y_star, p, horizon)
        else:
            draw = np.zeros((horizon + 1, n, n))

        draws.append(draw)

    return {"draws": draws}
