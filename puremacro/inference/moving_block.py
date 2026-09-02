"""Paparoditis-Politis (1994) moving-block bootstrap for VAR residuals.

This module provides the moving-block bootstrap used for inference in the
Rigobon heteroskedasticity identification (Notebook 05).  It resamples
overlapping blocks of residuals to preserve short-run autocorrelation
structure while breaking long-run dependence.

References
----------
Paparoditis, E. and Politis, D.N. (1994). The local bootstrap for kernel
    estimators under general dependence conditions. Ann. Stat.
Kilian, L. (1998). Small-sample confidence intervals for impulse response
    functions. Rev. Econ. Stat. 80(2), 218-230.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Low-level block sampler
# --------------------------------------------------------------------------- #

def _sample_blocks(
    residuals: np.ndarray,
    block_len: int,
    target_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw bootstrap residuals via overlapping moving blocks.

    Parameters
    ----------
    residuals : ndarray of shape (T, n)
        Original VAR residuals.
    block_len : int
        Length ℓ of each block.
    target_len : int
        Desired output length (T - p for a VAR(p)).
    rng : numpy Generator
        Random number generator.

    Returns
    -------
    ndarray of shape (target_len, n)
        Concatenated and truncated bootstrap residuals.
    """
    return residuals[_sample_block_indices(
        residuals.shape[0], block_len, target_len, rng), :]


def _sample_block_indices(
    T: int,
    block_len: int,
    target_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Row indices of a moving-block resample, shape ``(target_len,)``.

    Split out from :func:`_sample_blocks` so that a caller can carry a
    *per-observation covariate* along with the residual it belongs to. Pairing
    a reshuffled residual with the covariate of the calendar date it happened
    to land on destroys whatever cross-observation structure the estimator
    identifies from — which is exactly what happened to the regime labels in
    :func:`puremacro.var.identify.hetero.rigobon_svar`, whose whole identifying
    content is the contrast between two regimes' covariances.
    """
    # Maximum starting index for a complete block of length ℓ
    max_start = T - block_len  # starting indices 0 .. T-ℓ  (T-ℓ+1 choices)
    n_blocks = math.ceil(target_len / block_len)
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])
    return idx[:target_len]


# --------------------------------------------------------------------------- #
# VAR simulation helper
# --------------------------------------------------------------------------- #

def _simulate_var(
    Y_init: np.ndarray,
    A_list: list[np.ndarray],
    intercept: np.ndarray,
    bootstrap_residuals: np.ndarray,
) -> np.ndarray:
    """Simulate a VAR(p) forward from initial conditions using bootstrap residuals.

    Parameters
    ----------
    Y_init : ndarray of shape (p, n)
        The last p observations used as starting values (most-recent last).
    A_list : list of ndarray, each (n, n)
        VAR coefficient matrices A_1, ..., A_p (lag-1 first).
    intercept : ndarray of shape (n,)
        VAR intercept / constant.
    bootstrap_residuals : ndarray of shape (T_star, n)
        Bootstrap residuals e*_1, ..., e*_{T_star}.

    Returns
    -------
    ndarray of shape (p + T_star, n)
        Simulated Y* (includes initial p rows).
    """
    p = len(A_list)
    T_star, n = bootstrap_residuals.shape
    Y = np.empty((p + T_star, n))
    Y[:p] = Y_init
    for t in range(T_star):
        y_new = intercept.copy()
        for lag in range(p):
            y_new = y_new + A_list[lag] @ Y[p + t - lag - 1]
        y_new = y_new + bootstrap_residuals[t]
        Y[p + t] = y_new
    return Y


# --------------------------------------------------------------------------- #
# Main bootstrap function
# --------------------------------------------------------------------------- #

def moving_block_bootstrap(
    residuals: np.ndarray,
    Y: np.ndarray,
    A_list: list[np.ndarray],
    intercept: np.ndarray,
    n_draws: int = 500,
    pass_index: bool = False,
    block_len: Optional[int] = None,
    horizon: int = 20,
    irf_fn: Optional[Callable] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Paparoditis-Politis moving-block bootstrap for VAR IRF inference.

    Algorithm
    ---------
    For each bootstrap draw b = 1 ... B:
      1. Sample k = ceil((T-p)/ℓ) blocks uniformly from the T-ℓ+1 possible
         starting positions (with replacement).
      2. Concatenate to get e*_{1..T-p} of length T-p.
      3. Simulate Y* using the fitted VAR dynamics and e*.
      4. Re-estimate VAR(p) on Y*; compute IRFs via `irf_fn`.

    Parameters
    ----------
    residuals : ndarray of shape (T_eff, n)
        Fitted residuals from the VAR (length T - p, i.e. after burning lags).
    Y : ndarray of shape (T, n)
        Full original data matrix (length T).
    A_list : list of ndarray, each (n, n)
        Estimated VAR coefficients A_1, ..., A_p.
    intercept : ndarray of shape (n,)
        Estimated VAR intercept.
    n_draws : int
        Number of bootstrap repetitions B.
    block_len : int or None
        Block length ℓ.  If None, uses round(T_eff^(1/3)).
    horizon : int
        IRF horizon H passed through to `irf_fn`.
    irf_fn : callable or None
        Signature: ``irf_fn(Y_star, p, horizon) -> ndarray of shape (H+1, n, n)``.
        If None, a default OLS-then-Cholesky IRF is computed internally.
    rng : numpy Generator or None
        Random state.  If None, a fresh default_rng() is used.

    Returns
    -------
    dict with keys:
        "draws"     : list of arrays, each shape (horizon+1, n, n). Normally
                      ``n_draws`` long; an ``irf_fn`` may return ``None`` to
                      drop a draw it could not identify, in which case the list
                      is shorter and the caller owns the bookkeeping.
        "block_len" : int, the block length actually used
    """
    if rng is None:
        rng = np.random.default_rng()

    T_eff, n = residuals.shape
    p = len(A_list)
    T = Y.shape[0]
    ell = block_len if block_len is not None else round(T_eff ** (1 / 3))
    ell = max(ell, 1)

    # Initial conditions: last p rows of Y (burn-in for simulation)
    Y_init = Y[:p]   # shape (p, n)

    if irf_fn is None:
        irf_fn = _default_irf_fn

    draws = []
    for _ in range(n_draws):
        idx_star = _sample_block_indices(T_eff, ell, T_eff, rng)
        e_star = residuals[idx_star, :]
        Y_star = _simulate_var(Y_init, A_list, intercept, e_star)
        # `pass_index` hands the caller the rows this draw actually used, so a
        # per-observation covariate can travel with its residual. Off by
        # default, so every existing 3-argument `irf_fn` is untouched.
        draw = (irf_fn(Y_star, p, horizon, idx_star) if pass_index
                else irf_fn(Y_star, p, horizon))
        if draw is not None:
            draws.append(draw)

    return {"draws": draws, "block_len": ell}


# --------------------------------------------------------------------------- #
# Default IRF function (Cholesky, for general use)
# --------------------------------------------------------------------------- #

def _default_irf_fn(
    Y_star: np.ndarray,
    p: int,
    horizon: int,
) -> np.ndarray:
    """Re-estimate VAR(p) on Y_star and return Cholesky IRFs.

    Returns ndarray of shape (horizon+1, n, n). Pure puremacro path —
    no statsmodels dependency, so this works under Pyodide.
    """
    from .._linalg import safe_cholesky
    from ..var.estimate import estimate_var
    from ..var.irf import irf as compute_irf

    A_list, _, Sigma, _, _ = estimate_var(Y_star, p)
    P = safe_cholesky(Sigma, name="moving_block default IRF")
    return compute_irf(A_list, P, horizon)


# --------------------------------------------------------------------------- #
# Percentile bands helper
# --------------------------------------------------------------------------- #

def bootstrap_percentiles(
    draws: list[np.ndarray],
    q_lo: float = 16,
    q_hi: float = 84,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lo, median, hi) percentile bands from bootstrap draws.

    Parameters
    ----------
    draws : list of ndarray, each shape (H+1, n, n)
    q_lo, q_hi : float
        Lower and upper percentile levels (e.g. 16 and 84 for 68% bands).

    Returns
    -------
    lo, med, hi : each ndarray of shape (H+1, n, n)
    """
    stack = np.stack(draws, axis=0)   # (B, H+1, n, n)
    lo  = np.percentile(stack, q_lo,  axis=0)
    med = np.percentile(stack, 50,    axis=0)
    hi  = np.percentile(stack, q_hi,  axis=0)
    return lo, med, hi
