"""Overlapping block bootstrap (Paparoditis-Politis 2001).

Used by smooth LP and state-dependent LP to construct bootstrap confidence
bands.  The caller supplies a ``refit_fn`` that accepts a length-T bootstrap
residual array and returns a 1-D numpy array of statistics; this module
handles the block-sampling logic only.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def default_block_length(T: int) -> int:
    """Default block length: round(T^{1/3}), as in Paparoditis-Politis (2001)."""
    return round(T ** (1 / 3))


def block_bootstrap(
    residuals: np.ndarray,
    *,
    refit_fn: Callable[[np.ndarray], np.ndarray],
    B: int = 999,
    block_length: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Run the overlapping block bootstrap on *residuals*.

    Parameters
    ----------
    residuals:
        1-D array of length T (zero-mean recommended; re-centered inside).
    refit_fn:
        Callable(e_boot: ndarray[T]) -> ndarray[n_stats].  Called B times.
    B:
        Number of bootstrap replications.
    block_length:
        Block length ℓ. Defaults to round(T^{1/3}).
    rng:
        NumPy Generator for reproducibility.

    Returns
    -------
    draws : ndarray of shape (B, n_stats)
        Each row is one bootstrap draw of ``refit_fn``.
    """
    residuals = np.asarray(residuals, dtype=float)
    T = len(residuals)
    if block_length is None:
        block_length = default_block_length(T)
    if rng is None:
        rng = np.random.default_rng()

    ell = block_length
    n_starts = T - ell + 1          # number of possible starting points
    k = math.ceil(T / ell)           # blocks needed to cover length T

    draws_list: list[np.ndarray] = []
    for _ in range(B):
        starts = rng.integers(0, n_starts, size=k)
        # Build the bootstrap residual series by concatenating blocks
        boot = np.concatenate([residuals[s : s + ell] for s in starts])[:T]
        stat = refit_fn(boot)
        draws_list.append(np.asarray(stat, dtype=float))

    return np.stack(draws_list, axis=0)   # shape (B, n_stats)
