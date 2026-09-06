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
    n_jobs: int = 1,
) -> np.ndarray:
    """Run the overlapping block bootstrap on *residuals*.

    Parameters
    ----------
    residuals:
        1-D array of length T. Blocks are drawn from the array **as given**
        (no re-centering happens here): pass mean-zero residuals if the
        bootstrap DGP should have mean-zero innovations.
    refit_fn:
        Callable(e_boot: ndarray[T]) -> ndarray[n_stats]. Called B times.
    B:
        Number of bootstrap replications.
    block_length:
        Block length ℓ, ``1 <= ℓ <= T``. Defaults to round(T^{1/3}).
        A value outside that range raises ``ValueError``.
    rng:
        NumPy Generator for reproducibility.
    n_jobs:
        Number of parallel worker threads. Set to -1 to use all available CPU cores.

    Returns
    -------
    draws : ndarray of shape (B, n_stats)
        Each row is one bootstrap draw of ``refit_fn``.
    """
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 1:
        raise ValueError(
            f"block_bootstrap: residuals must be 1-D, got shape {residuals.shape}"
        )
    T = len(residuals)
    if T == 0:
        raise ValueError("block_bootstrap: residuals is empty")
    if block_length is None:
        block_length = default_block_length(T)
    block_length = int(block_length)
    if block_length < 1 or block_length > T:
        raise ValueError(
            f"block_bootstrap: block_length={block_length} must satisfy "
            f"1 <= block_length <= T={T}"
        )
    if rng is None:
        rng = np.random.default_rng()

    ell = block_length
    n_starts = T - ell + 1          # number of possible starting points
    k = math.ceil(T / ell)           # blocks needed to cover length T

    # Vectorized block indexing (identical random sequence)
    starts = rng.integers(0, n_starts, size=(B, k))
    offsets = np.arange(ell)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(B, k * ell)[:, :T]
    boot_matrix = residuals[idx]

    def _eval_draw(boot: np.ndarray) -> np.ndarray:
        return np.asarray(refit_fn(boot), dtype=float)

    if n_jobs == 1:
        draws_list = [_eval_draw(boot_matrix[b]) for b in range(B)]
    else:
        import concurrent.futures
        import os

        workers = os.cpu_count() or 1 if n_jobs < 0 else n_jobs
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            draws_list = list(ex.map(_eval_draw, boot_matrix))

    return np.stack(draws_list, axis=0)   # shape (B, n_stats)
