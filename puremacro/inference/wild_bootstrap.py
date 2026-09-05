"""Rademacher wild bootstrap for SVAR IRFs (Mertens-Ravn 2013).

Unlike residual bootstrap (which samples WITH replacement from residuals),
wild bootstrap multiplies each residual by ±1 preserving conditional heteroskedasticity.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np
from numpy.random import default_rng

from .bootstrap import _ols_var, _irf_from_var

#: Warn above this fraction of bootstrap draws failing identification. Mirrors
#: ``puremacro.var.identify.cholesky._BOOT_FAIL_WARN_THRESHOLD``, which is the
#: pattern CONTRIBUTING.md names as the one to follow.
_BOOT_FAIL_WARN_THRESHOLD = 0.05


def wild_bootstrap(
    residuals: np.ndarray,
    refit_fn: Callable[[np.ndarray], np.ndarray],
    n_boot: int = 999,
    rng: Optional[np.random.Generator] = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Rademacher wild bootstrap for scalar / LP regression inference.

    Multiplies each residual by an i.i.d. Rademacher weight (±1), then
    calls ``refit_fn`` to re-estimate the statistic of interest.

    Parameters
    ----------
    residuals : ndarray of shape (T,) or (T, n)
        Fitted residuals (zero-mean recommended).
    refit_fn : callable
        Signature: ``refit_fn(e_boot) -> ndarray``. Called ``n_boot`` times.
    n_boot : int
        Number of bootstrap replications.
    rng : numpy Generator or None
    n_jobs : int, default 1
        Number of parallel worker threads. Set to -1 to use all available CPU cores.

    Returns
    -------
    draws : ndarray of shape (n_boot, ...)
        Bootstrap draws of the statistic returned by ``refit_fn``.
    """
    if rng is None:
        rng = default_rng()
    residuals = np.asarray(residuals, dtype=float)
    T = residuals.shape[0]

    # Batch generate Rademacher weights (identical random sequence, vectorized)
    W = rng.choice(np.array([-1.0, 1.0]), size=(n_boot, T))

    def _eval_draw(w: np.ndarray) -> np.ndarray:
        if residuals.ndim == 1:
            e_boot = residuals * w
        else:
            e_boot = residuals * w[:, None]
        return np.asarray(refit_fn(e_boot), dtype=float)

    if n_jobs == 1:
        draws = [_eval_draw(W[b]) for b in range(n_boot)]
    else:
        import concurrent.futures
        import os

        workers = os.cpu_count() or 1 if n_jobs < 0 else n_jobs
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            draws = list(ex.map(_eval_draw, W))

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
    # Only identified draws contribute to the percentile bands. A draw whose
    # identification failed produced no statistic at all, so it carries no
    # information about the bootstrap distribution.
    accepted: list[np.ndarray] = []
    n_fail = 0
    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q

    W = rng.choice(np.array([-1.0, 1.0]), size=(n_boot, len(resid)))
    E_all = resid[None, :, :] * W[:, :, None]
    A_stack = np.hstack(A_list)
    Yb_all = np.empty((n_boot, T, n))
    Yb_all[:, :p, :] = Y[:p]
    for t in range(p, T):
        lags = np.concatenate([Yb_all[:, t - 1 - l, :] for l in range(p)], axis=1)
        Yb_all[:, t, :] = lags @ A_stack.T + c + E_all[:, t - p, :]

    for b in range(n_boot):
        A_b, _, Sigma_b, resid_b, _ = _ols_var(Yb_all[b], p)
        try:
            B_b = impact_fn(A_b, Sigma_b, resid_b)
        except np.linalg.LinAlgError:
            # Drop and warn -- the pattern CONTRIBUTING.md names, and which
            # this function was the counter-example to. Writing `point` here
            # instead placed a point mass exactly at the point estimate: with a
            # failure fraction f, the reported 100(1-2a)% band was really the
            # 100(1 - 2a/(1-f))% band, contracting monotonically in f, and at
            # f >= 1-2a collapsing to zero width. A 100%-failed bootstrap
            # returned lo == hi == point and said nothing about it.
            n_fail += 1
            continue
        accepted.append(_irf_from_var(A_b, horizon, B_b))

    if not accepted:
        raise np.linalg.LinAlgError(
            f"wild_bootstrap_var: all {n_boot} bootstrap draws failed "
            "identification. Likely causes: a weak proxy/instrument, too few "
            "observations for n*p coefficients, or a singular reduced-form Σ."
        )
    fail_rate = n_fail / n_boot
    if fail_rate > _BOOT_FAIL_WARN_THRESHOLD:
        warnings.warn(
            f"wild_bootstrap_var: {n_fail}/{n_boot} bootstrap draws "
            f"({fail_rate:.1%}) failed identification and were dropped. "
            "Bands are computed from the surviving draws and may be "
            "unreliable; consider more data, fewer lags, or a stronger "
            "instrument.",
            stacklevel=2,
        )
    draws = np.stack(accepted, axis=0)
    return point, np.quantile(draws, lo_q, axis=0), np.quantile(draws, hi_q, axis=0)
