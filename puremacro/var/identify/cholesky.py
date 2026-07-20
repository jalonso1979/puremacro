"""Cholesky-identified SVAR with residual bootstrap bands."""

from __future__ import annotations

import warnings

import numpy as np

from ..._linalg import safe_cholesky
from ..estimate import estimate_var
from ..irf import irf as compute_irf
from ._results import CholeskySVARResult

# Warn the caller when more than this fraction of bootstrap draws produce
# a non-PD reduced-form Σ. Above this rate, the percentile bands are no
# longer trustworthy as a description of sampling uncertainty.
_BOOT_FAIL_WARN_THRESHOLD = 0.05

# Percentile bands need cleaner draws than a single point estimate — a
# bootstrap Σ_b with cond > 1e8 (Cholesky pivot ratio < 1e-4) produces a
# numerically-defined but statistically meaningless impact matrix, so we
# treat such draws as failures. This is stricter than safe_cholesky's
# general-use threshold (cond > 1e14) on purpose: an estimator that
# returns wide bands is honest; one that returns narrow bands built on
# ill-conditioned draws is silent garbage.
_BOOT_COND_RATIO_FLOOR = 1e-4


def cholesky_factor(Sigma: np.ndarray) -> np.ndarray:
    """Lower-triangular Cholesky factor of Sigma (impact matrix B0)."""
    return safe_cholesky(Sigma, name="cholesky_factor")


def compute_chol_shocks(
    Y: np.ndarray,
    *,
    p: int,
    ordering: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify Cholesky structural shocks for a fitted reduced-form VAR.

    Mirrors the OLS step inside :func:`_residual_bootstrap_var` but
    returns the shock matrix instead of an IRF. Useful when downstream
    analysis (LP-IV, FEVD on extra variables, narrative inspection)
    needs the shocks themselves.

    Parameters
    ----------
    Y : ndarray, shape (T, n)
        Observations, rows time-ordered, columns variables.
    p : int
        Number of lags.
    ordering : list[int] | None
        Optional permutation of variable indices used for the
        Cholesky factorization (matches :func:`cholesky_svar`'s
        ``ordering`` semantics). Output columns are returned in the
        *original* variable order.

    Returns
    -------
    t_index : ndarray, shape (T - p,)
        Integer offsets of the rows in ``Y`` each shock corresponds to.
        Callers map these back to dates via the original DatetimeIndex.
    shocks : ndarray, shape (T - p, n)
        Structural shocks ``ε_t = L^{-1} u_t`` where ``u`` is the OLS
        residual and ``L`` is the lower-Cholesky factor of ``Σ_u``.
        Columns follow the ORIGINAL variable order, regardless of
        ``ordering``.
    """
    if ordering is not None:
        perm = np.array(ordering)
        Y_p = Y[:, perm]
    else:
        perm = None
        Y_p = Y

    _, _, Sigma, resid, _ = estimate_var(Y_p, p)
    L = safe_cholesky(Sigma, name="compute_chol_shocks")
    eps = np.linalg.solve(L, resid.T).T  # (T - p, n) in permuted order

    if perm is not None:
        inv = np.argsort(perm)
        eps = eps[:, inv]

    t_index = np.arange(p, Y.shape[0])
    return t_index, eps


def _residual_bootstrap_var(Y, p, horizon, n_boot=500, ci=0.9, seed=0, ordering=None):
    """Residual bootstrap for Cholesky-identified VAR IRFs.
    Returns (point, lower, upper, n_fail), arrays each (H+1, n, n).
    """
    rng = np.random.default_rng(seed)
    if ordering is not None:
        perm = np.array(ordering)
        Y = Y[:, perm]

    A_list, c, Sigma, resid, _ = estimate_var(Y, p)
    P = safe_cholesky(Sigma, name="cholesky_svar (point estimate)")
    point = compute_irf(A_list, P, horizon)  # (H+1, n, n)

    T, n = Y.shape
    accepted = []  # only PD-Σ draws contribute to the percentile bands
    n_fail = 0
    lo_q = (1 - ci) / 2 * 100
    hi_q = (1 - (1 - ci) / 2) * 100

    for b in range(n_boot):
        idx = rng.integers(0, len(resid), size=len(resid))
        eps_b = resid[idx]
        Yb = np.zeros_like(Y)
        Yb[:p] = Y[:p]
        for t in range(p, T):
            yt = c.copy()
            for l in range(p):
                yt += A_list[l] @ Yb[t - l - 1]
            yt += eps_b[t - p]
            Yb[t] = yt
        A_b, _, Sigma_b, _, _ = estimate_var(Yb, p)
        try:
            P_b = safe_cholesky(Sigma_b, name="cholesky_svar bootstrap Σ_b")
        except np.linalg.LinAlgError:
            n_fail += 1
            continue
        # LAPACK's potrf does not raise on ultra-degenerate Σ_b — it
        # silently produces NaN/Inf diagonals. Reject those, then reject
        # high-cond factors (cond > 1e8) for the reasons in the module
        # header. Both checks treat the draw as a bootstrap failure.
        diagP = np.abs(np.diag(P_b))
        if (not np.all(np.isfinite(diagP))
                or (diagP.size
                    and diagP.min() < _BOOT_COND_RATIO_FLOOR * diagP.max())):
            n_fail += 1
            continue
        accepted.append(compute_irf(A_b, P_b, horizon))

    if not accepted:
        raise np.linalg.LinAlgError(
            f"cholesky_svar: all {n_boot} bootstrap draws produced a "
            "non-PD reduced-form Σ. The data may be too short for p lags."
        )
    fail_rate = n_fail / n_boot
    if fail_rate > _BOOT_FAIL_WARN_THRESHOLD:
        warnings.warn(
            f"cholesky_svar: {n_fail}/{n_boot} bootstrap draws "
            f"({fail_rate:.1%}) produced a non-PD Σ and were dropped. "
            "Bands are computed from the surviving draws and may be "
            "unreliable; consider more data, fewer lags, or a different "
            "identification scheme.",
            stacklevel=2,
        )
    draws = np.stack(accepted, axis=0)
    lo = np.percentile(draws, lo_q, axis=0)
    hi = np.percentile(draws, hi_q, axis=0)

    if ordering is not None:
        perm = np.array(ordering)
        inv = np.argsort(perm)
        point = point[np.ix_(np.arange(horizon + 1), inv, inv)]
        lo = lo[np.ix_(np.arange(horizon + 1), inv, inv)]
        hi = hi[np.ix_(np.arange(horizon + 1), inv, inv)]

    # Return (n, n, H+1) for catalog consistency — transpose last axis to front
    # Actually keep as (H+1, n, n) which is what the new irf.py produces.
    return point, lo, hi, n_fail


def cholesky_svar(
    Y: np.ndarray,
    *,
    p: int,
    horizon: int,
    ordering: list[int] | None = None,
    n_boot: int = 500,
    ci: float = 0.9,
    seed: int = 0,
) -> CholeskySVARResult:
    """Cholesky SVAR. ``ordering`` re-orders variables before decomposition.

    Returns
    -------
    CholeskySVARResult
        Frozen dataclass with fields ``irf_point``, ``irf_lower``,
        ``irf_upper`` (each shaped ``(H+1, n, n)``), ``n_boot``,
        ``n_fail``, ``ci``.
    """
    point, lo, hi, n_fail = _residual_bootstrap_var(
        Y, p=p, horizon=horizon, n_boot=n_boot, ci=ci, seed=seed, ordering=ordering
    )
    return CholeskySVARResult(
        irf_point=point,
        irf_lower=lo,
        irf_upper=hi,
        n_boot=n_boot,
        n_fail=n_fail,
        ci=ci,
    )
