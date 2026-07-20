"""Giacomini-Kitagawa (2021) identified-set bands for sign-restricted SVAR.

Standard sign-restriction inference (Uhlig / RWZ) draws Q rotations and
reports posterior quantiles of the IRF *across draws*. That implicitly
puts a uniform prior on Q and so reports a posterior, not the
identified set. GK 2021 separate VAR uncertainty from set uncertainty:

    For each VAR estimate (or posterior draw) (A, Sigma):
        find max_Q  IRF[h, i, j]   s.t. sign restrictions hold
        find min_Q  IRF[h, i, j]   s.t. sign restrictions hold
    Aggregate the (min, max) bounds across VAR draws.

This module returns both the identified-set extremes and the within-set
median, all computed by Monte Carlo over a Haar prior on Q (standard
relaxation; for hard cases use a denser ``n_draws_per_estimate``).

Reference
---------
Giacomini, R. and Kitagawa, T. (2021). Robust Bayesian inference for
set-identified models. Econometrica, 89(4), 1519-1556.
"""
from __future__ import annotations

import numpy as np

from ..._linalg import safe_cholesky
from ..estimate import estimate_var
from ..irf import irf as compute_irf
from ._results import GKRobustBandsResult


def _haar(n: int, rng: np.random.Generator) -> np.ndarray:
    """One Haar-uniform orthogonal matrix."""
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    return Q @ np.diag(np.sign(np.diag(R)))


def _check_signs(ir: np.ndarray, restrictions: dict, shock_idx: int) -> bool:
    for h, sign_vec in restrictions.items():
        for i, s in enumerate(sign_vec):
            if s == 0:
                continue
            if np.sign(ir[h, i, shock_idx]) != s:
                return False
    return True


def gk_robust_bands(
    Y: np.ndarray,
    *,
    p: int,
    horizon: int,
    restrictions: dict,
    shock_idx: int = 0,
    n_var_draws: int = 200,
    n_q_per_draw: int = 500,
    ci: float = 0.9,
    seed: int = 0,
) -> GKRobustBandsResult:
    """Giacomini-Kitagawa robust bands for a sign-restricted SVAR.

    Parameters
    ----------
    Y : (T, n) ndarray
    p : int — VAR lag order.
    horizon : int — IRF horizon.
    restrictions : dict {h: [+1/-1/0, ...]}
        Sign restrictions on the impulse response of each variable to
        the structural shock at column ``shock_idx``.
    shock_idx : int — which structural shock the restrictions describe.
    n_var_draws : int — bootstrap draws of (A, Sigma) for VAR uncertainty.
        Set to 1 to skip VAR uncertainty (point identified-set only).
    n_q_per_draw : int — Haar Q draws per VAR estimate to map the set.
    ci : float — outer credibility level for VAR uncertainty.

    Returns
    -------
    GKRobustBandsResult
        Frozen dataclass with ``irf_lower``, ``irf_upper``,
        ``irf_median`` (each ``(H+1, n)``), ``n_accepted_per_draw``.
    """
    rng = np.random.default_rng(seed)
    A_list, intercept, Sigma, residuals, _ = estimate_var(Y, p)
    T, n = Y.shape

    set_min = np.empty((n_var_draws, horizon + 1, n))
    set_max = np.empty((n_var_draws, horizon + 1, n))
    set_med = np.empty((n_var_draws, horizon + 1, n))
    n_acc = np.zeros(n_var_draws, dtype=int)

    for d in range(n_var_draws):
        if d == 0 or n_var_draws == 1:
            A_d, Sigma_d = A_list, Sigma
        else:
            idx = rng.integers(0, len(residuals), size=len(residuals))
            u_b = residuals[idx]
            Y_b = np.copy(Y)
            for t in range(p, T):
                yt = intercept.copy()
                for l in range(p):
                    yt += A_list[l] @ Y_b[t - 1 - l]
                Y_b[t] = yt + u_b[t - p]
            try:
                A_d, _, Sigma_d, _, _ = estimate_var(Y_b, p)
            except np.linalg.LinAlgError:
                set_min[d] = set_max[d] = set_med[d] = np.nan
                continue

        try:
            P = safe_cholesky(Sigma_d, name="gk_robust_bands Σ_d")
        except np.linalg.LinAlgError:
            # Per-draw failure marker; aggregation skips NaN rows.
            set_min[d] = set_max[d] = set_med[d] = np.nan
            continue

        accepted = []
        for _ in range(n_q_per_draw):
            Q = _haar(n, rng)
            ir = compute_irf(A_d, P @ Q, horizon)
            if _check_signs(ir, restrictions, shock_idx):
                accepted.append(ir[:, :, shock_idx])
        if not accepted:
            set_min[d] = set_max[d] = set_med[d] = np.nan
            continue
        stack = np.stack(accepted, axis=0)
        set_min[d] = stack.min(axis=0)
        set_max[d] = stack.max(axis=0)
        set_med[d] = np.median(stack, axis=0)
        n_acc[d] = len(accepted)

    lo_q = (1 - ci) / 2 * 100
    hi_q = 100 - lo_q
    return GKRobustBandsResult(
        irf_lower=np.nanpercentile(set_min, lo_q, axis=0),
        irf_upper=np.nanpercentile(set_max, hi_q, axis=0),
        irf_median=np.nanmedian(set_med, axis=0),
        n_accepted_per_draw=n_acc,
    )


def gk_robust_bands_from_gibbs(
    A_draws: np.ndarray,
    Sigma_draws: np.ndarray,
    *,
    horizon: int,
    restrictions: dict,
    shock_idx: int = 0,
    n_q_per_draw: int = 300,
    ci: float = 0.9,
    seed: int = 0,
) -> GKRobustBandsResult:
    """Giacomini-Kitagawa robust bands using pre-computed BVAR posterior draws.

    Companion to :func:`gk_robust_bands` for users who already have
    posterior samples (e.g., from :func:`puremacro.var.bvar.minnesota_gibbs`).
    The advantage over the bootstrap path inside ``gk_robust_bands`` is
    that the VAR uncertainty is the proper *posterior* — the Bayesian
    + identified-set blend that GK 2021 actually advocates.

    Parameters
    ----------
    A_draws : (D, p, n, n) — VAR coefficient draws.
    Sigma_draws : (D, n, n) — covariance draws.
    horizon, restrictions, shock_idx, n_q_per_draw, ci, seed :
        Same semantics as :func:`gk_robust_bands`.

    Returns
    -------
    GKRobustBandsResult — same shape as :func:`gk_robust_bands`.
    """
    rng = np.random.default_rng(seed)
    D, p, n, _ = A_draws.shape
    set_min = np.empty((D, horizon + 1, n))
    set_max = np.empty((D, horizon + 1, n))
    set_med = np.empty((D, horizon + 1, n))
    n_acc = np.zeros(D, dtype=int)

    for d in range(D):
        A_list = [A_draws[d, k] for k in range(p)]
        try:
            P = safe_cholesky(Sigma_draws[d] + 1e-12 * np.eye(n),
                                name="gk_robust_bands_from_gibbs Σ_d")
        except np.linalg.LinAlgError:
            set_min[d] = set_max[d] = set_med[d] = np.nan
            continue
        accepted = []
        for _ in range(n_q_per_draw):
            Q = _haar(n, rng)
            ir = compute_irf(A_list, P @ Q, horizon)
            if _check_signs(ir, restrictions, shock_idx):
                accepted.append(ir[:, :, shock_idx])
        if not accepted:
            set_min[d] = set_max[d] = set_med[d] = np.nan
            continue
        stack = np.stack(accepted, axis=0)
        set_min[d] = stack.min(axis=0)
        set_max[d] = stack.max(axis=0)
        set_med[d] = np.median(stack, axis=0)
        n_acc[d] = len(accepted)

    lo_q = (1 - ci) / 2 * 100
    hi_q = 100 - lo_q
    return GKRobustBandsResult(
        irf_lower=np.nanpercentile(set_min, lo_q, axis=0),
        irf_upper=np.nanpercentile(set_max, hi_q, axis=0),
        irf_median=np.nanmedian(set_med, axis=0),
        n_accepted_per_draw=n_acc,
    )


__all__ = ["gk_robust_bands", "gk_robust_bands_from_gibbs"]
