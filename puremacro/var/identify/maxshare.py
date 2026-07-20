"""Uhlig (2003) / Faust (1998) maximum-share identification.

The 'max-share' shock at horizon h* is the linear combination of
reduced-form innovations that maximises the contribution of one shock
to Var(y_{j, t+h*}).

Barsky-Sims (2011) news shock: same engine at h -> infinity (large H).

Full pipeline
-------------
:func:`identify_maxshare` wraps VAR estimation, the closed-form
max-share optimisation, IRF/FEVD computation, and a residual bootstrap
into a single call returning a frozen :class:`MaxShareResult`. The
low-level :func:`maxshare` and :func:`news_maxshare` helpers are
unchanged and continue to return ``B0`` ndarrays.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import scipy.linalg

from ..._linalg import safe_cholesky
from ..estimate import estimate_var
from ._results import MaxShareResult

# Mirror the same thresholds as var/identify/cholesky.py so the bootstrap
# gate is consistent across all identification methods.
_BOOT_FAIL_WARN_THRESHOLD = 0.05
_BOOT_COND_RATIO_FLOOR = 1e-4


def maxshare(A_list, Sigma, *, target_var: int, horizon: int) -> np.ndarray:
    """Return B0 such that the first column of B0 is the rotation of
    chol(Sigma) that maximises the contribution of shock 1 to
    Var(y_{target_var, t+horizon}).

    Parameters
    ----------
    A_list : list of (n, n) ndarrays
        VAR coefficient matrices A_1, ..., A_p.
    Sigma : (n, n) ndarray
        Reduced-form residual covariance.
    target_var : int
        Index of the variable whose forecast-error variance we maximise.
    horizon : int
        Horizon h* over which the variance is summed.

    Returns
    -------
    B0 : (n, n) ndarray
        Lower-triangular Cholesky times an orthogonal Q whose first
        column is the optimal rotation.
    """
    P = safe_cholesky(Sigma, name="maxshare Σ")
    n = P.shape[0]

    # MA coefficients up to horizon
    Phi_list = [np.eye(n)]
    for h in range(1, horizon + 1):
        ph = np.zeros((n, n))
        for j in range(1, min(h, len(A_list)) + 1):
            ph += Phi_list[h - j] @ A_list[j - 1]
        Phi_list.append(ph)

    # M = sum_{h=0}^{H} (e_target' Phi_h P)' (e_target' Phi_h P)
    e = np.zeros(n); e[target_var] = 1.0
    M = np.zeros((n, n))
    for h in range(0, horizon + 1):
        v = Phi_list[h] @ P
        row = e @ v             # shape (n,)
        M += np.outer(row, row)

    # Top eigenvector of M is the optimal q1
    eigvals, eigvecs = np.linalg.eigh(M)
    q1 = eigvecs[:, -1]

    # Complete to orthonormal Q via Gram-Schmidt against random vectors
    rng = np.random.default_rng(0)
    Q = np.zeros((n, n))
    Q[:, 0] = q1
    for k in range(1, n):
        v = rng.standard_normal(n)
        for j in range(k):
            v -= (v @ Q[:, j]) * Q[:, j]
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            # Degenerate -- fallback to standard basis projection
            v = np.eye(n)[:, k] - sum((np.eye(n)[:, k] @ Q[:, j]) * Q[:, j]
                                       for j in range(k))
            norm = np.linalg.norm(v)
        Q[:, k] = v / norm
    return P @ Q


def news_maxshare(A_list, Sigma, *, target_var: int, horizon: int = 40) -> np.ndarray:
    """Barsky-Sims (2011) news shock -- max-share at large horizon."""
    return maxshare(A_list, Sigma, target_var=target_var, horizon=horizon)


# --------------------------------------------------------------------------- #
# Private helpers for the full pipeline (identify_maxshare)
# --------------------------------------------------------------------------- #

def _companion_irfs(
    A_list: list,
    horizon: int,
) -> np.ndarray:
    """Compute reduced-form MA coefficients Phi_h for h = 0 … H.

    Parameters
    ----------
    A_list : list of (n, n) arrays — A_1, ..., A_p (lag-1 first)
    horizon : int

    Returns
    -------
    ndarray (H+1, n, n) — Phi[0] = I_n
    """
    n = A_list[0].shape[0]
    p = len(A_list)
    Phi = np.zeros((horizon + 1, n, n))
    Phi[0] = np.eye(n)
    for h in range(1, horizon + 1):
        for lag in range(min(h, p)):
            Phi[h] += Phi[h - lag - 1] @ A_list[lag]
    return Phi


def _build_M_matrix(
    A_list: list,
    Sigma_u: np.ndarray,
    chol: np.ndarray,
    target_idx: int,
    max_fev_at: int,
) -> np.ndarray:
    """Build the matrix M whose top eigenvector is the optimal q.

    Derivation
    ----------
    FEV share of variable ``target_idx`` at horizon h* attributed to shock
    with impact vector c = chol(Sigma_u) @ q is::

        share(h*; q) = (sum_{s=0}^{h*-1} (e_i' Phi_s c)^2)
                       / (sum_{s=0}^{h*-1} e_i' Phi_s Sigma_u Phi_s' e_i)

    Numerator = q' M q where::

        M = sum_{s=0}^{h*-1} (Phi_s chol)' e_i e_i' (Phi_s chol)

    Maximizing q'Mq subject to q'q=1 gives the top eigenvector of M.
    The denominator is a positive constant w.r.t. q.

    Parameters
    ----------
    A_list : list of (n, n) arrays
    Sigma_u : ndarray (n, n)
    chol : ndarray (n, n) — lower-triangular Cholesky factor of Sigma_u
    target_idx : int
    max_fev_at : int — h* (number of horizons to accumulate, 1-indexed)

    Returns
    -------
    M : ndarray (n, n), symmetric PSD
    """
    n = A_list[0].shape[0]
    Phi = _companion_irfs(A_list, horizon=max_fev_at - 1)  # Phi[0..h*-1]
    e_i = np.zeros(n)
    e_i[target_idx] = 1.0
    M = np.zeros((n, n))
    for s in range(max_fev_at):
        v = Phi[s] @ chol                    # (n, n)
        col = v.T @ e_i                      # (n,) = (Phi_s chol)' e_i
        M += np.outer(col, col)
    return M


def _complete_B(B_col0: np.ndarray) -> np.ndarray:
    """Complete structural impact matrix B via Gram-Schmidt.

    Returns full (n, n) matrix whose first column is B_col0 and
    remaining columns span the orthogonal complement (standard metric).
    """
    n = len(B_col0)
    B = np.zeros((n, n))
    B[:, 0] = B_col0
    candidates = np.eye(n)
    idx = 1
    for j in range(n):
        v = candidates[:, j].copy()
        for k in range(idx):
            v -= (v @ B[:, k]) / (B[:, k] @ B[:, k]) * B[:, k]
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            B[:, idx] = v / norm
            idx += 1
        if idx == n:
            break
    return B


def _compute_fevd_from_irfs(irfs: np.ndarray) -> np.ndarray:
    """FEVD from structural IRFs. Returns (H+1, n, n)."""
    cum_sq = np.cumsum(irfs ** 2, axis=0)
    total = cum_sq.sum(axis=2, keepdims=True)
    total = np.where(total == 0, 1.0, total)
    return cum_sq / total


def _identify_one(
    A_list: list,
    Sigma_u: np.ndarray,
    chol: np.ndarray,
    target_idx: int,
    max_fev_at: int,
    horizon: int,
) -> tuple:
    """Core identification: given estimated (A_list, Sigma_u, chol), return
    (B, q, irfs, fevd, fev_share_at_target).
    """
    n = A_list[0].shape[0]
    M = _build_M_matrix(A_list, Sigma_u, chol, target_idx=target_idx, max_fev_at=max_fev_at)
    eigenvalues, eigenvectors = scipy.linalg.eigh(M)
    q = eigenvectors[:, -1]
    q = q / np.linalg.norm(q)

    B_col0 = chol @ q
    B = _complete_B(B_col0)

    Phi = _companion_irfs(A_list, horizon=horizon)
    irfs = np.einsum("hij,jk->hik", Phi, B)    # (H+1, n, n)
    fevd = _compute_fevd_from_irfs(irfs)

    # FEV share at the identification horizon (max_fev_at - 1 index, 0-based).
    # The M matrix accumulates horizons 0..max_fev_at-1 (inclusive).
    # We report the FEVD value at that same horizon.
    h_idx = max_fev_at - 1
    fev_share = float(fevd[h_idx, target_idx, 0])
    return B, q, irfs, fevd, fev_share


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #

def identify_maxshare(
    Y: np.ndarray,
    *,
    p: int = 2,
    target_idx: int = 0,
    max_fev_at: int = 1,
    horizon: int = 20,
    n_bootstrap: int = 500,
    ci: float = 0.68,
    seed: int = 0,
) -> MaxShareResult:
    """Faust-Uhlig max-share SVAR — full pipeline with residual bootstrap.

    Identifies the structural shock that maximises the cumulative forecast-
    error-variance (FEV) share of ``Y[:, target_idx]`` over horizons
    ``0..max_fev_at-1``. The solution is the top eigenvector of a symmetric
    PSD matrix, making the optimisation closed-form.

    Inference uses an inline residual bootstrap (Kilian 1998) gated by
    :func:`puremacro._linalg.safe_cholesky`: degenerate bootstrap draws
    (non-PD Sigma_b) are dropped.  If more than 5% of draws fail a warning
    is issued; if all draws fail a ``LinAlgError`` is raised.

    Low-level helpers :func:`maxshare` and :func:`news_maxshare` are
    unchanged and continue to return B0 ndarrays.

    Parameters
    ----------
    Y : ndarray (T, n)
        Data matrix.
    p : int
        VAR lag order.
    target_idx : int
        Index of the variable whose FEV share is maximised (0 = first).
    max_fev_at : int
        Horizon h* at which FEV share is maximised.  1 = impact (default).
    horizon : int
        IRF horizon H for output IRFs and FEVD.
    n_bootstrap : int
        Residual bootstrap repetitions.  0 to skip (bands are ``None``).
    ci : float
        Coverage for bootstrap bands.  ``ci=0.68`` gives 68% bands
        (i.e. the 16th–84th percentile range).  This is a *coverage*
        fraction, not a percentile level: internally ``lo_q = (1-ci)/2``,
        ``hi_q = 1 - lo_q``.
    seed : int
        Seed for the bootstrap RNG.

    Returns
    -------
    MaxShareResult
        Frozen dataclass with ``B``, ``q``, ``fev_share_at_target``,
        ``irfs``, ``fevd``, ``max_fev_at``, ``irf_lower``, ``irf_upper``,
        ``ci``.

    References
    ----------
    Barsky, R.B. and Sims, E.R. (2011). News shocks and business cycles.
        J. Monetary Econ. 58(3), 273-289.
    Faust, J. (1998). The robustness of identified VAR conclusions about
        money. Carnegie-Rochester Conference Series 49, 207-244.
    Kilian, L. (1998). Small-sample confidence intervals for impulse
        response functions. Rev. Econ. Stat. 80(2), 218-230.
    Uhlig, H. (2004). Do technology shocks lead to a fall in total hours
        worked? J. Eur. Econ. Assoc. 2(2-3), 361-371.
    """
    T, n = Y.shape
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # 1.  Estimate reduced-form VAR(p)
    # ------------------------------------------------------------------ #
    A_list, c, Sigma_u, residuals, _ = estimate_var(Y, p)
    chol = safe_cholesky(Sigma_u, name="identify_maxshare Sigma_u")

    # ------------------------------------------------------------------ #
    # 2-4.  Identification, IRFs, FEVD
    # ------------------------------------------------------------------ #
    B, q, irfs, fevd, fev_share = _identify_one(
        A_list, Sigma_u, chol, target_idx=target_idx,
        max_fev_at=max_fev_at, horizon=horizon,
    )

    # ------------------------------------------------------------------ #
    # 5.  Residual bootstrap
    # ------------------------------------------------------------------ #
    irf_lower = irf_upper = None
    if n_bootstrap > 0:
        lo_q = (1.0 - ci) / 2.0
        hi_q = 1.0 - lo_q

        accepted = []
        n_fail = 0

        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(residuals), size=len(residuals))
            eps_b = residuals[idx]
            Yb = np.zeros_like(Y)
            Yb[:p] = Y[:p]
            for t in range(p, T):
                yt = c.copy()
                for lag in range(p):
                    yt = yt + A_list[lag] @ Yb[t - lag - 1]
                yt = yt + eps_b[t - p]
                Yb[t] = yt

            A_b, _, Sigma_b, _, _ = estimate_var(Yb, p)

            try:
                chol_b = safe_cholesky(Sigma_b, name="identify_maxshare bootstrap Sigma_b")
            except np.linalg.LinAlgError:
                n_fail += 1
                continue

            # Reject ill-conditioned draws (cond > 1e8) — same gate as cholesky.py
            diag_b = np.abs(np.diag(chol_b))
            if (not np.all(np.isfinite(diag_b))
                    or (diag_b.size and diag_b.min() < _BOOT_COND_RATIO_FLOOR * diag_b.max())):
                n_fail += 1
                continue

            _, _, irfs_b, _, _ = _identify_one(
                A_b, Sigma_b, chol_b, target_idx=target_idx,
                max_fev_at=max_fev_at, horizon=horizon,
            )
            accepted.append(irfs_b)

        if not accepted:
            raise np.linalg.LinAlgError(
                f"identify_maxshare: all {n_bootstrap} bootstrap draws produced a "
                "non-PD reduced-form Sigma. The data may be too short for p lags."
            )

        fail_rate = n_fail / n_bootstrap
        if fail_rate > _BOOT_FAIL_WARN_THRESHOLD:
            warnings.warn(
                f"identify_maxshare: {n_fail}/{n_bootstrap} bootstrap draws "
                f"({fail_rate:.1%}) produced a non-PD Sigma and were dropped. "
                "Bands are computed from the surviving draws and may be "
                "unreliable; consider more data, fewer lags, or a different "
                "identification scheme.",
                stacklevel=2,
            )

        draws = np.stack(accepted, axis=0)   # (B_accepted, H+1, n, n)
        irf_lower = np.quantile(draws, lo_q, axis=0)
        irf_upper = np.quantile(draws, hi_q, axis=0)

    return MaxShareResult(
        B=B,
        q=q,
        fev_share_at_target=fev_share,
        irfs=irfs,
        fevd=fevd,
        max_fev_at=max_fev_at,
        irf_lower=irf_lower,
        irf_upper=irf_upper,
        ci=ci,
    )


__all__ = ["maxshare", "news_maxshare", "identify_maxshare"]
