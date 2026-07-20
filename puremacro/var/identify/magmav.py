"""Magnusson-Mavroeidis (2014) SVAR via continuous heteroskedasticity.

Identifies the structural impact matrix B from regime-specific variance
shifts in reduced-form residuals, where break dates are *not* prespecified
but selected endogenously by a sup-Wald scan + BIC.

References
----------
Magnusson, L.M. and Mavroeidis, S. (2014). Identification using stability
    restrictions. Econometrica 82(5), 1799-1851.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import scipy.optimize

from ..estimate import estimate_var
from ..irf import irf as compute_irf
from ._results import MagMavSVARResult


# --------------------------------------------------------------------------- #
# Break detection
# --------------------------------------------------------------------------- #

def _sup_wald_one_break(resid: np.ndarray, *, lo_frac: float = 0.15,
                        hi_frac: float = 0.85) -> tuple[int, float]:
    """Single-break sup-Wald scan on residual covariance.

    For each candidate tau in [lo_frac*T, hi_frac*T], compute the LR-style
    statistic comparing the homoskedastic baseline to a two-regime fit.
    Return (argmax tau, max stat).
    """
    T, n = resid.shape
    lo = max(int(lo_frac * T), n + 2)
    hi = min(int(hi_frac * T), T - n - 2)
    if hi <= lo:
        return T // 2, -np.inf  # guard against spurious break acceptance
    # Precompute cumulative outer-product array: cum[t] = sum_{s=0}^{t-1} u_s u_s^T
    # Reduces per-tau work from O(T*n^2) to O(n^2).
    cum = np.zeros((T + 1, n, n))
    for t in range(T):
        cum[t + 1] = cum[t] + np.outer(resid[t], resid[t])
    S_full = cum[T] / T
    sign_full, log_det_full = np.linalg.slogdet(S_full)
    if sign_full <= 0:
        return T // 2, -np.inf
    log_det_full = float(log_det_full)
    best_tau, best_stat = lo, -np.inf
    for tau in range(lo, hi + 1):
        S0 = cum[tau] / tau
        S1 = (cum[T] - cum[tau]) / (T - tau)
        sign0, ld0 = np.linalg.slogdet(S0)
        sign1, ld1 = np.linalg.slogdet(S1)
        if sign0 <= 0 or sign1 <= 0:
            continue
        stat = T * log_det_full - tau * float(ld0) - (T - tau) * float(ld1)
        if stat > best_stat:
            best_stat = stat
            best_tau = tau
    return best_tau, float(best_stat)


def _detect_k_breaks(resid: np.ndarray, *, k: int, lo_frac: float = 0.15,
                     hi_frac: float = 0.85, min_sep_frac: float = 0.05) -> tuple[int, ...]:
    """Greedy multi-break detection: find the single best break in each
    remaining sub-segment, enforcing minimum separation."""
    T = resid.shape[0]
    min_sep = max(int(min_sep_frac * T), 5)
    breaks: list[int] = []
    segments: list[tuple[int, int]] = [(0, T)]
    for _ in range(k):
        # In each pass, find the segment whose internal sup-Wald is largest.
        best = (-1, -np.inf, 0)  # (tau, stat, seg_idx)
        for si, (a, b) in enumerate(segments):
            if b - a < 2 * min_sep:
                continue
            sub = resid[a:b]
            tau_local, stat = _sup_wald_one_break(
                sub,
                lo_frac=lo_frac,
                hi_frac=hi_frac,
            )
            tau_global = a + tau_local
            # Enforce min separation from existing breaks
            if breaks and min(abs(tau_global - bk) for bk in breaks) < min_sep:
                continue
            if stat > best[1]:
                best = (tau_global, stat, si)
        if best[0] < 0:
            break
        tau_global, _, si = best
        breaks.append(tau_global)
        # Split that segment
        a, b = segments.pop(si)
        segments.append((a, tau_global))
        segments.append((tau_global, b))
        segments.sort()
    return tuple(sorted(breaks))


def _bic_k_breaks(resid: np.ndarray, k: int, breaks: tuple[int, ...]) -> float:
    """BIC for k-break heteroskedastic-Gaussian model."""
    T, n = resid.shape
    boundaries = (0,) + breaks + (T,)
    ll = 0.0
    for g in range(len(boundaries) - 1):
        a, b = boundaries[g], boundaries[g + 1]
        Tg = b - a
        if Tg <= n + 1:
            return np.inf
        Sg = resid[a:b].T @ resid[a:b] / Tg
        sign, logdet = np.linalg.slogdet(Sg)
        if sign <= 0:
            return np.inf
        ll += -0.5 * Tg * (n * np.log(2 * np.pi) + logdet + n)
    # Free parameters: k break dates + (k+1)*n diagonal structural variances.
    # B matrix params don't depend on k; they cancel from BIC differences.
    n_params = k + (k + 1) * n
    return -2.0 * ll + n_params * np.log(T)


def _select_k_breaks(resid: np.ndarray, *, k_grid: tuple[int, ...] = (0, 1, 2, 3, 4)) -> tuple[int, tuple[int, ...]]:
    """Return (k, breaks) minimising BIC over k_grid."""
    best_breaks: tuple[int, ...] = tuple()
    best_k, best_bic = 0, np.inf
    for k in k_grid:
        br: tuple[int, ...]
        if k == 0:
            br = tuple()
        else:
            br = _detect_k_breaks(resid, k=k)
            if len(br) < k:
                continue
        bic = _bic_k_breaks(resid, k, br)
        if bic < best_bic:
            best_k, best_bic, best_breaks = k, bic, br
    return best_k, best_breaks


# --------------------------------------------------------------------------- #
# B-matrix estimation
# --------------------------------------------------------------------------- #

def _unpack_B(theta: np.ndarray, n: int) -> np.ndarray:
    """Reshape the flat parameter vector ``theta`` into an (n, n) B matrix (row-major)."""
    return theta.reshape(n, n)


def _loss_B(theta: np.ndarray, n: int, Sigmas: list[np.ndarray]) -> float:
    """Sum_g ||Sigma_g - B D_g B^T||_F^2, with D_g recovered analytically given B."""
    B = _unpack_B(theta, n)
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        return 1e20
    loss = 0.0
    for Sigma_g in Sigmas:
        # Best D_g given B is diag(B^{-1} Sigma_g B^{-T}); clipped to positive.
        D_g = B_inv @ Sigma_g @ B_inv.T
        d = np.clip(np.diag(D_g), 1e-10, None)
        diff = Sigma_g - B @ np.diag(d) @ B.T
        loss += float(np.sum(diff * diff))
    return loss


def _solve_D_given_B(B: np.ndarray, Sigmas: list[np.ndarray]) -> np.ndarray:
    """Returns D, shape (G, n): per-regime structural variances."""
    n = B.shape[0]
    B_inv = np.linalg.inv(B)
    D = np.zeros((len(Sigmas), n))
    for g, Sigma_g in enumerate(Sigmas):
        Dg = B_inv @ Sigma_g @ B_inv.T
        D[g] = np.clip(np.diag(Dg), 1e-10, None)
    return D


def _estimate_B_from_regime_covariances(
    Sigmas: list[np.ndarray], *, n_starts: int = 3, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Minimise sum_g ||Sigma_g - B D_g B^T||_F^2 with multi-start BFGS.

    Returns (B, D, success).
    """
    rng = np.random.default_rng(seed)
    n = Sigmas[0].shape[0]
    # Sensible starting point: Cholesky of regime-pooled Sigma.
    Sigma_pool = sum(Sigmas) / len(Sigmas)
    try:
        L0 = np.linalg.cholesky(Sigma_pool)
    except np.linalg.LinAlgError:
        L0 = np.linalg.cholesky(Sigma_pool + 1e-8 * np.eye(n))
    best_B, best_loss, ok = None, np.inf, False
    for s in range(n_starts):
        if s == 0:
            theta0 = L0.flatten()
        else:
            jitter = 0.05 * L0 * rng.standard_normal((n, n))
            theta0 = (L0 + jitter).flatten()
        try:
            res = scipy.optimize.minimize(
                _loss_B, theta0, args=(n, Sigmas), method="BFGS",
                options={"maxiter": 300, "gtol": 1e-6},
            )
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            continue
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_B = _unpack_B(res.x, n)
            ok = bool(res.success)
    if best_B is None:
        D_fb = _solve_D_given_B(L0, Sigmas)
        return L0, D_fb, False
    D = _solve_D_given_B(best_B, Sigmas)
    # Absorb the geometric-mean structural variance into B columns to resolve the
    # column-scaling ambiguity: B_norm[:,j] = B[:,j] * sqrt(geom_mean_g D[g,j]).
    # After this normalisation every valid solution on the identification manifold
    # maps to the same canonical B (up to column permutation and sign).
    geom_mean_D = np.exp(np.mean(np.log(np.clip(D, 1e-10, None)), axis=0))
    best_B = best_B * np.sqrt(geom_mean_D)
    D = D / geom_mean_D
    return best_B, D, ok


def _normalise_B(B: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalise B such that:
        1. Columns ordered by descending cross-regime variance ratio
           r_j = max_g D[g, j] / min_g D[g, j].
        2. diag(B) >= 0 (sign-flip columns whose diagonal entry is negative).

    Returns (B_norm, D_norm, order).
    """
    n = B.shape[0]
    # Avoid divide-by-zero
    safe_min = np.clip(D.min(axis=0), 1e-12, None)
    ratios = D.max(axis=0) / safe_min
    order = np.argsort(-ratios)
    B = B[:, order].copy()
    D = D[:, order].copy()
    for j in range(n):
        if B[j, j] < 0:
            B[:, j] *= -1
    return B, D, order


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def _residuals_to_regime_covs(resid: np.ndarray, breaks: tuple[int, ...]) -> list[np.ndarray]:
    """Slice residuals into regimes and return covariance matrices."""
    T = resid.shape[0]
    bounds = (0,) + breaks + (T,)
    out = []
    for g in range(len(bounds) - 1):
        a, b = bounds[g], bounds[g + 1]
        out.append(resid[a:b].T @ resid[a:b] / (b - a))
    return out


def _regime_bootstrap_indices(T: int, breaks: tuple[int, ...], rng) -> np.ndarray:
    """Sample WITH replacement within each regime to preserve heteroskedasticity."""
    bounds = (0,) + breaks + (T,)
    idx = np.empty(T, dtype=int)
    for g in range(len(bounds) - 1):
        a, b = bounds[g], bounds[g + 1]
        idx[a:b] = rng.integers(a, b, size=b - a)
    return idx


def magmav_svar(
    Y: np.ndarray,
    *,
    p: int,
    horizon: int = 20,
    k_breaks: Optional[int] = None,
    n_boot: int = 500,
    ci: float = 0.9,
    seed: int = 0,
) -> MagMavSVARResult:
    """Magnusson-Mavroeidis (2014) SVAR via continuous heteroskedasticity.

    Parameters
    ----------
    Y : (T, n) reduced-form data.
    p : VAR lag order.
    horizon : IRF horizon H (output has shape (H+1, n, n)).
    k_breaks : if None, choose via BIC over {0, 1, 2, 3, 4}.
    n_boot, ci, seed : bootstrap controls.

    Returns
    -------
    MagMavSVARResult
    """
    Y = np.asarray(Y, dtype=float)
    rng = np.random.default_rng(seed)
    A_list, _, _, resid, _ = estimate_var(Y, p)
    T_eff = resid.shape[0]
    n = Y.shape[1]

    # 1. Select breaks
    if k_breaks is None:
        k_sel, breaks = _select_k_breaks(resid)
    else:
        k_sel = int(k_breaks)
        breaks = _detect_k_breaks(resid, k=k_sel) if k_sel > 0 else tuple()
    if len(breaks) < k_sel:
        # Sup-Wald couldn't find that many breaks; fall back.
        k_sel = len(breaks)

    # 2. Estimate B
    if k_sel == 0:
        # Homoskedastic fallback: Cholesky of Sigma; eu = (0, 0) signals failure.
        Sigma = resid.T @ resid / T_eff
        B = np.linalg.cholesky(Sigma)
        eu = (0, 0)
        warnings.warn(
            "magmav_svar: k_breaks selected as 0; identification not achieved; "
            "returning Cholesky fallback.",
            stacklevel=2,
        )
    else:
        Sigmas = _residuals_to_regime_covs(resid, breaks)
        B, D, success = _estimate_B_from_regime_covariances(Sigmas, n_starts=3, seed=seed)
        if not success:
            warnings.warn(
                "magmav_svar: B-matrix optimisation did not converge; returning best draw.",
                stacklevel=2,
            )
        B, D, _ = _normalise_B(B, D)
        eu = (1, 1) if success else (0, 0)

    irf_point = compute_irf(A_list, B, horizon)

    # 3. Bootstrap with regime-preserving resampling
    lo_pct = 100 * (1 - ci) / 2
    hi_pct = 100 * (1 + ci) / 2
    boot_irfs: list[np.ndarray] = []
    n_fail = 0
    for b in range(n_boot):
        idx = _regime_bootstrap_indices(T_eff, breaks, rng)
        u_boot = resid[idx]
        Y_boot = np.zeros((T_eff + p, n))
        Y_boot[:p] = Y[:p]
        for t in range(p, T_eff + p):
            x = sum(A_list[l] @ Y_boot[t - l - 1] for l in range(p))
            Y_boot[t] = x + u_boot[t - p]
        try:
            A_b, _, _, resid_b, _ = estimate_var(Y_boot, p)
            if k_sel == 0:
                B_b = np.linalg.cholesky(resid_b.T @ resid_b / resid_b.shape[0])
            else:
                Sigmas_b = _residuals_to_regime_covs(resid_b, breaks)
                # n_starts=1 in the bootstrap to keep total runtime tractable;
                # the point estimate above uses n_starts=3.
                B_b, _, _ = _estimate_B_from_regime_covariances(Sigmas_b, n_starts=1, seed=seed + b + 1)
                B_b, _, _ = _normalise_B(B_b, _solve_D_given_B(B_b, Sigmas_b))
            boot_irfs.append(compute_irf(A_b, B_b, horizon))
        # ValueError can leak from scipy.optimize on rare pathological draws.
        except (np.linalg.LinAlgError, ValueError):
            n_fail += 1
            continue
    if n_fail / max(n_boot, 1) > 0.05:
        warnings.warn(
            f"magmav_svar: {n_fail}/{n_boot} bootstrap draws failed "
            f"({n_fail / n_boot:.1%}).",
            stacklevel=2,
        )
    if len(boot_irfs) == 0:
        irf_lower = np.full_like(irf_point, np.nan)
        irf_upper = np.full_like(irf_point, np.nan)
    else:
        arr = np.stack(boot_irfs)
        irf_lower = np.percentile(arr, lo_pct, axis=0)
        irf_upper = np.percentile(arr, hi_pct, axis=0)

    return MagMavSVARResult(
        irf_point=irf_point,
        irf_lower=irf_lower,
        irf_upper=irf_upper,
        B=B,
        variance_change_dates=tuple(int(x) for x in breaks),
        k_breaks=int(k_sel),
        n_boot=int(n_boot),
        ci=float(ci),
        eu=eu,
        n_fail=int(n_fail),
    )


__all__ = ["magmav_svar"]
