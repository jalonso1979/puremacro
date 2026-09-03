"""SVAR identification via non-Gaussianity (Lanne-Meitz-Saikkonen 2017).

If reduced-form residuals u_t are mutually independent and at most one
is Gaussian, then the structural impact matrix B0 is identified up to
column ordering and sign — no sign / zero / proxy restrictions
required. Standard implementation: run FastICA on the whitened
residuals; the ICA un-mixing matrix gives the rotation Q such that
B0 = chol(Sigma) @ Q.

References
----------
Lanne, M., Meitz, M., and Saikkonen, P. (2017). Identification and
    estimation of non-Gaussian structural vector autoregressions.
    Journal of Econometrics 196(2), 288-304.
Hyvärinen, A. and Oja, E. (2000). Independent component analysis:
    algorithms and applications. Neural Networks 13(4-5), 411-430.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..._linalg import safe_cholesky

from ..estimate import estimate_var
from ..irf import irf as compute_irf
from ._results import NonGaussianSVARResult


def _fast_ica(X: np.ndarray, n_iter: int = 200, tol: float = 1e-6,
              rng=None) -> np.ndarray:
    """FastICA on rows of X (after whitening). Returns un-mixing matrix W.

    Uses the symmetric (parallel) version with the ``logcosh`` non-linearity
    g(u) = tanh(u), g'(u) = 1 - tanh(u)^2.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n, T = X.shape
    # Initialise random orthogonal W
    W = rng.standard_normal((n, n))
    W, _ = np.linalg.qr(W)
    for _ in range(n_iter):
        WX = W @ X
        gWX = np.tanh(WX)
        g_prime = 1.0 - gWX ** 2
        W_new = (gWX @ X.T) / T - np.diag(g_prime.mean(axis=1)) @ W
        # Symmetric decorrelation: W <- (W W')^{-1/2} W
        U, s, Vt = np.linalg.svd(W_new, full_matrices=False)
        W_new = U @ Vt
        if np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1.0)) < tol:
            W = W_new
            break
        W = W_new
    return W


def variance_decomposition_consistency(B0: np.ndarray, sigma_u: np.ndarray, *,
                                       tol: float = 1e-6) -> dict:
    """Sanity-check ``B0 @ B0.T ≈ Σ_u``.

    Parameters
    ----------
    B0 : np.ndarray
        Structural impact matrix, shape (n, n).
    sigma_u : np.ndarray
        Reduced-form residual covariance, shape (n, n).
    tol : float, optional
        Tolerance for the ``passed`` flag — 1e-6 is many orders of magnitude
        above typical floating-point accumulation noise for n ≤ 10 but well
        below any realistic identification error.  Default 1e-6.

    Returns
    -------
    dict
        Keys: ``max_abs_diff``, ``rms_diff``, ``passed`` (bool, True iff
        ``max_abs_diff < tol``).
    """
    diff = B0 @ B0.T - sigma_u
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff ** 2)))
    return {
        "max_abs_diff": max_abs,
        "rms_diff": rms,
        "passed": max_abs < tol,
    }


def gaussian_lr_test(B0: np.ndarray, residuals: np.ndarray) -> dict:
    """Kurtosis-based likelihood-ratio test for non-Gaussian structural shocks.

    Null: structural shocks ε = B0^{-1} u are i.i.d. Gaussian (excess
    kurtosis = 0 for every shock).  Alternative: at least one shock has
    non-zero excess kurtosis.

    The statistic is the Jarque-Bera kurtosis component summed over all n
    shocks::

        stat = (T / 24) · Σ_j K_j²

    where K_j is the sample excess kurtosis of shock j.  Under H₀ this is
    asymptotically χ²(n).  Unlike a KDE-based LR, this statistic has no
    in-sample overfitting problem and converges at rate √T.

    The chi-squared(n) calibration follows the Jarque-Bera kurtosis-only
    component: each shock contributes one degree of freedom via the
    sample fourth-moment CLT.  See Bai and Ng (2005, JBES) for the
    robustness of this statistic under serial dependence, and
    Lanne-Meitz-Saikkonen (2017) for the structural-shock context.

    Returns dict with ``stat``, ``df`` (= n), ``p_value``.
    """
    from scipy.stats import chi2
    T, n = residuals.shape
    eps = np.linalg.solve(B0, residuals.T).T  # structural shocks (T, n)
    excess_kurts = np.empty(n)
    for j in range(n):
        x = eps[:, j]
        mu4 = float(np.mean((x - x.mean()) ** 4))
        sigma2 = float(np.var(x, ddof=0))
        excess_kurts[j] = mu4 / max(sigma2 ** 2, 1e-24) - 3.0
    stat = float(T / 24.0 * np.sum(excess_kurts ** 2))
    df = n
    p_value = float(chi2.sf(stat, df=df))
    return {"stat": stat, "df": df, "p_value": p_value}


def _tiebreak_kurtosis_order(kurt: np.ndarray, src: np.ndarray, *,
                             tol: float = 1e-3) -> np.ndarray:
    """Sort indices by descending |kurt|; for entries within ``tol * max(|kurt|)``
    of each other, break ties by |skewness| then |5th central moment|, then by
    original index.

    Emits a warning when any tie-break is invoked.
    """
    n = kurt.size
    abs_k = np.abs(kurt)
    k_max = max(float(abs_k.max()), 1e-12)
    # Compute moments
    centred = src - src.mean(axis=0)
    std = centred.std(axis=0, ddof=0)
    std_safe = np.where(std > 0, std, 1.0)
    skew = ((centred / std_safe) ** 3).mean(axis=0)
    m5 = ((centred / std_safe) ** 5).mean(axis=0)
    # Build sort key as a tuple: (bucketed -|kurt|, -|skew|, -|m5|, original_idx)
    bucket = np.round(abs_k / (tol * k_max)) * (tol * k_max)
    keys = [(-bucket[i], -abs(skew[i]), -abs(m5[i]), i) for i in range(n)]
    order = np.array(sorted(range(n), key=lambda i: keys[i]))
    # Detect that tie-break actually changed the ordering.
    pure_order = np.argsort(-abs_k, kind="stable")
    needed = not np.array_equal(order, pure_order)
    if needed:
        warnings.warn(
            "non_gaussian_svar: kurtosis tiebreak invoked — column order "
            "resolved via skewness / 5th moment. Ordering may be sensitive "
            "to sample noise.",
            stacklevel=3,
        )
    return order


def non_gaussian_svar(
    Y: np.ndarray,
    *,
    p: int,
    horizon: int,
    seed: int = 0,
) -> NonGaussianSVARResult:
    """LMS (2017) non-Gaussian SVAR via FastICA on reduced-form residuals.

    Returns
    -------
    NonGaussianSVARResult
        Frozen dataclass with ``B0`` (n, n), ``Q`` (n, n), ``kurtosis``
        (n,), ``irf`` (H+1, n, n), ``ordering_by_kurt`` (n,).
    """
    A_list, _, Sigma, residuals, _ = estimate_var(Y, p)
    # Whiten residuals: u = chol(Sigma) e, so e = chol(Sigma)^{-1} u.
    L = safe_cholesky(Sigma, name="non_gaussian_svar Σ")
    e = np.linalg.solve(L, residuals.T)  # (n, T)
    # Centre rows
    e = e - e.mean(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    W = _fast_ica(e, rng=rng)
    # The structural shocks are W @ e; ensure unit variance by rescaling W.
    src = W @ e
    s_std = src.std(axis=1, ddof=0)
    W = (W.T / np.maximum(s_std, 1e-12)).T
    src = W @ e
    # Q = W^{-1} so that e = Q src and u = chol(Sigma) e = chol(Sigma) Q src.
    Q = np.linalg.inv(W)
    # Order columns of Q by descending |excess kurtosis|.
    kurt = (src ** 4).mean(axis=1) / np.maximum((src ** 2).mean(axis=1) ** 2, 1e-12) - 3.0
    order = np.argsort(-np.abs(kurt))
    Q = Q[:, order]
    kurt = kurt[order]
    # Sign normalisation: make the diagonal of B0 non-negative.
    B0 = L @ Q
    for j in range(B0.shape[1]):
        if B0[j, j] < 0:
            B0[:, j] *= -1
            Q[:, j] *= -1
    # 0.51.0: tiebreak when adjacent |kurt| values are close.
    tb_order = _tiebreak_kurtosis_order(kurt, src.T, tol=1e-3)
    if not np.array_equal(tb_order, np.arange(B0.shape[1])):
        B0 = B0[:, tb_order]
        Q = Q[:, tb_order]
        kurt = kurt[tb_order]
    irf_arr = compute_irf(A_list, B0, horizon)
    # 0.51.0: populate diagnostics.
    Sigma_u = residuals.T @ residuals / residuals.shape[0]
    lr = gaussian_lr_test(B0, residuals)
    cons = variance_decomposition_consistency(B0, Sigma_u)
    return NonGaussianSVARResult(
        B0=B0,
        Q=Q,
        kurtosis=kurt,
        irf=irf_arr,
        ordering_by_kurt=order[tb_order],
        lr_test=lr,
        consistency_check=cons,
    )


__all__ = [
    "non_gaussian_svar",
    "gaussian_lr_test",
    "variance_decomposition_consistency",
]
