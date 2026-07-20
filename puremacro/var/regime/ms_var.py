"""Markov-switching VAR with regime-dependent intercept and covariance.

Estimation by EM (Hamilton 1989 / Krolzig 1997):
  E-step: forward-backward Hamilton filter + Kim smoother — smoothed
          regime probabilities ``xi_{T|T}`` and pairwise smoothed
          probabilities ``xi_pair`` for the transition update.
  M-step: weighted OLS for shared autoregressive coefficients,
          weighted means for regime-specific intercepts ``mu_k``,
          weighted residual covariances ``Sigma_k``, and updated
          transition matrix from the smoothed pairwise probabilities.

The shared-A specification (regime-dependent mean and variance) is the
classic teaching version and the one that reproduces the Hamilton
(1989) recession dating result on US GNP growth. Full regime-dependent
``A_k`` is a small extension (see comments in ``_m_step_A``).

References
----------
Hamilton, J.D. (1989). A new approach to the economic analysis of
    nonstationary time series and the business cycle. Econometrica
    57(2), 357-384.
Krolzig, H.-M. (1997). Markov-Switching Vector Autoregressions.
    Springer Lecture Notes 454.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..._linalg import safe_cholesky


@dataclass
class MSVARResult:
    """Output of MS-VAR fit.

    Attributes
    ----------
    K : int — number of regimes.
    A : (n, n*p) ndarray — shared VAR coefficient matrix [A_1, ..., A_p].
    mu : (K, n) ndarray — regime-specific intercepts.
    Sigma : (K, n, n) ndarray — regime-specific covariances.
    P : (K, K) ndarray — transition probability matrix, P[i,j] = P(s_t=j | s_{t-1}=i).
    smoothed_probs : (T_eff, K) ndarray — Kim smoother output.
    filtered_probs : (T_eff, K) ndarray — Hamilton filter output.
    loglik : float — log-likelihood at convergence.
    n_iter : int — EM iterations used.
    converged : bool.
    """
    K: int
    A: np.ndarray
    mu: np.ndarray
    Sigma: np.ndarray
    P: np.ndarray
    smoothed_probs: np.ndarray
    filtered_probs: np.ndarray
    loglik: float
    n_iter: int
    converged: bool


def _gauss_density(y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    n = len(y)
    diff = y - mean
    try:
        L = safe_cholesky(cov + 1e-12 * np.eye(n), name="ms_var density")
    except np.linalg.LinAlgError:
        # Per the ms_var Hamilton-filter contract, return a tiny but
        # non-zero density rather than propagating the failure.
        return 1e-12
    z = np.linalg.solve(L, diff)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return float(np.exp(-0.5 * (n * np.log(2 * np.pi) + log_det + z @ z)))


def _hamilton_filter(densities: np.ndarray, P: np.ndarray, pi0: np.ndarray):
    """Forward filter. densities: (T, K). Returns (filtered, predicted, ll)."""
    T, K = densities.shape
    filt = np.empty((T, K))
    pred = np.empty((T, K))
    ll = 0.0
    pred_t = pi0
    for t in range(T):
        joint = pred_t * densities[t]
        s = joint.sum()
        if s <= 0:
            joint = np.full(K, 1e-12)
            s = float(joint.sum())
        ll += np.log(s)
        filt[t] = joint / s
        pred[t] = pred_t
        pred_t = filt[t] @ P
    return filt, pred, ll


def _kim_smoother(filt: np.ndarray, P: np.ndarray):
    """Kim 1994 backward recursion. Returns (smoothed, pairwise_smoothed)."""
    T, K = filt.shape
    sm = np.empty((T, K))
    sm[-1] = filt[-1]
    pairwise = np.empty((T - 1, K, K))
    for t in range(T - 2, -1, -1):
        pred_next = filt[t] @ P  # length K
        for j in range(K):
            if pred_next[j] <= 0:
                continue
            for i in range(K):
                pairwise[t, i, j] = (
                    filt[t, i] * P[i, j] * sm[t + 1, j] / pred_next[j]
                )
        sm[t] = pairwise[t].sum(axis=1)
    return sm, pairwise


def ms_var_fit(
    Y: np.ndarray,
    K: int = 2,
    p: int = 1,
    *,
    n_iter: int = 100,
    tol: float = 1e-5,
    seed: int = 0,
    verbose: bool = False,
) -> MSVARResult:
    """EM estimation of an MS-VAR(p) with K regimes.

    Parameters
    ----------
    Y : (T, n) ndarray.
    K : int — number of regimes.
    p : int — VAR lag order (shared across regimes).
    n_iter, tol : EM controls.
    seed : random seed for the regime initialisation.

    Returns
    -------
    MSVARResult.
    """
    Y = np.asarray(Y, dtype=float)
    T, n = Y.shape
    rng = np.random.default_rng(seed)

    # Build VAR design (without intercept; we'll add regime-specific mu_k).
    # y_t = mu_{s_t} + A_1 y_{t-1} + ... + A_p y_{t-p} + e_t
    Y_dep = Y[p:]
    X_lag = np.column_stack(
        [Y[p - k - 1: T - k - 1] for k in range(p)]
    )
    T_eff = Y_dep.shape[0]

    # --- initialisation ---
    # Initialise smoothed_probs by clustering on residuals from a single VAR.
    beta_init = np.linalg.lstsq(X_lag, Y_dep, rcond=None)[0]
    resid_init = Y_dep - X_lag @ beta_init
    score = (resid_init ** 2).sum(axis=1)
    qs = np.quantile(score, np.linspace(0, 1, K + 1)[1:-1])
    sm_init = np.zeros((T_eff, K))
    bins = np.digitize(score, qs)
    for t in range(T_eff):
        sm_init[t, bins[t]] = 1.0
    # Add a small jitter so EM can move
    sm_init = sm_init + 0.05 * rng.uniform(size=sm_init.shape)
    sm_init = sm_init / sm_init.sum(axis=1, keepdims=True)

    smoothed = sm_init
    A = np.zeros((n, n * p))  # shared
    mu = np.zeros((K, n))
    Sigma = np.zeros((K, n, n))
    P = np.full((K, K), 1.0 / K)
    pi0 = np.full(K, 1.0 / K)
    ll_prev = -np.inf
    converged = False

    for it in range(n_iter):
        # ----- M-step -----
        # 1) Shared A: weighted OLS pooling across regimes.
        #    For each row of A we minimise sum_t sum_k smoothed[t,k] * residual^2.
        w_total = smoothed.sum(axis=1)  # all 1s
        # Demean Y per regime (subtract regime-conditional mu).
        mu_new = np.zeros((K, n))
        for k in range(K):
            wk = smoothed[:, k]
            wk_sum = wk.sum()
            if wk_sum < 1e-8:
                mu_new[k] = Y_dep.mean(axis=0)
                continue
            # mu_k = weighted_mean(Y_dep - X_lag @ A) under current A.
            mu_new[k] = (wk[:, None] * (Y_dep - X_lag @ A.T)).sum(axis=0) / wk_sum
        mu = mu_new

        # Demean by regime-weighted mean
        Y_demeaned = Y_dep - smoothed @ mu  # weighted
        # Update A: OLS of Y_demeaned on X_lag (no intercept needed; regime mean absorbed).
        try:
            A_new = np.linalg.lstsq(X_lag, Y_demeaned, rcond=None)[0]
            A = A_new.T  # shape (n, n*p)
        except np.linalg.LinAlgError:
            pass

        # 2) Sigma_k: regime-conditional weighted residual covariance.
        for k in range(K):
            wk = smoothed[:, k]
            wk_sum = wk.sum()
            if wk_sum < 1e-8:
                Sigma[k] = np.eye(n)
                continue
            resid = Y_dep - mu[k] - X_lag @ A.T
            Sk = ((wk[:, None] * resid).T @ resid) / wk_sum
            Sigma[k] = Sk + 1e-8 * np.eye(n)  # ridge

        # 3) Transition matrix from pairwise smoothed probabilities.
        # Computed in the E-step below; here just keep prior P.

        # ----- E-step -----
        densities = np.empty((T_eff, K))
        for t in range(T_eff):
            for k in range(K):
                mean_kt = mu[k] + A @ X_lag[t]
                densities[t, k] = max(_gauss_density(Y_dep[t], mean_kt, Sigma[k]),
                                       1e-300)
        filt, pred, ll = _hamilton_filter(densities, P, pi0)
        smoothed, pairwise = _kim_smoother(filt, P)

        # Update transition matrix using pairwise smoothed probs.
        if pairwise.shape[0] > 0:
            P_new = pairwise.sum(axis=0)
            row_sums = P_new.sum(axis=1, keepdims=True)
            P = np.where(row_sums > 0, P_new / np.where(row_sums == 0, 1, row_sums), P)
        pi0 = smoothed[0]

        if verbose:
            print(f"  iter {it+1:>3d}: loglik = {ll:.3f}")

        if abs(ll - ll_prev) < tol:
            converged = True
            ll_prev = ll
            break
        ll_prev = ll

    return MSVARResult(
        K=K, A=A, mu=mu, Sigma=Sigma, P=P,
        smoothed_probs=smoothed, filtered_probs=filt,
        loglik=float(ll_prev), n_iter=it + 1, converged=converged,
    )


__all__ = ["ms_var_fit", "MSVARResult"]
