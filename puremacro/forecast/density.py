"""Forecast-density evaluation + combination weights.

Companions to ``compare.py`` (DM / GW): tools for evaluating
*probabilistic* forecasts rather than point forecasts.

- ``pit``                : probability integral transform.
- ``pit_uniformity_test``: Kolmogorov-Smirnov uniformity test on PITs.
- ``crps_gaussian``      : CRPS of a Gaussian forecast (closed form).
- ``crps_ensemble``      : CRPS from an ensemble (Gneiting-Raftery 2007
                           "fair" estimator).
- ``klic_amisano_giacomini`` : DM-style test on log-likelihood scores.
- ``combine_forecasts``  : equal-weight / inverse-MSE / Bates-Granger
                           combination weights.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.stats import norm, kstest, t as t_dist


def pit(y: np.ndarray, cdf_fn: Callable[[float, int], float]) -> np.ndarray:
    """Compute u_t = F_t(y_t) for a sequence of predictive CDFs.

    ``cdf_fn(y_t, t)`` returns F_t(y_t) — the predictive CDF for
    observation t evaluated at y_t. Under correct calibration, the
    sequence {u_t} is iid Uniform(0, 1).
    """
    y = np.asarray(y, dtype=float).ravel()
    return np.array([float(cdf_fn(y[t], t)) for t in range(len(y))])


def pit_uniformity_test(pits: np.ndarray) -> dict:
    """Kolmogorov-Smirnov test of H_0: PIT ~ Uniform(0, 1)."""
    pits = np.asarray(pits, dtype=float).ravel()
    ks = kstest(pits, "uniform")
    return {"stat": float(ks.statistic), "p_value": float(ks.pvalue)}


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Continuous ranked probability score under a Gaussian forecast.

    CRPS_t = sigma_t * [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ]
    where z = (y_t - mu_t) / sigma_t. Returns the per-t array; the mean
    is the usual scalar score.
    """
    y = np.asarray(y, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    z = (y - mu) / np.maximum(sigma, 1e-12)
    return sigma * (z * (2 * norm.cdf(z) - 1.0)
                    + 2 * norm.pdf(z)
                    - 1.0 / np.sqrt(np.pi))


def crps_ensemble(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Fair CRPS from an M-member ensemble (Gneiting-Raftery 2007 eq. 27).

    samples : (T, M) — M draws from the predictive at each t.
    Returns per-t CRPS.
    """
    y = np.asarray(y, dtype=float).ravel()
    s = np.asarray(samples, dtype=float)
    T, M = s.shape
    term1 = np.mean(np.abs(s - y[:, None]), axis=1)
    term2 = np.zeros(T)
    for t in range(T):
        st = s[t]
        diffs = np.abs(st[:, None] - st[None, :])
        term2[t] = diffs.sum() / (M * (M - 1)) if M > 1 else 0.0
    return term1 - 0.5 * term2


def klic_amisano_giacomini(
    log_lik_1: np.ndarray,
    log_lik_2: np.ndarray,
    h: int = 1,
) -> dict:
    """Amisano-Giacomini (2007) weighted likelihood-ratio test.

    Tests equal predictive ability via DM-style asymptotic test on
    d_t = log f_1(y_t) - log f_2(y_t). Long-run variance via Bartlett
    kernel with bandwidth h - 1.
    """
    d = np.asarray(log_lik_1, dtype=float) - np.asarray(log_lik_2, dtype=float)
    T = len(d)
    d_bar = float(np.mean(d))
    g0 = float(np.sum((d - d_bar) ** 2) / T)
    s2 = g0
    L = max(0, h - 1)
    for ell in range(1, L + 1):
        if ell >= T:
            break
        cov = float(np.sum((d[ell:] - d_bar) * (d[:-ell] - d_bar)) / T)
        s2 += 2.0 * (1.0 - ell / (L + 1.0)) * cov
    if s2 <= 0:
        return {"stat": np.nan, "p_value": np.nan, "n_obs": int(T)}
    stat = d_bar / np.sqrt(s2 / T)
    p_value = 2.0 * t_dist.sf(abs(stat), df=T - 1)
    return {"stat": float(stat), "p_value": float(p_value), "n_obs": int(T)}


def combine_forecasts(
    forecasts: np.ndarray,
    method: str = "equal",
    errors: np.ndarray | None = None,
    rolling: int | None = None,
) -> dict:
    """Point-forecast combination.

    Parameters
    ----------
    forecasts : (T, K) array of K competing forecasts.
    method : {"equal", "inv_mse", "bates_granger"}.
        * equal: w_k = 1/K
        * inv_mse: w_k proportional to 1 / MSE_k (computed from `errors`)
        * bates_granger: w = (e' Sigma^{-1} 1) / (1' Sigma^{-1} 1) where
          Sigma is the sample covariance of the forecast errors.
    errors : (T, K) array of forecast errors, required for inv_mse and
        bates_granger.
    rolling : if int, compute rolling weights: the weights applied to
        ``forecasts[t]`` are built from ``errors[max(0, t - rolling):t]`` —
        the last ``rolling`` errors *strictly before* ``t`` — so the
        combination is out-of-sample at every ``t``. Rows with fewer than
        two past errors (``t < 2``) use equal weights. Else, weights are
        constant and computed on all of ``errors`` (in-sample).

    Returns
    -------
    dict with
        weights  : (K,) or (T, K) — the chosen weights
        combined : (T,) — the combined forecast
        method   : echo
    """
    forecasts = np.asarray(forecasts, dtype=float)
    if forecasts.ndim != 2:
        raise ValueError("forecasts must be (T, K)")
    T, K = forecasts.shape
    if rolling is not None and int(rolling) < 1:
        raise ValueError(f"rolling must be a positive int; got {rolling!r}")

    if method == "equal":
        w = np.full(K, 1.0 / K)
        return {"weights": w, "combined": forecasts @ w, "method": method}

    if errors is None:
        raise ValueError(f"method={method!r} requires `errors`")
    err = np.asarray(errors, dtype=float)
    if err.shape != (T, K):
        raise ValueError(f"errors must have the same shape as forecasts {(T, K)}; got {err.shape}")

    def _weights(e: np.ndarray) -> np.ndarray:
        if method == "inv_mse":
            mse = (e ** 2).mean(axis=0)
            inv = 1.0 / np.maximum(mse, 1e-12)
            return inv / inv.sum()
        if method == "bates_granger":
            S = (e.T @ e) / e.shape[0]
            try:
                S_inv = np.linalg.inv(S + 1e-10 * np.eye(K))
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)
            ones = np.ones(K)
            w = S_inv @ ones
            return w / (ones @ w)
        raise ValueError(f"unknown method {method!r}")

    if rolling is None:
        w = _weights(err)
        return {"weights": w, "combined": forecasts @ w, "method": method}

    weights = np.empty((T, K))
    win = int(rolling)
    for t in range(T):
        past = err[max(0, t - win):t]            # errors strictly before t
        weights[t] = _weights(past) if past.shape[0] >= 2 else np.full(K, 1.0 / K)
    combined = (forecasts * weights).sum(axis=1)
    return {"weights": weights, "combined": combined, "method": method}


def berkowitz_test(pits: np.ndarray) -> dict:
    """Berkowitz (2001) joint LR test on PIT-transformed scores.

    For PIT u_t under the null of correct calibration,
    z_t = Phi^{-1}(u_t) is iid N(0, 1). Berkowitz tests the joint
    null H_0: μ = 0, σ² = 1, ρ = 0 in the AR(1)

        z_t = μ + ρ z_{t-1} + σ ε_t,   ε_t ~ N(0, 1).

    The unrestricted log-likelihood is the OLS profile of (μ, ρ, σ²);
    the restricted log-likelihood plugs in (0, 0, 1). The LR statistic
    is 2 * (ll_unrestricted - ll_restricted) ~ χ²(3) under the null.

    More powerful than KS uniformity because it tests calibration *and*
    independence jointly.
    """
    pits = np.asarray(pits, dtype=float).ravel()
    # Map (0, 1) to the real line; clip to avoid +/- infinity at boundaries.
    eps = 1e-6
    u = np.clip(pits, eps, 1.0 - eps)
    z = norm.ppf(u)
    T = len(z)
    # Unrestricted AR(1) log-likelihood (profile out σ² via residual variance).
    z_lag = z[:-1]
    z_dep = z[1:]
    X = np.column_stack([np.ones(T - 1), z_lag])
    beta = np.linalg.lstsq(X, z_dep, rcond=None)[0]
    resid = z_dep - X @ beta
    sigma2_hat = float(np.var(resid, ddof=0))
    if sigma2_hat <= 0:
        sigma2_hat = 1e-12
    ll_unr = -0.5 * (T - 1) * (np.log(2 * np.pi) + np.log(sigma2_hat) + 1.0)
    # Restricted log-likelihood: μ=0, ρ=0, σ²=1 — i.e., z_t ~ N(0, 1).
    ll_r = -0.5 * np.sum(np.log(2 * np.pi) + z ** 2)
    # LR uses the same (T-1) contribution, so add the missing term to ll_r.
    ll_r_aligned = -0.5 * np.sum(np.log(2 * np.pi) + z[1:] ** 2)
    LR = 2.0 * (ll_unr - ll_r_aligned)
    from scipy.stats import chi2
    p_value = chi2.sf(LR, df=3) if LR >= 0 else np.nan
    return {
        "stat": float(LR),
        "p_value": float(p_value),
        "df": 3,
        "mu_hat": float(beta[0]),
        "rho_hat": float(beta[1]),
        "sigma2_hat": float(sigma2_hat),
    }


__all__ = [
    "pit",
    "pit_uniformity_test",
    "crps_gaussian",
    "crps_ensemble",
    "klic_amisano_giacomini",
    "combine_forecasts",
    "berkowitz_test",
]
