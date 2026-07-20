"""MCMC convergence diagnostics for puremacro Gibbs samplers
(``minnesota_gibbs``, ``ms_var_fit``, ``tvp_var_fit``, ``tvp_var_sv_fit``).

Currently exposes:
    - ``geweke_z``                  : Geweke (1992) z-score on
                                      first-vs-last segment of one chain.
    - ``gelman_rubin``              : R̂ across multiple chains.
    - ``effective_sample_size``     : autocorrelation-adjusted ESS.
    - ``autocorrelations``          : full ACF for diagnostic plotting.
    - ``trace_summary``             : summary statistics across chains.
"""
from __future__ import annotations

import numpy as np


def _spectral_density_zero(x: np.ndarray, max_lag: int | None = None) -> float:
    """Bartlett-kernel estimate of the spectral density at frequency 0.

    Used as a long-run variance for Geweke's z-test, where simple sample
    variance over-estimates the precision of an autocorrelated chain.
    """
    n = len(x)
    if max_lag is None:
        max_lag = int(4 * (n / 100) ** 0.2)
    x_dem = x - x.mean()
    g0 = float((x_dem ** 2).mean())
    s = g0
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        cov = float((x_dem[lag:] * x_dem[:-lag]).mean())
        s += 2.0 * (1.0 - lag / (max_lag + 1.0)) * cov
    return max(s, 1e-12)


def geweke_z(chain: np.ndarray, first: float = 0.1, last: float = 0.5) -> float:
    """Geweke (1992) z-score for chain convergence.

    Compares the mean of the first ``first`` fraction with the mean of
    the last ``last`` fraction (default: first 10% vs last 50%). Under
    the null of stationarity, the z-score is N(0, 1). |z| > 1.96 ⇒
    suspect non-stationarity.
    """
    chain = np.asarray(chain, dtype=float).ravel()
    n = len(chain)
    n_first = max(2, int(first * n))
    n_last = max(2, int(last * n))
    first_seg = chain[:n_first]
    last_seg = chain[-n_last:]
    s_first = _spectral_density_zero(first_seg)
    s_last = _spectral_density_zero(last_seg)
    diff = float(first_seg.mean() - last_seg.mean())
    se = float(np.sqrt(s_first / n_first + s_last / n_last))
    return diff / se if se > 0 else float("nan")


def gelman_rubin(chains: np.ndarray) -> dict:
    """Gelman-Rubin R̂ statistic.

    chains : (M, n) ndarray of M chains, each of length n. R̂ near 1
    indicates between-chain variance has shrunk to within-chain variance
    — i.e., all chains agree on the marginal distribution.
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2:
        raise ValueError("chains must be 2-D (M, n).")
    M, n = chains.shape
    chain_means = chains.mean(axis=1)
    grand_mean = chain_means.mean()
    B = n * ((chain_means - grand_mean) ** 2).sum() / max(M - 1, 1)
    W = ((chains - chain_means[:, None]) ** 2).sum() / (M * max(n - 1, 1))
    var_plus = ((n - 1) / n) * W + B / n
    R_hat = float(np.sqrt(var_plus / W)) if W > 0 else float("nan")
    return {"R_hat": R_hat, "W": float(W), "B": float(B), "var_plus": float(var_plus)}


def autocorrelations(chain: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """Sample ACF up to ``max_lag``."""
    chain = np.asarray(chain, dtype=float).ravel()
    x = chain - chain.mean()
    var = float((x ** 2).mean())
    if var == 0:
        return np.zeros(max_lag + 1)
    rho = np.zeros(max_lag + 1)
    rho[0] = 1.0
    for lag in range(1, max_lag + 1):
        if lag >= len(x):
            break
        rho[lag] = float((x[lag:] * x[:-lag]).mean()) / var
    return rho


def effective_sample_size(chain: np.ndarray, max_lag: int | None = None) -> float:
    """Effective sample size via Geyer's initial monotone sequence.

    ESS = n / (1 + 2 * sum_{k=1}^{m} rho_k)

    where m is chosen by the initial-monotone-sequence rule (Geyer 1992):
    sum sequential pairs of even/odd autocorrelations and stop when the
    pair sum becomes negative or non-decreasing.
    """
    chain = np.asarray(chain, dtype=float).ravel()
    n = len(chain)
    if max_lag is None:
        max_lag = min(n - 1, 1000)
    rho = autocorrelations(chain, max_lag=max_lag)
    pair_sums = []
    for k in range(1, len(rho) - 1, 2):
        s = rho[k] + rho[k + 1]
        if s < 0:
            break
        pair_sums.append(s)
    tau = 1.0 + 2.0 * float(sum(pair_sums))
    return float(n / tau) if tau > 0 else float(n)


def trace_summary(chains: np.ndarray) -> dict:
    """Summary stats across an (M, n) bundle of chains."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2:
        chains = chains[None, :]
    M, n = chains.shape
    out = {
        "n_chains": M,
        "n_iter": n,
        "mean": float(chains.mean()),
        "std": float(chains.std(ddof=1)),
        "geweke_z": [float(geweke_z(c)) for c in chains],
        "ess_per_chain": [float(effective_sample_size(c)) for c in chains],
    }
    if M >= 2:
        out["R_hat"] = gelman_rubin(chains)["R_hat"]
    return out


def random_walk_metropolis(
    log_posterior_fn,
    init,
    proposal_cov,
    n_draws: int,
    *,
    seed: int = 0,
    accept_target: float = 0.25,
    adapt_burnin: int = 0,
) -> dict:
    """Random-Walk Metropolis-Hastings with optional scalar-c proposal adaptation.

    Parameters
    ----------
    log_posterior_fn : callable(np.ndarray) -> float
        Target log-density (may return -np.inf).
    init : np.ndarray, shape (n_params,)
        Initial parameter vector.
    proposal_cov : np.ndarray, shape (n_params, n_params)
        Base proposal covariance. Actual proposal at iteration t is
        N(0, c**2 * proposal_cov) where c is adapted during adapt_burnin
        (no covariance adaptation; only scalar c).
    n_draws : int
        Post-burn-in iterations to retain.
    seed : int
        RNG seed.
    accept_target : float, default 0.25
        Adaptation target acceptance rate (only used if adapt_burnin > 0).
    adapt_burnin : int, default 0
        Number of additional iterations BEFORE n_draws used for scalar-c
        adaptation. Every 100 iterations, scale c by 1.1 if recent
        accept-rate > accept_target * 1.2, divide by 1.1 if < accept_target * 0.8.
        c is frozen at the end of burn-in.

    Returns
    -------
    dict with keys:
        chain       : (n_draws, n_params) — post-burn-in samples
        log_post    : (n_draws,)          — log-density at each retained sample
        accept_rate : float               — over the n_draws iterations only
        final_scale : float               — scalar c after adaptation (1.0 if adapt_burnin=0)
    """
    init = np.asarray(init, dtype=float).ravel()
    n_params = len(init)
    cov = np.asarray(proposal_cov, dtype=float)
    if cov.shape != (n_params, n_params):
        raise ValueError(
            f"proposal_cov shape {cov.shape} doesn't match init length {n_params}"
        )
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov + 1e-10 * np.eye(n_params)
        L = np.linalg.cholesky(cov)

    rng = np.random.default_rng(seed)

    x = init.copy()
    lp = log_posterior_fn(x)
    c = 1.0
    if not np.isfinite(lp):
        raise RuntimeError(
            "log_posterior_fn(init) is not finite; pick a better starting point"
        )

    # Adaptation phase.
    adapt_window = 100
    adapt_accepts = 0
    adapt_count = 0
    for it in range(adapt_burnin):
        z = rng.standard_normal(n_params)
        x_new = x + c * (L @ z)
        lp_new = log_posterior_fn(x_new)
        log_alpha = lp_new - lp
        if np.log(rng.uniform()) < log_alpha:
            x = x_new
            lp = lp_new
            adapt_accepts += 1
        adapt_count += 1
        if (it + 1) % adapt_window == 0:
            rate = adapt_accepts / adapt_count
            if rate > accept_target * 1.2:
                c *= 1.1
            elif rate < accept_target * 0.8:
                c /= 1.1
            adapt_accepts = 0
            adapt_count = 0

    # Retained chain.
    chain = np.empty((n_draws, n_params))
    log_post = np.empty(n_draws)
    accepts = 0
    for it in range(n_draws):
        z = rng.standard_normal(n_params)
        x_new = x + c * (L @ z)
        lp_new = log_posterior_fn(x_new)
        log_alpha = lp_new - lp
        if np.log(rng.uniform()) < log_alpha:
            x = x_new
            lp = lp_new
            accepts += 1
        chain[it] = x
        log_post[it] = lp

    return {
        "chain": chain,
        "log_post": log_post,
        "accept_rate": accepts / n_draws,
        "final_scale": c,
    }


__all__ = [
    "geweke_z", "gelman_rubin",
    "autocorrelations", "effective_sample_size",
    "trace_summary",
    "random_walk_metropolis",
]
