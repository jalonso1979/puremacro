"""Shapley decomposition of aggregate volatility.

Aggregate growth: ``g_t = sum_i w_i g_{i,t}``. Aggregate variance:
``V = w' Sigma w``. The Shapley value assigns each component ``i`` a
non-negative share ``phi_i`` such that ``sum_i phi_i = V`` (efficiency),
based on its average marginal contribution across all permutations of
the components.

For a subset ``S`` of components, the *characteristic function* is

.. math::

    v(S) = \\sum_{i,j \\in S} w_i w_j \\Sigma_{ij}.

The Shapley value of component ``i`` is

.. math::

    \\phi_i = \\sum_{S \\subseteq N \\setminus \\{i\\}}
             \\frac{|S|!\\,(n-|S|-1)!}{n!}\\,
             \\bigl[v(S \\cup \\{i\\}) - v(S)\\bigr].

For ``n <= 14`` we compute it exactly via subset enumeration; for larger
``n`` we use Monte Carlo permutation sampling (Castro, Gomez & Tejada
2009). A 95% confidence interval for the MC estimate is returned with
``return_se=True``.

Public API
----------
``shapley_var(sigma, R, w, *, n_perm=None, seed=0, return_se=False)``
"""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np


def _build_sigma(sigma: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Build covariance matrix Sigma = diag(sigma) R diag(sigma)."""
    return np.diag(sigma) @ R @ np.diag(sigma)


def _v_subset(Sigma: np.ndarray, w: np.ndarray, S_mask: np.ndarray) -> float:
    """Characteristic function: w[S]' Sigma[S,S] w[S]."""
    if not S_mask.any():
        return 0.0
    idx = np.where(S_mask)[0]
    Ws = w[idx]
    return float(Ws @ Sigma[np.ix_(idx, idx)] @ Ws)


def _shapley_exact(Sigma: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = len(w)
    phi = np.zeros(n)
    n_fact = factorial(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for k in range(n):
            for S in combinations(others, k):
                S_mask = np.zeros(n, dtype=bool)
                S_mask[list(S)] = True
                v_S = _v_subset(Sigma, w, S_mask)
                S_mask[i] = True
                v_Si = _v_subset(Sigma, w, S_mask)
                weight = factorial(k) * factorial(n - k - 1) / n_fact
                phi[i] += weight * (v_Si - v_S)
    return phi


def _shapley_mc(
    Sigma: np.ndarray, w: np.ndarray, n_perm: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo Shapley: sample ``n_perm`` permutations and average
    incremental contributions; return (mean, standard error)."""
    n = len(w)
    accum = np.zeros((n_perm, n))
    for p in range(n_perm):
        order = rng.permutation(n)
        S_mask = np.zeros(n, dtype=bool)
        v_prev = 0.0
        for idx in order:
            S_mask[idx] = True
            v_now = _v_subset(Sigma, w, S_mask)
            accum[p, idx] = v_now - v_prev
            v_prev = v_now
    mean = accum.mean(axis=0)
    se = accum.std(axis=0, ddof=1) / np.sqrt(n_perm)
    return mean, se


def shapley_var(
    sigma: np.ndarray,
    R: np.ndarray,
    w: np.ndarray,
    *,
    n_perm: int | None = None,
    seed: int = 0,
    return_se: bool = False,
):
    """Shapley decomposition of ``V = w' Sigma w`` across components.

    Parameters
    ----------
    sigma : (n,) array of per-component standard deviations.
    R     : (n, n) correlation matrix.
    w     : (n,) weights (typically share of aggregate; can be unnormalized).
    n_perm: if None, decide automatically (exact for n <= 14, else 2000 MC).
    seed  : seed for the Monte-Carlo permutation sampler.
    return_se: if True, also return an estimate of the SE per component.

    Returns
    -------
    phi : (n,) array of Shapley values that sum to ``w' Sigma w``.
    se  : (n,) array of standard errors (only when return_se=True).

    Notes
    -----
    Efficiency holds exactly for the exact path; for the MC path it holds
    in expectation, with sampling error in ``se``.
    """
    sigma = np.asarray(sigma, dtype=float).ravel()
    R = np.asarray(R, dtype=float)
    w = np.asarray(w, dtype=float).ravel()
    n = len(w)
    if R.shape != (n, n):
        raise ValueError(f"R must be ({n},{n}); got {R.shape}")
    if len(sigma) != n:
        raise ValueError(f"sigma must have length {n}; got {len(sigma)}")

    Sigma = _build_sigma(sigma, R)

    if n_perm is None:
        if n <= 14:
            phi = _shapley_exact(Sigma, w)
            if return_se:
                return phi, np.zeros(n)
            return phi
        n_perm = 2000

    rng = np.random.default_rng(seed)
    phi, se = _shapley_mc(Sigma, w, n_perm=n_perm, rng=rng)
    if return_se:
        return phi, se
    return phi


def shapley_table(
    sigma: np.ndarray,
    R: np.ndarray,
    w: np.ndarray,
    labels: list[str] | None = None,
    *,
    n_perm: int | None = None,
    seed: int = 0,
):
    """Convenience wrapper that returns a tidy DataFrame.

    Columns: ``component``, ``w``, ``sigma``, ``phi``, ``share``, ``se``.
    ``share`` = ``phi / sum(phi)``.
    """
    import pandas as pd

    n = len(w)
    if labels is None:
        labels = [f"c{i}" for i in range(n)]
    phi, se = shapley_var(sigma, R, w, n_perm=n_perm, seed=seed, return_se=True)
    total = float(phi.sum()) if abs(phi.sum()) > 1e-15 else 1.0
    return pd.DataFrame({
        "component": labels,
        "w": np.asarray(w, dtype=float),
        "sigma": np.asarray(sigma, dtype=float),
        "phi": phi,
        "share": phi / total,
        "se": se,
    })


def _V(sigma: np.ndarray, R: np.ndarray, w: np.ndarray) -> float:
    Sigma = np.diag(sigma) @ R @ np.diag(sigma)
    return float(w @ Sigma @ w)


def shapley_factors(pre: dict, post: dict) -> dict:
    """Three-factor Shapley decomposition of ``Delta V = V_post - V_pre``.

    Allocates the change in aggregate variance to changes in three
    primitives: weights (``w``), per-component standard deviations
    (``sigma``), and the correlation matrix (``R``).

    For 3 factors there are :math:`3! = 6` orderings of the change. For
    each, the marginal contribution of factor :math:`F` is the difference
    in :math:`V` after updating that factor (with the others held at the
    current state in the ordering). Shapley averages contributions across
    all 6 orderings.

    Returns a dict with: ``V_pre``, ``V_post``, ``deltaV``,
    ``phi_w``, ``phi_sigma``, ``phi_R``, plus normalized shares.
    """
    from itertools import permutations
    keys = ("w", "sigma", "R")
    def get(state, k):
        return np.asarray(state[k], dtype=float)

    contribs = {k: 0.0 for k in keys}
    for order in permutations(keys):
        cur = {k: get(pre, k).copy() for k in keys}
        for k in order:
            v_before = _V(cur["sigma"], cur["R"], cur["w"])
            cur[k] = get(post, k).copy()
            v_after = _V(cur["sigma"], cur["R"], cur["w"])
            contribs[k] += (v_after - v_before)
    for k in keys:
        contribs[k] /= 6

    V_pre = _V(get(pre, "sigma"), get(pre, "R"), get(pre, "w"))
    V_post = _V(get(post, "sigma"), get(post, "R"), get(post, "w"))
    deltaV = V_post - V_pre
    denom = deltaV if abs(deltaV) > 1e-15 else 1.0
    return {
        "V_pre": V_pre, "V_post": V_post, "deltaV": deltaV,
        "phi_w": contribs["w"],
        "phi_sigma": contribs["sigma"],
        "phi_R": contribs["R"],
        "share_w": contribs["w"] / denom,
        "share_sigma": contribs["sigma"] / denom,
        "share_R": contribs["R"] / denom,
    }


__all__ = ["shapley_var", "shapley_table", "shapley_factors"]
