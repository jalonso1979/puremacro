"""Stationary agent distribution over a solved VFI policy (VFIToolkit StationaryDist_Case1).

Given the optimal next-state policy a'(a,z) and the exogenous Markov transition
P_z, the joint state (a,z) evolves under a known transition; this finds its
stationary measure mu(a,z) (unit mass, sums to 1). The workhorse is the Tan
(2020) two-step iteration: (1) push mass through the DETERMINISTIC a'-policy
(a scatter along the a-axis, z fixed), then (2) apply the z-transition
(mu @ P_z). This avoids materialising the full (n_a*n_z)^2 joint operator.
``joint_transition_matrix`` builds that explicit operator for small problems /
validation. numpy-only: it consumes VFISolution.policy_aprime (always numpy).
"""
from __future__ import annotations

import numpy as np


def push_distribution(mu, policy_aprime, P_z):
    """One forward step of the agent distribution (the Tan 2020 two-step).

    (1) push mass through the deterministic a'-policy (scatter along a, z fixed);
    (2) apply the z-transition (mu @ P_z). Mass-conserving. ``mu`` and the result
    are (n_a, n_z). This is the single-period operator that
    ``stationary_distribution`` iterates and that the transition path applies
    period by period.
    """
    mu = np.asarray(mu, dtype=float)
    pol = np.asarray(policy_aprime)
    P = np.asarray(P_z, dtype=float)
    n_a, n_z = pol.shape
    mu_half = np.zeros((n_a, n_z))
    z_idx = np.broadcast_to(np.arange(n_z), (n_a, n_z))
    np.add.at(mu_half, (pol.astype(np.intp), z_idx), mu)
    return mu_half @ P


def lottery_push(mu, aprime_values, a_grid, P_z):
    """One forward step for a CONTINUOUS next-asset policy (lottery/histogram).

    Each (a,z) sends mass mu[a,z] to a' = aprime_values[a,z], split between the
    bracketing grid nodes by linear weights (so the expected next asset equals
    a'); a' outside [a_grid[0], a_grid[-1]] goes entirely to the nearest endpoint.
    Then apply the z-transition (mu @ P_z). Mass-conserving. Returns (n_a, n_z).
    """
    mu = np.asarray(mu, dtype=float)
    ap = np.asarray(aprime_values, dtype=float)
    a = np.asarray(a_grid, dtype=float)
    P = np.asarray(P_z, dtype=float)
    n_a, n_z = mu.shape
    # bracketing lower index j in [0, n_a-2]
    j = np.clip(np.searchsorted(a, ap) - 1, 0, n_a - 2)
    a_lo = a[j]
    a_hi = a[j + 1]
    w_lo = np.clip((a_hi - ap) / (a_hi - a_lo), 0.0, 1.0)     # weight to node j
    z_idx = np.broadcast_to(np.arange(n_z), (n_a, n_z))
    mu_half = np.zeros((n_a, n_z))
    np.add.at(mu_half, (j, z_idx), mu * w_lo)
    np.add.at(mu_half, (j + 1, z_idx), mu * (1.0 - w_lo))
    return mu_half @ P


def lottery_distribution(aprime_values, a_grid, P_z, *, tol: float = 1e-12,
                         max_iter: int = 100_000, mu0=None):
    """Stationary distribution for a CONTINUOUS next-asset policy (lottery method).

    Iterates ``lottery_push`` to a sup-norm tolerance. ``aprime_values`` is
    (n_a, n_z) continuous next assets (e.g. ``solve_egm(...).aprime``); returns
    an (n_a, n_z) measure summing to 1.
    """
    ap = np.asarray(aprime_values, dtype=float)
    n_a, n_z = ap.shape
    if mu0 is None:
        mu = np.full((n_a, n_z), 1.0 / (n_a * n_z))
    else:
        mu = np.asarray(mu0, dtype=float).copy()
        mu = mu / mu.sum()
    for _ in range(max_iter):
        mu_new = lottery_push(mu, ap, a_grid, P_z)
        if np.max(np.abs(mu_new - mu)) < tol:
            return mu_new / mu_new.sum()
        mu = mu_new
    raise RuntimeError(
        f"lottery_distribution did not converge in {max_iter} iterations"
    )


def stationary_distribution(policy_aprime, P_z, *, tol: float = 1e-12,
                            max_iter: int = 100_000, mu0=None):
    """Stationary distribution mu(a,z) (n_a, n_z), summing to 1.

    policy_aprime: (n_a, n_z) int indices into the a-grid (a' chosen at each
    (a,z)). P_z: (n_z, n_z) row-stochastic. Iterates the Tan two-step to a
    sup-norm tolerance; raises if it does not converge.
    """
    pol = np.asarray(policy_aprime)
    P = np.asarray(P_z, dtype=float)
    n_a, n_z = pol.shape
    if P.shape != (n_z, n_z):
        raise ValueError(f"P_z must be ({n_z},{n_z}); got {P.shape}")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("P_z rows must sum to 1")
    if pol.size and (pol.min() < 0 or pol.max() >= n_a):
        raise ValueError("policy_aprime must hold valid a-grid indices in [0, n_a)")
    if mu0 is None:
        mu = np.full((n_a, n_z), 1.0 / (n_a * n_z))
    else:
        mu = np.asarray(mu0, dtype=float).copy()
        mu = mu / mu.sum()
    for _ in range(max_iter):
        mu_new = push_distribution(mu, pol, P)
        if np.max(np.abs(mu_new - mu)) < tol:
            return mu_new / mu_new.sum()
        mu = mu_new
    raise RuntimeError(
        f"stationary_distribution did not converge in {max_iter} iterations"
    )


def joint_transition_matrix(policy_aprime, P_z):
    """Explicit row-stochastic joint transition over flattened (a,z) [C-order a*n_z+z].

    Gamma[(a,z),(a',z')] = 1[a'==policy(a,z)] * P_z[z,z']. numpy-only,
    O((n_a*n_z)^2) memory -- for small problems and validating
    ``stationary_distribution`` (mu.reshape(-1) is its dominant left eigenvector).
    """
    pol = np.asarray(policy_aprime)
    P = np.asarray(P_z, dtype=float)
    n_a, n_z = pol.shape
    N = n_a * n_z
    G = np.zeros((N, N))
    for a in range(n_a):
        for z in range(n_z):
            ap = int(pol[a, z])
            i = a * n_z + z
            for zp in range(n_z):
                G[i, ap * n_z + zp] = P[z, zp]
    return G


__all__ = ["push_distribution", "stationary_distribution", "joint_transition_matrix",
           "lottery_push", "lottery_distribution"]
