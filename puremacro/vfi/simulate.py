"""Monte Carlo agent panel simulation (VFIToolkit SimulateTimeSeries).

The joint state (a, z) is a Markov chain: a' is deterministic given (a, z) (the
optimal policy index), z' is stochastic (P_z). ``simulate_panel`` simulates
n_agents over n_periods, vectorised across agents, and returns integer index
panels. ``empirical_distribution`` bins a panel into an (n_a, n_z) measure --
which converges to ``distribution.stationary_distribution`` as the panel grows
(an independent cross-check of the solve -> policy -> distribution chain).
"""
from __future__ import annotations

import numpy as np


def _z_stationary(P_z):
    P = np.asarray(P_z, dtype=float)
    w, V = np.linalg.eig(P.T)
    v = np.real(V[:, np.argmin(np.abs(w - 1.0))])
    return v / v.sum()


def simulate_panel(policy_aprime, P_z, *, n_agents: int, n_periods: int,
                   seed: int = 0, a0_index: int = 0, z0_index=None,
                   burn_in: int = 0):
    """Simulate (a, z) index paths for ``n_agents`` over ``n_periods``.

    a_{t+1} = policy_aprime[a_t, z_t] (deterministic); z_{t+1} ~ P_z[z_t]
    (inverse-CDF). All agents start at asset index ``a0_index`` and shock index
    ``z0_index`` (default: drawn from the z-stationary distribution). ``burn_in``
    periods are simulated and discarded before recording. Returns
    (a_path, z_path), each (n_agents, n_periods) int arrays of grid indices.
    """
    pol = np.asarray(policy_aprime)
    n_a, n_z = pol.shape
    P = np.asarray(P_z, dtype=float)
    if int(burn_in) < 0:
        raise ValueError(f"burn_in must be >= 0; got {burn_in}")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("P_z rows must sum to 1")
    cdf = np.cumsum(P, axis=1)
    rng = np.random.default_rng(seed)

    a_idx = np.full(n_agents, int(a0_index), dtype=np.intp)
    if z0_index is None:
        z_idx = rng.choice(n_z, size=n_agents, p=_z_stationary(P)).astype(np.intp)
    else:
        z_idx = np.full(n_agents, int(z0_index), dtype=np.intp)

    a_path = np.empty((n_agents, n_periods), dtype=np.intp)
    z_path = np.empty((n_agents, n_periods), dtype=np.intp)
    total = int(burn_in) + int(n_periods)
    for t in range(total):
        if t >= burn_in:
            a_path[:, t - burn_in] = a_idx
            z_path[:, t - burn_in] = z_idx
        a_next = pol[a_idx, z_idx]
        u = rng.random(n_agents)
        # inverse-CDF sample of z' from row z_idx: first column where u < cdf
        z_next = (u[:, None] < cdf[z_idx]).argmax(axis=1).astype(np.intp)
        a_idx, z_idx = a_next.astype(np.intp), z_next
    return a_path, z_path


def empirical_distribution(a_path, z_path, n_a: int, n_z: int):
    """Empirical (n_a, n_z) distribution from index panels (sums to 1)."""
    a = np.asarray(a_path)
    z = np.asarray(z_path)
    if a.shape != z.shape:
        raise ValueError(f"a_path and z_path must have the same shape; got {a.shape} vs {z.shape}")
    flat = a.reshape(-1) * n_z + z.reshape(-1)
    counts = np.bincount(flat, minlength=n_a * n_z).reshape(n_a, n_z)
    if counts.sum() == 0:
        raise ValueError("empty panel: no samples to form a distribution")
    return counts / counts.sum()


__all__ = ["simulate_panel", "empirical_distribution"]
