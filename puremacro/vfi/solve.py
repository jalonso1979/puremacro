"""Low-level & high-level VFI solvers."""
from __future__ import annotations

import time
from typing import Callable, Dict, Any, Tuple
import numpy as np

from puremacro import _backend as _bk
from puremacro.vfi.kernels import bellman_step


def solve_vfi(
    R: np.ndarray,
    P: np.ndarray,
    beta: float,
    howard: bool = True,
    n_howard: int = 20,
    tol: float = 1e-8,
    max_iter: int = 10_000,
    *,
    xp=np,
) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
    """Low-level VFI solver operating on return tensor R (n_d, n_ap, n_a, n_z).

    Returns (V, flat_policy, n_ap, n_iter, sup_norm).
    """
    R_arr = xp.asarray(R)
    P_arr = xp.asarray(P)

    if R_arr.ndim == 3:
        n_ap, n_a, n_z = R_arr.shape
        R_arr = R_arr.reshape(1, n_ap, n_a, n_z)
    elif R_arr.ndim != 4:
        raise ValueError(f"R must be 3-D or 4-D; got shape {R_arr.shape}")

    n_d, n_ap, n_a, n_z = R_arr.shape

    V = xp.zeros((n_a, n_z), dtype=R_arr.dtype)
    flat_policy = xp.zeros((n_a, n_z), dtype=xp.int64)

    n_iter = 0
    sup_norm = 1.0

    while sup_norm > tol and n_iter < max_iter:
        n_iter += 1
        V_old = xp.array(V)

        if n_a == n_ap:
            EV = V_old @ P_arr.T
        else:
            EV = V_old[:n_ap, :] @ P_arr.T

        V_new, flat_idx = bellman_step(R_arr, EV, beta, xp=xp)

        if xp.any(xp.isneginf(V_new)) or xp.any(xp.isnan(V_new)):
            raise RuntimeError("no feasible action for state")

        V = V_new
        flat_policy = flat_idx

        if howard and n_howard > 0:
            d_idx = flat_idx // n_ap
            ap_idx = flat_idx % n_ap

            for _ in range(n_howard):
                if n_a == n_ap:
                    EV_h = V @ P_arr.T
                else:
                    EV_h = V[:n_ap, :] @ P_arr.T

                V_next = xp.zeros_like(V)
                for z_i in range(n_z):
                    for a_i in range(n_a):
                        idd = d_idx[a_i, z_i]
                        iap = ap_idx[a_i, z_i]
                        V_next[a_i, z_i] = R_arr[idd, iap, a_i, z_i] + beta * EV_h[iap, z_i]
                V = V_next

        sup_norm = float(xp.max(xp.abs(V - V_old)))

    if sup_norm > tol:
        raise RuntimeError(f"VFI did not converge in {max_iter} iterations (sup_norm={sup_norm:.3e} > tol={tol:.3e})")

    return V, flat_policy, n_ap, n_iter, sup_norm


def solve_vfi_howard(
    return_fn: Callable[[float, float, float], float],
    a_grid: np.ndarray,
    z_grid: np.ndarray,
    P: np.ndarray,
    beta: float,
    max_iter: int = 2000,
    tol: float = 1e-7,
    howard_steps: int = 20,
) -> Dict[str, Any]:
    """High-level VFI solver with 3D tensor vectorization and Howard acceleration."""
    a_grid = np.asarray(a_grid, dtype=float)
    z_grid = np.asarray(z_grid, dtype=float)
    Na = len(a_grid)
    Nz = len(z_grid)

    t_start = time.time()

    R_mat = np.zeros((Na, Na, Nz))
    for k in range(Nz):
        z_val = z_grid[k]
        for i in range(Na):
            a_val = a_grid[i]
            for j in range(Na):
                ap_val = a_grid[j]
                u = return_fn(a_val, ap_val, z_val)
                R_mat[i, j, k] = -1e10 if (u is None or np.isnan(u) or not np.isreal(u)) else float(u)

    V = np.zeros((Na, Nz))
    policy_idx = np.zeros((Na, Nz), dtype=int)
    iter_count = 0
    dist = 1.0

    while dist > tol and iter_count < max_iter:
        iter_count += 1
        V_old = V.copy()

        EV = V_old @ P.T

        W = np.zeros((Na, Na, Nz))
        for k in range(Nz):
            W[:, :, k] = R_mat[:, :, k] + beta * EV[:, k]

        policy_idx = np.argmax(W, axis=1)
        V = np.max(W, axis=1)

        if howard_steps > 0:
            u_policy = np.zeros((Na, Nz))
            for k in range(Nz):
                for i in range(Na):
                    u_policy[i, k] = R_mat[i, policy_idx[i, k], k]

            for _ in range(howard_steps):
                EV_h = V @ P.T
                for k in range(Nz):
                    for i in range(Na):
                        V[i, k] = u_policy[i, k] + beta * EV_h[policy_idx[i, k], k]

        dist = float(np.max(np.abs(V - V_old)))

    policy_a = np.zeros((Na, Nz))
    for k in range(Nz):
        policy_a[:, k] = a_grid[policy_idx[:, k]]

    elapsed = time.time() - t_start

    return {
        "V": V,
        "policy_idx": policy_idx,
        "policy_a": policy_a,
        "n_iter": iter_count,
        "elapsed": elapsed,
    }


def simulate_vfi_panel(
    policy_a: np.ndarray,
    a_grid: np.ndarray,
    z_grid: np.ndarray,
    P: np.ndarray,
    n_agents: int = 2000,
    T_periods: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Simulate panel of agents and compute Gini inequality."""
    rng = np.random.default_rng(seed)
    Na = len(a_grid)
    Nz = len(z_grid)

    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    p_stat = np.real(eigvecs[:, idx])
    p_stat = np.maximum(p_stat / np.sum(p_stat), 0)

    z_states = rng.choice(Nz, size=n_agents, p=p_stat)
    a_states = np.full(n_agents, Na // 2, dtype=int)

    asset_panel = np.zeros((n_agents, T_periods))

    for t in range(T_periods):
        for i in range(n_agents):
            a_i = a_states[i]
            z_i = z_states[i]
            asset_panel[i, t] = a_grid[a_i]

            a_next = policy_a[a_i, z_i]
            a_states[i] = int(np.argmin(np.abs(a_grid - a_next)))
            z_states[i] = int(rng.choice(Nz, p=P[z_i, :]))

    final_assets = np.sort(asset_panel[:, -1])
    mean_assets = float(np.mean(final_assets))

    n_a = len(final_assets)
    cum_a = np.cumsum(final_assets - np.min(final_assets))
    gini = 1.0 - 2.0 * np.sum(cum_a) / (n_a * max(cum_a[-1], 1e-10)) + 1.0 / n_a

    return {
        "mean_assets": mean_assets,
        "gini": float(gini),
        "asset_panel": asset_panel,
    }
