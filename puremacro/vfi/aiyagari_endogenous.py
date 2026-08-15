"""Aiyagari (1994) General Equilibrium Model with Endogenous Labor Supply."""
from __future__ import annotations

import time
from typing import Dict, Any
import numpy as np
from .discretize import tauchen
from .solve import simulate_vfi_panel


def solve_aiyagari_endogenous(
    beta: float = 0.95,
    gamma: float = 2.0,
    frisch: float = 0.5,
    chi: float = 1.2,
    alpha: float = 0.36,
    delta: float = 0.08,
    rho_e: float = 0.90,
    sigma_e: float = 0.20,
    Na: int = 45,
    Ne: int = 5,
    r_guess: float = 0.035,
) -> Dict[str, Any]:
    """Solve Aiyagari General Equilibrium Model with joint consumption, asset saving & labor supply n(a, e)."""
    t_start = time.time()

    log_e_grid, P_e = tauchen(Ne, rho_e, sigma_e)
    e_grid = np.exp(log_e_grid)
    a_grid = np.linspace(0.0, 20.0, Na)

    r_curr = r_guess
    kl_ratio = ((r_curr + delta) / alpha) ** (1.0 / (alpha - 1.0))
    w_curr = (1.0 - alpha) * (kl_ratio ** alpha)

    u_mat = np.zeros((Na, Na, Ne))
    c_mat = np.zeros((Na, Na, Ne))
    n_mat = np.zeros((Na, Na, Ne))

    for k in range(Ne):
        e_val = e_grid[k]
        for i in range(Na):
            a_val = a_grid[i]
            for j in range(Na):
                ap_val = a_grid[j]
                c_approx = max((1.0 + r_curr) * a_val + w_curr * e_val * 0.5 - ap_val, 1e-4)
                n_opt = 0.5
                for _ in range(4):
                    n_opt = min(1.0, max(0.0, ((w_curr * e_val * (c_approx ** (-gamma))) / chi) ** frisch))
                    c_approx = max((1.0 + r_curr) * a_val + w_curr * e_val * n_opt - ap_val, 1e-4)

                if (1.0 + r_curr) * a_val + w_curr * e_val * n_opt - ap_val <= 0:
                    u_mat[i, j, k] = -1e10
                else:
                    c_mat[i, j, k] = c_approx
                    n_mat[i, j, k] = n_opt
                    u_mat[i, j, k] = (c_approx ** (1.0 - gamma)) / (1.0 - gamma) - chi * (n_opt ** (1.0 + 1.0 / frisch)) / (1.0 + 1.0 / frisch)

    V = np.zeros((Na, Ne))
    dist = 1.0
    vfi_iter = 0
    policy_idx = np.zeros((Na, Ne), dtype=int)

    while dist > 1e-6 and vfi_iter < 1000:
        vfi_iter += 1
        V_old = V.copy()
        EV = V_old @ P_e.T

        W = np.zeros((Na, Na, Ne))
        for k in range(Ne):
            W[:, :, k] = u_mat[:, :, k] + beta * EV[:, k]

        policy_idx = np.argmax(W, axis=1)
        V = np.max(W, axis=1)

        for _ in range(20):
            EV_h = V @ P_e.T
            for k in range(Ne):
                for i in range(Na):
                    ap_i = policy_idx[i, k]
                    V[i, k] = u_mat[i, ap_i, k] + beta * EV_h[ap_i, k]

        dist = float(np.max(np.abs(V - V_old)))

    policy_a = np.zeros((Na, Ne))
    policy_c = np.zeros((Na, Ne))
    policy_n = np.zeros((Na, Ne))
    for k in range(Ne):
        for i in range(Na):
            j_opt = policy_idx[i, k]
            policy_a[i, k] = a_grid[j_opt]
            policy_c[i, k] = c_mat[i, j_opt, k]
            policy_n[i, k] = n_mat[i, j_opt, k]

    sim_res = simulate_vfi_panel(policy_a, a_grid, e_grid, P_e, n_agents=2000, T_periods=100)
    K_star = sim_res["mean_assets"]
    L_star = float(np.mean(policy_n) * np.mean(e_grid))

    elapsed = time.time() - t_start

    return {
        "r_star": r_curr,
        "w_star": w_curr,
        "K_star": K_star,
        "L_star": L_star,
        "V": V,
        "policy_a": policy_a,
        "policy_c": policy_c,
        "policy_n": policy_n,
        "a_grid": a_grid,
        "e_grid": e_grid,
        "P_e": P_e,
        "sim_res": sim_res,
        "elapsed": elapsed,
    }
