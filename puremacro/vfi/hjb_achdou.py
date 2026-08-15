"""Achdou, Han, Lasry, Lions & Moll (2022) Continuous-Time HJB Finite-Difference Solver."""
from __future__ import annotations

import time
from typing import Dict, Any
import numpy as np


def solve_hjb_achdou(
    r_rate: float = 0.03,
    w_rate: float = 1.0,
    rho_val: float = 0.05,
    gamma_r: float = 2.0,
    Na: int = 100,
    max_iter: int = 500,
    tol: float = 1e-7,
) -> Dict[str, Any]:
    """Continuous-time HJB Upwind Finite-Difference solver for consumption-saving models."""
    t_start = time.time()

    a_grid = np.linspace(0.0, 30.0, Na)
    da = a_grid[1] - a_grid[0]
    e_grid = np.array([0.2, 1.0])
    Ne = 2

    # Initial Value Function guess
    V = np.zeros((Na, Ne))
    for k in range(Ne):
        V[:, k] = ((r_rate * a_grid + w_rate * e_grid[k]) ** (1.0 - gamma_r)) / ((1.0 - gamma_r) * rho_val)

    dt = 0.02
    c_policy = np.zeros((Na, Ne))
    s_drift = np.zeros((Na, Ne))

    for iter_count in range(1, max_iter + 1):
        V_old = V.copy()

        V_forward = np.zeros((Na, Ne))
        V_backward = np.zeros((Na, Ne))

        for k in range(Ne):
            V_forward[:-1, k] = (V[1:, k] - V[:-1, k]) / da
            V_forward[-1, k] = (r_rate * a_grid[-1] + w_rate * e_grid[k]) ** (-gamma_r)

            V_backward[1:, k] = (V[1:, k] - V[:-1, k]) / da
            V_backward[0, k] = (r_rate * a_grid[0] + w_rate * e_grid[k]) ** (-gamma_r)

        c_forward = np.maximum(np.maximum(V_forward, 1e-8) ** (-1.0 / gamma_r), 1e-6)
        s_forward = r_rate * a_grid[:, None] + w_rate * e_grid[None, :] - c_forward

        c_backward = np.maximum(np.maximum(V_backward, 1e-8) ** (-1.0 / gamma_r), 1e-6)
        s_backward = r_rate * a_grid[:, None] + w_rate * e_grid[None, :] - c_backward

        c_zero = r_rate * a_grid[:, None] + w_rate * e_grid[None, :]
        V_zero = c_zero ** (-gamma_r)

        use_fwd = (s_forward > 0)
        use_bwd = (s_backward < 0) & (~use_fwd)
        use_zero = (~use_fwd) & (~use_bwd)

        c_policy = c_forward * use_fwd + c_backward * use_bwd + c_zero * use_zero
        s_drift = s_forward * use_fwd + s_backward * use_bwd

        u_val = (c_policy ** (1.0 - gamma_r)) / (1.0 - gamma_r)
        V = V_old + dt * (u_val + s_drift * (use_fwd * V_forward + use_bwd * V_backward + use_zero * V_zero) - rho_val * V_old)

        dist = float(np.max(np.abs(V - V_old)))
        if dist < tol:
            break

    elapsed = time.time() - t_start

    return {
        "V": V,
        "c_policy": c_policy,
        "s_drift": s_drift,
        "a_grid": a_grid,
        "e_grid": e_grid,
        "n_iter": iter_count,
        "elapsed": elapsed,
    }
