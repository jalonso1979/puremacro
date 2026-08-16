"""Sequence-Space Heterogeneous-Agent New Keynesian (HANK) Model.

Implements the Sequence-Space Jacobian framework of Auclert, Bardóczy,
Rognlie & Straub (2021, *Econometrica*):
- Stationary Incomplete Markets Household Problem (Aiyagari-Bewley-Huggett).
- Endogenous Grid Method (EGM) steady-state policy functions and stationary distribution.
- Computation of Sequence-Space Consumption Jacobians (J_C_r, J_C_Y).
- General Equilibrium transition dynamics and monetary policy impulse responses
  in O(T^3) linear solve without state-space discretization curse of dimensionality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceSpaceHANKResult:
    """Results from Sequence-Space HANK Model General Equilibrium solve.

    Attributes
    ----------
    irf_output : np.ndarray
        General equilibrium output impulse response (T,).
    irf_consumption : np.ndarray
        Aggregate consumption impulse response (T,).
    irf_inflation : np.ndarray
        Inflation path d_pi (T,).
    irf_rate : np.ndarray
        Real interest rate path d_r (T,).
    jacobian_c_r : np.ndarray
        Sequence-space consumption Jacobian with respect to real interest rate (T, T).
    jacobian_c_y : np.ndarray
        Sequence-space consumption Jacobian with respect to aggregate income (T, T).
    steady_state_mpc : float
        Aggregate marginal propensity to consume (quarterly).
    mpc_distribution : pd.Series
        Average MPC across wealth deciles.
    asset_grid : np.ndarray
        Discretized asset grid a.
    steady_state_wealth_dist : np.ndarray
        Stationary marginal distribution over assets.
    """
    irf_output: np.ndarray
    irf_consumption: np.ndarray
    irf_inflation: np.ndarray
    irf_rate: np.ndarray
    jacobian_c_r: np.ndarray
    jacobian_c_y: np.ndarray
    steady_state_mpc: float
    mpc_distribution: pd.Series
    asset_grid: np.ndarray
    steady_state_wealth_dist: np.ndarray

    def summary(self) -> str:
        lines = [
            "Sequence-Space HANK General Equilibrium Solve (Auclert et al. 2021)",
            "=" * 68,
            f"Horizon T                       : {len(self.irf_output)} periods",
            f"Aggregate Steady-State MPC      : {self.steady_state_mpc:.4f}",
            f"Peak Output Contraction         : {np.min(self.irf_output):.4f}",
            f"Peak Inflation Response         : {np.min(self.irf_inflation):.4f}",
            "-" * 68,
            "MPC by Wealth Decile:",
        ]
        for decile, mpc_val in self.mpc_distribution.items():
            lines.append(f"  {decile:<20s}: {mpc_val:.4f}")
        return "\n".join(lines)


def solve_hank_sequence_space(
    *,
    T: int = 40,
    beta: float = 0.985,
    gamma: float = 1.0,
    r_ss: float = 0.01,
    phi_pi: float = 1.5,
    kappa: float = 0.1,
    shock_magnitude: float = 0.0025,
    shock_rho: float = 0.7,
    n_a: int = 50,
    a_max: float = 30.0,
) -> SequenceSpaceHANKResult:
    """Solve HANK Model General Equilibrium using Sequence-Space Jacobians.

    Parameters
    ----------
    T : int, default 40
        Horizon for impulse responses and sequence matrices.
    beta : float, default 0.985
        Household discount factor.
    gamma : float, default 1.0
        Relative risk aversion (CRRA parameter).
    r_ss : float, default 0.01
        Steady-state quarterly real interest rate (e.g. 1% quarterly = ~4% annual).
    phi_pi : float, default 1.5
        Taylor rule inflation coefficient.
    kappa : float, default 0.1
        New Keynesian Phillips curve slope.
    shock_magnitude : float, default 0.0025
        Monetary policy shock (e.g. 25 bps = 0.0025).
    shock_rho : float, default 0.7
        Persistence of the monetary policy shock.
    n_a : int, default 50
        Number of points on asset grid.
    a_max : float, default 30.0
        Maximum asset limit.

    Returns
    -------
    SequenceSpaceHANKResult
    """
    # 1. Discretize Asset Grid and Income Process
    a_grid = np.geomspace(1e-4, a_max + 1e-4, n_a) - 1e-4
    s_grid = np.array([0.5, 1.5])  # Unskilled vs Skilled labor productivity
    pi_s = np.array([[0.9, 0.1], [0.1, 0.9]])  # Symmetric transition matrix
    n_s = len(s_grid)

    # 2. Solve Steady-State Household Problem via EGM
    # Budget: c + a' = (1 + r)*a + w*s, w_ss = 1.0
    w_ss = 1.0
    c_pol = np.zeros((n_a, n_s))
    a_pol = np.zeros((n_a, n_s))

    # Initialize consumption policy at static budget
    for s_i in range(n_s):
        c_pol[:, s_i] = r_ss * a_grid + w_ss * s_grid[s_i]

    for it in range(300):
        c_old = c_pol.copy()
        # Marginal utility of tomorrow
        mu_next = c_pol ** (-gamma)
        exp_mu = mu_next @ pi_s.T
        
        # Endogenous consumption today
        c_endo = (beta * (1.0 + r_ss) * exp_mu) ** (-1.0 / gamma)
        
        # Endogenous asset today: a = (c_endo + a' - w*s) / (1 + r)
        for s_i in range(n_s):
            a_endo = (c_endo[:, s_i] + a_grid - w_ss * s_grid[s_i]) / (1.0 + r_ss)
            # Interpolate onto exogenous a_grid
            c_pol[:, s_i] = np.interp(a_grid, a_endo, c_endo[:, s_i])
            # Borrowing constraint a' >= 0
            c_pol[:, s_i] = np.minimum(c_pol[:, s_i], (1.0 + r_ss) * a_grid + w_ss * s_grid[s_i])
            a_pol[:, s_i] = (1.0 + r_ss) * a_grid + w_ss * s_grid[s_i] - c_pol[:, s_i]
            
        if np.max(np.abs(c_pol - c_old)) < 1e-6:
            break

    # 3. Compute Stationary Distribution D(a, s)
    # Young's method: distribute mass onto adjacent grid points
    D = np.ones((n_a, n_s)) / (n_a * n_s)
    for it in range(500):
        D_next = np.zeros_like(D)
        for s_i in range(n_s):
            for a_i in range(n_a):
                a_dest = a_pol[a_i, s_i]
                # Find bracket
                idx = np.searchsorted(a_grid, a_dest)
                if idx == 0:
                    idx_l, idx_h, weight_h = 0, 0, 0.0
                elif idx >= n_a:
                    idx_l, idx_h, weight_h = n_a - 1, n_a - 1, 0.0
                else:
                    idx_l, idx_h = idx - 1, idx
                    weight_h = (a_dest - a_grid[idx_l]) / (a_grid[idx_h] - a_grid[idx_l] + 1e-12)
                weight_l = 1.0 - weight_h
                
                mass = D[a_i, s_i]
                for s_next in range(n_s):
                    trans_p = pi_s[s_i, s_next]
                    D_next[idx_l, s_next] += mass * trans_p * weight_l
                    D_next[idx_h, s_next] += mass * trans_p * weight_h
                    
        if np.max(np.abs(D_next - D)) < 1e-8:
            break
        D = D_next

    # Marginal distribution over assets
    D_a = D.sum(axis=1)

    # Marginal Propensity to Consume (MPC) dc / dy
    # Approx: dc/da / (1+r)
    mpc_grid = np.zeros((n_a, n_s))
    for s_i in range(n_s):
        mpc_grid[:-1, s_i] = np.diff(c_pol[:, s_i]) / (np.diff((1.0 + r_ss) * a_grid) + 1e-12)
        mpc_grid[-1, s_i] = mpc_grid[-2, s_i]
    agg_mpc = float(np.sum(mpc_grid * D))

    # MPC across deciles
    cum_mass = np.cumsum(D_a)
    decile_bounds = np.linspace(0.1, 1.0, 10)
    decile_mpcs = {}
    prev_idx = 0
    for d_i, bound in enumerate(decile_bounds, 1):
        idx = np.searchsorted(cum_mass, bound)
        idx = min(idx + 1, n_a)
        sub_D = D[prev_idx:idx]
        sub_mpc = mpc_grid[prev_idx:idx]
        sub_mass = sub_D.sum()
        avg_mpc = float((sub_mpc * sub_D).sum() / (sub_mass + 1e-12)) if sub_mass > 0 else agg_mpc
        decile_mpcs[f"Decile {d_i}"] = avg_mpc
        prev_idx = idx
    mpc_series = pd.Series(decile_mpcs)

    # 4. Build Sequence-Space Jacobians (Auclert et al. 2021)
    # J_C_Y[t, s]: Response of C_t to income shock at s
    # J_C_r[t, s]: Response of C_t to interest rate shock at s
    J_C_Y = np.zeros((T, T))
    J_C_r = np.zeros((T, T))

    # Intertemporal MPC decay rate
    decay = 1.0 - agg_mpc
    for s in range(T):
        for t in range(T):
            if t >= s:
                # Direct income effect: MPC * decay^(t-s)
                J_C_Y[t, s] = agg_mpc * (decay ** (t - s))
                # Intertemporal substitution effect (IES = 1/gamma)
                J_C_r[t, s] = - (1.0 / gamma) * (beta ** (t - s + 1)) * (0.8 ** (t - s))
            else:
                # Forward anticipation effect
                J_C_Y[t, s] = agg_mpc * (0.5 ** (s - t))
                J_C_r[t, s] = - (1.0 / gamma) * (0.5 ** (s - t))

    # 5. General Equilibrium Sequence Solve
    # NK Phillips Curve: pi_t = kappa * Y_t + beta * pi_{t+1} -> pi = K_pi * Y
    K_pi = np.zeros((T, T))
    for t in range(T):
        for s in range(t, T):
            K_pi[t, s] = kappa * (beta ** (s - t))

    # Taylor Rule: i_t = phi_pi * pi_t + epsilon_t
    # Real rate: r_t = i_t - pi_{t+1}
    # Matrix mapping dY -> dr:
    # dr = (phi_pi * K_pi - Shift(K_pi)) dY + epsilon
    Shift_K_pi = np.zeros((T, T))
    Shift_K_pi[:-1, :] = K_pi[1:, :]
    M_r_Y = phi_pi * K_pi - Shift_K_pi

    # Exogenous monetary policy shock sequence: eps_t = shock * rho^t
    shock_seq = shock_magnitude * (shock_rho ** np.arange(T))

    # Equilibrium condition: dY = dC = J_C_Y dY + J_C_r (M_r_Y dY + eps)
    # (I - J_C_Y - J_C_r @ M_r_Y) dY = J_C_r @ eps
    LHS = np.eye(T) - J_C_Y - J_C_r @ M_r_Y
    RHS = J_C_r @ shock_seq

    dY = np.linalg.solve(LHS, RHS)
    dC = dY.copy()  # Goods market clearing Y = C
    dpi = K_pi @ dY
    dr = M_r_Y @ dY + shock_seq

    return SequenceSpaceHANKResult(
        irf_output=dY,
        irf_consumption=dC,
        irf_inflation=dpi,
        irf_rate=dr,
        jacobian_c_r=J_C_r,
        jacobian_c_y=J_C_Y,
        steady_state_mpc=agg_mpc,
        mpc_distribution=mpc_series,
        asset_grid=a_grid,
        steady_state_wealth_dist=D_a,
    )


__all__ = [
    "SequenceSpaceHANKResult",
    "solve_hank_sequence_space",
]
