"""Aiyagari (1994) General Equilibrium Model with Endogenous Labor Supply.

Households face idiosyncratic productivity risk, save in a risk-free asset and
choose hours intratemporally; a representative Cobb-Douglas firm hires both
factors. The equilibrium interest rate is the one at which the capital the
households want to hold equals the capital the firm wants to rent, with BOTH
sides evaluated at the same prices -- labor included, which here is an
equilibrium object rather than a parameter, since hours respond to the wage
that ``r`` implies.

Until 1.3.2 this module solved the household problem at ``r_guess`` and
returned that same number as ``r_star``: there was no market-clearing step at
all, so every "equilibrium" it reported was whatever rate the caller happened
to pass in, and ``L_star`` was a product of unweighted grid means rather than
an integral against the stationary distribution.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
from scipy.optimize import brentq

from .discretize import tauchen
from .distribution import stationary_distribution
from .solve import simulate_vfi_panel


def _factor_prices(r: float, alpha: float, delta: float) -> tuple[float, float]:
    """Capital-labor ratio and wage consistent with ``r`` under Cobb-Douglas."""
    kl = ((r + delta) / alpha) ** (1.0 / (alpha - 1.0))
    return kl, (1.0 - alpha) * (kl ** alpha)


def _period_payoffs(a_grid, e_grid, r, w, gamma, frisch, chi):
    """``u(a, a', e)`` with hours chosen intratemporally, plus ``c`` and ``n``.

    Hours solve ``chi n^(1/frisch) = w e c^(-gamma)`` given consumption, and
    consumption follows from the budget given hours; four sweeps of that pair
    converge to the intratemporal fixed point. Infeasible ``(a, a', e)`` cells
    are marked with a large negative payoff rather than dropped, so the value
    function's argmax never selects them.
    """
    a = a_grid[:, None, None]
    ap = a_grid[None, :, None]
    e = e_grid[None, None, :]

    unearned = (1.0 + r) * a - ap                      # cash before labor income
    c = np.maximum(unearned + w * e * 0.5, 1e-4)       # seed at half-time work
    n = np.full(np.broadcast(unearned, e).shape, 0.5)
    for _ in range(4):
        n = np.clip(((w * e * c ** (-gamma)) / chi) ** frisch, 0.0, 1.0)
        c = np.maximum(unearned + w * e * n, 1e-4)

    feasible = (unearned + w * e * n) > 0.0
    u = ((c ** (1.0 - gamma)) / (1.0 - gamma)
         - chi * (n ** (1.0 + 1.0 / frisch)) / (1.0 + 1.0 / frisch))
    zero = np.zeros_like(u)
    return (np.where(feasible, u, -1e10),
            np.where(feasible, c, zero),
            np.where(feasible, n, zero))


def _solve_household(u, P_e, beta, *, tol=1e-6, max_iter=1000, n_howard=20):
    """Value-function iteration with Howard improvement; returns ``(V, argmax)``."""
    Na, _, Ne = u.shape
    V = np.zeros((Na, Ne))
    policy_idx = np.zeros((Na, Ne), dtype=int)
    for _ in range(max_iter):
        V_old = V
        # EV[a', e] -- the continuation is indexed by the CHOICE, so it
        # broadcasts along axis 1 of the (a, a', e) payoff array.
        EV = V_old @ P_e.T
        W = u + beta * EV[None, :, :]
        policy_idx = np.argmax(W, axis=1)
        V = np.max(W, axis=1)
        u_pol = np.take_along_axis(u, policy_idx[:, None, :], axis=1)[:, 0, :]
        for _ in range(n_howard):
            EV_h = V @ P_e.T
            V = u_pol + beta * np.take_along_axis(EV_h, policy_idx, axis=0)
        if np.max(np.abs(V - V_old)) < tol:
            break
    return V, policy_idx


def _policies(policy_idx, a_grid, c_mat, n_mat):
    """Read the chosen ``a'``, ``c`` and ``n`` off the (a, a', e) grids."""
    take = policy_idx[:, None, :]
    return (a_grid[policy_idx],
            np.take_along_axis(c_mat, take, axis=1)[:, 0, :],
            np.take_along_axis(n_mat, take, axis=1)[:, 0, :])


def _aggregates(policy_idx, policy_n, a_grid, e_grid, P_e):
    """Stationary distribution and the two aggregates it implies.

    Effective labor is ``int e * n(a, e) dmu`` -- weighted by where households
    actually are. The mean of a policy array over the raw grid is a different
    quantity, and the mean of a product is not the product of means.
    """
    mu = stationary_distribution(policy_idx, P_e)
    K = float(np.sum(mu * a_grid[:, None]))
    L = float(np.sum(mu * policy_n * e_grid[None, :]))
    return mu, K, L


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
    a_max: float = 20.0,
    r_bracket: tuple[float, float] | None = None,
    xtol: float = 1e-6,
) -> Dict[str, Any]:
    """Solve the Aiyagari general equilibrium with endogenous labor supply.

    Returns the market-clearing ``r_star`` and the wage, aggregates, policy
    functions and stationary distribution that go with it.

    ``r_guess`` no longer pins the answer -- it only seeds the search bracket,
    which is otherwise ``(1e-4, 1/beta - 1 - 1e-4)``: above ``1/beta - 1`` a
    household's assets diverge, so no stationary distribution exists there.
    Pass ``r_bracket`` to override. Raises ``ValueError`` when excess demand
    does not change sign across the bracket, which usually means ``a_max`` is
    binding and the asset grid, not the economy, is setting the supply.
    """
    t_start = time.time()

    log_e_grid, P_e = tauchen(Ne, rho_e, sigma_e)
    e_grid = np.exp(log_e_grid)
    a_grid = np.linspace(0.0, a_max, Na)

    def solve_at(r: float):
        kl, w = _factor_prices(r, alpha, delta)
        u, c_mat, n_mat = _period_payoffs(a_grid, e_grid, r, w, gamma, frisch, chi)
        V, policy_idx = _solve_household(u, P_e, beta)
        policy_a, policy_c, policy_n = _policies(policy_idx, a_grid, c_mat, n_mat)
        mu, K, L = _aggregates(policy_idx, policy_n, a_grid, e_grid, P_e)
        return {
            "kl": kl, "w": w, "V": V, "policy_idx": policy_idx,
            "policy_a": policy_a, "policy_c": policy_c, "policy_n": policy_n,
            "mu": mu, "K": K, "L": L, "excess": K - kl * L,
        }

    n_evals = {"n": 0}

    def excess(r: float) -> float:
        n_evals["n"] += 1
        return solve_at(r)["excess"]

    lo, hi = r_bracket if r_bracket is not None else (1e-4, 1.0 / beta - 1.0 - 1e-4)
    f_lo, f_hi = excess(lo), excess(hi)
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"excess capital demand does not change sign on r in ({lo:.4f}, "
            f"{hi:.4f}): K - K_demand is {f_lo:+.4f} at the low end and "
            f"{f_hi:+.4f} at the high end. With both positive the asset grid "
            f"is binding (raise a_max above {a_max}); with both negative the "
            f"households never accumulate enough to clear the market."
        )
    r_star = float(brentq(excess, lo, hi, xtol=xtol))

    eq = solve_at(r_star)
    sim_res = simulate_vfi_panel(eq["policy_a"], a_grid, e_grid, P_e,
                                 n_agents=2000, T_periods=100)

    return {
        "r_star": r_star,
        "w_star": eq["w"],
        "K_star": eq["K"],
        "L_star": eq["L"],
        "excess_demand": eq["excess"],
        "distribution": eq["mu"],
        "n_evals": n_evals["n"],
        "V": eq["V"],
        "policy_a": eq["policy_a"],
        "policy_c": eq["policy_c"],
        "policy_n": eq["policy_n"],
        "a_grid": a_grid,
        "e_grid": e_grid,
        "P_e": P_e,
        "sim_res": sim_res,
        "elapsed": time.time() - t_start,
    }
