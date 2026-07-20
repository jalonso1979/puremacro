"""Deterministic perfect-foresight transition paths between stationary equilibria
(VFIToolkit TransitionPath_Case1).

Given an initial distribution mu_0, a terminal (final-steady-state) value
function V_T, a period-t problem builder, and a map from the realized aggregates
to implied prices, solve for the price path {p_t}_{t=0..T-1} that is consistent
along the transition. The algorithm is a damped fixed point:

  repeat until ||p_new - p|| < tol:
    backward: V_next = V_T; for t = T-1..0: one Bellman step at price p_t with
              continuation V_next; store policy_t; V_next = V_t.
    forward:  mu_0 given; for t = 0..T-1: mu_{t+1} = push_distribution(mu_t, policy_t).
    p_new = implied_price_path(dists, policies, p); p = damping*p_new + (1-damping)*p

The exogenous Markov P_z is constant along the path (only prices move). Reuses
build_return_tensor + bellman_step (backward) and push_distribution (forward).
numpy-only (it consumes/produces numpy policies and distributions).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from puremacro.vfi.distribution import push_distribution
from puremacro.vfi.kernels import bellman_step
from puremacro.vfi.returnfn import build_return_tensor


@dataclass(frozen=True)
class TransitionPath:
    """A solved deterministic transition path."""
    price_path: np.ndarray          # (T,)
    distributions: list             # length T+1: mu_0 .. mu_T
    policies: list                  # length T: policy_aprime per period (n_a, n_z) int
    n_iter: int
    gap: float


def transition_path(mu0, terminal_V, build_problem, implied_price_path, price_path0,
                    *, damping: float = 0.5, tol: float = 1e-5, max_iter: int = 300):
    """Solve the perfect-foresight transition (damped fixed point on the price path).

    mu0: initial distribution (n_a, n_z). terminal_V: final-SS value V_T (n_a, n_z).
    build_problem(t, price_path) -> VFIProblem (period-t household problem).
    implied_price_path(distributions, policies, price_path) -> (T,) implied prices.
    Returns a TransitionPath; raises RuntimeError if not converged in max_iter.
    """
    p = np.asarray(price_path0, dtype=float).copy()
    T = int(p.shape[0])
    mu0 = np.asarray(mu0, dtype=float)
    terminal_V = np.asarray(terminal_V, dtype=float)
    P_z = np.asarray(build_problem(0, p).P_z, dtype=float)  # constant along the path

    gap = float("inf")
    dists: list = []
    policies: list = []
    for it in range(1, max_iter + 1):
        # --- backward: time-varying VFI from the terminal value ---
        V_next = terminal_V
        policies = [None] * T
        for t in range(T - 1, -1, -1):
            prob = build_problem(t, p)
            a = np.asarray(prob.a_grid, dtype=float)
            z = np.asarray(prob.z_grid, dtype=float)
            n_a = a.size
            d = None if prob.d_grid is None else np.asarray(prob.d_grid, dtype=float)
            R = build_return_tensor(prob.return_fn, a, z, prob.params, d_grid=d, xp=np)
            EV = V_next @ P_z.T
            V_t, flat = bellman_step(R, EV, float(prob.beta), xp=np)
            policies[t] = (np.asarray(flat).astype(np.int64) % n_a)
            V_next = V_t
        # --- forward: push the distribution period by period ---
        dists = [None] * (T + 1)
        dists[0] = mu0
        for t in range(T):
            dists[t + 1] = push_distribution(dists[t], policies[t], P_z)
        # --- update the price path ---
        p_new = np.asarray(implied_price_path(dists, policies, p), dtype=float)
        gap = float(np.max(np.abs(p_new - p)))
        p = damping * p_new + (1.0 - damping) * p
        if gap < tol:
            return TransitionPath(price_path=p, distributions=dists,
                                  policies=policies, n_iter=it, gap=gap)
    raise RuntimeError(
        f"transition_path did not converge in {max_iter} iterations "
        f"(price-path sup-gap {gap:.3e} > tol {tol:.1e})"
    )


__all__ = ["transition_path", "TransitionPath"]
