"""Aggregate risk in dp models: the Krusell and Smith (1998) forecast-rule fixed point.

A model with ``aggregate_shock("Z", ...)`` and ``aggregate_state("K", grid,
mean="a")`` is a household VFI whose exogenous state is (idiosyncratic shocks,
Z, K). Agents perceive ``log K(+1) = b0[Z] + b1[Z]*log K``; the K-transition
puts that forecast on the K-grid by linear lottery weights
(``vfi.krusell_smith.ks_exog_transition``). The loop solves the household,
simulates the wealth distribution along one drawn path of Z
(``ks_simulate``, policies interpolated in K at the realised mean of ``a``),
regresses log K(+1) on log K per Z state, and damps the update. Prices and the
firm are whatever the model's equations say, so any technology and any reward
the VFI compiler accepts work.

With ``method="egm"`` the household is solved by EGM (the return r(Z, K) may
vary with the exogenous state) and the simulation pushes the distribution
with the continuous policy. Discrete VFI policies are pinned to grid nodes, so
on a coarse asset grid a one-percent TFP shock can leave every choice, and
hence aggregate capital, unchanged; EGM policies move continuously with (Z, K).
"""
from __future__ import annotations

import numpy as np

from puremacro.dp._expr import ModelSpecError
from puremacro.vfi.distribution import lottery_distribution, lottery_push, stationary_distribution
from puremacro.vfi.krusell_smith import (
    KSEquilibrium,
    _regress_forecast,
    _simulate_markov_path,
    ks_exog_transition,
    ks_simulate,
)


def _simulate_continuous(aprime, a, P_idio, K_grid, Z_path, mu0):
    """``ks_simulate`` for a continuous next-asset policy (n_a, n_idio, nZ, nK)."""
    mu = np.asarray(mu0, dtype=float) / np.sum(mu0)
    T = len(Z_path)
    K_path = np.empty(T + 1)
    for t in range(T):
        K_t = float(np.sum(mu * a[:, None]))
        K_path[t] = K_t
        Kc = min(max(K_t, K_grid[0]), K_grid[-1])
        j = int(np.clip(np.searchsorted(K_grid, Kc) - 1, 0, K_grid.size - 2))
        w = (K_grid[j + 1] - Kc) / (K_grid[j + 1] - K_grid[j])
        iZ = int(Z_path[t])
        mu = lottery_push(mu, w * aprime[:, :, iZ, j] + (1.0 - w) * aprime[:, :, iZ, j + 1], a, P_idio)
    K_path[T] = float(np.sum(mu * a[:, None]))
    return K_path


def solve_ks(comp, *, method="vfi", backend="numpy", options=None, T: int = 3000, burn_in: int = 300, seed: int = 0,
             damping: float = 0.5, tol: float = 1e-3, max_outer: int = 60, mu0=None, b0=None, b1=None):
    """Krusell-Smith loop for a compiled dp model; returns a ``DPSolution`` whose
    ``raw`` is the household VFI at the final forecast rule and ``equilibrium`` a
    ``vfi.krusell_smith.KSEquilibrium`` (``no_agg_risk_K`` is nan here).

    ``mu0`` is the (n_a, n_idio) wealth distribution the simulation starts from;
    by default the stationary distribution of the household policy when agents
    expect K to stay at the middle of its grid, in the most likely aggregate state.
    ``b0``/``b1`` are the initial forecast coefficients (default: K stays put).
    """
    from puremacro.dp._results import DPSolution

    if not 0 < damping <= 1:
        raise ModelSpecError(f"ks damping must be in (0, 1]; got {damping}")
    if not 0 <= burn_in < T - 2:
        raise ModelSpecError(f"ks burn_in must be in [0, T - 2); got burn_in={burn_in}, T={T}")
    a = comp.state_grids[0]
    K_grid = comp.z_grids[-1]
    P_idio, P_Z = comp.P_idio, comp.P_agg
    n_idio, nZ, nK = P_idio.shape[0], P_Z.shape[0], K_grid.size
    params = comp.param_values()
    options = dict(options or {})

    plan = None
    if method == "egm":
        from puremacro.dp._egm import EGMPlan

        plan = EGMPlan(comp)
    warm = {"c": None}

    def household(b0_, b1_):
        comp.P, _ = ks_exog_transition(np.arange(n_idio), P_idio, np.arange(nZ), P_Z, K_grid, b0_, b1_)
        if plan is not None:
            sol = plan.solve_household(params, tol=options.get("tol", 1e-8),
                                       max_iter=options.get("max_iter", 10_000), c0=warm["c"])
            warm["c"] = sol.c
            return sol, sol.aprime.reshape(a.size, n_idio, nZ, nK)
        sol = comp._vfi_problem(params, options).solve(backend)
        return sol, sol.policy_aprime.reshape(a.size, n_idio, nZ, nK)

    b0 = np.zeros(nZ) if b0 is None else np.asarray(b0, dtype=float).copy()
    b1 = np.ones(nZ) if b1 is None else np.asarray(b1, dtype=float).copy()
    if mu0 is None:
        from puremacro.vfi.discretize import markov_stationary

        k_mid = nK // 2
        _, pol = household(np.full(nZ, np.log(K_grid[k_mid])), np.zeros(nZ))
        iz = int(np.argmax(markov_stationary(P_Z)))
        mu0 = (lottery_distribution(pol[:, :, iz, k_mid], a, P_idio) if plan is not None
               else stationary_distribution(pol[:, :, iz, k_mid], P_idio))
    Z_path = _simulate_markov_path(P_Z, int(T), np.random.default_rng(seed))

    converged, n_outer = False, 0
    for outer in range(int(max_outer)):
        n_outer = outer + 1
        sol, pol = household(b0, b1)
        K_path = (_simulate_continuous(pol, a, P_idio, K_grid, Z_path, mu0) if plan is not None
                  else ks_simulate(pol, a, P_idio, K_grid, Z_path, mu0=mu0))
        nb0, nb1, R2 = _regress_forecast(K_path, Z_path, nZ, int(burn_in))
        diff = max(float(np.max(np.abs(nb0 - b0))), float(np.max(np.abs(nb1 - b1))))
        b0 = damping * nb0 + (1.0 - damping) * b0
        b1 = damping * nb1 + (1.0 - damping) * b1
        if diff < tol:
            converged = True
            break
    eq = KSEquilibrium(b0=b0, b1=b1, r_squared=R2, K_path=K_path, mean_K=float(np.mean(K_path[burn_in:])),
                       no_agg_risk_K=float("nan"), n_outer=n_outer, converged=converged)
    return DPSolution(comp, sol, params=params, aggregates={comp.shock_names[-1]: eq.mean_K},
                      equilibrium=eq, method=f"{method}+ks")
