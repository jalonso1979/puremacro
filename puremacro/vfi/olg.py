"""Overlapping-generations (OLG) stationary general equilibrium.

Composes the finite-horizon household (``FiniteHorizonProblem``) with demographic
weighting and market clearing. ``stationary_age_weights`` gives the population
mass by age; ``olg_aggregate`` integrates an age- and policy-dependent quantity
over the cross-section of cohorts (the right tool for endogenous labor, where
hours are chosen per age); ``olg_stationary_equilibrium`` (see Task 2) finds the
market-clearing price. Endogenous labor (extensive + intensive margins) is the
decision ``d`` = hours on a grid that includes 0 (0 = non-participation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from puremacro.vfi.aggregate import evaluate_on_grid
from puremacro.vfi.finite_horizon import life_cycle_distribution


def stationary_age_weights(horizon, *, survival=None, pop_growth: float = 0.0):
    """Stationary population mass by age, length ``horizon``, summing to 1.

    ``ell_0 = 1``; ``ell_j = ell_{j-1} * s_{j-1} / (1 + pop_growth)`` where the
    conditional survival ``s`` (default all ones) is the same length-``horizon``
    vector ``FiniteHorizonProblem`` takes. Uniform ``1/J`` with no mortality and
    no growth; declining with age under mortality or positive population growth.
    """
    J = int(horizon)
    if J < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")
    s = np.ones(J) if survival is None else np.asarray(survival, dtype=float)
    if s.shape != (J,):
        raise ValueError(f"survival must have shape ({J},); got {s.shape}")
    if np.any(s <= 0.0) or np.any(s > 1.0):
        raise ValueError("survival probabilities must lie in (0, 1]")
    g = 1.0 + float(pop_growth)
    if g <= 0.0:
        raise ValueError(f"pop_growth must be > -1; got {pop_growth}")
    ell = np.empty(J)
    ell[0] = 1.0
    for j in range(1, J):
        ell[j] = ell[j - 1] * s[j - 1] / g
    return ell / ell.sum()


def olg_aggregate(fn, life_cycle_dist, age_weights, solution, a_grid, z_grid, *,
                  d_grid=None, params=None):
    """Per-age, demographically weighted integral ``sum_j ell_j * E_{cohort j}[fn]``.

    ``fn`` follows the finite-horizon eval convention -- ``fn([d,] a'_1.., a_1..,
    z, age, *params, xp=np)`` -- and is evaluated at each age ``j`` over cohort
    ``j``'s measure ``life_cycle_dist[j]`` using age ``j``'s policy
    (``solution.policy_aprime[j]`` and, if present, ``solution.policy_d[j]``).
    Correctly aggregates age- and policy-dependent quantities (labor, consumption)
    as well as pure state quantities (assets). Returns a float.
    """
    lcd = np.asarray(life_cycle_dist, dtype=float)
    w = np.asarray(age_weights, dtype=float)
    J = lcd.shape[0]
    if w.shape != (J,):
        raise ValueError(f"age_weights must have shape ({J},); got {w.shape}")
    pol_a = np.asarray(solution.policy_aprime)
    pol_d = None if solution.policy_d is None else np.asarray(solution.policy_d)
    extra = params or {}
    total = 0.0
    for j in range(J):
        pdj = None if pol_d is None else pol_d[j]
        vals = evaluate_on_grid(
            fn, pol_a[j], a_grid, z_grid,
            params={"age": j, **extra}, policy_d=pdj, d_grid=d_grid,
        )
        total += float(w[j]) * float(np.sum(lcd[j] * vals))
    return total


@dataclass(frozen=True)
class OLGEquilibrium:
    """A solved stationary OLG general equilibrium."""
    price: float
    residual: float
    solution: object         # FiniteHorizonSolution at the equilibrium price
    life_cycle_dist: np.ndarray
    age_weights: np.ndarray
    problem: object          # FiniteHorizonProblem at the equilibrium price
    n_evals: int


def olg_stationary_equilibrium(build_problem, market_residual, price_bracket, *,
                               age_weights, backend: str = "numpy",
                               xtol: float = 1e-6, max_evals: int = 100):
    """Market-clearing price for a stationary OLG economy (scalar brentq).

    ``build_problem(price) -> FiniteHorizonProblem`` builds the life-cycle
    household at ``price``; ``market_residual(price, solution, life_cycle_dist,
    age_weights, problem) -> float`` (= 0 at equilibrium) aggregates the cleared
    market(s) -- typically capital and (endogenous) labor via ``olg_aggregate``.
    ``age_weights`` is the demographic mass by age (e.g. ``stationary_age_weights``).
    Returns an ``OLGEquilibrium``.
    """
    w = np.asarray(age_weights, dtype=float)
    counter = {"n": 0}

    def _resid(price):
        counter["n"] += 1
        prob = build_problem(price)
        sol = prob.solve(backend)
        lcd = life_cycle_distribution(sol, np.asarray(prob.P_z, dtype=float))
        return float(market_residual(price, sol, lcd, w, prob))

    lo, hi = price_bracket
    p_star = brentq(_resid, lo, hi, xtol=xtol, maxiter=max_evals)

    prob = build_problem(p_star)
    sol = prob.solve(backend)
    lcd = life_cycle_distribution(sol, np.asarray(prob.P_z, dtype=float))
    resid = float(market_residual(p_star, sol, lcd, w, prob))
    return OLGEquilibrium(price=float(p_star), residual=resid, solution=sol,
                          life_cycle_dist=lcd, age_weights=w, problem=prob,
                          n_evals=counter["n"])


__all__ = ["stationary_age_weights", "olg_aggregate", "olg_stationary_equilibrium", "OLGEquilibrium"]
