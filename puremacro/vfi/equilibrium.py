"""Stationary general equilibrium for heterogeneous-agent models
(VFIToolkit HeteroAgentStationaryEqm_Case1).

Find the price at which markets clear, wrapping the solve -> stationary
distribution -> aggregate pipeline in a root-find. v1: a single scalar price
(the canonical Aiyagari interest-rate case), via scipy.optimize.brentq on a
user-supplied market-clearing residual. The user provides:

    build_problem(price) -> VFIProblem
    market_residual(price, solution, mu, problem) -> float   # = 0 at equilibrium

The engine runs the inner solve + stationary distribution at each candidate
price. (Vector-price GE via a multivariate root-finder is a deferred extension.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from puremacro.vfi.permanent_types import solve_permanent_types


@dataclass(frozen=True)
class EquilibriumResult:
    """A solved stationary general equilibrium."""
    price: float
    residual: float
    solution: object        # VFISolution at the equilibrium price
    distribution: np.ndarray  # stationary mu(a,z) at the equilibrium price
    problem: object         # VFIProblem at the equilibrium price
    n_evals: int            # root-finder residual evaluations (total solves = n_evals + 1)


def stationary_equilibrium(build_problem, market_residual, price_bracket, *,
                           backend: str = "numpy", dist_options=None,
                           xtol: float = 1e-6, max_evals: int = 100):
    """Find the market-clearing price in ``price_bracket=(lo, hi)`` (scalar brentq).

    ``build_problem(price) -> VFIProblem``; ``market_residual(price, solution,
    mu, problem) -> float``. The residual must change sign across the bracket
    (else brentq raises ValueError). Returns an EquilibriumResult evaluated at
    the solved price.
    """
    dist_options = dist_options or {}
    counter = {"n": 0}

    def _resid(price):
        counter["n"] += 1
        prob = build_problem(price)
        sol = prob.solve(backend)
        mu = prob.stationary_distribution(sol, **dist_options)
        return float(market_residual(price, sol, mu, prob))

    lo, hi = price_bracket
    p_star = brentq(_resid, lo, hi, xtol=xtol, maxiter=max_evals)

    prob = build_problem(p_star)
    sol = prob.solve(backend)
    mu = prob.stationary_distribution(sol, **dist_options)
    resid = float(market_residual(p_star, sol, mu, prob))
    return EquilibriumResult(
        price=float(p_star), residual=resid, solution=sol,
        distribution=mu, problem=prob, n_evals=counter["n"],
    )


@dataclass(frozen=True)
class PermanentTypesEquilibrium:
    """A solved stationary GE with permanent types."""
    price: float
    residual: float
    pt_solution: object     # PermanentTypesSolution at the equilibrium price
    n_evals: int            # residual evaluations (total PType solves = n_evals + 1)


def stationary_equilibrium_types(build_problem, weights, market_residual,
                                 price_bracket, *, backend: str = "numpy",
                                 dist_tol: float = 1e-12,
                                 dist_max_iter: int = 100_000,
                                 xtol: float = 1e-6, max_evals: int = 100):
    """Market-clearing price for a model with permanent types (scalar brentq).

    ``build_problem(price, t) -> VFIProblem`` builds type ``t``'s problem at
    ``price``; ``weights`` are the type masses (>= 0, sum 1); ``market_residual(
    price, pt_solution) -> float`` (= 0 at equilibrium) reads population
    aggregates off the ``PermanentTypesSolution`` (e.g. ``pt_solution.aggregate(
    fn)``). The residual must change sign across ``price_bracket=(lo, hi)``.
    Returns a ``PermanentTypesEquilibrium`` at the solved price.
    """
    counter = {"n": 0}

    def _resid(price):
        counter["n"] += 1
        pt = solve_permanent_types(
            lambda t: build_problem(price, t), weights, backend=backend,
            dist_tol=dist_tol, dist_max_iter=dist_max_iter,
        )
        return float(market_residual(price, pt))

    lo, hi = price_bracket
    p_star = brentq(_resid, lo, hi, xtol=xtol, maxiter=max_evals)

    pt = solve_permanent_types(
        lambda t: build_problem(p_star, t), weights, backend=backend,
        dist_tol=dist_tol, dist_max_iter=dist_max_iter,
    )
    resid = float(market_residual(p_star, pt))
    return PermanentTypesEquilibrium(price=float(p_star), residual=resid,
                                     pt_solution=pt, n_evals=counter["n"])


__all__ = ["stationary_equilibrium", "EquilibriumResult", "stationary_equilibrium_types", "PermanentTypesEquilibrium"]
