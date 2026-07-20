"""Permanent types (VFIToolkit PType): fixed ex-ante heterogeneity.

Solve a separate ``VFIProblem`` for each permanent type, build each type's
stationary distribution, and aggregate across types by their population masses
``weights`` (which sum to 1). The user supplies ``build_problem(t) -> VFIProblem``
(the same factory seam used by ``stationary_equilibrium`` / ``transition_path``),
so types may differ in ANY parameter or grid. Type-weighted aggregates
``sum_t w_t * E_{mu_t}[fn]`` are grid-agnostic (each type integrates over its own
policy and grids); the population mixture measure ``sum_t w_t * mu_t`` is only
defined when the types share an (a, z) grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from puremacro.vfi.problem import VFIProblem, VFISolution


@dataclass(frozen=True)
class PermanentTypesSolution:
    """Per-type solved problems / distributions plus their population masses."""
    problems: tuple
    solutions: tuple
    distributions: tuple        # per-type stationary mu, each (n_a, n_z)
    weights: np.ndarray         # (n_types,) masses summing to 1

    def type_aggregates(self, fn, *, per_type_params=None) -> np.ndarray:
        """Per-type integral E_{mu_t}[fn] (each type uses its own problem/grids)."""
        out = np.empty(len(self.weights))
        for t in range(len(self.weights)):
            params = None if per_type_params is None else per_type_params[t]
            out[t] = self.problems[t].aggregate(
                self.solutions[t], self.distributions[t], fn, params=params
            )
        return out

    def aggregate(self, fn, *, per_type_params=None) -> float:
        """Type-weighted population aggregate ``sum_t w_t * E_{mu_t}[fn]``."""
        return float(np.dot(self.weights, self.type_aggregates(
            fn, per_type_params=per_type_params)))

    def mixture_distribution(self) -> np.ndarray:
        """Population mixture measure ``sum_t w_t * mu_t`` (requires a shared grid).

        Raises if the per-type distributions do not all share the same shape.
        """
        shapes = {mu.shape for mu in self.distributions}
        if len(shapes) != 1:
            raise ValueError(
                "mixture_distribution requires all types to share the same shape "
                f"(a, z) grid; got {sorted(shapes)}"
            )
        mix = np.zeros(self.distributions[0].shape)
        for w, mu in zip(self.weights, self.distributions):
            mix += float(w) * np.asarray(mu, dtype=float)
        return mix


def solve_permanent_types(build_problem, weights, *, backend: str = "numpy",
                          dist_tol: float = 1e-12,
                          dist_max_iter: int = 100_000) -> PermanentTypesSolution:
    """Solve a model with permanent types; see the module docstring.

    ``build_problem(t)`` returns the ``VFIProblem`` for type index ``t`` in
    ``range(len(weights))``. ``weights`` are the population masses (>= 0, sum 1).
    Returns a ``PermanentTypesSolution``.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size < 1:
        raise ValueError("weights must be 1-D with >= 1 entry")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    total = float(w.sum())
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f"weights must sum to 1; got {total}")
    problems, solutions, dists = [], [], []
    for t in range(w.size):
        prob = build_problem(t)
        if not isinstance(prob, VFIProblem):
            raise TypeError(
                f"build_problem({t}) must return a VFIProblem; got {type(prob)}"
            )
        sol = prob.solve(backend)
        mu = prob.stationary_distribution(sol, tol=dist_tol, max_iter=dist_max_iter)
        problems.append(prob)
        solutions.append(sol)
        dists.append(mu)
    return PermanentTypesSolution(
        problems=tuple(problems), solutions=tuple(solutions),
        distributions=tuple(dists), weights=w,
    )


__all__ = ["PermanentTypesSolution", "solve_permanent_types"]
