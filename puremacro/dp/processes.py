"""Exogenous shock processes for puremacro.dp models.

Each discrete-time process discretizes to a Markov chain ``(grid, P)`` with the
existing ``puremacro.vfi.discretize`` routines, so a dp model and a hand-built
``VFIProblem`` see exactly the same chain. ``Jump`` is the continuous-time
counterpart (a generator) for models with a ``drift``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from puremacro.vfi.discretize import rouwenhorst, tauchen


@dataclass(frozen=True)
class AR1:
    """Gaussian AR(1) ``x' = rho*x + eps``, ``eps ~ N(0, sigma^2)``, on ``n`` states.

    ``method`` is ``"tauchen"`` (grid half-width ``m`` unconditional standard
    deviations) or ``"rouwenhorst"``. The grid is in levels of ``x``; write
    ``exp(z)`` in the model when the process is for log income.
    """
    rho: float
    sigma: float
    n: int
    method: str = "tauchen"
    m: float = 3.0

    def discretize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.method == "tauchen":
            return tauchen(n=int(self.n), rho=float(self.rho), sigma=float(self.sigma), m=float(self.m))
        if self.method == "rouwenhorst":
            return rouwenhorst(int(self.n), float(self.rho), float(self.sigma))
        raise ValueError(f"AR1 method must be 'tauchen' or 'rouwenhorst'; got {self.method!r}")


@dataclass(frozen=True)
class Markov:
    """An explicit Markov chain: state values ``grid`` and row-stochastic ``P``."""
    grid: object
    P: object

    def discretize(self) -> tuple[np.ndarray, np.ndarray]:
        g = np.asarray(self.grid, dtype=float).ravel()
        P = np.asarray(self.P, dtype=float)
        if P.shape != (g.size, g.size):
            raise ValueError(f"Markov P must be ({g.size},{g.size}); got {P.shape}")
        if np.any(P < 0) or not np.allclose(P.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("Markov P must be row-stochastic")
        return g, P


@dataclass(frozen=True)
class Jump:
    """A continuous-time Markov chain (Poisson jumps) for continuous-time models:
    state values ``grid`` and generator ``rates`` (non-negative off-diagonal
    intensities, zero row sums), e.g. ``Jump([0.1, 0.2], [[-1.2, 1.2], [1.2, -1.2]])``."""
    grid: object
    rates: object

    def generator(self) -> tuple[np.ndarray, np.ndarray]:
        g = np.asarray(self.grid, dtype=float).ravel()
        A = np.asarray(self.rates, dtype=float)
        if A.shape != (g.size, g.size):
            raise ValueError(f"Jump rates must be ({g.size},{g.size}); got {A.shape}")
        off = A - np.diag(np.diag(A))
        if np.any(off < 0) or not np.allclose(A.sum(axis=1), 0.0, atol=1e-10):
            raise ValueError("Jump rates must have non-negative off-diagonal entries and zero row sums")
        return g, A


__all__ = ["AR1", "Jump", "Markov"]
