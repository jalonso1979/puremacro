"""Epstein-Zin recursive preferences (VFIToolkit exotic-preferences port).

Separates risk aversion ``gamma`` from the intertemporal elasticity of
substitution ``psi``. The return function returns the PERIOD FELICITY
``u(d,a',a,z) > 0`` (feasible) / ``-inf`` (infeasible), and the value solves the
recursion

    V = [ (1-beta) u^rho + beta * CE^rho ]^(1/rho),   rho = 1 - 1/psi,
    CE(a',z) = ( E_{z'|z}[ V(a',z')^(1-gamma) ] )^(1/(1-gamma)),

maximised over (d, a'). When ``gamma == 1/psi`` this collapses to time-separable
expected utility (the reduction oracle). numpy-only (a specialised, non-additive
recursion; no GPU/numba path). Requires positive felicity, ``psi != 1`` and
``gamma != 1`` (the log limits are excluded).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from puremacro.vfi.returnfn import build_return_tensor


@dataclass(frozen=True)
class EpsteinZinSolution:
    """Solved value and greedy policy under Epstein-Zin preferences (numpy)."""
    V: np.ndarray                 # (n_a, n_z), strictly positive
    policy_aprime: np.ndarray     # (n_a, n_z) int a'-indices
    policy_d: np.ndarray | None   # (n_a, n_z) int d-indices, or None
    n_iter: int
    sup_norm: float


@dataclass(frozen=True)
class EpsteinZinProblem:
    """Infinite-horizon VFI with Epstein-Zin recursive preferences (see module doc)."""
    a_grid: np.ndarray
    z_grid: np.ndarray
    P_z: np.ndarray
    return_fn: Callable
    beta: float
    gamma: float
    psi: float
    params: dict = field(default_factory=dict)
    d_grid: np.ndarray | None = None
    options: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 < self.beta < 1.0):
            raise ValueError(f"beta must be in (0,1); got {self.beta}")
        if self.gamma <= 0.0 or self.gamma == 1.0:
            raise ValueError(f"gamma must be > 0 and != 1 (log limit); got {self.gamma}")
        if self.psi <= 0.0 or self.psi == 1.0:
            raise ValueError(f"psi must be > 0 and != 1 (log limit); got {self.psi}")

    def solve(self) -> EpsteinZinSolution:
        a = np.asarray(self.a_grid, dtype=float)
        z = np.asarray(self.z_grid, dtype=float)
        P = np.asarray(self.P_z, dtype=float)
        d = None if self.d_grid is None else np.asarray(self.d_grid, dtype=float)
        beta = float(self.beta)
        rho = 1.0 - 1.0 / float(self.psi)
        omg = 1.0 - float(self.gamma)
        opt = self.options
        tol = float(opt.get("tol", 1e-9))
        max_iter = int(opt.get("max_iter", 10_000))

        U = build_return_tensor(self.return_fn, a, z, self.params, d_grid=d, xp=np)
        n_d, n_ap, n_a, n_z = U.shape
        feasible = np.isfinite(U) & (U > 0.0)
        U_safe = np.where(feasible, U, 1.0)
        Urho = np.where(feasible, U_safe ** rho, 0.0)
        PzT = P.T

        V = np.ones((n_a, n_z))
        flat = np.zeros((n_a, n_z), dtype=np.int64)
        sup = np.inf
        for it in range(1, max_iter + 1):
            CE = ((V ** omg) @ PzT) ** (1.0 / omg)            # (n_ap, n_z) certainty equiv
            CErho = (CE ** rho).reshape(1, n_ap, 1, n_z)
            inner = (1.0 - beta) * Urho + beta * CErho        # (n_d,n_ap,n_a,n_z)
            Vcand = np.where(feasible, np.where(feasible, inner, 1.0) ** (1.0 / rho), -np.inf)
            Qflat = Vcand.reshape(n_d * n_ap, n_a, n_z)
            V_new = Qflat.max(axis=0)
            flat = Qflat.argmax(axis=0)
            if not np.all(np.isfinite(V_new)):
                raise RuntimeError(
                    "EpsteinZin: a state has no feasible action (felicity must be "
                    "positive somewhere for every (a, z))"
                )
            sup = float(np.max(np.abs(V_new - V)))
            V = V_new
            if sup < tol:
                break
        else:
            raise RuntimeError(
                f"EpsteinZinProblem did not converge in {max_iter} iterations "
                f"(sup-norm {sup:.3e} > tol {tol:.1e})"
            )

        flat = flat.astype(np.int64)
        policy_aprime = flat % n_ap
        policy_d = (flat // n_ap) if d is not None else None
        return EpsteinZinSolution(V=V, policy_aprime=policy_aprime, policy_d=policy_d,
                                  n_iter=it, sup_norm=sup)


__all__ = ["EpsteinZinProblem", "EpsteinZinSolution"]
