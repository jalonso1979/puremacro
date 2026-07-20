"""Case 2 value-function iteration: the next state follows an exogenous rule.

In VFIToolkit's "Case 2", the next endogenous state is NOT freely chosen but
determined by a rule a' = Phi(d, a, z) of a contemporaneous decision d. The
Bellman is

    V(a,z) = max_d  R(d,a,z) + beta * E[V(Phi(d,a,z), z') | z],

so the return tensor is (n_d, n_a, n_z) (no a' axis) and the continuation is
GATHERED at the Phi indices instead of maximised over a'. Two user functions:

    return_fn(d, a, z, *params, xp=np)       -> (n_d, n_a, n_z) payoff (VALUE grids)
    aprime_fn(d_idx, a_idx, z_idx, *params)  -> (n_d, n_a, n_z) a'-INDICES (INDEX grids)

numpy-only in v1 (a multi-backend version is deferred). With d_grid == a_grid and
Phi(d_idx,·,·) = d_idx this reduces exactly to Case 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Case2Solution:
    """Solved Case-2 value and decision policy (numpy)."""
    V: np.ndarray            # (n_a, n_z)
    policy_d: np.ndarray     # (n_a, n_z) int indices into d_grid
    n_iter: int
    sup_norm: float


@dataclass(frozen=True)
class Case2Problem:
    """Case-2 VFI problem (exogenous next-state rule Phi). See module docstring.

    ``options`` keys: howard (bool, default True), n_howard (int, 20),
    tol (float, 1e-8), max_iter (int, 10_000).
    """
    a_grid: np.ndarray
    z_grid: np.ndarray
    P_z: np.ndarray
    d_grid: np.ndarray
    return_fn: Callable
    aprime_fn: Callable
    beta: float
    params: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 < self.beta < 1.0):
            raise ValueError(f"beta must be in (0,1); got {self.beta}")
        a = np.asarray(self.a_grid); z = np.asarray(self.z_grid)
        d = np.asarray(self.d_grid); P = np.asarray(self.P_z)
        if a.ndim != 1 or a.size < 2:
            raise ValueError("a_grid must be 1-D with >= 2 points")
        if z.ndim != 1 or z.size < 1:
            raise ValueError("z_grid must be 1-D with >= 1 points")
        if d.ndim != 1 or d.size < 1:
            raise ValueError("d_grid must be 1-D with >= 1 points")
        if P.shape != (z.size, z.size):
            raise ValueError(f"P_z must be ({z.size},{z.size}); got {P.shape}")
        if not np.allclose(P.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("P_z rows must sum to 1")

    def solve(self) -> Case2Solution:
        a = np.asarray(self.a_grid, dtype=float)
        z = np.asarray(self.z_grid, dtype=float)
        d = np.asarray(self.d_grid, dtype=float)
        P = np.asarray(self.P_z, dtype=float)
        n_d, n_a, n_z = d.size, a.size, z.size
        opt = self.options
        howard = bool(opt.get("howard", True))
        n_howard = int(opt.get("n_howard", 20))
        tol = float(opt.get("tol", 1e-8))
        max_iter = int(opt.get("max_iter", 10_000))
        pvals = tuple(self.params.values())

        # return tensor R(d,a,z) on broadcast VALUE grids (broadcast-up partials)
        R = self.return_fn(d.reshape(n_d, 1, 1), a.reshape(1, n_a, 1),
                           z.reshape(1, 1, n_z), *pvals, xp=np)
        R = np.asarray(R, dtype=float)
        if R.shape != (n_d, n_a, n_z):
            R = R + np.zeros((n_d, n_a, n_z))

        # Phi index tensor on broadcast INDEX grids
        di = np.arange(n_d).reshape(n_d, 1, 1)
        ai = np.arange(n_a).reshape(1, n_a, 1)
        zi = np.arange(n_z).reshape(1, 1, n_z)
        aprime_idx = np.asarray(self.aprime_fn(di, ai, zi, *pvals))
        if aprime_idx.shape != (n_d, n_a, n_z):
            aprime_idx = aprime_idx + np.zeros((n_d, n_a, n_z), dtype=int)
        aprime_idx = aprime_idx.astype(np.intp)
        if aprime_idx.min() < 0 or aprime_idx.max() >= n_a:
            raise ValueError("aprime_fn must return valid a-grid indices in [0, n_a)")

        z_bcast = np.broadcast_to(np.arange(n_z), (n_d, n_a, n_z))
        flat = aprime_idx * n_z + z_bcast            # into EV.reshape(-1)
        PzT = P.T
        V = np.zeros((n_a, n_z))
        sup = np.inf
        for it in range(1, max_iter + 1):
            EV = (V @ PzT).reshape(-1)
            Q = R + self.beta * EV[flat]             # (n_d, n_a, n_z)
            V_new = Q.max(axis=0)
            pol_d = Q.argmax(axis=0)
            sup = float(np.max(np.abs(V_new - V)))
            V = V_new
            if howard and n_howard:
                # gather R and a' at the greedy decision, hold fixed
                aa = np.arange(n_a)[:, None]; zz = np.arange(n_z)[None, :]
                R_pol = R[pol_d, aa, zz]                       # (n_a, n_z)
                ap_pol = aprime_idx[pol_d, aa, zz]             # (n_a, n_z)
                for _ in range(n_howard):
                    EVh = V @ PzT
                    V = R_pol + self.beta * EVh[ap_pol, zz]
            if sup < tol:
                return Case2Solution(V=V, policy_d=pol_d.astype(np.int64),
                                     n_iter=it, sup_norm=sup)
        raise RuntimeError(
            f"Case2 VFI did not converge in {max_iter} iterations "
            f"(sup-norm {sup:.3e} > tol {tol:.1e})"
        )


__all__ = ["Case2Problem", "Case2Solution"]
