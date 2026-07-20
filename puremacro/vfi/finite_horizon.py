"""Finite-horizon (life-cycle) value-function iteration for puremacro.vfi.

A finite-horizon problem has ages j = 0..J-1 and value
    V_j(a,z) = max_{d,a'} R_j(d,a',a,z) + beta * E[V_{j+1}(a',z') | z],
with terminal continuation V_J = terminal_value (default 0). This is NOT a
fixed-point iteration -- it is exactly J backward applications of the one-step
Bellman operator, so the infinite-horizon kernels (bellman_step,
build_return_tensor) are reused per age. The return function is age-dependent:

    no decision:   return_fn(aprime, a, z, age, *params, xp=np)
    with decision: return_fn(d, aprime, a, z, age, *params, xp=np)

(``age`` is injected as the first positional param.) Output V/policy are
age-indexed, shape (J, n_a, n_z). xp-generic (numpy/mlx/cupy); a numba per-age
twin is deferred.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from puremacro import _backend as _bk
from puremacro.vfi.distribution import push_distribution
from puremacro.vfi.kernels import bellman_step
from puremacro.vfi.returnfn import build_return_tensor


@dataclass(frozen=True)
class FiniteHorizonSolution:
    """Age-indexed value and policy from a life-cycle solve (always numpy)."""
    V: np.ndarray                  # (J, n_a, n_z)
    policy_aprime: np.ndarray      # (J, n_a, n_z) int a'-indices
    policy_d: np.ndarray | None    # (J, n_a, n_z) int d-indices, or None
    horizon: int
    endo_shape: tuple = ()        # component sizes of the endogenous grid; (n_a,) if 1-D

    def policy_components(self):
        """Per-asset next-state index arrays, each (J, n_a, n_z), unravelled from
        the flat ``policy_aprime`` over ``endo_shape``. 1-tuple for a single asset."""
        shape = self.endo_shape if self.endo_shape else (self.policy_aprime.shape[1],)
        return tuple(np.unravel_index(np.asarray(self.policy_aprime), shape))


@dataclass(frozen=True)
class FiniteHorizonProblem:
    """Life-cycle VFI problem solved by backward induction over ``horizon`` ages.

    ``return_fn`` is age-dependent (see module docstring). ``terminal_value`` is
    the (n_a, n_z) continuation value after the last age (default zeros).
    ``survival`` is an optional length-``horizon`` array of conditional survival
    probabilities ``s_j`` in (0, 1]: the effective discount from age ``j`` to
    ``j+1`` is ``beta * s_j`` (mortality risk; default all ones = no mortality).
    ``options`` is currently unused (reserved). Solve on numpy/mlx/cupy.
    """
    a_grid: np.ndarray
    z_grid: np.ndarray
    P_z: np.ndarray
    return_fn: Callable
    beta: float
    horizon: int
    params: dict = field(default_factory=dict)
    d_grid: np.ndarray | None = None
    terminal_value: np.ndarray | None = None
    survival: np.ndarray | None = None
    options: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 < self.beta < 1.0):
            raise ValueError(f"beta must be in (0,1); got {self.beta}")
        if int(self.horizon) < 1:
            raise ValueError(f"horizon must be >= 1; got {self.horizon}")
        if "age" in self.params:
            raise ValueError(
                '"age" is reserved: the finite-horizon solver injects the age '
                "as the first return-fn parameter, so it must not appear in params"
            )
        z = np.asarray(self.z_grid)
        P = np.asarray(self.P_z)
        if isinstance(self.a_grid, (list, tuple)):
            grids = [np.asarray(g) for g in self.a_grid]
            if len(grids) < 1:
                raise ValueError("a_grid list must contain >= 1 endogenous grid")
            for g in grids:
                if g.ndim != 1 or g.size < 1:
                    raise ValueError("each endogenous grid must be 1-D with >= 1 point")
            n_a_total = int(np.prod([g.size for g in grids]))
            if n_a_total < 2:
                raise ValueError("the product endogenous grid must have >= 2 points")
        else:
            a = np.asarray(self.a_grid)
            if a.ndim != 1 or a.size < 2:
                raise ValueError("a_grid must be 1-D with >= 2 points")
            n_a_total = int(a.size)
        if z.ndim != 1 or z.size < 1:
            raise ValueError("z_grid must be 1-D with >= 1 points")
        if P.shape != (z.size, z.size):
            raise ValueError(f"P_z must be ({z.size},{z.size}); got {P.shape}")
        if not np.allclose(P.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("P_z rows must sum to 1")
        if self.terminal_value is not None:
            tv = np.asarray(self.terminal_value)
            if tv.shape != (n_a_total, z.size):
                raise ValueError(
                    f"terminal_value must be ({n_a_total},{z.size}); got {tv.shape}"
                )
        if self.survival is not None:
            s = np.asarray(self.survival)
            if s.shape != (int(self.horizon),):
                raise ValueError(
                    f"survival must have shape ({int(self.horizon)},); got {s.shape}"
                )
            if np.any(s <= 0.0) or np.any(s > 1.0):
                raise ValueError("survival probabilities must lie in (0, 1]")

    def solve(self, backend: str = "numpy") -> FiniteHorizonSolution:
        if backend == "numba":
            raise ValueError(
                "finite-horizon solve supports numpy/mlx/cupy; numba is deferred"
            )
        xp = _bk.get_array_namespace(backend)
        z = np.asarray(self.z_grid, dtype=float)
        n_z = z.size
        if isinstance(self.a_grid, (list, tuple)):
            grids = [np.asarray(g, dtype=float) for g in self.a_grid]
            endo_shape = tuple(int(g.size) for g in grids)
            a_arg = grids
        else:
            a_arr = np.asarray(self.a_grid, dtype=float)
            endo_shape = (int(a_arr.size),)
            a_arg = a_arr
        n_a = int(np.prod(endo_shape))
        J = int(self.horizon)
        d = None if self.d_grid is None else np.asarray(self.d_grid, dtype=float)
        surv = (np.ones(J) if self.survival is None
                else np.asarray(self.survival, dtype=float))
        Pt = xp.array(np.asarray(self.P_z, dtype=float)).T
        if self.terminal_value is None:
            V_next = xp.zeros((n_a, n_z))
        else:
            V_next = xp.array(np.asarray(self.terminal_value, dtype=float))

        Vs: list = [None] * J
        flats: list = [None] * J
        for j in range(J - 1, -1, -1):
            EV = V_next @ Pt
            R_j = build_return_tensor(
                self.return_fn, a_arg, z, {"age": j, **self.params}, d_grid=d, xp=xp
            )
            V_j, flat_j = bellman_step(R_j, EV, float(self.beta) * float(surv[j]), xp=xp)
            Vs[j] = V_j
            flats[j] = flat_j
            V_next = V_j

        V = np.stack([_bk.to_numpy(v) for v in Vs], axis=0)             # (J,n_a,n_z)
        flat = np.stack([_bk.to_numpy(f) for f in flats], axis=0).astype(np.int64)
        n_ap = n_a
        policy_aprime = flat % n_ap
        policy_d = (flat // n_ap) if d is not None else None
        return FiniteHorizonSolution(
            V=V, policy_aprime=policy_aprime, policy_d=policy_d, horizon=J,
            endo_shape=endo_shape
        )


def _z_stationary(P_z):
    P = np.asarray(P_z, dtype=float)
    w, V = np.linalg.eig(P.T)
    v = np.real(V[:, np.argmin(np.abs(w - 1.0))])
    return v / v.sum()


def life_cycle_distribution(solution, P_z, newborn_dist=None):
    """Age-indexed cohort distribution (J, n_a, n_z) for a solved life-cycle model.

    Cohorts are born at age 0 with ``newborn_dist`` (default: all mass at the
    lowest asset, z ~ the z-stationary distribution of P_z) and age
    deterministically: mu_{j+1} = push_distribution(mu_j, policy_j, P_z). Each
    age-slice is a probability measure (cohort mass 1).
    """
    pol = np.asarray(solution.policy_aprime)
    J, n_a, n_z = pol.shape
    P = np.asarray(P_z, dtype=float)
    if newborn_dist is None:
        mu0 = np.zeros((n_a, n_z))
        mu0[0, :] = _z_stationary(P)
    else:
        mu0 = np.asarray(newborn_dist, dtype=float)
        if mu0.shape != (n_a, n_z):
            raise ValueError(f"newborn_dist must be ({n_a},{n_z}); got {mu0.shape}")
        mu0 = mu0 / mu0.sum()
    dist = np.empty((J, n_a, n_z))
    dist[0] = mu0
    for j in range(J - 1):
        dist[j + 1] = push_distribution(dist[j], pol[j], P)
    return dist


def cross_section(life_cycle_dist, age_weights=None):
    """Cross-sectional distribution = sum_j w_j * mu_j (default uniform cohorts)."""
    d = np.asarray(life_cycle_dist, dtype=float)
    J = d.shape[0]
    if age_weights is None:
        w = np.full(J, 1.0 / J)
    else:
        w = np.asarray(age_weights, dtype=float)
        if w.shape != (J,):
            raise ValueError(f"age_weights must have length {J}; got {w.shape}")
        w = w / w.sum()
    return np.tensordot(w, d, axes=(0, 0))


def age_profile(life_cycle_dist, a_grid):
    """Mean assets by age: profile[j] = sum_{a,z} mu_j(a,z) * a_grid[a]."""
    d = np.asarray(life_cycle_dist, dtype=float)
    a = np.asarray(a_grid, dtype=float)
    return np.einsum("jaz,a->j", d, a)


__all__ = [
    "FiniteHorizonProblem",
    "FiniteHorizonSolution",
    "life_cycle_distribution",
    "cross_section",
    "age_profile",
]
