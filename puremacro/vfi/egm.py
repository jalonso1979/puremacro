"""Endogenous Grid Method (Carroll 2006) for the one-asset income-fluctuation problem.

Solves  max E sum beta^t u(c),  u(c) = c^(1-gamma)/(1-gamma),  subject to
    c + a' = (1+r) a + y(z),    a' >= a_grid[0]   (borrowing constraint),
with z an AR(1) Markov shock (P_z) and income y(z). EGM inverts the Euler
equation on a grid of NEXT-period assets instead of maximising over a' on a
discrete grid -- O(n) per iteration, and the policy is continuous (interpolated)
rather than pinned to grid nodes. Returns continuous c(a,z) and a'(a,z) policies.
numpy-only (the EGM interpolation is not a GPU/backend path).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EGMSolution:
    """Continuous consumption and next-asset policies from EGM (numpy)."""
    c: np.ndarray            # (n_a, n_z) consumption policy
    aprime: np.ndarray       # (n_a, n_z) next-asset policy (values, >= a_grid[0])
    n_iter: int
    sup_norm: float


def solve_egm(a_grid, z_grid, income, P_z, *, beta, r, gamma,
              tol: float = 1e-9, max_iter: int = 10_000):
    """EGM solve of the CRRA income-fluctuation problem; see module docstring.

    ``income`` is y(z), shape (n_z,). Returns an EGMSolution with continuous
    policies on ``a_grid``. (The top of ``a_grid`` should be high enough that the
    saving policy is interior there; np.interp clamps above the endogenous range.)
    """
    a = np.asarray(a_grid, dtype=float)
    inc = np.asarray(income, dtype=float)
    P = np.asarray(P_z, dtype=float)
    n_a, n_z = a.size, np.asarray(z_grid).shape[0]
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0,1); got {beta}")
    if r <= -1.0:
        raise ValueError(f"r must be > -1; got {r}")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0; got {gamma}")
    if inc.shape != (n_z,):
        raise ValueError(f"income must have shape ({n_z},); got {inc.shape}")
    if not np.all(np.diff(a) > 0):
        raise ValueError("a_grid must be strictly increasing")
    if P.shape != (n_z, n_z) or not np.allclose(P.sum(axis=1), 1.0, atol=1e-8) or np.any(P < -1e-12):
        raise ValueError("P_z must be (n_z,n_z) and row-stochastic")
    R = 1.0 + r
    a_min = a[0]
    coh = R * a[:, None] + inc[None, :]                  # cash-on-hand (n_a, n_z)
    c = np.maximum(coh - a_min, 1e-10)                   # initial guess: save the minimum
    sup = np.inf
    for it in range(1, max_iter + 1):
        Emu = (c ** (-gamma)) @ P.T                      # E_{z'|z}[u'(c')], indexed (a', z)
        rhs = beta * R * Emu
        c_endog = rhs ** (-1.0 / gamma)                  # current c if next assets = a-node
        a_endog = (c_endog + a[:, None] - inc[None, :]) / R   # endogenous current assets
        c_new = np.empty((n_a, n_z))
        for zi in range(n_z):
            ae = a_endog[:, zi]
            ce = c_endog[:, zi]
            c_un = np.interp(a, ae, ce)                  # unconstrained (clamped outside)
            c_con = coh[:, zi] - a_min                   # constrained: a' = a_min
            c_new[:, zi] = np.where(a <= ae[0], c_con, c_un)
        c_new = np.maximum(c_new, 1e-10)
        sup = float(np.max(np.abs(c_new - c)))
        c = c_new
        if sup < tol:
            break
    else:
        raise RuntimeError(
            f"solve_egm did not converge in {max_iter} iterations "
            f"(sup-norm {sup:.3e} > tol {tol:.1e})"
        )
    aprime = np.clip(coh - c, a_min, a[-1])
    c = coh - aprime   # re-impose the budget after clamping a' to the grid range
    return EGMSolution(c=c, aprime=aprime, n_iter=it, sup_norm=sup)


__all__ = ["solve_egm", "EGMSolution"]
