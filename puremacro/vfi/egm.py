"""Endogenous Grid Method (Carroll 2006) for the one-asset income-fluctuation problem.

Solves  max E sum beta^t u(c),  u(c) = c^(1-gamma)/(1-gamma),  subject to
    c + a' = (1+r) a + y(z),    a' >= a_grid[0]   (borrowing constraint),
with z an AR(1) Markov shock (P_z) and income y(z). EGM inverts the Euler
equation on a grid of NEXT-period assets instead of maximising over a' on a
discrete grid -- O(n) per iteration, and the policy is continuous (interpolated)
rather than pinned to grid nodes. Returns continuous c(a,z) and a'(a,z) policies.
numpy-only (the EGM interpolation is not a GPU/backend path).

Utility is CRRA by default. Any strictly increasing, strictly concave felicity
works by passing its marginal utility ``u_prime`` and the inverse
``u_prime_inv`` (vectorised callables) instead of ``gamma``; ``puremacro.dp``
derives both from a symbolic reward when it compiles a model to EGM.
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


def solve_egm(a_grid, z_grid, income, P_z, *, beta, r, gamma=None,
              u_prime=None, u_prime_inv=None, c0=None,
              tol: float = 1e-9, max_iter: int = 10_000):
    """EGM solve of the income-fluctuation problem; see module docstring.

    ``income`` is y(z), shape (n_z,). Utility is CRRA with ``gamma`` unless both
    ``u_prime`` (c -> u'(c)) and ``u_prime_inv`` (u' -> c) are given. ``c0`` is an
    optional (n_a, n_z) initial consumption guess (warm start). Returns an
    EGMSolution with continuous policies on ``a_grid``. (The top of ``a_grid``
    should be high enough that the saving policy is interior there; np.interp
    clamps above the endogenous range.)
    """
    a = np.asarray(a_grid, dtype=float)
    inc = np.asarray(income, dtype=float)
    P = np.asarray(P_z, dtype=float)
    n_a, n_z = a.size, np.asarray(z_grid).shape[0]
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0,1); got {beta}")
    if r <= -1.0:
        raise ValueError(f"r must be > -1; got {r}")
    if (u_prime is None) != (u_prime_inv is None):
        raise ValueError("pass both u_prime and u_prime_inv, or neither (CRRA with gamma)")
    if u_prime is None:
        if gamma is None or gamma <= 0.0:
            raise ValueError(f"gamma must be > 0; got {gamma}")
        g = float(gamma)

        def u_prime(c):
            return c ** (-g)

        def u_prime_inv(x):
            return x ** (-1.0 / g)
    if inc.shape != (n_z,):
        raise ValueError(f"income must have shape ({n_z},); got {inc.shape}")
    if not np.all(np.diff(a) > 0):
        raise ValueError("a_grid must be strictly increasing")
    if P.shape != (n_z, n_z) or not np.allclose(P.sum(axis=1), 1.0, atol=1e-8) or np.any(P < -1e-12):
        raise ValueError("P_z must be (n_z,n_z) and row-stochastic")
    R = 1.0 + r
    a_min = a[0]
    coh = R * a[:, None] + inc[None, :]                  # cash-on-hand (n_a, n_z)
    if c0 is None:
        c = np.maximum(coh - a_min, 1e-10)               # initial guess: save the minimum
    else:
        c = np.maximum(np.asarray(c0, dtype=float), 1e-10)
        if c.shape != (n_a, n_z):
            raise ValueError(f"c0 must have shape ({n_a}, {n_z}); got {c.shape}")
    sup = np.inf
    for it in range(1, max_iter + 1):
        Emu = u_prime(c) @ P.T                          # E_{z'|z}[u'(c')], indexed (a', z)
        rhs = beta * R * Emu
        c_endog = u_prime_inv(rhs)                       # current c if next assets = a-node
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
