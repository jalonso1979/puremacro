from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi import VFIProblem, tauchen
from puremacro.vfi.egm import EGMSolution, solve_egm

ALPHA = None  # unused; keep imports tidy


def _setup(n_a=300, n_z=5, gamma=2.0, r=0.03, rho=0.9, sigma=0.2):
    z_grid, P = tauchen(n=n_z, rho=rho, sigma=sigma)
    income = np.exp(z_grid)                       # y(z) = exp(z)
    a_grid = np.linspace(0.0, 60.0, n_a)          # borrowing constraint at 0
    return a_grid, z_grid, P, income, gamma, r


def test_shapes_and_budget_and_monotone():
    a_grid, z_grid, P, income, gamma, r = _setup()
    sol = solve_egm(a_grid, z_grid, income, P, beta=0.96, r=r, gamma=gamma)
    assert isinstance(sol, EGMSolution)
    assert sol.c.shape == (len(a_grid), len(z_grid))
    assert sol.aprime.shape == sol.c.shape
    # budget holds exactly: c + a' = (1+r)a + y(z)
    coh = (1.0 + r) * a_grid[:, None] + income[None, :]
    np.testing.assert_allclose(sol.c + sol.aprime, coh, atol=1e-9)
    assert np.all(sol.c > 0.0)
    assert np.all(sol.aprime >= a_grid[0] - 1e-12)            # respects borrowing constraint
    assert np.all(np.diff(sol.c, axis=0) >= -1e-9)            # consumption rises with assets
    assert np.all(np.diff(sol.aprime, axis=0) >= -1e-9)       # savings rise with assets


def test_constrained_at_low_assets():
    # at a=0 (lowest), low-income agents are borrowing-constrained: a' = a_min
    a_grid, z_grid, P, income, gamma, r = _setup()
    sol = solve_egm(a_grid, z_grid, income, P, beta=0.96, r=r, gamma=gamma)
    # the lowest-income, lowest-asset agent should be at (or essentially at) a_min
    assert sol.aprime[0, 0] == pytest.approx(a_grid[0], abs=1e-8)
    assert sol.aprime[0, 1] == pytest.approx(a_grid[0], abs=1e-8)


def test_euler_residual_unconstrained_small():
    # at the EGM solution the Euler eq c^-g = beta(1+r) E[c'(a',z')^-g] holds where
    # a' is interior (unconstrained). Interpolate next-period c at the implied a'.
    a_grid, z_grid, P, income, gamma, r = _setup()
    beta = 0.96
    sol = solve_egm(a_grid, z_grid, income, P, beta=beta, r=r, gamma=gamma)
    n_a, n_z = sol.c.shape
    a_min = a_grid[0]
    max_resid = 0.0
    for ai in range(n_a):
        for zi in range(n_z):
            ap = sol.aprime[ai, zi]
            if ap <= a_min + 1e-6 or ap >= a_grid[-1] - 5.0:
                continue                            # skip constrained/near-top states
            lhs = sol.c[ai, zi] ** (-gamma)
            # next-period consumption at a' for each z', interpolated
            cprime = np.array([np.interp(ap, a_grid, sol.c[:, zp]) for zp in range(n_z)])
            rhs = beta * (1.0 + r) * float(P[zi] @ (cprime ** (-gamma)))
            max_resid = max(max_resid, abs(lhs - rhs) / lhs)
    assert max_resid < 1e-3                                    # Euler holds in the interior


def test_agrees_with_discrete_vfi_interior():
    # EGM (continuous) must agree with a fine discrete VFI of the SAME problem on
    # the interior asset range (discretization-limited).
    a_grid, z_grid, P, income, gamma, r = _setup(n_a=400)
    beta = 0.96
    sol = solve_egm(a_grid, z_grid, income, P, beta=beta, r=r, gamma=gamma)

    def rf(ap, a, z, xp=np):
        c = (1.0 + r) * a + np.exp(z) - ap
        # CRRA gamma=2: u = -1/c (monotone increasing); -inf if infeasible
        return xp.where(c > 0.0, -1.0 / xp.maximum(c, 1e-12), -np.inf)

    vsol = VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf, beta=beta,
                      options=dict(tol=1e-10, n_howard=50)).solve("numpy")
    c_vfi = (1.0 + r) * a_grid[:, None] + income[None, :] - a_grid[vsol.policy_aprime]
    # compare consumption on the interior (away from both grid edges)
    mask = (a_grid > 2.0) & (a_grid < 50.0)
    rel = np.abs(sol.c[mask] - c_vfi[mask]) / np.abs(c_vfi[mask])
    assert np.median(rel) < 0.02              # typically <2% (EGM is the accurate one)
    assert np.max(rel) < 0.10                 # no interior point off by >10%


def test_validation():
    a_grid, z_grid, P, income, gamma, r = _setup(n_a=10, n_z=3)
    with pytest.raises(ValueError, match="beta"):
        solve_egm(a_grid, z_grid, income, P, beta=1.0, r=r, gamma=gamma)
    with pytest.raises(ValueError, match="gamma"):
        solve_egm(a_grid, z_grid, income, P, beta=0.96, r=r, gamma=0.0)
    with pytest.raises(ValueError, match="income"):
        solve_egm(a_grid, z_grid, income[:-1], P, beta=0.96, r=r, gamma=gamma)
    with pytest.raises(ValueError, match="r must be"):
        solve_egm(a_grid, z_grid, income, P, beta=0.96, r=-1.0, gamma=gamma)
    with pytest.raises(ValueError, match="strictly increasing"):
        solve_egm(np.array([0.0, 2.0, 1.0, 3.0]), z_grid, income, P, beta=0.96, r=r, gamma=gamma)
    P_neg = np.array([[-0.1, 1.1, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
    with pytest.raises(ValueError, match="row-stochastic"):
        solve_egm(a_grid, z_grid, income, P_neg, beta=0.96, r=r, gamma=gamma)


def test_egm_exported():
    from puremacro.vfi import EGMSolution as S
    from puremacro.vfi import solve_egm as fn

    assert fn is solve_egm
    assert S is EGMSolution
