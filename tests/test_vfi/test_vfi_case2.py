from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi import VFIProblem
from puremacro.vfi.case2 import Case2Problem, Case2Solution


def test_reduces_to_case1():
    # Case 2 with d_grid == a_grid and Phi(d_idx,a,z) = d_idx is exactly Case 1
    # (the decision IS the chosen next asset). V and policy must match Case 1.
    a_grid = np.linspace(1e-3, 12.0, 40)
    z_grid, P = np.array([-0.2, 0.0, 0.2]), np.array(
        [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]]
    )
    beta = 0.95

    def util(c, xp):
        return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -1e10)

    # Case 1: choose a'
    def rf1(ap, a, z, xp=np):
        return util(np.exp(z) + 1.04 * a - ap, xp)

    sol1 = VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf1, beta=beta,
                      options=dict(tol=1e-10, n_howard=40)).solve("numpy")

    # Case 2: choose d (= the next-asset index); a' = Phi(d) = d
    def rf2(d, a, z, xp=np):
        # d is the next-asset VALUE here (d_grid == a_grid), so consumption uses d as a'
        return util(np.exp(z) + 1.04 * a - d, xp)

    def phi(di, ai, zi):
        return di + 0 * ai + 0 * zi          # a'-index = decision index

    sol2 = Case2Problem(a_grid=a_grid, z_grid=z_grid, P_z=P, d_grid=a_grid,
                        return_fn=rf2, aprime_fn=phi, beta=beta,
                        options=dict(tol=1e-10, n_howard=40)).solve()

    assert isinstance(sol2, Case2Solution)
    np.testing.assert_allclose(sol2.V, sol1.V, atol=1e-7)
    # the chosen next-asset index matches (policy_d indexes d_grid==a_grid == a'-index)
    np.testing.assert_array_equal(sol2.policy_d, sol1.policy_aprime)


def test_nontrivial_phi_runs():
    # a' = min(a_idx + d_idx, n_a-1): each decision moves up the asset grid by d steps
    n_a = 20
    a_grid = np.linspace(0.0, 9.5, n_a)
    z_grid, P = np.array([0.0]), np.array([[1.0]])
    d_grid = np.array([0.0, 1.0, 2.0])     # deposit 0, 1, or 2 grid steps

    def rf(d, a, z, xp=np):
        # cost of depositing d steps; reward log-consumption of income minus deposit cost
        c = 2.0 + 0.0 * a + 0.0 * z - 0.3 * d
        return xp.log(xp.maximum(c, 1e-12)) + 0.01 * a   # mild reward for assets

    def phi(di, ai, zi):
        nxt = ai + di
        return np.minimum(nxt, n_a - 1)        # cap at the top asset index

    sol = Case2Problem(a_grid=a_grid, z_grid=z_grid, P_z=P, d_grid=d_grid,
                       return_fn=rf, aprime_fn=phi, beta=0.95,
                       options=dict(tol=1e-10)).solve()
    assert sol.V.shape == (n_a, 1)
    assert sol.policy_d.shape == (n_a, 1)
    assert np.all(np.isfinite(sol.V))
    assert sol.policy_d.min() >= 0 and sol.policy_d.max() < len(d_grid)


def test_howard_matches_pure():
    a_grid = np.linspace(1e-3, 8.0, 25)
    z_grid, P = np.array([0.0, 0.3]), np.array([[0.8, 0.2], [0.3, 0.7]])

    def rf(d, a, z, xp=np):
        c = np.exp(z) + a - d
        return xp.where(c > 0, xp.log(xp.maximum(c, 1e-12)), -1e10)

    def phi(di, ai, zi):
        return di + 0 * ai + 0 * zi

    kw = dict(a_grid=a_grid, z_grid=z_grid, P_z=P, d_grid=a_grid, return_fn=rf,
              aprime_fn=phi, beta=0.95)
    vp = Case2Problem(**kw, options=dict(howard=False, tol=1e-11)).solve()
    vh = Case2Problem(**kw, options=dict(howard=True, n_howard=30, tol=1e-11)).solve()
    np.testing.assert_allclose(vp.V, vh.V, atol=1e-7)


def test_validation():
    a = np.linspace(0, 1, 3); z = np.array([0.0]); P = np.array([[1.0]])
    rf = lambda d, a, z, xp=np: 0.0 * (d + a + z)
    phi = lambda di, ai, zi: di + 0 * ai + 0 * zi
    with pytest.raises(ValueError, match="beta"):
        Case2Problem(a_grid=a, z_grid=z, P_z=P, d_grid=a, return_fn=rf,
                     aprime_fn=phi, beta=1.0)
    with pytest.raises(ValueError, match="rows must sum"):
        Case2Problem(a_grid=a, z_grid=z, P_z=np.array([[0.5]]), d_grid=a,
                     return_fn=rf, aprime_fn=phi, beta=0.9)


def test_case2_exported():
    from puremacro.vfi import Case2Problem as P
    from puremacro.vfi import Case2Solution as S

    assert P is Case2Problem
    assert S is Case2Solution
