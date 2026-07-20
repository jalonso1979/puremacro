from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi import (
    VFIProblem,
    combine_markov_chains,
    markov_stationary,
    stationary_distribution,
    tauchen,
)
from puremacro.vfi.aggregate import aggregate


def test_exact_reduction_trivial_second_shock():
    # a 2-shock problem whose 2nd shock is a single trivial point reduces to the
    # 1-shock problem (P_combined = kron(P1, [[1]]) = P1).
    z_grid, P = tauchen(5, 0.9, 0.2)
    a_grid = np.linspace(0.0, 30.0, 60)
    beta = 0.95
    opt = dict(tol=1e-10, n_howard=30)

    def rf1(ap, a, z, xp=np):
        c = 1.03 * a + np.exp(z) - ap
        return xp.where(c > 0.0, -1.0 / xp.maximum(c, 1e-12), -np.inf)

    s1 = VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf1,
                    beta=beta, options=opt).solve("numpy")

    z2 = np.array([0.0])
    _, Pc = combine_markov_chains((z_grid, P), (z2, np.array([[1.0]])))

    def rf2(ap, a, z1, z2_, xp=np):                 # 2nd shock trivial (z2=0)
        c = 1.03 * a + np.exp(z1) - ap
        return xp.where(c > 0.0, -1.0 / xp.maximum(c, 1e-12), -np.inf)

    s2 = VFIProblem(a_grid=a_grid, z_grid=[z_grid, z2], P_z=Pc, return_fn=rf2,
                    beta=beta, options=opt).solve("numpy")
    np.testing.assert_allclose(s2.V.reshape(60, 5), s1.V, atol=1e-9)
    np.testing.assert_array_equal(s2.policy_aprime.reshape(60, 5), s1.policy_aprime)


def test_genuine_two_shock_solve_and_distribution():
    # persistent income shock z1 + a separate rate shock z2 entering the budget
    # differently. Combined chain via combine_markov_chains.
    z1g, P1 = tauchen(4, 0.9, 0.2)          # log income
    z2g, P2 = tauchen(3, 0.6, 0.05)         # interest-rate shock (around mean 0)
    _, Pc = combine_markov_chains((z1g, P1), (z2g, P2))
    a_grid = np.linspace(0.0, 40.0, 50)

    def rf(ap, a, z1, z2, xp=np):
        r = 0.03 + z2                       # rate shock shifts the return
        c = (1.0 + r) * a + np.exp(z1) - ap
        return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -np.inf)

    sol = VFIProblem(a_grid=a_grid, z_grid=[z1g, z2g], P_z=Pc, return_fn=rf,
                     beta=0.95, options=dict(tol=1e-9, n_howard=30)).solve("numpy")
    assert sol.V.shape == (50, 12)          # n_z = 4*3 = 12 flat exogenous states
    mu = stationary_distribution(sol.policy_aprime, Pc)
    np.testing.assert_allclose(mu.sum(), 1.0, atol=1e-10)
    np.testing.assert_allclose(mu.sum(axis=0), markov_stationary(Pc), atol=1e-8)

    # aggregate with a z-component-dependent eval fn (mean income exp(z1))
    def income(ap, a, z1, z2, xp=np):
        return np.exp(z1) + 0.0 * a
    Y = aggregate(income, mu, sol.policy_aprime, a_grid, [z1g, z2g])
    z1v, z2v = np.meshgrid(z1g, z2g, indexing="ij")
    direct = float(np.sum(mu * np.exp(z1v.reshape(-1))[None, :]))
    assert Y == pytest.approx(direct)


def test_multi_shock_validation():
    z1 = np.array([0.0, 1.0])
    z2 = np.array([0.0, 1.0])               # flat n_z = 4
    a_grid = np.linspace(0.0, 5.0, 6)

    def rf(ap, a, z1_, z2_, xp=np):
        return -ap

    with pytest.raises(ValueError, match="P_z"):     # P_z must be (4,4)
        VFIProblem(a_grid=a_grid, z_grid=[z1, z2], P_z=np.eye(2), return_fn=rf,
                   beta=0.95)
