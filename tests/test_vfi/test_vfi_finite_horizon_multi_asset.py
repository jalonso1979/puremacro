from __future__ import annotations

import numpy as np

from puremacro.vfi import tauchen
from puremacro.vfi.finite_horizon import (
    FiniteHorizonProblem,
    life_cycle_distribution,
)


def test_single_asset_endo_shape_unchanged():
    z_grid, P = tauchen(3, 0.8, 0.2)
    a_grid = np.linspace(0.0, 20.0, 25)

    def rf(ap, a, z, age, r, xp=np):
        c = (1.0 + r) * a + np.exp(z) - ap
        return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -np.inf)

    sol = FiniteHorizonProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                               beta=0.96, horizon=10, params={"r": 0.03}).solve("numpy")
    assert sol.endo_shape == (25,)
    assert sol.V.shape == (10, 25, 3)
    comps = sol.policy_components()
    assert len(comps) == 1
    np.testing.assert_array_equal(comps[0], sol.policy_aprime)


def test_exact_reduction_trivial_second_asset():
    z_grid, P = tauchen(5, 0.9, 0.2)
    a_grid = np.linspace(0.0, 30.0, 60)
    J = 12

    def rf1(ap, a, z, age, r, xp=np):
        c = (1.0 + r) * a + np.exp(z) - ap
        return xp.where(c > 0.0, -1.0 / xp.maximum(c, 1e-12), -np.inf)

    s1 = FiniteHorizonProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf1,
                              beta=0.95, horizon=J, params={"r": 0.03}).solve("numpy")

    def rf2(m_p, k_p, m, k, z, age, r, xp=np):
        c = (1.0 + r) * m + np.exp(z) - m_p
        return xp.where(c > 0.0, -1.0 / xp.maximum(c, 1e-12), -np.inf)

    s2 = FiniteHorizonProblem(a_grid=[a_grid, np.array([0.0])], z_grid=z_grid, P_z=P,
                              return_fn=rf2, beta=0.95, horizon=J,
                              params={"r": 0.03}).solve("numpy")
    np.testing.assert_allclose(s2.V, s1.V, atol=1e-9)          # identical age-indexed value
    assert s2.endo_shape == (60, 1)
    m_pol, k_pol = s2.policy_components()
    np.testing.assert_array_equal(m_pol, s1.policy_aprime)     # asset-1 policy identical
    np.testing.assert_array_equal(k_pol, np.zeros_like(k_pol))


def test_genuine_two_asset_life_cycle():
    # liquid + illiquid over the life cycle, no bequest: shapes, cohort mass,
    # terminal spend-down, and per-asset decode.
    z_grid, P = tauchen(3, 0.9, 0.2)
    J = 20
    m_grid = np.linspace(0.0, 12.0, 13)
    k_grid = np.linspace(0.0, 12.0, 13)
    ages = np.arange(J)
    kappa = np.exp(0.1 * ages - 0.004 * ages ** 2)            # hump earnings

    def rf(m_p, k_p, m, k, z, age, xp=np):
        c = 1.01 * m + 1.05 * k + kappa[age] * np.exp(z) - m_p - k_p
        return xp.where(c > 0.0, -1.0 / xp.maximum(c, 1e-12), -np.inf)

    sol = FiniteHorizonProblem(a_grid=[m_grid, k_grid], z_grid=z_grid, P_z=P,
                               return_fn=rf, beta=0.96, horizon=J).solve("numpy")
    assert sol.endo_shape == (13, 13)
    assert sol.V.shape == (J, 169, 3)
    # no bequest: at the last age the agent saves nothing (flat index 0 = (m'=0,k'=0))
    assert np.all(sol.policy_aprime[-1] == 0)
    m_pol, k_pol = sol.policy_components()
    assert m_pol.shape == (J, 169, 3) and k_pol.shape == (J, 169, 3)
    assert m_pol.max() < 13 and k_pol.max() < 13

    dist = life_cycle_distribution(sol, P)
    assert dist.shape == (J, 169, 3)
    np.testing.assert_allclose(dist.sum(axis=(1, 2)), np.ones(J), atol=1e-10)


def test_terminal_value_shape_validation_multi():
    import pytest

    z_grid, P = tauchen(2, 0.5, 0.2)

    def rf(m_p, k_p, m, k, z, age, xp=np):
        return -m_p - k_p

    # product n_a = 3*2 = 6; a wrong-shaped terminal_value must raise
    with pytest.raises(ValueError, match="terminal_value"):
        FiniteHorizonProblem(a_grid=[np.linspace(0, 1, 3), np.array([0.0, 1.0])],
                             z_grid=z_grid, P_z=P, return_fn=rf, beta=0.95,
                             horizon=3, terminal_value=np.zeros((3, 2)))
