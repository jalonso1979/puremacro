from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi import (
    VFIProblem,
    markov_stationary,
    stationary_distribution,
    tauchen,
)


def test_single_asset_endo_shape_and_decode():
    z_grid, P = tauchen(3, 0.8, 0.2)
    a_grid = np.linspace(0.0, 20.0, 30)

    def rf(ap, a, z, xp=np):
        c = 1.03 * a + np.exp(z) - ap
        return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -np.inf)

    sol = VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                     beta=0.95, options=dict(tol=1e-9, n_howard=30)).solve("numpy")
    assert sol.endo_shape == (30,)
    comps = sol.policy_components()
    assert len(comps) == 1
    np.testing.assert_array_equal(comps[0], sol.policy_aprime)


def test_exact_reduction_trivial_second_asset():
    z_grid, P = tauchen(5, 0.9, 0.2)
    a_grid = np.linspace(0.0, 40.0, 80)
    beta = 0.95
    opt = dict(tol=1e-10, n_howard=30)

    def rf1(ap, a, z, xp=np):
        c = 1.03 * a + np.exp(z) - ap
        return xp.where(c > 0.0, -1.0 / xp.maximum(c, 1e-12), -np.inf)

    s1 = VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf1,
                    beta=beta, options=opt).solve("numpy")

    def rf2(m_p, k_p, m, k, z, xp=np):           # k is a trivial single point {0}
        c = 1.03 * m + np.exp(z) - m_p
        return xp.where(c > 0.0, -1.0 / xp.maximum(c, 1e-12), -np.inf)

    s2 = VFIProblem(a_grid=[a_grid, np.array([0.0])], z_grid=z_grid, P_z=P,
                    return_fn=rf2, beta=beta, options=opt).solve("numpy")
    np.testing.assert_allclose(s2.V, s1.V, atol=1e-9)        # identical value
    assert s2.endo_shape == (80, 1)
    m_pol, k_pol = s2.policy_components()
    np.testing.assert_array_equal(m_pol, s1.policy_aprime)   # asset-1 policy identical
    np.testing.assert_array_equal(k_pol, np.zeros_like(k_pol))


def test_perfect_substitutes_value_matches_one_asset():
    # two assets, same gross return 1.03, no friction: only total wealth matters.
    z_grid, P = tauchen(3, 0.8, 0.2)
    beta = 0.95
    opt = dict(tol=1e-10, n_howard=40)
    grid = np.array([0.0, 1.0, 2.0, 3.0])         # m,k each in {0,1,2,3}
    W_grid = np.arange(7.0)                        # achievable totals {0..6}

    def rf1(Wp, W, z, xp=np):
        c = 1.03 * W + np.exp(z) - Wp
        return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -np.inf)

    s1 = VFIProblem(a_grid=W_grid, z_grid=z_grid, P_z=P, return_fn=rf1,
                    beta=beta, options=opt).solve("numpy")

    def rf2(m_p, k_p, m, k, z, xp=np):
        c = 1.03 * (m + k) + np.exp(z) - (m_p + k_p)
        return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -np.inf)

    s2 = VFIProblem(a_grid=[grid, grid], z_grid=z_grid, P_z=P, return_fn=rf2,
                    beta=beta, options=opt).solve("numpy")
    assert s2.endo_shape == (4, 4)
    # V_2asset(m,k) == V_1asset(m+k) (the split is irrelevant; total wealth governs)
    for mi in range(4):
        for ki in range(4):
            flat = mi * 4 + ki                     # C-order, k fastest
            W_idx = mi + ki                        # grid is 0,1,2,3 => value==index
            np.testing.assert_allclose(s2.V[flat], s1.V[W_idx], atol=1e-9)


def test_genuine_two_asset_solve_sane():
    z_grid, P = tauchen(3, 0.8, 0.2)
    beta, r_m, r_k = 0.96, 0.01, 0.05
    m_grid = np.linspace(0.0, 15.0, 16)
    k_grid = np.linspace(0.0, 15.0, 16)

    def rf(m_p, k_p, m, k, z, xp=np):
        c = (1.0 + r_m) * m + (1.0 + r_k) * k + np.exp(z) - m_p - k_p
        return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -np.inf)

    sol = VFIProblem(a_grid=[m_grid, k_grid], z_grid=z_grid, P_z=P, return_fn=rf,
                     beta=beta, options=dict(tol=1e-9, n_howard=30)).solve("numpy")
    assert sol.V.shape == (256, 3)
    assert sol.endo_shape == (16, 16)
    m_pol, k_pol = sol.policy_components()
    assert m_pol.shape == (256, 3) and k_pol.shape == (256, 3)
    assert m_pol.min() >= 0 and m_pol.max() < 16
    assert k_pol.min() >= 0 and k_pol.max() < 16
    assert k_pol.max() > 0                         # the high-return asset is used

    mu = stationary_distribution(sol.policy_aprime, P)
    assert mu.shape == (256, 3)
    np.testing.assert_allclose(mu.sum(), 1.0, atol=1e-9)
    np.testing.assert_allclose(mu.sum(axis=0), markov_stationary(P), atol=1e-8)


def test_two_asset_numba_matches_numpy():
    pytest.importorskip("numba")
    z_grid, P = tauchen(3, 0.8, 0.2)
    beta = 0.96
    m_grid = np.linspace(0.0, 12.0, 13)
    k_grid = np.linspace(0.0, 12.0, 13)

    def rf(m_p, k_p, m, k, z, xp=np):
        c = 1.02 * m + 1.05 * k + np.exp(z) - m_p - k_p
        return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -np.inf)

    kw = dict(a_grid=[m_grid, k_grid], z_grid=z_grid, P_z=P, return_fn=rf,
              beta=beta, options=dict(tol=1e-10, n_howard=30))
    snp = VFIProblem(**kw).solve("numpy")
    snb = VFIProblem(**kw).solve("numba")
    np.testing.assert_allclose(snb.V, snp.V, atol=1e-7)
    np.testing.assert_array_equal(snb.policy_aprime, snp.policy_aprime)


def test_validation_multi_grid():
    z_grid, P = tauchen(2, 0.5, 0.2)

    def rf(m_p, k_p, m, k, z, xp=np):
        return -m_p - k_p

    with pytest.raises(ValueError, match="1-D"):
        VFIProblem(a_grid=[np.zeros((2, 2)), np.array([0.0, 1.0])],
                   z_grid=z_grid, P_z=P, return_fn=rf, beta=0.95)
    with pytest.raises(ValueError, match=">= 2"):     # product grid too small
        VFIProblem(a_grid=[np.array([0.0]), np.array([0.0])],
                   z_grid=z_grid, P_z=P, return_fn=rf, beta=0.95)
