from __future__ import annotations

import numpy as np

from puremacro.vfi import FiniteHorizonProblem
from puremacro.vfi.finite_horizon import age_profile, cross_section, life_cycle_distribution


def _stationary_of(M):
    w, V = np.linalg.eig(M.T)
    v = np.real(V[:, np.argmin(np.abs(w - 1.0))])
    return v / v.sum()


def _solved_life_cycle(n_a=30, n_z=3, J=8):
    a_grid = np.linspace(0.0, 12.0, n_a)
    z_grid = np.array([-0.3, 0.0, 0.3])[:n_z] if n_z == 3 else np.linspace(-0.3, 0.3, n_z)
    P = np.full((n_z, n_z), 1.0 / n_z) * 0.2 + np.eye(n_z) * 0.8
    P = P / P.sum(axis=1, keepdims=True)
    # income peaks early then falls (classic lifecycle); agents save for retirement
    def rf(ap, a, z, age, xp=np):
        income = 3.0 - 0.3 * min(age, 4)        # declining earnings -> save while young
        c = income * xp.exp(z) + 1.04 * a - ap
        return xp.where(c > 1e-12, xp.log(xp.maximum(c, 1e-12)), -1e10)
    prob = FiniteHorizonProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                                beta=0.97, horizon=J)
    return a_grid, z_grid, P, prob.solve("numpy")


def test_life_cycle_distribution_masses_and_shape():
    a_grid, z_grid, P, sol = _solved_life_cycle()
    J, n_a, n_z = sol.policy_aprime.shape
    dist = life_cycle_distribution(sol, P)
    assert dist.shape == (J, n_a, n_z)
    for j in range(J):
        np.testing.assert_allclose(dist[j].sum(), 1.0, atol=1e-12)  # cohort mass 1
        assert np.all(dist[j] >= -1e-15)


def test_newborn_default_is_low_asset_z_stationary():
    a_grid, z_grid, P, sol = _solved_life_cycle()
    dist = life_cycle_distribution(sol, P)
    pi_z = _stationary_of(P)
    # all mass at the lowest asset, z ~ z-stationary
    np.testing.assert_allclose(dist[0][0, :], pi_z, atol=1e-12)
    np.testing.assert_allclose(dist[0][1:, :], 0.0, atol=1e-15)


def test_z_marginal_is_stationary_at_every_age():
    # newborn z-marginal = pi_z and push preserves it -> every age's z-marg = pi_z
    a_grid, z_grid, P, sol = _solved_life_cycle()
    dist = life_cycle_distribution(sol, P)
    pi_z = _stationary_of(P)
    for j in range(dist.shape[0]):
        np.testing.assert_allclose(dist[j].sum(axis=0), pi_z, atol=1e-10)


def test_custom_newborn_distribution_used():
    a_grid, z_grid, P, sol = _solved_life_cycle()
    n_a, n_z = len(a_grid), len(z_grid)
    psi0 = np.zeros((n_a, n_z)); psi0[2, :] = _stationary_of(P)  # born at asset index 2
    dist = life_cycle_distribution(sol, P, newborn_dist=psi0)
    np.testing.assert_allclose(dist[0], psi0, atol=1e-12)


def test_cross_section_sums_to_one_and_default_uniform():
    a_grid, z_grid, P, sol = _solved_life_cycle()
    dist = life_cycle_distribution(sol, P)
    cs = cross_section(dist)
    np.testing.assert_allclose(cs.sum(), 1.0, atol=1e-12)
    # uniform age weights => cross-section is the simple mean over ages
    np.testing.assert_allclose(cs, dist.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(cs.sum(axis=0), _stationary_of(P), atol=1e-10)


def test_age_profile_assets_accumulate():
    a_grid, z_grid, P, sol = _solved_life_cycle()
    dist = life_cycle_distribution(sol, P)
    prof = age_profile(dist, a_grid)
    assert prof.shape == (dist.shape[0],)
    assert prof[0] == 0.0                       # born with zero assets
    assert prof[3] > prof[0]                    # assets accumulate over early life
    assert np.all(prof >= -1e-12)


def test_age_weights_validation():
    import pytest
    a_grid, z_grid, P, sol = _solved_life_cycle()
    dist = life_cycle_distribution(sol, P)
    with pytest.raises(ValueError, match="age_weights"):
        cross_section(dist, age_weights=np.ones(dist.shape[0] + 1))


def test_lifecycle_dist_exported():
    from puremacro.vfi import age_profile as ap
    from puremacro.vfi import cross_section as cs
    from puremacro.vfi import life_cycle_distribution as lcd

    assert lcd is life_cycle_distribution
    assert cs is cross_section
    assert ap is age_profile
