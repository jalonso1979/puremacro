from __future__ import annotations

import numpy as np

from puremacro.vfi.distribution import stationary_distribution
from puremacro.vfi.simulate import empirical_distribution, simulate_panel


def _stationary_of(M):
    w, V = np.linalg.eig(M.T)
    v = np.real(V[:, np.argmin(np.abs(w - 1.0))])
    return v / v.sum()


def _random_policy(n_a=8, n_z=3, seed=0):
    rng = np.random.default_rng(seed)
    pol = rng.integers(0, n_a, size=(n_a, n_z))
    P = rng.random((n_z, n_z)) + 0.1
    P = P / P.sum(axis=1, keepdims=True)
    return pol, P


def test_panel_shapes_and_index_ranges():
    pol, P = _random_policy()
    n_a, n_z = pol.shape
    a_path, z_path = simulate_panel(pol, P, n_agents=50, n_periods=20, seed=1)
    assert a_path.shape == (50, 20) and z_path.shape == (50, 20)
    assert a_path.min() >= 0 and a_path.max() < n_a
    assert z_path.min() >= 0 and z_path.max() < n_z


def test_deterministic_under_seed():
    pol, P = _random_policy()
    a1, z1 = simulate_panel(pol, P, n_agents=30, n_periods=10, seed=42)
    a2, z2 = simulate_panel(pol, P, n_agents=30, n_periods=10, seed=42)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(z1, z2)
    a3, _ = simulate_panel(pol, P, n_agents=30, n_periods=10, seed=43)
    assert not np.array_equal(a1, a3)   # different seed -> different draws


def test_transition_follows_policy_and_chain():
    # a_{t+1} must equal policy[a_t, z_t] exactly (deterministic asset choice)
    pol, P = _random_policy()
    a_path, z_path = simulate_panel(pol, P, n_agents=40, n_periods=15, seed=7)
    for t in range(14):
        np.testing.assert_array_equal(a_path[:, t + 1], pol[a_path[:, t], z_path[:, t]])


def test_empirical_distribution_matches_stationary():
    # the key anchor: a long panel's empirical (a,z) dist -> analytic stationary dist
    pol, P = _random_policy(n_a=8, n_z=3, seed=3)
    a_path, z_path = simulate_panel(pol, P, n_agents=4000, n_periods=400,
                                    seed=2024, burn_in=100)
    emp = empirical_distribution(a_path, z_path, n_a=8, n_z=3)
    np.testing.assert_allclose(emp.sum(), 1.0, atol=1e-12)
    analytic = stationary_distribution(pol, P)
    # Monte Carlo error ~ 1/sqrt(N*T); 8x3 bins over ~1.6M samples -> tight
    assert np.max(np.abs(emp - analytic)) < 0.01


def test_z_marginal_matches_chain_stationary():
    pol, P = _random_policy(n_a=6, n_z=4, seed=9)
    a_path, z_path = simulate_panel(pol, P, n_agents=3000, n_periods=300,
                                    seed=11, burn_in=80)
    emp = empirical_distribution(a_path, z_path, n_a=6, n_z=4)
    np.testing.assert_allclose(emp.sum(axis=0), _stationary_of(P), atol=0.01)


def test_empirical_distribution_validation():
    import pytest
    with pytest.raises(ValueError, match="same shape"):
        empirical_distribution(np.zeros((2, 3), dtype=int),
                               np.zeros((2, 4), dtype=int), n_a=5, n_z=5)


def test_simulate_exported():
    from puremacro.vfi import empirical_distribution as ed
    from puremacro.vfi import simulate_panel as sp

    assert sp is simulate_panel
    assert ed is empirical_distribution


def test_simulate_rejects_negative_burn_in():
    import pytest

    pol, P = _random_policy()
    with pytest.raises(ValueError, match="burn_in"):
        simulate_panel(pol, P, n_agents=5, n_periods=3, burn_in=-1)


def test_simulate_rejects_nonstochastic_P():
    import pytest

    pol, _ = _random_policy(n_a=4, n_z=3)
    with pytest.raises(ValueError, match="rows must sum"):
        simulate_panel(pol, np.full((3, 3), 0.3), n_agents=5, n_periods=3)


def test_empirical_distribution_rejects_empty_panel():
    import pytest

    with pytest.raises(ValueError, match="empty panel"):
        empirical_distribution(np.empty((0, 5), dtype=int),
                               np.empty((0, 5), dtype=int), n_a=2, n_z=3)
