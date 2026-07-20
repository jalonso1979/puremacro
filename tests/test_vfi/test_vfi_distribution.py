from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi.distribution import joint_transition_matrix, stationary_distribution


def _stationary_of(M):
    # left eigenvector (eigenvalue 1) of a row-stochastic matrix M, normalised
    w, V = np.linalg.eig(M.T)
    v = np.real(V[:, np.argmin(np.abs(w - 1.0))])
    return v / v.sum()


def test_sums_to_one_and_nonneg():
    rng = np.random.default_rng(0)
    n_a, n_z = 6, 3
    pol = rng.integers(0, n_a, size=(n_a, n_z))
    P = rng.random((n_z, n_z)); P = P / P.sum(1, keepdims=True)
    mu = stationary_distribution(pol, P)
    assert mu.shape == (n_a, n_z)
    np.testing.assert_allclose(mu.sum(), 1.0, atol=1e-10)
    assert np.all(mu >= -1e-15)


def test_absorbing_policy_known_dist():
    pol = np.array([[1], [1]])
    P = np.array([[1.0]])
    mu = stationary_distribution(pol, P)
    np.testing.assert_allclose(mu[:, 0], [0.0, 1.0], atol=1e-10)


def test_is_stationary_fixed_point():
    rng = np.random.default_rng(1)
    n_a, n_z = 5, 4
    pol = rng.integers(0, n_a, size=(n_a, n_z))
    P = rng.random((n_z, n_z)); P = P / P.sum(1, keepdims=True)
    mu = stationary_distribution(pol, P)
    mu_half = np.zeros((n_a, n_z))
    z_idx = np.broadcast_to(np.arange(n_z), (n_a, n_z))
    np.add.at(mu_half, (pol, z_idx), mu)
    mu_step = mu_half @ P
    np.testing.assert_allclose(mu_step, mu, atol=1e-9)


def test_matches_joint_eigenvector():
    rng = np.random.default_rng(2)
    n_a, n_z = 4, 3
    pol = rng.integers(0, n_a, size=(n_a, n_z))
    P = rng.random((n_z, n_z)); P = P / P.sum(1, keepdims=True)
    mu = stationary_distribution(pol, P)
    G = joint_transition_matrix(pol, P)
    np.testing.assert_allclose(G.sum(axis=1), np.ones(n_a * n_z), atol=1e-12)
    np.testing.assert_allclose(mu.reshape(-1), _stationary_of(G), atol=1e-8)


def test_z_marginal_matches_exogenous_stationary():
    rng = np.random.default_rng(3)
    n_a, n_z = 7, 4
    pol = rng.integers(0, n_a, size=(n_a, n_z))
    P = rng.random((n_z, n_z)); P = P / P.sum(1, keepdims=True)
    mu = stationary_distribution(pol, P)
    np.testing.assert_allclose(mu.sum(axis=0), _stationary_of(P), atol=1e-8)


def test_validation_errors():
    with pytest.raises(ValueError, match="rows must sum"):
        stationary_distribution(np.array([[0], [1]]), np.array([[0.5]]))
    with pytest.raises(ValueError, match="valid a-grid"):
        stationary_distribution(np.array([[5]]), np.array([[1.0]]))


def test_composition_with_solve():
    from puremacro.vfi import VFIProblem, tauchen

    z_grid, P = tauchen(n=4, rho=0.8, sigma=0.1)
    a_grid = np.linspace(1e-3, 30.0, 60)

    def rf(ap, a, z, r, w, xp=np):
        c = w * xp.exp(z) + (1.0 + r) * a - ap
        return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -1e10)

    prob = VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                      beta=0.95, params={"r": 0.03, "w": 1.0})
    sol = prob.solve("numpy")
    mu = prob.stationary_distribution(sol)
    assert mu.shape == (60, 4)
    np.testing.assert_allclose(mu.sum(), 1.0, atol=1e-10)
    assert np.all(mu >= -1e-15)
    np.testing.assert_allclose(mu.sum(axis=0), _stationary_of(P), atol=1e-6)


def test_push_distribution_one_step_matches_manual():
    rng = np.random.default_rng(11)
    n_a, n_z = 6, 3
    pol = rng.integers(0, n_a, size=(n_a, n_z))
    P = rng.random((n_z, n_z)); P = P / P.sum(1, keepdims=True)
    mu = rng.random((n_a, n_z)); mu = mu / mu.sum()
    from puremacro.vfi.distribution import push_distribution
    got = push_distribution(mu, pol, P)
    # manual Tan step
    mu_half = np.zeros((n_a, n_z))
    z_idx = np.broadcast_to(np.arange(n_z), (n_a, n_z))
    np.add.at(mu_half, (pol, z_idx), mu)
    expected = mu_half @ P
    np.testing.assert_allclose(got, expected, atol=1e-15)
    np.testing.assert_allclose(got.sum(), 1.0, atol=1e-12)  # mass conserved


def test_push_iterated_equals_stationary():
    from puremacro.vfi.distribution import push_distribution
    rng = np.random.default_rng(12)
    n_a, n_z = 5, 4
    pol = rng.integers(0, n_a, size=(n_a, n_z))
    P = rng.random((n_z, n_z)); P = P / P.sum(1, keepdims=True)
    mu = np.full((n_a, n_z), 1.0 / (n_a * n_z))
    for _ in range(20000):
        nxt = push_distribution(mu, pol, P)
        if np.max(np.abs(nxt - mu)) < 1e-14:
            mu = nxt
            break
        mu = nxt
    np.testing.assert_allclose(mu, stationary_distribution(pol, P), atol=1e-10)
