from __future__ import annotations

import numpy as np

from puremacro.vfi import VFIProblem, solve_egm, tauchen
from puremacro.vfi.distribution import (
    lottery_distribution,
    lottery_push,
    stationary_distribution,
)


def _stationary_of(M):
    w, V = np.linalg.eig(M.T)
    v = np.real(V[:, np.argmin(np.abs(w - 1.0))])
    return v / v.sum()


def test_lottery_push_conserves_mass_and_expected_asset():
    a_grid = np.array([0.0, 1.0, 2.0, 3.0])
    P = np.array([[1.0]])
    # a' = 1.5 for the single (a,z)=(0,0) agent with all mass -> splits 0.5/0.5 to nodes 1,2
    mu = np.array([[1.0], [0.0], [0.0], [0.0]])
    ap = np.array([[1.5], [1.5], [1.5], [1.5]])
    out = lottery_push(mu, ap, a_grid, P)
    np.testing.assert_allclose(out.sum(), 1.0, atol=1e-12)                   # mass conserved
    np.testing.assert_allclose(out[:, 0], [0.0, 0.5, 0.5, 0.0], atol=1e-12)  # 0.5 to a=1, 0.5 to a=2
    # expected next asset under the lottery equals a' = 1.5
    np.testing.assert_allclose(out[:, 0] @ a_grid, 1.5, atol=1e-12)


def test_lottery_push_clamps_out_of_grid():
    a_grid = np.array([0.0, 1.0, 2.0])
    P = np.array([[1.0]])
    mu = np.array([[0.5], [0.0], [0.5]])
    ap = np.array([[-1.0], [1.0], [5.0]])      # below grid, on-node, above grid
    out = lottery_push(mu, ap, a_grid, P)
    # a'=-1 -> all to node 0; a'=5 -> all to node 2 (top); masses: 0.5 at node0, 0.5 at node2
    np.testing.assert_allclose(out[:, 0], [0.5, 0.0, 0.5], atol=1e-12)


def test_lottery_distribution_sums_to_one_and_z_marginal():
    rng = np.random.default_rng(0)
    n_a, n_z = 20, 3
    a_grid = np.linspace(0.0, 10.0, n_a)
    P = rng.random((n_z, n_z)); P = P / P.sum(1, keepdims=True)
    # a continuous a'-policy: mean-revert toward the middle of the grid
    ap = 0.5 * a_grid[:, None] + 2.0 + 0.3 * np.arange(n_z)[None, :]
    mu = lottery_distribution(ap, a_grid, P)
    assert mu.shape == (n_a, n_z)
    np.testing.assert_allclose(mu.sum(), 1.0, atol=1e-10)
    assert np.all(mu >= -1e-15)
    np.testing.assert_allclose(mu.sum(axis=0), _stationary_of(P), atol=1e-10)  # z-marginal


def test_lottery_is_stationary_fixed_point():
    rng = np.random.default_rng(1)
    n_a, n_z = 15, 4
    a_grid = np.linspace(0.0, 8.0, n_a)
    P = rng.random((n_z, n_z)); P = P / P.sum(1, keepdims=True)
    ap = np.clip(0.6 * a_grid[:, None] + 1.0, a_grid[0], a_grid[-1]) + 0.0 * np.arange(n_z)
    mu = lottery_distribution(ap, a_grid, P)
    np.testing.assert_allclose(lottery_push(mu, ap, a_grid, P), mu, atol=1e-9)  # fixed point


def test_lottery_matches_discrete_grid_policy():
    # if a' lands exactly on grid nodes, the lottery reduces to the deterministic
    # push and the stationary dist equals stationary_distribution(indices)
    rng = np.random.default_rng(2)
    n_a, n_z = 10, 3
    a_grid = np.linspace(0.0, 9.0, n_a)            # integer-spaced
    P = rng.random((n_z, n_z)); P = P / P.sum(1, keepdims=True)
    pol_idx = rng.integers(0, n_a, size=(n_a, n_z))
    ap_values = a_grid[pol_idx]                    # a' exactly on grid nodes
    mu_lottery = lottery_distribution(ap_values, a_grid, P)
    mu_index = stationary_distribution(pol_idx, P)
    np.testing.assert_allclose(mu_lottery, mu_index, atol=1e-9)


def test_lottery_egm_agrees_with_discrete_vfi_dist():
    # the headline bridge: the lottery dist of an EGM policy ~ the stationary dist
    # of the discrete-VFI policy for the SAME income-fluctuation problem.
    z_grid, P = tauchen(n=5, rho=0.9, sigma=0.2)
    income = np.exp(z_grid)
    a_grid = np.linspace(0.0, 60.0, 200)
    beta, r, gamma = 0.96, 0.03, 2.0

    egm = solve_egm(a_grid, z_grid, income, P, beta=beta, r=r, gamma=gamma)
    mu_egm = lottery_distribution(egm.aprime, a_grid, P)

    def rf(ap, a, z, xp=np):
        c = (1.0 + r) * a + np.exp(z) - ap
        return xp.where(c > 0.0, -1.0 / xp.maximum(c, 1e-12), -np.inf)

    vsol = VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf, beta=beta,
                      options=dict(tol=1e-10, n_howard=50)).solve("numpy")
    mu_vfi = stationary_distribution(vsol.policy_aprime, P)
    # both approximate the same continuous distribution; compare aggregate assets
    K_egm = float(np.sum(mu_egm * a_grid[:, None]))
    K_vfi = float(np.sum(mu_vfi * a_grid[:, None]))
    assert abs(K_egm - K_vfi) / K_vfi < 0.05        # mean assets agree within 5%
    np.testing.assert_allclose(mu_egm.sum(), 1.0, atol=1e-10)
    np.testing.assert_allclose(mu_egm.sum(axis=0), _stationary_of(P), atol=1e-8)


def test_lottery_exported():
    from puremacro.vfi import lottery_distribution as ld
    from puremacro.vfi import lottery_push as lp

    assert ld is lottery_distribution
    assert lp is lottery_push
