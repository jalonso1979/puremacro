from __future__ import annotations

import numpy as np
import pytest

from puremacro.models.nested_dmp import kernels as K


def test_matching_cobb_douglas_values():
    # m(u,v) = mu * u^alpha * v^(1-alpha); alpha=0.5 -> geometric mean.
    assert K.matching_rate(0.05, 0.05, mu=1.0, alpha=0.5) == pytest.approx(0.05)
    assert K.matching_rate(0.04, 0.09, mu=1.0, alpha=0.5) == pytest.approx(0.06)


def test_q_and_f_consistency():
    # f(theta) = theta * q(theta); q(1)=f(1)=mu.
    theta = 1.7
    assert K.q_theta(1.0, mu=1.0, alpha=0.5) == pytest.approx(1.0)
    assert K.f_theta(1.0, mu=1.0, alpha=0.5) == pytest.approx(1.0)
    assert K.f_theta(theta, mu=1.2, alpha=0.4) == pytest.approx(
        theta * K.q_theta(theta, mu=1.2, alpha=0.4)
    )


def test_belief_update_midpoint_returns_prior():
    # Likelihoods equal when r_obs is the midpoint of mu_D and mu_H;
    # with prior 0.5 the posterior is exactly 0.5.
    post = K.belief_update(
        0.5, r_obs=-1.0, sigma=1.0, r_star=0.0,
        phi_D=-3.0, phi_H=1.0, signal_sd=1.0,
    )
    assert post == pytest.approx(0.5)


def test_belief_update_cut_raises_dove_posterior():
    # mu_D=-3 (dove cut), mu_H=+1 (hawk hike). Observing a deep cut (r=-3)
    # pushes the posterior toward dove; a hike (r=+1) toward hawk.
    args = dict(sigma=1.0, r_star=0.0, phi_D=-3.0, phi_H=1.0, signal_sd=1.0)
    post_cut = K.belief_update(0.5, r_obs=-3.0, **args)
    post_hike = K.belief_update(0.5, r_obs=1.0, **args)
    assert post_cut > 0.9
    assert post_hike < 0.1
    assert post_cut > 0.5 > post_hike


def _row_stochastic(n, rng):
    A = rng.random((n, n)) + 1e-3
    return A / A.sum(axis=1, keepdims=True)


def test_bellman_no_separation_matches_closed_form(rng):
    # When the continuation value is always positive, the max(.,0) is
    # inactive and V solves V = flow + beta P V. With flow = c*1 and P
    # row-stochastic (P 1 = 1), the fixed point is V = c/(1-beta) * 1.
    n, beta, c = 6, 0.95, 1.0
    P = _row_stochastic(n, rng)
    flow = np.full(n, c)
    V = K.bellman_iterate(flow, P, beta, tol=1e-12, max_iter=100_000)
    np.testing.assert_allclose(V, c / (1.0 - beta), rtol=1e-6)


def test_bellman_negative_flow_separates_to_zero(rng):
    # Deeply negative flow -> continuation always < 0 -> V pinned at the
    # outside option (separate), normalised to 0.
    n, beta = 6, 0.95
    P = _row_stochastic(n, rng)
    flow = np.full(n, -10.0)
    V = K.bellman_iterate(flow, P, beta, tol=1e-12, max_iter=100_000)
    np.testing.assert_allclose(V, 0.0, atol=1e-9)


def test_rouwenhorst_base_case_and_rows_sum_to_one():
    grid, P = K.rouwenhorst(2, rho=0.9, sigma_eps=0.1)
    p = (1.0 + 0.9) / 2.0
    np.testing.assert_allclose(P, [[p, 1 - p], [1 - p, p]])
    np.testing.assert_allclose(P.sum(axis=1), 1.0)


def test_rouwenhorst_grid_symmetric_and_sized():
    n = 7
    grid, P = K.rouwenhorst(n, rho=0.95, sigma_eps=0.1)
    assert grid.shape == (n,)
    assert P.shape == (n, n)
    np.testing.assert_allclose(grid[0], -grid[-1])
    np.testing.assert_allclose(P.sum(axis=1), 1.0)


def test_rouwenhorst_rejects_n_below_2():
    with pytest.raises(ValueError, match="n must be >= 2"):
        K.rouwenhorst(1, rho=0.9, sigma_eps=0.1)


def test_bellman_raises_on_nonconvergence(rng):
    # Too few iterations at an impossibly tight tol must fail loudly,
    # not silently return an unconverged value.
    n, beta = 6, 0.99
    P = _row_stochastic(n, rng)
    flow = np.full(n, 1.0)
    with pytest.raises(RuntimeError, match="did not converge"):
        K.bellman_iterate(flow, P, beta, tol=1e-15, max_iter=2)


def test_tauchen_transition_on_fixed_grid():
    from puremacro.models.nested_dmp.kernels import tauchen_transition, rouwenhorst
    # Use a Rouwenhorst grid as the fixed grid; discretize the SAME AR(1) on it.
    rho, sd = 0.95, 0.10
    grid, _ = rouwenhorst(15, rho, sd)
    P = tauchen_transition(grid, rho, sd)
    assert P.shape == (15, 15)
    # Rows are proper probability distributions.
    np.testing.assert_allclose(P.sum(axis=1), np.ones(15))
    assert np.all(P >= 0.0)
    # Conditional mean at an interior node ~ rho * x_i (discretization-accurate).
    i = 7  # central node (grid is symmetric about 0; grid[7]~0)
    cond_mean = float(P[i] @ grid)
    assert cond_mean == pytest.approx(rho * grid[i], abs=5e-2)
    # Higher innovation sd spreads mass: P(stay in same cell) falls.
    P_wide = tauchen_transition(grid, rho, 2.0 * sd)
    assert P_wide[i, i] < P[i, i]


def test_phi_sigma_raises_beta_uniformly_without_moving_pi_star():
    # The 'lean against uncertainty' lever phi_sigma cuts BOTH type-rates by
    # phi_sigma*sigma: E[beta] rises for any pi, but the type gap -- and hence the
    # sign-flip threshold pi* = phi_H/(phi_H-phi_D) -- is invariant.
    phiD, phiH, sigma = -0.032, 0.015, 1.0
    kw = dict(r_star=0.01, phi_D=phiD, phi_H=phiH)
    for pi in (0.1, 0.5, 0.9):
        b0 = K.expected_discount(pi, sigma, **kw, phi_sigma=0.0)
        b1 = K.expected_discount(pi, sigma, **kw, phi_sigma=0.01)
        assert b1 > b0  # leaning against uncertainty raises the effective discount
    # The dove-minus-hawk beta gap is unchanged by phi_sigma (pi* is invariant).
    gap0 = (K.expected_discount(0.9, sigma, **kw, phi_sigma=0.0)
            - K.expected_discount(0.1, sigma, **kw, phi_sigma=0.0))
    gap1 = (K.expected_discount(0.9, sigma, **kw, phi_sigma=0.02)
            - K.expected_discount(0.1, sigma, **kw, phi_sigma=0.02))
    assert gap1 == pytest.approx(gap0, rel=0.05)
