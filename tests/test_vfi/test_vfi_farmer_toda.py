from __future__ import annotations

import numpy as np

from puremacro.vfi.discretize import farmer_toda, tauchen


def _stationary_of(P):
    w, V = np.linalg.eig(P.T)
    v = np.real(V[:, np.argmin(np.abs(w - 1.0))])
    return v / v.sum()


def test_shapes_and_rows_sum_to_one():
    grid, P = farmer_toda(n=9, rho=0.9, sigma=0.1)
    assert grid.shape == (9,) and P.shape == (9, 9)
    np.testing.assert_allclose(P.sum(axis=1), np.ones(9), atol=1e-10)
    assert np.all(P >= -1e-12)


def test_grid_symmetric_and_monotone():
    grid, _ = farmer_toda(n=7, rho=0.0, sigma=1.0)
    np.testing.assert_allclose(grid, -grid[::-1], atol=1e-12)
    assert np.all(np.diff(grid) > 0)


def test_conditional_moments_matched_interior():
    # for interior "from" states the conditional mean and variance match exactly
    n, rho, sigma = 15, 0.7, 0.2
    grid, P = farmer_toda(n=n, rho=rho, sigma=sigma)
    interior = range(3, n - 3)
    for i in interior:
        mu = float(P[i] @ grid)
        var = float(P[i] @ (grid - mu) ** 2)
        np.testing.assert_allclose(mu, rho * grid[i], atol=1e-6)
        np.testing.assert_allclose(var, sigma ** 2, rtol=5e-3)


def test_more_accurate_than_tauchen_at_high_persistence():
    # the Farmer-Toda selling point: the discretized chain's UNCONDITIONAL variance
    # is closer to the true sigma_z^2 than Tauchen's for high rho.
    rho, sigma, n = 0.95, 0.1, 9
    sigma_z2 = sigma ** 2 / (1.0 - rho ** 2)
    g_ft, P_ft = farmer_toda(n=n, rho=rho, sigma=sigma)
    g_t, P_t = tauchen(n=n, rho=rho, sigma=sigma)
    pi_ft = _stationary_of(P_ft); pi_t = _stationary_of(P_t)
    var_ft = float(pi_ft @ (g_ft - pi_ft @ g_ft) ** 2)
    var_t = float(pi_t @ (g_t - pi_t @ g_t) ** 2)
    err_ft = abs(var_ft - sigma_z2)
    err_t = abs(var_t - sigma_z2)
    assert err_ft < err_t          # Farmer-Toda strictly closer to the true variance
    assert err_ft / sigma_z2 < 0.05  # and within 5% of the truth


def test_iid_rows_identical():
    _, P = farmer_toda(n=6, rho=0.0, sigma=0.5)
    for i in range(1, 6):
        np.testing.assert_allclose(P[i], P[0], atol=1e-8)


def test_validation():
    import pytest
    with pytest.raises(ValueError, match="n must be"):
        farmer_toda(n=1, rho=0.5, sigma=0.1)
    with pytest.raises(ValueError, match="sigma"):
        farmer_toda(n=5, rho=0.5, sigma=0.0)


def test_farmer_toda_exported():
    from puremacro.vfi import farmer_toda as ft

    assert ft is farmer_toda


def test_high_persistence_no_numerical_warnings():
    # the log-space (logsumexp) formulation must not produce log(0)/0-div/overflow
    # even at rho=0.99, where the far-tail base density underflows; rows still valid
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        grid, P = farmer_toda(n=9, rho=0.99, sigma=0.1)
    np.testing.assert_allclose(P.sum(axis=1), np.ones(9), atol=1e-10)
    assert np.all(P >= -1e-12)
