from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi.aggregate import (
    aggregate,
    evaluate_on_grid,
    lorenz_and_gini,
    weighted_quantile,
)


def _setup(rng):
    n_a, n_z = 5, 3
    a_grid = np.linspace(0.0, 4.0, n_a)
    z_grid = np.array([-0.2, 0.0, 0.2])
    pol = rng.integers(0, n_a, size=(n_a, n_z))
    mu = rng.random((n_a, n_z)); mu = mu / mu.sum()
    return n_a, n_z, a_grid, z_grid, pol, mu


def test_evaluate_shape_and_realized_aprime():
    rng = np.random.default_rng(0)
    n_a, n_z, a_grid, z_grid, pol, mu = _setup(rng)
    vals = evaluate_on_grid(lambda ap, a, z, xp=np: ap, pol, a_grid, z_grid)
    assert vals.shape == (n_a, n_z)
    np.testing.assert_allclose(vals, a_grid[pol])  # realized next-asset values


def test_aggregate_constant_is_total_mass():
    rng = np.random.default_rng(1)
    _, _, a_grid, z_grid, pol, mu = _setup(rng)
    agg = aggregate(lambda ap, a, z, xp=np: 1.0 + 0.0 * (ap + a + z),
                    mu, pol, a_grid, z_grid)
    assert agg == pytest.approx(1.0)


def test_aggregate_current_assets():
    rng = np.random.default_rng(2)
    _, _, a_grid, z_grid, pol, mu = _setup(rng)
    agg = aggregate(lambda ap, a, z, xp=np: a + 0.0 * (ap + z),
                    mu, pol, a_grid, z_grid)
    assert agg == pytest.approx(float(np.sum(mu * a_grid[:, None])))


def test_aggregate_realized_savings():
    rng = np.random.default_rng(3)
    _, _, a_grid, z_grid, pol, mu = _setup(rng)
    agg = aggregate(lambda ap, a, z, xp=np: ap + 0.0 * (a + z),
                    mu, pol, a_grid, z_grid)
    assert agg == pytest.approx(float(np.sum(mu * a_grid[pol])))


def test_aggregate_with_decision_axis():
    rng = np.random.default_rng(4)
    n_a, n_z = 4, 2
    a_grid = np.linspace(0.0, 3.0, n_a); z_grid = np.array([0.0, 1.0])
    d_grid = np.array([0.0, 10.0])
    pol = rng.integers(0, n_a, size=(n_a, n_z))
    pol_d = rng.integers(0, 2, size=(n_a, n_z))
    mu = rng.random((n_a, n_z)); mu = mu / mu.sum()
    # fn = d (realized decision value) -> aggregate = sum mu * d_grid[pol_d]
    agg = aggregate(lambda d, ap, a, z, xp=np: d + 0.0 * (ap + a + z),
                    mu, pol, a_grid, z_grid, policy_d=pol_d, d_grid=d_grid)
    assert agg == pytest.approx(float(np.sum(mu * d_grid[pol_d])))


def test_gini_degenerate_is_zero():
    mu = np.array([[0.25, 0.25], [0.25, 0.25]])
    values = np.full((2, 2), 3.0)  # all equal -> perfect equality
    _, _, g = lorenz_and_gini(mu, values)
    assert g == pytest.approx(0.0, abs=1e-12)


def test_gini_two_point_half_half():
    # half the mass at value 0, half at value 1 -> Gini = 0.5
    mu = np.array([[0.5], [0.5]])
    values = np.array([[0.0], [1.0]])
    pop, val, g = lorenz_and_gini(mu, values)
    assert g == pytest.approx(0.5, abs=1e-12)
    assert pop[0] == 0.0 and pop[-1] == pytest.approx(1.0)
    assert val[0] == 0.0 and val[-1] == pytest.approx(1.0)
    assert np.all(val <= pop + 1e-12)  # Lorenz below 45-degree line


def test_gini_rejects_negative():
    with pytest.raises(ValueError, match="nonnegative"):
        lorenz_and_gini(np.array([[1.0]]), np.array([[-1.0]]))


def test_gini_rejects_nan():
    # NaN must not silently slip past the guard and produce a NaN Gini
    with pytest.raises(ValueError, match="nonnegative"):
        lorenz_and_gini(np.array([[0.5], [0.5]]), np.array([[1.0], [np.nan]]))


def test_weighted_quantile_median():
    mu = np.array([0.25, 0.25, 0.25, 0.25])
    values = np.array([0.0, 1.0, 2.0, 3.0])
    med = weighted_quantile(mu, values, 0.5)
    assert 1.0 <= med <= 2.0  # median between the two middle values


def test_weighted_quantile_vectorised():
    mu = np.array([0.5, 0.5]); values = np.array([10.0, 20.0])
    qs = weighted_quantile(mu, values, [0.0, 1.0])
    np.testing.assert_allclose(qs, [10.0, 20.0])


def test_composition_aggregate_assets():
    from puremacro.vfi import VFIProblem, tauchen

    z_grid, P = tauchen(n=4, rho=0.8, sigma=0.1)
    a_grid = np.linspace(1e-3, 30.0, 50)

    def rf(ap, a, z, r, w, xp=np):
        c = w * xp.exp(z) + (1.0 + r) * a - ap
        return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -1e10)

    prob = VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                      beta=0.95, params={"r": 0.03, "w": 1.0})
    sol = prob.solve("numpy")
    mu = prob.stationary_distribution(sol)
    # aggregate current assets via the convenience method
    K = prob.aggregate(sol, mu, lambda ap, a, z, r, w, xp=np: a + 0.0 * (ap + z))
    assert K == pytest.approx(float(np.sum(mu * a_grid[:, None])), rel=1e-9)
    assert K > 0.0
