"""Analytical limits and an independent discrete dynamic-programming oracle."""
import numpy as np
import pandas as pd
import pytest

from puremacro.trade.adjustment import solve_sourcing_adjustment


@pytest.mark.parametrize("sigma", [.5, 1., 1+1e-10, 3.])
def test_frictionless_cost_and_hicksian_demands(sigma):
    p = pd.DataFrame([[1., 1.], [1.2, .8]], columns=["home", "foreign"])
    a = pd.Series({"home": .6, "foreign": .4})
    r = solve_sourcing_adjustment(p, a, elasticity=sigma, adjustment_cost=0)
    if abs(sigma-1) < 1e-8:
        index = p.home.to_numpy()**.6 * p.foreign.to_numpy()**.4
    else:
        index = (.6*p.home.to_numpy()**(1-sigma)+.4*p.foreign.to_numpy()**(1-sigma))**(1/(1-sigma))
    q = a.to_numpy() * (index[:, None]/p.to_numpy())**sigma
    np.testing.assert_allclose(r.quantities, q, rtol=1e-9)
    np.testing.assert_allclose(r.accounts.total_unit_cost, index, rtol=1e-9)
    np.testing.assert_allclose(r.accounts.output, 1, atol=1e-12)


@pytest.mark.parametrize("cost", [0., 1., 100.])
def test_common_origin_prices_have_no_sourcing_adjustment(cost):
    price = np.array([1., 1.2, .9, 1.])
    r = solve_sourcing_adjustment(pd.DataFrame({"home": price, "foreign": price}),
        pd.Series({"home": .6, "foreign": .4}), elasticity=2., adjustment_cost=cost)
    np.testing.assert_allclose(r.quantities, np.tile([.6,.4], (4,1)), atol=1e-10)
    np.testing.assert_allclose(r.accounts.total_unit_cost, price, atol=1e-10)


def test_dynamic_solution_against_independent_grid_bellman_oracle():
    # For sigma=2, F(q)=(sqrt(alpha0*q0)+sqrt(alpha1*q1))^2.
    # Parameterize the unit isoquant by unnormalized quantity mix s.
    a = np.array([.6, .4])
    prices = np.array([[1., 1.], [1., 1.5], [1., 1.]])
    discount, cost = .97, 1.5
    r = solve_sourcing_adjustment(pd.DataFrame(prices, columns=["home", "foreign"]),
        pd.Series(a, index=["home", "foreign"]), elasticity=2., adjustment_cost=cost,
        discount=discount)
    s = np.linspace(.2, .9, 2001)
    raw = np.column_stack([s, 1-s])
    q = raw / (np.sqrt(raw*a).sum(axis=1)**2)[:, None]
    switching = cost/2*np.sum((q[:,None,:]-q[None,:,:])**2/a, axis=2)
    future = np.zeros(len(q))
    for p in prices[::-1]:
        future = np.min(q@p + switching + discount*future[None,:], axis=1)
    initial_cost = cost/2*np.sum((q-a)**2/a, axis=1)
    # Recompute at the actual initial state rather than a nearby grid node.
    tail = np.zeros(len(q))
    for p in prices[:0:-1]:
        tail = np.min(q@p + switching + discount*tail[None,:], axis=1)
    oracle = float(np.min(q@prices[0]+initial_cost+discount*tail))
    value = r.diagnostics["discounted_total_cost"]
    assert value <= oracle + 1e-8
    assert oracle - value < 2e-6
    # Direct resource/production accounting, independent of production helper.
    solved = r.quantities.to_numpy()
    output = np.sqrt(solved*a).sum(axis=1)**2
    np.testing.assert_allclose(output, 1., atol=1e-7)
    change = solved - np.vstack([a,solved[:-1]])
    costs = (solved*prices).sum(axis=1)+cost/2*np.sum(change**2/a,axis=1)
    assert value == pytest.approx(discount**np.arange(3)@costs)
    assert solved[0,1] < .4  # costly adjustment starts before the announced shock.


def test_failed_optimizer_cannot_be_used_as_a_path():
    with pytest.raises(RuntimeError, match="not accepted"):
        solve_sourcing_adjustment(pd.DataFrame({"home":[1.,1.,1.],"foreign":[1.,2.,2.]}),
            pd.Series({"home":.6,"foreign":.4}), elasticity=3, adjustment_cost=5, max_iter=1)


@pytest.mark.pyodide_smoke
def test_portable_dynamic_sourcing():
    test_common_origin_prices_have_no_sourcing_adjustment(1.)
