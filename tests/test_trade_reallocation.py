"""Independent limits, physical accounts and dynamic equilibrium timing."""
from dataclasses import replace

import numpy as np
import pytest

from puremacro.trade.reallocation import (
    ReallocationEconomy, ReallocationSolveError, audit_reallocation,
    solve_reallocation_steady_state, solve_reallocation_transition,
    run_reallocation_horizon_ladder,
)


@pytest.fixture(scope="module")
def economy():
    return ReallocationEconomy()


@pytest.fixture(scope="module")
def steady(economy):
    return solve_reallocation_steady_state(economy)


def tariff_path(horizon):
    tariffs = np.zeros((horizon, 2, 2))
    tariffs[2:8, 0, 0] = .1
    return tariffs


def test_symmetric_benchmark_matches_closed_form(economy, steady):
    a, beta, delta = economy.capital_share, economy.beta, economy.delta
    rental = 1/beta-1+delta
    wage = (1-a)*(a/rental)**(a/(1-a))
    output = wage/(1-a)
    capital = a*output/rental
    v = steady.values
    np.testing.assert_allclose(v["producer_prices"], 1., atol=1e-10)
    np.testing.assert_allclose(v["wages"], wage, atol=1e-10)
    np.testing.assert_allclose(v["rentals"], rental, atol=1e-10)
    np.testing.assert_allclose(v["output"], output/2, atol=1e-10)
    np.testing.assert_allclose(v["capital"], capital/2, atol=1e-10)
    np.testing.assert_allclose(v["consumption"], output-delta*capital, atol=1e-10)
    np.testing.assert_allclose(v["labor"], .5, atol=1e-10)
    np.testing.assert_allclose(v["supplier_log_ratio"], 0., atol=1e-10)
    assert steady.converged


def test_no_shock_path_and_welfare_are_stationary(economy, steady):
    run = solve_reallocation_transition(economy, np.zeros((8, 2, 2)), initial=steady, terminal=steady)
    for name in ("output", "capital", "consumption", "labor", "supplier_log_ratio"):
        np.testing.assert_allclose(run.paths[name], np.broadcast_to(steady.values[name], run.paths[name].shape), atol=1e-10)
    np.testing.assert_allclose(run.welfare["consumption_equivalent_pct"], 0., atol=1e-10)
    assert run.converged and run.diagnostics["terminal_gap"] < 1e-10
    assert audit_reallocation(run)["accepted"]


@pytest.mark.parametrize("elasticity", [.7, 1., 1.+1e-8, 4.])
def test_zero_sourcing_cost_recovers_analytic_ces(elasticity):
    model = ReallocationEconomy(trade_elasticity=elasticity, sourcing_cost=0.)
    ss = solve_reallocation_steady_state(model)
    run = solve_reallocation_transition(model, tariff_path(10), initial=ss, terminal=ss)
    for c in range(2):
        home, foreign = run.paths["producer_prices"][:, c], run.paths["producer_prices"][:, 1-c]
        expected = -elasticity*np.log((1+run.tariffs[:, c])*foreign/home)
        np.testing.assert_allclose(run.paths["supplier_log_ratio"][:, c], expected, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(run.paths["sourcing_costs"], 0.)


def test_zero_labor_and_capital_costs_recover_frictionless_conditions():
    model = ReallocationEconomy(labor_cost=0., capital_cost=0.)
    ss = solve_reallocation_steady_state(model)
    run = solve_reallocation_transition(model, tariff_path(10), initial=ss, terminal=ss)
    v = run.paths
    np.testing.assert_allclose(v["wages"][:, :, 0], v["wages"][:, :, 1], atol=1e-7)
    m = model.beta*(v["consumption"][1:]/v["consumption"][:-1])**(-model.risk_aversion)
    rhs = m[..., None]*(v["rentals"][1:]/v["price_index"][1:, :, None]+1-model.delta)
    np.testing.assert_allclose(rhs, 1., atol=1e-7)
    np.testing.assert_array_equal(v["capital_costs"], 0.)
    np.testing.assert_array_equal(v["labor_costs"], 0.)


@pytest.mark.parametrize("kwargs", [dict(beta=1.), dict(delta=0.), dict(capital_share=1.),
    dict(trade_elasticity=0.), dict(capital_cost=-1.), dict(labor_cost=np.inf),
    dict(sourcing_cost=np.nan), dict(home_shares=np.ones((2, 2))),
    dict(sector_weights=np.ones((2, 2))), dict(labor_supply=[1., 0.]),
    dict(countries=("A", "A")), dict(productivity=np.ones((2, 3)))])
def test_invalid_economies_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        ReallocationEconomy(**kwargs)


def test_invalid_paths_and_mismatched_boundaries(economy, steady):
    for bad in (np.zeros((4, 2)), np.full((4, 2, 2), -.1), np.full((4, 2, 2), np.nan)):
        with pytest.raises(ValueError):
            solve_reallocation_transition(economy, bad, initial=steady)
    other = replace(economy, beta=.95)
    with pytest.raises(ValueError, match="economy"):
        solve_reallocation_transition(other, tariff_path(10), initial=steady)
    with pytest.raises(ValueError, match="terminal"):
        solve_reallocation_transition(economy, np.full((4, 2, 2), .1), initial=steady, terminal=steady)


def test_solver_budget_does_not_fabricate_convergence(economy, steady):
    with pytest.raises(ReallocationSolveError) as failure:
        solve_reallocation_transition(economy, tariff_path(10), initial=steady, max_evaluations=1)
    assert not failure.value.diagnostics["accepted"]


@pytest.fixture(scope="module")
def transition(economy, steady):
    return solve_reallocation_transition(economy, tariff_path(12), initial=steady, terminal=steady)


def test_rebuilt_accounts_include_omitted_budget_and_real_resources(transition):
    v, model = transition.paths, transition.economy
    for t in range(len(transition.tariffs)):
        for c in range(2):
            p, output = v["producer_prices"][t, c], v["output"][t, c]
            imports = v["bilateral_quantities"][t, c, :, 1-c]
            foreign_prices = v["producer_prices"][t, 1-c]
            tariff_revenue = np.sum(transition.tariffs[t, c]*foreign_prices*imports)
            spending = v["price_index"][t, c]*(v["consumption"][t, c]
                +np.sum(v["investment"][t, c]+v["capital_costs"][t, c]+v["sourcing_costs"][t, c])
                +v["labor_costs"][t, c])
            assert spending == pytest.approx(np.sum(p*output)+tariff_revenue, abs=1e-8)
            assert np.sum(imports*foreign_prices) == pytest.approx(
                np.sum(v["bilateral_quantities"][t, 1-c, :, c]*p), abs=1e-8)
            assert v["rebates"][t, c] == pytest.approx(tariff_revenue, abs=1e-12)
    certificate = audit_reallocation(transition)
    assert certificate["accepted"] and certificate["max_error"] < 1e-8
    assert {"all_household_budgets", "external_balance_0", "external_balance_1",
            "physical_resources", "capital_euler_and_boundary", "sourcing_optimality"} <= set(certificate["checks"])
    assert max(np.max(v[key]) for key in ("capital_costs", "labor_costs", "sourcing_costs")) > 0
    np.testing.assert_allclose(transition.capital_path[1:],
        (1-model.delta)*transition.capital_path[:-1]+v["investment"], atol=1e-12)


@pytest.mark.parametrize("field,check", [("rebates", "government_budget"),
    ("bilateral_quantities", "bilateral_deliveries"), ("sourcing_costs", "sourcing_adjustment_resources"),
    ("labor_costs", "labor_adjustment_resources"), ("capital_costs", "capital_adjustment_resources")])
def test_audit_detects_interventions_in_saved_arrays(transition, field, check):
    paths = {name: value.copy() for name, value in transition.paths.items()}
    paths[field][3] += .001
    changed = replace(transition, paths=paths)
    audit = audit_reallocation(changed)
    assert not audit["accepted"]
    assert audit["checks"][check] > 1e-6


def test_sparse_complex_step_matches_independent_centered_products(economy, steady):
    from puremacro.trade.reallocation import _equations, _jacobian, _scales

    tariffs = tariff_path(5)
    x = np.tile(steady.coordinates, (5, 1)).ravel()+np.random.default_rng(2).normal(0, .01, 80)
    fun = lambda z: _equations(economy, z, tariffs, _scales(economy, steady.values), steady.values, steady.values)
    jac = _jacobian(fun, 5)(x)
    for seed in (1, 2, 3):
        direction = np.random.default_rng(seed).normal(size=len(x))
        step = 2e-6
        reference = (fun(x+step*direction)-fun(x-step*direction))/(2*step)
        np.testing.assert_allclose(jac@direction, reference, atol=2e-8, rtol=2e-8)


def test_path_matches_dense_finite_difference_solver(economy, steady):
    from scipy.optimize import root
    from puremacro.trade.reallocation import _equations, _scales

    tariffs = np.zeros((6, 2, 2))
    tariffs[1:4, 0, 0] = .07
    actual = solve_reallocation_transition(economy, tariffs, initial=steady, terminal=steady)
    fun = lambda z: _equations(economy, z, tariffs, _scales(economy, steady.values), steady.values, steady.values)
    independent = root(fun, np.tile(steady.coordinates, (6, 1)).ravel(), method="hybr", options={"xtol": 1e-10})
    assert independent.success and np.max(abs(fun(independent.x))) < 1e-9
    np.testing.assert_allclose(actual.coordinates.ravel(), independent.x, atol=1e-8, rtol=0)


def test_each_adjustment_cost_changes_joint_equilibrium(transition):
    for field, allocation, cost in (("capital_cost", "capital_next", "capital_costs"),
        ("labor_cost", "labor", "labor_costs"), ("sourcing_cost", "supplier_log_ratio", "sourcing_costs")):
        model = replace(transition.economy, **{field: 0.})
        run = solve_reallocation_transition(model, transition.tariffs)
        assert np.max(abs(run.paths[allocation]-transition.paths[allocation])) > 1e-5
        np.testing.assert_array_equal(run.paths[cost], 0.)
        assert audit_reallocation(run)["accepted"]


def test_tariff_is_anticipated_and_initial_capital_is_inherited(transition):
    assert np.count_nonzero(transition.tariffs[:2]) == 0
    np.testing.assert_array_equal(transition.paths["capital"][0], transition.initial.values["capital"])
    assert np.max(abs(transition.paths["capital_next"][0]-transition.initial.values["capital"])) > 1e-6
    assert np.max(abs(transition.paths["labor"][0]-transition.initial.values["labor"])) > 1e-6
    assert np.max(abs(transition.paths["supplier_log_ratio"][0])) > 1e-6


def test_owned_inputs_readonly_results_and_fresh_frames(transition):
    data = np.ones((2, 2))
    model = ReallocationEconomy(productivity=data)
    data[:] = 2.
    np.testing.assert_array_equal(model.productivity, 1.)
    with pytest.raises(ValueError):
        transition.paths["capital"][0] = 0.
    with pytest.raises(TypeError):
        transition.paths["capital"] = np.zeros((12, 2, 2))
    frame = transition.to_frame()
    frame.iloc[0, 0] = -999.
    assert transition.to_frame().iloc[0, 0] > 0
    assert "AAA" in transition.to_markdown()
    assert "tabular" in transition.to_latex()
    assert "table" in transition.to_typst()


def test_horizon_validation_retains_failures_and_rejects_different_policies(economy):
    unresolved = run_reallocation_horizon_ladder(economy, tariff_path, horizons=(10, 12), window=4,
        path_tolerance=1e-12, welfare_tolerance=1e-12, terminal_tolerance=1e-12)
    assert not unresolved.converged and unresolved.accepted is None
    assert len(unresolved.runs) == 2 and len(unresolved.attempts) == 2
    failed = run_reallocation_horizon_ladder(economy, tariff_path, horizons=(10, 12), window=4, max_evaluations=1)
    assert not failed.converged and len(failed.runs) == 0
    assert len(failed.attempts) == 2 and all(not a["solved"] for a in failed.attempts)

    def inconsistent(T):
        rates = tariff_path(T)
        if T > 10:
            rates[0, 0, 0] = .01
        return rates
    with pytest.raises(ValueError, match="prefixes"):
        run_reallocation_horizon_ladder(economy, inconsistent, horizons=(10, 12), window=4)
