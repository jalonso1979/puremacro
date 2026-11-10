"""Independent household variations, symmetry and horizon references."""
from dataclasses import replace

import numpy as np
import pytest

from puremacro.trade.reallocation import (
    ReallocationEconomy, ReallocationSolveError, audit_reallocation, reallocation_stability,
    solve_reallocation_steady_state, solve_reallocation_transition, run_reallocation_horizon_ladder,
)


def tariff_path(T):
    rates = np.zeros((T, 2, 2))
    rates[2:8, 0, 0] = .1
    return rates


@pytest.fixture(scope="module")
def run():
    return solve_reallocation_transition(ReallocationEconomy(), tariff_path(12))


def test_household_sourcing_variations_at_fixed_market_prices(run):
    m, v, c = run.economy, run.paths, 0
    T = len(run.tariffs)
    weights = m.beta**np.arange(T)*v["consumption"][:, c]**(-m.risk_aversion)/v["price_index"][:, c]
    fixed_resources = (v["consumption"][:, c]+v["investment"][:, c].sum(axis=1)
                       +v["capital_costs"][:, c].sum(axis=1)+v["labor_costs"][:, c])
    alpha = np.stack([m.home_shares[c], 1-m.home_shares[c]], axis=-1)
    delivered = np.stack([v["producer_prices"][:, c],
                          (1+run.tariffs[:, c])*v["producer_prices"][:, 1-c]], axis=-1)

    def expenditure(log_ratio):
        # Construct CES quantities directly, without the solver's recipe code.
        rho = 1-1/m.trade_elasticity
        raw = np.stack([np.ones_like(log_ratio), np.exp(log_ratio)], axis=-1)
        quantity = alpha*raw/(np.sum(alpha*raw**rho, axis=-1)**(1/rho))[..., None]
        sector_prices = np.sum(delivered*quantity, axis=-1)
        price = np.prod(sector_prices**m.sector_weights[c], axis=-1)
        changes = log_ratio-np.vstack([run.initial.values["supplier_log_ratio"][c], log_ratio[:-1]])
        costs = .5*m.sourcing_cost*m.labor_supply[c]*np.sum(m.sector_weights[c]*changes**2, axis=-1)
        return weights@(price*(fixed_resources+costs))

    for t, s in ((0, 0), (4, 0), (7, 1), (T-1, 0)):
        step = 2e-5
        up, down = v["supplier_log_ratio"][:, c].copy(), v["supplier_log_ratio"][:, c].copy()
        up[t, s] += step
        down[t, s] -= step
        derivative = (expenditure(up)-expenditure(down))/(2*step)
        scale = weights[t]*v["price_index"][t, c]*m.labor_supply[c]*m.sector_weights[c, s]
        assert abs(derivative/scale) < 2e-8


def test_household_labor_variations_include_next_period_cost(run):
    m, v, c = run.economy, run.paths, 1
    T = len(run.tariffs)
    weights = m.beta**np.arange(T)*v["consumption"][:, c]**(-m.risk_aversion)/v["price_index"][:, c]

    def net_expenditure(first_labor):
        labor = np.column_stack([first_labor, m.labor_supply[c]-first_labor])
        changes = labor-np.vstack([run.initial.values["labor"][c], labor[:-1]])
        costs = .5*m.labor_cost*np.sum(changes**2, axis=-1)/m.labor_supply[c]
        return weights@(v["price_index"][:, c]*costs-np.sum(v["wages"][:, c]*labor, axis=-1))

    for t in (0, 4, T-1):
        step = 2e-5
        up, down = v["labor"][:, c, 0].copy(), v["labor"][:, c, 0].copy()
        up[t] += step
        down[t] -= step
        derivative = (net_expenditure(up)-net_expenditure(down))/(2*step)
        assert abs(derivative/(weights[t]*v["price_index"][t, c])) < 2e-8


def test_household_capital_variations_match_terminal_value(run):
    m, v, c = run.economy, run.paths, 0
    T = len(run.tariffs)
    weights = m.beta**np.arange(T)*v["consumption"][:, c]**(-m.risk_aversion)/v["price_index"][:, c]

    def net_expenditure(next_capital):
        capital = np.vstack([run.initial.values["capital"][c], next_capital[:-1]])
        investment = next_capital-(1-m.delta)*capital
        costs = .5*m.capital_cost*(next_capital-capital)**2/capital
        out = np.sum(v["price_index"][:, c, None]*(investment+costs)-v["rentals"][:, c]*capital, axis=-1)
        terminal_value = m.beta**(T-1)*run.terminal.values["consumption"][c]**(-m.risk_aversion)*next_capital[-1].sum()
        return weights@out-terminal_value

    for t, s in ((0, 0), (5, 1), (T-1, 0)):
        step = 2e-5
        up, down = v["capital_next"][:, c].copy(), v["capital_next"][:, c].copy()
        up[t, s] += step
        down[t, s] -= step
        derivative = (net_expenditure(up)-net_expenditure(down))/(2*step)
        assert abs(derivative/(weights[t]*v["price_index"][t, c])) < 2e-8


@pytest.mark.parametrize("axis", [0, 1])
def test_country_and_sector_relabeling_preserves_real_allocations(axis):
    model = ReallocationEconomy(productivity=np.array([[1.1, .9], [.95, 1.05]]),
        sector_weights=np.array([[.4, .6], [.55, .45]]), home_shares=np.array([[.7, .8], [.65, .75]]),
        labor_supply=np.array([1., 1.3]))
    kwargs = {name: np.flip(getattr(model, name), axis=axis) for name in
              ("productivity", "sector_weights", "home_shares")}
    if axis == 0:
        kwargs.update(countries=model.countries[::-1], labor_supply=model.labor_supply[::-1])
    else:
        kwargs.update(sectors=model.sectors[::-1])
    swapped = replace(model, **kwargs)
    rates = tariff_path(10)
    base = solve_reallocation_transition(model, rates)
    other = solve_reallocation_transition(swapped, np.flip(rates, axis=axis+1))
    for name in ("output", "capital", "capital_next", "labor", "supplier_log_ratio"):
        np.testing.assert_allclose(other.paths[name], np.flip(base.paths[name], axis=axis+1), atol=1e-7, rtol=1e-8)
    expected_C = base.paths["consumption"][:, ::-1] if axis == 0 else base.paths["consumption"]
    np.testing.assert_allclose(other.paths["consumption"], expected_C, atol=1e-7)


def test_world_size_scaling_leaves_per_worker_paths_and_welfare_unchanged(run):
    factor = 10.
    bigger = replace(run.economy, labor_supply=factor*run.economy.labor_supply)
    other = solve_reallocation_transition(bigger, run.tariffs)
    for name in ("consumption", "output", "capital", "labor", "gross_absorption"):
        np.testing.assert_allclose(other.paths[name]/factor, run.paths[name], atol=1e-8, rtol=1e-8)
    for name in ("producer_prices", "price_index", "wages", "supplier_log_ratio"):
        np.testing.assert_allclose(other.paths[name], run.paths[name], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(other.welfare.consumption_equivalent_pct, run.welfare.consumption_equivalent_pct, atol=1e-8)


def test_local_stability_keeps_the_rejected_high_cost_case():
    model = ReallocationEconomy()
    good = reallocation_stability(solve_reallocation_steady_state(model))
    assert good["stable_graph"] and good["stable_roots"] == good["required_stable_roots"] == 16
    assert good["lag_rank"] == good["lead_rank"] == 10
    assert good["quadratic_residual_max_abs"] < 1e-10
    unstable = replace(model, sourcing_cost=1.)
    bad = reallocation_stability(solve_reallocation_steady_state(unstable))
    assert not bad["stable_graph"] and bad["stable_roots"] == 15
    with pytest.raises(ReallocationSolveError, match="stable graph") as failure:
        solve_reallocation_transition(unstable, tariff_path(10))
    assert failure.value.diagnostics["stage"] == "terminal_stability"
    finite = solve_reallocation_transition(unstable, np.zeros((4, 2, 2)), require_stable_terminal=False)
    assert finite.converged and not finite.diagnostics["terminal_stability"]["stable_graph"]
    assert not finite.diagnostics["stable_terminal_required"]


def test_horizon_extension_cannot_add_unannounced_policy_events():
    def inconsistent(T):
        p = np.zeros((T, 2, 2))
        if T > 10:
            p[10, 0, 0] = .1
        return p
    with pytest.raises(ValueError, match="later policy events"):
        run_reallocation_horizon_ladder(ReallocationEconomy(), inconsistent, horizons=(10, 12), window=4)


def test_horizon_refinement_preserves_early_rejections():
    result = run_reallocation_horizon_ladder(ReallocationEconomy(), tariff_path)
    assert result.converged and result.accepted is not None
    assert [a["verified"] for a in result.attempts] == [False, False, True]
    assert [a["horizon"] for a in result.attempts] == [40, 80, 160]
    last = result.attempts[-1]
    assert last["path_gap"] < 1e-6 and last["welfare_gap_pp"] < 1e-4 and last["terminal_gap"] < 1e-5
    assert audit_reallocation(result.accepted)["accepted"]


@pytest.mark.parametrize("gamma", [1., 1.+1e-8, 2.])
def test_welfare_matches_independent_discounted_utility_with_capital_tail(gamma):
    model = ReallocationEconomy(risk_aversion=gamma)
    result = solve_reallocation_transition(model, tariff_path(12))
    consumption = result.paths["consumption"]
    base, terminal = result.initial.values["consumption"], result.terminal.values["consumption"]
    T = len(consumption)
    power = 1-gamma

    def utility(ratio):
        return np.log(ratio) if power == 0 else np.expm1(power*np.log(ratio))/power

    discounted = np.sum(model.beta**np.arange(T)[:, None]*utility(consumption/base), axis=0)
    discounted += model.beta**T/(1-model.beta)*utility(terminal/base)
    capital_value = model.beta**(T-1)*terminal**(-gamma)*np.sum(
        result.capital_path[-1]-result.terminal.values["capital"], axis=1)/base**(1-gamma)
    discounted += capital_value
    log_equivalent = ((1-model.beta)*discounted if power == 0 else
                      np.log1p(power*(1-model.beta)*discounted)/power)
    expected = 100*np.expm1(log_equivalent)
    np.testing.assert_allclose(result.welfare.consumption_equivalent_pct, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(result.welfare.terminal_capital_utility_correction, capital_value, atol=1e-13)
    assert np.max(abs(capital_value)) > 1e-4


def test_permanent_tariff_uses_distinct_audited_terminal_state():
    model = ReallocationEconomy()
    policy = np.zeros((80, 2, 2))
    policy[2:, 0, 0] = .05
    result = solve_reallocation_transition(model, policy)
    assert not np.array_equal(result.initial.tariffs, result.terminal.tariffs)
    np.testing.assert_array_equal(result.terminal.tariffs, policy[-1])
    assert audit_reallocation(result)["accepted"]
    assert np.max(abs(result.terminal.coordinates-result.initial.coordinates)) > .01
    assert np.isfinite(result.welfare.consumption_equivalent_pct).all()


@pytest.mark.parametrize("labels", [None, "AA", ("A", "A"), ([1], "B"), ("", "B")])
def test_invalid_country_labels_are_rejected(labels):
    with pytest.raises(ValueError, match="distinct nonempty labels"):
        ReallocationEconomy(countries=labels)
