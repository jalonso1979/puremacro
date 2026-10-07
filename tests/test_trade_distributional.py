"""Independent formulas, aggregation invariants, and audited GE incidence."""
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from puremacro.trade.distributional import (
    prepare_household_groups, compute_distributional_welfare,
    distributional_welfare_from_results,
)
from puremacro.trade.household import HouseholdDomainError, household_prices_from_result
from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium


def example_groups(*, scale=1.0, exposures=None):
    spending = pd.DataFrame([[60., 40.], [40., 160.]],
                            index=["low", "high"], columns=["food", "other"]) * scale
    if exposures is None:
        exposures = pd.DataFrame([[80., 20.], [100., 100.]],
                                 index=spending.index, columns=["labor", "capital"]) * scale
    return prepare_household_groups(
        spending, pd.Series([3., 1.], index=spending.index),
        income_exposure=exposures, monetary_unit="fixture_currency", period="quarter",
        provenance={"source": "analytic synthetic fixture", "is_synthetic": True})


def prices(food=1.2, other=.9):
    return pd.Series([food, other], index=["food", "other"])


@pytest.mark.parametrize("rule", ["fixed_baskets", "cobb_douglas"])
def test_independent_two_group_formulas_and_weighted_aggregation(rule):
    g = example_groups()
    changes = pd.Series([.1, -.05], index=["labor", "capital"])
    # Per-unit allocation scores (2,1), denominator 3*2+1*1=7.
    result = compute_distributional_welfare(
        g, prices(1, 1), prices(), factor_income_changes=changes,
        transfer_total=14., transfer_weights=pd.Series([2., 1.], index=["low", "high"]), rule=rule)
    shares = np.array([[.6, .4], [.2, .8]])
    index = (shares @ np.array([1.2, .9]) if rule == "fixed_baskets" else
             np.exp(shares @ np.log([1.2, .9])))
    m0 = np.array([100., 200.])
    m1 = m0 + np.array([7., 5.]) + np.array([4., 2.])
    expected_ev = m1 / index - m0
    expected_cv = m1 - m0 * index
    assert_allclose(result.groups.ev, expected_ev, atol=1e-12)
    assert_allclose(result.groups.cv, expected_cv, atol=1e-12)
    assert_allclose(result.groups.transfer, [4, 2])
    assert_allclose(result.aggregate.total_transfer, 14)
    assert_allclose(result.aggregate.total_ev, [3, 1] @ expected_ev)
    assert_allclose(result.aggregate.ev_pct, 100 * ([3, 1] @ expected_ev) / 500)
    assert_allclose(result.aggregate.mean_ev_pct, np.average(100 * expected_ev / m0, weights=[3, 1]))
    assert abs(result.aggregate.ev_pct - result.aggregate.mean_ev_pct) > .01
    for measure in ("ev", "cv"):
        assert_allclose(result.groups[[f"price_{measure}", f"income_{measure}", f"transfer_{measure}"]].sum(axis=1),
                        result.groups[measure], atol=1e-13)


@pytest.mark.parametrize("rule", ["fixed_baskets", "cobb_douglas", "ces", "stone_geary"])
def test_nominal_homogeneity_and_currency_rescaling(rule):
    g = example_groups()
    nominal = compute_distributional_welfare(
        g, prices(2, 4), prices(2.2, 4.4), rule=rule, ces_elasticity=1.5,
        factor_income_changes=pd.Series(.1, index=g.income_exposure.columns))
    assert_allclose(nominal.groups[["ev", "cv"]], 0, atol=1e-12)
    original = compute_distributional_welfare(g, prices(2, 4), prices(2.4, 3.6), rule=rule,
                                              transfer_total=20, ces_elasticity=1.5)
    scaled = compute_distributional_welfare(example_groups(scale=100), prices(2, 4), prices(2.4, 3.6),
                                            rule=rule, transfer_total=2000, ces_elasticity=1.5)
    assert_allclose(scaled.groups.ev, original.groups.ev * 100, atol=1e-9)
    assert_allclose(scaled.groups.ev_pct, original.groups.ev_pct, atol=1e-12)


def test_label_permutation_and_equal_per_unit_transfers():
    g = example_groups()
    permuted = prepare_household_groups(
        g.expenditure.iloc[::-1, ::-1], g.population_weights,
        income_exposure=g.income_exposure.iloc[:, ::-1], monetary_unit=g.monetary_unit,
        period=g.period, provenance=g.provenance)
    changes = pd.Series({"labor": .08, "capital": -.02})
    a = compute_distributional_welfare(g, prices(1, 1), prices(), factor_income_changes=changes, transfer_total=-8)
    b = compute_distributional_welfare(permuted, prices(1, 1).iloc[::-1], prices(),
                                       factor_income_changes=changes.iloc[::-1], transfer_total=-8)
    pd.testing.assert_frame_equal(a.groups, b.groups.reindex(a.groups.index))
    assert_allclose(a.groups.transfer, [-2, -2])
    assert_allclose(a.aggregate.total_transfer, -8)


def test_exposed_income_can_exceed_consumption_but_incremental_spending_is_explicit():
    exposure = pd.DataFrame({"cash_wages": [200, 400]}, index=["low", "high"])
    g = example_groups(exposures=exposure)
    result = compute_distributional_welfare(g, prices(1, 1), prices(1, 1),
                                            factor_income_changes=pd.Series({"cash_wages": .1}))
    assert_allclose(result.groups.income_change, [20, 40])
    assert_allclose(result.groups.ev, [20, 40])
    assert "all incremental" in result.metadata["income_spending_assumption"]


def test_les_assumptions_and_domain_are_preserved():
    g = example_groups()
    eta = pd.DataFrame([[.8, 1.3], [.7, 1.075]], index=g.expenditure.index, columns=g.expenditure.columns)
    result = compute_distributional_welfare(g, prices(1, 1), prices(), rule="stone_geary",
                                            expenditure_elasticities=eta,
                                            supernumerary_share=pd.Series({"high": .6, "low": .5}))
    assert result.metadata["preference_calibration"]["supernumerary_share_default_is_assumption"] is False
    assert_allclose(result.metadata["preference_calibration"]["supernumerary_share"], [.5, .6])
    # Price-only state is outside LES surplus domain, even with a final rescue transfer.
    with pytest.raises(HouseholdDomainError):
        compute_distributional_welfare(g, prices(1, 1), prices(3, 3), rule="stone_geary", transfer_total=10000)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_rejects_invalid_prices(bad):
    with pytest.raises(ValueError):
        compute_distributional_welfare(example_groups(), prices(1, 1), prices(bad, 1))


@pytest.mark.parametrize("mutation, match", [
    (lambda g: replace(g, population_weights=pd.Series([0, 1], index=g.expenditure.index)), "positive"),
    (lambda g: replace(g, baseline_budget=g.baseline_budget + 1), "row sums"),
    (lambda g: replace(g, income_exposure=-g.income_exposure), "nonnegative"),
    (lambda g: replace(g, provenance={}), "source"),
    (lambda g: replace(g, monetary_unit=""), "monetary_unit"),
    (lambda g: replace(g, expenditure=g.expenditure.rename(columns={"other": "food"})), "unique"),
])
def test_adapter_validation_rechecks_mutated_group_frames(mutation, match):
    with pytest.raises(ValueError, match=match):
        compute_distributional_welfare(mutation(example_groups()), prices(1, 1), prices())


def test_rejects_mismatched_shocks_and_invalid_allocation():
    g = example_groups()
    with pytest.raises(ValueError, match="labels"):
        compute_distributional_welfare(g, prices(1, 1), pd.Series({"food": 1, "missing": 1}))
    with pytest.raises(ValueError, match="labels"):
        compute_distributional_welfare(g, prices(1, 1), prices(), factor_income_changes=pd.Series({"labor": .1}))
    with pytest.raises(ValueError, match="below -1"):
        compute_distributional_welfare(g, prices(1, 1), prices(), factor_income_changes=pd.Series({"labor": -2, "capital": 0}))
    for scores in ([0, 0], [-1, 1], [np.nan, 1]):
        with pytest.raises(ValueError):
            compute_distributional_welfare(g, prices(1, 1), prices(), transfer_weights=pd.Series(scores, index=g.expenditure.index))


@pytest.fixture(scope="module")
def ge_pair():
    # Balanced synthetic two-country/two-sector accounting; no empirical claim.
    data = np.zeros((7, 10))
    data[:4, :4] = 10.
    data[4, :4] = 5.
    data[5, :4] = 35.
    data[6, :4] = 20.
    data[:4, 4:] = 10.
    calib = calibrate_trade_model(data, nc=2, ns=2, nfd=3,
                                  country_codes=["AAA", "BBB"], sector_codes=["food", "other"])
    calib.metadata.update(source="analytic GE fixture", is_regression_fixture=True)
    baseline = solve_trade_equilibrium(calib, accounting="consistent", tol=1e-10)
    counterfactual = solve_trade_equilibrium(calib, tau=np.array([.1, 0]), tau_fd=np.array([.1, 0]),
                                            accounting="consistent", tol=1e-10)
    assert baseline.converged and counterfactual.converged
    return calib, baseline, counterfactual


def test_ge_bridge_reuses_audited_prices_preserves_provenance(ge_pair):
    calib, base, cf = ge_pair
    g = example_groups()
    changes = pd.Series({"labor": .05, "capital": -.02})
    bridge = distributional_welfare_from_results(g, calib, base, cf, country="AAA",
                                                 factor_income_changes=changes, transfer_total=12)
    p0, _ = household_prices_from_result(base, calib)
    p1, _ = household_prices_from_result(cf, calib)
    direct = compute_distributional_welfare(g, pd.Series(p0[:, 0], index=calib.sector_codes),
                                            pd.Series(p1[:, 0], index=calib.sector_codes),
                                            factor_income_changes=changes, transfer_total=12)
    pd.testing.assert_frame_equal(bridge.groups, direct.groups)
    assert bridge.metadata["is_regression_fixture"] is True
    assert bridge.metadata["ge_calibration_provenance"]["source"] == "analytic GE fixture"
    assert bridge.metadata["source"] == "analytic synthetic fixture"


def test_ge_bridge_rejects_unconverged_misaligned_and_unauditable_states(ge_pair):
    calib, base, cf = ge_pair
    g = example_groups()
    kwargs = dict(country="AAA", factor_income_changes=pd.Series({"labor": 0, "capital": 0}))
    with pytest.raises(ValueError, match="converge"):
        distributional_welfare_from_results(g, calib, base, replace(cf, converged=False), **kwargs)
    with pytest.raises(ValueError, match="labels"):
        distributional_welfare_from_results(g, calib, base, replace(cf, sector_codes=("other", "food")), **kwargs)
    with pytest.raises(ValueError):
        distributional_welfare_from_results(g, calib, base, replace(cf, w_sol=cf.w_sol * 2), **kwargs)
    with pytest.raises(ValueError, match="explicit"):
        distributional_welfare_from_results(g, calib, base, cf, country="AAA", factor_income_changes=None)


def test_ge_bridge_refuses_placeholder_prices_for_positive_survey_spending():
    # Independent balanced raw-flow construction: country AAA consumes only
    # 'other' in category 0. Food production remains positive for other uses,
    # and the tariff counterfactual moves its producer prices.
    raw = np.zeros((7, 10))
    raw[:4, :4] = 10.
    raw[4, :4] = 5.
    raw[5, :4] = [25., 45., 25., 45.]
    raw[6, :4] = 20.
    raw[:4, 4:] = 10.
    raw[:4, 4] = [0., 20., 0., 20.]
    calib = calibrate_trade_model(raw, nc=2, ns=2, nfd=3,
                                  country_codes=["AAA", "BBB"], sector_codes=["food", "other"])
    assert_allclose(calib.afd[:, 0, 0], [0., .5, 0., .5])
    base = solve_trade_equilibrium(calib, accounting="consistent", tol=1e-10)
    cf = solve_trade_equilibrium(calib, accounting="consistent", tau=np.array([.3, 0.]),
                                 tau_fd=np.array([.3, 0.]), tol=1e-10)
    assert base.converged and cf.converged
    prices_cf, _ = household_prices_from_result(cf, calib)
    assert prices_cf[0, 0] == 1.  # structural-zero sentinel, not a price estimate
    assert not np.allclose(cf.p_sol, base.p_sol)
    kwargs = {"country": "AAA", "factor_income_changes": pd.Series({"labor": 0., "capital": 0.})}
    with pytest.raises(ValueError, match="no GE origin-basket support.*food"):
        distributional_welfare_from_results(example_groups(), calib, base, cf, **kwargs)
    # The same sentinel is harmless if the survey also has zero expenditure
    # in that sector; no household welfare then depends on its invented price.
    original = example_groups()
    spending = original.expenditure.copy()
    spending["food"] = 0.
    supported = prepare_household_groups(spending, original.population_weights,
                                          income_exposure=original.income_exposure,
                                          monetary_unit=original.monetary_unit, period=original.period,
                                          provenance=original.provenance)
    result = distributional_welfare_from_results(supported, calib, base, cf, **kwargs)
    assert result.metadata["ge_structural_zero_sectors"] == ["food"]
    assert "fixed origin weights" in result.metadata["ge_sector_price_assumption"]


def test_exports_plot_and_input_provenance_are_independent():
    import matplotlib.pyplot as plt
    g = example_groups()
    result = compute_distributional_welfare(g, prices(1, 1), prices())
    assert "Conditional household" in result.summary()
    assert "low" in result.to_markdown()
    assert "tabular" in result.to_latex()
    assert "table" in result.to_typst()
    assert result.plot().axes
    plt.close("all")
    frame = result.to_frame()
    frame.iloc[0, 0] = -1
    assert result.groups.population_weight.iloc[0] == 3
    g.provenance["source"] = "changed after evaluation"
    assert result.metadata["source"] == "analytic synthetic fixture"
