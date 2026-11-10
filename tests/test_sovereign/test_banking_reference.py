"""Independent optimization, equilibrium enumeration and invariance references."""
from dataclasses import replace
from itertools import product
from math import log

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize_scalar

from puremacro.sovereign.banking import (
    SovereignBankingParameters, audit_sovereign_banking,
    solve_sovereign_banking, sovereign_bank_accounts,
)


def reference(p, y, a, probability, grid):
    records = {}
    for bi, debt in enumerate(grid):
        for choices in product((False, True), repeat=len(y)):
            q = sum(w for w, default in zip(probability, choices) if not default)/p.gross_rate
            safe = p.initial_equity-p.domestic_share*q*debt
            if safe < 0: continue
            values, loans = [], []
            for income, productivity in zip(y, a):
                branch_values, branch_loans = [], []
                for default in (False, True):
                    equity = p.gross_rate*safe+(not default)*p.domestic_share*debt
                    cap = p.leverage*equity
                    # Solve the concave credit surplus maximization numerically;
                    # do not use the production solver's closed-form loan demand.
                    surplus = lambda x: productivity*np.log1p(x/p.credit_scale)-p.gross_rate*x
                    trial = minimize_scalar(lambda x: -surplus(x), bounds=(0, cap), method="bounded",
                                            options={"xatol": 1e-12}).x if cap else 0.
                    loan = max((0., cap, trial), key=surplus)
                    c1 = income*(1-p.default_loss*default)-(not default)*debt
                    c2 = p.terminal_income+surplus(loan)+p.gross_rate*equity
                    branch_values.append(log(c1)+p.beta*log(c2) if c1 > 0 else -np.inf)
                    branch_loans.append(loan)
                values.append(branch_values); loans.append(branch_loans)
            decisions = tuple(v[1] > v[0]+1e-12 for v in values)
            if decisions == choices:
                mask = sum(int(d)*2**j for j, d in enumerate(choices))
                welfare = log(p.initial_income-p.initial_equity+q*debt)+p.beta*sum(
                    w*v[int(d)] for w, v, d in zip(probability, values, choices))
                records[bi, mask] = q, welfare, [v[int(d)] for v, d in zip(loans, choices)]
    return records


@pytest.mark.parametrize("share,penalty", [(0, .1), (.4, 0), (.4, .1), (.8, .5)])
@pytest.mark.parametrize("selection", ["highest", "lowest"])
def test_numerically_optimized_credit_and_exhaustive_price_reference(share, penalty, selection):
    p = SovereignBankingParameters(domestic_share=share, default_loss=penalty)
    y, a, weights, grid = [.7, 1, 1.3], [.6, .9, 1.2], [.25, .5, .25], [0, .1, .2, .3, .6]
    expected = reference(p, y, a, weights, grid)
    r = solve_sovereign_banking(parameters=p, income=y, productivity=a, probabilities=weights,
                                debt_grid=grid, price_selection=selection)
    assert {(e.debt_index, e.mask) for e in r.equilibria} == set(expected)
    for e in r.equilibria:
        q, value, loans = expected[e.debt_index, e.mask]
        assert_allclose(e.price, q, atol=1e-14)
        assert_allclose(e.value, value, atol=1e-12)
        assert_allclose(e.to_frame().credit, loans, atol=3e-8)
    choose = max if selection == "highest" else min
    available = [choose((key for key in expected if key[0] == bi), key=lambda key: expected[key][0])
                 for bi in range(len(grid)) if any(key[0] == bi for key in expected)]
    optimum = max(available, key=lambda key: (expected[key][1], -key[0]))
    assert (r.selected.debt_index, r.selected.mask) == optimum


def test_multiple_equilibria_are_retained_and_selection_is_explicit():
    kwargs = dict(income=[.7, 1, 1.3], productivity=[.6, .9, 1.2],
                  probabilities=[.25, .5, .25], debt_grid=[0, .25])
    high, low = (solve_sovereign_banking(**kwargs, price_selection=s) for s in ("highest", "lowest"))
    records = [e for e in high.equilibria if e.debt_index == 1]
    assert {e.mask for e in records} == {1, 3, 7}
    assert_allclose(sorted(e.price for e in records), [0, .25/1.02, .75/1.02])
    assert high.equilibria[high.selected_indices[1]].mask == 1
    assert low.equilibria[low.selected_indices[1]].mask == 7
    assert high.diagnostics["multiple_equilibrium_debt_points"] == 1
    assert_allclose(high.candidates, low.candidates)


def test_no_feasible_pure_price_is_recorded_and_never_filled_in():
    p = SovereignBankingParameters(domestic_share=.8, default_loss=.9)
    r = solve_sovereign_banking(parameters=p, income=[1.], productivity=[2.], probabilities=[1.],
                                debt_grid=[0., .3], price_selection="highest")
    assert r.selected_indices[1] == -1
    assert r.debt_frame().status.iloc[1] == "no_pure_equilibrium"
    assert list(r.candidates_frame().status.iloc[2:]) == ["infeasible_portfolio", "inconsistent_default"]
    assert r.diagnostics["debt_points_without_pure_equilibrium"] == 1
    assert audit_sovereign_banking(r)["accepted"]


def test_splitting_identical_joint_states_does_not_create_extra_equilibria():
    base = dict(income=[.7, 1.], productivity=[.6, .9], probabilities=[.25, .75],
                debt_grid=[0, .1, .2, .3], price_selection="highest")
    r = solve_sovereign_banking(**base)
    base.update(income=[.7, 1., 1.], productivity=[.6, .9, .9], probabilities=[.25, .375, .375])
    s = solve_sovereign_banking(**base)
    assert len(s.equilibria) == len(r.equilibria)
    assert_allclose(s.equilibria_frame()[["debt", "price", "value", "expected_credit"]],
                    r.equilibria_frame()[["debt", "price", "value", "expected_credit"]], atol=1e-13)


@pytest.mark.parametrize("scale", [.001, 1000.])
def test_good_units_transform_accounts_and_utility_consistently(scale):
    p = SovereignBankingParameters()
    kwargs = dict(income=np.array([.7,1.,1.3]), productivity=np.array([.6,.9,1.2]),
                  probabilities=[.25,.5,.25], debt_grid=np.array([0,.1,.25,.3]), price_selection="highest")
    r = solve_sovereign_banking(parameters=p, **kwargs)
    fields = {key: getattr(p, key)*scale for key in ("initial_income", "terminal_income", "initial_equity", "credit_scale")}
    for key in ("income", "productivity", "debt_grid"): kwargs[key] *= scale
    s = solve_sovereign_banking(parameters=replace(p, **fields), **kwargs)
    assert np.array_equal(r.candidates, s.candidates)
    assert r.selected_index == s.selected_index
    for e, f in zip(r.equilibria, s.equilibria):
        assert_allclose(f.value-e.value, (1+p.beta+p.beta**2)*log(scale), atol=1e-12)
        assert_allclose(f.to_frame().credit, e.to_frame().credit*scale, atol=1e-10)
        assert_allclose(f.price, e.price, atol=1e-14)


def test_debt_zero_tie_rule_and_exact_portfolio_endpoint():
    p = SovereignBankingParameters(gross_rate=1., initial_equity=.2, domestic_share=.5, default_loss=.9)
    r = solve_sovereign_banking(parameters=p, income=[1.], productivity=[2.], probabilities=[1.],
                                debt_grid=[0., .4], price_selection="highest")
    e = [e for e in r.equilibria if e.debt_index == 1][0]
    assert e.initial_reserves == 0.
    assert e.mask == 0
    assert not r.equilibria[0].default.any()


def test_large_enumeration_rejected_before_allocation():
    with pytest.raises(ValueError, match="enumeration limit"):
        solve_sovereign_banking(income=[1.]*12, productivity=[1.]*12, probabilities=[1/12]*12,
            debt_grid=np.linspace(0,.2,50), price_selection="highest")


def test_default_decision_tolerance_is_reported_and_regret_bounded():
    p = SovereignBankingParameters(domestic_share=0, default_loss=.1)
    r = solve_sovereign_banking(parameters=p, income=[1.], productivity=[1.], probabilities=[1.],
        debt_grid=[0, .1+1e-10], price_selection="highest", decision_tolerance=1e-8)
    assert not r.equilibria[-1].default[0]
    assert 0 < r.diagnostics["maximum_default_regret"] < 1e-8
    assert r.diagnostics["near_indifference_state_decisions"] >= 1


def test_audit_rejects_noninteger_candidate_codes_and_wrong_selection():
    r = solve_sovereign_banking(income=[1.], productivity=[1.], probabilities=[1.],
                                debt_grid=[0, .05], price_selection="highest")
    broken = r.candidates.astype(float); broken[0, 1] = .1
    assert not audit_sovereign_banking(replace(r, candidates=broken))["accepted"]
    assert not audit_sovereign_banking(replace(r, selected_index=1-r.selected_index))["accepted"]


@pytest.mark.parametrize("changes", [{"default": [0]}, {"price": -1}, {"debt": -.2},
    {"price": 1.1}, {"debt": 4., "price": .9}, {"income": [float("inf")]}])
def test_invalid_contracts_are_rejected(changes):
    kwargs = dict(income=[1.], productivity=[1.], default=[True], debt=.2, price=.8)
    kwargs.update(changes)
    with pytest.raises(ValueError): sovereign_bank_accounts(SovereignBankingParameters(), **kwargs)
