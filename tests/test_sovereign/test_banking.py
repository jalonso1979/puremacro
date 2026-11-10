"""Economic identities and contract limits for sovereign-bank equilibria."""
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from puremacro.sovereign.banking import (
    SovereignBankingParameters, audit_sovereign_banking,
    solve_sovereign_banking, sovereign_bank_accounts,
)


def solve(**kwargs):
    options = dict(income=[.7, 1., 1.3], productivity=[.6, .9, 1.2],
                   probabilities=[.25, .5, .25], debt_grid=np.linspace(0, .5, 21),
                   price_selection="highest")
    options.update(kwargs)
    return solve_sovereign_banking(**options)


def test_zero_exposure_analytic_default_price_and_bank_reference():
    p = SovereignBankingParameters(domestic_share=0., default_loss=.2)
    r = solve(parameters=p)
    for i, b in enumerate(r.debt_grid):
        records = [e for e in r.equilibria if e.debt_index == i]
        assert len(records) == 1
        e = records[0]
        expected = b > p.default_loss*r.income + 1e-13
        assert np.array_equal(e.default, expected)
        assert_allclose(e.price, (1-r.probabilities@expected)/p.gross_rate, atol=1e-14)
        frame = e.to_frame()
        assert_allclose(frame.bank_equity, p.gross_rate*p.initial_equity)
        assert_allclose(frame.credit, np.minimum(np.maximum(r.productivity/p.gross_rate-p.credit_scale, 0),
                                                p.leverage*p.gross_rate*p.initial_equity))
    assert audit_sovereign_banking(r)["accepted"]


def test_fixed_contract_loss_timing_and_binding_credit_multiplier():
    p = SovereignBankingParameters(initial_equity=.2, domestic_share=.5, leverage=3.)
    kw = dict(parameters=p, income=[1.], productivity=[4.], debt=.25, price=.8)
    rep = sovereign_bank_accounts(**kw, default=[False]).iloc[0]
    default = sovereign_bank_accounts(**kw, default=[True]).iloc[0]
    assert_allclose(rep.bank_equity-default.bank_equity, .125)
    assert_allclose(rep.credit-default.credit, .375)
    assert_allclose(rep.bank_equity, p.gross_rate*(.2-.8*.125)+.125)
    assert default.default_resource_loss == p.default_loss
    assert rep.default_resource_loss == 0


def test_no_credit_effect_when_both_branches_unconstrained():
    p = SovereignBankingParameters(initial_equity=.4, domestic_share=.2)
    kw = dict(parameters=p, income=[1.], productivity=[.2], debt=.2, price=.8)
    a = sovereign_bank_accounts(**kw, default=[False])
    b = sovereign_bank_accounts(**kw, default=[True])
    assert_allclose(a.credit, b.credit)
    assert_allclose(a.loan_rate, p.gross_rate)


def test_zero_loan_demand_and_zero_equity_have_finite_accounts():
    p = SovereignBankingParameters(initial_equity=.2, domestic_share=.5, gross_rate=1.)
    f = sovereign_bank_accounts(p, income=[1., 1.], productivity=[.01, 1.],
                               debt=.4, price=1., default=[False, True])
    assert_allclose(f.credit, 0.)
    assert_allclose(f.foreign_funding, 0.)
    assert np.isfinite(f.to_numpy(dtype=float)).all()
    assert f.bank_equity.iloc[1] == 0


def test_all_sector_budgets_and_real_resources():
    r = solve()
    p = r.parameters
    for e in r.equilibria:
        f = e.to_frame(); b = r.debt_grid[e.debt_index]
        assert_allclose(e.initial_consumption+e.initial_reserves,
                        p.initial_income+(1-p.domestic_share)*e.price*b, atol=1e-14)
        assert_allclose(f.credit+f.safe_assets, f.bank_equity+f.foreign_funding)
        assert_allclose(f.consumption1+f.repayment+f.default_resource_loss, r.income)
        assert_allclose(f.consumption2, p.terminal_income+f.firm_profit+f.bank_dividend)
        assert_allclose(f.consumption1+f.credit+f.safe_assets,
            r.income-f.default_resource_loss+p.gross_rate*e.initial_reserves
            -(1-p.domestic_share)*f.repayment+f.foreign_funding)
        assert_allclose(f.consumption2+p.gross_rate*f.foreign_funding,
                        p.terminal_income+f.project_output+p.gross_rate*f.safe_assets)


def test_without_banks_or_exogenous_cost_positive_debt_is_worthless():
    r = solve(parameters=SovereignBankingParameters(domestic_share=0, default_loss=0))
    assert r.selected.debt_index == 0
    for e in r.equilibria:
        if e.debt_index:
            assert e.price == 0
            assert e.default.all()


def test_infeasible_repayment_retained_as_default_not_floored_consumption():
    r = solve(parameters=SovereignBankingParameters(domestic_share=0), debt_grid=[0., 2.])
    e = r.equilibria[-1]
    assert e.default.all()
    f = r.counterfactual_frame(e)
    rep = f[f.default == 0]
    assert not rep.feasible.any()
    assert rep.continuation_value.isna().all()
    assert (rep.consumption1 < 0).all()


def test_owned_arrays_fresh_tables_and_fingerprints():
    income = np.array([.7, 1., 1.3])
    r = solve(income=income)
    original = r.income.copy(); income[:] = 4
    assert_allclose(r.income, original)
    for a in (r.income, r.productivity, r.probabilities, r.debt_grid, r.candidates,
              r.selected.accounts, r.selected.default, r.selected_indices):
        assert not a.flags.writeable
    f = r.to_frame(); f.iloc[0, 0] = -100
    assert r.to_frame().iloc[0, 0] != -100
    assert r.fingerprint == solve().fingerprint
    assert r.fingerprint != solve(price_selection="lowest").fingerprint
    assert r.to_markdown() and r.to_latex() and r.to_typst()


def test_state_permutation_preserves_selection_and_values():
    r = solve(); order = [2, 0, 1]
    s = solve(income=r.income[order], productivity=r.productivity[order], probabilities=r.probabilities[order])
    assert s.selected.debt_index == r.selected.debt_index
    assert_allclose(s.selected.value, r.selected.value, atol=1e-13)
    assert_allclose(s.selected.price, r.selected.price, atol=1e-13)
    assert_allclose(s.selected.accounts, r.selected.accounts[order], atol=1e-13)


@pytest.mark.parametrize("kwargs", [
    {"domestic_share": 1.}, {"domestic_share": -.1}, {"leverage": .9},
    {"initial_equity": 0}, {"initial_equity": 1.}, {"default_loss": 1.},
    {"beta": 0.}, {"gross_rate": 0.}, {"credit_scale": -1.},
    {"terminal_income": float("nan")}, {"beta": True},
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError): SovereignBankingParameters(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"income": [0, 1, 2]}, {"productivity": [1, -1, 1]},
    {"probabilities": [.3, .3, .3]}, {"probabilities": [0, .5, .5]},
    {"debt_grid": [.1, .2]}, {"debt_grid": [0, .2, .1]},
    {"debt_grid": [0, .1, .1]}, {"price_selection": "automatic"},
    {"decision_tolerance": 1e-3},
    {"income": [1.]*13, "productivity": [1.]*13, "probabilities": [1/13]*13},
])
def test_invalid_specifications(kwargs):
    with pytest.raises(ValueError): solve(**kwargs)


def test_auditor_rejects_corrupted_price_accounts_coverage_and_selection():
    r = solve()
    index = r.selected_index; e = r.selected
    changes = [replace(e, price=e.price+.05), replace(e, value=e.value+.1)]
    accounts = e.accounts.copy(); accounts[0, 5] += .1
    changes.append(replace(e, accounts=accounts))
    for broken in changes:
        entries = list(r.equilibria); entries[index] = broken
        assert not audit_sovereign_banking(replace(r, equilibria=tuple(entries)))["accepted"]
    codes = r.candidates.copy(); codes[0, 0] = 0
    assert not audit_sovereign_banking(replace(r, candidates=codes))["accepted"]
    assert not audit_sovereign_banking(replace(r, equilibria=r.equilibria[:-1]))["accepted"]
