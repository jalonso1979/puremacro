"""Economic timing, accounting and analytical limits for adaptation/debt."""
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from puremacro.sovereign.adaptation import (
    SovereignAdaptationParameters, solve_sovereign_adaptation,
    audit_sovereign_adaptation, audit_adaptation_distribution,
)


def solve(**kwargs):
    options = dict(income=[1., 1.], damage=[0., .4], disaster=[False, True],
                   transition=[[.9, .1], [.75, .25]], debt_grid=[0., .1, .3, .6],
                   adaptation_grid=[0., .05, .15], horizon=3)
    options.update(kwargs)
    return solve_sovereign_adaptation(**options)


def test_terminal_analytic_default_and_no_new_investment_or_borrowing():
    p = SovereignAdaptationParameters(disaster_relief=.5)
    r = solve(horizon=1, parameters=p)
    net = r.income[None, :]*(1-r.damage[None, :]*np.exp(-p.effectiveness*r.adaptation_grid[:, None]))
    service = r.debt_grid[:, None, None]*(1-p.disaster_relief*r.disaster[None, None, :])
    expected = service > p.default_loss*net[None, :, :] + 1e-13
    assert np.array_equal(r.default[0], expected)
    assert_allclose(r.excluded_value[0], np.log((1-p.default_loss)*net))
    f = r.to_frame()
    assert_allclose(f.investment, 0.)
    assert_allclose(f.new_debt, 0.)
    assert_allclose(f.scrapping, 0.)
    assert_allclose(f.next_stock, (1-p.depreciation)*f.stock)
    assert (r.repayment_adaptation_policy == -1).all()
    assert_allclose(r.prices, 0.)
    assert audit_sovereign_adaptation(r)["accepted"]


def test_lender_prices_next_default_and_contract_relief_exactly():
    p = SovereignAdaptationParameters(disaster_relief=.6)
    r = solve(parameters=p)
    payoff = 1-p.disaster_relief*r.disaster
    for t in range(r.horizon-1):
        for b in range(len(r.debt_grid)):
            for a in range(len(r.adaptation_grid)):
                expected = r.transition@((1-r.default[t+1,b,a])*payoff)/p.gross_rate
                assert_allclose(r.prices[t,b,a], expected, atol=1e-14)
    assert_allclose(r.prices[:-1,0,0], np.broadcast_to(r.transition@payoff/p.gross_rate, (r.horizon-1,2)))


def test_stock_protection_is_predetermined_and_grants_close_resources():
    r = solve(parameters=SovereignAdaptationParameters(grant_rate=.5))
    p = r.parameters
    for t in range(r.horizon):
        for excluded in (False, True):
            f = r.to_frame(date=t, excluded=excluded)
            assert_allclose(f.disaster_damage, f.income*f.damage_fraction*np.exp(-p.effectiveness*f.stock))
            assert_allclose(f.next_stock, (1-p.depreciation)*f.stock+f.investment-f.scrapping)
            assert_allclose(f.grant, p.grant_rate*f.investment)
            assert_allclose(f.consumption+f.investment+f.debt_service,
                            f.income-f.disaster_damage-f.exclusion_loss+f.borrowing_proceeds+f.grant)
            assert_allclose(f.consumption+f.net_tax, f.income-f.disaster_damage-f.exclusion_loss)
            assert_allclose(f.debt, f.debt_service+f.contract_relief+f.default_writeoff)
            assert (f.consumption > 0).all()


@pytest.mark.parametrize("kwargs", [{"damage": [0.,0.]}, {"parameters": SovereignAdaptationParameters(effectiveness=0.)}])
def test_without_damage_benefits_no_new_adaptation(kwargs):
    r = solve(**kwargs)
    d = r.distribution([1.,0.])
    assert_allclose(d.to_frame().investment, 0.)
    baseline = solve(adaptation_grid=[0.], **kwargs)
    assert_allclose(r.access_value[:,:,0], baseline.access_value[:,:,0], atol=1e-13)


def test_zero_default_penalty_implies_zero_priced_positive_debt():
    r = solve(parameters=SovereignAdaptationParameters(default_loss=0.))
    assert_allclose(r.prices[:-1,1:], 0.)
    assert_allclose(r.distribution([1.,0.]).to_frame().new_debt, 0.)


def test_full_disaster_relief_has_no_payment_and_is_not_default():
    r = solve(horizon=1, parameters=SovereignAdaptationParameters(disaster_relief=1.))
    assert not r.default[...,1].any()
    f = r.to_frame().query("state == 1")
    assert_allclose(f.debt_service, 0.)
    assert_allclose(f.contract_relief, f.debt)
    assert_allclose(f.default_writeoff, 0.)


def test_default_entry_and_permanent_exclusion_mass_are_distinct():
    r = solve(debt_grid=[0., 4.])
    d = r.distribution([0.,1.], debt_index=1)
    f = d.to_frame()
    assert_allclose(f.new_default_probability, [1.,0.,0.])
    assert_allclose(f.excluded_probability, 1.)
    assert_allclose(f.new_debt, 0.)
    assert audit_adaptation_distribution(d)["accepted"]
    assert_allclose(d.value, r.access_value[0,1,0,1], atol=1e-13)


def test_distribution_value_equals_initial_bellman_value_and_log_ce():
    r = solve()
    d = r.distribution([.6,.4], debt_index=1, adaptation_index=1)
    assert_allclose(d.value, r.access_value[0,1,1]@[.6,.4], atol=1e-12)
    assert_allclose(d.access_mass.sum(axis=(1,2,3))+d.excluded_mass.sum(axis=(1,2)), 1.)
    assert d.consumption_equivalent(d) == 0.
    grant = solve(parameters=SovereignAdaptationParameters(grant_rate=.5)).distribution([.6,.4], debt_index=1, adaptation_index=1)
    assert_allclose(grant.consumption_equivalent(d), np.expm1((grant.value-d.value)/sum(r.parameters.beta**t for t in range(r.horizon))))
    with pytest.raises(ValueError): d.consumption_equivalent(solve(horizon=1).distribution([1.,0.]))


def test_input_ownership_fresh_frames_and_fingerprints():
    grid=np.array([0.,.05,.15]); r=solve(adaptation_grid=grid); grid[:]=9
    assert_allclose(r.adaptation_grid,[0.,.05,.15])
    for name in ("income","damage","disaster","transition","debt_grid","adaptation_grid","access_value",
                 "repayment_value","excluded_value","prices","default","repayment_debt_policy"):
        assert not getattr(r,name).flags.writeable
    frame=r.to_frame(); frame.loc[0,"consumption"]=-1
    assert r.to_frame().consumption.min()>0
    assert r.fingerprint==solve().fingerprint
    assert r.fingerprint!=solve(horizon=2).fingerprint
    assert r.to_markdown() and r.to_latex() and r.to_typst()


def test_infeasible_repayment_is_retained_not_clipped():
    r=solve(horizon=1,debt_grid=[0.,4.])
    assert np.isneginf(r.repayment_value[:,1:]).all()
    f=r.to_frame().query("debt_index == 1")
    assert not f.repayment_feasible.any()
    assert f.repayment_value.isna().all()
    assert f.new_default.all()


def test_auditor_rejects_price_policy_value_and_mass_corruption():
    r=solve()
    for name in ("access_value","excluded_value","prices"):
        a=getattr(r,name).copy(); a.flat[0]+=.1
        assert not audit_sovereign_adaptation(replace(r,**{name:a}))["accepted"]
    policy=r.excluded_adaptation_policy.copy(); policy[0,0,0]=99
    assert not audit_sovereign_adaptation(replace(r,excluded_adaptation_policy=policy))["accepted"]
    d=r.distribution([1.,0.]); mass=d.access_mass.copy(); mass.flat[0]+=.1
    assert not audit_adaptation_distribution(replace(d,access_mass=mass))["accepted"]


@pytest.mark.parametrize("kwargs", [
    {"beta":0},{"gross_rate":0},{"depreciation":1.1},{"effectiveness":-1},
    {"default_loss":1},{"grant_rate":-.1},{"disaster_relief":1.1},{"beta":True},
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError): SovereignAdaptationParameters(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"income":[0,1]},{"damage":[0,1]},{"disaster":[0,1]},
    {"damage":[.1,.4]}, {"transition":[[.9,.2],[.1,.9]]},
    {"transition":[[1.,0.],[-.1,1.1]]}, {"horizon":True},{"horizon":0},
    {"debt_grid":[.1,.2]},{"adaptation_grid":[0,.1,.1]},
    {"decision_tolerance":1e-3},
])
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError): solve(**kwargs)


@pytest.mark.parametrize("kwargs", [{"debt_index":-1},{"adaptation_index":9},{"debt_index":True},
    {"market_access":False,"debt_index":1}])
def test_invalid_distribution_initial_state(kwargs):
    with pytest.raises(ValueError): solve().distribution([1.,0.],**kwargs)
