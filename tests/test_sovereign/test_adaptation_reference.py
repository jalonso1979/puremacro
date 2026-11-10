"""Scalar dynamic program, analytic investment and full-history references."""
from dataclasses import replace
from functools import lru_cache
from itertools import product
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import brentq

from puremacro.sovereign.adaptation import SovereignAdaptationParameters, solve_sovereign_adaptation


def small(**kwargs):
    spec=dict(income=[1.,.9],damage=[0.,.35],disaster=[False,True],transition=[[.8,.2],[.6,.4]],
              debt_grid=[0.,.12,.28],adaptation_grid=[0.,.08,.16],horizon=3)
    spec.update(kwargs)
    return solve_sovereign_adaptation(**spec)


def scalar_reference(r):
    p=r.parameters; T=r.horizon; b=r.debt_grid; a=r.adaptation_grid; nz=len(r.income)
    payoff=1-p.disaster_relief*r.disaster

    def income(ai,z):
        return float(r.income[z]*(1-r.damage[z]*math.exp(-p.effectiveness*a[ai])))

    @lru_cache(None)
    def excluded(t,ai,z):
        if t==T-1: return math.log((1-p.default_loss)*income(ai,z)),-1
        choices=[]
        for nxt,stock in enumerate(a):
            investment=max(float(stock-(1-p.depreciation)*a[ai]),0.)
            c=(1-p.default_loss)*income(ai,z)-(1-p.grant_rate)*investment
            if c<=0: continue
            value=math.log(c)+p.beta*sum(r.transition[z,zz]*excluded(t+1,nxt,zz)[0] for zz in range(nz))
            choices.append((value,nxt))
        return max(choices,key=lambda item:(item[0],-item[1]))

    @lru_cache(None)
    def access(t,bi,ai,z):
        candidates=[]
        actions=[(0,-1)] if t==T-1 else product(range(len(b)),range(len(a)))
        for bn,an in actions:
            if t==T-1:
                c=income(ai,z)-payoff[z]*b[bi]; future=0.
            else:
                investment=max(float(a[an]-(1-p.depreciation)*a[ai]),0.)
                price=sum(r.transition[z,zz]*payoff[zz]*(not access(t+1,bn,an,zz)[1]) for zz in range(nz))/p.gross_rate
                c=income(ai,z)-payoff[z]*b[bi]-(1-p.grant_rate)*investment+price*b[bn]
                future=p.beta*sum(r.transition[z,zz]*access(t+1,bn,an,zz)[0] for zz in range(nz))
            value=math.log(c)+future if c>0 else -math.inf
            candidates.append((value,bn,an))
        rep=max(candidates,key=lambda item:(item[0],-item[1],-item[2]))
        ev,ea=excluded(t,ai,z)
        default=ev>rep[0]+r.decision_tolerance
        return (ev if default else rep[0],default,rep[0],rep[1],rep[2],ea)
    return access,excluded


@pytest.mark.parametrize("params", [SovereignAdaptationParameters(),
    SovereignAdaptationParameters(grant_rate=.5),SovereignAdaptationParameters(disaster_relief=1.),
    SovereignAdaptationParameters(depreciation=1.,default_loss=0.)])
def test_independent_scalar_dynamic_program_matches_every_state(params):
    r=small(parameters=params); access,excluded=scalar_reference(r)
    for t,bi,ai,z in product(range(r.horizon),range(len(r.debt_grid)),range(len(r.adaptation_grid)),range(len(r.income))):
        value,default,rep,_,_,_=access(t,bi,ai,z)
        assert_allclose(r.access_value[t,bi,ai,z],value,atol=2e-13)
        assert_allclose(r.repayment_value[t,bi,ai,z],rep,atol=2e-13)
        assert r.default[t,bi,ai,z]==default
        assert_allclose(r.excluded_value[t,ai,z],excluded(t,ai,z)[0],atol=2e-13)


@pytest.mark.parametrize("initial_access", [True,False])
def test_complete_shock_histories_match_forward_mass_accounts(initial_access):
    r=small(parameters=SovereignAdaptationParameters(grant_rate=.3,disaster_relief=.4))
    p=r.parameters; probabilities=[.4,.6]; initial_b=1 if initial_access else 0
    d=r.distribution(probabilities,debt_index=initial_b,adaptation_index=1,market_access=initial_access)
    columns=["consumption","investment","disaster_damage","new_default_probability","excluded_probability","grant","new_debt","scrapping"]
    expected=np.zeros((r.horizon,len(columns))); expected_utility=0.
    for states in product(range(len(r.income)),repeat=r.horizon):
        probability=probabilities[states[0]]
        for t in range(1,r.horizon): probability*=r.transition[states[t-1],states[t]]
        bi,ai,market=initial_b,1,initial_access
        for t,z in enumerate(states):
            default=bool(market and r.default[t,bi,ai,z]); excluded=not market or default
            stock=float(r.adaptation_grid[ai]); debt=float(r.debt_grid[bi])
            damage=float(r.income[z]*r.damage[z]*math.exp(-p.effectiveness*stock))
            net=float(r.income[z]-damage)
            if t<r.horizon-1:
                bn=0 if excluded else int(r.repayment_debt_policy[t,bi,ai,z])
                an=int(r.excluded_adaptation_policy[t,ai,z] if excluded else r.repayment_adaptation_policy[t,bi,ai,z])
                investment=max(float(r.adaptation_grid[an]-(1-p.depreciation)*stock),0.)
                scrap=max(float((1-p.depreciation)*stock-r.adaptation_grid[an]),0.)
                new_debt=float(r.debt_grid[bn]); proceeds=0. if excluded else float(r.prices[t,bn,an,z]*new_debt)
            else: bn,an,investment,scrap,new_debt,proceeds=0,ai,0.,0.,0.,0.
            grant=p.grant_rate*investment
            service=0. if excluded else (1-p.disaster_relief*r.disaster[z])*debt
            consumption=(1-p.default_loss*excluded)*net-service-investment+grant+proceeds
            expected[t]+=probability*np.array([consumption,investment,damage,default,excluded,grant,new_debt,scrap])
            expected_utility+=probability*p.beta**t*math.log(consumption)
            bi,ai,market=bn,an,not excluded
    assert_allclose(d.to_frame()[columns],expected,atol=2e-13)
    assert_allclose(d.value,expected_utility,atol=2e-13)


def test_two_date_autarky_investment_near_independent_analytic_optimum():
    p=SovereignAdaptationParameters(beta=.96,effectiveness=6.,grant_rate=.2)
    r=solve_sovereign_adaptation(income=[1.,1.],damage=[0.,.4],disaster=[False,True],
        transition=[[0.,1.],[0.,1.]],debt_grid=[0.],adaptation_grid=np.linspace(0,.5,101),horizon=2,parameters=p)
    def derivative(x):
        e=math.exp(-6*x)
        return -.8/(1-.8*x)+.96*.4*6*e/(1-.4*e)
    optimum=brentq(derivative,0.,.5,xtol=1e-14)
    choice=r.distribution([1.,0.]).to_frame().investment.iloc[0]
    assert abs(choice-optimum)<.005
    assert choice>0
    assert_allclose(r.to_frame().query('debt_index == 0 and adaptation_index == 0 and state == 0').disaster_damage,0.)


def test_free_disposal_has_no_sale_proceeds_or_created_stock():
    r=small(parameters=SovereignAdaptationParameters(effectiveness=0.,depreciation=.25))
    f=r.to_frame().query('adaptation_index == 2')
    assert_allclose(f.investment,0.)
    assert_allclose(f.next_stock,0.)
    assert_allclose(f.scrapping,.12)
    assert_allclose(f.borrowing_proceeds,f.new_debt*f.bond_price)


def test_grant_weakly_improves_no_borrowing_value():
    base=small(debt_grid=[0.])
    grant=small(debt_grid=[0.],parameters=SovereignAdaptationParameters(grant_rate=.5))
    assert np.min(grant.access_value-base.access_value)>-1e-13
    assert np.min(grant.excluded_value-base.excluded_value)>-1e-13


@pytest.mark.parametrize("scale", [.01,100.])
def test_units_transform_stock_effectiveness_and_log_values(scale):
    r=small()
    s=small(income=r.income*scale,debt_grid=r.debt_grid*scale,adaptation_grid=r.adaptation_grid*scale,
            parameters=replace(r.parameters,effectiveness=r.parameters.effectiveness/scale))
    assert np.array_equal(r.default,s.default)
    assert_allclose(r.prices,s.prices,atol=1e-14)
    for t in range(r.horizon):
        shift=sum(r.parameters.beta**j for j in range(r.horizon-t))*math.log(scale)
        assert_allclose(s.access_value[t],r.access_value[t]+shift,atol=1e-12)
    assert_allclose(s.distribution([1.,0.]).to_frame().investment,r.distribution([1.,0.]).to_frame().investment*scale,atol=1e-12)


def test_state_permutation_preserves_full_solution():
    r=small(); ix=[1,0]
    s=small(income=r.income[ix],damage=r.damage[ix],disaster=r.disaster[ix],transition=r.transition[np.ix_(ix,ix)])
    assert_allclose(s.access_value,r.access_value[:,:,:,ix],atol=1e-13)
    assert_allclose(s.prices,r.prices[:,:,:,ix],atol=1e-13)
    assert_allclose(s.distribution([0.,1.]).to_frame(),r.distribution([1.,0.]).to_frame(),atol=1e-13)


def test_memory_work_limits_reject_before_allocating():
    with pytest.raises(ValueError,match="grid budget"):
        small(debt_grid=np.linspace(0,1,1000),adaptation_grid=np.linspace(0,1,1000))


def test_pricing_frame_rejects_terminal_quotes_and_invalid_initial_mass():
    r=small(horizon=1)
    with pytest.raises(ValueError): r.pricing_frame()
    with pytest.raises(ValueError): r.distribution([.3,.3])
    with pytest.raises(ValueError): r.distribution([-.1,1.1])
