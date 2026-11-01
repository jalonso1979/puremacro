import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from puremacro.asset_pricing import simulate_brock_hommes


def scalar_reference(periods,g,b,c,R,beta,memory,risk,initial,shocks):
    # Literal scalar equations, independent of production vectorization/helpers.
    history=list(initial)
    utility=[(history[-1]-R*history[-2])*(g[j]*history[-3]+b[j]-R*history[-2])/risk-c[j] for j in range(len(g))]
    output=[]; fractions=[]
    for t in range(periods):
        scores=[math.exp(beta*(v-max(utility))) for v in utility]
        shares=[s/sum(scores) for s in scores]
        now=(sum(shares[j]*(g[j]*history[-1]+b[j]) for j in range(len(g)))+shocks[t])/R
        utility=[(now-R*history[-1])*(g[j]*history[-2]+b[j]-R*history[-1])/risk-c[j]+memory*utility[j] for j in range(len(g))]
        history.append(now);output.append(now);fractions.append(shares)
    return np.array(output),np.array(fractions)


def test_independent_profit_and_selection_timing_with_memory_and_costs():
    g=[0,.9,1.1];b=[0,.1,-.1];c=[.02,0,.01];R=1.01;beta=3;mem=.5;risk=.7
    initial=[.03,-.02,.05]; shocks=np.random.default_rng(4).normal(0,.01,60)
    out=simulate_brock_hommes(60,slopes=g,biases=b,costs=c,gross_rate=R,intensity=beta,
        memory=mem,risk_scale=risk,initial_deviations=initial,shocks=shocks)
    x,n=scalar_reference(60,g,b,c,R,beta,mem,risk,initial,shocks)
    assert_allclose(out.deviations,x,atol=1e-13,rtol=1e-12)
    assert_allclose(out.fractions,n,atol=1e-13,rtol=1e-12)
    assert_allclose(out.fractions.sum(1),1,atol=1e-15)


def test_homogeneous_fundamentalists_recover_fundamental_price():
    out=simulate_brock_hommes(20,slopes=[0,0],biases=[0,0],intensity=90.5)
    assert_allclose(out.deviations,0,atol=0)
    assert_allclose(out.prices,100,atol=0)


def test_zero_intensity_has_closed_form_linear_dynamics():
    out=simulate_brock_hommes(30,slopes=[0,.8],biases=[0,.2],gross_rate=1.1,
                              intensity=0,initial_deviations=[0,0,.1])
    a=.4/1.1; b=.1/1.1; fixed=b/(1-a)
    expected=fixed+(.1-fixed)*a**np.arange(1,31)
    assert_allclose(out.deviations,expected,atol=2e-16)
    assert_allclose(out.fractions,.5,atol=0)


def test_published_four_type_parameters_against_literal_equations():
    # Hommes chapter Fig.11 parameters; independent recurrence, not digitized figure parity.
    g=[0,.9,.9,1.01]; b=[0,.2,-.2,0]; initial=[.1,.12,.13]
    out=simulate_brock_hommes(30,slopes=g,biases=b,intensity=90.5,initial_deviations=initial)
    x,n=scalar_reference(30,g,b,[0]*4,1.01,90.5,0,1,initial,[0]*30)
    assert_allclose(out.deviations,x,atol=1e-11,rtol=1e-10)
    assert_allclose(out.fractions,n,atol=1e-10)


def test_shock_timing_seed_and_prefix_causality():
    shocks=np.zeros(30); shocks[12]=.2
    out=simulate_brock_hommes(30,shocks=shocks)
    zero=simulate_brock_hommes(30,shocks=np.zeros(30))
    assert_allclose(out.deviations[:12],zero.deviations[:12],atol=0,rtol=0)
    assert_allclose(out.fractions[:13],zero.fractions[:13],atol=0,rtol=0)
    assert out.deviations[12]-zero.deviations[12]==pytest.approx(.2/1.01)
    a=simulate_brock_hommes(30,noise_std=.01,seed=12)
    b=simulate_brock_hommes(30,noise_std=.01,seed=12)
    assert_allclose(a.deviations,b.deviations,atol=0,rtol=0)


def test_extreme_logit_and_divergence_are_explicit():
    out=simulate_brock_hommes(3,initial_fitness=[1e200,-1e200],intensity=1e200)
    assert_allclose(out.fractions[0],[1,0])
    with pytest.raises(RuntimeError,match='period'):
        simulate_brock_hommes(30,slopes=[100],biases=[0],max_abs_deviation=10)
    with pytest.raises(ValueError):
        simulate_brock_hommes(3,shocks=[0]*3,noise_std=.1)
    assert out.to_latex() and out.to_typst() and out.to_markdown()
    assert out.plot() is not None
