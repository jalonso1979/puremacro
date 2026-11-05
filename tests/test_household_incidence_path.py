"""Lifetime CEV identities independent of the transition evaluator."""
import numpy as np
import pandas as pd
import pytest

from puremacro.trade.distributional import prepare_household_groups, compute_distributional_welfare
from puremacro.trade.incidence_transition import household_incidence_path


def groups():
    return prepare_household_groups(pd.DataFrame({"food":[60.,40.],"other":[40.,160.]},index=["a","b"]),
        pd.Series({"a":3.,"b":1.}),monetary_unit="fixture",period="quarter",provenance={"source":"synthetic"})


@pytest.mark.parametrize("gamma", [.5, 1., 1+1e-9, 2., 4.])
def test_independent_two_date_cev(gamma):
    g=groups()
    p=pd.DataFrame({"food":[1.1,1.3],"other":[1.,1.]})
    r=household_incidence_path(g,p,discount=.95,risk_aversion=gamma)
    u=p.food.to_numpy()[:,None]**(-np.array([.6,.2]))
    w=np.array([1,.95])/1.95
    if abs(gamma-1)<1e-7:
        expected=np.exp(w@np.log(u))-1
    else:
        expected=(w@u**(1-gamma))**(1/(1-gamma))-1
    np.testing.assert_allclose(r.households.cev_pct,100*expected,atol=1e-8)


def test_constant_path_equals_static_ev_and_one_period_has_no_curvature_effect():
    g=groups()
    p=pd.DataFrame({"food":[1.2]*5,"other":[1.]*5})
    static=compute_distributional_welfare(g,p.iloc[0]*0+1,p.iloc[0],rule="ces",ces_elasticity=2)
    for gamma in [.5,1,5]:
        result=household_incidence_path(g,p,risk_aversion=gamma,ces_elasticity=2)
        np.testing.assert_allclose(result.households.cev,static.groups.ev,atol=1e-10)


def test_signed_fiscal_paths_conserve_resources_and_missing_funding_fails():
    g=groups()
    p=pd.DataFrame({"food":[1.,1.],"other":[1.,1.]})
    tr=pd.DataFrame({"a":[2.,4.],"b":[-6.,-12.]})
    r=household_incidence_path(g,p,transfers=tr)
    np.testing.assert_allclose(r.periods.total_transfers,0,atol=1e-12)
    assert r.households.loc['a','cev']>0
    assert r.households.loc['b','cev']<0
    with pytest.raises(ValueError,match="funding"):
        household_incidence_path(g,p,transfers=tr+1)


@pytest.mark.pyodide_smoke
def test_portable_lifetime_incidence():
    test_independent_two_date_cev(2.)
