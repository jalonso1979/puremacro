"""Official-data identity, explicit exclusions and observed-data evaluation."""
import numpy as np
import pytest
from puremacro.examples.household_trade_policy import load_inputs,run_study


@pytest.mark.pyodide_smoke
def test_official_sample_is_authenticated_and_zero_budgets_are_recorded():
    groups,data,manifest=load_inputs()
    assert manifest['source_rows']==90102
    assert len(data)==90033
    assert manifest['excluded_zero_spending_rows']==69
    assert manifest['population_households']==37560123
    assert manifest['excluded_zero_spending_population']==21945
    assert int(groups.population_weights.sum())==37538178
    assert manifest['weighted_means_all_rows']['ing_cor']==pytest.approx(63695.45790004601)
    assert not manifest['is_synthetic']
    np.testing.assert_allclose(groups.expenditure.sum(axis=1),data.gasto_mon)


def test_observed_study_has_fiscal_and_independent_welfare_evidence():
    r=run_study(price_changes=(.10,),elasticities=(1.,))
    assert len(r['aggregate'])==4
    assert len(r['incidence'])==20
    assert r['metadata']['maximum_scaled_independent_ev_gap']<1e-10
    assert np.max(np.abs(r['aggregate'].fiscal_residual))<1e-3
    comparison=r['comparisons']['bottom40_mean_ev']
    assert comparison.summary.loc['bottom_40_income','undominated']
    assert comparison.metadata['probabilities'] is None


def test_near_unit_elasticities_remain_distinct_and_match_the_cobb_douglas_limit():
    r = run_study(price_changes=(.1,), elasticities=(1.-1e-10, 1., 1.+1e-10))
    assert r['aggregate'].state.nunique() == 3
    assert r['metadata']['maximum_scaled_independent_ev_gap'] < 1e-10
    for _, group in r['aggregate'].groupby('policy'):
        assert np.ptp(group.mean_ev) < 1e-6
