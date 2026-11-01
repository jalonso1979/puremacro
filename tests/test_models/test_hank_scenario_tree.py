"""Deterministic reduction, information sets and independent node accounts."""
import numpy as np
import pandas as pd
import pytest

from puremacro.models.one_asset_hank import calibrate_one_asset_hank, solve_one_asset_hank
from puremacro.models.hank_scenario_tree import monetary_scenario_tree, solve_hank_scenario_tree


@pytest.fixture(scope="module")
def steady():
    return calibrate_one_asset_hank(n_a=60,n_e=3,a_max=150.)


@pytest.mark.parametrize("magnitude",[-.001,.001])
def test_chain_matches_existing_published_model_solver(steady,magnitude):
    tree=monetary_scenario_tree(horizon=6,innovations=(magnitude,),probabilities=(1.,),branch_date=0)
    result=solve_hank_scenario_tree(steady,tree)
    reference=solve_one_asset_hank(steady,horizon=6,magnitude=magnitude,nonlinear=True)
    assert result.converged and reference.converged
    np.testing.assert_allclose(result.paths,reference.paths[result.paths.columns],atol=3e-8,rtol=0)


def test_duplicate_histories_reduce_to_the_same_announced_path(steady):
    single=solve_hank_scenario_tree(steady,monetary_scenario_tree(horizon=5,innovations=(-.002,),probabilities=(1.,)))
    duplicate=solve_hank_scenario_tree(steady,monetary_scenario_tree(horizon=5,innovations=(-.002,-.002),probabilities=(.3,.7)))
    assert single.converged and duplicate.converged
    np.testing.assert_allclose(single.expected_paths(),duplicate.expected_paths(),atol=3e-8,rtol=0)
    np.testing.assert_allclose(single.policy_c[0],duplicate.policy_c[0],atol=1e-8)


def test_uncertainty_is_known_before_realization_and_node_budgets_clear(steady):
    tree=monetary_scenario_tree(horizon=6,innovations=(-.005,.005))
    r=solve_hank_scenario_tree(steady,tree)
    assert r.converged
    p=steady.calibration
    # An unresolved mean-zero shock is not the zero-shock deterministic path.
    assert abs(r.paths.iloc[0].C)>1e-6
    assert np.max(np.abs(r.policy_c[0]-steady.policy_c))>1e-5
    assert r.metadata["household_distribution_solved"]
    assert not r.metadata["general_stationary_aggregate_risk_solver"]
    # Before different children observe the branch, inherited distributions coincide.
    np.testing.assert_array_equal(r.distributions[1],r.distributions[2])
    assert np.max(np.abs(r.policy_a[1]-r.policy_a[2]))>1e-4
    np.testing.assert_allclose(r.nodes.groupby("period").probability.sum(),1,atol=1e-14)
    for i,node in enumerate(r.nodes.index):
        row=r.paths.loc[node]
        realized=p['r']+row.r
        if i==0:
            nominal=p['r']
        else:
            parent=r.nodes.loc[node,'parent']
            nominal=p['r']+tree.loc[parent,'monetary_shock']+p['phi']*r.paths.loc[parent,'pi']
        assert 1+realized==pytest.approx((1+nominal)/(1+row.pi))
        tax=steady.aggregates['Tax']+row.Tax
        assert tax==pytest.approx(realized*p['B'])
        wage=p['w']+row.w
        div=steady.aggregates['Div']+row.Div
        budget=(1+realized)*steady.asset_grid[:,None]+wage*steady.e_grid*r.policy_n[i]+(div-tax)*steady.e_grid
        np.testing.assert_allclose(r.policy_c[i]+r.policy_a[i],budget,atol=1e-10)
        # Conditional, not unconditional, probabilities belong in each NKPC.
        children=r.nodes.index[r.nodes.parent==node]
        expected=0.
        for child in children:
            q=tree.loc[child,'conditional_probability']
            other=r.paths.loc[child]
            expected+=q*(1+other.Y)/(1+row.Y)*np.log1p(other.pi)/(1+p['r']+other.r)
        assert abs(p['kappa']*row.w+expected-np.log1p(row.pi))<1e-8
    assert r.diagnostics['mass_error']<1e-12
    assert r.diagnostics['household_budget_error']<1e-10


def test_probabilities_change_the_pre_realization_solution(steady):
    a=solve_hank_scenario_tree(steady,monetary_scenario_tree(horizon=4,probabilities=(.2,.8)))
    b=solve_hank_scenario_tree(steady,monetary_scenario_tree(horizon=4,probabilities=(.8,.2)))
    assert a.converged and b.converged
    assert abs(a.paths.iloc[0].C-b.paths.iloc[0].C)>1e-4


@pytest.mark.pyodide_smoke
def test_repeated_branches_preserve_distinct_history_distributions(steady):
    # Two unresolved innovations: leaves sharing a parent inherit the same
    # distribution, while histories with different past shocks do not.
    tree = pd.DataFrame({
        "parent": [None, "root", "root", "low", "low", "high", "high"],
        "conditional_probability": [1., .4, .6, .25, .75, .8, .2],
        "monetary_shock": [0., -.004, .004, -.003, .001, -.001, .003],
    }, index=["root", "low", "high", "ll", "lh", "hl", "hh"])
    result = solve_hank_scenario_tree(steady, tree)
    assert result.converged
    np.testing.assert_allclose(result.nodes.probability, [1., .4, .6, .1, .3, .48, .12])
    np.testing.assert_array_equal(result.distributions[3], result.distributions[4])
    np.testing.assert_array_equal(result.distributions[5], result.distributions[6])
    assert np.max(np.abs(result.distributions[3]-result.distributions[5])) > 1e-6
    leaf_c = result.paths.loc[["ll", "lh", "hl", "hh"], "C"].to_numpy()
    assert result.expected_paths().loc[2, "C"] == pytest.approx(np.dot([.1, .3, .48, .12], leaf_c))


def test_failed_solve_does_not_claim_equilibrium(steady):
    r=solve_hank_scenario_tree(steady,monetary_scenario_tree(horizon=4),max_evaluations=1)
    assert not r.converged
    assert r.diagnostics['market_residual_max_abs']>r.diagnostics['tolerance']


def test_hybrid_root_matches_a_separate_least_squares_optimizer(steady, monkeypatch):
    from scipy.optimize import least_squares
    import puremacro.models.hank_scenario_tree as model
    tree = monetary_scenario_tree(horizon=6, innovations=(-.005, .005), probabilities=(.3, .7))
    hybrid = solve_hank_scenario_tree(steady, tree)

    def reference_optimizer(fun, x, **kwargs):
        return least_squares(fun, x, xtol=1e-11, ftol=1e-11, gtol=1e-11, x_scale="jac")

    monkeypatch.setattr(model, "root", reference_optimizer)
    reference = solve_hank_scenario_tree(steady, tree)
    assert hybrid.converged and reference.converged
    np.testing.assert_allclose(hybrid.paths, reference.paths, atol=3e-8, rtol=0)
    np.testing.assert_allclose(hybrid.policy_c, reference.policy_c, atol=3e-8, rtol=0)


@pytest.mark.parametrize("mutation",[
    lambda t:t.assign(conditional_probability=.7),
    lambda t:t.assign(monetary_shock=np.nan),
    lambda t:t.rename(index={t.index[1]:t.index[0]}),
    lambda t:t.iloc[:-1],
])
def test_invalid_tree_refused(steady,mutation):
    with pytest.raises(ValueError):
        solve_hank_scenario_tree(steady,mutation(monetary_scenario_tree(horizon=4)))


@pytest.mark.pyodide_smoke
def test_tree_probabilities_are_conditional_and_normalized():
    t=monetary_scenario_tree(horizon=4,probabilities=(.25,.75),branch_date=2)
    assert t.loc['branch0:2','conditional_probability']==.25
    assert t.loc['branch1:3','conditional_probability']==1.
    assert len(t)==6
