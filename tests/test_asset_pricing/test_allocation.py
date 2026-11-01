import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from puremacro.asset_pricing import estimate_covariance, equal_weight, optimize_portfolio, efficient_frontier


def test_diagonal_minimum_variance_and_scale_invariance():
    sigma = np.diag([.01, .04, .09])
    expected = 1 / np.diag(sigma); expected /= expected.sum()
    for scale in [1e-8, 1, 1e8]:
        out = optimize_portfolio(sigma*scale)
        assert_allclose(out.weights, expected, atol=2e-7)
        assert out.optimality_gap < 1e-7*scale
        assert np.isnan(out.expected_return)
        assert out.to_markdown() and out.to_latex() and out.to_typst()


def test_caps_and_mean_variance_against_exhaustive_two_asset_reference():
    sigma = np.array([[.04, .01], [.01, .09]])
    mu = np.array([.02, .03])
    w0 = np.linspace(.2, .65, 100001)
    w = np.column_stack([w0, 1-w0])
    objective = .5*3*np.einsum('ti,ij,tj->t', w, sigma, w)-w@mu
    out = optimize_portfolio(sigma, expected_returns=mu, risk_aversion=3, max_weight=[.65,.8])
    assert_allclose(out.weights, w[objective.argmin()], atol=5e-6)
    assert out.constraint_violation < 1e-12


def test_turnover_penalty_retains_previous_and_frontier_hits_floor():
    out = optimize_portfolio(np.diag([.01,.1]), previous_weights=[.4,.6], turnover_penalty=1)
    assert_allclose(out.weights, [.4,.6], atol=1e-8)
    frontier = efficient_frontier(np.diag([.01,.04]), [.01,.05], [.01,.03,.045])
    # Unconstrained min variance has weight .8 in the first asset, return .018.
    assert_allclose(frontier.expected_return, [.018,.03,.045], atol=1e-8)
    assert_allclose(frontier['weight:a0'], [.8,.5,.125], atol=1e-7)


def test_ledoit_wolf_matches_independent_outer_product_formula():
    x = np.random.default_rng(63).normal(size=(80,5)) * np.arange(1,6)*.005
    data = pd.DataFrame(x, index=pd.date_range('2000',periods=len(x)), columns=list('abcde'))
    z = x-x.mean(0)
    outer = np.array([np.outer(row,row) for row in z])
    s = outer.mean(0); target = np.eye(5)*np.trace(s)/5
    error = ((outer-s)**2).sum()/len(x)**2
    alpha = min(1, error/((s-target)**2).sum())
    out = estimate_covariance(data)
    assert_allclose(out.covariance, (1-alpha)*s+alpha*target, atol=1e-16)
    assert_allclose(out.shrinkage, alpha, atol=1e-14)
    assert_allclose(estimate_covariance(data,method='sample').covariance, np.cov(x,rowvar=False))
    with pytest.raises(ValueError):
        out.covariance[0,0] = 0


def test_labels_and_sample_means_are_not_implicit_forecasts():
    sigma = pd.DataFrame(np.diag([.01,.04]), index=['a','b'],columns=['a','b'])
    out = optimize_portfolio(sigma, expected_returns=pd.Series({'b':.02,'a':.01}))
    assert list(out.as_series().index) == ['a','b']
    assert_allclose(equal_weight(['b','a']), [.5,.5])
    with pytest.raises(ValueError,match='labels'):
        optimize_portfolio(sigma, expected_returns=pd.Series({'b':.02,'c':.01}))


@pytest.mark.parametrize('kwargs', [dict(max_weight=.4), dict(expected_returns=[.01,.02],target_return=.03),
                                    dict(turnover_penalty=.1), dict(previous_weights=[-.1,1.1])])
def test_infeasible_or_underspecified_allocation_raises(kwargs):
    with pytest.raises(ValueError):
        optimize_portfolio(np.eye(2), **kwargs)


def test_indefinite_and_asymmetric_matrices_rejected():
    for s in [np.array([[1,2],[2,1]]), np.array([[1,.1],[.3,1]])]:
        with pytest.raises(ValueError):
            optimize_portfolio(s)


@pytest.mark.parametrize("negative", [-1e-12, -1e-16])
def test_nearly_indefinite_input_never_reports_negative_risk(negative):
    with pytest.raises((ValueError, RuntimeError), match="positive semidefinite|negative portfolio variance"):
        optimize_portfolio(np.diag([1., negative]))
