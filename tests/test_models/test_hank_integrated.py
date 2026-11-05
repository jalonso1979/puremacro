"""Joint state/control reduction, nonlinear continuation and capital accounts."""
import numpy as np
import pytest

from puremacro.models import (CapitalHANKSteadyState, ReducedStochasticHANKPolicy,
    calibrate_capital_hank, solve_hank_reduced_policy, simulate_hank_nonlinear)


@pytest.fixture(scope="module")
def capital_policy():
    # This declares a smaller aggregate-risk domain; it does not relax any
    # equation/refinement criterion. The original coarse-grid sigma=1e-4
    # branch-depth failure is retained in the development evidence.
    steady = calibrate_capital_hank(n_a=15, n_b=15, a_max=200., b_max=30.)
    return solve_hank_reduced_policy(steady, distribution_rank=48, marginal_rank=128,
                                    krylov_steps=96, shock_std=1e-5)


def test_public_factory_preserves_productive_capital_closure(capital_policy):
    policy = capital_policy
    assert isinstance(policy, ReducedStochasticHANKPolicy)
    assert isinstance(policy.steady_state, CapitalHANKSteadyState)
    assert policy._kernel.model == "two_asset_productive_capital"
    assert policy.aggregate_state_names == ("lag_nominal", "monetary_shock", "log_capital")
    assert policy.shock_index == len(policy.transition)-2
    assert policy._kernel._derivatives is None
    assert policy.diagnostics["projected_equilibrium_residual_max_abs"] < 1e-8


def test_capital_receding_history_clears_both_markets_and_independent_accounts(capital_policy):
    policy = capital_policy
    result = simulate_hank_nonlinear(policy, np.array([-1e-6, 2e-6, 0.]),
        horizons=(4, 8, 16), branch_depths=(1, 2),
        horizon_tolerance=1e-6, branching_tolerance=1e-6, tolerance=1e-8)
    assert result.converged
    assert result.diagnostics["completed_periods"] == 3
    kernel, p = policy._kernel, policy._kernel.p
    h = kernel.hh
    previous_q, previous_i, K = 1., p["r"], p["K"]
    for date, v in result.prices.iterrows():
        d = result.distributions[date]
        c, a, b, chi = (result.policies[key][date] for key in ("c", "a", "b", "chi"))
        budget = c+a+b+chi-(1+v.r_a)*h.a_grid[:, None, None]-(1+v.r)*h.b_grid[None, :, None]-v.income*h.s_grid
        np.testing.assert_allclose(budget, 0., atol=1e-10, rtol=0)
        assert v.K == pytest.approx(K, abs=1e-12)
        assert v.lag_q == pytest.approx(previous_q, abs=1e-14)
        assert 1+v.r == pytest.approx((1+previous_i)/(1+v.pi), abs=1e-14)
        assert np.sum(d*a) == pytest.approx(v.q*v.Knext, abs=1e-8)
        assert np.sum(d*b) == pytest.approx(p["B"], abs=1e-8)
        assert np.sum(d*h.a_grid[:, None, None]) == pytest.approx(previous_q*K, abs=1e-8)
        assert v.tax == pytest.approx(v.r*p["B"]+p["G"], abs=1e-14)
        assert v.Knext == pytest.approx((1-p["delta"])*K+v.I, abs=1e-12)
        resources = v.Y-np.sum(d*(c+chi))-v.investment_goods-v.adj-p["G"]
        assert resources == pytest.approx(result.paths.loc[date, "goods_mkt"], abs=1e-12)
        assert abs(resources) < 1e-8
        assert (1+v.r_a)*previous_q*K == pytest.approx(
            v.rental*K-v.investment_goods+v.q*v.Knext, abs=1e-11)
        assert result.states[date, -1] == pytest.approx(np.log(K/p["K"]), abs=1e-12)
        previous_q, previous_i, K = v.q, v.nominal, v.Knext
    assert np.max(abs(result.distributions.sum(axis=(1, 2, 3))-1)) < 1e-12
    assert result.distributions.min() >= 0
    assert not result.diagnostics["global_accuracy_certified"]
    assert not result.diagnostics["household_grid_accuracy_certified"]
