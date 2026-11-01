"""Independent conditional accounts, nonanticipation and refinement failures."""
from dataclasses import replace

import numpy as np
import pytest

from puremacro.models import calibrate_one_asset_hank, calibrate_two_asset_equilibrium, solve_hank_stochastic_policy
from puremacro.models.hank_nonlinear_continuation import (
    HANKContinuationAccuracyError, correct_hank_stochastic_state,
    simulate_hank_nonlinear, solve_hank_nonlinear_continuation)


@pytest.fixture(scope="module")
def policy():
    ss = calibrate_one_asset_hank(n_a=40, n_e=2, a_max=120.)
    return solve_hank_stochastic_policy(ss, distribution_rank=12, shock_std=1e-4)


@pytest.fixture(scope="module")
def full(policy):
    return solve_hank_stochastic_policy(policy.steady_state, distribution_rank=None, shock_std=1e-4)


@pytest.fixture(scope="module")
def correction(policy):
    return solve_hank_nonlinear_continuation(policy, aggregate_state=np.array([0., -1e-4]),
                                             horizon=4, branch_depth=2)


@pytest.mark.pyodide_smoke
def test_dense_and_matrix_free_conditional_equilibria_agree(policy, correction):
    dense = solve_hank_nonlinear_continuation(policy, aggregate_state=np.array([0., -1e-4]),
                                             horizon=4, branch_depth=2, method="hybr")
    assert correction.converged and dense.converged
    np.testing.assert_allclose(correction.tree.paths, dense.tree.paths, atol=2e-8, rtol=2e-5)
    np.testing.assert_allclose(correction.post_distributions, dense.post_distributions, atol=2e-8)
    assert dense.diagnostics["dense_jacobian_assembled"]
    assert not correction.diagnostics["dense_jacobian_assembled"]


def test_independent_node_expectations_accounts_and_full_transport(policy, correction):
    result, ss, p = correction, policy.steady_state, policy.steady_state.calibration
    nodes = result.tree.nodes
    for i, label in enumerate(nodes.index):
        d, prices = result.tree.distributions[i], result.prices.iloc[i]
        c, a, labor = [result.tree.policies[name][i] for name in ("c", "a", "n")]
        budget = c+a-(1+prices.r)*ss.asset_grid[:, None]-prices.w*ss.e_grid*labor-prices.transfer*ss.e_grid
        np.testing.assert_allclose(budget, 0., atol=2e-10)
        assert abs(np.sum(d*a)-p["B"]) < 1e-8
        assert abs(np.sum(d*labor*ss.e_grid)-prices.Y) < 1e-8
        assert abs(prices.Y-np.sum(d*c)-prices.adj) < 1e-8
        child_positions = np.flatnonzero(nodes.parent.to_numpy() == label)
        if len(child_positions):
            weights = nodes.iloc[child_positions].conditional_probability.to_numpy()
            expected = np.einsum("i,ij->j", weights, result.marginal_values[child_positions])
            np.testing.assert_allclose(result.expected_marginals[i], expected, atol=1e-13)
            child = result.prices.iloc[child_positions]
            continuation = np.sum(weights*child.Y/prices.Y*np.log1p(child.pi)/(1+child.r))
            for j in child_positions:
                np.testing.assert_array_equal(result.tree.distributions[j], result.post_distributions[i])
            assert abs(p["kappa"]*(prices.w-p["w"])-np.log1p(prices.pi)+continuation) < 1e-8
    assert np.min(result.expected_marginals) > 0
    assert np.min(result.tree.distributions) >= 0
    assert np.min(result.post_distributions) >= 0
    np.testing.assert_allclose(result.post_distributions.sum(axis=(1, 2)), 1., atol=1e-13)


def test_conditional_shock_moments_and_mean_tail(policy, correction):
    nodes = correction.tree.nodes
    for label, parent in nodes.iterrows():
        children = nodes[nodes.parent == label]
        if children.empty:
            continue
        probabilities = children.conditional_probability.to_numpy()
        innovations = children.monetary_shock.to_numpy()-policy.rho*parent.monetary_shock
        assert np.dot(probabilities, innovations) == pytest.approx(0., abs=1e-16)
        variance = np.dot(probabilities, innovations**2)
        expected = policy.shock_std**2 if parent.period < 2 else 0.
        assert variance == pytest.approx(expected, abs=1e-20)


def test_later_realizations_cannot_change_earlier_actions(policy):
    options = dict(horizons=(4, 8), branch_depths=(1, 2), keep_trees=True)
    a = simulate_hank_nonlinear(policy, np.array([-1e-4, 4e-5, 1e-5]), **options)
    b = simulate_hank_nonlinear(policy, np.array([-1e-4, 4e-5, -9e-5]), **options)
    assert a.converged and b.converged
    np.testing.assert_array_equal(a.paths.iloc[:2], b.paths.iloc[:2])
    np.testing.assert_array_equal(a.distributions[:3], b.distributions[:3])
    np.testing.assert_array_equal(a.expected_marginals[:2], b.expected_marginals[:2])
    assert not np.array_equal(a.paths.iloc[2], b.paths.iloc[2])
    np.testing.assert_allclose(a.states[1, -2], a.prices.iloc[0].nominal-policy._kernel.r, atol=1e-15)
    np.testing.assert_allclose(a.states[1, -1], .7*(-1e-4)+4e-5, atol=1e-15)
    np.testing.assert_array_equal(a.corrections[1].tree.distributions[0], a.distributions[1])


def test_failed_refinement_is_retained_and_never_transported(policy):
    options = dict(horizons=(4, 8), branch_depths=(1, 2),
                   horizon_tolerance=1e-15, branching_tolerance=1e-15)
    failed = correct_hank_stochastic_state(policy, aggregate_state=np.array([0., -1e-4]),
                                            raise_on_failure=False, **options)
    assert not failed.converged
    assert failed.tree.converged  # Equations solved; approximation criterion failed.
    assert failed.diagnostics["adaptive_status"] == "refinement_limits_exhausted"
    assert not failed.diagnostics["horizon_converged"]
    assert not failed.diagnostics["branching_converged"]
    with pytest.raises(HANKContinuationAccuracyError) as caught:
        correct_hank_stochastic_state(policy, aggregate_state=np.array([0., -1e-4]), **options)
    assert not caught.value.result.converged
    stopped = simulate_hank_nonlinear(policy, np.array([-1e-4, 1e-4]), raise_on_failure=False, **options)
    assert not stopped.converged and stopped.diagnostics["completed_periods"] == 0
    assert stopped.diagnostics["failed_correction"]["period"] == 0
    np.testing.assert_array_equal(stopped.distributions, policy._kernel.d[None])
    assert stopped.states.shape == (0, policy.basis.rank+2)


@pytest.mark.parametrize("horizon_tolerance,branching_tolerance,expected", [
    (1e-15, 1e-3, (False, True)), (1e-3, 1e-15, (True, False))])
def test_horizon_and_branching_have_separate_statuses(policy, horizon_tolerance, branching_tolerance, expected):
    result = correct_hank_stochastic_state(policy, aggregate_state=np.array([0., -1e-4]),
        horizons=(4, 8), branch_depths=(1, 2), horizon_tolerance=horizon_tolerance,
        branching_tolerance=branching_tolerance, raise_on_failure=False)
    assert (result.diagnostics["horizon_converged"], result.diagnostics["branching_converged"]) == expected
    assert not result.converged


def test_true_nonlinear_future_branches_change_current_expectations(policy):
    risky = replace(policy, shock_std=1e-3)
    certain = replace(policy, shock_std=0.)
    a = solve_hank_nonlinear_continuation(risky, horizon=4, branch_depth=2)
    b = solve_hank_nonlinear_continuation(certain, horizon=4, branch_depth=2)
    assert a.converged and b.converged
    # Linear expected marginals at the same zero state have no variance shift.
    # Optimized nonlinear child marginals do; this is more than tail arithmetic.
    assert np.max(abs(a.expected_marginals[0]-b.expected_marginals[0])) > 1e-7
    assert np.max(abs(a.aggregate_coordinates[0]-b.aggregate_coordinates[0])) > 1e-7
    np.testing.assert_array_equal(risky.feedback, certain.feedback)


def test_longer_horizon_reference_and_separate_error_dimensions(full):
    short = correct_hank_stochastic_state(full, aggregate_state=np.array([0., -1e-4]))
    long = solve_hank_nonlinear_continuation(full, aggregate_state=np.array([0., -1e-4]),
                                            horizon=32, branch_depth=3)
    assert short.converged and long.converged
    np.testing.assert_allclose(short.aggregate_coordinates[0], long.aggregate_coordinates[0], atol=2e-6)
    assert long.diagnostics["market_residual_max_abs"] < 1e-8
    assert long.diagnostics["euler_kkt_max"] > 1e-5
    assert long.diagnostics["terminal_compression_l1"] < 1e-7
    assert not long.diagnostics["global_accuracy_certified"]
    assert not long.diagnostics["household_grid_accuracy_certified"]


def test_quadrature_order_is_distinct_from_branch_depth(policy):
    options = dict(horizon=4, branch_depth=2, aggregate_state=np.array([0., -1e-4]))
    two = solve_hank_nonlinear_continuation(policy, quadrature_order=2, **options)
    three = solve_hank_nonlinear_continuation(policy, quadrature_order=3, **options)
    assert two.converged and three.converged
    assert len(two.tree.nodes) == 11 and len(three.tree.nodes) == 22
    assert two.diagnostics["quadrature_order_converged"] is None
    np.testing.assert_allclose(two.aggregate_coordinates[0], three.aggregate_coordinates[0], atol=1e-6)


def test_simulation_preserves_passed_criterion_on_branch_failure(policy):
    stopped = simulate_hank_nonlinear(policy, np.array([-1e-4]), raise_on_failure=False,
        horizons=(4, 8), branch_depths=(1, 2), horizon_tolerance=1e-3, branching_tolerance=1e-15)
    assert not stopped.converged
    assert stopped.diagnostics["equation_converged"]
    assert stopped.diagnostics["horizon_converged"]
    assert not stopped.diagnostics["branching_converged"]


def test_actual_previous_consol_quote_is_not_reinferred():
    ss = calibrate_two_asset_equilibrium(n_a=10, n_b=10, a_max=80.)
    policy = solve_hank_stochastic_policy(ss, distribution_rank=12)
    inherited_q = 1.0001
    result = solve_hank_nonlinear_continuation(policy, lag_q=inherited_q, horizon=3, branch_depth=1)
    root = result.prices.iloc[0]
    assert result.diagnostics["inherited_q"] == inherited_q
    np.testing.assert_allclose(1+root.r_a, (root.q+ss.calibration["r_a"])/inherited_q, atol=1e-14)
    assert abs((1+root.r_a)-(root.q+ss.calibration["r_a"])) > 1e-5
    # This deliberately inconsistent inherited quote/distribution does not
    # silently pass the resource-accounting gate.
    assert not result.converged


@pytest.mark.parametrize("options,match", [
    (dict(horizon=1), "horizon"), (dict(horizon=4, branch_depth=4), "branch_depth"),
    (dict(lag_q=0.), "lag_q"), (dict(max_nodes=3), "max_nodes"),
    (dict(max_workspace_bytes=100), "budget"), (dict(method="unknown"), "method"),
    (dict(method="hybr", max_workspace_bytes=400000), "dense reference"),
    (dict(quadrature_order=33), "quadrature_order"),
    (dict(horizon=10000000), "max_nodes")])
def test_resource_and_domain_guards(policy, options, match):
    with pytest.raises(ValueError, match=match):
        solve_hank_nonlinear_continuation(policy, **options)


def test_bad_distribution_and_refinement_design_are_rejected(policy):
    bad = policy._kernel.d.copy()
    bad[0, 0] = -.01
    with pytest.raises(ValueError, match="distribution"):
        solve_hank_nonlinear_continuation(policy, distribution=bad)
    with pytest.raises(ValueError, match="two increasing"):
        correct_hank_stochastic_state(policy, horizons=(8,))
    with pytest.raises(ValueError, match="two horizons"):
        correct_hank_stochastic_state(policy, horizons=(4, 8), branch_depths=(1, 4))
    with pytest.raises(ValueError, match="nonempty"):
        simulate_hank_nonlinear(policy, np.array([]))
