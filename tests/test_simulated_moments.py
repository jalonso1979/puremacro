"""Simulation uncertainty checked against independent covariance/GLS algebra."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from puremacro.structural import (
    MomentTargets, MomentReplication, SimulationPlan, SimulationMomentError,
    simulate_moments, moment_error_covariance, fit_simulated_moments,
)


def _linear_problem():
    design = np.array([[1., .2], [.3, 1.], [1., -1.]])
    loading = np.array([[.5, 0., 0.], [.2, .4, 0.], [-.1, .1, .3]])
    targets = MomentTargets([.8, -.25, 1.15],
                            [[.04, .012, 0.], [.012, .09, .01], [0., .01, .03]],
                            ("a", "b", "c"), units=("level",)*3)
    plan = SimulationPlan(seed=731, replications=32, periods=10)
    def simulator(theta, rng, plan):
        return design @ theta + (rng.standard_normal((plan.periods, 3)) @ loading.T).mean(axis=0)
    return design, targets, plan, simulator


def test_joint_simulation_covariance_is_variance_of_average_without_extra_period_scaling():
    vectors = iter([[1., 2.], [3., 4.], [5., 0.], [7., 2.]])
    plan = SimulationPlan(seed=9, replications=4, periods=1000)
    batch = simulate_moments(lambda t, r, p: next(vectors), [0.], plan,
                             labels=("a", "b"), definition_id="known-replicates")
    np.testing.assert_allclose(batch.mean, [4., 2.])
    np.testing.assert_allclose(batch.covariance, [[5/3, -1/3], [-1/3, 2/3]], atol=1e-14)
    np.testing.assert_allclose(batch.replicate_covariance, 4*batch.covariance)
    targets = MomentTargets([4., 2.], np.eye(2), ("a", "b"))
    total = moment_error_covariance(targets, batch, assume_independent=True)
    np.testing.assert_allclose(total, np.eye(2)+batch.covariance)


def test_plan_replay_extension_order_roles_and_global_rng():
    plan = SimulationPlan(seed=100, replications=5, periods=10)
    before = np.random.get_state()
    first = np.array([plan.rng(i).standard_normal(6) for i in range(5)])
    reordered = {i: plan.rng(i).standard_normal(6) for i in [4, 1, 0, 3, 2]}
    extended = replace(plan, replications=8)
    for i in range(5):
        np.testing.assert_array_equal(first[i], reordered[i])
        np.testing.assert_array_equal(first[i], extended.rng(i).standard_normal(6))
    assert not np.array_equal(first[0], replace(plan, role="validation").rng(0).standard_normal(6))
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    assert plan.plan_id == replace(plan).plan_id and plan.plan_id != extended.plan_id


def test_linear_gls_estimate_and_full_sampling_simulation_sandwich():
    design, targets, plan, simulator = _linear_problem()
    result = fit_simulated_moments(simulator, targets, [0., 0.], plan=plan,
                                   definition_id="linear-gaussian", jac=lambda t: design,
                                   assume_independent=True, assume_smooth=True)
    noise = simulate_moments(simulator, [0., 0.], plan, labels=targets.labels,
                             units=targets.units, definition_id="linear-gaussian")
    weight = np.linalg.inv(targets.covariance)
    influence = np.linalg.solve(design.T @ weight @ design, design.T @ weight)
    expected = influence @ (targets.values - noise.mean)
    assert result.success and result.inference_valid, result.inference_reasons
    np.testing.assert_allclose(result.theta, expected, atol=2e-9)
    np.testing.assert_allclose(result.influence, influence, atol=2e-9)
    np.testing.assert_allclose(result.sampling_covariance, influence @ targets.covariance @ influence.T)
    np.testing.assert_allclose(result.simulation_covariance, influence @ noise.covariance @ influence.T)
    np.testing.assert_allclose(result.covariance, influence @ (targets.covariance+noise.covariance) @ influence.T)
    np.testing.assert_allclose(result.covariance, result.sampling_covariance+result.simulation_covariance)
    residual = result.simulation.mean-targets.values
    np.testing.assert_allclose(result.objective, residual @ weight @ residual)
    assert result.n_replications_evaluated == result.n_evals * plan.replications
    assert "negligible model/simulation error" not in result.diagnostics["inference_assumptions"]


def test_cross_covariance_requires_declaration_and_joint_psd():
    design, targets, plan, simulator = _linear_problem()
    batch = simulate_moments(simulator, [0., 0.], plan, labels=targets.labels,
                             units=targets.units, definition_id="cross-covariance")
    with pytest.raises(ValueError, match="independ"):
        moment_error_covariance(targets, batch)
    with pytest.raises(ValueError, match="not both"):
        moment_error_covariance(targets, batch, assume_independent=True, cross_covariance=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="positive semidefinite"):
        moment_error_covariance(targets, batch, cross_covariance=np.eye(3))
    cross = np.eye(3)*1e-4
    result = fit_simulated_moments(simulator, targets, [0., 0.], plan=plan,
                                   definition_id="cross-covariance", cross_covariance=cross,
                                   jac=lambda t: design, assume_smooth=True)
    assert result.inference_valid
    expected = targets.covariance+result.simulation.covariance-cross-cross.T
    np.testing.assert_allclose(result.moment_covariance, expected)
    np.testing.assert_allclose(result.cross_covariance_adjustment,
                               -result.influence @ (cross+cross.T) @ result.influence.T)
    np.testing.assert_allclose(result.covariance, result.sampling_covariance+
                               result.simulation_covariance+result.cross_covariance_adjustment)


def test_custom_weight_and_moment_unit_changes_preserve_covariance():
    design, targets, plan, simulator = _linear_problem()
    weight = np.array([[3., .2, 0.], [.2, 1., .1], [0., .1, 2.]])
    first = fit_simulated_moments(simulator, targets, [0., 0.], plan=plan, weight=weight,
                                  definition_id="units", assume_independent=True, assume_smooth=True)
    units = np.array([1000., .01, 2.])
    second = fit_simulated_moments(lambda t, r, p: simulator(t, r, p)*units,
                                   targets.rescale(units, units=("rescaled",)*3), [0., 0.], plan=plan,
                                   weight=weight/units[:, None]/units[None, :],
                                   definition_id="rescaled-units", assume_independent=True, assume_smooth=True)
    assert first.inference_valid and second.inference_valid
    np.testing.assert_allclose(second.theta, first.theta, atol=2e-8)
    np.testing.assert_allclose(second.covariance, first.covariance, rtol=2e-6, atol=1e-9)


def test_pilot_weights_use_fit_replications_and_stay_fixed_with_parameter_dependent_variance():
    plan = SimulationPlan(seed=51, replications=40, periods=12)
    pilot = replace(plan, role="pilot", replications=7)
    validation = replace(plan, role="validation", periods=25, replications=16)
    targets = MomentTargets([.8, 1.4], [[.05, .01], [.01, .08]], ("a", "b"))
    def simulator(theta, rng, p):
        u = rng.standard_normal((p.periods, 2)).mean(axis=0)
        return np.array([theta[0]+(2+theta[0])*u[0], 2*theta[0]+(1+.3*theta[0])*u[1]])
    result = fit_simulated_moments(simulator, targets, [.1], plan=plan, pilot_plan=pilot,
                                   validation_plan=validation, definition_id="pilot-budget",
                                   assume_independent=True, assume_smooth=True)
    assert result.inference_valid
    frozen_cov = targets.covariance+result.pilot.replicate_covariance/plan.replications
    np.testing.assert_allclose(result.weight, np.linalg.inv(frozen_cov), rtol=1e-12)
    np.testing.assert_allclose(result.moment_covariance, targets.covariance+result.simulation.covariance)
    assert not np.allclose(result.moment_covariance, frozen_cov)
    u = np.array([plan.rng(i).standard_normal((plan.periods, 2)).mean(axis=0)
                  for i in range(plan.replications)]).mean(axis=0)
    slope, intercept = np.array([1+u[0], 2+.3*u[1]]), np.array([2*u[0], u[1]])
    expected = (slope @ result.weight @ (targets.values-intercept))/(slope @ result.weight @ slope)
    np.testing.assert_allclose(result.theta, [expected], atol=2e-8)
    assert result.validation.plan.role == "validation"
    np.testing.assert_array_equal(result.validation.theta, result.theta)
    assert result.diagnostics["validation_scope"].startswith("descriptive")


@pytest.mark.parametrize("draw", [
    np.array([np.nan, 2.]),
    MomentReplication([0., 2.], counts=[0, 5]),
    MomentReplication([1., 2.], status="solver_failed"),
])
def test_invalid_replicates_are_retained_not_dropped_or_imputed(draw):
    plan = SimulationPlan(seed=2, replications=3, periods=5)
    batches = iter([np.array([3., 4.]), draw, np.array([5., 6.])])
    batch = simulate_moments(lambda t, r, p: next(batches), [0.], plan,
                             labels=("a", "b"), definition_id="invalid-test")
    assert not batch.valid and batch.replicates.shape == (3, 2)
    assert not batch.valid_replicates[1]
    assert np.isnan(batch.mean).all() and np.isnan(batch.covariance).all()
    with pytest.raises(SimulationMomentError) as info:
        moment_error_covariance(MomentTargets([1., 2.], np.eye(2), ("a", "b")), batch,
                                assume_independent=True)
    assert info.value.batch is batch


def test_expected_invalidity_and_callback_bugs_do_not_become_optimizer_penalties():
    plan = SimulationPlan(seed=2, replications=4, periods=10)
    target = MomentTargets([1.], [[.1]], ("m",))
    with pytest.raises(SimulationMomentError) as info:
        fit_simulated_moments(lambda t, r, p: MomentReplication([0.], counts=[0]),
                               target, [0.], plan=plan, definition_id="missing-conditional",
                               assume_independent=True)
    assert info.value.batch.counts.min() == 0
    def broken(t, r, p):
        raise TypeError("inside simulation")
    with pytest.raises(TypeError, match="inside simulation"):
        fit_simulated_moments(broken, target, [0.], plan=plan,
                               definition_id="broken", assume_independent=True)


def test_replay_detects_ignored_rng_and_attaches_both_batches():
    count = [0]
    def not_reproducible(t, r, p):
        count[0] += 1
        return np.array([t[0]+count[0]])
    with pytest.raises(SimulationMomentError, match="reproduc") as info:
        fit_simulated_moments(not_reproducible, MomentTargets([1.], [[1.]], ("m",)), [0.],
                               plan=SimulationPlan(seed=1, replications=3, periods=4),
                               definition_id="bad-rng", assume_independent=True)
    assert info.value.reference is not None


def test_inference_requires_smoothness_and_verified_derivatives():
    plan = SimulationPlan(seed=1, replications=10, periods=10)
    target = MomentTargets([.3], [[.1]], ("mean",))
    sim = lambda t, rng, p: t+ rng.normal(size=p.periods).mean()
    result = fit_simulated_moments(sim, target, [0.], plan=plan, definition_id="smooth",
                                   assume_independent=True)
    assert result.success and not result.inference_valid
    assert "smoothness_not_asserted" in result.inference_reasons
    assert np.isnan(result.standard_errors).all()
    bad = fit_simulated_moments(sim, target, [0.], plan=plan, definition_id="bad-jac",
                                assume_independent=True, assume_smooth=True, jac=lambda t: [[2.]])
    assert not bad.inference_valid and "jacobian_mismatch" in bad.inference_reasons


def test_boundary_and_rank_failures_withhold_covariance():
    plan = SimulationPlan(seed=10, replications=8, periods=10)
    target = MomentTargets([-10.], [[1.]], ("m",))
    with pytest.warns(UserWarning):
        bound = fit_simulated_moments(lambda t, r, p: t+r.normal(size=p.periods).mean(),
                                      target, [1.], bounds=[(0., 2.)], plan=plan,
                                      definition_id="boundary", assume_independent=True, assume_smooth=True)
    assert "parameter_on_boundary" in bound.inference_reasons
    with pytest.warns(UserWarning):
        rank = fit_simulated_moments(lambda t, r, p: np.array([t.sum()])+r.normal(size=p.periods).mean(),
                                     target, [0., 0.], plan=plan, definition_id="rank",
                                     assume_independent=True, assume_smooth=True)
    assert "rank_deficient" in rank.inference_reasons
    assert np.isnan(rank.covariance).all()


@pytest.mark.parametrize("changes", [
    {"seed": -1}, {"replications": 1}, {"periods": 0}, {"burn_in": -1},
    {"stream": True}, {"role": "unknown"},
])
def test_invalid_plan(changes):
    kwargs = {"seed": 1, "replications": 4, "periods": 10} | changes
    with pytest.raises(ValueError):
        SimulationPlan(**kwargs)


def test_metadata_labels_counts_and_snapshot_ownership():
    plan = SimulationPlan(seed=2, replications=4, periods=10, burn_in=3)
    theta = np.array([2.])
    seen = []
    def simulator(t, rng, p):
        seen.append((p.periods, p.burn_in))
        return MomentReplication(pd.Series([t[0]+rng.normal()], index=["mean"]), counts=[p.periods])
    batch = simulate_moments(simulator, theta, plan, labels=("mean",), definition_id="counted")
    theta[:] = 99.
    assert batch.theta[0] == 2 and seen == [(10, 3)]*4
    with pytest.raises(ValueError):
        batch.replicates[0, 0] = 1.
    assert batch.to_frame().loc[0, "label"] == "mean"
    with pytest.raises(ValueError, match="labels"):
        simulate_moments(simulator, [2.], plan, labels=("wrong",), definition_id="bad-label")
    with pytest.raises(ValueError, match="counts"):
        MomentReplication([1.], counts=[1.5])


def test_role_budget_and_reporting_contracts():
    design, targets, plan, sim = _linear_problem()
    with pytest.raises(ValueError, match="pilot"):
        fit_simulated_moments(sim, targets, [0., 0.], plan=plan, pilot_plan=plan,
                               definition_id="bad-role", assume_independent=True)
    with pytest.raises(ValueError, match="periods"):
        fit_simulated_moments(sim, targets, [0., 0.], plan=plan,
                               pilot_plan=replace(plan, role="pilot", periods=11),
                               definition_id="bad-budget", assume_independent=True)
    result = fit_simulated_moments(sim, targets, [0., 0.], plan=plan, definition_id="report",
                                   assume_independent=True, assume_smooth=True, jac=lambda t: design)
    assert "simulation_mc_se" in result.moment_fit()
    assert "estimate" in result.to_frame()
    assert "tabular" in result.to_latex() and "table" in result.to_typst()
    assert "theta_0" in result.to_markdown()


def test_kink_with_nonzero_central_derivative_withholds_local_inference():
    result = fit_simulated_moments(
        lambda t, r, p: t+.2*abs(t), MomentTargets([0.], [[1.]], ("kink",)), [0.],
        plan=SimulationPlan(seed=9, replications=4, periods=10), definition_id="kink",
        assume_independent=True, assume_smooth=True)
    assert result.success and not result.inference_valid
    assert "inconsistent_stencil" in result.inference_reasons
    assert np.isnan(result.covariance).all()


def test_unequal_cross_covariance_orientation_and_snapshot_protection():
    _, targets, plan, sim = _linear_problem()
    batch = simulate_moments(sim, [0., 0.], plan, labels=targets.labels,
                             units=targets.units, definition_id="asymmetric-cross")
    cross = np.array([[1., 2., 0.], [0., 1., 0.], [0., 0., 1.]])*1e-5
    total = moment_error_covariance(targets, batch, cross_covariance=cross)
    np.testing.assert_allclose(total, targets.covariance+batch.covariance-cross-cross.T)
    with pytest.raises(ValueError):
        batch.replicates.setflags(write=True)


def test_validation_does_not_change_fit_and_uses_separate_random_streams():
    design, targets, plan, sim = _linear_problem()
    kwargs = dict(plan=plan, definition_id="validation", assume_independent=True,
                  assume_smooth=True, jac=lambda t: design)
    first = fit_simulated_moments(sim, targets, [0., 0.], **kwargs)
    second = fit_simulated_moments(sim, targets, [0., 0.],
                                   validation_plan=replace(plan, role="validation"), **kwargs)
    np.testing.assert_array_equal(first.theta, second.theta)
    np.testing.assert_array_equal(first.covariance, second.covariance)
    assert not np.array_equal(second.validation.replicates, second.simulation.replicates)
