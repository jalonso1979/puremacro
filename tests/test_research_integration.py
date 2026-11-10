"""Use the public modules together with independent algebraic references."""
import numpy as np

from puremacro.core import EquilibriumSystem, continue_equilibrium, linearize_equilibrium
from puremacro.expectations import GaussianSignal, build_gaussian_information
from puremacro.models import solve_information_nk
from puremacro.structural import (
    MomentTargets, SimulationPlan, fit_simulated_moments, simulate_moments,
    simulated_moment_test, evaluate_moment_grid, fit_structural,
)


def test_equilibrium_derivatives_feed_smm_and_point_null_inference():
    # Unique real root of k + k**3 = theta; the reference root is closed form.
    system = EquilibriumSystem(lambda k, theta: k + k**3 - theta, ("capital",), ("technology",))

    def root(theta):
        return 2/np.sqrt(3)*np.sinh(np.arcsinh(1.5*np.sqrt(3)*theta)/3)

    branch = continue_equilibrium(system, [1.], [2.], max_steps=4)
    np.testing.assert_allclose(branch.states, root(branch.parameters), rtol=0, atol=1e-9)
    for k, theta in zip(branch.states, branch.parameters):
        sensitivity = linearize_equilibrium(system, k, theta).sensitivity()
        assert sensitivity.valid
        np.testing.assert_allclose(sensitivity.jacobian, [[1/(1+3*k[0]**2)]], rtol=1e-7)

    plan = SimulationPlan(seed=93, replications=24, periods=8)
    targets = MomentTargets([1.], [[.0025]], ("capital",), units="level")

    def simulator(theta, rng, budget):
        return root(theta) + .1*rng.normal(size=budget.periods).mean()

    def jac(theta):
        return linearize_equilibrium(system, root(theta), theta).sensitivity().jacobian

    fit = fit_simulated_moments(simulator, targets, [1.8], plan=plan,
        definition_id="equilibrium-capital-plus-mean-noise-v1", jac=jac,
        assume_independent=True, assume_smooth=True, parameter_names=("technology",))
    noise = np.array([.1*plan.rng(i).normal(size=plan.periods).mean() for i in range(plan.replications)])
    implied_k = targets.values[0]-noise.mean()
    expected_theta = implied_k+implied_k**3
    expected_covariance = (1+3*implied_k**2)**2*(.0025+noise.var(ddof=1)/len(noise))
    assert fit.success and fit.inference_valid
    np.testing.assert_allclose(fit.theta, [expected_theta], atol=1e-8)
    np.testing.assert_allclose(fit.covariance, [[expected_covariance]], rtol=1e-6)
    point = simulated_moment_test(targets, fit.simulation, assume_independent=True)
    assert point.valid and point.accepted

    def at(theta):
        batch = simulate_moments(simulator, [theta], plan, labels=targets.labels, units=targets.units,
            definition_id="equilibrium-capital-plus-mean-noise-v1")
        return simulated_moment_test(targets, batch, assume_independent=True)

    grid = evaluate_moment_grid(at, fit.theta[0]+np.array([-.7, 0., .7]), parameter_name="technology")
    np.testing.assert_array_equal(grid.accepted, [False, True, False])
    assert not grid.continuous_complete
    assert set(grid.unresolved_cells().reason) == {"unsampled_interior"}


def test_information_forecasts_use_explicit_moment_measurement():
    info = build_gaussian_information(np.eye(2), [[GaussianSignal(0, [1., 1.], "noisy_news")]], horizon=3)
    rates = np.array([[0., 0.], [0., 0.], [.01, 0.]])

    def model(theta):
        path = solve_information_nk(info, rates*theta[0]).evaluate([1., 0.])
        return np.array([path.output[0], path.inflation[0]])

    loading = model([1.])
    covariance = np.diag([1e-5, 2e-6])
    observed = model([1.4])+np.array([.001, -.0002])
    targets = MomentTargets(observed, covariance, ("output_0", "inflation_0"), units="deviation")
    result = fit_structural(model, targets, [.8], parameter_names=("news_amplitude",))
    precision = np.linalg.inv(covariance)
    reference = float(loading@precision@observed/(loading@precision@loading))
    assert result.success
    np.testing.assert_allclose(result.theta, [reference], rtol=1e-7)
    # Reuse never changes the information set or the original policy array.
    np.testing.assert_array_equal(rates, [[0., 0.], [0., 0.], [.01, 0.]])


def test_invalid_simulation_remains_unresolved_in_a_composed_grid():
    targets = MomentTargets([1.], [[.01]], ("mean",))
    plan = SimulationPlan(seed=3, replications=4, periods=1)

    def test_at(theta):
        batch = simulate_moments(lambda x, rng, p: [np.nan if x[0] == 1 else x[0]], [theta], plan,
            labels=targets.labels, definition_id="deliberately-undefined-at-one")
        return simulated_moment_test(targets, batch, assume_independent=True)

    grid = evaluate_moment_grid(test_at, [0., 1., 2.])
    assert grid.tests[1].accepted is None
    assert grid.tests[1].simulation is not None
    assert not grid.tests[1].simulation.valid
    assert grid.to_frame().accepted.isna().tolist() == [False, True, False]
    assert set(grid.unresolved_cells().reason) == {"invalid_endpoint"}
