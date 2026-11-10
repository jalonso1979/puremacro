"""Dense simultaneous equations and statistical Gaussian forecast references."""
import numpy as np
import pytest

from puremacro.expectations import GaussianSignal, build_gaussian_information
from puremacro.models.information_nk import InformationNKParameters, solve_information_nk


def small_information(T=5):
    return build_gaussian_information(np.array([[1., .2, 0.], [.2, 1., .1], [0., .1, 1.]]), (
        (GaussianSignal(0, [1., 1., 0.], "a"), GaussianSignal(3, [1, 0, 0], "reveal")),
        (GaussianSignal(1, [1., 0., 2.], "b"), GaussianSignal(3, [1, 0, 0], "reveal")),
    ), horizon=T, weights=[.3, .7])


def test_simultaneous_dense_reference_matches_backward_accumulators():
    info, model = small_information(), InformationNKParameters()
    T, n = info.horizon, len(info.covariance)
    R = np.zeros((T, n))
    R[3, 0] = -.0025
    firm_weights = np.array([.8, .2])
    solution = solve_information_nk(info, R, firm_weights=firm_weights)

    def equations(flat, rates):
        Y, Pi = flat.reshape(2, T, n)
        demand, pricing = Y.copy(), Pi-model.kappa*Y
        for t in range(T):
            Pc = sum(info.weights[g]*info.projectors[t, g] for g in range(2))
            Pf = sum(firm_weights[g]*info.projectors[t, g] for g in range(2))
            for j in range(t, T):
                demand[t] += model.intertemporal_elasticity*model.beta**(j-t)*rates[j]@Pc
                if j+1 < T:
                    demand[t] -= model.intertemporal_elasticity*model.beta**(j-t)*Pi[j+1]@Pc
                if j > t:
                    demand[t] -= (1-model.beta)*model.beta**(j-t-1)*Y[j]@Pc
                    pricing[t] -= (model.beta*model.price_stickiness)**(j-t)*(
                        model.kappa*Y[j]+(1-model.price_stickiness)/model.price_stickiness*Pi[j])@Pf
        return np.r_[demand.ravel(), pricing.ravel()]

    size = 2*T*n
    matrix = np.column_stack([equations(column, np.zeros_like(R)) for column in np.eye(size)])
    rhs = -equations(np.zeros(size), R)
    expected = np.linalg.solve(matrix, rhs).reshape(2, T, n)
    np.testing.assert_allclose(solution.output_loadings, expected[0], atol=1e-12)
    np.testing.assert_allclose(solution.inflation_loadings, expected[1], atol=1e-12)


def test_averaging_forecasts_in_textbook_euler_is_not_the_same_model():
    info = small_information()
    R = np.zeros((5, 3))
    R[3, 0] = -.0025
    result = solve_information_nk(info, R)
    errors = []
    for t in range(4):
        P = info.average_projectors[t]
        wrong_euler = (result.output_loadings[t+1]-(R[t]-result.inflation_loadings[t+1]))@P
        errors.append(np.max(abs(result.output_loadings[t]-wrong_euler)))
    assert max(errors) > 1e-5


def test_seeded_forecast_errors_match_conditional_variance_and_orthogonality():
    rng = np.random.default_rng(88154)
    draws = rng.normal(size=(50000, 3))
    info = build_gaussian_information(np.eye(3), (
        (GaussianSignal(0, [1., .5, 0.], "a"),),
        (GaussianSignal(0, [1., 0., 2.], "b"),),
    ), horizon=2)
    for g, noise in enumerate((.5, 2.)):
        signal = draws@info.signals[g][0].loading
        expected_mean = signal/(1+noise**2)
        calculated_mean = draws@info.projectors[0, g, 0]
        np.testing.assert_allclose(calculated_mean, expected_mean, atol=1e-12)
        error = draws[:, 0]-calculated_mean
        variance = noise**2/(1+noise**2)
        assert abs(error.mean()) < 4*np.sqrt(variance/len(draws))
        assert abs(np.mean(error**2)/variance-1) < .04
        product = error*signal
        assert abs(product.mean()) < 4*product.std(ddof=1)/np.sqrt(len(draws))
        assert info.posterior_covariances[0, g, 0, 0] == pytest.approx(variance)


def test_policy_linearity_and_owned_input_arrays():
    info = small_information()
    R = np.zeros((5, 3))
    R[3, 0] = -.0025
    first = solve_information_nk(info, R)
    other = solve_information_nk(info, -3*R)
    np.testing.assert_allclose(other.output_loadings, -3*first.output_loadings, atol=1e-12)
    R[:] = 100
    assert first.rate_loadings[0, 0] == 0
    assert not first.output_loadings.flags.writeable


def test_prior_unit_rescaling_preserves_economic_forecasts():
    info = small_information()
    units = np.diag([10., .1, 2.])
    inverse = np.linalg.inv(units)
    changed = build_gaussian_information(units@info.covariance@units, tuple(
        tuple(GaussianSignal(s.date, s.loading@inverse, s.label) for s in group) for group in info.signals),
        horizon=info.horizon, weights=info.weights)
    R = np.zeros((5, 3))
    R[3, 0] = -.0025
    base = solve_information_nk(info, R)
    other = solve_information_nk(changed, R@inverse)
    np.testing.assert_allclose(other.output_loadings@units, base.output_loadings, atol=1e-12)
    np.testing.assert_allclose(other.inflation_loadings@units, base.inflation_loadings, atol=1e-12)


def test_signal_precision_limit_recovers_observed_policy_direction():
    T = 4
    R = np.zeros((T, 3))
    R[3, 0] = -.0025
    errors = []
    full = build_gaussian_information(np.eye(3), ((GaussianSignal(0, [1, 0, 0], "truth"),),), horizon=T)
    reference = solve_information_nk(full, R)
    for noise in (1., .1, .01, .001):
        info = build_gaussian_information(np.eye(3), (
            (GaussianSignal(0, [1, noise, 0], "a"), GaussianSignal(3, [1, 0, 0], "reveal")),
            (GaussianSignal(0, [1, 0, noise], "b"), GaussianSignal(3, [1, 0, 0], "reveal")),
        ), horizon=T)
        result = solve_information_nk(info, R)
        errors.append(np.max(abs(result.output_loadings[:, 0]-reference.output_loadings[:, 0])))
    assert all(a > b for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 1e-8
