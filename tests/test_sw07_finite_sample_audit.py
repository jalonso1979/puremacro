"""Independent Gaussian quadratic-form audit of finite-sample SW07 moments.

Small AR(1)/VAR(1) systems expose exact centering and lag alignment. The
simulation test integrates a linear simulator over deterministic Gaussian
second-moment quadrature points; it is not a Monte Carlo experiment.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from puremacro.dsge.sw07_observation import OBSERVED_VARS
from puremacro.state_space import StateSpaceModel
from puremacro.structural import sw07_sampling as sampling


LEFT, RIGHT = "gdp_growth", "infl"
SELECTED = [OBSERVED_VARS.index(LEFT), OBSERVED_VARS.index(RIGHT)]


def _toy_system(transition, innovation, intercept=None):
    transition = np.asarray(transition, dtype=float)
    innovation = np.asarray(innovation, dtype=float)
    dimension = len(transition)
    # A Kronecker linear solve, independent of the production Lyapunov
    # routine, provides the stationary state covariance for the oracle.
    state_covariance = np.linalg.solve(
        np.eye(dimension**2) - np.kron(transition, transition), innovation.ravel()
    ).reshape(dimension, dimension)
    observation = np.zeros((7, dimension))
    observation[SELECTED[0], 0] = 1
    observation[SELECTED[1], -1] = 1
    model = StateSpaceModel(
        T=transition, Z=observation, R=np.eye(dimension), Q=innovation,
        H=np.eye(7) * 123.,  # deliberately large: sampling must omit H
        d=np.arange(7, dtype=float) if intercept is None else np.asarray(intercept),
    )
    diagnostics = {"transition_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(transition)))),
                   "lyapunov_relative_residual": 0., "measurement_error_included": False}
    return model, state_covariance, diagnostics


def _patch_system(monkeypatch, system):
    monkeypatch.setattr(sampling, "_stationary_system", lambda params: system)


def _dense_stationary_covariance(model, state_covariance, nobs):
    observation = model.Z[SELECTED]
    gammas = [observation @ np.linalg.matrix_power(model.T, h) @ state_covariance @ observation.T
              for h in range(nobs)]
    # Time-major order: [left_0,right_0,left_1,right_1,...].
    return np.block([[gammas[t - s] if t >= s else gammas[s - t].T
                      for s in range(nobs)] for t in range(nobs)])


def _quadratic_form_oracle(model, state_covariance, moments, nobs, common_max_lag, demean):
    covariance = _dense_stationary_covariance(model, state_covariance, nobs)
    time_centering = np.eye(nobs) - np.ones((nobs, nobs)) / nobs if demean else np.eye(nobs)
    centering = np.kron(time_centering, np.eye(2))
    operators = []
    positions = {LEFT: 0, RIGHT: 1}
    for left, right, lag in moments:
        operator = np.zeros((2 * nobs, 2 * nobs))
        for t in range(common_max_lag, nobs):
            i, j = 2 * t + positions[left], 2 * (t - lag) + positions[right]
            operator[i, j] += .5 / (nobs - common_max_lag)
            operator[j, i] += .5 / (nobs - common_max_lag)
        operators.append(centering.T @ operator @ centering)
    loadings = [operator @ covariance for operator in operators]
    means = np.array([np.trace(loading) for loading in loadings])
    # For zero-mean Gaussian z, E(z'Az)=tr(AK) and the covariance of two
    # symmetric quadratic forms is 2*tr(A K B K). The production code uses
    # elementwise paired products, so this is an independent dense oracle.
    variances = np.array([[2 * np.trace(a @ b) for b in loadings] for a in loadings])
    return means, variances


@pytest.mark.parametrize("rho", [0., -.55, .73, .97])
@pytest.mark.parametrize("demean", [False, True])
def test_ar1_exact_joint_moments_match_dense_gaussian_quadratic_forms(monkeypatch, rho, demean):
    system = _toy_system([[rho]], [[.64]])
    _patch_system(monkeypatch, system)
    moments = ((LEFT, LEFT, 0), (LEFT, LEFT, 1), (LEFT, LEFT, 4))
    actual = sampling.sw07_finite_sample_moments({}, moments, nobs=13,
                                                common_max_lag=4, demean=demean)
    means, covariance = _quadratic_form_oracle(*system[:2], moments, 13, 4, demean)
    assert_allclose(actual.values, means, rtol=2e-12, atol=1e-13)
    assert_allclose(actual.covariance, covariance, rtol=3e-12, atol=1e-12)
    assert np.linalg.eigvalsh(actual.covariance).min() > 0
    if not demean:
        assert_allclose(actual.values, .64 / (1 - rho**2) * rho ** np.array([0, 1, 4]))


def test_iid_closed_forms_expose_sample_mean_bias_and_covariance_scaling(monkeypatch):
    variance, nobs = .64, 156
    _patch_system(monkeypatch, _toy_system([[0.]], [[variance]]))
    moments = ((LEFT, LEFT, 0),)
    centered = sampling.sw07_finite_sample_moments({}, moments, nobs=nobs, demean=True)
    known_mean = sampling.sw07_finite_sample_moments({}, moments, nobs=nobs, demean=False)
    assert_allclose(centered.values, [variance * (nobs - 1) / nobs], atol=1e-14)
    assert_allclose(centered.covariance, [[2 * variance**2 * (nobs - 1) / nobs**2]], atol=1e-14)
    assert_allclose(known_mean.values, [variance], atol=1e-14)
    assert_allclose(known_mean.covariance, [[2 * variance**2 / nobs]], atol=1e-14)
    # Estimated demeaning creates negative expected sample autocovariance
    # even for IID observations, including a truncated common time window.
    lagged = sampling.sw07_finite_sample_moments(
        {}, ((LEFT, LEFT, 0), (LEFT, LEFT, 1), (LEFT, LEFT, 4)),
        nobs=nobs, common_max_lag=4, demean=True)
    assert_allclose(lagged.values, [variance * (1 - 1 / nobs), -variance / nobs, -variance / nobs],
                    atol=1e-14)


@pytest.mark.parametrize("demean", [False, True])
def test_nonsymmetric_crosslags_and_full_joint_covariance_match_dense_oracle(monkeypatch, demean):
    system = _toy_system([[.68, .24], [-.13, .36]], [[.7, .16], [.16, .4]])
    _patch_system(monkeypatch, system)
    moments = ((LEFT, LEFT, 0), (LEFT, RIGHT, 0), (LEFT, RIGHT, 1),
               (RIGHT, LEFT, 1), (RIGHT, LEFT, 4))
    actual = sampling.sw07_finite_sample_moments({}, moments, nobs=12,
                                                common_max_lag=4, demean=demean)
    expected, covariance = _quadratic_form_oracle(*system[:2], moments, 12, 4, demean)
    assert abs(expected[2] - expected[3]) > .02
    assert_allclose(actual.values, expected, rtol=2e-12, atol=1e-13)
    assert_allclose(actual.covariance, covariance, rtol=2e-12, atol=1e-13)
    assert np.max(np.abs(covariance - np.diag(np.diag(covariance)))) > .001


def test_explicit_common_window_preserves_selected_joint_moments(monkeypatch):
    _patch_system(monkeypatch, _toy_system([[.87]], [[.5]]))
    moments = ((LEFT, LEFT, 0), (LEFT, LEFT, 1), (LEFT, LEFT, 4))
    all_moments = sampling.sw07_finite_sample_moments({}, moments, nobs=15)
    same_window = sampling.sw07_finite_sample_moments({}, moments[:2], nobs=15, common_max_lag=4)
    assert_allclose(same_window.values, all_moments.values[:2], atol=1e-14)
    assert_allclose(same_window.covariance, all_moments.covariance[:2, :2], atol=1e-14)
    changed_window = sampling.sw07_finite_sample_moments({}, moments[:2], nobs=15)
    assert np.max(np.abs(changed_window.values - same_window.values)) > .001


@pytest.mark.parametrize("singular", [False, True])
def test_simulator_has_exact_stationary_initial_covariance_under_deterministic_quadrature(monkeypatch, singular):
    system = (_toy_system([[.7, 0.], [0., .3]], [[.5, 0.], [0., 0.]]) if singular else
              _toy_system([[.68, .24], [-.13, .36]], [[.7, .16], [.16, .4]]))
    _patch_system(monkeypatch, system)
    model, stationary, _ = system
    nobs = 7
    dimension = len(model.T) + (nobs - 1) * len(model.Q)

    class FixedNormal:
        def __init__(self, values):
            self.values, self.used = values, 0

        def standard_normal(self, size):
            count = int(np.prod(size))
            values = self.values[self.used:self.used + count]
            self.used += count
            return values.reshape(size)

    samples = []
    for coordinate in range(dimension):
        for sign in (-1., 1.):
            normal = np.zeros(dimension)
            normal[coordinate] = sign * np.sqrt(dimension)
            rng = FixedNormal(normal)
            monkeypatch.setattr(sampling.np.random, "default_rng", lambda seed: rng)
            simulated = sampling.simulate_sw07_sample({}, nobs=nobs, seed=0)
            assert tuple(simulated.columns) == OBSERVED_VARS
            assert rng.used == dimension
            samples.append(simulated.loc[:, [LEFT, RIGHT]].to_numpy().ravel())
    samples = np.asarray(samples)
    population_mean = np.tile(model.d[SELECTED], nobs)
    assert_allclose(samples.mean(axis=0), population_mean, atol=1e-14)
    deviations = samples - population_mean
    exact_covariance = deviations.T @ deviations / len(samples)
    expected = _dense_stationary_covariance(model, stationary, nobs)
    assert_allclose(exact_covariance, expected, rtol=2e-12, atol=1e-13)
    # A zero-initialized state would have zero first-period variance and
    # fail both this check and the complete time covariance above.
    assert exact_covariance[0, 0] > .5


@pytest.mark.filterwarnings("ignore:klein_solve.*:RuntimeWarning")
def test_nonstationary_sw07_parameters_cannot_produce_exact_stationary_moments():
    with pytest.raises(np.linalg.LinAlgError, match="stationar|Blanchard"):
        sampling.sw07_finite_sample_moments({"crhoa": 1.0}, nobs=20)
    with pytest.raises(np.linalg.LinAlgError, match="stationar|Blanchard"):
        sampling.simulate_sw07_sample({"crhoa": 1.0}, nobs=20, seed=0)
