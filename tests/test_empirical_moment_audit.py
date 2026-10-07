"""Independent numerical oracles for the empirical SW07 moment application.

The covariance oracle uses a dense time kernel, and the population oracle
uses a moving-average sum rather than solving the Lyapunov equation.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
from puremacro.dsge.sw07_observation import OBSERVED_VARS, make_state_space
from puremacro.structural.empirical_sw07 import (
    ALL_MOMENTS,
    covariance_moment_targets,
    sw07_covariance_moments,
)


AUDIT_MOMENTS = (
    ("gdp_growth", "gdp_growth", 0),
    ("gdp_growth", "infl", 0),
    ("infl", "infl", 0),
    ("ffr", "ffr", 0),
    ("gdp_growth", "infl", 1),
    ("infl", "gdp_growth", 1),
    ("ffr", "gdp_growth", 4),
)
UNITS = {"gdp_growth": "quarterly percent", "infl": "quarterly percent", "ffr": "quarterly percent"}


def _sample():
    rng = np.random.default_rng(7180)
    transition = np.array([[.58, .17, 0], [-.14, .72, .08], [.2, 0, .47]])
    values = np.empty((160, 3))
    values[0] = rng.normal(size=3)
    for t in range(1, len(values)):
        values[t] = transition @ values[t - 1] + rng.normal(size=3)
    values += [1.2, -3.1, 7.4]
    return pd.DataFrame(values, columns=list(UNITS),
                        index=pd.period_range("1980Q1", periods=len(values), freq="Q"))


def _targets(data, moments=AUDIT_MOMENTS, bandwidth=6):
    return covariance_moment_targets(data, moments, units=UNITS, bandwidth=bandwidth)


@pytest.mark.parametrize("bandwidth", [0, 1, 6, 19])
def test_joint_sampling_covariance_matches_dense_bartlett_quadratic_form(bandwidth):
    data = _sample()
    first = max(h for _, _, h in AUDIT_MOMENTS)
    # Assemble each scalar product explicitly on the common window. The
    # population estimand uses full-series means, not pair-specific means.
    means = {name: np.mean(data[name].to_numpy()) for name in data}
    products = np.array([
        [(data[left].iloc[t] - means[left]) * (data[right].iloc[t - lag] - means[right])
         for left, right, lag in AUDIT_MOMENTS]
        for t in range(first, len(data))
    ])
    estimates = products.mean(axis=0)
    scores = products - estimates
    n_eff = len(products)
    distance = np.abs(np.arange(n_eff)[:, None] - np.arange(n_eff)[None, :])
    kernel = np.maximum(1 - distance / (bandwidth + 1), 0)
    expected_covariance = scores.T @ kernel @ scores / n_eff**2
    actual = _targets(data, bandwidth=bandwidth)
    assert_allclose(actual.values, estimates, rtol=2e-13, atol=1e-14)
    assert_allclose(actual.covariance, expected_covariance, rtol=2e-12, atol=1e-14)
    # A diagonal-only implementation would lose substantive joint information.
    offdiagonal = actual.covariance - np.diag(np.diag(actual.covariance))
    assert np.max(np.abs(offdiagonal)) > 1e-3
    assert np.linalg.eigvalsh(actual.covariance)[0] > 0


def test_estimated_centering_is_translation_invariant_and_scales_covariance_twice():
    data = _sample()
    base = _targets(data)
    translated = _targets(data + pd.Series({"gdp_growth": 24., "infl": -43., "ffr": 80.}))
    assert_allclose(translated.values, base.values, rtol=2e-12, atol=2e-13)
    assert_allclose(translated.covariance, base.covariance, rtol=2e-12, atol=2e-13)
    # Cov(left, right) scales by a_left*a_right; its sampling covariance
    # scales on both moment axes, including a sign reversal of cross moments.
    scales = {"gdp_growth": 100., "infl": -2., "ffr": .25}
    rescaled = _targets(data * pd.Series(scales))
    moment_scales = np.array([scales[a] * scales[b] for a, b, _ in AUDIT_MOMENTS])
    assert_allclose(rescaled.values, base.values * moment_scales, rtol=2e-12, atol=1e-11)
    assert_allclose(rescaled.covariance,
                    moment_scales[:, None] * base.covariance * moment_scales[None, :],
                    rtol=2e-12, atol=1e-10)


def test_moment_permutation_preserves_cross_covariance_and_common_window():
    data = _sample()
    base = _targets(data)
    order = np.array([6, 2, 4, 0, 5, 3, 1])
    permuted = _targets(data, tuple(AUDIT_MOMENTS[i] for i in order))
    assert_allclose(permuted.values, base.values[order], atol=1e-14)
    assert_allclose(permuted.covariance, base.covariance[np.ix_(order, order)], atol=1e-14)
    assert permuted.labels == tuple(base.labels[i] for i in order)


def test_complex_empirical_values_are_not_silently_projected_to_real_part():
    data = _sample()
    data["infl"] = data["infl"].astype(complex) + .01j
    with pytest.raises(ValueError, match="real"):
        _targets(data)


@pytest.fixture(scope="module")
def population_oracle():
    params = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS, "crr": .72, "em": .15}
    model = make_state_space(params)
    # White-noise impulse responses incorporate the shock standard deviations.
    # This convergent MA representation is independent of the production
    # discrete Lyapunov solver and has no Kalman measurement-error ridge.
    loading = model.R @ np.linalg.cholesky(model.Q)
    responses = []
    state_covariance = np.zeros_like(model.T)
    # The fixed calibration includes technology persistence .9977, so a
    # few thousand terms would still leave a material tail in state levels.
    for _ in range(16005):
        responses.append(model.Z @ loading)
        state_covariance += loading @ loading.T
        loading = model.T @ loading
    assert np.linalg.norm(loading) < 1e-11
    responses = np.array(responses)
    gamma = {h: np.einsum("tij,tkj->ik", responses[h:h + 16000], responses[:16000])
             for h in (0, 1, 2, 4)}
    return params, model, state_covariance, gamma


def test_sw07_population_moments_match_independent_ma_sum(population_oracle):
    params, model, _, gamma = population_oracle
    moments = tuple(ALL_MOMENTS) + (("gdp_growth", "ffr", 1), ("ffr", "gdp_growth", 1))
    actual = sw07_covariance_moments(params, moments)
    expected = [gamma[h][OBSERVED_VARS.index(left), OBSERVED_VARS.index(right)]
                for left, right, h in moments]
    assert_allclose(np.asarray(actual), expected, rtol=5e-10, atol=2e-11)
    assert_allclose(actual.attrs["transition_spectral_radius"],
                    np.max(np.abs(np.linalg.eigvals(model.T))), atol=1e-13)
    assert actual.attrs["lyapunov_relative_residual"] < 1e-9
    assert actual.attrs["measurement_error_included"] is False
    # Γ(1) is generally nonsymmetric: reversing the ordered pair changes
    # the estimand, unlike zero-lag covariance.
    assert abs(expected[-1] - expected[-2]) > 1e-3
    assert model.H[0, 0] > 0
    # Tighter-than-ridge tolerance above detects accidental inclusion of H.
    assert 2e-11 < model.H[0, 0]


def test_population_covariances_satisfy_joint_psd_and_match_stationary_oracle(population_oracle):
    params, model, state_covariance, gamma = population_oracle
    innovation_covariance = model.R @ model.Q @ model.R.T
    residual = state_covariance - model.T @ state_covariance @ model.T.T - innovation_covariance
    assert np.linalg.norm(residual) / np.linalg.norm(state_covariance) < 1e-12
    assert np.max(np.abs(np.linalg.eigvals(model.T))) < 1
    zero_specs = tuple((a, b, 0) for a in OBSERVED_VARS for b in OBSERVED_VARS)
    zero = sw07_covariance_moments(params, zero_specs).to_numpy().reshape(7, 7)
    assert_allclose(zero, gamma[0], rtol=5e-10, atol=2e-11)
    assert_allclose(zero, zero.T, atol=1e-10)
    assert np.linalg.eigvalsh(zero)[0] > 0
    # Autocovariances need not themselves be PSD. The joint covariance of
    # [y_t, y_(t-h)] must be PSD, which also audits lag orientation.
    for h in (1, 2, 4):
        specs = tuple((a, b, h) for a in OBSERVED_VARS for b in OBSERVED_VARS)
        lagged = sw07_covariance_moments(params, specs).to_numpy().reshape(7, 7)
        assert_allclose(lagged, gamma[h], rtol=5e-10, atol=2e-11)
        joint = np.block([[zero, lagged], [lagged.T, zero]])
        assert np.linalg.eigvalsh(joint)[0] > -1e-10


@pytest.mark.filterwarnings("ignore:klein_solve.*:RuntimeWarning")
def test_unit_root_calibration_has_no_reported_unconditional_covariance():
    with pytest.raises(np.linalg.LinAlgError, match="stationary"):
        sw07_covariance_moments({"crhoa": 1.0})


def test_inaccurate_lyapunov_solution_is_refused(monkeypatch):
    from puremacro.structural import empirical_sw07

    # Simulate a solver returning a finite, symmetric, PSD but incorrect
    # solution: PSD alone cannot validate unconditional model moments.
    monkeypatch.setattr(empirical_sw07, "solve_discrete_lyapunov",
                        lambda transition, innovation: np.zeros_like(transition))
    with pytest.raises(np.linalg.LinAlgError, match="Lyapunov"):
        sw07_covariance_moments({})
