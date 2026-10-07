"""Sampling API contracts; independent exact oracles live in the audit tests."""
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge.smets_wouters import SW07_SHOCK_STDS
from puremacro.dsge.sw07_observation import OBSERVED_VARS
from puremacro.structural.empirical_sw07 import (
    ALL_MOMENTS, FIT_MOMENTS, _PUBLISHED_MODE, sw07_covariance_moments,
)
from puremacro.structural.sw07_sampling import (
    _psd_root, simulate_sw07_sample, sw07_finite_sample_moments,
)


def test_known_mean_control_equals_population_covariances():
    for params in ({}, _PUBLISHED_MODE):
        targets = sw07_finite_sample_moments(params, demean=False)
        population = sw07_covariance_moments(params)
        np.testing.assert_allclose(targets.values, population, atol=2e-11, rtol=2e-12)
        assert targets.labels == tuple(population.index)
        assert targets.metadata["effective_observations"] == 152
        assert "not raw products" in targets.metadata["mean_treatment"]
        assert "not generally Gaussian" in targets.metadata["distribution_warning"]
        assert "not HAC" in targets.metadata["covariance_method"]
        assert np.linalg.eigvalsh(targets.covariance).min() > 0


def test_full_window_preserved_for_fit_subset():
    all_targets = sw07_finite_sample_moments(_PUBLISHED_MODE)
    subset = sw07_finite_sample_moments(_PUBLISHED_MODE, FIT_MOMENTS, common_max_lag=4)
    np.testing.assert_allclose(subset.values, all_targets.values[:9], rtol=2e-12, atol=1e-13)
    np.testing.assert_allclose(subset.covariance, all_targets.covariance[:9, :9], rtol=2e-12, atol=1e-13)
    own_window = sw07_finite_sample_moments(_PUBLISHED_MODE, FIT_MOMENTS)
    assert own_window.metadata["effective_observations"] == 155
    assert not np.allclose(own_window.covariance, subset.covariance, rtol=1e-4, atol=1e-12)


def test_reproducible_stationary_samples_include_population_means():
    first = simulate_sw07_sample({}, seed=52)
    repeated = simulate_sw07_sample({}, seed=52)
    other = simulate_sw07_sample({}, seed=53)
    pd.testing.assert_frame_equal(first, repeated)
    assert first.shape == (156, 7)
    assert tuple(first.columns) == OBSERVED_VARS
    assert not np.allclose(first, other)
    assert first.attrs["is_synthetic"]
    assert first.attrs["observation_intercepts_included"]
    assert first.attrs["population_means"]["infl"] == .7
    assert not first.attrs["measurement_error_included"]


def test_zero_innovation_variances_have_no_added_ridge_or_initial_noise():
    params = dict.fromkeys(SW07_SHOCK_STDS, 0.)
    sample = simulate_sw07_sample(params, nobs=20)
    means = np.array([sample.attrs["population_means"][name] for name in OBSERVED_VARS])
    np.testing.assert_array_equal(sample.to_numpy(), np.tile(means, (20, 1)))
    targets = sw07_finite_sample_moments(params, nobs=20)
    np.testing.assert_array_equal(targets.values, np.zeros(len(ALL_MOMENTS)))
    np.testing.assert_array_equal(targets.covariance, np.zeros((len(ALL_MOMENTS),)*2))


@pytest.mark.parametrize("scale", [1., 1e-20, 1e20])
def test_psd_guard_is_relative_to_covariance_units(scale):
    with pytest.raises(np.linalg.LinAlgError, match="positive semidefinite"):
        _psd_root(scale*np.diag([1., -.001]), "test covariance")
    factor, diagnostics = _psd_root(scale*np.diag([1., -1e-16]), "test covariance")
    np.testing.assert_allclose(factor @ factor.T / scale, np.diag([1., 0.]), atol=1e-15)
    assert diagnostics["negative_eigenvalues_clipped_for_roundoff"] == 1


@pytest.mark.parametrize("kwargs", [
    {"nobs": True}, {"nobs": 1}, {"nobs": 4}, {"demean": "true"},
    {"common_max_lag": True}, {"common_max_lag": 3}, {"common_max_lag": 156},
])
def test_invalid_sampling_moment_controls(kwargs):
    with pytest.raises(ValueError):
        sw07_finite_sample_moments({}, **kwargs)


@pytest.mark.parametrize("kwargs", [{"seed": -1}, {"seed": True}, {"seed": .1},
                                   {"nobs": 0}, {"nobs": 2.5}])
def test_invalid_simulation_controls(kwargs):
    with pytest.raises(ValueError):
        simulate_sw07_sample({}, **kwargs)


@pytest.mark.parametrize("params", [{"em": -.1}, {"crr": np.nan}, {"em": .2j}, {"unknown": 1}])
def test_both_paths_reject_invalid_calibrations(params):
    for function in (simulate_sw07_sample, sw07_finite_sample_moments):
        with pytest.raises(ValueError):
            function(params)
