"""Fast SW07 expectations agree with the audited full finite-sample oracle."""
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from puremacro.dsge.smets_wouters import SW07_SHOCK_STDS
from puremacro.dsge.sw07_observation import OBSERVED_VARS
from puremacro.structural import sw07_expectations as fast
from puremacro.structural.empirical_sw07 import (
    ALL_MOMENTS, FIT_MOMENTS, _PUBLISHED_MODE, sw07_covariance_moments,
)
from puremacro.structural.sw07_sampling import sw07_finite_sample_moments


CALIBRATIONS = [{}, dict(_PUBLISHED_MODE)] + [
    {**mode, "crr": crr, "em": em}
    for mode in ({}, _PUBLISHED_MODE)
    for crr, em in ((.2, .01), (.2, 1.), (.98, .01), (.98, 1.))
] + [{**_PUBLISHED_MODE, "crr": .83825764, "em": .2540856}]


@pytest.mark.parametrize("params", CALIBRATIONS)
@pytest.mark.parametrize("demean", [False, True])
def test_both_calibrations_and_estimator_bounds_match_full_oracle(params, demean):
    actual = fast.sw07_finite_sample_expectations(params, nobs=156, demean=demean)
    expected = sw07_finite_sample_moments(params, nobs=156, demean=demean)
    assert tuple(actual.index) == expected.labels
    assert_allclose(actual.to_numpy(), expected.values, rtol=3e-12, atol=2e-13)
    assert actual.attrs["effective_observations"] == 152
    assert actual.attrs["measurement_error_included"] is False


@pytest.mark.parametrize("nobs,window", [(2, 1), (7, 3), (31, 8), (156, 12)])
def test_nonsymmetric_crosslags_additional_observables_and_windows(nobs, window):
    moments = (("gdp_growth", "infl", 1), ("infl", "gdp_growth", 1),
               ("log_hours", "wage_growth", 0), ("inv_growth", "cons_growth", 1))
    actual = fast.sw07_finite_sample_expectations(_PUBLISHED_MODE, moments,
                                                 nobs=nobs, common_max_lag=window)
    expected = sw07_finite_sample_moments(_PUBLISHED_MODE, moments,
                                         nobs=nobs, common_max_lag=window)
    assert_allclose(actual, expected.values, rtol=3e-12, atol=2e-13)
    if nobs > 2:
        assert abs(actual.iloc[0] - actual.iloc[1]) > .001


def test_fit_subset_preserves_original_window_only_when_requested():
    full = fast.sw07_finite_sample_expectations(_PUBLISHED_MODE)
    subset = fast.sw07_finite_sample_expectations(_PUBLISHED_MODE, FIT_MOMENTS, common_max_lag=4)
    default_subset = fast.sw07_finite_sample_expectations(_PUBLISHED_MODE, FIT_MOMENTS)
    assert_allclose(subset, full.iloc[:len(FIT_MOMENTS)], atol=1e-14)
    assert np.max(np.abs(subset - default_subset)) > 1e-4


def test_known_mean_control_equals_population_covariances():
    params = {**_PUBLISHED_MODE, "ctrend": 2.1, "constepinf": 3.2}
    actual = fast.sw07_finite_sample_expectations(params, demean=False)
    assert_allclose(actual, sw07_covariance_moments(params), rtol=3e-12, atol=2e-13)
    assert "known model population mean" in actual.attrs["mean_treatment"]


@pytest.mark.parametrize("demean", [False, True])
def test_independent_dense_quadratic_form_with_nonsymmetric_var(monkeypatch, demean):
    transition = np.array([[.68, .24], [-.13, .36]])
    innovation = np.array([[.7, .16], [.16, .4]])
    stationary = np.linalg.solve(np.eye(4) - np.kron(transition, transition),
                                 innovation.ravel()).reshape(2, 2)
    observation = np.zeros((len(OBSERVED_VARS), 2))
    observation[OBSERVED_VARS.index("gdp_growth"), 0] = 1
    observation[OBSERVED_VARS.index("infl"), 1] = 1
    model = SimpleNamespace(T=transition, Z=observation)
    monkeypatch.setattr(fast, "_stationary_system", lambda params: (model, stationary, {}))
    nobs, window = 11, 4
    moments = (("gdp_growth", "gdp_growth", 0), ("gdp_growth", "infl", 1),
               ("infl", "gdp_growth", 1), ("infl", "gdp_growth", 4))
    gammas = [np.linalg.matrix_power(transition, h) @ stationary for h in range(nobs)]
    covariance = np.block([[gammas[t-s] if t >= s else gammas[s-t].T
                            for s in range(nobs)] for t in range(nobs)])
    center = np.eye(nobs) - np.ones((nobs, nobs)) / nobs if demean else np.eye(nobs)
    center = np.kron(center, np.eye(2))
    expected = []
    indices = {"gdp_growth": 0, "infl": 1}
    for left, right, lag in moments:
        operator = np.zeros_like(covariance)
        for t in range(window, nobs):
            operator[2*t + indices[left], 2*(t-lag) + indices[right]] = 1 / (nobs-window)
        expected.append(np.trace(center.T @ operator @ center @ covariance))
    actual = fast.sw07_finite_sample_expectations({}, moments, nobs=nobs,
                                                 common_max_lag=window, demean=demean)
    assert_allclose(actual, expected, rtol=2e-12, atol=1e-14)
    assert abs(actual.iloc[1] - actual.iloc[2]) > .02


def test_iid_exact_negative_autocovariance_after_sample_demeaning(monkeypatch):
    observation = np.zeros((len(OBSERVED_VARS), 1))
    observation[OBSERVED_VARS.index("gdp_growth"), 0] = 1
    model = SimpleNamespace(T=np.zeros((1, 1)), Z=observation)
    variance, nobs = .64, 156
    monkeypatch.setattr(fast, "_stationary_system",
                        lambda params: (model, np.array([[variance]]), {}))
    moments = tuple(("gdp_growth", "gdp_growth", h) for h in (0, 1, 4))
    actual = fast.sw07_finite_sample_expectations({}, moments, nobs=nobs, common_max_lag=4)
    assert_allclose(actual, [variance * (1-1/nobs), -variance/nobs, -variance/nobs], atol=1e-15)


def test_zero_shock_limit_is_exactly_zero():
    params = {name: 0. for name in SW07_SHOCK_STDS}
    actual = fast.sw07_finite_sample_expectations(params)
    assert_allclose(actual, np.zeros(len(ALL_MOMENTS)), atol=0.)


@pytest.mark.parametrize("kwargs", [
    {"nobs": True}, {"nobs": 1}, {"nobs": 156.}, {"demean": 1},
    {"common_max_lag": False}, {"common_max_lag": -1},
    {"common_max_lag": 3}, {"common_max_lag": 156},
    {"moments": ()}, {"moments": (("unknown", "infl", 0),)},
    {"moments": (("infl", "infl", True),)},
    {"moments": (("infl", "infl", 0), ("infl", "infl", 0))},
])
def test_invalid_sample_or_moment_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        fast.sw07_finite_sample_expectations({}, **kwargs)


@pytest.mark.parametrize("params", [{"unknown": .5}, {"em": -.1}, {"em": np.nan},
                                     {"crr": True}, {"crr": .5 + .1j}, None])
def test_calibration_validation_is_preserved(params):
    with pytest.raises(ValueError):
        fast.sw07_finite_sample_expectations(params)


@pytest.mark.filterwarnings("ignore:klein_solve.*:RuntimeWarning")
def test_nonstationary_calibration_is_rejected():
    with pytest.raises(np.linalg.LinAlgError, match="stationar|Blanchard"):
        fast.sw07_finite_sample_expectations({"crhoa": 1.})
