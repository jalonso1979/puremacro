"""Analytic Gaussian information sets, including genuine higher-order differences."""
from dataclasses import replace

import numpy as np
import pytest

from puremacro.expectations import (
    GaussianSignal, build_gaussian_information, audit_gaussian_information,
)


def panel(horizon=4):
    return build_gaussian_information(np.eye(3), (
        (GaussianSignal(0, [1., 1., 0.], "a"), GaussianSignal(2, [1., 0., 0.], "truth")),
        (GaussianSignal(0, [1., 0., 2.], "b"), GaussianSignal(2, [1., 0., 0.], "truth")),
    ), horizon=horizon, weights=[.4, .6], group_names=("A", "B"), state_names=("news", "noise_a", "noise_b"))


def test_normal_posterior_and_dated_revelation():
    info = panel()
    z = np.array([1., .2, -.3])
    beliefs = info.beliefs(z)
    assert beliefs[0, 0, 0] == pytest.approx((z[0]+z[1])/2)
    assert beliefs[0, 1, 0] == pytest.approx((z[0]+2*z[2])/5)
    np.testing.assert_allclose(info.posterior_covariances[0, :, 0, 0], [.5, .8])
    np.testing.assert_allclose(beliefs[2:, :, 0], 1., atol=1e-12)
    np.testing.assert_allclose(info.posterior_covariances[2:, :, 0, 0], 0., atol=1e-12)
    assert audit_gaussian_information(info)["accepted"]


def test_individual_tower_and_average_tower_are_distinct():
    info = panel()
    for group in range(2):
        P, Q = info.projectors[0, group], info.projectors[2, group]
        np.testing.assert_allclose(P@Q, P, atol=1e-12)
        np.testing.assert_allclose(P@P, P, atol=1e-12)
    mean = info.average_projectors[0]
    assert np.max(abs(mean@mean-mean)) > .1


def test_no_information_and_full_information_limits():
    none = build_gaussian_information(np.array([[2., .4], [.4, 1.]]), ((),), horizon=3)
    np.testing.assert_array_equal(none.projectors, 0.)
    np.testing.assert_allclose(none.posterior_covariances, np.broadcast_to(none.covariance, (3, 1, 2, 2)))
    full = build_gaussian_information(none.covariance, ((GaussianSignal(0, [1, 0], "x"),
        GaussianSignal(0, [0, 1], "y")),), horizon=3)
    np.testing.assert_allclose(full.projectors, np.broadcast_to(np.eye(2), (3, 1, 2, 2)), atol=1e-12)


def test_no_future_signal_leakage():
    info = build_gaussian_information(np.eye(2), ((GaussianSignal(2, [1, 1], "later"),),), horizon=4)
    np.testing.assert_array_equal(info.beliefs([1, 3])[:2], info.beliefs([5, -7])[:2])
    assert info.beliefs([1, 3])[2, 0, 0] != info.beliefs([5, -7])[2, 0, 0]


def test_duplicate_signals_do_not_double_count_information():
    once = build_gaussian_information(np.eye(2), ((GaussianSignal(0, [1, 2], "one"),),), horizon=2)
    twice = build_gaussian_information(np.eye(2), ((GaussianSignal(0, [1, 2], "one"),
        GaussianSignal(0, [2, 4], "duplicate")),), horizon=2)
    np.testing.assert_allclose(once.projectors, twice.projectors, atol=1e-12)
    assert twice.ranks[0, 0] == 1


def test_near_rank_threshold_is_explicitly_rejected():
    with pytest.raises(ValueError, match="ambiguous"):
        build_gaussian_information(np.eye(2), ((GaussianSignal(0, [1, 0], "one"),
            GaussianSignal(0, [1, 2e-12], "near")),), horizon=2)


def test_signal_rescaling_and_order_invariance():
    covariance = np.array([[2., .4], [.4, 1.]])
    one = build_gaussian_information(covariance, ((GaussianSignal(0, [1, 2], "a"),
        GaussianSignal(1, [3, -1], "b")),), horizon=3)
    other = build_gaussian_information(covariance, ((GaussianSignal(1, [-3e9, 1e9], "b"),
        GaussianSignal(0, [1e-9, 2e-9], "a")),), horizon=3)
    np.testing.assert_allclose(one.projectors, other.projectors, atol=1e-12)


def test_batch_conditioning_matches_sequential_scalar_updates():
    info = panel()
    z = np.array([.7, -.3, .4])
    for g, signals in enumerate(info.signals):
        mean, cov = np.zeros(3), np.eye(3)
        for signal in signals:
            h = signal.loading
            gain = cov@h/(h@cov@h)
            mean += gain*(h@z-h@mean)
            cov -= np.outer(gain, h@cov)
            np.testing.assert_allclose(info.beliefs(z)[signal.date, g], mean, atol=1e-12)
            np.testing.assert_allclose(info.posterior_covariances[signal.date, g], cov, atol=1e-12)


def test_inputs_owned_forecasts_and_frames_are_fresh():
    covariance = np.eye(2)
    loading = np.array([1., 1.])
    signal = GaussianSignal(0, loading, "s")
    info = build_gaussian_information(covariance, ((signal,),), horizon=3)
    covariance[0, 0] = 10
    loading[0] = 7
    assert info.covariance[0, 0] == 1 and signal.loading[0] == 1
    assert not info.projectors.flags.writeable
    assert info.forecast([2., 0.], [1., 3.], date=0, group=0) == pytest.approx(4.)
    frame = info.belief_frame([1., 3.])
    frame.iloc[0, 0] = -1e9
    assert info.belief_frame([1., 3.]).iloc[0, 0] != -1e9


def test_audit_rejects_hidden_extra_information():
    info = panel()
    bad = info.projectors.copy()
    bad[0, 0] = np.eye(3)
    assert not audit_gaussian_information(replace(info, projectors=bad))["accepted"]


@pytest.mark.parametrize("covariance", [np.zeros((2, 2)), [[1, 2], [2, 1]], [[1, .2], [0, 1]], [[np.nan]]])
def test_invalid_prior(covariance):
    with pytest.raises(ValueError):
        build_gaussian_information(covariance, ((),), horizon=2)


@pytest.mark.parametrize("weights", [[-1, 2], [0, 0], [.2, .2], [np.nan, 1]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        build_gaussian_information(np.eye(1), ((), ()), horizon=2, weights=weights)


@pytest.mark.parametrize("date", [-1, True, .5])
def test_invalid_release_dates(date):
    with pytest.raises(ValueError):
        GaussianSignal(date, [1], "s")


@pytest.mark.parametrize("signals", [
    ((GaussianSignal(3, [1], "late"),),),
    ((GaussianSignal(0, [1, 1], "wide"),),),
    ((GaussianSignal(0, [1], "same"), GaussianSignal(1, [2], "same")),),
])
def test_incompatible_signal_panels(signals):
    with pytest.raises(ValueError):
        build_gaussian_information([[1]], signals, horizon=3)


def test_dense_memory_limit_precedes_panel_allocation():
    with pytest.raises(ValueError, match="storage"):
        build_gaussian_information(np.eye(3), ((), ()), horizon=10**9)


def test_corrupt_covariance_and_weights_fail_audit():
    info = panel()
    for changed in (replace(info, covariance=np.zeros((3, 3))), replace(info, weights=[.5, .7])):
        assert not audit_gaussian_information(changed)["accepted"]


@pytest.mark.parametrize("date,group", [(-1, 0), (4, 0), (0, 2), (True, 0)])
def test_forecast_domain_checks(date, group):
    with pytest.raises(ValueError):
        panel().forecast([1, 0, 0], [1, 0, 0], date=date, group=group)
