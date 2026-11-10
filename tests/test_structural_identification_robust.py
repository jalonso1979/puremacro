"""Independent point-null algebra and complete scalar inverse images."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, norm

from puremacro.structural import (
    MomentTargets, MomentReplication, SimulationPlan, simulate_moments,
    QuadraticMomentCurve, moment_restriction_test, simulated_moment_test,
    invert_quadratic_moments, evaluate_moment_grid,
)


def _curve(loading=1., linear=1., quadratic=0., **kwargs):
    return QuadraticMomentCurve([0.], [loading], ("m",), "scalar-reference",
                                 linear=linear, quadratic=quadratic, **kwargs)


def test_correlated_point_null_uses_q_degrees_and_estimator_covariance():
    target = MomentTargets([.4, -.2, 1.], [[.04, .01, 0], [.01, .09, .02], [0, .02, .16]], ("a", "b", "c"))
    prediction = np.array([.1, -.3, .8])
    result = moment_restriction_test(target, prediction, calibration="gaussian_known")
    error = target.values-prediction
    expected = error @ np.linalg.solve(target.covariance, error)
    assert result.valid and result.df == 3
    assert result.statistic == pytest.approx(expected, rel=1e-13)
    assert result.pvalue == pytest.approx(chi2.sf(expected, 3))
    assert result.critical_value == pytest.approx(chi2.ppf(.95, 3))
    assert result.accepted == (expected <= chi2.ppf(.95, 3))
    assert result.calibration == "gaussian_known"


def test_moment_units_leave_the_test_and_set_unchanged():
    target = MomentTargets([.3, .7], [[.04, .01], [.01, .09]], ("a", "b"))
    curve = QuadraticMomentCurve([.1, .2], [1., .5], target.labels, "units")
    factors = np.array([1000., .01])
    scaled = target.rescale(factors, units=("scaled", "scaled"))
    transformed = replace(curve, offset=curve.offset*factors, loading=curve.loading*factors, units=scaled.units)
    first = invert_quadratic_moments(curve, target)
    second = invert_quadratic_moments(transformed, scaled)
    assert first.resolved and second.resolved
    np.testing.assert_allclose(first.intervals, second.intervals, rtol=1e-12, atol=1e-12)
    p1 = moment_restriction_test(target, curve(.2))
    p2 = moment_restriction_test(scaled, transformed(.2))
    assert p1.statistic == pytest.approx(p2.statistic, rel=1e-13)


@pytest.mark.parametrize("loading", [1., .02, 1e-14, -2.])
def test_affine_continuous_set_matches_normal_interval_even_when_weak(loading):
    target = MomentTargets([.3], [[.04]], ("m",))
    result = invert_quadratic_moments(_curve(loading), target, calibration="gaussian_known")
    z = norm.ppf(.975)
    expected = np.sort((np.array([.3-z*.2, .3+z*.2]))/loading)
    assert result.resolved and result.global_geometry == "bounded"
    np.testing.assert_allclose(result.intervals, [expected], rtol=1e-12)
    assert not result.truncated and result.contains(.3/loading)


def test_correlated_overidentified_curve_retains_orthogonal_residual():
    target = MomentTargets([.4, 1.1], [[.04, .01], [.01, .09]], ("a", "b"))
    curve = QuadraticMomentCurve([.1, .2], [1., 2.], target.labels, "overidentified")
    w, b, a = np.linalg.inv(target.covariance), curve.loading, target.values-curve.offset
    center = (b @ w @ a)/(b @ w @ b)
    residual = a-b*center
    radius = np.sqrt((chi2.ppf(.95, 2)-residual @ w @ residual)/(b @ w @ b))
    result = invert_quadratic_moments(curve, target)
    np.testing.assert_allclose(result.intervals, [[center-radius, center+radius]], atol=1e-12)
    assert result.df == 2  # not overidentification degrees q-p


@pytest.mark.parametrize("sign", [1., -1.])
def test_quadratic_inverse_keeps_disconnected_components(sign):
    target = MomentTargets([sign*1.], [[.01]], ("m",))
    result = invert_quadratic_moments(_curve(linear=0., quadratic=sign), target)
    low, high = np.sqrt([1-norm.ppf(.975)*.1, 1+norm.ppf(.975)*.1])
    expected = [[-high, -low], [low, high]]
    assert result.resolved
    np.testing.assert_allclose(result.intervals, expected, atol=1e-12)
    assert not result.contains(0.) and result.contains(-1.) and result.contains(1.)
    for theta in np.linspace(-2, 2, 101):
        test = moment_restriction_test(target, [sign*theta**2])
        assert result.contains(theta) == test.accepted


def test_shifted_scaled_curve_and_nonzero_linear_term():
    curve = _curve(linear=2., quadratic=1., center=1000., scale=.01)
    target = MomentTargets([2.], [[.04]], ("m",))
    result = invert_quadratic_moments(curve, target)
    l, u = 2-norm.ppf(.975)*.2, 2+norm.ppf(.975)*.2
    expected_z = np.array([[-1-np.sqrt(1+u), -1-np.sqrt(1+l)],
                           [-1+np.sqrt(1+l), -1+np.sqrt(1+u)]])
    np.testing.assert_allclose(result.intervals, 1000+.01*expected_z, atol=2e-12)
    assert result.resolved


def test_exact_quadratic_tangency_returns_a_singleton_not_an_empty_set():
    radius = np.sqrt(chi2.isf(.05, 1))
    target = MomentTargets([-radius], [[1.]], ("m",))
    result = invert_quadratic_moments(_curve(linear=0., quadratic=1.), target)
    assert result.resolved
    np.testing.assert_array_equal(result.intervals, [[0., 0.]])
    assert result.contains(0.) and not result.contains(1e-5)


@pytest.mark.parametrize("observation,expected", [(0., "all_real"), (10., "empty")])
def test_absent_identification_can_give_all_real_or_empty(observation, expected):
    result = invert_quadratic_moments(_curve(0.), MomentTargets([observation], [[1.]], ("m",)))
    assert result.resolved and result.global_geometry == expected
    if expected == "all_real":
        np.testing.assert_array_equal(result.intervals, [[-np.inf, np.inf]])
        assert result.contains(1e100)
    else:
        assert result.intervals.shape == (0, 2) and not result.contains(0.)


def test_constant_index_does_not_fake_identification_from_nonzero_loading():
    result = invert_quadratic_moments(_curve(2., linear=0., quadratic=0.),
                                      MomentTargets([0.], [[1.]], ("m",)))
    assert result.global_geometry == "all_real"


def test_search_domain_retains_truncation_and_global_geometry():
    target = MomentTargets([0.], [[1.]], ("m",))
    result = invert_quadratic_moments(_curve(), target, domain=(0., .5))
    np.testing.assert_allclose(result.intervals, [[0., .5]])
    assert result.truncated and result.boundary_contact == (True, True)
    assert result.global_geometry == "bounded" and len(result.global_intervals) == 1
    absent = invert_quadratic_moments(_curve(0.), target, domain=(-1., 1.))
    assert absent.truncated and absent.global_geometry == "all_real"
    outside = invert_quadratic_moments(_curve(), target, domain=(5., 6.))
    assert outside.intervals.shape == (0, 2) and outside.truncated
    assert outside.global_geometry == "bounded"  # not globally empty


def test_ambiguous_near_tangency_and_overflow_are_unresolved():
    crit = chi2.ppf(.95, 2)
    target = MomentTargets([0., np.sqrt(crit)*(1+2e-15)], np.eye(2), ("a", "b"))
    curve = QuadraticMomentCurve([0., 0.], [1., 0.], target.labels, "near-tangent")
    result = invert_quadratic_moments(curve, target)
    assert not result.resolved and result.global_geometry == "unresolved"
    with pytest.raises(ValueError, match="unresolved"):
        result.contains(0.)
    huge = invert_quadratic_moments(_curve(1e-320), MomentTargets([1.], [[1.]], ("m",)))
    assert not huge.resolved


def test_singular_covariance_is_unresolved_without_pseudoinverse():
    target = MomentTargets([1., 1.], [[1., 1.], [1., 1.]], ("a", "b"))
    point = moment_restriction_test(target, [0., 0.])
    assert not point.valid and point.accepted is None and np.isnan(point.pvalue)
    curve = QuadraticMomentCurve([0., 0.], [1., 1.], target.labels, "singular")
    result = invert_quadratic_moments(curve, target)
    assert not result.resolved and "covariance" in result.status


def test_simulation_test_uses_covariance_at_candidate_and_is_asymptotic():
    target = MomentTargets([.4], [[.01]], ("m",))
    plan = SimulationPlan(seed=4, replications=32, periods=10)
    def simulate(theta, rng, budget):
        return [theta[0]+(1+theta[0]**2)*rng.normal(size=budget.periods).mean()]
    results = []
    for theta in (0., 2.):
        batch = simulate_moments(simulate, [theta], plan, labels=target.labels, definition_id="candidate-variance")
        result = simulated_moment_test(target, batch, assume_independent=True)
        expected = (target.values[0]-batch.mean[0])**2/(.01+batch.covariance[0, 0])
        assert result.valid and result.calibration == "asymptotic"
        assert result.statistic == pytest.approx(expected)
        assert result.simulation is batch
        results.append(result)
    assert results[1].covariance[0, 0] > results[0].covariance[0, 0]
    cross = np.array([[.0001]])
    crossed = simulated_moment_test(target, batch, cross_covariance=cross)
    np.testing.assert_allclose(crossed.covariance, target.covariance+batch.covariance-2*cross)


def test_invalid_simulation_is_retained_as_unresolved_not_rejected():
    target = MomentTargets([0.], [[1.]], ("m",))
    plan = SimulationPlan(seed=5, replications=4, periods=3)
    batch = simulate_moments(lambda t, r, p: MomentReplication([np.nan], counts=[0]),
                             [0.], plan, labels=target.labels, definition_id="undefined")
    point = simulated_moment_test(target, batch, assume_independent=True)
    assert not point.valid and point.accepted is None and point.simulation is batch


def test_grid_missing_an_island_never_claims_a_continuous_empty_set():
    target = MomentTargets([.25], [[1e-4]], ("m",))
    test_at = lambda theta: moment_restriction_test(target, [theta**2])
    grid = evaluate_moment_grid(test_at, [-1., 0., 1.], parameter_name="theta")
    assert not grid.continuous_complete and not grid.accepted.any()
    assert len(grid.unresolved_cells()) == 2
    analytic = invert_quadratic_moments(_curve(linear=0., quadratic=1.), target)
    assert len(analytic.intervals) == 2 and analytic.contains(.5)


def test_grid_keeps_gaps_invalid_points_and_callback_errors():
    target = MomentTargets([0.], [[1.]], ("m",))
    def point(theta):
        return moment_restriction_test(target, [np.nan if theta == 0 else theta])
    grid = evaluate_moment_grid(point, [-1., 0., 1.])
    assert grid.valid.tolist() == [True, False, True]
    assert len(grid.accepted_runs()) == 2
    assert pd.isna(grid.to_frame().loc[1, "accepted"])
    assert grid.boundary_status == ("accepted", "accepted")
    def broken(theta):
        raise TypeError("callback bug")
    with pytest.raises(TypeError, match="callback bug"):
        evaluate_moment_grid(broken, [-1., 1.])


@pytest.mark.parametrize("alpha", [0., 1., -.1, np.nan, True])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        moment_restriction_test(MomentTargets([0.], [[1.]], ("m",)), [0.], alpha=alpha)


def test_labels_domains_calibration_and_immutable_reporting():
    target = MomentTargets([0.], [[1.]], ("m",))
    with pytest.raises(ValueError, match="labels"):
        moment_restriction_test(target, pd.Series([0.], index=["wrong"]))
    with pytest.raises(ValueError, match="calibration"):
        moment_restriction_test(target, [0.], calibration="exact_from_estimated_covariance")
    with pytest.raises(ValueError, match="domain"):
        invert_quadratic_moments(_curve(), target, domain=(1., 0.))
    with pytest.raises(ValueError, match="increasing"):
        evaluate_moment_grid(lambda x: moment_restriction_test(target, [x]), [1., 0.])
    result = invert_quadratic_moments(_curve(), target)
    with pytest.raises(ValueError):
        result.intervals.setflags(write=True)
    assert "lower" in result.to_frame()
    assert "tabular" in result.to_latex() and "table" in result.to_typst()
    assert "lower" in result.to_markdown()


def test_extreme_tail_critical_value_is_finite_and_calibrated():
    alpha = 1e-100
    result = moment_restriction_test(MomentTargets([0.], [[1.]], ("m",)), [0.], alpha=alpha)
    assert np.isfinite(result.critical_value)
    assert chi2.sf(result.critical_value, 1) == pytest.approx(alpha, rel=1e-11, abs=0.)


def test_reports_distinguish_unresolved_empty_and_unsampled_sets():
    singular = MomentTargets([0.], [[0.]], ("m",))
    unresolved = invert_quadratic_moments(_curve(), singular)
    empty = invert_quadratic_moments(_curve(0.), MomentTargets([10.], [[1.]], ("m",)))
    assert "unresolved_covariance" in unresolved.to_markdown()
    assert "empty" in empty.to_markdown()
    assert r"unresolved\_covariance" in unresolved.to_latex()
    grid = evaluate_moment_grid(lambda t: moment_restriction_test(MomentTargets([0.], [[1.]], ("m",)), [t]), [-1., 1.])
    assert "sampled point" in grid.to_markdown()


def test_constant_curve_avoids_irrelevant_overflow():
    curve = _curve(0., quadratic=1e300)
    np.testing.assert_array_equal(curve(1e300), [0.])


def test_parameter_names_do_not_overwrite_grid_statistics():
    target = MomentTargets([0.], [[1.]], ("m",))
    grid = evaluate_moment_grid(lambda t: moment_restriction_test(target, [t]),
                                [-1., 1.], parameter_name="alpha")
    frame = grid.to_frame()
    assert frame["parameter"].tolist() == ["alpha", "alpha"]
    np.testing.assert_array_equal(frame["value"], grid.grid)
    np.testing.assert_allclose(frame["alpha"], .05)
