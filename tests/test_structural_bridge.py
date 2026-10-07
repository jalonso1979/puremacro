"""Independent GLS/analytic oracles and failure contracts for structural fitting."""
from dataclasses import FrozenInstanceError

import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.stats import chi2

from puremacro.structural import MomentTargets, StructuralFitResult, fit_structural


def _linear_problem():
    # An overidentified regression with correlated, unequal-variance estimates.
    design = np.array([[1., -1.], [1., 0.], [1., 1.], [1., 2.]])
    values = np.array([0.5, 1.7, 2.1, 3.5])
    covariance = np.array([[.09, .03, .01, 0], [.03, .16, .04, .01],
                           [.01, .04, .25, .06], [0, .01, .06, .36]])
    targets = MomentTargets(values, covariance, ("m0", "m1", "m2", "m3"),
                            "pp per 1pp shock", {"sample": "1980Q1:2020Q4", "n_obs": 164})
    # Normal equations give an independent oracle, not the whitening/SVD code.
    precision_design = np.linalg.solve(covariance, design)
    info = design.T @ precision_design
    parameter_covariance = np.linalg.inv(info)
    theta = np.linalg.solve(info, design.T @ np.linalg.solve(covariance, values))
    return design, targets, theta, parameter_covariance


@pytest.mark.parametrize("analytic", [False, True])
def test_linear_gls_covariance_and_j_have_no_sample_size_multiplier(analytic):
    design, targets, expected, variance = _linear_problem()
    result = fit_structural(lambda t: design @ t, targets, [0., 0.],
                            parameter_names=("level", "slope"),
                            jac=(lambda t: design) if analytic else None,
                            assume_correct_specification=True)
    assert isinstance(result, StructuralFitResult)
    assert result.success and result.inference_valid
    assert result.identification_rank == 2
    assert_allclose(result.theta, expected, atol=2e-9)
    assert_allclose(result.covariance, variance, atol=2e-9)
    assert_allclose(result.standard_errors, np.sqrt(np.diag(variance)), atol=2e-9)
    residual = design @ expected - targets.values
    objective = residual @ np.linalg.solve(targets.covariance, residual)
    assert_allclose(result.objective, objective, atol=1e-12)
    assert_allclose(result.j_statistic, objective, atol=1e-12)
    assert_allclose(result.j_pvalue, chi2.sf(objective, 2), atol=1e-12)
    assert result.j_df == 2
    diagonal_fit = fit_structural(lambda t: design @ t,
                                  MomentTargets(targets.values, np.diag(np.diag(targets.covariance)), targets.labels),
                                  [0., 0.], jac=lambda t: design)
    assert np.linalg.norm(diagonal_fit.theta - result.theta) > .01


def test_custom_weight_sandwich_is_not_inverse_hessian_or_chisquare():
    design, targets, _, _ = _linear_problem()
    weight = np.diag([1., 2., 4., 3.])
    inverse = np.linalg.inv(design.T @ weight @ design)
    influence = inverse @ design.T @ weight
    theta = influence @ targets.values
    variance = influence @ targets.covariance @ influence.T
    result = fit_structural(lambda t: design @ t, targets, [0., 0.], weight=weight,
                            jac=lambda t: design, assume_correct_specification=True)
    assert result.inference_valid
    assert_allclose(result.theta, theta, atol=1e-10)
    assert_allclose(result.covariance, variance, atol=1e-12)
    assert not np.allclose(result.covariance, inverse)
    assert np.isnan(result.j_pvalue) and np.isnan(result.j_statistic)
    assert result.diagnostics["weighting"] == "custom"


@pytest.mark.parametrize("factor", [1e-30, 1e-20, 1e20, 1e30])
def test_arbitrary_overall_weight_scale_changes_only_reported_objective(factor):
    design, targets, _, _ = _linear_problem()
    weight = np.diag([1., 2., 4., 3.])
    original = fit_structural(lambda t: design @ t, targets, [0., 0.],
                              weight=weight, jac=lambda t: design)
    rescaled = fit_structural(lambda t: design @ t, targets, [0., 0.],
                              weight=weight * factor, jac=lambda t: design)
    assert original.inference_valid and rescaled.inference_valid
    assert_allclose(rescaled.theta, original.theta, atol=1e-9)
    assert_allclose(rescaled.covariance, original.covariance, rtol=1e-10)
    assert_allclose(rescaled.objective / factor, original.objective, rtol=1e-10)


def test_nonlinear_analytical_recovery_and_unused_moments():
    # Exponential impulse response: h=0 identifies impact, h=1 identifies decay.
    h = np.arange(5)
    truth = np.array([1.4, .35])
    model = lambda t: t[0] * np.exp(-t[1] * h)
    jac = lambda t: np.column_stack([np.exp(-t[1] * h), -h * t[0] * np.exp(-t[1] * h)])
    targets = MomentTargets(model(truth), np.eye(5) * .01, tuple(f"h{i}" for i in h))
    unused = MomentTargets([99.], [[.02]], ("long_run",), "level")
    result = fit_structural(model, targets, [.8, .8], parameter_names=("impact", "decay"),
                            jac=jac, bounds=[(0, 3), (0, 2)], held_out=unused,
                            held_out_moments_at=lambda t: [t[0] / t[1]])
    assert_allclose(result.theta, truth, atol=1e-9)
    assert result.inference_valid and result.objective < 1e-15
    holdout = result.moment_fit(held_out=True)
    assert holdout["used_in_fit"].tolist() == [False]
    assert_allclose(holdout["model"], [4.])
    assert_allclose(holdout["residual"], [-95.])
    assert np.isnan(result.j_pvalue)  # explicit assertion is needed for J inference.


def test_covariance_and_fit_are_invariant_to_mixed_moment_units():
    design, targets, _, _ = _linear_problem()
    factors = np.array([1000., -.01, 100., 1e-5])
    scaled = targets.rescale(factors, units=("a", "b", "c", "d"))
    first = fit_structural(lambda t: design @ t, targets, [0., 0.], jac=lambda t: design)
    second = fit_structural(lambda t: factors * (design @ t), scaled, [0., 0.],
                            jac=lambda t: factors[:, None] * design)
    assert_allclose(first.theta, second.theta, atol=1e-10)
    assert_allclose(first.covariance, second.covariance, atol=1e-12)
    assert_allclose(first.objective, second.objective, atol=1e-12)
    assert_allclose(first.singular_values, second.singular_values, atol=1e-12)
    assert scaled.metadata["sample"] == targets.metadata["sample"]


@pytest.mark.parametrize("kind", ["collinear", "underidentified", "zero", "ill_conditioned"])
def test_no_inference_for_failed_local_rank_or_conditioning(kind):
    designs = {"collinear": [[1, 1], [2, 2], [3, 3]], "underidentified": [[1, 2]],
               "zero": [[0, 0], [0, 0]], "ill_conditioned": [[1, 0], [0, 1e-9]]}
    design = np.asarray(designs[kind], dtype=float)
    targets = MomentTargets(design @ np.ones(2), np.eye(len(design)), tuple(f"m{i}" for i in range(len(design))))
    with pytest.warns(UserWarning, match="inference unavailable"):
        result = fit_structural(lambda t: design @ t, targets, [1., 1.],
                                jac=lambda t: design, assume_correct_specification=True)
    assert result.success
    assert not result.inference_valid
    assert np.all(np.isnan(result.standard_errors)) and np.all(np.isnan(result.covariance))
    assert np.isnan(result.j_pvalue)
    if kind == "ill_conditioned":
        assert result.identification_rank == 2
        assert "ill_conditioned_jacobian" in result.diagnostics["inference_unavailable_reasons"]
    else:
        assert result.identification_rank < 2


def test_boundary_solution_suppresses_normal_inference():
    targets = MomentTargets([2., 4.], np.eye(2), ("a", "b"))
    with pytest.warns(UserWarning, match="parameter_on_boundary"):
        result = fit_structural(lambda t: t[0] * np.array([1., 2.]), targets, [.2], bounds=[(0., 1.)])
    assert result.success and not result.inference_valid
    assert_allclose(result.theta, [1.], atol=1e-8)
    assert result.boundary.tolist() == [True]
    assert np.isnan(result.standard_errors[0])


def test_optimizer_budget_failure_never_claims_inference():
    targets = MomentTargets([3., 9.], np.eye(2), ("a", "b"))
    with pytest.warns(UserWarning, match="optimizer_failed"):
        result = fit_structural(lambda t: np.array([t[0], t[0] ** 2]), targets, [.1], max_nfev=1)
    assert not result.success and not result.inference_valid
    assert result.n_evals == 1
    assert np.all(np.isnan(result.standard_errors))


def test_exact_identification_has_no_j_test():
    targets = MomentTargets([2.], [[.25]], ("a",))
    result = fit_structural(lambda t: t, targets, [0.], assume_correct_specification=True)
    assert_allclose(result.theta, [2.])
    assert_allclose(result.standard_errors, [.5])
    assert result.j_df == 0 and np.isnan(result.j_pvalue)


@pytest.mark.parametrize("slope", [1e-14, 1e-12, 1., 1e14])
@pytest.mark.parametrize("analytic", [False, True])
def test_parameter_units_do_not_turn_tiny_gradient_into_false_convergence(slope, analytic):
    # The same well-identified model in different parameter units. An absolute
    # gtol accepted theta=0 for slope=1e-14 despite a ten-standard-error miss.
    targets = MomentTargets([1.], [[.01]], ("m",))
    result = fit_structural(lambda t: np.array([slope * t[0]]), targets, [0.],
                            jac=(lambda t: np.array([[slope]])) if analytic else None)
    assert result.success and result.inference_valid
    assert_allclose(result.theta * slope, [1.], rtol=1e-9)
    assert_allclose(result.standard_errors * slope, [.1], rtol=1e-8)
    assert result.objective < 1e-16
    assert result.diagnostics["projected_residual_norm"] <= result.diagnostics["stationarity_tolerance"]


def test_stationarity_gate_accepts_noisy_overidentified_gls_in_large_parameter_units():
    design = np.array([1., 2., 3.])
    values = np.array([1.1, 1.8, 3.4])
    targets = MomentTargets(values, np.eye(3) * .01, ("a", "b", "c"))
    slope = 1e-14
    result = fit_structural(lambda t: design * slope * t[0], targets, [0.],
                            jac=lambda t: (design * slope)[:, None], assume_correct_specification=True)
    assert result.success and result.inference_valid
    assert result.objective > 1
    assert_allclose(result.theta * slope, [(design @ values) / (design @ design)], rtol=1e-9)
    assert result.diagnostics["projected_residual_norm"] < result.diagnostics["stationarity_tolerance"]
    assert np.isfinite(result.j_pvalue)


def test_centered_coordinates_prevent_large_parameter_offset_xtol_false_success():
    # In raw coordinates SciPy's relative xtol passes at the initial guess.
    # Centering the internal solve keeps this identical to (theta-0)^2=100.
    center = 1e12
    targets = MomentTargets([100.], [[1.]], ("m",))
    result = fit_structural(lambda t: np.array([(t[0] - center) ** 2]),
                            targets, [center + 1.],
                            jac=lambda t: np.array([[2 * (t[0] - center)]]))
    assert result.success and result.inference_valid
    assert_allclose(result.theta - center, [10.], atol=1e-8)


def test_optimizer_success_without_first_order_stationarity_withholds_inference(monkeypatch):
    from types import SimpleNamespace
    import puremacro.structural.bridge as bridge
    # An optimizer can report stagnation as successful termination. Inject
    # such a status while retaining a real, nonstationary moment discrepancy.
    monkeypatch.setattr(bridge, "least_squares", lambda fun, x0, **kw:
                        SimpleNamespace(x=x0, active_mask=np.zeros(len(x0)), success=True,
                                        status=2, optimality=1., nfev=1, message="stagnation"))
    targets = MomentTargets([1.], [[.01]], ("m",))
    with pytest.warns(UserWarning, match="nonstationary_solution"):
        result = fit_structural(lambda t: np.array([1e-14 * t[0]]), targets, [0.],
                                jac=lambda t: np.array([[1e-14]]))
    assert result.success  # preserves the documented raw optimizer status
    assert not result.inference_valid
    assert result.objective > 99
    assert np.isnan(result.standard_errors).all()
    assert result.diagnostics["projected_residual_norm"] > result.diagnostics["stationarity_tolerance"]


def test_singular_covariance_can_be_stored_but_not_silently_regularized():
    targets = MomentTargets([1., 1.], [[1., 1.], [1., 1.]], ("a", "b"))
    with pytest.raises(np.linalg.LinAlgError, match="structural covariance"):
        fit_structural(lambda t: np.repeat(t, 2), targets, [0.])


@pytest.mark.parametrize("covariance,match", [
    ([[1., 2.], [2., 1.]], "positive semidefinite"),
    ([[1., .1], [.2, 1.]], "symmetric"),
    ([[1., np.nan], [0., 1.]], "finite"),
    ([1., 1.], "2-dimensional"),
    (np.eye(3), "shape"),
    ([[0., 1e-15], [1e-15, 1.]], "zero covariance"),
    ([[0., 0.], [1e-15, 1.]], "zero covariance"),
    ([[-1., 0.], [0., 1.]], "negative diagonal"),
])
def test_invalid_covariances(covariance, match):
    with pytest.raises(ValueError, match=match):
        MomentTargets([1., 2.], covariance, ("a", "b"))


def test_target_labels_shapes_and_immutability():
    for values, labels in [([1., np.inf], ("a", "b")), ([[1., 2.]], ("a", "b")),
                           ([1., 2.], ("a", "a")), ([1., 2.], ("a",)), ([], ())]:
        with pytest.raises(ValueError):
            MomentTargets(values, np.eye(2), labels)
    source = np.array([1., 2.])
    metadata = {"sample": ["1980", "2020"], "shock": {"unit": "pp"}}
    targets = MomentTargets(source, np.eye(2), ("a", "b"), metadata=metadata)
    source[0] = 100
    metadata["shock"]["unit"] = "wrong"
    assert targets.values[0] == 1 and targets.metadata["shock"]["unit"] == "pp"
    with pytest.raises(ValueError):
        targets.values[0] = 2
    with pytest.raises(TypeError):
        targets.metadata["sample"] = "wrong"
    with pytest.raises(FrozenInstanceError):
        targets.units = ("x", "y")


def test_frame_irf_adapters_preserve_sample_and_shock_metadata():
    frame = pd.DataFrame({"h": [0, 1], "beta": [.4, .2], "se": [.2, .3]})
    frame.attrs.update({"sample": "1990:2020", "shock_size": 25, "shock_unit": "bp", "n_obs": 100})
    covariance = [[.04, .02], [.02, .09]]
    targets = MomentTargets.from_irf(frame, covariance=covariance,
                                     response="output", shock="policy", units="percent per 25bp")
    assert targets.labels == ("output<-policy[h=0]", "output<-policy[h=1]")
    assert targets.metadata["shock_size"] == 25
    assert targets.metadata["n_obs"] == 100
    assert targets.units == ("percent per 25bp",) * 2
    roundtrip = MomentTargets.from_frame(targets.to_frame(), covariance=targets.covariance,
                                         metadata=targets.metadata)
    assert_allclose(roundtrip.values, targets.values)
    assert roundtrip.labels == targets.labels
    with pytest.raises(ValueError, match="row order"):
        MomentTargets.from_irf(frame, covariance=covariance, response="output", shock="policy",
                                units="pp", horizons=[1, 0])
    with pytest.raises(ValueError, match="1-dimensional"):
        MomentTargets.from_irf(np.zeros((2, 2)), covariance=covariance, response="y", shock="x", units="pp")


def test_joint_draws_use_sample_covariance_not_variance_of_draw_mean():
    draws = np.array([[1., 2.], [2., 3.], [3., 5.], [4., 4.]])
    values = [1.7, 3.6]
    targets = MomentTargets.from_draws(draws, values=values, labels=("a", "b"), units="pp")
    centered = draws - draws.mean(axis=0)
    assert_allclose(targets.covariance, centered.T @ centered / 3)
    assert_allclose(targets.values, values)
    assert targets.metadata["n_draws"] == 4
    scalar = MomentTargets.from_draws(draws[:, :1], values=[2.], labels=("a",))
    assert scalar.covariance.shape == (1, 1)
    for invalid in [draws[:1], draws[:, :1], draws * np.nan]:
        with pytest.raises(ValueError):
            MomentTargets.from_draws(invalid, values=values, labels=("a", "b"))


def test_stacking_requires_explicit_dependence_and_preserves_components():
    one = MomentTargets([1.], [[.04]], ("a",), "percent", {"sample": "first"})
    two = MomentTargets([2., 3.], [[.09, .01], [.01, .16]], ("b", "c"), "pp", {"sample": "second"})
    with pytest.raises(ValueError, match="cross_covariance"):
        one.stack(two)
    with pytest.raises(ValueError, match="not both"):
        one.stack(two, cross_covariance=[[0, 0]], assume_independent=True)
    with pytest.raises(ValueError, match="boolean"):
        one.stack(two, assume_independent="yes")
    joint = one.stack(two, cross_covariance=[[.02, -.01]])
    assert_allclose(joint.covariance, [[.04, .02, -.01], [.02, .09, .01], [-.01, .01, .16]])
    assert joint.metadata["components"][1]["sample"] == "second"
    assert joint.units == ("percent", "pp", "pp")
    independent = one.stack(two, assume_independent=True)
    assert independent.covariance[0, 1] == 0
    assert independent.metadata["cross_covariance_assumption"] == "independent"
    with pytest.raises(ValueError, match="positive semidefinite"):
        one.stack(two, cross_covariance=[[5, 5]])
    with pytest.raises(ValueError, match="unique"):
        one.stack(one, assume_independent=True)
    selected = joint.select(("c", "a"))
    assert selected.labels == ("c", "a")
    assert_allclose(selected.covariance, [[.16, -.01], [-.01, .04]])


def test_rescaling_rejects_complex_factors_and_large_covariance_remains_finite():
    targets = MomentTargets([1.], [[1e308]], ("a",))
    assert np.isfinite(targets.covariance).all()
    with pytest.raises(ValueError, match="real"):
        targets.rescale([1j], units="pp")


def test_labeled_inputs_refuse_silent_reordering():
    covariance = pd.DataFrame(np.eye(2), index=["b", "a"], columns=["b", "a"])
    with pytest.raises(ValueError, match="labels"):
        MomentTargets([1., 2.], covariance, ("a", "b"))
    with pytest.raises(ValueError, match="labels"):
        MomentTargets(pd.Series([1., 2.], index=["b", "a"]), np.eye(2), ("a", "b"))
    targets = MomentTargets([1., 2.], np.eye(2), ("a", "b"))
    with pytest.raises(ValueError, match="moment_labels"):
        fit_structural(lambda t: t, targets, [0., 0.], moment_labels=("b", "a"))
    with pytest.raises(ValueError, match="model moment labels"):
        fit_structural(lambda t: pd.Series(t, index=["b", "a"]), targets, [0., 0.])
    with pytest.raises(ValueError, match="jacobian labels"):
        fit_structural(lambda t: t, targets, [0., 0.], parameter_names=("x", "y"),
                        jac=lambda t: pd.DataFrame(np.eye(2), index=["b", "a"], columns=["x", "y"]))


@pytest.mark.parametrize("kwargs", [
    {"max_nfev": 0}, {"max_nfev": True}, {"tol": 0}, {"rank_rtol": -1},
    {"condition_limit": 1}, {"bounds": [(1, 0)]}, {"bounds": [(2, 3)]},
    {"parameter_names": ("a", "b")}, {"assume_correct_specification": "yes"},
    {"weight": [[0.]]},
])
def test_invalid_fit_arguments(kwargs):
    targets = MomentTargets([1.], [[1.]], ("a",))
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        fit_structural(lambda t: t, targets, [0.], **kwargs)


def test_invalid_model_jacobian_and_holdout_fail_explicitly():
    targets = MomentTargets([1.], [[1.]], ("a",))
    for model in [lambda t: [np.nan], lambda t: [1., 2.], lambda t: [1j]]:
        with pytest.raises(ValueError):
            fit_structural(model, targets, [0.])
    with pytest.raises(ValueError, match="jacobian"):
        fit_structural(lambda t: t, targets, [0.], jac=lambda t: [[1., 2.]])
    with pytest.raises(ValueError, match="together"):
        fit_structural(lambda t: t, targets, [0.], held_out=targets)
    with pytest.raises(ValueError, match="disjoint"):
        fit_structural(lambda t: t, targets, [0.], held_out=targets, held_out_moments_at=lambda t: t)


def test_result_reporting_and_readonly_arrays():
    targets = MomentTargets([1.], [[.04]], ("m",))
    result = fit_structural(lambda t: t, targets, [0.], parameter_names=("impact",))
    table = result.summary(ci=.9)
    assert set(table.columns) == {"parameter", "estimate", "se", "ci_lower", "ci_upper", "boundary"}
    assert table.attrs["inference_valid"]
    assert "impact" in result.to_markdown() and "impact" in result.to_latex() and "impact" in result.to_typst()
    assert result.moment_fit()["unit"].tolist() == ["unspecified"]
    with pytest.raises(ValueError):
        result.theta[0] = 0
    with pytest.raises(ValueError, match="held-out"):
        result.moment_fit(held_out=True)
    with pytest.raises(ValueError, match="ci"):
        result.summary(ci=1.)


def test_boundary_estimate_never_leaves_the_bounds():
    """theta = theta0 + scale * z can round an ulp past a bound the optimizer sits on.

    The SW07 study then failed on Linux CI when it warm-started a second fit
    from such an estimate ("theta0 lies outside bounds"). The estimate must
    satisfy the bounds it was fitted under, exactly, and be reusable as a start.
    """
    targets = MomentTargets([2., 4.], np.eye(2), ("a", "b"))
    outside = 0
    for lower, upper, start, slope in [(lo, hi, s, k)
                                       for lo, hi in ((.2, .98), (.01, 1.), (.1, .7), (1e-3, .3))
                                       for s in np.linspace(.15, .95, 9)
                                       for k in (.3, 7., 50.)]:
        start = float(np.clip(start, lower, upper))
        model = lambda t, k=slope: t[0] * k * np.array([1., 2.])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = fit_structural(model, targets, [start], bounds=[(lower, upper)])
        outside += int(not lower <= fit.theta[0] <= upper)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_structural(model, targets, fit.theta, bounds=[(lower, upper)])
    assert outside == 0
