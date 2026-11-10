"""Independent curves and failure contracts for equilibrium continuation."""
from dataclasses import replace

import numpy as np
import pytest

from puremacro.core import EquilibriumSystem, continue_equilibrium


def cubic_system():
    return EquilibriumSystem(
        lambda x, p: x**3 - x - p,
        ("capital",), ("policy",),
        state_jacobian=lambda x, p: np.array([[3*x[0]**2 - 1]]),
        parameter_jacobian=lambda x, p: np.array([[-1.]]),
    )


def test_cubic_traverses_two_refined_folds_and_three_known_roots():
    branch = continue_equilibrium(cubic_system(), [-1.5], [-1.875],
                                  step=.08, max_step=.12, max_steps=100,
                                  parameter_bounds=(-2., 2.))
    x, p = branch.states[:, 0], branch.parameters[:, 0]
    np.testing.assert_allclose(x**3 - x - p, 0., atol=1e-9)
    assert x[-1] > 1.5 and np.all(np.diff(x) > 0)
    assert len(branch.folds) == 2
    for fold, expected in zip(branch.folds, [-1/np.sqrt(3), 1/np.sqrt(3)]):
        assert fold.status == "simple_fold"
        np.testing.assert_allclose(fold.state, [expected], atol=2e-6)
        np.testing.assert_allclose(fold.parameters, [expected**3 - expected], atol=2e-6)
        assert fold.equilibrium_residual <= 1e-9
        assert abs(fold.curvature) > 1
    # An independent algebraic root calculation gives three equilibria at p=0.
    roots = np.sort(np.roots([1., 0., -1., 0.]).real)
    crossings = []
    for i in range(len(p)-1):
        if p[i] * p[i+1] <= 0:
            candidates = roots[(roots >= x[i]) & (roots <= x[i+1])]
            crossings.extend(candidates)
    np.testing.assert_allclose(crossings, [-1., 0., 1.])
    np.testing.assert_allclose(np.linalg.norm(branch.tangents, axis=1), 1., atol=1e-12)


def test_coupled_equations_hold_other_parameters_fixed_through_fold():
    system = EquilibriumSystem(
        lambda x, p: np.array([x[0]**2 + x[1] - p[0], x[1] - p[1]]),
        ("k", "price"), ("technology", "fixed"),
    )
    branch = continue_equilibrium(system, [1., 2.], [3., 2.], parameter="technology",
                                  direction=-1, step=.1, max_step=.1, max_steps=35)
    assert branch.states[-1, 0] < -.5
    np.testing.assert_allclose(branch.states[:, 1], 2., atol=1e-12)
    np.testing.assert_array_equal(branch.parameters[:, 1], 2.)
    np.testing.assert_allclose(branch.states[:, 0]**2 + 2., branch.parameters[:, 0], atol=1e-9)
    assert len(branch.folds) == 1 and branch.folds[0].status == "simple_fold"


def test_forward_reverse_restart_uses_scaled_tangent():
    first = continue_equilibrium(cubic_system(), [-1.2], [-.528], step=.05, max_step=.05, max_steps=12)
    reverse = continue_equilibrium(
        first.system, first.states[-1], first.parameters[-1],
        initial_tangent=-first.tangents[-1], step=.05, max_step=.05, max_steps=12,
    )
    # Hyperplane steps in opposite directions need not hit identical points.
    assert abs(reverse.states[-1, 0] - first.states[0, 0]) < .003
    np.testing.assert_allclose(reverse.states[:, 0]**3 - reverse.states[:, 0],
                               reverse.parameters[:, 0], atol=1e-9)


def test_coordinate_and_equation_units_preserve_the_physical_path():
    a = np.array([[2., 1.], [-1., 3.]])
    b = np.array([1., -2.])
    def system(xu, pu, ru):
        return EquilibriumSystem(
            lambda x, p: (a @ (x/xu) - b*p[0]/pu[0]) * ru,
            ("x", "y"), ("p",), state_scale=xu, parameter_scale=pu, residual_scale=ru,
            state_jacobian=lambda x, p: ru[:, None]*a/xu,
            parameter_jacobian=lambda x, p: (-ru*b/pu[0])[:, None],
        )
    plain = continue_equilibrium(system(np.ones(2), np.ones(1), np.ones(2)),
                                 [0., 0.], [0.], step=.03, max_step=.05, max_steps=15)
    xu, pu, ru = np.array([1000., .01]), np.array([100.]), np.array([.001, 100.])
    scaled = continue_equilibrium(system(xu, pu, ru), [0., 0.], [0.],
                                  step=.03, max_step=.05, max_steps=15)
    np.testing.assert_allclose(scaled.states / xu, plain.states, atol=1e-10)
    np.testing.assert_allclose(scaled.parameters / pu, plain.parameters, atol=1e-10)


def test_degenerate_turning_point_is_not_a_simple_fold():
    system = EquilibriumSystem(lambda x, p: x**4 - p, ("x",), ("p",),
                               state_jacobian=lambda x, p: np.array([[4*x[0]**3]]),
                               parameter_jacobian=lambda x, p: np.array([[-1.]]))
    branch = continue_equilibrium(system, [-.5], [.0625], direction=-1,
                                  step=.04, max_step=.04, max_steps=30)
    assert branch.states[-1, 0] > .3
    assert len(branch.folds) == 1
    assert branch.folds[0].status == "degenerate_or_unresolved"


def test_fold_seed_needs_orientation_and_branch_point_seed_is_rejected():
    fold = EquilibriumSystem(lambda x, p: x*x - p, ("x",), ("p",))
    with pytest.raises(ValueError, match="initial_tangent"):
        continue_equilibrium(fold, [0.], [0.])
    branch = continue_equilibrium(fold, [0.], [0.], initial_tangent=[1., 0.], max_steps=5)
    assert np.all(branch.states[1:, 0] > 0)
    assert branch.folds[0].status == "simple_fold"
    branching = EquilibriumSystem(lambda x, p: x**3 - p*x, ("x",), ("p",))
    with pytest.raises(ValueError, match="rank_deficient"):
        continue_equilibrium(branching, [0.], [0.], initial_tangent=[0., 1.])


def test_retry_preserves_failures_and_exhaustion_returns_only_accepted_points():
    system = EquilibriumSystem(lambda x, p: x*x-p, ("x",), ("p",))
    branch = continue_equilibrium(system, [1.], [1.], direction=-1, step=.4, max_step=.4,
                                  max_newton=1, max_retries=14, max_steps=3)
    assert len(branch.states) == 4
    assert any(not a.accepted for a in branch.attempts)
    np.testing.assert_allclose(branch.states[:, 0]**2, branch.parameters[:, 0], atol=1e-9)
    failed = continue_equilibrium(system, [1.], [1.], direction=-1, step=.4, max_step=.4,
                                  max_newton=1, max_retries=0, max_steps=3)
    assert len(failed.states) == 1 and failed.status == "newton_failed"
    assert len(failed.attempts) == 1 and not failed.attempts[0].accepted


def test_bounds_and_regimes_stop_without_accepting_invalid_points():
    system = EquilibriumSystem(lambda x, p: x-p, ("x",), ("p",))
    bounded = continue_equilibrium(system, [0.], [0.], parameter_bounds=(-.2, .2), max_steps=100)
    assert bounded.status == "parameter_bound"
    assert np.all(bounded.parameters <= .2)
    piecewise = replace(system, regime=lambda x, p: bool(p[0] >= 0))
    branch = continue_equilibrium(piecewise, [-.1], [-.1], min_step=1e-4, max_steps=100)
    assert branch.status == "regime_change"
    assert np.all(branch.parameters < 0)
    assert any(a.reason == "regime_change" for a in branch.attempts)


def test_seed_errors_derivative_errors_and_callback_errors_are_not_hidden():
    system = EquilibriumSystem(lambda x, p: x-p, ("x",), ("p",))
    with pytest.raises(ValueError, match="not_equilibrium"):
        continue_equilibrium(system, [1.], [0.])
    wrong = replace(system, parameter_jacobian=lambda x, p: np.ones((1, 1)))
    with pytest.raises(ValueError, match="derivative_mismatch"):
        continue_equilibrium(wrong, [0.], [0.])
    def broken(x, p):
        raise TypeError("inside economic equations")
    with pytest.raises(TypeError, match="inside economic equations"):
        continue_equilibrium(replace(system, residual=broken), [0.], [0.])


@pytest.mark.parametrize("options", [
    {"step": 0.}, {"min_step": .2, "max_step": .1}, {"direction": 0},
    {"max_steps": True}, {"parameter_bounds": (1., -1.)},
    {"initial_tangent": [0., 0.]}, {"parameter": "unknown"},
])
def test_invalid_continuation_options(options):
    system = EquilibriumSystem(lambda x, p: x-p, ("x",), ("p",))
    with pytest.raises(ValueError):
        continue_equilibrium(system, [0.], [0.], **options)


def test_owned_results_reports_and_state_budget():
    system = EquilibriumSystem(lambda x, p: x-p, ("x",), ("p",))
    x, p = np.zeros(1), np.zeros(1)
    result = continue_equilibrium(system, x, p, max_steps=3, branch_id="line")
    x[:] = 9.
    p[:] = 9.
    assert result.states[0, 0] == 0 and result.parameters[0, 0] == 0
    with pytest.raises(ValueError):
        result.states[0, 0] = 1.
    assert "state:x" in result.to_frame()
    assert "max_steps" in result.summary().to_string()
    assert "state:x" in result.to_markdown()
    assert "tabular" in result.to_latex()
    assert "table" in result.to_typst()
    big = EquilibriumSystem(lambda x, p: (_ for _ in ()).throw(AssertionError("must not run")),
                            ("x", "y"), ("p",))
    with pytest.raises(ValueError, match="max_states"):
        continue_equilibrium(big, [0., 0.], [0.], max_states=1)


def test_fixed_nonsmooth_parameter_is_not_differentiated():
    system = EquilibriumSystem(lambda x, p: x-p[0]-abs(p[1]), ("x",), ("moving", "fixed"))
    branch = continue_equilibrium(system, [0.], [0., 0.], parameter="moving", max_steps=4)
    assert branch.status == "max_steps"
    np.testing.assert_allclose(branch.states[:, 0], branch.parameters[:, 0], atol=1e-10)
    np.testing.assert_array_equal(branch.parameters[:, 1], 0.)


def test_truncation_error_does_not_establish_rank_at_a_branching_seed():
    system = EquilibriumSystem(lambda x, p: 10*x**3-p*x, ("x",), ("p",))
    with pytest.raises(ValueError, match="rank_"):
        continue_equilibrium(system, [0.], [0.], initial_tangent=[0., 1.])


def test_primary_branch_does_not_silently_cross_a_detectable_branch_point():
    system = EquilibriumSystem(lambda x, p: x**3-p*x, ("x",), ("p",))
    branch = continue_equilibrium(system, [0.], [-.11], step=.04, max_step=.04,
                                  min_step=1e-4, max_steps=50)
    assert branch.status in ("rank_deficient", "rank_unresolved", "possible_branch_point")
    assert np.all(branch.parameters < 0)


def test_exhausted_fold_refinement_is_reported_without_a_simple_fold_claim():
    branch = continue_equilibrium(cubic_system(), [-1.5], [-1.875], step=.1, max_step=.1,
                                  max_steps=70, fold_max_iter=1)
    assert len(branch.folds) == 2
    assert all(fold.status == "unresolved" for fold in branch.folds)
