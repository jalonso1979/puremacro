"""Independent numerical and failure contracts for shared equilibrium derivatives."""
from dataclasses import replace

import numpy as np
import pytest

from puremacro.core.equilibrium import EquilibriumSystem, linearize_equilibrium


def test_coupled_linear_solution_products_and_adjoint():
    a = np.array([[3.0, 1.0], [-1.0, 2.0]])
    b = np.array([[2.0, -0.5], [0.0, 3.0]])
    theta = np.array([0.4, 0.2])
    state = np.linalg.solve(a, b @ theta)
    system = EquilibriumSystem(
        lambda x, p: a @ x - b @ p, ("capital", "price"), ("productivity", "tax"),
        state_jacobian=lambda x, p: a, parameter_jacobian=lambda x, p: -b,
    )
    lin = linearize_equilibrium(system, state, theta)
    result = lin.sensitivity()
    assert result.valid and result.status == "regular"
    np.testing.assert_allclose(result.jacobian, np.linalg.solve(a, b), atol=1e-13)
    v, w = np.array([0.2, -0.7]), np.array([0.4, 1.1])
    np.testing.assert_allclose(lin.jvp(v), a @ v)
    np.testing.assert_allclose(lin.vjp(w), a.T @ w)
    np.testing.assert_allclose(w @ lin.jvp(v), v @ lin.vjp(w))
    np.testing.assert_allclose(lin.solve(w), np.linalg.solve(a, w))
    np.testing.assert_allclose(lin.solve(w, transpose=True), np.linalg.solve(a.T, w))
    assert lin.state_derivative_source == "supplied"
    assert result.linearized_residual < 1e-12


def test_unit_changes_preserve_physical_sensitivities():
    a = np.array([[3.0, 1.0], [0.5, 2.0]])
    b = np.array([[2.0], [1.0]])
    theta = np.array([0.3])
    state = np.linalg.solve(a, b @ theta)
    xunits, punits, runits = np.array([1000.0, 0.01]), np.array([100.0]), np.array([0.001, 100.0])
    system = EquilibriumSystem(
        lambda x, p: (a @ (x / xunits) - b @ (p / punits)) * runits,
        ("k", "r"), ("theta",), state_scale=xunits, parameter_scale=punits, residual_scale=runits,
    )
    result = linearize_equilibrium(system, state * xunits, theta * punits).sensitivity()
    assert result.valid
    np.testing.assert_allclose(result.jacobian / xunits[:, None] * punits, np.linalg.solve(a, b), rtol=1e-8)


@pytest.mark.parametrize("slope", [0.0, 1e-13])
def test_singular_scalar_never_uses_regularized_sensitivity(slope):
    system = EquilibriumSystem(lambda x, p: slope * x - p, ("x",), ("p",))
    lin = linearize_equilibrium(system, [0.0], [0.0])
    result = lin.sensitivity()
    assert not result.valid and "singular" in result.reasons
    assert np.isnan(result.jacobian).all()
    with pytest.raises(ValueError, match="singular"):
        lin.solve([1.0])


def test_equilibrium_error_and_nonsmoothness_are_explicit():
    system = EquilibriumSystem(lambda x, p: x - p, ("x",), ("p",))
    result = linearize_equilibrium(system, [2.0], [1.0]).sensitivity()
    assert not result.valid and "not_equilibrium" in result.reasons
    kink = EquilibriumSystem(lambda x, p: x - np.abs(p), ("x",), ("p",))
    result = linearize_equilibrium(kink, [0.0], [0.0]).sensitivity()
    assert not result.valid and "inconsistent_stencil" in result.reasons


def test_declared_regime_change_withholds_ordinary_derivative():
    system = EquilibriumSystem(lambda x, p: x - p, ("x",), ("p",),
                               regime=lambda x, p: tuple(p >= 0))
    result = linearize_equilibrium(system, [0.0], [0.0]).sensitivity()
    assert not result.valid and "regime_change" in result.reasons


def test_owned_arrays_and_snapshot_reporting():
    a = np.array([[2.0]])
    point = np.array([1.0])
    system = EquilibriumSystem(lambda x, p: 2 * x - p, ("x",), ("p",),
                               state_jacobian=lambda x, p: a)
    lin = linearize_equilibrium(system, point, [2.0], branch_id="benchmark")
    point[0] = 9.0
    a[0, 0] = 99.0
    assert lin.state[0] == 1.0 and lin.state_jacobian[0, 0] == 2.0
    with pytest.raises(ValueError):
        lin.state[0] = 3.0
    result = lin.sensitivity()
    assert result.branch_id == "benchmark"
    assert list(result.to_frame().columns) == ["p"]
    assert "x" in result.to_markdown()
    assert "tabular" in result.to_latex()
    assert "table" in result.to_typst()


def test_supplied_wrong_derivative_and_callback_errors_are_visible():
    system = EquilibriumSystem(lambda x, p: x - 2 * p, ("x",), ("p",),
                               parameter_jacobian=lambda x, p: np.zeros((1, 1)))
    assert "derivative_mismatch" in linearize_equilibrium(system, [2.0], [1.0]).reasons
    def broken(x, p):
        raise TypeError("inside user equation")
    with pytest.raises(TypeError, match="inside user equation"):
        linearize_equilibrium(replace(system, residual=broken), [2.0], [1.0])


@pytest.mark.parametrize("kwargs", [
    {"state_names": ("x", "x")}, {"state_scale": [0.0]},
    {"parameter_names": ()}, {"residual_scale": [float("nan")]},
])
def test_invalid_system_contract(kwargs):
    args = {"residual": lambda x, p: x - p, "state_names": ("x",), "parameter_names": ("p",)}
    with pytest.raises((ValueError, TypeError)):
        EquilibriumSystem(**(args | kwargs))


def test_dense_budget_checked_before_equations_run():
    def forbidden(x, p):
        raise AssertionError("must reject before residual evaluation")
    system = EquilibriumSystem(forbidden, ("x", "y"), ("p",))
    with pytest.raises(ValueError, match="max_states"):
        linearize_equilibrium(system, [0.0, 0.0], [1.0], max_states=1)


def test_ill_conditioning_threshold_and_scaled_transpose_solve():
    a = np.diag([1., 1e-6])
    system = EquilibriumSystem(lambda x, p: a @ x - p[0], ("x", "y"), ("p",))
    lin = linearize_equilibrium(system, [0., 0.], [0.], condition_limit=1e5)
    assert lin.reasons == ("ill_conditioned",)
    with pytest.raises(ValueError, match="ill_conditioned"):
        lin.solve([1., 1.])
    system = EquilibriumSystem(lambda x, p: a @ x - p[0], ("x", "y"), ("p",),
                               state_scale=[2., 3.], residual_scale=[4., 1e-5])
    lin = linearize_equilibrium(system, [0., 0.], [0.])
    rhs = np.array([[2., 3.], [4., 5.]])
    np.testing.assert_allclose(lin.solve(rhs, transpose=True), np.linalg.solve(a.T, rhs), rtol=1e-12)
