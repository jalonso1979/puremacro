"""Model re-solves independently validate the shared equilibrium derivatives."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from puremacro.vfi import (
    CollocationProblem, linearize_aiyagari, linearize_projection,
    solve_aiyagari_continuous,
)


@pytest.fixture(scope="module", params=[1.0, 2.0], ids=["log", "crra"])
def projection(request):
    kss = (.36 * .96) ** (1 / (1 - .36))
    problem = CollocationProblem(
        domain=(.5 * kss, 1.5 * kss), orders=6, method="euler", beta=.96,
        params={"alpha": .36, "delta": 1., "sigma": request.param, "z": 1.},
        options={"tol": 1e-12, "max_iter": 500},
    )
    solution = problem.solve()
    assert solution.converged
    return problem, solution


def test_projection_coefficients_and_policies_match_resolved_models(projection):
    problem, solution = projection
    lin = linearize_projection(solution, problem, parameters=("alpha", "beta", "sigma", "z"))
    result = lin.sensitivity()
    assert result.valid, result.reasons
    states = np.linspace(*problem.domain[0], 23)
    phi = solution.basis.evaluate(states)
    for j, name in enumerate(result.parameter_names):
        h = 1e-5
        if name == "beta":
            plus, minus = replace(problem, beta=problem.beta + h), replace(problem, beta=problem.beta - h)
        else:
            theta = problem.params[name]
            plus = replace(problem, params=problem.params | {name: theta + h})
            minus = replace(problem, params=problem.params | {name: theta - h})
        sp, sm = plus.solve(), minus.solve()
        assert sp.converged and sm.converged
        reference = (sp.coefficients - sm.coefficients) / (2 * h)
        np.testing.assert_allclose(result.jacobian[:, j], reference, rtol=2e-5, atol=2e-8)
        np.testing.assert_allclose(phi @ result.jacobian[:, j], phi @ reference, rtol=2e-5, atol=2e-8)


def test_projection_retains_equations_and_owns_inputs(projection):
    problem, solution = deepcopy(projection)
    lin = linearize_projection(solution, problem)
    original = lin.system.evaluate(lin.state, lin.parameters)
    changed = lin.parameters.copy()
    changed[0] += .001
    assert np.max(np.abs(lin.system.evaluate(lin.state, changed) - original)) > .001
    problem.params["alpha"] = .7
    solution.coefficients[:] = 0
    np.testing.assert_array_equal(lin.system.evaluate(lin.state, lin.parameters), original)


def test_projection_refuses_unsupported_and_clipped_models(projection):
    problem, solution = projection
    with pytest.raises(ValueError, match="custom model"):
        linearize_projection(solution, replace(problem, euler_residual_fn=lambda x, y, z: x))
    with pytest.raises(ValueError, match="unsupported"):
        linearize_projection(solution, problem, parameters=("rho",))
    with pytest.raises(ValueError, match="not converged"):
        linearize_projection(replace(solution, converged=False), problem)
    clipped = replace(solution, coefficients=np.zeros_like(solution.coefficients))
    with pytest.raises(ValueError, match="clipping"):
        linearize_projection(clipped, problem)


@pytest.fixture(scope="module", params=[False, True], ids=["persistent_income", "custom_income"])
def aiyagari(request):
    options = dict(beta=.95, n_a=30, N_k=60, n_z=3, xtol=1e-11, egm_tol=1e-11, tol_ge=1e-6)
    if request.param:
        options.update(beta=.94, n_z=2, rho_z=0., P_z=np.array([[.85, .15], [.2, .8]]),
                       z_grid=np.array([.6, 1.4]))
    solution = solve_aiyagari_continuous(**options)
    assert solution.converged
    lin = linearize_aiyagari(solution, parameters=("alpha", "beta", "gamma", "delta"))
    return options, solution, lin


def _aiyagari_state(solution):
    return np.r_[solution.household_solution.policy_c.ravel(), solution.distribution.pdf.ravel(), solution.r]


def test_aiyagari_joint_sensitivities_match_independent_resolves(aiyagari):
    options, solution, lin = aiyagari
    result = lin.sensitivity()
    assert result.valid, result.reasons
    nc = solution.household_solution.policy_c.size
    nk, nz = solution.distribution.pdf.shape
    assets = solution.distribution.asset_grid[:, None]
    np.testing.assert_allclose(result.jacobian[nc:-1].sum(axis=0), 0., atol=1e-10)
    for j, name in enumerate(result.parameter_names):
        theta, h = solution.metadata["params"][name], 1e-5
        plus = solve_aiyagari_continuous(**(options | {name: theta + h}))
        minus = solve_aiyagari_continuous(**(options | {name: theta - h}))
        assert plus.converged and minus.converged
        reference = (_aiyagari_state(plus) - _aiyagari_state(minus)) / (2 * h)
        # A full-state check includes the distribution tails and constrained
        # consumption nodes, not just a matching equilibrium interest rate.
        np.testing.assert_allclose(result.jacobian[:, j], reference, rtol=1e-4, atol=2e-7)
        np.testing.assert_allclose(result.jacobian[-1, j], reference[-1], rtol=2e-5, atol=2e-8)
        capital = float(np.sum(result.jacobian[nc:-1, j].reshape(nk, nz) * assets))
        np.testing.assert_allclose(capital, (plus.K - minus.K) / (2 * h), rtol=2e-5, atol=2e-6)
    assert result.jacobian[-1, result.parameter_names.index("beta")] < 0


def test_aiyagari_preserves_actual_chain_and_equation_snapshot(aiyagari):
    _, solution, _ = aiyagari
    owned = deepcopy(solution)
    lin = linearize_aiyagari(owned, parameters=("beta",))
    original = lin.system.evaluate(lin.state, lin.parameters)
    changed = lin.parameters + .001
    assert np.max(np.abs(lin.system.evaluate(lin.state, changed) - original)) > 1e-5
    owned.household_solution.P_z[:] = np.eye(len(owned.household_solution.P_z))
    owned.household_solution.policy_c[:] = 99.
    owned.distribution.pdf[:] = 0.
    np.testing.assert_array_equal(lin.system.evaluate(lin.state, lin.parameters), original)
    assert lin.equilibrium_residual < 1e-7


def test_aiyagari_refuses_unsupported_inputs_and_reports_bad_equilibrium(aiyagari):
    _, solution, _ = aiyagari
    with pytest.raises(ValueError, match="unsupported"):
        linearize_aiyagari(solution, parameters=("rho_z",))
    with pytest.raises(ValueError, match="not converged"):
        linearize_aiyagari(replace(solution, converged=False))
    with pytest.raises(ValueError, match="max_states"):
        linearize_aiyagari(solution, max_states=2)
    bad = deepcopy(solution)
    bad.distribution.pdf[:] *= .9
    result = linearize_aiyagari(bad).sensitivity()
    assert "not_equilibrium" in result.reasons
    assert not result.valid and np.isnan(result.jacobian).all()
