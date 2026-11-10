"""Continuation of the actual model-adapter residuals, checked by re-solving."""
from dataclasses import replace

import numpy as np

from puremacro.core import continue_equilibrium
from puremacro.vfi import (
    CollocationProblem, linearize_projection, linearize_aiyagari, solve_aiyagari_continuous,
)


def test_growth_continuation_agrees_with_independent_projection_solves():
    kss = (.36*.96)**(1/(1-.36))
    problem = CollocationProblem(
        domain=(.5*kss, 1.5*kss), orders=6, method="euler", beta=.96,
        params={"alpha": .36, "delta": 1., "sigma": 1.},
        options={"tol": 1e-12, "max_iter": 500},
    )
    solution = problem.solve()
    lin = linearize_projection(solution, problem)
    branch = continue_equilibrium(
        lin.system, lin.state, lin.parameters, parameter="beta",
        step=.002, max_step=.002, max_steps=8, branch_id="growth-beta",
    )
    assert branch.status == "max_steps" and len(branch.states) == 9
    assert not branch.folds and branch.parameters[-1, 1] > .97
    np.testing.assert_array_equal(branch.parameters[:, 0], .36)
    for i in (4, 8):
        reference = replace(problem, beta=branch.parameters[i, 1]).solve()
        assert reference.converged
        np.testing.assert_allclose(branch.states[i], reference.coefficients, rtol=1e-6, atol=2e-8)


def test_aiyagari_continuation_preserves_probability_and_matches_resolves():
    options = dict(beta=.94, n_a=20, N_k=40, n_z=2,
                   P_z=np.array([[.85, .15], [.2, .8]]), z_grid=np.array([.6, 1.4]),
                   xtol=1e-11, egm_tol=1e-11, tol_ge=1e-6)
    solution = solve_aiyagari_continuous(**options)
    lin = linearize_aiyagari(solution, parameters=("beta",))
    branch = continue_equilibrium(lin.system, lin.state, lin.parameters,
                                  step=.0002, max_step=.0002, max_steps=4)
    assert branch.status == "max_steps" and len(branch.states) == 5
    nc = solution.household_solution.policy_c.size
    np.testing.assert_allclose(branch.states[:, nc:-1].sum(axis=1), 1., atol=1e-10)
    for i in (2, 4):
        reference = solve_aiyagari_continuous(**(options | {"beta": branch.parameters[i, 0]}))
        assert reference.converged
        state = np.r_[reference.household_solution.policy_c.ravel(), reference.distribution.pdf.ravel(), reference.r]
        np.testing.assert_allclose(branch.states[i], state, rtol=1e-6, atol=2e-8)
    assert branch.states[-1, -1] < branch.states[0, -1]
