from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi import VFIProblem, VFISolution


def _simple_problem(**over):
    a = np.linspace(0.1, 5.0, 20)
    z = np.array([0.0, 0.5])
    P = np.array([[0.7, 0.3], [0.3, 0.7]])
    kw = dict(
        a_grid=a, z_grid=z, P_z=P,
        return_fn=lambda ap, a, z, r, xp=np: (
            xp.where(((1 + r) * a + xp.exp(z) - ap) > 1e-12,
                     xp.log(xp.maximum((1 + r) * a + xp.exp(z) - ap, 1e-12)),
                     -1e10)
        ),
        beta=0.95, params={"r": 0.03},
    )
    kw.update(over)
    return VFIProblem(**kw)


def test_beta_out_of_range_rejected():
    with pytest.raises(ValueError, match="beta must be in"):
        _simple_problem(beta=1.0)


def test_non_stochastic_matrix_rejected():
    with pytest.raises(ValueError, match="rows must sum"):
        _simple_problem(P_z=np.array([[0.7, 0.7], [0.3, 0.7]]))


def test_solve_numpy_returns_solution_shapes():
    sol = _simple_problem().solve("numpy")
    assert isinstance(sol, VFISolution)
    assert sol.V.shape == (20, 2)
    assert sol.policy_aprime.shape == (20, 2)
    assert sol.policy_d is None
    assert sol.backend == "numpy"
    assert sol.sup_norm < 1e-8
    assert np.all(np.diff(sol.V, axis=0) >= -1e-9)


def test_unavailable_backend_raises():
    with pytest.raises(ValueError, match="not available"):
        _simple_problem().solve("cupy")  # no NVIDIA GPU in this environment


def test_decision_axis_policy_returned():
    a = np.linspace(0.1, 3.0, 10)
    z = np.array([0.0])
    P = np.array([[1.0]])
    d = np.array([0.0, 1.0])
    prob = VFIProblem(
        a_grid=a, z_grid=z, P_z=P,
        return_fn=lambda dd, ap, a, z, xp=np: dd + 0.0 * (a - ap + z),
        beta=0.5, d_grid=d,
    )
    sol = prob.solve("numpy")
    assert sol.policy_d is not None
    assert sol.policy_d.shape == (10, 1)
    assert np.all(sol.policy_d == 1)  # always pick the higher d
