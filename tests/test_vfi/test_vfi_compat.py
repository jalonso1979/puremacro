from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi import VFIProblem
from puremacro.vfi.compat import value_fn_iter_case1


def _savings_rf(ap, a, z, xp=np):
    c = np.exp(z) + 1.04 * a - ap
    return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -1e10)


def test_shim_matches_vfiproblem_no_decision():
    # the shim's (V, Policy) must equal a native VFIProblem solve of the same model
    a = np.linspace(1e-3, 10.0, 30)
    z = np.array([0.0, 0.3])
    P = np.array([[0.8, 0.2], [0.3, 0.7]])
    V, policy = value_fn_iter_case1(0, 30, 2, None, a, z, P, _savings_rf, 0.95)
    sol = VFIProblem(a_grid=a, z_grid=z, P_z=P, return_fn=_savings_rf,
                     beta=0.95).solve("numpy")
    np.testing.assert_allclose(V, sol.V, atol=1e-12)
    np.testing.assert_array_equal(policy, sol.policy_aprime)
    assert policy.shape == (30, 2)


def test_shim_with_decision_stacks_policy():
    # with a decision, Policy is np.stack([policy_d, policy_aprime]) (2, n_a, n_z)
    a = np.linspace(0.0, 4.0, 12)
    z = np.array([0.0])
    P = np.array([[1.0]])
    d = np.array([0.0, 0.5, 1.0])

    def rf(dd, ap, a, z, xp=np):
        c = 1.0 + a - ap - 0.1 * dd
        return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -1e10)

    V, policy = value_fn_iter_case1(3, 12, 1, d, a, z, P, rf, 0.95)
    sol = VFIProblem(a_grid=a, z_grid=z, P_z=P, return_fn=rf, beta=0.95,
                     d_grid=d).solve("numpy")
    assert policy.shape == (2, 12, 1)
    np.testing.assert_array_equal(policy[0], sol.policy_d)
    np.testing.assert_array_equal(policy[1], sol.policy_aprime)
    np.testing.assert_allclose(V, sol.V, atol=1e-12)


def test_shim_forwards_options_and_backend():
    a = np.linspace(1e-3, 8.0, 20)
    z = np.array([0.0])
    P = np.array([[1.0]])
    # forwarding howard/tol must not change the converged answer vs the default
    V1, p1 = value_fn_iter_case1(0, 20, 1, None, a, z, P, _savings_rf, 0.95,
                                 howard=False, tol=1e-11)
    V2, p2 = value_fn_iter_case1(0, 20, 1, None, a, z, P, _savings_rf, 0.95)
    np.testing.assert_allclose(V1, V2, atol=1e-7)
    np.testing.assert_array_equal(p1, p2)


def test_shim_size_validation():
    a = np.linspace(0, 1, 5)
    z = np.array([0.0])
    P = np.array([[1.0]])
    with pytest.raises(ValueError, match="n_a"):
        value_fn_iter_case1(0, 4, 1, None, a, z, P, _savings_rf, 0.95)  # n_a=4 != 5
    with pytest.raises(ValueError, match="n_z"):
        value_fn_iter_case1(0, 5, 2, None, a, z, P, _savings_rf, 0.95)  # n_z=2 != 1
    with pytest.raises(ValueError, match="n_d"):
        value_fn_iter_case1(3, 5, 1, np.array([0.0, 1.0]), a, z, P,
                            lambda dd, ap, a, z, xp=np: 0.0 * (dd + ap + a + z),
                            0.95)  # n_d=3 != len(d_grid)=2


def test_shim_exported():
    from puremacro.vfi import value_fn_iter_case1 as fn

    assert fn is value_fn_iter_case1
