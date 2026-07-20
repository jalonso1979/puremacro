from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi.finite_horizon import FiniteHorizonProblem, FiniteHorizonSolution


def test_shapes_and_horizon():
    n_a, n_z, J = 6, 2, 4
    a_grid = np.linspace(0.0, 5.0, n_a)
    z_grid = np.array([0.0, 0.5])
    P = np.array([[0.6, 0.4], [0.4, 0.6]])
    prob = FiniteHorizonProblem(
        a_grid=a_grid, z_grid=z_grid, P_z=P,
        return_fn=lambda ap, a, z, age, xp=np: -((a - ap) ** 2) + 0.0 * (z + age),
        beta=0.95, horizon=J,
    )
    sol = prob.solve("numpy")
    assert isinstance(sol, FiniteHorizonSolution)
    assert sol.V.shape == (J, n_a, n_z)
    assert sol.policy_aprime.shape == (J, n_a, n_z)
    assert sol.policy_d is None
    assert sol.horizon == J


def test_last_period_consumes_everything():
    # Terminal continuation 0, log utility, no income: in the LAST age the agent
    # saves nothing (a' = a_grid[0] = 0), consuming all current assets.
    n_a = 20
    a_grid = np.linspace(0.0, 4.0, n_a)
    z_grid = np.array([0.0])
    P = np.array([[1.0]])

    def rf(ap, a, z, age, xp=np):
        c = a - ap
        return xp.where(c > 1e-12, xp.log(xp.maximum(c, 1e-12)), -1e10)

    sol = FiniteHorizonProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                               beta=0.9, horizon=3).solve("numpy")
    # last age index = horizon-1 = 2: policy a' = index 0 (a_grid[0]=0) for all a>0
    assert np.all(sol.policy_aprime[2, 1:, 0] == 0)


def test_two_period_log_closed_form():
    # 2-period log, no income, terminal 0. Age 1 (last): consume all -> a''=0.
    # Age 0: max log(a-a') + beta*log(a'); FOC -> a' = beta/(1+beta) * a.
    beta = 0.9
    n_a = 400
    a_grid = np.linspace(1e-4, 5.0, n_a)
    z_grid = np.array([0.0]); P = np.array([[1.0]])

    def rf(ap, a, z, age, xp=np):
        c = a - ap
        return xp.where(c > 1e-12, xp.log(xp.maximum(c, 1e-12)), -1e10)

    sol = FiniteHorizonProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                               beta=beta, horizon=2).solve("numpy")
    # age 1 (last): consume all -> a' index 0
    assert np.all(sol.policy_aprime[1, 1:, 0] == 0)
    # age 0: a' = beta/(1+beta) * a on the interior
    aprime0 = a_grid[sol.policy_aprime[0, :, 0]]
    target = beta / (1.0 + beta) * a_grid
    spacing = a_grid[1] - a_grid[0]
    mask = (a_grid > 0.3) & (a_grid < 4.7)
    np.testing.assert_allclose(aprime0[mask], target[mask], atol=2.0 * spacing)


def test_terminal_value_is_used():
    # A large terminal value on the top asset should pull the last-age policy
    # toward saving (a' high) instead of consuming everything.
    n_a = 10
    a_grid = np.linspace(0.0, 9.0, n_a)
    z_grid = np.array([0.0]); P = np.array([[1.0]])
    term = np.zeros((n_a, 1)); term[-1, 0] = 1e15  # huge bequest value at top asset (must exceed 1e10 penalty)

    def rf(ap, a, z, age, xp=np):
        c = a - ap
        return xp.where(c > 1e-12, xp.log(xp.maximum(c, 1e-12)), -1e10)

    sol = FiniteHorizonProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                               beta=0.95, horizon=2, terminal_value=term).solve("numpy")
    # last age with a huge bequest at the top -> the richest state saves to the top
    assert sol.policy_aprime[1, -1, 0] == n_a - 1


def test_age_dependent_return_used():
    # return depends on age (income arrives only at age 0); verify age is threaded.
    n_a = 5
    a_grid = np.linspace(0.0, 4.0, n_a); z_grid = np.array([0.0]); P = np.array([[1.0]])
    income = np.array([2.0, 0.0])  # age 0 gets income 2, age 1 gets 0

    def rf(ap, a, z, age, xp=np):
        c = income[age] + a - ap
        return xp.where(c > 1e-12, xp.log(xp.maximum(c, 1e-12)), -1e10)

    sol = FiniteHorizonProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                               beta=0.95, horizon=2).solve("numpy")
    # with income at age 0, even the a=0 state has positive consumption available
    assert sol.V[0, 0, 0] > -1e9  # feasible (income makes c>0), not the -1e10 floor


def test_validation():
    a = np.linspace(0, 1, 3); z = np.array([0.0]); P = np.array([[1.0]])
    rf = lambda ap, a, z, age, xp=np: -((a - ap) ** 2) + 0.0 * (z + age)
    with pytest.raises(ValueError, match="horizon"):
        FiniteHorizonProblem(a_grid=a, z_grid=z, P_z=P, return_fn=rf, beta=0.9, horizon=0)
    with pytest.raises(ValueError, match="beta"):
        FiniteHorizonProblem(a_grid=a, z_grid=z, P_z=P, return_fn=rf, beta=1.5, horizon=2)


def test_age_is_reserved_param():
    # "age" is injected by the solver; a user "age" param would silently shadow it
    a = np.linspace(0, 1, 3); z = np.array([0.0]); P = np.array([[1.0]])
    rf = lambda ap, a, z, age, xp=np: -((a - ap) ** 2) + 0.0 * (z + age)
    with pytest.raises(ValueError, match="reserved"):
        FiniteHorizonProblem(a_grid=a, z_grid=z, P_z=P, return_fn=rf, beta=0.9,
                             horizon=2, params={"age": 5})


def test_mlx_parity():
    import importlib.util

    if importlib.util.find_spec("mlx") is None:
        import pytest
        pytest.skip("mlx not installed")
    n_a, n_z, J = 12, 3, 5
    a_grid = np.linspace(1e-3, 10.0, n_a)
    z_grid = np.array([-0.2, 0.0, 0.2])
    P = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])

    def rf(ap, a, z, age, w, xp=np):
        c = w * xp.exp(z) + a - ap + 0.1 * age
        return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -1e10)

    prob = FiniteHorizonProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                                beta=0.96, horizon=J, params={"w": 1.0})
    sol_np = prob.solve("numpy")
    sol_mlx = prob.solve("mlx")
    np.testing.assert_allclose(sol_mlx.V, sol_np.V, rtol=1e-4, atol=1e-3)
    np.testing.assert_array_equal(sol_mlx.policy_aprime, sol_np.policy_aprime)


def test_importable_from_package():
    from puremacro.vfi import FiniteHorizonProblem as FHP
    from puremacro.vfi import FiniteHorizonSolution as FHS

    assert FHP is FiniteHorizonProblem
    assert FHS is FiniteHorizonSolution
