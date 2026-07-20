from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")  # DC kernel needs numba; skip the whole file if absent

from puremacro.vfi import tauchen
from puremacro.vfi.kernels_numba import solve_vfi_dc_numba, solve_vfi_numba
from puremacro.vfi.returnfn import build_return_tensor
from puremacro.vfi.solve import solve_vfi


def _savings_problem(n_a=120, n_z=5, beta=0.95, r=0.03, rho=0.9, sigma=0.2):
    """A concave consumption-savings problem: the a'-policy is monotone in a, so
    divide-and-conquer is EXACT and must equal brute force bit-for-bit."""
    z_grid, P = tauchen(n=n_z, rho=rho, sigma=sigma)
    a_grid = np.linspace(0.0, 40.0, n_a)

    def rf(ap, a, z, xp=np):
        c = (1.0 + r) * a + np.exp(z) - ap
        return xp.where(c > 0.0, -1.0 / xp.maximum(c, 1e-12), -np.inf)

    R = build_return_tensor(rf, a_grid, z_grid, {}, d_grid=None, xp=np)
    return np.ascontiguousarray(R), P, beta


def _labor_choice_problem(n_a=60, n_d=4, n_z=3, beta=0.95, r=0.03):
    """A problem WITH a decision d (labor), still supermodular in (a,a')."""
    z_grid, P = tauchen(n=n_z, rho=0.8, sigma=0.2)
    a_grid = np.linspace(0.0, 25.0, n_a)
    d_grid = np.linspace(0.2, 1.0, n_d)               # labor supply

    def rf(d, ap, a, z, xp=np):
        c = (1.0 + r) * a + np.exp(z) * d - ap
        u = xp.where(c > 0.0, -1.0 / xp.maximum(c, 1e-12), -np.inf)
        return u - 0.5 * d ** 2                       # labor disutility
    R = build_return_tensor(rf, a_grid, z_grid, {}, d_grid=d_grid, xp=np)
    return np.ascontiguousarray(R), P, beta


def test_dc_matches_brute_numba_no_decision():
    R, P, beta = _savings_problem()
    Vb, polb, _, _ = solve_vfi_numba(R, P, beta, True, 20, 1e-10, 10_000)
    Vd, pold, _, _ = solve_vfi_dc_numba(R, P, beta, True, 20, 1e-10, 10_000)
    np.testing.assert_allclose(Vd, Vb, atol=1e-9)         # value identical
    np.testing.assert_array_equal(pold, polb)             # policy identical (monotone => same first-max)


def test_dc_matches_numpy_oracle_no_decision():
    R, P, beta = _savings_problem()
    Vo, flat_o, n_ap, _, _ = solve_vfi(R, P, beta, howard=True, n_howard=20, tol=1e-10)
    Vd, pold, _, _ = solve_vfi_dc_numba(R, P, beta, True, 20, 1e-10, 10_000)
    np.testing.assert_allclose(Vd, np.asarray(Vo), atol=1e-9)
    np.testing.assert_array_equal(pold, np.asarray(flat_o).astype(np.int64))


def test_dc_matches_brute_no_howard():
    R, P, beta = _savings_problem(n_a=80)
    Vb, polb, _, _ = solve_vfi_numba(R, P, beta, False, 0, 1e-10, 20_000)
    Vd, pold, _, _ = solve_vfi_dc_numba(R, P, beta, False, 0, 1e-10, 20_000)
    np.testing.assert_allclose(Vd, Vb, atol=1e-9)
    np.testing.assert_array_equal(pold, polb)


def test_dc_matches_brute_with_decision():
    R, P, beta = _labor_choice_problem()
    Vb, polb, _, _ = solve_vfi_numba(R, P, beta, True, 20, 1e-10, 10_000)
    Vd, pold, _, _ = solve_vfi_dc_numba(R, P, beta, True, 20, 1e-10, 10_000)
    np.testing.assert_allclose(Vd, Vb, atol=1e-9)
    # value-at-policy must match exactly even if a tie picks a different equal-value arg
    n_d, n_ap, n_a, n_z = R.shape
    EV = (Vb @ P.T)                                     # (n_a==n_ap, n_z)
    def q_at(flat):
        dd = flat // n_ap; ap = flat % n_ap
        out = np.empty((n_a, n_z))
        for ia in range(n_a):
            for iz in range(n_z):
                out[ia, iz] = R[dd[ia, iz], ap[ia, iz], ia, iz] + beta * EV[ap[ia, iz], iz]
        return out
    np.testing.assert_allclose(q_at(pold), q_at(polb), atol=1e-9)


def test_dc_matches_brute_large_grid():
    # equivalence holds at scale (where DC's sub-quadratic greedy actually pays off)
    R, P, beta = _savings_problem(n_a=400, n_z=7)
    Vb, polb, _, _ = solve_vfi_numba(R, P, beta, True, 30, 1e-10, 10_000)
    Vd, pold, _, _ = solve_vfi_dc_numba(R, P, beta, True, 30, 1e-10, 10_000)
    np.testing.assert_allclose(Vd, Vb, atol=1e-9)
    np.testing.assert_array_equal(pold, polb)


def test_dc_infeasible_state_raises():
    # a state with no feasible action (all -inf) must raise, like brute force
    R, P, beta = _savings_problem(n_a=20, n_z=2)
    R[:, :, 5, :] = -np.inf                              # state ia=5 infeasible for all a'
    with pytest.raises(RuntimeError):
        solve_vfi_dc_numba(R, P, beta, True, 20, 1e-10, 10_000)


from puremacro.vfi import VFIProblem


def _problem(n_a=120, n_z=5, beta=0.95, r=0.03):
    z_grid, P = tauchen(n=n_z, rho=0.9, sigma=0.2)
    a_grid = np.linspace(0.0, 40.0, n_a)

    def rf(ap, a, z, xp=np):
        c = (1.0 + r) * a + np.exp(z) - ap
        return xp.where(c > 0.0, -1.0 / xp.maximum(c, 1e-12), -np.inf)
    return a_grid, z_grid, P, rf, beta


def test_problem_dc_matches_plain_numba():
    a_grid, z_grid, P, rf, beta = _problem()
    base = dict(tol=1e-10, n_howard=20)
    plain = VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                       beta=beta, options=base).solve("numba")
    dc = VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf, beta=beta,
                    options={**base, "divide_and_conquer": True}).solve("numba")
    np.testing.assert_allclose(dc.V, plain.V, atol=1e-9)
    np.testing.assert_array_equal(dc.policy_aprime, plain.policy_aprime)


def test_problem_dc_requires_numba_backend():
    a_grid, z_grid, P, rf, beta = _problem(n_a=20, n_z=2)
    prob = VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf, beta=beta,
                      options={"divide_and_conquer": True})
    with pytest.raises(ValueError, match="divide_and_conquer"):
        prob.solve("numpy")
