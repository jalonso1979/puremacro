from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi import VFIProblem, tauchen
from puremacro.vfi.epstein_zin import EpsteinZinProblem, EpsteinZinSolution


def _income_felicity(r):
    def u(ap, a, z, xp=np):
        c = (1.0 + r) * a + np.exp(z) - ap
        return xp.where(c > 0.0, c, -np.inf)          # period felicity = c (>0 feasible)
    return u


def test_reduces_to_time_separable_when_gamma_eq_inv_psi():
    # gamma = 1/psi with psi>1 (rho=1-1/psi>0): EZ collapses to time-separable.
    # Policy matches the standard VFIProblem with return (1-beta)*u^rho, and
    # V_ez^rho == V_std.
    psi, gamma = 2.0, 0.5                              # gamma == 1/psi
    rho = 1.0 - 1.0 / psi                              # = 0.5 > 0
    beta, r = 0.95, 0.04
    z_grid, P = tauchen(3, 0.8, 0.2)
    a_grid = np.linspace(0.1, 25.0, 40)               # a_min>0 keeps felicity c>0
    u = _income_felicity(r)

    ez = EpsteinZinProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=u,
                           beta=beta, gamma=gamma, psi=psi,
                           options=dict(tol=1e-12)).solve()
    assert isinstance(ez, EpsteinZinSolution)

    def R_add(ap, a, z, xp=np):
        c = (1.0 + r) * a + np.exp(z) - ap
        return xp.where(c > 0.0, (1.0 - beta) * xp.maximum(c, 1e-12) ** rho, -np.inf)

    std = VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=R_add, beta=beta,
                     options=dict(tol=1e-12, n_howard=0, max_iter=20000)).solve("numpy")

    np.testing.assert_array_equal(ez.policy_aprime, std.policy_aprime)   # exact policy match
    np.testing.assert_allclose(ez.V ** rho, std.V, rtol=1e-6, atol=1e-8)  # value transform


def test_shapes_and_positive_value():
    psi, gamma = 1.5, 5.0
    z_grid, P = tauchen(4, 0.85, 0.2)
    a_grid = np.linspace(0.1, 20.0, 30)
    sol = EpsteinZinProblem(a_grid=a_grid, z_grid=z_grid, P_z=P,
                            return_fn=_income_felicity(0.03), beta=0.95,
                            gamma=gamma, psi=psi).solve()
    assert sol.V.shape == (30, 4)
    assert sol.policy_aprime.shape == (30, 4)
    assert sol.policy_d is None
    assert np.all(sol.V > 0.0)                         # EZ value is positive
    assert np.all(np.diff(sol.policy_aprime, axis=0) >= 0)  # savings rise with assets (monotone)


def test_risk_aversion_matters():
    # holding psi fixed, raising gamma (more risk-averse) changes the policy:
    # EZ is NOT degenerate in gamma (unlike time-separable CRRA where only one
    # parameter governs both risk and EIS).
    z_grid, P = tauchen(5, 0.9, 0.25)
    a_grid = np.linspace(0.1, 30.0, 40)
    kw = dict(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=_income_felicity(0.03),
              beta=0.95, psi=1.5)
    low = EpsteinZinProblem(**kw, gamma=2.0).solve()
    high = EpsteinZinProblem(**kw, gamma=20.0).solve()
    assert not np.array_equal(low.policy_aprime, high.policy_aprime)


def test_with_labor_decision():
    # EZ with a decision d (labor): solves, valid shapes/policy.
    z_grid, P = tauchen(3, 0.8, 0.2)
    a_grid = np.linspace(0.1, 15.0, 25)
    d_grid = np.array([0.5, 1.0])

    def u(d, ap, a, z, xp=np):
        c = 1.03 * a + np.exp(z) * d - ap
        return xp.where(c > 0.0, c * (1.5 - d), -np.inf)   # felicity from c and leisure

    sol = EpsteinZinProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=u,
                            beta=0.95, gamma=4.0, psi=1.5, d_grid=d_grid).solve()
    assert sol.policy_d is not None
    assert sol.policy_d.shape == (25, 3)
    assert np.all(sol.V > 0.0)


def test_validation():
    z_grid, P = tauchen(2, 0.5, 0.2)
    a_grid = np.linspace(0.1, 5.0, 8)
    u = _income_felicity(0.03)
    with pytest.raises(ValueError, match="beta"):
        EpsteinZinProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=u,
                          beta=1.0, gamma=2.0, psi=1.5)
    with pytest.raises(ValueError, match="psi"):       # psi=1 (log) excluded
        EpsteinZinProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=u,
                          beta=0.95, gamma=2.0, psi=1.0)
    with pytest.raises(ValueError, match="gamma"):     # gamma=1 (log risk) excluded
        EpsteinZinProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=u,
                          beta=0.95, gamma=1.0, psi=1.5)


def test_ez_exported():
    from puremacro.vfi import EpsteinZinProblem as Pcls
    from puremacro.vfi import EpsteinZinSolution as Scls

    assert Pcls is EpsteinZinProblem and Scls is EpsteinZinSolution
