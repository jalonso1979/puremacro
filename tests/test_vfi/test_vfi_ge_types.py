from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi import VFIProblem, aiyagari_steady_state, tauchen
from puremacro.vfi.equilibrium import (
    PermanentTypesEquilibrium,
    stationary_equilibrium_types,
)


def _assets(aprime, a, z, *params, xp=np):   # *params: ignore forwarded prices (r, w)
    return a + xp.zeros_like(aprime)


def _aiyagari_pieces(alpha=0.36, delta=0.08, n_z=5, n_a=120, a_max=80.0):
    z_grid, P = tauchen(n=n_z, rho=0.9, sigma=0.2)
    w_eig, V_eig = np.linalg.eig(P.T)
    pi = np.real(V_eig[:, np.argmin(np.abs(w_eig - 1.0))])
    pi = pi / pi.sum()
    L = float(pi @ np.exp(z_grid))
    a_grid = np.linspace(1e-4, a_max, n_a)

    def KL_of_r(r):
        return (alpha / (r + delta)) ** (1.0 / (1.0 - alpha))

    def wage(r):
        return (1.0 - alpha) * KL_of_r(r) ** alpha

    return z_grid, P, a_grid, L, KL_of_r, wage


def _build_factory(z_grid, P, a_grid, wage, beta):
    def build(r, t):
        w = wage(r)

        def rf(ap, a, z, r_, w_, xp=np):
            c = w_ * np.exp(z) + (1.0 + r_) * a - ap
            return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -np.inf)

        return VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                          beta=beta, params={"r": r, "w": w},
                          options=dict(tol=1e-9, n_howard=40))
    return build


def test_one_type_ge_matches_plain_aiyagari():
    beta = 0.96
    z_grid, P, a_grid, L, KL_of_r, wage = _aiyagari_pieces(n_a=120)
    build = _build_factory(z_grid, P, a_grid, wage, beta)

    def resid(r, pt):
        return pt.aggregate(_assets) - L * KL_of_r(r)

    bracket = (0.005, 1.0 / beta - 1.0 - 0.002)
    eq = stationary_equilibrium_types(build, [1.0], resid, bracket)
    assert isinstance(eq, PermanentTypesEquilibrium)

    plain = aiyagari_steady_state(beta=beta, n_z=5, n_a=120, a_max=80.0)
    # one type reproduces the plain Aiyagari GE exactly (identical discrete problem):
    # same clearing price AND same residual (both sit at the same coarse-grid K(r) kink).
    assert eq.price == pytest.approx(plain["r"], abs=1e-4)
    assert eq.residual == pytest.approx(plain["equilibrium"].residual, abs=1e-6)


def test_two_identical_types_equal_one_type():
    beta = 0.96
    z_grid, P, a_grid, L, KL_of_r, wage = _aiyagari_pieces(n_a=120)
    build = _build_factory(z_grid, P, a_grid, wage, beta)

    def resid(r, pt):
        return pt.aggregate(_assets) - L * KL_of_r(r)

    bracket = (0.005, 1.0 / beta - 1.0 - 0.002)
    one = stationary_equilibrium_types(build, [1.0], resid, bracket)
    two = stationary_equilibrium_types(build, [0.5, 0.5], resid, bracket)
    assert two.price == pytest.approx(one.price, abs=1e-5)
    per = two.pt_solution.type_aggregates(_assets)
    np.testing.assert_allclose(per[0], per[1], atol=1e-9)


def test_two_beta_types_clears_and_patient_holds_more():
    betas = [0.94, 0.97]
    z_grid, P, a_grid, L, KL_of_r, wage = _aiyagari_pieces(n_a=120)

    def build(r, t):
        w = wage(r)

        def rf(ap, a, z, r_, w_, xp=np):
            c = w_ * np.exp(z) + (1.0 + r_) * a - ap
            return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -np.inf)

        return VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                          beta=betas[t], params={"r": r, "w": w},
                          options=dict(tol=1e-9, n_howard=40))

    def resid(r, pt):
        return pt.aggregate(_assets) - L * KL_of_r(r)

    # bracket below the LEAST patient... actually below the MOST patient type's 1/beta-1
    bracket = (0.005, 1.0 / max(betas) - 1.0 - 0.002)
    eq = stationary_equilibrium_types(build, [0.5, 0.5], resid, bracket)
    assert abs(eq.residual) < 1e-4
    per = eq.pt_solution.type_aggregates(_assets)
    assert per[1] > per[0]                                   # patient type holds more capital
    assert 0.0 < eq.price < 1.0 / min(betas) - 1.0


def test_weights_validation_propagates():
    z_grid, P, a_grid, L, KL_of_r, wage = _aiyagari_pieces(n_a=40, n_z=3)
    build = _build_factory(z_grid, P, a_grid, wage, 0.96)

    def resid(r, pt):
        return pt.aggregate(_assets) - L * KL_of_r(r)

    with pytest.raises(ValueError, match="sum to 1"):
        stationary_equilibrium_types(build, [0.5, 0.4], resid, (0.005, 0.04))


def test_ge_types_exported():
    from puremacro.vfi import PermanentTypesEquilibrium as E
    from puremacro.vfi import stationary_equilibrium_types as fn

    assert fn is stationary_equilibrium_types
    assert E is PermanentTypesEquilibrium
