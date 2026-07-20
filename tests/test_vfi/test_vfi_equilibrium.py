from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi import VFIProblem
from puremacro.vfi.equilibrium import EquilibriumResult, stationary_equilibrium


def _trivial_problem():
    # a valid 2-asset, 1-shock problem whose solve/dist always succeed
    a = np.array([0.1, 1.0])
    z = np.array([0.0])
    P = np.array([[1.0]])
    return VFIProblem(
        a_grid=a, z_grid=z, P_z=P,
        return_fn=lambda ap, a, z, xp=np: -((a - ap) ** 2) + 0.0 * z,
        beta=0.9,
    )


def test_root_find_locates_clearing_price():
    # residual independent of the (ignored) solved objects; root at price = 0.05
    def build_problem(price):
        return _trivial_problem()

    def market_residual(price, solution, mu, problem):
        return price - 0.05

    eq = stationary_equilibrium(build_problem, market_residual, (0.0, 1.0), xtol=1e-9)
    assert isinstance(eq, EquilibriumResult)
    assert eq.price == pytest.approx(0.05, abs=1e-6)
    assert abs(eq.residual) < 1e-6
    assert eq.distribution.shape == (2, 1)
    np.testing.assert_allclose(eq.distribution.sum(), 1.0, atol=1e-10)
    assert eq.n_evals >= 1


def test_bracket_without_sign_change_raises():
    def build_problem(price):
        return _trivial_problem()

    def market_residual(price, solution, mu, problem):
        return price + 1.0  # strictly positive on [0,1] -> no root

    with pytest.raises(ValueError):
        stationary_equilibrium(build_problem, market_residual, (0.0, 1.0))


def _stationary_of(M):
    w, V = np.linalg.eig(M.T)
    v = np.real(V[:, np.argmin(np.abs(w - 1.0))])
    return v / v.sum()


def test_aiyagari_general_equilibrium():
    # Canonical Aiyagari (1994) with log utility: incomplete markets + idiosyncratic
    # labor risk; firm Cobb-Douglas; solve for the interest rate that clears the
    # capital market (capital supply from household savings == firm capital demand).
    from puremacro.vfi import stationary_equilibrium, tauchen

    alpha, delta, beta = 0.36, 0.08, 0.96
    z_grid, P = tauchen(n=5, rho=0.9, sigma=0.2)
    pi_z = _stationary_of(P)
    L = float(pi_z @ np.exp(z_grid))            # aggregate effective labor (inelastic)
    a_grid = np.linspace(1e-4, 80.0, 100)

    def KL_of_r(r):
        return (alpha / (r + delta)) ** (1.0 / (1.0 - alpha))

    def wage_of_r(r):
        return (1.0 - alpha) * KL_of_r(r) ** alpha

    def build_problem(r):
        w = wage_of_r(r)

        def rf(ap, a, z, r_, w_, xp=np):
            c = w_ * xp.exp(z) + (1.0 + r_) * a - ap
            return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -1e10)

        return VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                          beta=beta, params={"r": r, "w": w},
                          options=dict(tol=1e-9, n_howard=40))

    def market_residual(r, sol, mu, prob):
        K_supply = float(np.sum(mu * a_grid[:, None]))   # aggregate household assets
        K_demand = L * KL_of_r(r)                          # firm FOC capital demand
        return K_supply - K_demand

    r_max = 1.0 / beta - 1.0                              # impatience bound
    eq = stationary_equilibrium(build_problem, market_residual,
                                (0.005, r_max - 0.002), xtol=1e-5)

    # equilibrium r strictly inside (0, 1/beta - 1)
    assert 0.0 < eq.price < r_max
    # capital market clears
    assert abs(eq.residual) < 1e-2
    # positive, finite aggregate capital
    K = float(np.sum(eq.distribution * a_grid[:, None]))
    assert K > 0.0 and np.isfinite(K)
    # Walras / goods-market clearing -- NOT imposed by the solver, so this is an
    # INDEPENDENT consistency check of the whole equilibrium (a circular firm-FOC
    # check r=alpha*(K/L)^(alpha-1)-delta would just re-derive market clearing).
    # In steady state aggregate consumption + depreciation must equal output:
    #   C + delta*K == Y = K^alpha * L^(1-alpha).
    C = eq.problem.aggregate(
        eq.solution, eq.distribution,
        lambda ap, a, z, r_, w_, xp=np: w_ * np.exp(z) + (1.0 + r_) * a - ap,
    )
    Y = K ** alpha * L ** (1.0 - alpha)
    np.testing.assert_allclose(C + delta * K, Y, rtol=2e-2)
    # distribution is a valid measure
    np.testing.assert_allclose(eq.distribution.sum(), 1.0, atol=1e-9)
    assert np.all(eq.distribution >= -1e-15)
