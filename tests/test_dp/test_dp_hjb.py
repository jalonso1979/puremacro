"""Continuous-time dp models compiled to the HJB upwind scheme (``method="hjb"``).

Parity is exact against the hand-called ``vfi.solve_hjb_achdou`` and the
hard-coded continuous-time Aiyagari equilibrium. dp's ``crra`` carries the
``-1`` shift, so values differ from the solver's CRRA by ``1/((1 - gamma) rho)``.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro import dp
from puremacro.dp import ModelSpecError
from puremacro.vfi.hjb_achdou import solve_aiyagari_continuous_hjb, solve_hjb_achdou

A2 = np.array([[-0.1, 0.1], [0.1, -0.1]])


def household(reward="crra(c, gamma)", drift="r*a + w*e - c", **params):
    m = dp.Model("ct household")
    m.parameters(rho=0.05, gamma=2.0, r=0.03, w=1.0, **params)
    m.exogenous("e", dp.Jump([0.2, 1.0], A2))
    m.state("a", np.linspace(0.0, 30.0, 100))
    m.control("c")
    m.drift("a", drift)
    m.reward(reward)
    m.aggregate(A="a")
    return m


def test_crra_household_matches_solve_hjb_achdou():
    sol = household().solve()
    ref = solve_hjb_achdou()
    assert sol.method == "hjb"
    np.testing.assert_allclose(sol.raw.c_policy, ref.c_policy, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sol.V, ref.V - 1.0 / ((1.0 - 2.0) * 0.05), rtol=0, atol=1e-10)
    np.testing.assert_allclose(sol.raw.g_dist, ref.g_dist, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sol.policy("c"), ref.c_policy, rtol=0, atol=1e-12)
    assert sol.distribution.sum() == pytest.approx(1.0, abs=1e-12)
    assert sol.aggregates["A"] == pytest.approx(sol.mean("a"), abs=1e-14)


def test_cara_household_matches_solver_with_custom_utility():
    th = 1.5
    sol = household("-exp(-th*c)/th", th=th).solve()
    ref = solve_hjb_achdou(utility=lambda c: -np.exp(-th * c) / th, u_prime=lambda c: np.exp(-th * c),
                           u_prime_inv=lambda x: -np.log(x) / th)
    np.testing.assert_allclose(sol.raw.c_policy, ref.c_policy, rtol=0, atol=1e-8)


def test_aiyagari_equilibrium_matches_the_hard_coded_solver():
    ref = solve_aiyagari_continuous_hjb(tol_ge=1e-8)
    m = dp.Model("ct aiyagari")
    m.parameters(rho=0.05, gamma=2.0, alpha=0.33, delta=0.05)
    m.prices(r=(0.005, 0.048))
    m.exogenous("e", dp.Jump([0.2, 1.0], A2))
    m.state("a", np.linspace(0.0, 30.0, 100))
    m.local("w = (1 - alpha)*(alpha/(r + delta))^(alpha/(1 - alpha))")
    m.control("c")
    m.drift("a", "r*a + w*e - c")
    m.reward("crra(c, gamma)")
    m.aggregate(K="a", L="e")
    m.clear("K - L*(alpha/(r + delta))^(1/(1 - alpha))")
    sol = m.solve(xtol=1e-12)
    assert sol.prices["r"] == pytest.approx(ref.r_star, abs=1e-9)
    assert sol.aggregates["K"] == pytest.approx(ref.K_star, abs=1e-7)
    assert sol.aggregates["L"] == pytest.approx(ref.L_star, abs=1e-12)


def test_two_independent_jumps_equal_their_product_chain():
    B = np.array([[-0.3, 0.3], [0.5, -0.5]])

    def model(two):
        m = dp.Model("two jumps")
        m.parameters(rho=0.05, gamma=2.0, r=0.02)
        if two:
            m.exogenous("e", dp.Jump([0.2, 1.0], A2))
            m.exogenous("h", dp.Jump([0.9, 1.1], B))
            m.drift("a", "r*a + e*h - c")
        else:
            prod = np.kron(A2, np.eye(2)) + np.kron(np.eye(2), B)
            m.exogenous("y", dp.Jump([0.18, 0.22, 0.9, 1.1], prod))
            m.drift("a", "r*a + y - c")
        m.state("a", np.linspace(0.0, 20.0, 80))
        m.control("c")
        m.reward("crra(c, gamma)")
        return m

    np.testing.assert_allclose(model(True).solve().V, model(False).solve().V, rtol=0, atol=1e-10)


def test_huggett_bond_market_clears_with_a_negative_limit():
    m = dp.Model("ct huggett")
    m.parameters(rho=0.05, gamma=2.0)
    m.prices(r=(-0.05, 0.045))
    m.exogenous("z", dp.Jump([0.1, 0.2], [[-1.2, 1.2], [1.2, -1.2]]))
    m.state("a", np.linspace(-0.15, 5.0, 400))
    m.control("c")
    m.drift("a", "r*a + z - c")
    m.reward("crra(c, gamma)")
    m.aggregate(B="a")
    m.clear("B")
    sol = m.solve(xtol=1e-12)
    assert abs(sol.aggregates["B"]) < 1e-8
    assert -0.05 < sol.prices["r"] < 0.05
    assert np.all(sol.policy("c") > 0)


@pytest.mark.parametrize("change, match", [
    (lambda m: m.drift("a", "r*a + w*e - 2*c"), "slope -1"),
    (lambda m: m.drift("a", "r*a + w*e - c^2"), "slope -1"),
    (lambda m: m.drift("a", "r*a(+1) + w*e - c"), "no timing"),
    (lambda m: m.subject_to("c > 0"), "no subject_to"),
    (lambda m: m.reward("crra(c, gamma) + a"), "only through 'c'"),
    (lambda m: m.control("x"), "exactly one control"),
    (lambda m: m.horizon(3), "infinite horizon"),
])
def test_models_outside_hjb_are_rejected(change, match):
    m = household()
    change(m)
    with pytest.raises(ModelSpecError, match=match):
        m.solve()


def test_continuous_and_discrete_time_pieces_do_not_mix():
    with pytest.raises(ModelSpecError, match="method='hjb'"):
        household().solve("vfi")
    m = household()
    m._shocks = [("e", dp.AR1(rho=0.9, sigma=0.1, n=3))]
    with pytest.raises(ModelSpecError, match="dp.Jump"):
        m.solve()
    v = dp.Model("vfi with jump")
    v.parameters(beta=0.96)
    v.exogenous("e", dp.Jump([0.2, 1.0], A2))
    v.state("a", np.linspace(0.0, 1.0, 5))
    v.reward("a")
    with pytest.raises(ModelSpecError, match="continuous time"):
        v.solve()
